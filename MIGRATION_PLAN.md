# Hybrid Attention + L1 KVWeave 量化：迁移到 vllm-v0.26.0 + lmcache-main 的计划

## 背景

两个历史 patch：

- `lmcache-mp-hybrid.patch`：在 vLLM v0.23.0+ / LMCache v0.4.7 上，给 MP connector 加 hybrid attention（Mamba+attention 混合模型，如 Qwen3.5）在 XPU 上的支持。
- `lmcache-v0.4.7-mp-hybrid-to-kvweave.patch`：在上一个 patch 基础上，加 L1（内存层）KVWeave 4bit 量化支持。

结论先说：**lmcache-main 已经独立演化出了比这两个 patch 更完整、设计不同的 hybrid/多 object-group 基础设施**。因此大部分 patch 1 的内容已经是 no-op，不需要移植；patch 2（KVWeave 量化）里除了少数几个非 GPU 传输路径的限制外，基础设施大多也已就位，真正需要写的新代码集中在：**engine-driven（原 non_gpu）传输路径的多 group 支持**、**KVWeave 量化编解码器本身**、**Mamba conv/ssm 状态的量化**，以及把这些接到已有的 `group_layout_descs` / `AttnWindowDesc` 机制上。

以下按文件/模块列出现状 gap 和需要做的事，按依赖顺序分 Phase。

---

## 近期线上问题复盘与修正（2026-08-13）

这部分记录迁移过程中在真机联调（XPU + hybrid + MP）里遇到的实际问题，避免后续回归。

1. **LMCache HTTP 清缓存接口 404**
   - 现象：`POST /clear-cache` 返回 404。
   - 原因：`lmcache-main` 的 MP HTTP API 已统一为 `POST /cache/clear`（注意：这条只针对 MP HTTP API；`lmcache/v1/internal_api_server/vllm/cache_api.py` 里还有一个独立的、非 MP 场景用的 `DELETE /cache/clear`，二者不是同一个路由）。
   - 修正：测试脚本 `libraries.ai.kvweave/tests/vllm-bench-two-waves.sh` 中两处调用已改为 `"${LMCACHE_HTTP_URL}/cache/clear"`。
   - 备注：`/cache/clear` body 可省略，默认清 L1。

2. **XPU 启动时偶发 `free=0` 导致 vLLM 误判 OOM**
   - 现象：vLLM 在 `request_memory()` 前的 `MemorySnapshot.free_memory` 偶发为 0，报
     `Free memory on device xpu:0 (0.0/...) ... less than desired GPU memory utilization`。
   - 原因：在部分 XPU runtime / PyTorch 组合下，`torch.accelerator.get_memory_info(xpu)` 启动早期存在瞬时不稳定值。
   - 修正：`vllm-v0.26.0/vllm/utils/mem_utils.py` 已加 XPU fallback：当 `free==0` 时改用 `torch.xpu.mem_get_info()`（并保留原路径回退）。
   - 结论：这属于探测口径/时序问题，不是“设备真实无显存”。

3. **XPU 不支持 `Event(interprocess=True)`**
   - 现象：
     `TypeError: Event.__new__() got an unexpected keyword argument 'interprocess'`
     出现在 `lmcache_mp_connector.py` 的 `wait_for_save` / load 路径。
   - 原因：`torch.xpu.Event` 构造签名仅支持 `enable_timing`，不支持 CUDA 的 `interprocess` 参数。
   - 修正：`lmcache/integration/vllm/lmcache_mp_connector.py` 新增 `_new_transfer_event()`，优先尝试 `interprocess=True`，遇到 `TypeError` 自动回退 `Event()`。

4. **Hybrid 多 group 在 async engine-driven store 路径触发 shape 越界**
   - 现象：
     `IndexError: tuple index out of range`，堆栈在
     `gather_paged_kv_to_cpu -> get_head_size -> NL_X_NB_TWO_BS_NH_HS_Spec.head_size()`。
   - 根因：engine-driven 路径把注册时单次计算的 `self._engine_kv_format` 强行复用于所有 group；在 attention + mamba 混合 group 时，某些 group 的真实 tensor 维度与该 format 不匹配。
   - 修正：
     - `worker_transfer.py` 的 `submit_store`/`submit_retrieve`
     - `async_engine_driven.py` 的 async gather 路径
     均改为按 group 自动探测 format（`engine_kv_format=None`），不再强制使用全局 format。
   - 影响：修复了 hybrid + engine-driven 下的异步 store 崩溃，且对单 group 路径保持兼容。

5. **Qwen3.5-0.8B 参数对齐提醒（避免误报与性能退化）**
   - `Qwen3.5-0.8B` 推荐统一块大小 `N=544`。
   - 建议：`lmcache server --chunk-size 544`，vLLM 侧 `--max_num_batched_tokens 1087`（`2N-1`）。
   - 在 XPU 上建议显式 `lmcache.mp.mp_transfer_mode=engine_driven`，避免因默认路由/环境差异引入不确定性。

---

## vLLM 0.26 + lmcache-main Mamba SSM quant 调试结论（2026-08-20）

### 已确认结论

1. **当前 lmcache-main 的 SSM int4 native 调用参数不是直接传错。**
   - 用同一组 shape/seed/参数对比：`lmcache-v0.4.7` 的 `mamba_quant.py` 与当前 `lmcache-main` 的 `_KVWeaveCodec.quantize_mamba_substate_4bit(..., qbit=4)`。
   - 三组 SSM 配置：`per_tensor`、`per_channel+RH`、`per_token` 的 dequant 输出 bitwise 一致。
   - payload hash 不完全一致是因为 scale id/header 字段不同；dequant tensor 完全一致。
   - 因此，`blocks_num/head_num/head_dim/chunks`、`rh/asym/scaling_method`、decode header 解析这些 SSM int4 kernel 调参语义，目前看与 0.4.7 等价。

2. **vLLM 0.26 路径里之前确实存在非量化误差来源，已修正。**
   - Mamba page 必须按 vLLM 0.26 的真实 byte layout 处理：`conv_state | ssm_state | pad`，不能当普通 attention KV tensor 解释。
   - Mamba payload 需要显式 `conv_len + conv_payload + ssm_len + ssm_payload` framing，不能让 fixed-size slot padding 混入 SSM payload。
   - async engine-driven store 里，`gather_paged_kv_to_cpu()` 可能只 enqueue D2H copy；CPU 侧 KVWeave encode 之前必须等待 copy 完成，否则会读到未完成的 staging buffer。
   - group 3（单平面 attention/raw group）已验证 retrieve 与 scatter round-trip bit-exact，不是当前乱码根因。

3. **当前可跑通配置是 `conv 4bit + ssm 16bit`，但这不是最终解释“为什么老版 int4 可以”。**
   - `conv_state` 在真实服务里动态范围较大，`per_tensor` 误差过粗；改为 `conv per_channel + qbit=4` 后 mean error 降到约 `0.005`。
   - `ssm_state` 使用 4/8bit 有损恢复时，vLLM 0.26 + lmcache-main 的 wave2 命中路径仍会生成乱码。
   - `SSM_QUANT_ENABLED=0` 时 wave2 正常。
   - `SSM_QBIT=16`（fp16 value payload，仍走 SSM 独立 encode/decode 路径）时 wave2 正常。
   - 所以 `SSM_QBIT=16` 是当前安全运行配置；它不等价于“SSM int4 天生不行”，也不应作为最终根因闭环。

### 当前推荐运行配置

```bash
LMCACHE_MP_KVWEAVE_CONV_SCALING_METHOD=per_channel
LMCACHE_MP_KVWEAVE_CONV_QBIT=4
LMCACHE_MP_KVWEAVE_SSM_SCALING_METHOD=per_tensor
LMCACHE_MP_KVWEAVE_SSM_QBIT=16
LMCACHE_MP_KVWEAVE_SSM_QUANT_ENABLED=1
LMCACHE_MP_L1_KVWEAVE_QUANT=1
```

这套配置已经在 `vLLM 0.26 + lmcache-main` 上通过 1024-token 两波验证：wave1 cold 正常，wave2 `cached_tokens=1024` 正常。

### 下一步计划：真正解决 SSM int4 量化问题

目标不是关闭 SSM quant，也不是长期停留在 `SSM_QBIT=16`，而是解释并修复：为什么 `vLLM 0.23 + LMCache 0.4.7` 服务级可以容忍任意 SSM int4 配置，而 `vLLM 0.26 + lmcache-main` 当前不能。

1. **建立 0.23/0.4.7 与 0.26/main 的服务级 SSM dump 对照。**
   - 使用同一个模型、同一段 prompt、同样 token 边界、同样 cache chunk 长度。
   - dump store-time 的真实 `ssm_state`，不要只测 synthetic/standalone tensor。
   - 对比每个 Mamba group/layer/head 的 min/max/mean/std/percentile、zero/inf/nan、per-head 动态范围。

2. **确认两边 restore 的语义位置是否一致。**
   - 对齐 `mamba_cache_mode=align` 下 snapshot 的 token 边界。
   - 对齐 `tokens_per_block`、`block_id`、prefix cache reset 后的物理 block 重分配。
   - 确认 0.23/0.4.7 恢复的是同一个 recurrent state 语义点，而不是不同 step 或不同 page 视图。

3. **在真实服务路径中只替换 SSM int4，其余全部 raw/低损。**
   - attention/raw group 保持 bit-exact。
   - conv 使用已验证的 `per_channel + qbit=4` 或直接 raw 作为对照。
   - 只切换 SSM：raw / fp16 / int8 / int4，记录首 token logits 或最早发散位置。

4. **如果确认是 SSM int4 误差放大，继续查 int4 scaling 轴和 RH 语义，而不是直接判死。**
   - 验证 `per_tensor/per_channel/per_token` 在真实服务 SSM 分布上的误差，而不是只看随机 tensor。
   - 对比 0.4.7 服务实际用到的 scale 分块、RH transform length、preconditioner seed、payload header。
   - 若 0.26 的 SSM 分布显著更尖/更大，尝试更细粒度的 SSM int4 scaling（例如 per-head/per-state 方向）或校正 RH 分块，而不是关闭 SSM quant。

5. **验收标准。**
   - `LMCACHE_MP_KVWEAVE_SSM_QUANT_ENABLED=1`
   - `LMCACHE_MP_KVWEAVE_SSM_QBIT=4`
   - 两波请求中 wave2 `cached_tokens=1024` 且输出与 wave1 语义一致。
   - 同时保留当前 layout/framing/async D2H 同步修复。

### Step 1 执行结果（2026-08-20）

**环境搭建**：`kvweave-xpu-test` conda env 之前未装旧栈，本次补齐：`lmcache-v0.4.7`（`BUILD_WITH_SYCL=1 pip install -e .`）、kvweave native ext（`libraries.ai.kvweave/`，`KVWEAVE_XPU=1 KVWEAVE_COMPILER=icpx pip install -e .`）、`vllm-v0.23.0`（`VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation -e .`，另需先 `pip install setuptools_rust`）。装完后发现三个与本次调查无关、但阻塞旧栈起服务的环境/移植问题，均已修复：

1. **torch/torchvision/torchaudio 版本不匹配**：vLLM 的 XPU 安装把 `torch` 升级到 `2.11.0+xpu`，但 `torchvision`/`torchaudio` 仍是不带 `+xpu` 后缀的 CPU 编译版本，导致 `torchvision::nms`/`libcudart.so.13` 相关的 `OSError`/`RuntimeError`。修复：分别 `pip install --no-deps torchvision==0.26.0+xpu`/`torchaudio==2.11.0+xpu`（`--extra-index-url https://download.pytorch.org/whl/xpu`）。
2. **`triton`/`triton-xpu` 命名空间冲突**：`xgrammar` 依赖拉入了纯 CPU/CUDA 版 `triton==3.7.1`（来自 PyPI），与 `triton-xpu` 共用 `triton/` 安装路径，导致 `triton.next_power_of_2` 缺失（`AttributeError`），vLLM 的 Mamba/GDN layernorm kernel（`vllm/model_executor/layers/fla/ops/layernorm_guard.py`）因此崩溃。修复：`pip install --force-reinstall --no-deps triton-xpu==3.7.0`（走 pytorch xpu index）覆盖回正确文件。
3. **`vllm-v0.23.0` 的 Mamba `align` 模式指针溢出 bug**：`vllm/v1/worker/mamba_utils.py` 的 `collect_mamba_copy_meta()` 把 `state.data_ptr()`（XPU/Level-Zero USM 指针，可能 ≥ 2^63）写入 `torch.int64` 缓冲区，触发 `OverflowError: Python int too large to convert to C long`。这是指针值被当成有符号数解释导致的溢出，不是逻辑错误。修复：`src_ptrs`/`dst_ptrs` 缓冲区 dtype 从 `torch.int64` 改为 `torch.uint64`（`mamba_utils.py` 两处 `make_buffer` 调用），与用户提供的现成 fix 一致（见 `libraries.ai.kvweave/upstream/kv-quant-offload/integration/lmcache/vllm/docker/Dockerfile` 里对应的 `sed` 脚本）。
4. **XPU 启动时偶发大幅 free-memory 低估**：`vllm-v0.23.0` 没有 Phase 0 记录的那条"`free==0` 时 fallback 到 `torch.xpu.mem_get_info()`"修复（那条只进了 `vllm-v0.26.0`），实测中 vLLM 启动时探测到的 free memory 会稳定卡在一个比 `torch.xpu.mem_get_info()` 直接读到的值低很多的数（如探测到 4.5GB，独立查询显示 7-25GB），导致反复 `Free memory ... is less than desired GPU memory utilization` 或 `No available memory for the cache blocks`。这与 `env_vars.md` 里"关闭 vLLM 后需要 `sudo echo 3 > /proc/sys/vm/drop_caches`"的提示一致——本质是页缓存占用让 XPU 侧的 free-memory 探测偏低；执行一次 `drop_caches` 后 free memory 从 ~4-8GB 回升到 25GB，问题消失。

**Step 1 dump 对照结果**：双栈用相同模型（`Qwen3.5-0.8B`）、相同 prompt/`INPUT_LEN=1024`/`--chunk-size 1024`、相同 KVWeave 配置（`conv: per_channel qbit=4`，`ssm: per_tensor qbit=16`，即文档"当前推荐运行配置"）跑 `vllm-bench-two-waves.sh` 的 wave1（cold store），在 `_KVWeaveCodec.encode_chunk()`（新栈）/`quantize_mamba_substate_4bit()`（旧栈）的 split-after/quantize-before 插入点 dump 真实 `conv_state`/`ssm_state`（新增环境变量 `LMCACHE_MP_KVWEAVE_SSM_DUMP_DIR`，两栈对称实现，默认关闭、零开销）。用新写的 `compare_ssm_dumps.py`（仓库顶层）对比：

- 两栈的 `conv_state`/`ssm_state` shape 完全一致（`conv: [6,1,3,6144] fp16`，`ssm: [6,1,16,128,128] fp32`）。
- min/max/mean/std/p1/p50/p99 的差异都在 1e-2 到 1e-4 量级（推理路径本身的非确定性/不同随机 prompt 内容造成的正常波动），**没有观察到系统性的分布偏移或尖峰**。
- zero/nan/inf 计数两边完全相同（如 `ssm_chunk1`: 896/0/0 vs 896/0/0，共 1572864 元素）。
- per-head 动态范围（每个 head 的 max-min 分布区间）两栈几乎重合（如 `ssm_chunk1`: `[1.457, 19.876]` vs `[1.452, 19.796]`）。

**结论**：**store-time 的真实 `ssm_state`/`conv_state` 数值分布在两栈之间没有显著差异**——排除了"0.26 的 SSM 分布本身发生了偏移/更尖锐，导致 int4 量化误差被放大"这个假设。这意味着根因更可能在 Step 2（restore 语义位置是否对齐，例如 block 重分配/token 边界错位）而不是量化输入数据本身。


---

## 现状总结（各文件对照结论）

| 文件 | Patch 1 (hybrid) | Patch 2 (kvweave) | 结论 |
|---|---|---|---|
| `lmcache/integration/vllm/kv_cache_group_edits.py` | 已存在，lmcache-main 更进一步 | 已完成 | 已加入 `MambaSubStateLayout`/`MambaRealLayout`/`real_layout()` |
| `lmcache/integration/vllm/kv_cache_groups.py` | N/A（patch1 不改这个文件） | 已完成 | 已加入 cache category、Mamba layout 和 group merge 逻辑 |
| `lmcache/v1/multiprocess/group_view.py` | N/A | 已完成 | 已加入 `MambaSubStateWireLayout` 和 `EngineGroupInfo` 字段 |
| `lmcache/integration/vllm/lmcache_mp_connector.py` | 核心逻辑已被 lmcache-main 自己的实现取代（`AttnWindowDesc`/`create_engine_group_infos_from_vllm`），**patch1 hybrid 部分不适用** | debug 日志辅助函数 + `_record_transfer_event` 非 CUDA 兼容 fallback 缺失（cosmetic，非阻塞） | 只需移植日志/事件兼容性两个小改动 |
| `lmcache/v1/multiprocess/custom_types.py` | `RegisterEngineDrivenContextPayload`（对应老的 `RegisterNonGpuContextPayload`）没有 `engine_group_infos` | 没有 `group_layout_descs`/`enable_l1_kvweave_quant`/`SerializedMemoryLayoutDesc` | 需要在**正确的** payload 类上加字段 |
| `lmcache/v1/multiprocess/engine_context.py` | 已被更完整的 `AttnWindowDesc` + `group_layout_descs: dict[int, MemoryLayoutDesc]` 机制取代 | 同上，patch 的 union-type 方案已过时 | **不需要按 patch 移植**，只需确保新代码调用现有 API |
| `lmcache/v1/multiprocess/modules/lookup.py` | 核心已被 `_chunk_major_object_keys`/`fold_unfold_ranked` 取代，更完整（含 sliding window 折叠） | patch 的 dict 类型 guard 不适用（`find()` 从不返回 dict） | 不需要移植 |
| `lmcache/v1/multiprocess/transfer_context/worker_transfer.py` （原 `non_gpu_transfer.py` 的 worker 端，改名为 `EngineDrivenTransferContext`） | **仍然只支持单 group**：`_single_group_block_ids()` 在多 group 时直接抛异常，`register()` 里 `engine_group_infos` 是 no-op | 依赖多 group 支持才能工作 | **这是真正需要写的新代码**：实现 per-group gather/scatter |
| `lmcache/v1/multiprocess/modules/engine_driven_transfer.py`（原 `non_gpu_transfer.py` 的 server 端） | 同上，单 group 限制 | `enable_l1_kvweave_quant` 校验、`group_layout_descs` 反序列化都缺失 | 需要移植 + 适配新类名 |
| `lmcache/v1/multiprocess/modules/server_transfer.py` | `TransferStrategy.prepare_store`/`commit_store` 的 context 参数仍是单一 `EngineDrivenContextMetadata`，不支持按 group 列表 | 同上 | 需要移植 `_reserve_write_by_group` 等价逻辑 |
| `lmcache/v1/distributed/serde/kvweave/kvweave_config.py` | — | 已实现 | 配置、协议常量、dtype/scaling 映射、P&D 矩阵和 scale id 管理 |
| `lmcache/v1/distributed/serde/kvweave/kvweave_serde.py` | — | 已实现 | attention KV 与 Mamba conv/ssm codec，统一使用 `_KVWeaveCodec` 类方法 |
| `lmcache/v1/distributed/l1_manager.py` + `l1_manager_protocol.py` + `memory_manager/{gds,l1}_memory_manager.py` | — | `is_variable_size()` 缺失 | **可直接照搬移植**，零冲突 |
| `lmcache/v1/distributed/storage_controllers/prefetch_controller.py` | — | lmcache-main 已经原生支持 `group_layout_descs: dict[int, MemoryLayoutDesc]`（`_reserve_load_buffers` 已按 `object_group_id` groupby） | **不需要按 patch 移植**，只需确保新量化代码产出的 `MemoryLayoutDesc` 符合现有分组预期 |
| `lmcache/v1/distributed/storage_manager.py` | — | 需要加 `LMCACHE_MP_L1_KVWEAVE_QUANT` 开关 + L2 serde 双重量化保护 + `is_l1_variable_size()` | **需要适配**：lmcache-main 把 adapter 构造逻辑重构到了 `_build_l2_adapter()`（被 `__init__` 和 `add_l2_adapter` 共用），逻辑要放这里而不是 patch 假设的 `__init__` 内联循环，否则运行时动态加的 adapter 不会被保护 |
| `lmcache/v1/multiprocess/server.py` | — | 启动日志缺失 | 可直接照搬移植 |
| `lmcache/integration/vllm/vllm_multi_process_adapter.py` | `ParallelStrategy` 缺少灵活构造签名；`LMCacheMPWorkerAdapter` 缺 `use_mla`/`is_first_rank_of_pp_group` | — | 需要适配：lmcache-main 的 `ParallelStrategy` 字段已变（`mla_only` 不是 `use_mla`，多了 `n_servers` 分片逻辑），且已有 `_normalize_adapter_init_args()` 兼容层，需先确认 patch 的目标是否已被这个新机制覆盖，避免重复实现 |
| `setup.py` / build 后端探测 | XPU 自动探测缺失 | — | **不需要移植**：已用 `setup_extensions/policy.py` + `build_profiles/{sycl,rocm,musa,cuda}.py` 的 profile/`detect()` 插件模式实现，功能等价 |

---

## Phase 0：前置确认（已完成，2026-08-13）

1. **[已确认] 真正承载 hybrid 元信息的 IPC payload 与消息路径**
   - `RegisterEngineDrivenContextPayload` 仅用于 engine-driven 路径：`RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`。
   - GPU 主路径使用另一条消息：`RequestType.REGISTER_KV_CACHE`，其 payload 已包含 `list[EngineGroupInfo]`（由 `create_engine_group_infos_from_vllm` 产出并传入）。
   - 结论：`group_layout_descs` / `enable_l1_kvweave_quant` 不应加到 GPU 注册消息上，应加在 engine-driven 对应 payload/流程里。

2. **[已确认，2026-08-17 更新] `upstream` conda 环境已装好 kvweave native extension + editable lmcache/vllm**
   - `conda activate upstream` 后：
     - `kvweave`/`kvweave_quant` 可正常 import，`.so` 位于 `/home/user/miniforge3/envs/upstream/lib/python3.12/site-packages/kvweave/`。
     - `lmcache` 是 editable 安装，指向本目录下的 `lmcache-main`。
     - `vllm` 是 editable 安装，指向本目录下的 `vllm-v0.26.0`（版本 `0.26.0+xpu`）。
     - `torch_device_type=xpu`，说明 torch 也已装好 XPU 后端。
   - 结论：**Phase 3/4 的验证依赖已就绪**，不再是"可开发、暂不可验证"状态；后续所有 Phase 的单测/集成验证都应在 `conda activate upstream` 环境下跑，且改动会直接体现在这两个 editable 安装的源码目录里（改了就是改了当前环境用的代码，不需要重装）。

3. **[已确认] `_normalize_adapter_init_args()` 已覆盖旧/新构造参数兼容**
   - 该兼容层已处理两种入口：
     - 新接口：直接传 `ParallelStrategy`。
     - 旧接口：`(kv_world_size, kv_worker_id, legacy_block_size)`。
   - 同时 `ParallelStrategy` 已演进到 `mla_only + n_servers` 语义。
   - 结论：构造参数兼容问题已被 `_normalize_adapter_init_args()` 完全覆盖；原计划设想的 adapter 属性补充（`use_mla` / `is_first_rank_of_pp_group`）经 2026-08-18 复核，在当前代码里没有调用方，已确认不适用（见文末"建议实施顺序"一节，原 Phase 6 已删除）。

### Phase 0 结果对后续 Phase 的影响

- Phase 1/2/5 可按计划推进（Phase 1 已完成，见下）。
- Phase 3/4 codec 核心已完成并在 `upstream` 环境验证；Phase 5/6 已将其接入 worker/server 的 engine-driven 链路。
- 所有后续开发、跑测试统一使用 `conda activate upstream`。

---

## vLLM v0.26.0 KV Cache Shape 对照（Full / Hybrid）

这一节把“vLLM 侧真实 shape”与“LMCache（lmcache-main）迁移后目标形态”放在一起，便于后续 Phase 1/2/3/4 对齐。

### A. Full Attention

1. **vLLM v0.26.0（事实）**
   - 标准 attention spec 使用 `AttentionSpec/FullAttentionSpec`，其 page 字节大小为：
     - `2 * block_size * num_kv_heads * head_size * dtype_size`（未考虑量化打包时的额外规则）。
   - 典型后端（如 flash-attn）逻辑 shape 为：
     - `(num_blocks, num_kv_heads, block_size, 2 * head_size)`
     - K/V 打包在最后一维（`2 * head_size`）。
   - 物理 stride/layout 可因后端而异（NHD/HND），但逻辑维度语义一致。

2. **LMCache（lmcache-main，迁移后目标）**
   - **LMCache-driven 路径（GPU）**：保持按 vLLM engine-group 传输，`tokens_per_block = kv_cache_spec.block_size`，不改 Full Attention 的语义。
   - **Engine-driven 路径（XPU/CPU）**：注册元信息使用统一 layout：
     - 非 MLA：`[2, num_layers, chunk_tokens, hidden_dim]`
     - MLA：`[num_layers, chunk_tokens, hidden_dim]`
     - 其中 `hidden_dim = num_heads * head_size`。
   - Full Attention 迁移重点不是改 shape 定义，而是确保 Phase 1 多 group 下按 group 正确 gather/scatter 与 reserve。

### B. Hybrid Attention（Attention + Mamba）

1. **vLLM v0.26.0（事实）**
   - Mamba 层在 v1 接口中用 `MambaSpec(shapes, dtypes, block_size, ...)` 描述。
   - 对 Qwen3.5/GDN 这条线，state 仍是二元：
     - `conv_state`（`self_kv_cache[0]`）
     - `ssm_state`（`self_kv_cache[1]`）
   - 即：`kv_cache` 在运行时是 `[conv_state, ssm_state]` 两个 tensor，而不是 attention 的单一 K/V 打包 tensor。

2. **LMCache（lmcache-main，迁移后目标）**
   - 对 Hybrid 模型，最终要同时满足：
   - Attention 子组：仍按 attention group 的 `tokens_per_block` 与既有 group 机制处理。
   - Mamba 子组：补齐 `mamba_real_layout`（conv/ssm 的 `byte_offset/byte_length/dtype/shape`）并随 `EngineGroupInfo` 传输。
   - Engine-driven（XPU）侧要从“单 group”升级到“按 group 逐组 gather/scatter”，否则 hybrid 多 group block_ids 无法工作。
   - 量化 codec（Phase 3/4）依赖上述布局信息：
     - attention 走 KVWeave codec；
     - mamba 按 conv/ssm 子状态拆分后量化与回填。

### C. 本次迁移验收时应看到的 shape/布局结果

1. **Full Attention only**：
   - 能稳定以单 group 或多 attention group 运行；
   - Engine-driven 下 `MemoryLayoutDesc` 与 chunk gather/scatter shape 一致；
   - 不出现 block_size 被误判（尤其是子分页/压缩场景）。

2. **Hybrid Attention（Qwen3.5 类）**：
   - group infos 中能区分 attention 与 mamba；
   - mamba 组携带可用的 `mamba_real_layout`（conv+ssm）；
   - XPU engine-driven 路径可处理多 group block_ids；
   - 开启 `LMCACHE_MP_L1_KVWEAVE_QUANT=1` 后，attention 与 mamba 的序列化/反序列化均能 round-trip。

---

## Phase 1：engine-driven（原 non_gpu）传输路径的多 group 支持 —— 本次迁移中最实质的新工作 —— ✅ 已完成（2026-08-13）

**完成情况**：按下面的设计方案全部落地，`upstream` 环境下验证通过。

- `custom_types.py`：`RegisterEngineDrivenContextPayload` 加了 `engine_group_infos` 字段。
- `worker_transfer.py`：新增 `_kv_caches_for_group`/`_blocks_per_chunk_for_group`/`_group_chunk_shape`/`_iter_transfer_groups` 模块级 helper，`EngineDrivenTransferContext` 新增 `iter_transfer_groups()`/`group_chunk_shape()` 公开方法供子类复用；`register()`/`submit_store()`/`submit_retrieve()` 按 group 循环 gather/scatter，拼接成一个 flat 列表，仍然只发一次 MQ round-trip。
- `async_engine_driven.py`：`AsyncEngineDrivenTransferContext.submit_store()` 的后台线程闭包内部改成按 group 循环 gather（pickle 模式下每组用 `group_chunk_shape()` 算出自己的 staging buffer 形状，不再假设所有组同形状）。
- `engine_driven_transfer.py`（server 端）：`EngineDrivenContextEntry` 加 `metadata_by_group` 字段；`register_kv_cache_engine_driven_context` 按 group 各自的 `layer_indices`/`tokens_per_block` 构造独立的 `EngineDrivenContextMetadata`，并通过 `AttnWindowDesc(num_chunks_in_sw=[-1]*num_groups)` + `group_layout_descs` 接入已有的 `layout_desc_registry.register()` 多组机制；新增 `_resolve_obj_keys`/`_resolve_group_obj_keys`，`prepare_store`/`commit_store`/`prepare_retrieve`/`commit_retrieve` 都按 group 循环调用现有的（未改动的）`server_transfer.py` API，各自处理结果拼接（`slots`/`chunk_indices` 带偏移量拼接；pickle payload 先整体 unpickle 再按组切片重新 pickle，避免位置错位）。
- **后续热修（同日）**：`worker_transfer.py` 与 `async_engine_driven.py` 的 gather/scatter 调用不再强绑 `self._engine_kv_format`，改为每个 group 按 `group_kv_caches` 自动探测，修复混合组格式误用导致的 `IndexError`。
- **`server_transfer.py` 确认未改动一行**，符合设计目标。
- 测试：新建 `tests/v1/multiprocess/test_engine_driven_multi_group.py`（16 个用例，覆盖 worker 侧纯函数 helper + `submit_store`/`submit_retrieve` 端到端拼接）和 `tests/v1/multiprocess/test_engine_driven_transfer_multi_group_server.py`（8 个用例，覆盖 server 侧注册/拍平/store/retrieve 循环）。全部通过，且不依赖访问模块私有属性（`# noqa: SLF001`），改用可观测的公开行为断言。
- 回归验证：`tests/v1/multiprocess/` 全量 582 个测试通过；`tests/v1/`（除 `cache_controller`/`mp_coordinator`/`storage_backend/test_gds_backend.py` 这几个因环境缺 `pytest-asyncio` 插件而失败、以及 `test_xpu_sglang_connector.py` 12 个失败已确认是改动前就存在的、与本次工作无关的 XPU 原生 kernel 环境问题）全部通过，共 3243 通过。

---

**以下是原始设计记录（写代码前定的方案，代码落地后与此一致）**：

**状态（2026-08-13）**：已通读 `worker_transfer.py`（`EngineDrivenTransferContext`/`AsyncEngineDrivenTransferContext`）、`transfer_context/base.py`（`gather_paged_kv_to_cpu`/`scatter_cpu_to_paged_kv`/`EngineDrivenContext`）、`transfer_context/shm.py`/`pickle.py`、server 端 `modules/engine_driven_transfer.py`/`modules/server_transfer.py`、`engine_context.py`（`resolve_obj_keys`）、`custom_types.py`（`RegisterEngineDrivenContextPayload`）、`protocols/engine.py`（wire 协议定义）以及现有测试 `tests/v1/multiprocess/test_engine_driven_transfer.py`/`test_async_engine_driven_transfer_context.py`。下面是重新设计后的方案，**与最初写这份计划时设想的方案不同**（那时还没读过 lmcache-main 这几个文件的实际实现，是照着 patch 的思路直接套的）。写代码前先把设计定下来。

### 现状确认的关键事实

1. CUDA 路径（`LMCacheDrivenTransferContext`）通过 IPC handle 传输，`STORE`/`RETRIEVE` 协议的 `block_ids: list[list[int]]` **已经是**按 LMCache group 索引的多 group 设计，天然支持 hybrid。
2. CPU/XPU 路径（`EngineDrivenTransferContext`，`AsyncEngineDrivenTransferContext` 是其异步子类，XPU 因为支持 Stream/Event/pinned memory 实际用的是这个异步版本）目前硬编码只支持单 group：`worker_transfer.py:177` 的 `_single_group_block_ids()` 在 `len(block_ids) != 1` 时直接抛 `RuntimeError`；`register()` 的 docstring（`worker_transfer.py:675-679`）明确写着 `engine_group_infos` 目前是 no-op。这是唯一需要真正写新代码的架构性 gap。
3. `gather_paged_kv_to_cpu`/`scatter_cpu_to_paged_kv`（`transfer_context/base.py:317`/`562`）本身已经是"给一个 `kv_caches: dict[str, Tensor]` 子集 + 该子集自己的 `block_ids`/`blocks_per_chunk`，产出/消费一组 chunk tensor"的通用工具，不需要改——按 group 切好 `kv_caches`/`block_ids`/`blocks_per_chunk` 后可以直接复用，调用 N 次（N=group 数）即可。
4. Server 端 `engine_context.py:279` 的 `MPCacheServerContext.resolve_obj_keys(key, object_group_ids)` **已经原生支持多 group**，返回 `list[list[ObjectKey]]`，按 group 索引。`server_transfer.py` 里 `TransferStrategy.prepare_store`/`commit_store` 等方法接受的 `resolve_obj_keys: Callable[[key], list[ObjectKey]]` 回调本身是不关心 group 概念的纯 flat 列表——**只要调用方（`engine_driven_transfer.py`）在回调里把多 group 的 `resolve_obj_keys(key, [0..N-1])` 结果拍平拼接成一个 flat list，`server_transfer.py` 完全不用改**。
5. `ObjectKey`（`distributed/api.py:83`）已经带有 `object_group_id: int = 0` 字段（默认 0，向后兼容）。这是能把"多个 group 的 obj_keys 混在一个 flat list 里传输，同时保留分组信息"的关键——不需要新增字段。
6. Wire 协议（`protocols/engine.py`）里 `PREPARE_STORE`/`COMMIT_STORE`/`PREPARE_RETRIEVE`/`COMMIT_RETRIEVE`/`REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT` 的 payload 都是位置参数列表（`[KeyType, int]` 等），不是可扩展的 struct。改这几个协议的 payload 形状（比如加一个 `group_index: int`）会牵涉到 `shm.py`/`pickle.py`/`engine_driven_transfer.py`/`protocols/engine.py`，以及一个独立的 benchmark 工具 `cli/commands/bench/server_bench/helpers.py`（它绕开 `EngineDrivenContext` 直接拼 RPC payload）。改动面偏大，且这类协议 payload 改动通常意味着新旧 worker/server 版本不兼容，风险较高。

### 设计决策：**不改 wire 协议**，每次 store/retrieke 仍是一次 MQ round-trip，多 group 数据拼在一个 flat 请求里传输

放弃了最初设想的"每个 group 一次独立 MQ round-trip"方案（那个方案需要给 `instance_id` 加虚拟复合 key 或者给协议加 `group_index` 字段，见下面"考虑过但放弃的方案"）。改为：

- **worker 侧**（`EngineDrivenTransferContext`/`AsyncEngineDrivenTransferContext`）：`register()` 时保存 `self._engine_group_infos: list[EngineGroupInfo]`（为空时保持现在的单 group 行为，完全向后兼容）。`submit_store`/`submit_retrieve` 里，如果有多 group，按 group 顺序切出每组的 `kv_caches` 子集（按 `layer_indices` 过滤）、`block_ids[group_idx]`、按 `tokens_per_block` 算出该组的 `blocks_per_chunk`，依次调用 `gather_paged_kv_to_cpu`/`scatter_cpu_to_paged_kv`，把每组产出的 chunk tensor 列表按顺序拼接成一个 flat `list[Tensor]`（第 0..k0-1 个是 group0 的 chunk，第 k0..k0+k1-1 个是 group1 的，以此类推）。**仍然只发一次 `PREPARE_STORE`/`COMMIT_STORE` RPC**，`chunks` 参数就是这个拼接后的 flat 列表。
- **server 侧**（`engine_driven_transfer.py`）：`register_kv_cache_engine_driven_context` 收到 payload 里的 `engine_group_infos` 后，给这个 instance 存一份"每组的 chunk 数/`EngineDrivenContextMetadata`"信息（沿用 patch 思路的 `metadata_by_group`）。`resolve_obj_keys` 从单 group 的 `self._ctx.resolve_obj_keys(key, [0])[0]` 换成多 group 版本：调用 `self._ctx.resolve_obj_keys(key, list(range(num_groups)))`，然后按 group 顺序拍平拼接成一个 flat list（拼接顺序必须和 worker 侧拼接 chunk 的顺序一致，都是 group 0, 1, 2... 顺序）——**每个 obj_key 自带 `object_group_id`，所以拼接后仍能在下游区分是哪个 group 的**。这个拼接后的 flat obj_keys 列表原样传给 `server_transfer.py` 现有的 `TransferStrategy.prepare_store(..., resolve_obj_keys=...)`，`server_transfer.py` **不需要改一行**，因为它本来就只关心"一个 flat 列表"。
- **一个需要留意的点**：`server_transfer.py` 里 `context: EngineDrivenContextMetadata` 参数目前只有一份 `layout_desc`（单一 `MemoryLayoutDesc`），但 hybrid 场景下不同 group 的 chunk shape 不同（attention 组 `num_layers` 更大，mamba 组更小）。`reserve_write(obj_keys, layout_desc, mode)` 对整批 `obj_keys` 只应用一个 `layout_desc`，这在多 group 且 shape 不同时会出错。所以这一步**确实需要**在 `engine_driven_transfer.py` 里，把单次 `prepare_store`/`commit_store` 调用按 group 拆成 N 次分别调用 `strategy.prepare_store(...)`（每次传该 group 自己的 `EngineDrivenContextMetadata`、该 group 切片后的 `resolve_obj_keys` 闭包），再把 N 次调用的 `PrepareStoreResponse.context["slots"]`/`["chunk_indices"]` 按顺序拼接成一个总的 response 返回给 worker（`chunk_indices` 要整体偏移，让 worker 侧仍能用一份 flat `chunks` 列表定位）。**这样 `server_transfer.py` 本身仍然不用改**，只是 `engine_driven_transfer.py` 在服务端多循环几次、拼接结果。

### 考虑过但放弃的方案

1. **每组一次独立 MQ round-trip，用虚拟复合 `instance_id`（`instance_id * K + group_idx`）区分**：会破坏 PING 存活检测——PING 用的是 worker 的真实 `instance_id`（见 `management.py:137` 的 `ping(instance_id)` → 各 target 的 `touch_instance(instance_id)`），如果 `_engine_driven_contexts` 字典改成用虚拟复合 key 存储，PING 永远刷新不到这些虚拟 key，会导致组注册被误判超时回收。放弃。
2. **给协议 payload 加 `group_index: int` 字段，每组一次独立 round-trip**：需要改 `protocols/engine.py` 的 5 个 `ProtocolDefinition`、`shm.py`/`pickle.py`/`engine_driven_transfer.py` 的所有调用点，以及独立的 `cli/commands/bench/server_bench/helpers.py`（它直接手工拼 RPC payload，不经过 `EngineDrivenContext`）。改动面大、且是 wire 协议破坏性变更（worker/server 版本必须同步升级）。当前"多组数据拼一个 flat 请求"的方案能在不碰协议的前提下解决问题，优先选用；仅当未来发现 N 次独立 round-trip 有实质性能收益（比如减少一次性内存峰值）时才重新考虑。

### 具体改动清单

1. `lmcache/v1/multiprocess/custom_types.py`
   - `RegisterEngineDrivenContextPayload`（第 123 行）追加字段 `engine_group_infos: list[EngineGroupInfo] = msgspec.field(default_factory=list)`，空列表表示非 hybrid 单 group（向后兼容）。

2. `lmcache/v1/multiprocess/transfer_context/worker_transfer.py`（`EngineDrivenTransferContext`，633-841 行附近）
   - `register()`：把 `engine_group_infos` 存到 `self._engine_group_infos`（不再是 no-op），并把它编码进 `RegisterEngineDrivenContextPayload`。**不需要**额外存 `chunk_size_tokens`：`submit_store`/`submit_retrieve` 每次调用本来就会传入 `blocks_in_chunk`（默认组的 blocks-per-chunk），乘以 `register()` 时算好的默认组 `block_size` 现算即可，不用在 `register()` 里持久化一份冗余状态。
   - 新增 helper（对齐 patch 思路但适配 lmcache-main 现有签名）：
     - `_kv_caches_for_group(kv_caches, group_info) -> dict[str, Tensor]`：按 `layer_indices` 过滤。
     - `_blocks_per_chunk_for_group(group_info, default_blocks_in_chunk, default_block_size) -> int`：`tokens_per_block <= 0` 时回退 `default_blocks_in_chunk`；否则用调用时刻现算的 `chunk_size_tokens = default_blocks_in_chunk * default_block_size`，要求其能被该组 `tokens_per_block` 整除，返回商。
     - `_iter_transfer_groups(kv_caches, block_ids, blocks_in_chunk)`：`self._engine_group_infos` 为空时 yield 一次 `(None, kv_caches, block_ids[0], blocks_in_chunk)`（完全retain旧行为）；非空时按顺序对每个 group yield 过滤后的四元组。
   - `submit_store`/`submit_retrieve`：改成对 `_iter_transfer_groups()` 循环调用 `gather_paged_kv_to_cpu`/`scatter_cpu_to_paged_kv`，把多组的 chunk 列表按顺序 `extend` 成一个 flat 列表，仍只发一次 `PREPARE_STORE`+`COMMIT_STORE`（或 `PREPARE_RETRIEVE`+`COMMIT_RETRIEVE`）。
   - 去掉 `_single_group_block_ids` 的强制单 group 校验，或保留作为 `_engine_group_infos` 为空时内部的 fallback 实现（复用其逻辑而不是删除，减少改动面）。

3. `lmcache/v1/multiprocess/transfer_context/async_engine_driven.py`（`AsyncEngineDrivenTransferContext`）
   - `register()` 直接继承基类不用改。`submit_store()` 目前用 `_single_group_block_ids(block_ids)` 拿到 flat block_ids 后一次性 gather——需要同步改成按 `_iter_transfer_groups()`（从基类继承的方法）循环 gather，把多组 chunk 拼接后再整体 commit。注意这里的 gather 是在后台线程里做的（`_prepare_gather_and_commit` 闭包），循环逻辑要在闭包内部做，不能破坏原有的三阶段 prepare/gather/commit 异步结构。

4. `lmcache/v1/multiprocess/modules/engine_driven_transfer.py`（server 端）
   - `EngineDrivenContextEntry` 增加 `metadata_by_group: list[EngineDrivenContextMetadata]` 字段（空列表表示非 hybrid，回退到现有单 `metadata` 字段）。
   - `register_kv_cache_engine_driven_context`：如果 payload 带了非空 `engine_group_infos`，为每个 group 算一份 `EngineDrivenContextMetadata`（`num_layers=len(group_info.layer_indices)`、`block_size=group_info.tokens_per_block or payload.block_size`），存进 `metadata_by_group`。
   - `_resolve_single_group_obj_keys` 换成 `_resolve_obj_keys(key, instance_id)`：查这个 instance 注册时的 group 数（0 或空→当作 1），调用 `self._ctx.resolve_obj_keys(key, list(range(num_groups)))`，按 group 顺序拍平拼接。
   - `prepare_store`/`commit_store`/`prepare_retrieve`/`commit_retrieve`：如果 `metadata_by_group` 非空，按 group 循环调用 `strategy.prepare_store(..., context=metadata_by_group[g], resolve_obj_keys=<该组切片闭包>)`，把每组返回的 `slots`/`chunk_indices` 按顺序拼接（`chunk_indices` 要整体加上前面各组已用掉的 chunk 数偏移量）成一个总 response；否则走现有单 group 路径不变。
   - 顺带把 patch 2 的 KVWeave 相关字段（`enable_l1_kvweave_quant`、`group_layout_descs`）接进来，见 Phase 3/5 依赖，这部分等 Phase 3/4 的量化编解码器写完后再回来接。

5. `lmcache/v1/multiprocess/modules/server_transfer.py`
   - **确认不需要改动**（结论有变化，之前的草稿以为要加 `_reserve_write_by_group`，重新设计后不需要）：只要 `engine_driven_transfer.py` 按 group 分别调用现有的 `strategy.prepare_store`/`commit_store`（每次一个 group 自己的 `context`/`resolve_obj_keys`），`server_transfer.py` 现有的单一 `EngineDrivenContextMetadata` 签名完全够用，不用改类型签名或加新 helper。

**验收**：新增/改写测试覆盖：worker 侧多 group gather/scatter 拼接（对应旧 patch 里 `test_worker_transfer.py` 的用例思路，但适配新架构）、server 侧 `_resolve_obj_keys` 多 group 拍平、`prepare_store`/`commit_store` 按 group 循环且 `chunk_indices` 偏移正确、`register_kv_cache_engine_driven_context` 携带 `engine_group_infos` 时正确存储 `metadata_by_group`。全部基于 `tests/v1/multiprocess/test_engine_driven_transfer.py`/`test_async_engine_driven_transfer_context.py` 现有的 mock/fixture 风格扩展，而不是照抄旧 patch 的 `test_non_cuda_data_transfer.py`（该测试文件对应的模块结构在 lmcache-main 里已经不存在，仅供设计思路参考）。

---

## Phase 2：移植 KVWeave 专用的 hybrid 元信息（低风险，机械式移植）—— ✅ 已完成（2026-08-17）

**目标**：让 `EngineGroupInfo` 能携带 `cache_category`（`"attention"`/`"mamba"`/`"unknown"`）和 `mamba_real_layout`（conv/ssm 各自的真实字节布局），为后续量化代码提供依据。

1. `lmcache/v1/multiprocess/group_view.py`
   - 新增 `MambaSubStateWireLayout(msgspec.Struct, frozen=True)`：`byte_offset`/`byte_length`/`dtype_str`/`shape`。
   - 给 `EngineGroupInfo` 追加两个字段：`cache_category: str = "unknown"`、`mamba_real_layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout] | None = None`。
   - 直接追加在现有字段（`sw_size_tokens`）之后即可，msgspec Struct 字段是追加式的，不影响现有序列化兼容性。

2. `lmcache/integration/vllm/kv_cache_group_edits.py`
   - 新增 `MambaSubStateLayout`（dataclass，非 wire 版）和 `MambaRealLayout`（含 `pad_byte_offset`/`pad_byte_length` property）。
   - 给 `_MambaPageViewEdit` 加 `real_layout(self, spec: KVCacheSpec) -> MambaRealLayout` 方法。
   - **注意结构性差异**：lmcache-main 的 `_MambaPageViewEdit.apply()` 签名已经是 `apply(self, spec, kv_cache, _layout_hints)`（3 参数，第三个参数名为 `_layout_hints`，下划线前缀、当前未使用），不是 patch 里的旧 2 参数版本。`real_layout()` 本身只需要 `spec` 参数，不受影响，但插入位置要跟当前 3 参数 `apply()` 的写法保持一致的代码风格。

3. `lmcache/integration/vllm/kv_cache_groups.py`
   - 新增 `_cache_spec_category(spec) -> str`、`_mamba_real_layout_wire(spec) -> tuple[...] | None`、`_merge_layer_cache_categories(...)`、`_merge_layer_mamba_real_layouts(...)`。
   - 在 `create_engine_group_infos_from_vllm` 里补充 `per_layer_cache_category`/`per_layer_mamba_real_layout` 的收集逻辑，并在最终构造 `EngineGroupInfo(...)` 时传入 `cache_category=`/`mamba_real_layout=`。
   - **注意结构性差异**：lmcache-main 的分组 identity 对象已经从 tuple 索引（patch 里的 `identity[4]`）重构为属性访问（`identity.engine_group_idx`）。新增的两个 kwargs 要按 `identity.engine_group_idx` 风格取值，不要照抄 patch 的 `identity[4]`。

### Phase 2 实际落地变更（2026-08-17）

- `lmcache/v1/multiprocess/group_view.py`：新增 `MambaSubStateWireLayout`，并为 `EngineGroupInfo` 增加 `cache_category` 与 `mamba_real_layout` 字段；保留旧 payload 的默认值兼容性。
- `lmcache/integration/vllm/kv_cache_group_edits.py`：新增 `MambaSubStateLayout`、`MambaRealLayout` 及 `_MambaPageViewEdit.real_layout()`，准确描述 conv/ssm 的 dtype、shape、字节范围和尾部 padding。
- `lmcache/integration/vllm/kv_cache_groups.py`：新增 cache category 判断、Mamba wire layout 生成及 per-layer metadata 合并逻辑，并在 `create_engine_group_infos_from_vllm()` 中输出对应元信息。
- 测试：新增 `tests/v1/test_kv_cache_group_edits.py`，并补充 group view 的 msgspec round-trip 与 vLLM attention/Mamba 分类测试。
- 验证：`upstream` 环境下 Phase 2 相关测试共 33 个通过，`git diff --check` 通过。

**验收**：可以对着 `tests/v1/test_kv_cache_groups.py` / `tests/v1/test_vllm_kv_cache_groups.py`（如果里面有 mamba 相关 fixture）跑一下，确认没有破坏现有 group 划分逻辑；如果没有对应 real_layout 的测试用例，需要新增。

---

## Phase 3：新建 KVWeave L1 量化编解码器——✅ 已完成（2026-08-17）

依赖 Phase 2（`mamba_real_layout`/`cache_category`）和 Phase 0 的 native extension 确认。

1. 新建 `lmcache/v1/distributed/serde/kvweave/kvweave_config.py`：基本可以照搬 patch 内容（`KVWeaveCodecConfig` dataclass + P&D 矩阵生成/加载逻辑），这个文件不依赖 lmcache 内部类型，迁移风险最低。
2. 新建 `lmcache/v1/distributed/serde/kvweave/kvweave_serde.py`：照搬 `_KVWeaveCodec` 类，但要检查：
   - `from lmcache.v1.memory_management import MemoryFormat` 的 import 路径在 lmcache-main 是否还有效（如果 `MemoryFormat` 挪了位置要调整）。
   - `MemoryLayoutDesc` 的 import 路径（`lmcache.v1.distributed.api`）在 lmcache-main 是否还成立。
   - 这个文件本身不直接依赖 `NonGpuContextMetadata`，可以独立于 Phase 1 先写。

### Phase 3 实际落地变更（2026-08-17）

- 参考 `lmcache-v0.4.7-mp-hybrid-to-kvweave.patch`，新增 `lmcache/v1/distributed/serde/kvweave/kvweave_config.py`，提供 `KVWeaveCodecConfig`、P&D 矩阵生成/加载和线程安全缓存。
- 新增 `lmcache/v1/distributed/serde/kvweave/kvweave_serde.py`，实现 `_KVWeaveCodec`：支持 `[T, 2, H]` 与 `[2, L, T, H]` attention-shaped tensor、KVW3 4bit payload、raw fallback、scale blob、反量化回填和序列化大小估算。
- `_KVWeaveCodec` 同时承载 attention 与 Mamba 方法，不新增独立 `MambaCodec` 类，也不保留重复的模块级 Mamba 实现。
- codec 通过 `from kvweave import kvweave_quant` 调用 native `kvweave_serialize_chunk` / `kvweave_dequantize_chunk_into_4d`，与当前 `upstream` 环境的 extension 布局一致。
- 新增 `tests/v1/distributed/serde/test_kvweave_serde.py`，覆盖 P&D 配置、3D/4D quant round-trip、raw round-trip、大小上界和非法 shape。
- 验证：Phase 3 codec 测试 `6 passed`。

**验收**：已通过独立的 `_KVWeaveCodec.serialize_tensor`/`deserialize_tensor` round-trip
单测验证 attention-shaped tensor，且不依赖 MP 传输链路。

---

## Phase 4：新建 Mamba conv/ssm 量化——codec 核心已完成（2026-08-17）

依赖 Phase 2（`mamba_real_layout` 提供 conv/ssm 各自的字节布局和 dtype）。

1. 在 `lmcache/v1/distributed/serde/kvweave/kvweave_serde.py` 中实现 `split_mamba_chunk`/`merge_mamba_chunk`/`MambaChunkSplit` 及量化逻辑。这部分逻辑不依赖 lmcache 的传输框架类型，主要依赖 Phase 2 加的 `MambaSubStateWireLayout`。
2. `_KVWeaveCodec` 的 Mamba 方法已由 Phase 6 接到 `worker_transfer.py` 的 gather/scatter 前后；engine-driven worker 的 store/retrieve 已通过统一实例方法完成编码、解码和 round-trip。

### Phase 4 实际落地变更（2026-08-17）

- 在 `lmcache/v1/distributed/serde/kvweave/kvweave_serde.py` 的 `_KVWeaveCodec` 类中实现 `MambaChunkSplit`、`split_mamba_chunk()` 和 `merge_mamba_chunk()`，按 Phase 2 的 `MambaSubStateWireLayout` 恢复 conv/ssm 的真实 dtype、shape 和字节区域。
- 在 `_KVWeaveCodec` 类中实现 Mamba 单子状态 4bit payload：`quantize_mamba_substate_4bit()` / `dequantize_mamba_substate_4bit()`，调用 native `kvweave_serialize_chunk_state()` / `kvweave_dequantize_chunk_state()`。
- 在 `_KVWeaveCodec` 类中实现 `pack_mamba_payloads()` / `unpack_mamba_payloads()`，将 conv 与 ssm 两个量化 payload 安全封装为一个存储 blob。
- 新增 `tests/v1/multiprocess/test_mamba_quant.py`，覆盖 split/merge、非连续输入、边界校验、payload framing 和 conv/ssm native quant round-trip。
- 验证：Phase 4 测试 `7 passed`；与 Phase 3 codec 测试合计 `13 passed`。

**验收**：codec 核心已通过 split/merge 和 native quant round-trip 测试；Phase 6
进一步通过 engine-driven store/retrieve 的同步、异步 Mamba round-trip 测试验证真实接入。

---

## Phase 5：engine-driven 路径接入 L1 KVWeave 量化——✅ 已完成并修复两处正确性 bug（2026-08-18）

**范围说明（与本节早前草稿的差异）**：早前草稿曾计划把这一阶段收缩为"只保留 engine-driven 注册字段，删除 storage 层开关/`server.py`日志/`l1_manager.is_variable_size()`"等。实际推进后确认：worker 侧 `register()`/`submit_store()`/`submit_retrieve()` 要把 `LMCACHE_MP_L1_KVWEAVE_QUANT=1` 落到真正的量化数据路径（而不仅仅是注册 metadata），必须依赖 `storage_manager.is_l1_variable_size()`（对应 `l1_manager.is_variable_size()`）在 server 侧做双重量化保护，以及 `server.py` 的启动日志用于确认生产环境实际生效的量化开关。因此这些内容予以保留并完成，而不是删除。

**已落地的改动**：

1. `custom_types.py`：`RegisterEngineDrivenContextPayload` 保留 `group_layout_descs: list[SerializedMemoryLayoutDesc] | None` 与 `enable_l1_kvweave_quant: bool`；新增 `serialize_memory_layout_desc()`/`deserialize_memory_layout_desc()` 做 `MemoryLayoutDesc` 的 msgpack 编解码。
2. `worker_transfer.py`（`EngineDrivenTransferContext`）：
   - `register()` 按 group 计算原始 layout，若 `LMCACHE_MP_L1_KVWEAVE_QUANT=1` 且非 MLA，则用 `KVWeaveCodec.estimate_serialized_size()` 逐组比较量化后/原始字节数，只在量化更小时才把该组标记为量化（`_group_layout_descs`/`_group_raw_layout_descs`/`_kvweave_quant_enabled`），并把量化后的 `group_layout_descs` 编码进注册 payload。
   - `submit_store()`：量化组不再直接 gather 到 SHM/输出 buffer，而是先 gather 到原始形状的临时 buffer，再用 `KVWeaveCodec.serialize_tensor()` 编码后写入 SHM 槽位或新分配的 `uint8` chunk。
   - `submit_retrieve()`：**新增**对量化组的 `KVWeaveCodec.deserialize_tensor()` 解码步骤——按 `_group_raw_layout_descs` 分配真实形状/dtype 的目标张量，把取回的 `uint8` 编码字节解码回真实 KV 数值后再 `scatter_cpu_to_paged_kv()`。
3. `async_engine_driven.py`（`AsyncEngineDrivenTransferContext`，XPU 实际使用的异步 store 路径）：`submit_store()` 的三阶段后台 gather 循环补齐了与同步路径对称的量化逻辑（量化组一律先 gather 到 pinned 暂存区，再编码进 SHM 槽位或新 `uint8` chunk；`submit_retrieve()` 直接继承基类，天然获得上面的解码修复）。
4. `engine_driven_transfer.py`（server 端）：`payload.enable_l1_kvweave_quant` 时要求 `storage_manager.is_l1_variable_size()` 为真（否则拒绝注册）；`payload.group_layout_descs` 非空时按组反序列化为 `metadata_by_group`，否则退回按 `engine_group_infos` 现算 layout 的旧路径。
5. `storage_manager.py`：新增 `is_l1_variable_size()`（委托 `l1_manager.is_variable_size()`）；`_build_l2_adapter()` 里当 L1 量化开关打开时，禁止同一 adapter 再叠加 `serde_config.type == "kvweave"` 的 L2 serde 包装（防止 L1/L2 双重量化），并对其它 serde 类型跳过 `SerdeL2AdapterWrapper` 包装。
6. `l1_manager.py` + `memory_manager/{l1,gds}_memory_manager.py` + `l1_manager_protocol.py`：新增 `is_variable_size()`（DRAM 用 `MixedMemoryAllocator`/`LazyMemoryAllocator` 判断，GDS 固定返回 `False`）。
7. `server.py`：启动时若 `LMCACHE_MP_L1_KVWEAVE_QUANT=1`，记录一条 info 日志说明量化及 `LMCACHE_MP_KVWEAVE_LINEAR_QUANT_ENABLED` 的生效状态。
8. `kvweave_serde.py`/`kvweave_config.py`：把魔数、scale-id 分配、`_quantized_bytes` 等常量/工具方法收拢进 `KVWeaveCodecConfig`，消除 `_KVWeaveCodec` 内部的重复实现（纯重构，不改变外部行为）。

**本轮 review 中发现并修复的两处正确性 bug（均在验证阶段发现，已修复+补测试）**：

- `unpack_mamba_payloads()` 重构时误删了 `size > len(blob) - 4` 的越界校验，导致损坏/截断的 Mamba payload 会静默返回错误切片而不是报错。已恢复校验。
- **`submit_retrieve()`（同步 `EngineDrivenTransferContext` 与继承它的异步路径共用）此前完全没有反量化步骤**：量化组的取回数据是仍处于 KVWeave 编码状态的 `uint8` 字节，会被直接当作真实 KV 张量 `scatter_cpu_to_paged_kv()`，在任何开启 `LMCACHE_MP_L1_KVWEAVE_QUANT=1` 的部署下都会导致取出的 KV 数值损坏。已补上 `KVWeaveCodec.deserialize_tensor()` 解码步骤。
- 此外 `AsyncEngineDrivenTransferContext.submit_store()`（XPU 实际使用的路径）本身完全没有走量化逻辑——它有自己独立的 gather 循环，未复用同步路径新加的量化代码；同时初版补丁在"量化组+SHM 直写"分支里错误设置了 `used_shm_direct=True`（会在混合量化/非量化 group 场景下，误跳过对非量化 group 暂存 buffer 的释放）并把已写入 SHM 的量化 chunk 又重复追加进 `commit_store` 的 flat 列表。已重写为：量化组一律先 gather 到 pinned 暂存区、量化后立即释放暂存区、按 SHM 是否存在选择写入 SHM 槽位或追加新 `uint8` chunk，且不影响非量化 group 的 `used_shm_direct`/`staged_chunks` 记账。

**新增测试**：
- `tests/v1/multiprocess/test_custom_types.py::test_engine_driven_payload_roundtrip_preserves_kvweave_fields`
- `tests/v1/multiprocess/test_engine_driven_multi_group.py::test_submit_store_retrieve_round_trips_kvweave_quantized_groups`（同步路径，真实 gather/scatter + 真实 native 量化/反量化）
- `tests/v1/multiprocess/test_async_engine_driven_transfer_context.py::test_async_submit_store_quantizes_then_submit_retrieve_dequantizes`（异步路径，真实 gather/scatter + 真实 native 量化/反量化）

**验证**：`conda activate upstream` 环境下，`tests/v1/multiprocess/` 全量 606 passed / 32 skipped；`tests/v1/`（排除 `cache_controller`/`mp_coordinator`/`test_gds_backend.py`/`test_xpu_sglang_connector.py`，与 Phase 1 记录的排除范围一致）3946 passed / 451 skipped / 3 failed——这 3 个失败（`test_basic_check.py::TestMain`）经 `git stash` 验证在本阶段改动前就已存在（环境缺 `pytest-asyncio` 插件），与本阶段工作无关。

---

## Phase 6：修复 Mamba group 误用 attention KVWeave 量化导致的乱码——✅ 已完成（2026-08-18）

**背景**：真机跑 E2E（Qwen3.5 hybrid + XPU + MP + `LMCACHE_MP_L1_KVWEAVE_QUANT=1`）时发现输出变成乱码/重复文本。

**根因**：`worker_transfer.py::EngineDrivenTransferContext.register()`（927-994 行）的量化决策循环对所有 LMCache group（包括 Mamba group）一视同仁地套用 `KVWeaveCodec.estimate_serialized_size()`/`serialize_tensor()`——这是 Phase 3 专为 attention K/V 张量设计的通用 4bit 量化编解码器，从未检查 Phase 2 已经加好的 `EngineGroupInfo.cache_category`（`"attention"`/`"mamba"`/`"unknown"`）。

Mamba group 的原始 page-view chunk 形状恰好也是 `[2, layers, tokens, hidden]`（与 attention fused-K/V 形状撞车），能通过 `_estimate_shape()` 里 `kv_size==2` 的校验，但内容其实是 conv_state/ssm_state 按字节偏移打包的真实数据，不是浮点 K/V。被误当成 attention K/V 走 4bit 量化后，Mamba 的递归状态（尤其 `ssm_state`，这也是本文档反复强调的坑）被破坏，导致模型输出乱码。

`AsyncEngineDrivenTransferContext.submit_store()`（XPU 实际使用的异步路径，见 Phase 5）有自己独立的 gather 循环，未复用同步路径逻辑，同样的误用存在两份拷贝。`submit_retrieve()` 未被异步类重写，只有一处需要修。

Phase 4 已经写好了专用于 Mamba 的量化方法（`split_mamba_chunk`/`merge_mamba_chunk`/`quantize_mamba_substate_4bit`/`dequantize_mamba_substate_4bit`/`pack_mamba_payloads`/`unpack_mamba_payloads`，都在 `kvweave_serde.py` 的 `_KVWeaveCodec` 类）；本 Phase 将这些方法接入传输链路，并落实 `env_vars.md` 中设计的 `LMCACHE_MP_KVWEAVE_LINEAR_*`/`CONV_*`/`SSM_*` 系列环境变量。

### 实施结果

Phase 6 已完成：Mamba codec 配置已迁移到 `kvweave_config.py`，runtime 环境变量由
`KVWeaveRuntimeConfig.from_env()` 集中解析；同步和异步 engine-driven 路径均通过
`_KVWeaveCodec.encode_chunk()`/`decode_chunk()` 按 cache category 分派，避免 Mamba
状态误用 attention codec。相关单测覆盖 runtime 配置解析和 attention chunk 分派，
多 group 及异步 round-trip 测试也已更新为新实例方法结构。

### 实际落地变更

1. **按 group category 做量化决策**：`EngineDrivenTransferContext.register()` 只调用一次
   `KVWeaveRuntimeConfig.from_env()`，并保存 `cache_category`、Mamba wire layout、每组
   token/block 信息和 `mamba_options`。attention/unknown 组使用 attention codec；Mamba
   组在启用 `linear_quant_enabled` 且有真实 layout 时使用
   `estimate_mamba_serialized_size()` 与 `linear_max_size_ratio` 判断是否量化；缺少
   layout 时安全回退为原始传输。MLA attention 仍不进入 attention 量化分支。

2. **集中配置与 dataclass 归属**：`MambaCodecOptions` 已从 `worker_transfer.py` 移到
   `kvweave_config.py`，由 `MambaCodecOptions.from_env()` 解析 conv/ssm 的 scaling、RH
   和共享 asym。`KVWeaveRuntimeConfig.from_env()` 集中解析总开关、linear 开关、大小比率
   以及 attention codec kwargs；`worker_transfer.py::register()` 不再直接读取这些变量。
   `LINEAR_PRECOND` 和 `NUM_THREADS` 保持未接入，因为 Mamba native API 不提供对应参数，
   `NUM_THREADS` 仅属于 attention codec 配置。

3. **实例方法编解码**：`_encode_quantized_chunk()`/`_decode_quantized_chunk()` 模块级
   helper 未保留。`_KVWeaveCodec.encode_chunk()`/`decode_chunk()` 按 category 分派：Mamba
   路径执行 split、conv/ssm 独立量化或反量化、framing 和 merge；其它 category 复用
   attention 的 `serialize_tensor()`/`deserialize_tensor()`。同步
   `worker_transfer.py` 与异步 `async_engine_driven.py` 的 store 路径都调用同一 codec
   实例方法，retrieve 通过基类的 `decode_chunk()` 获得对称处理。

4. **大小估算与传输布局**：`estimate_mamba_serialized_size()` 使用
   `KVWeaveCodecConfig.quantized_bytes()` 以及 header/scale 保守余量，同时用于注册阶段
   的量化判定和 SHM/pickle 槽位规划，避免编码 payload 超出目标 slot。Mamba 的编码 blob
   仍由 `pack_mamba_payloads()` framing，解码时保留截断和长度校验。

5. **测试与验证**：`test_kvweave_serde.py` 覆盖 runtime 环境变量解析和
   `encode_chunk()`/`decode_chunk()` 的 attention 分派；Mamba codec、同步/异步
   engine-driven store/retrieve round-trip、混合 group 互不干扰及 payload 字段 round-trip
   测试位于对应的 multiprocess 测试文件中。以下 5 个直接涉及本次改动的测试文件已在
   `upstream` conda 环境下通过：`test_kvweave_serde.py`、`test_mamba_quant.py`、
   `test_custom_types.py`、`test_engine_driven_multi_group.py` 和
   `test_async_engine_driven_transfer_context.py`，合计 **56 passed / 4 skipped**。
   本次验证未运行 LMCache 其它目录的全量测试。

---

## 建议实施顺序

Phase 0（确认，已完成）→ Phase 1（多 group 传输，已完成）→ Phase 2（元信息移植，已完成）→ Phase 3（KVWeave attention codec，核心已完成）→ Phase 4（Mamba conv/ssm codec，核心已完成）→ Phase 5（engine-driven 路径接入 L1 KVWeave 量化，已完成，含 store/retrieve 量化-反量化 round-trip 修复）→ Phase 6（修复 Mamba group 误用 attention 量化导致乱码，已完成）。

原计划中还有两个 Phase（`vllm_multi_process_adapter.py` 适配、`lmcache_mp_connector.py` debug 可观测性）已于 2026-08-18 确认为不适用并删除：

- 前者想解决的构造参数兼容问题已被 lmcache-main 自带的 `_normalize_adapter_init_args()` 解决；原计划里提到要补的 `use_mla`/`is_first_rank_of_pp_group` 两个 adapter 属性，经全仓库搜索确认在当前代码里没有任何调用方（`use_mla` 语义已经通过 `ParallelStrategy.mla_only`/`adapter.mla_only` 落地；PP rank 判断走的是 vLLM 自身的 `get_pp_group()`，不经过 adapter），补上也不会被消费，纯属过时条目。
- 后者唯一的功能性修复（`Event(interprocess=True)` 的 XPU 兼容 fallback）已在 2026-08-13 落地并验证；剩余内容仅是"移植老 patch 的 debug 日志"，本身标注可选/非阻塞，且当前 `lmcache_mp_connector.py` 日志覆盖已经足够，无实际缺口。

每个 Phase 完成后建议先跑对应模块的单测，再跑一次端到端的 hybrid 模型（如 Qwen3.5）在 XPU 上、开启 `LMCACHE_MP_L1_KVWEAVE_QUANT=1` 的 MP 模式集成测试，确认 cache hit 后取出的 KV 数值正确（尤其关注 Mamba `ssm_state` 的 fp32 精度是否被误读成别的 dtype）。截至 2026-08-18，Phase 0-6 已完成；端到端 XPU 验证仍取决于对应硬件和模型环境。
