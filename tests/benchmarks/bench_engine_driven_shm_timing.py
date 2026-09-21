# SPDX-License-Identifier: Apache-2.0
"""Per-stage timing of the XPU engine-driven SHM store/retrieve path.

Manual timing script (run directly with ``python``, not through pytest):
mocks Qwen3.5-9B's real hybrid *registration* as observed in a real vLLM +
LMCache production log -- 1 attention group (8 layers, fused K/V,
``EngineKVFormat.NL_X_NB_BS_NH_CS``) and 3 Mamba groups (8 layers each, 24
Mamba/linear-attention layers total, identical real conv/ssm layout) -- to
measure, per :class:`EngineGroupInfo` bucketed by
``EngineGroupInfo.cache_category`` (``"attention"`` vs ``"mamba"``), each
real sub-stage of
``EngineDrivenTransferContext.submit_store``/``submit_retrieve``. The LMCache
chunk size is 1024 tokens (1 physical block/chunk for both categories),
matching that production log.

The script runs the *same* store stages and retrieve stages under two
scenarios, so quantization overhead can be compared directly against not
quantizing at all. Retrieve has no separate "read bytes back out of SHM"
stage: production's ``prepare_retrieve`` hands ``decode_chunk`` a zero-copy
``torch.frombuffer`` view straight over the SHM slot (see
``EngineDrivenContextShm._build_slot_tensors`` and
``EngineDrivenTransferContext._decode_group_chunks`` in
``worker_transfer.py``), so this script's dequantize stage reads directly
off the SHM slot too, matching that contract exactly rather than measuring
an extra copy production never makes:

Scenario A -- Mamba + attention quantization ENABLED:
    Stage 1  gather (device -> CPU, D2H copy) into a plain CPU buffer.
    Stage 2  Mamba: recover the real ``conv_state``/``ssm_state`` bytes out
             of the opaque, padding-including synthetic page view into
             pool-level split scratch, then release the raw gather scratch;
             (``_KVWeaveCodec.split_mamba_chunk``, see
             ``lmcache/integration/vllm/kv_cache_group_edits.py``'s
             ``_MambaPageViewEdit`` docstring for why the raw gathered chunk
             is not already conv/ssm-shaped); attention: real 4-bit fused
             K/V quantization via ``_KVWeaveCodec.encode_chunk`` directly
             (native ``kvweave_quant`` extension).
    Stage 3  Mamba: real 4-bit conv/ssm quantization directly from split
             scratch into the
             durable SHM slot; attention: fused K/V quantization directly
             into its durable SHM slot. The native API still returns an
             internal Python ``bytes`` payload, but no separate uint8 staging
             tensor is allocated.
    Stage 4  folded into Stage 3 for quantized groups.
    Stage R1 dequantize directly off the SHM slot into a pool-level scratch
             view (``decode_chunk(..., out=scratch)``). For Mamba this also
             re-adds padding by writing only real conv/ssm bytes to the
             scratch-backed opaque page; unused padding is left untouched.
    Stage R2 Mamba only: folded into Stage R1 by the latest production path.
    Stage R3 scatter (CPU -> device, H2D copy) using the dequantized chunk.

Scenario B -- Mamba + attention quantization DISABLED:
    Stage 1  same gather as scenario A.
    Stage 2/3 skipped (nothing to split/quantize for the wire).
    Stage 4  ``submit_store`` gathers *directly* into the SHM slot for an
             unquantized group, so stages 1 and 4 collapse into one D2H
             copy -- measured here by gathering straight into a real SHM
             slot tensor instead of a throwaway CPU buffer.
    Stage R1/R2 skipped (nothing was quantized, nothing to decode/re-pad).
    Stage R3 scatter using the raw (never quantized) gathered chunk.

Quantization-dependent stages require the native ``kvweave_quant``
extension (part of the ``kvweave`` package, e.g. installed in the
``dev_lmcache`` conda env). If it is not importable, Scenario A's
quantize/dequantize/re-pad stages are skipped with a printed note instead of
crashing; Scenario B is unaffected.

Requires a real XPU device (exits early otherwise). Run with:

    python tests/benchmarks/bench_engine_driven_shm_timing.py
"""

# Standard
from unittest.mock import MagicMock
import os
import statistics
import sys
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    AttentionPlaneLayout,
    KVWeaveCodecConfig,
    MambaCodecOptions,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import (
    MambaChunkSplit,
    _KVWeaveCodec,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo, MambaSubStateWireLayout
from lmcache.v1.multiprocess.posix_shm import (
    shm_create_readwrite,
    shm_munmap,
    shm_unlink,
)
from lmcache.v1.multiprocess.protocols.engine import (
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transfer_context.shm import EngineDrivenContextShm
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    GroupTransferPlan,
    _raw_gather_shape,
)

try:
    from kvweave import kvweave_quant  # noqa: F401

    _KVWEAVE_QUANT_AVAILABLE = True
except ImportError:
    _KVWEAVE_QUANT_AVAILABLE = False

# Qwen3.5-9B hybrid geometry, matching a real vLLM + LMCache production
# registration log (see MIGRATION_PLAN.md / test_kvweave_serde.py's
# ``test_estimate_mamba_serialized_size_compression_ratio_qwen35_9b`` for the
# per-layer conv/ssm numbers): 8 full-attention layers (fused K/V), 24 Mamba
# layers split into 3 real ``EngineGroupInfo`` groups of 8 layers each (as
# vLLM itself partitioned them), one physical block per 1024-token chunk for
# both categories.
_NUM_ATTENTION_LAYERS = 8
_ATTENTION_NUM_KV_HEADS = 4
_ATTENTION_HEAD_DIM = 256
_NUM_MAMBA_LAYERS = 24
_NUM_MAMBA_GROUPS = 3
_MAMBA_LAYERS_PER_GROUP = _NUM_MAMBA_LAYERS // _NUM_MAMBA_GROUPS
_BLOCK_SIZE = 1024
_BLOCKS_IN_CHUNK = 1
_CHUNK_SIZE_TOKENS = _BLOCKS_IN_CHUNK * _BLOCK_SIZE

# vLLM's real hybrid KV cache allocator unifies ``page_size_bytes`` across
# attention and Mamba groups so blocks are byte-interchangeable in one
# shared pool -- it inflates the attention *logical* block's token count
# until its byte size matches Mamba's fixed page size (see
# ``_SubpagedAttentionViewEdit.apply()``'s ``kernel_page_bytes * ratio ==
# spec.page_size_bytes`` assertion and "vLLM unifies page sizes across
# hybrid groups" in
# ``docs/design/integration/vllm/kv_cache_group_edits.md``). So one Mamba
# page's byte size must equal one attention layer's per-chunk byte size --
# NOT be independently derived from Mamba's own conv/ssm content, which is
# smaller and only occupies the *front* of the (real, padded) page.
_ATTENTION_LAYER_CHUNK_BYTES = (
    _BLOCK_SIZE * _ATTENTION_NUM_KV_HEADS * (2 * _ATTENTION_HEAD_DIM) * 2  # fp16
)

# One Mamba page's real (conv_state, ssm_state) content -- see
# ``_MambaPageViewEdit.real_layout``'s Qwen3.5-9B numbers reused from
# ``test_kvweave_serde.py``'s ``test_estimate_mamba_serialized_size_
# compression_ratio_qwen35_9b``: conv (kernel_history=3, conv_dim=8192)
# fp16, ssm (num_heads=32, head_dim=128, state_size=128) fp32. This content
# occupies only the front of the real (unified-size) page; the remainder is
# real trailing padding (see ``docs/design/integration/vllm/
# kv_cache_group_edits.md``'s "one padded page (``conv | ssm | pad``)").
_MAMBA_CONV_ELEMS = 3 * 8192
_MAMBA_CONV_BYTES = _MAMBA_CONV_ELEMS * 2  # fp16
_MAMBA_SSM_ELEMS = 32 * 128 * 128
_MAMBA_SSM_BYTES = _MAMBA_SSM_ELEMS * 4  # fp32
_MAMBA_PAGE_BYTES = _MAMBA_CONV_BYTES + _MAMBA_SSM_BYTES

# ``_MambaPageViewEdit.apply()`` re-strides one page into a synthetic
# ``[num_blocks, 2, block_size, 1, head_size]`` view in the conv state's
# dtype (fp16) -- see ``_synthetic_attention_shape``. ``head_size`` is
# solved so the synthetic view's byte size equals the unified
# ``page_size_bytes`` (``_ATTENTION_LAYER_CHUNK_BYTES``), not from
# ``_MAMBA_PAGE_BYTES`` (which would silently ignore real padding).
_MAMBA_SYNTHETIC_HEAD_SIZE = _ATTENTION_LAYER_CHUNK_BYTES // (2 * _BLOCK_SIZE * 2)
_MAMBA_REAL_PAGE_BYTES = 2 * _BLOCK_SIZE * _MAMBA_SYNTHETIC_HEAD_SIZE * 2  # fp16=2B
if _MAMBA_REAL_PAGE_BYTES != _ATTENTION_LAYER_CHUNK_BYTES:
    raise ValueError(
        f"Mamba synthetic page ({_MAMBA_REAL_PAGE_BYTES} bytes) does not "
        f"match the unified page size "
        f"({_ATTENTION_LAYER_CHUNK_BYTES} bytes) -- vLLM's hybrid "
        f"allocator requires attention and Mamba pages to be byte-equal"
    )
if _MAMBA_PAGE_BYTES > _MAMBA_REAL_PAGE_BYTES:
    raise ValueError(
        f"Mamba conv+ssm content ({_MAMBA_PAGE_BYTES} bytes) does not fit "
        f"in the unified page ({_MAMBA_REAL_PAGE_BYTES} bytes)"
    )

# Real per-block byte layout of the two Mamba sub-states at the *front* of
# the opaque synthetic page above -- conv_state first, ssm_state right
# after it, with real trailing padding making up the rest of the page. This
# is what ``_MambaPageViewEdit.real_layout()`` would compute from the
# engine's registered ``(conv_state, ssm_state)`` shapes/dtypes; the bench
# script reconstructs it directly since it never talks to vLLM.
_MAMBA_REAL_LAYOUT = (
    MambaSubStateWireLayout(0, _MAMBA_CONV_BYTES, "torch.float16", (3, 8192)),
    MambaSubStateWireLayout(
        _MAMBA_CONV_BYTES, _MAMBA_SSM_BYTES, "torch.float32", (32, 128, 128)
    ),
)
_NUM_ITERATIONS = 20
_NUM_WARMUP = 3
# Real raw chunk sizes this script gathers straight into SHM (see the
# unquantized Stage 4 path): attention (fused K/V) is
# _NUM_ATTENTION_LAYERS * _BLOCKS_IN_CHUNK * _BLOCK_SIZE *
# _ATTENTION_NUM_KV_HEADS * (2 * _ATTENTION_HEAD_DIM) * 2 bytes (fp16) =~ 32
# MiB; each 8-layer Mamba group's opaque page chunk is the *same* ~32 MiB
# (vLLM's unified ``page_size_bytes`` -- see ``_MAMBA_REAL_PAGE_BYTES``
# above -- makes one Mamba page byte-equal to one attention layer's
# per-chunk bytes). The pool must fit either one (offset
# ``_SHM_RAW_CHUNK_OFFSET`` onward) plus a small region at offset 0 for the
# quantized payload (attention's or a Mamba group's -- only one group's
# quantized payload occupies this region at a time, so they can share it),
# with no overlap between the two regions. 48 MiB leaves headroom for
# either category's 4-bit quantized payload at this geometry.
_SHM_QUANTIZED_REGION_BYTES = 48 * 1024 * 1024
_SHM_RAW_CHUNK_OFFSET = _SHM_QUANTIZED_REGION_BYTES
_SHM_POOL_SIZE = 512 * 1024 * 1024


def _attention_codec() -> _KVWeaveCodec:
    """Real Qwen3.5-9B attention K/V quantization codec: per-channel scaling,
    randomized-Hadamard off (matches ``test_kvweave_serde.py``'s ``_codec()``
    helper), asymmetric 4-bit, sized to this script's fused-K/V attention
    tensors (``_BLOCK_SIZE`` tokens/block, ``_ATTENTION_NUM_KV_HEADS`` heads,
    ``_ATTENTION_HEAD_DIM`` per head).
    """
    return _KVWeaveCodec(
        {
            "quantize": True,
            "qbit": 4,
            "scaling_method": "per_channel",
            "rh": False,
            "asym": True,
            "block_size": _BLOCK_SIZE,
            "num_kv_heads": _ATTENTION_NUM_KV_HEADS,
            "head_dim": _ATTENTION_HEAD_DIM,
        }
    )


def _mamba_codec_options() -> MambaCodecOptions:
    """Real Qwen3.5-9B Mamba quantization options (see ``test_mamba_quant.py``'s
    ``test_qwen35_mamba_page_quant_round_trip_is_deterministic``): per-channel
    scaling, randomized-Hadamard off (8192/128 conv/ssm widths keep RH on in
    production, but per-channel + RH off is the config validated in tests),
    asymmetric 4-bit for both sub-states.
    """
    return MambaCodecOptions(
        conv_scaling_method="per_channel",
        conv_rh=False,
        ssm_scaling_method="per_channel",
        ssm_rh=False,
        asym=True,
    )


def _attention_kv_caches() -> dict[str, torch.Tensor]:
    """Paged fused-K/V attention tensors, one per full-attention layer.

    Matches a real vLLM non-MLA blocks-first fused-K/V backend's per-layer
    registered shape ``[NB, BS, NH, CS]`` (``NL_X_NB_BS_NH_CS_Spec`` in
    ``lmcache/v1/gpu_connector/kv_format/specs/nl_x_nb_bs_nh_cs.py``), where
    the trailing content axis packs K and V together
    (``CS == 2 * _ATTENTION_HEAD_DIM``). Real full attention uses the
    engine's own ``block_size`` (here ``_BLOCK_SIZE``, matching
    ``_BLOCKS_IN_CHUNK`` paged blocks per LMCache chunk), so one chunk spans
    ``_BLOCKS_IN_CHUNK`` physical blocks.
    """
    shape = (
        _BLOCKS_IN_CHUNK,
        _BLOCK_SIZE,
        _ATTENTION_NUM_KV_HEADS,
        2 * _ATTENTION_HEAD_DIM,
    )
    return {
        f"attn_layer_{i}": torch.randn(
            shape, dtype=torch.float16, device=torch_device_type
        )
        for i in range(_NUM_ATTENTION_LAYERS)
    }


def _mamba_kv_caches() -> dict[str, torch.Tensor]:
    """Paged Mamba/GDN recurrent-state tensors, one per linear-attn layer.

    Each Mamba page is one recurrent-state snapshot -- not a sliding window
    of multiple blocks -- so one LMCache chunk maps to exactly *one* Mamba
    block regardless of ``_BLOCKS_IN_CHUNK`` (see
    ``EngineGroupInfo.recurrent_state`` and ``_MambaPageViewEdit``'s
    docstring: "equivalent for caching purposes to one block of a
    sliding-window attention layer with window == block_size"). Shaped as
    ``[#blocks, 2, block_size, 1, head_size]`` with the real Qwen3.5-9B
    per-page byte volume (see ``_MAMBA_SYNTHETIC_HEAD_SIZE``) at the
    engine's real ``block_size`` -- addressing metadata only; the
    conv/ssm/pad bytes underneath are opaque to gather/scatter (see
    ``lmcache/integration/vllm/kv_cache_group_edits.py``).
    """
    shape = (1, 2, _BLOCK_SIZE, 1, _MAMBA_SYNTHETIC_HEAD_SIZE)
    return {
        f"mamba_layer_{i}": torch.randn(
            shape, dtype=torch.float16, device=torch_device_type
        )
        for i in range(_NUM_MAMBA_LAYERS)
    }


def _qwen35_9b_groups() -> list[EngineGroupInfo]:
    """Group 0: full attention layers. Groups 1-3: Mamba/linear layers.

    Mamba is split into ``_NUM_MAMBA_GROUPS`` separate groups of
    ``_MAMBA_LAYERS_PER_GROUP`` layers each, matching a real vLLM + LMCache
    production log where vLLM's own ``kv_cache_config.kv_cache_groups``
    partitioned the 24 Mamba layers into 3 groups of 8
    (``create_engine_group_infos_from_vllm`` in
    ``lmcache/integration/vllm/kv_cache_groups.py`` mirrors vLLM's own
    partition rather than inventing one) -- all 3 groups share the same
    real conv/ssm layout here.

    ``EngineGroupInfo.layer_indices`` are positions into the *worker's full,
    registration-ordered* ``kv_caches`` dict (see
    ``_kv_caches_for_group``/``_select_kv_caches_by_index``), not indices
    relative to each group's own layer count -- so each Mamba group's range
    must start where the previous group's layers end, matching
    ``_qwen35_9b_kv_caches``'s layer order.

    Each Mamba group's ``tokens_per_block`` is set to the *whole LMCache
    chunk size* (``_CHUNK_SIZE_TOKENS``), not the engine's real per-token
    block size -- that is what makes ``_blocks_per_chunk_for_group``
    resolve to ``1`` Mamba block per chunk (see
    :func:`_mamba_kv_caches`'s docstring for why that is the real semantics
    for a recurrent-state group).
    """
    groups = [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=tuple(range(_NUM_ATTENTION_LAYERS)),
            tokens_per_block=_BLOCK_SIZE,
            cache_category="attention",
        ),
    ]
    for mamba_group_idx in range(_NUM_MAMBA_GROUPS):
        start = _NUM_ATTENTION_LAYERS + mamba_group_idx * _MAMBA_LAYERS_PER_GROUP
        end = start + _MAMBA_LAYERS_PER_GROUP
        groups.append(
            EngineGroupInfo(
                engine_group_id=mamba_group_idx + 1,
                layer_indices=tuple(range(start, end)),
                tokens_per_block=_CHUNK_SIZE_TOKENS,
                recurrent_state=True,
                cache_category="mamba",
                mamba_real_layout=_MAMBA_REAL_LAYOUT,
            )
        )
    return groups


def _qwen35_9b_kv_caches() -> dict[str, torch.Tensor]:
    """Attention layers first, then Mamba layers -- matching the layer index
    ranges assigned in :func:`_qwen35_9b_groups`."""
    kv_caches = _attention_kv_caches()
    kv_caches.update(_mamba_kv_caches())
    return kv_caches


def _register_shm_context(
    kv_caches: dict[str, torch.Tensor],
    shm_name: str,
    pool_size: int,
) -> tuple[EngineDrivenTransferContext, EngineDrivenContextShm]:
    """Register a real ``EngineDrivenContextShm`` with a mocked RPC client.

    Only the RPC layer is mocked (no live server process); the SHM segment
    itself, its pinning, and every tensor-view read/write are real, matching
    ``test_engine_driven_context_shm_store_retrieve_flow_with_mocked_mq``'s
    pattern in ``tests/v1/multiprocess/test_engine_driven_transfer.py``.

    Returns the transfer context plus the underlying ``EngineDrivenContextShm``
    so the caller can carve out extra real SHM slots for timing the
    write-into-SHM stage independently of ``submit_store``.
    """
    created_shm_contexts: list[EngineDrivenContextShm] = []

    def _create_shm_context(
        metadata, *_args: object, **_kwargs: object
    ) -> EngineDrivenContextShm:
        shm_context = EngineDrivenContextShm(
            metadata=metadata,
            req_client=MagicMock(),
            mq_timeout=5.0,
            shm_name=shm_name,
            pool_size=pool_size,
            scratch_offset=_SHM_RAW_CHUNK_OFFSET,
            scratch_size=_SHM_POOL_SIZE - _SHM_RAW_CHUNK_OFFSET,
        )
        created_shm_contexts.append(shm_context)
        return shm_context

    original_factory = worker_transfer.create_engine_driven_context
    worker_transfer.create_engine_driven_context = _create_shm_context
    try:
        future = MagicMock()
        future.result.return_value = RegisterEngineDrivenContextResponse(
            shm_name=shm_name,
            pool_size=pool_size,
            scratch_offset=_SHM_RAW_CHUNK_OFFSET,
            scratch_size=_SHM_POOL_SIZE - _SHM_RAW_CHUNK_OFFSET,
        )
        req_client = MagicMock()
        req_client.register_kv_cache_engine_driven_context.return_value = future

        ctx = EngineDrivenTransferContext()
        ctx.register(
            instance_id=1,
            kv_caches=kv_caches,
            model_name="qwen3.5-9b",
            world_size=1,
            blocks_in_chunk=_BLOCKS_IN_CHUNK,
            req_client=req_client,
            mq_timeout=5.0,
            engine_group_infos=_qwen35_9b_groups(),
        )
        return ctx, created_shm_contexts[0]
    finally:
        worker_transfer.create_engine_driven_context = original_factory


def _shm_slot_tensor(
    shm_ctx: EngineDrivenContextShm, offset: int, num_bytes: int
) -> torch.Tensor:
    """Build a real SHM-backed ``uint8`` tensor view at ``offset``.

    This is exactly what ``EngineDrivenContextShm.prepare_store`` hands back
    to ``submit_store`` in production -- a ``torch.frombuffer`` view straight
    over the shared-memory segment (see ``ShmSlotDescriptor``) -- built here
    directly instead of round-tripping through a mocked server response, the
    same pattern ``test_engine_driven_transfer.py``'s
    ``test_make_tensor_view_reads_shm_contents`` uses.
    """
    return shm_ctx._make_tensor_view(
        offset=offset, length=num_bytes, shape=[num_bytes], dtype_str="uint8"
    )


def _group_by_category(
    ctx: EngineDrivenTransferContext,
    kv_caches: dict[str, torch.Tensor],
) -> list[tuple[str, str, GroupTransferPlan, dict[str, torch.Tensor], list[int]]]:
    """Pair each group's plan/KV-cache subset with its own block-id list.

    Returns an ordered ``(label, cache_category, plan, group_kv_caches,
    group_block_ids)`` list rather than a ``dict`` keyed by category, since
    3 distinct Mamba groups share the category ``"mamba"`` and would
    otherwise collide on one dict key. ``label`` disambiguates them for
    printing (e.g. ``"mamba[1]"``); ``cache_category`` stays the plain
    dispatch key consumed by ``_run_store_stages``/``_run_retrieve_stages``.

    ``block_ids`` is indexed by LMCache group id (see
    ``iter_transfer_groups``), in the same order ``_qwen35_9b_groups``
    registered them: attention first, then each Mamba group. Every group
    here covers its chunk with exactly ``1`` physical block
    (``_BLOCKS_IN_CHUNK == 1`` for attention; Mamba's recurrent-state page
    always covers 1 block regardless of ``_BLOCKS_IN_CHUNK``).
    """
    attention_block_ids = list(range(_BLOCKS_IN_CHUNK))
    mamba_block_ids = [0]
    block_ids = [attention_block_ids] + [mamba_block_ids] * _NUM_MAMBA_GROUPS
    result: list[
        tuple[str, str, GroupTransferPlan, dict[str, torch.Tensor], list[int]]
    ] = []
    for plan, group_kv_caches, group_block_ids in ctx.iter_transfer_groups(
        kv_caches, block_ids, _BLOCKS_IN_CHUNK
    ):
        if plan.group_info is None:
            continue
        cache_category = plan.group_info.cache_category
        label = (
            cache_category
            if cache_category != "mamba"
            else f"mamba[{plan.group_info.engine_group_id}]"
        )
        result.append((label, cache_category, plan, group_kv_caches, group_block_ids))
    return result


def _time_calls(fn, iterations: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch_dev.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return samples_ms


def _print_stats(label: str, samples_ms: list[float], num_bytes: int = 0) -> None:
    mean_ms = statistics.mean(samples_ms)
    stdev_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    bandwidth_str = ""
    if num_bytes > 0:
        mean_s = mean_ms / 1000.0
        gib_per_s = (num_bytes / (1024**3)) / mean_s if mean_s > 0 else 0.0
        size_mib = num_bytes / (1024**2)
        bandwidth_str = f"  size={size_mib:9.3f}MiB  bw={gib_per_s:7.2f}GiB/s"
    print(
        f"  {label:<28s} "
        f"mean={mean_ms:8.3f}ms  min={min(samples_ms):8.3f}ms  "
        f"max={max(samples_ms):8.3f}ms  stdev={stdev_ms:7.3f}ms  "
        f"n={len(samples_ms)}{bandwidth_str}"
    )


def _run_store_stages(
    label: str,
    cache_category: str,
    plan: GroupTransferPlan,
    group_kv_caches: dict[str, torch.Tensor],
    group_block_ids: list[int],
    shm_ctx: EngineDrivenContextShm,
    mamba_options: MambaCodecOptions,
    codec: _KVWeaveCodec,
    attention_codec: _KVWeaveCodec,
    *,
    quantize_mamba: bool,
    quantize_attention: bool,
) -> dict[str, object]:
    """Run this group's store-side Stage 1-4, printing each stage's timing.

    Returns the intermediate results the matching retrieve stages need:
    always ``"gathered_chunk"`` (the raw gather output); for a Mamba group
    with ``quantize_mamba`` True and quantization available, also
    ``"quantized_payload"`` and ``"quant_shm_slot"``; for an attention
    group with ``quantize_attention`` True and quantization available, the
    same two keys via the attention codec's ``encode_chunk``.
    """
    is_mamba = cache_category == "mamba"
    is_attention = cache_category == "attention"
    run_mamba_quant = is_mamba and quantize_mamba and _KVWEAVE_QUANT_AVAILABLE
    run_attention_quant = (
        is_attention and quantize_attention and _KVWEAVE_QUANT_AVAILABLE
    )

    scratch_allocation = None
    scratch_chunks = None
    if run_mamba_quant or run_attention_quant:
        raw_dtype = plan.raw_layout_desc.dtypes[0]
        scratch = shm_ctx.allocate_scratch_tensors(
            _raw_gather_shape(plan), raw_dtype, 1, wait=False
        )
        if scratch is not None:
            scratch_chunks, scratch_allocation = scratch

    gather_target = (
        scratch_chunks
        if scratch_chunks is not None
        else None
    )
    gather_description = (
        "reused SHM scratch buffer"
        if gather_target is not None
        else "a freshly allocated CPU buffer"
    )
    print(
        "Stage 1 - gather (device -> CPU, D2H copy) into "
        f"{gather_description}:"
    )
    gathered_chunk: list[torch.Tensor] = []

    def _gather() -> None:
        gathered_chunk[:] = gather_paged_kv_to_cpu(
            group_kv_caches,
            group_block_ids,
            plan.blocks_per_chunk,
            engine_kv_format=plan.engine_kv_format,
            out=gather_target,
        )

    samples_ms = _time_calls(_gather, _NUM_ITERATIONS, _NUM_WARMUP)
    raw_chunk = gathered_chunk[0]
    _print_stats(
        label, samples_ms, raw_chunk.numel() * raw_chunk.element_size()
    )

    result: dict[str, object] = {
        "gathered_chunk": raw_chunk,
        "scratch_allocation": scratch_allocation,
    }

    if is_mamba:
        if run_mamba_quant:
            print(
                "\nStage 2 - Mamba padding removal / conv+ssm recovery "
                "(split_mamba_chunk, CPU):"
            )
            split_result: list[MambaChunkSplit] = []
            conv_layout, ssm_layout = _MAMBA_REAL_LAYOUT
            layers = raw_chunk.shape[1] if raw_chunk.dim() == 4 else raw_chunk.shape[0]
            blocks = raw_chunk.shape[-2] // _BLOCK_SIZE
            split_scratch = shm_ctx.allocate_scratch_tensor_groups(
                [
                    (
                        torch.Size((layers, blocks, *conv_layout.shape)),
                        KVWeaveCodecConfig.mamba_dtype(conv_layout.dtype_str),
                    ),
                    (
                        torch.Size((layers, blocks, *ssm_layout.shape)),
                        KVWeaveCodecConfig.mamba_dtype(ssm_layout.dtype_str),
                    ),
                ],
                1,
                wait=False,
            )
            split_scratch_allocation = None
            if split_scratch is not None:
                split_views, split_scratch_allocation = split_scratch
                split_destination = MambaChunkSplit(*split_views[0])
                split_description = "pool-level SHM split scratch"
            else:
                split_destination = None
                split_description = "temporary CPU split buffers"

            def _split() -> None:
                split_result[:] = [
                    _KVWeaveCodec.split_mamba_chunk(
                        raw_chunk,
                        _MAMBA_REAL_LAYOUT,
                        _BLOCK_SIZE,
                        out=split_destination,
                    )
                ]

            print(f"  Stage 2 destination: {split_description}")
            samples_ms = _time_calls(_split, _NUM_ITERATIONS, _NUM_WARMUP)
            split_preview = split_result[0]
            split_bytes = (
                split_preview.conv.numel() * split_preview.conv.element_size()
                + split_preview.ssm.numel() * split_preview.ssm.element_size()
            )
            _print_stats(
                label,
                samples_ms,
                split_bytes,
            )
            if scratch_allocation is not None:
                shm_ctx.free_scratch(scratch_allocation)
                result["scratch_allocation"] = None
                scratch_allocation = None

            payload = codec.encode_chunk(
                "mamba",
                _MAMBA_REAL_LAYOUT,
                _BLOCK_SIZE,
                mamba_options,
                raw_chunk,
            )

            if len(payload) > _SHM_QUANTIZED_REGION_BYTES:
                raise RuntimeError(
                    f"quantized Mamba payload ({len(payload)} bytes) "
                    f"exceeds the reserved SHM region "
                    f"({_SHM_QUANTIZED_REGION_BYTES} bytes)"
                )
            quant_shm_slot = _shm_slot_tensor(
                shm_ctx, offset=0, num_bytes=len(payload)
            )
            print(
                "\nStage 3 - quantization directly into SHM "
                "(no uint8 staging tensor):"
            )
            samples_ms = _time_calls(
                lambda: codec.encode_chunk_into(
                    "mamba", _MAMBA_REAL_LAYOUT, _BLOCK_SIZE,
                    mamba_options, raw_chunk, quant_shm_slot,
                    mamba_split=split_preview,
                ),
                _NUM_ITERATIONS,
                _NUM_WARMUP,
            )
            _print_stats(label, samples_ms, len(payload))
            print("\nStage 4 - folded into Stage 3 (direct SHM slot write).")
            if split_scratch_allocation is not None:
                shm_ctx.free_scratch(split_scratch_allocation)

            result["quantized_payload"] = payload
            result["quant_shm_slot"] = quant_shm_slot
            return result

        skip_reason = (
            "quantization disabled for this scenario"
            if not quantize_mamba
            else "kvweave_quant unavailable"
        )
        print(
            "\nStage 2/3 - Mamba padding removal + quantization: "
            f"skipped ({skip_reason})."
        )

    if is_attention:
        if run_attention_quant:
            payload = attention_codec.encode_chunk(
                "attention",
                None,
                _BLOCK_SIZE,
                None,
                raw_chunk,
                attention_plane_layout=AttentionPlaneLayout.FUSED_KV,
            )

            if len(payload) > _SHM_QUANTIZED_REGION_BYTES:
                raise RuntimeError(
                    f"quantized attention payload ({len(payload)} bytes) "
                    f"exceeds the reserved SHM region "
                    f"({_SHM_QUANTIZED_REGION_BYTES} bytes)"
                )
            quant_shm_slot = _shm_slot_tensor(
                shm_ctx, offset=0, num_bytes=len(payload)
            )
            print(
                "\nStage 2 - quantization directly into SHM "
                "(no uint8 staging tensor):"
            )
            samples_ms = _time_calls(
                lambda: attention_codec.encode_chunk_into(
                    "attention", None, _BLOCK_SIZE, None, raw_chunk,
                    quant_shm_slot,
                    attention_plane_layout=AttentionPlaneLayout.FUSED_KV,
                ),
                _NUM_ITERATIONS,
                _NUM_WARMUP,
            )
            _print_stats(label, samples_ms, len(payload))
            print("\nStage 3 - folded into Stage 2 (direct SHM slot write).")
            if scratch_allocation is not None:
                shm_ctx.free_scratch(scratch_allocation)
                result["scratch_allocation"] = None

            result["quantized_payload"] = payload
            result["quant_shm_slot"] = quant_shm_slot
            return result

        skip_reason = (
            "quantization disabled for this scenario"
            if not quantize_attention
            else "kvweave_quant unavailable"
        )
        print(f"\nStage 2/3 - attention quantization: skipped ({skip_reason}).")

    print(
        "\nStage 4 - write into SHM (gather straight into a real SHM slot "
        "-- collapses stages 1+4 into one D2H copy):"
    )
    chunk_bytes = raw_chunk.numel() * raw_chunk.element_size()
    if _SHM_RAW_CHUNK_OFFSET + chunk_bytes > _SHM_POOL_SIZE:
        raise RuntimeError(
            f"{cache_category} raw chunk ({chunk_bytes} bytes) does not "
            f"fit in the SHM pool ({_SHM_POOL_SIZE} bytes) at offset "
            f"{_SHM_RAW_CHUNK_OFFSET}"
        )
    raw_shm_slot = (
        _shm_slot_tensor(shm_ctx, offset=_SHM_RAW_CHUNK_OFFSET, num_bytes=chunk_bytes)
        .view(raw_chunk.dtype)
        .view(raw_chunk.shape)
    )

    def _gather_into_shm() -> None:
        gather_paged_kv_to_cpu(
            group_kv_caches,
            group_block_ids,
            plan.blocks_per_chunk,
            engine_kv_format=plan.engine_kv_format,
            out=[raw_shm_slot],
        )

    samples_ms = _time_calls(_gather_into_shm, _NUM_ITERATIONS, _NUM_WARMUP)
    _print_stats(label, samples_ms, chunk_bytes)

    result["raw_shm_slot"] = raw_shm_slot
    return result


def _run_retrieve_stages(
    label: str,
    cache_category: str,
    plan: GroupTransferPlan,
    group_kv_caches: dict[str, torch.Tensor],
    group_block_ids: list[int],
    store_result: dict[str, object],
    attention_codec: _KVWeaveCodec,
    shm_ctx: EngineDrivenContextShm,
    *,
    quantize_mamba: bool,
    quantize_attention: bool,
) -> None:
    """Run this group's retrieve-side Stage R1-R3, printing each stage's timing."""
    is_mamba = cache_category == "mamba"
    is_attention = cache_category == "attention"
    run_mamba_quant = (
        is_mamba and quantize_mamba and "quantized_payload" in store_result
    )
    run_attention_quant = (
        is_attention and quantize_attention and "quantized_payload" in store_result
    )
    raw_chunk = store_result["gathered_chunk"]
    decode_scratch_allocation = None
    decode_target: torch.Tensor | None = None
    if run_mamba_quant or run_attention_quant:
        scratch = shm_ctx.allocate_scratch_tensors(
            _raw_gather_shape(plan), raw_chunk.dtype, 1, wait=False
        )
        if scratch is not None:
            scratch_chunks, decode_scratch_allocation = scratch
            decode_target = scratch_chunks[0]
        else:
            requested_mib = (
                raw_chunk.numel() * raw_chunk.element_size() / (1024**2)
            )
            print(
                "  retrieve scratch unavailable; falling back to temporary "
                f"CPU buffer (requested={requested_mib:.3f}MiB)"
            )

    scatter_chunk = raw_chunk
    if run_mamba_quant:
        # No separate "read from SHM" step: production's ``prepare_retrieve``
        # hands ``decode_chunk`` a zero-copy ``torch.frombuffer`` view
        # straight over the SHM slot (see
        # ``EngineDrivenContextShm._build_slot_tensors``/``_make_tensor_view``
        # and ``EngineDrivenTransferContext._decode_group_chunks`` in
        # ``worker_transfer.py``), so dequantization reads directly off
        # ``quant_shm_slot`` here too -- an extra ``.clone()`` before this
        # would measure a copy production never makes.
        print(
            "\nStage R1 - dequantization + scratch-backed opaque-page "
            "rebuild (4-bit conv+ssm, CPU, reads directly off the SHM slot):"
        )
        quant_shm_slot = store_result["quant_shm_slot"]
        decoded_chunk: list[torch.Tensor] = []

        def _dequantize() -> None:
            decoded_chunk[:] = [
                attention_codec.decode_chunk(
                    "mamba",
                    _MAMBA_REAL_LAYOUT,
                    _BLOCK_SIZE,
                    raw_chunk.shape,
                    raw_chunk.dtype,
                    quant_shm_slot,
                    out=decode_target,
                )
            ]

        samples_ms = _time_calls(_dequantize, _NUM_ITERATIONS, _NUM_WARMUP)
        _print_stats(
            label,
            samples_ms,
            raw_chunk.numel() * raw_chunk.element_size(),
        )
        print("\nStage R2 - folded into Stage R1 (scratch-backed decode_chunk).")
        scatter_chunk = decoded_chunk[0]
    elif is_mamba:
        skip_reason = (
            "quantization disabled for this scenario"
            if not quantize_mamba
            else "kvweave_quant unavailable"
        )
        print(
            "\nStage R1/R2 - dequantize + re-pad: "
            f"skipped ({skip_reason})."
        )
    elif run_attention_quant:
        # Same zero-copy contract as the Mamba path above: decode reads
        # directly off ``quant_shm_slot``, no intermediate clone.
        print(
            "\nStage R1 - dequantization (4-bit fused K/V, CPU, "
            "native kvweave_quant, reads directly off the SHM slot):"
        )
        quant_shm_slot = store_result["quant_shm_slot"]
        decoded_chunk: list[torch.Tensor] = []

        def _decode_attention() -> None:
            decoded_chunk[:] = [
                attention_codec.decode_chunk(
                    "attention",
                    None,
                    _BLOCK_SIZE,
                    raw_chunk.shape,
                    raw_chunk.dtype,
                    quant_shm_slot,
                    attention_plane_layout=AttentionPlaneLayout.FUSED_KV,
                    out=decode_target,
                )
            ]

        samples_ms = _time_calls(_decode_attention, _NUM_ITERATIONS, _NUM_WARMUP)
        _print_stats(
            label, samples_ms, raw_chunk.numel() * raw_chunk.element_size()
        )
        scatter_chunk = decoded_chunk[0]
    elif is_attention:
        skip_reason = (
            "quantization disabled for this scenario"
            if not quantize_attention
            else "kvweave_quant unavailable"
        )
        print(f"\nStage R1 - dequantize: skipped ({skip_reason}).")

    print(
        "\nStage R3 - scatter (CPU -> device, H2D copy):"
    )
    chunks = [scatter_chunk]
    samples_ms = _time_calls(
        lambda: scatter_cpu_to_paged_kv(
            group_kv_caches,
            group_block_ids,
            chunks,
            plan.blocks_per_chunk,
            engine_kv_format=plan.engine_kv_format,
        ),
        _NUM_ITERATIONS,
        _NUM_WARMUP,
    )
    _print_stats(
        label,
        samples_ms,
        scatter_chunk.numel() * scatter_chunk.element_size(),
    )
    if decode_scratch_allocation is not None:
        shm_ctx.free_scratch(decode_scratch_allocation)


def main() -> int:
    if not (torch_device_type == "xpu" and torch_dev.is_available()):
        print("Skipping: requires an available XPU runtime.")
        return 0

    if not _KVWEAVE_QUANT_AVAILABLE:
        print(
            "Note: native 'kvweave_quant' extension not importable -- "
            "Scenario A's quantize/dequantize/re-pad stages will be "
            "skipped. Run inside an env with the 'kvweave' package "
            "installed (e.g. the 'dev_lmcache' conda env) to measure "
            "them.\n"
        )

    kv_caches = _qwen35_9b_kv_caches()
    shm_name = f"lmcache_bench_shm_{os.getpid()}"
    addr = shm_create_readwrite(shm_name, _SHM_POOL_SIZE)
    try:
        ctx, shm_ctx = _register_shm_context(kv_caches, shm_name, _SHM_POOL_SIZE)
        groups = _group_by_category(ctx, kv_caches)
        mamba_options = _mamba_codec_options()
        codec = _KVWeaveCodec()
        attention_codec = _attention_codec()

        print(
            f"Qwen3.5-9B engine-driven SHM timing "
            f"({_NUM_ATTENTION_LAYERS} attention layers x "
            f"{_BLOCKS_IN_CHUNK} block x {_BLOCK_SIZE} tokens/block "
            f"(fused K/V), {_NUM_MAMBA_LAYERS} mamba layers split into "
            f"{_NUM_MAMBA_GROUPS} groups x {_MAMBA_LAYERS_PER_GROUP} layers, "
            f"each 1 recurrent-state block covering the same "
            f"{_CHUNK_SIZE_TOKENS}-token chunk, "
            f"{_NUM_ITERATIONS} iterations after {_NUM_WARMUP} warmup)"
        )

        scenarios = [
            (True, "Scenario A -- Mamba + attention quantization ENABLED"),
            (False, "Scenario B -- Mamba + attention quantization DISABLED"),
        ]
        for quantize_all, scenario_label in scenarios:
            print(f"\n{'=' * 70}\n{scenario_label}\n{'=' * 70}")
            for label, cache_category, plan, group_kv_caches, group_block_ids in groups:
                print(f"\n--- group: {label} ---")
                store_result = _run_store_stages(
                    label,
                    cache_category,
                    plan,
                    group_kv_caches,
                    group_block_ids,
                    shm_ctx,
                    mamba_options,
                    codec,
                    attention_codec,
                    quantize_mamba=quantize_all,
                    quantize_attention=quantize_all,
                )
                _run_retrieve_stages(
                    label,
                    cache_category,
                    plan,
                    group_kv_caches,
                    group_block_ids,
                    store_result,
                    attention_codec,
                    shm_ctx,
                    quantize_mamba=quantize_all,
                    quantize_attention=quantize_all,
                )
                scratch_allocation = store_result.get("scratch_allocation")
                if scratch_allocation is not None:
                    shm_ctx.free_scratch(scratch_allocation)

        return 0
    finally:
        shm_munmap(addr, _SHM_POOL_SIZE)
        shm_unlink(shm_name)


if __name__ == "__main__":
    sys.exit(main())
