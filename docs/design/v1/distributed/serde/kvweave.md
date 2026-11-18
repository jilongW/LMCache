# `lmcache.v1.distributed.serde.kvweave` — KVWeave Serde Backend

## Scope

KVWeave is a 4/8-bit KV quantization codec with native CPU and XPU kernels.
This serde applies it on the **L1 → L2 path**: L1 keeps full-precision KV, the
store controller hands a chunk to a serde-wrapped L2 adapter, the serde
quantizes it while serializing, and the load path dequantizes it back into the
L1 destination chunk. Only L2 is lossy, so nothing has to guard against two
lossy layers.

It plugs into the generic serde framework
([`README.md`](README.md)) and the L2 wrapper
([`../l2_adapters/serde_wrapper.md`](../l2_adapters/serde_wrapper.md)); it does
not touch `StorageManager` or the L2 adapter APIs.

## Configuration

```json
{"type": "fs", "base_path": "/cache",
 "serde": {"type": "kvweave", "qbit": 4, "scaling_method": "per_channel",
           "device": "xpu", "num_kv_heads": 4, "head_dim": 128}}
```

| Key | Default | Meaning |
|---|---|---|
| `qbit` | `4` | `4` or `8` bits per value |
| `scaling_method` | `per_channel` | `per_channel`, `per_token` or `per_tensor` |
| `rh`, `asym`, `precond` | `true`, `true`, `true` | randomized Hadamard, asymmetric range, preconditioner |
| `device` | `cpu` | `cpu` or `xpu`; XPU must be available |
| `num_threads` | `4` | OpenMP threads of the CPU kernels |
| `kv_fp8_dtype` | none | format of an fp8 (uint8) KV cache; fused layout only |
| `num_kv_heads`, `head_dim` | `1`, `0` | geometry for groups the engine did not describe |
| `conv_*` / `ssm_*` options | see `MambaCodecOptions.from_kwargs` | per-sub-state scaling, RH, bits |
| (conv Q/K/V split) | engine-reported | not configurable; a Mamba group whose engine reports no `channel_segments` (non-GDN kinds) is stored raw |
| `linear_max_size_ratio` | `1.2` | quantize a Mamba chunk only if it is not larger than this × raw |
| `groups`, `default_group` | none | static per-group facts (see below) |
| `max_workers` | `1` | serde thread-pool size |

## What a chunk is: per-group routing

What the codec must do depends on what the chunk holds, so every key is routed
by its `object_group_id`:

| Group holds | Chunk shape | Handling |
|---|---|---|
| attention, split K/V | `[2, L, T, H]` | quantize both planes |
| attention, fused-packed K/V | `[L, T, H]` | fused codec |
| Mamba / linear state | opaque page view | split into `conv` + `ssm`, each in its own dtype |
| MLA, unclassified, or not smaller | any | bytes copied through unchanged |

An object group's chunk is one memory object holding **one tensor per kernel
group** (a *part*). A multi-part chunk is stored as `[int32 length | payload]`
per part; a single-part chunk is the bare payload.

### Where the facts come from

Quantizing needs the group's category, head geometry, K/V packing and (for
Mamba) the real conv/ssm byte layout. The engine registration publishes them as
`SerdePartInfo` into the engine-neutral registry
`lmcache.v1.distributed.serde.group_info`; the serde interprets them, so the
registration code stays serde-agnostic.

```
vLLM worker                      server                         serde
EngineGroupInfo ──register──►  KernelGroupInfo.serde_part_info() ──► SerdeGroupInfoRegistry
 cache_category                   (heads/head size from the page          │
 mamba_real_layout                shape, K/V packing from the KV          ▼
                                  format spec)                     KVWeaveGroupSpec
```

- *lmcache-driven* registration (the server holds the engine's tensors) builds
  the facts from its `KVLayerGroupsManager`.
- *engine-driven* registration (SHM/inline) carries them in
  `GroupLayout.serde_part`; workers that predate the field publish nothing, so
  their groups are stored untransformed.
- `groups` / `default_group` in the serde config give the same facts statically
  (for tests, or setups without an engine registration).

## Size contract

The temp buffer is allocated **before** serializing, so the size must be known
from the layout alone. KVWeave's payload size depends only on shape, `qbit` and
`scaling_method` — not on the data, `rh` or `asym` — so `estimate_serialized_size`
is an exact formula, not a margin:

```
split  [2,L,T,H]: 28 + 2*(4 + 12*S + 4*L) + 2*ceil(L*T*H*qbit/8)
fused  [L,T,H]  : 25 + 12*S + 4*native_layers + ceil(L*T*H*qbit/8)
Mamba           : framing + per-sub-state payloads (see estimate_mamba_serialized_size)
```

`S` is the scale count of the scaling method. The tests assert the estimate
equals the bytes actually written, on CPU and XPU. Because the size can differ
between groups that share a shape, the wrapper sizes each temp buffer through
`estimate_serialized_size_for_key` (see the wrapper doc).

## Decision per chunk

`kvweave_l2._PlanResolver` decides once per `(group spec, part shape, dtype)` and
caches the result. A chunk is copied through (never quantized) when its group is
MLA/unclassified, its rank or hidden size does not match the declared layout, the
dtype is unsupported (a uint8 fp8 cache needs the fused layout and
`kv_fp8_dtype`), or quantizing would not make it smaller. The native RH kernel
needs a power-of-two transform length; a shape that cannot satisfy it is encoded
without RH (the header records the choice, and the size is unchanged).

Payloads carry a magic number, so decoding a buffer that is not a KVWeave payload
fails the deserialize task instead of returning garbage. The group
configuration must be the same on store and load: a pass-through chunk has no
header to check.

## Module layout

```
serde/kvweave/
  kvweave_config.py   # AttentionPlaneLayout, MambaCodecOptions, KVWeaveCodecConfig
  kvweave_serde.py    # KVWeaveCodec: the native-kernel codec (attention + Mamba)
  kvweave_l2.py       # per-group routing, Serializer/Deserializer, factory registration
serde/group_info.py   # SerdePartInfo registry shared with the engine registration
```
