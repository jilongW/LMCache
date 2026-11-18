# SPDX-License-Identifier: Apache-2.0
"""KVWeave L2 serde: quantize L1 chunks while they are written to L2.

The L1 pool keeps full-precision KV. When the store controller moves a chunk
to an L2 adapter wrapped by :class:`SerdeL2AdapterWrapper`, this serde encodes
it with the KVWeave codec (4/8-bit, on CPU or XPU); the load path decodes it
back into the L1 destination chunk. Because only L2 is lossy, the double-lossy
guard of L1 quantization does not apply.

What a chunk *is* depends on the object group it belongs to, so every key is
routed through a per-group :class:`KVWeaveGroupSpec`:

* ``attention`` + ``split_kv``: ``[2, L, T, H]``, K and V as separate planes.
* ``attention`` + ``fused_kv``: ``[L, T, H]``, K and V packed on the last axis.
* ``mamba``: an opaque recurrent-state page view holding a ``conv`` and an
  ``ssm`` sub-state, each quantized in its own dtype.
* anything else (MLA, unclassified, or not worth quantizing): the bytes are
  copied through unchanged.

Group facts come from the engine's registration (published to the neutral
``serde.group_info`` registry) and/or from the serde's JSON kwargs
(``"groups"`` / ``"default_group"``).

Example::

    {"type": "fs", "base_path": "/cache",
     "serde": {"type": "kvweave", "qbit": 4, "scaling_method": "per_channel",
               "device": "xpu", "num_kv_heads": 4, "head_dim": 128}}
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
import threading

# Third Party
import msgspec
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.common.group_view import MambaSubStateWireLayout
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import (
    Deserializer,
    SerdeProcessor,
    Serializer,
)
from lmcache.v1.distributed.serde.factory import register_serde_factory
from lmcache.v1.distributed.serde.group_info import (
    SerdePartInfo,
    get_serde_group_info_registry,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    AttentionPlaneLayout,
    ConvQKVSplit,
    KVWeaveCodecConfig,
    MambaCodecOptions,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import KVWeaveCodec
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

_SUPPORTED_DTYPES = frozenset(KVWeaveCodecConfig.DTYPE_TO_CODE) | {torch.uint8}

# Length prefix of each part of a multi-part chunk (int32).
_FRAME_BYTES = 4

# Mamba is quantized only when the quantized chunk is not larger than this
# multiple of the raw one (mirrors the L1 path's ``linear_max_size_ratio``).
_DEFAULT_LINEAR_MAX_SIZE_RATIO = 1.20


# ============================================================================
# Per-group metadata
# ============================================================================


@dataclass(frozen=True)
class KVWeaveGroupSpec:
    """What the codec needs to know about one object group.

    Attributes:
        cache_category: ``"attention"``, ``"mamba"`` or ``"unknown"``
            (``"unknown"`` groups are copied through unquantized).
        attention_plane_layout: K/V plane layout of an attention group.
        tokens_per_block: Tokens per paged block (Mamba block count).
        mamba_layout: ``(conv, ssm)`` real byte layout of a Mamba group.
        num_kv_heads: KV heads of an attention group; ``0`` = use the serde
            default.
        head_dim: Per-head width of an attention group; ``0`` = use the serde
            default.
    """

    cache_category: str = "unknown"
    attention_plane_layout: AttentionPlaneLayout = AttentionPlaneLayout.UNKNOWN
    tokens_per_block: int = 0
    mamba_layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout] | None = None
    num_kv_heads: int = 0
    head_dim: int = 0

    @classmethod
    def from_part_info(cls, info: SerdePartInfo) -> "KVWeaveGroupSpec":
        """Interpret engine-published facts as a codec spec."""
        if info.cache_category == "mamba":
            return cls(
                cache_category="mamba",
                tokens_per_block=info.tokens_per_block,
                mamba_layout=info.mamba_real_layout,
            )
        if info.cache_category != "attention":
            return cls()
        if info.is_mla:
            plane = AttentionPlaneLayout.MLA
        elif info.is_fused_packed:
            plane = AttentionPlaneLayout.FUSED_KV
        else:
            plane = AttentionPlaneLayout.SPLIT_KV
        return cls(
            cache_category="attention",
            attention_plane_layout=plane,
            tokens_per_block=info.tokens_per_block,
            num_kv_heads=info.num_heads,
            head_dim=info.head_size,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "KVWeaveGroupSpec":
        """Build a spec from a JSON-config dict.

        Keys: ``cache_category``, ``attention_plane_layout``,
        ``tokens_per_block``, ``num_kv_heads``, ``head_dim`` and
        ``mamba_layout`` (a ``[conv, ssm]`` pair of
        ``{byte_offset, byte_length, dtype_str, shape}`` dicts).
        """
        layout = raw.get("mamba_layout")
        mamba_layout = None
        if layout is not None:
            if len(layout) != 2:
                raise ValueError("mamba_layout must be a [conv, ssm] pair")
            mamba_layout = (
                msgspec.convert(layout[0], MambaSubStateWireLayout),
                msgspec.convert(layout[1], MambaSubStateWireLayout),
            )
        category = str(raw.get("cache_category", "unknown"))
        return cls(
            cache_category=category,
            attention_plane_layout=AttentionPlaneLayout(
                str(raw.get("attention_plane_layout", "unknown"))
            ),
            tokens_per_block=int(raw.get("tokens_per_block", 0)),
            mamba_layout=mamba_layout,
            num_kv_heads=int(raw.get("num_kv_heads", 0)),
            head_dim=int(raw.get("head_dim", 0)),
        )


# ============================================================================
# Per-(group, shape) plan
# ============================================================================


@dataclass(frozen=True)
class _Plan:
    """How one chunk of a given group/shape/dtype is (de)serialized."""

    mode: str  # "raw" | "split" | "fused" | "mamba"
    size: int  # exact upper bound of the serialized bytes
    codec: KVWeaveCodec | None = None
    spec: KVWeaveGroupSpec | None = None
    mamba_options: MambaCodecOptions | None = None


def _raw_nbytes(shape: Sequence[int], dtype: torch.dtype) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n * torch.empty((), dtype=dtype).element_size()


def _transform_length_ok(length: int) -> bool:
    return length > 0 and not length & (length - 1)


def _mamba_rh_length(
    substate: str,
    sub_shape: tuple[int, ...],
    layers: int,
    blocks: int,
    scaling_method: str,
) -> int:
    per_layer = blocks
    for dim in sub_shape:
        per_layer *= int(dim)
    _, _, _, chunks = KVWeaveCodecConfig.mamba_layout(
        substate, (layers, blocks, *sub_shape), scaling_method
    )
    return max(per_layer // chunks, 1)


def _engine_conv_split(spec: KVWeaveGroupSpec) -> ConvQKVSplit | None:
    """The conv q/k/v split the engine reported for this Mamba group, if any."""
    assert spec.mamba_layout is not None
    segments = spec.mamba_layout[0].channel_segments
    if len(segments) != 3:
        return None
    return ConvQKVSplit(key_dim=segments[0], value_dim=segments[2])


def _resolve_mamba_rh(
    spec: KVWeaveGroupSpec,
    shape: tuple[int, ...],
    options: MambaCodecOptions,
    split: ConvQKVSplit,
) -> MambaCodecOptions:
    """Drop ``conv_rh``/``ssm_rh`` where the transform length is not 2^k.

    The native RH kernel rejects non-power-of-two lengths; turning RH off per
    sub-state keeps the serialized size unchanged (it never depends on RH).
    """
    assert spec.mamba_layout is not None
    conv_layout, ssm_layout = spec.mamba_layout
    layers = int(shape[1]) if len(shape) == 4 else int(shape[0])
    blocks = max(int(shape[-2]) // spec.tokens_per_block, 1)
    updates: dict[str, bool] = {}
    if options.conv_rh:
        for sub_dim in (split.key_dim, split.key_dim, split.value_dim):
            length = _mamba_rh_length(
                "conv",
                (*conv_layout.shape[:-1], sub_dim),
                layers,
                blocks,
                options.conv_scaling_method,
            )
            if not _transform_length_ok(length):
                updates["conv_rh"] = False
                break
    if options.ssm_rh:
        length = _mamba_rh_length(
            "ssm", tuple(ssm_layout.shape), layers, blocks, options.ssm_scaling_method
        )
        if not _transform_length_ok(length):
            updates["ssm_rh"] = False
    return replace(
        options,
        conv_rh=updates.get("conv_rh", options.conv_rh),
        ssm_rh=updates.get("ssm_rh", options.ssm_rh),
    )


def _plane_layout(mode: str, spec: KVWeaveGroupSpec) -> AttentionPlaneLayout | None:
    return spec.attention_plane_layout if mode in ("split", "fused") else None


def encode_chunk_into_with_plan(
    plan: "_Plan", raw: torch.Tensor, destination: torch.Tensor
) -> int:
    """Encode ``raw`` per ``plan`` straight into ``destination``.

    Args:
        plan: A quantizing plan (``split``, ``fused`` or ``mamba``).
        raw: The CPU chunk tensor.
        destination: Flat contiguous ``uint8`` buffer of at least ``plan.size``.

    Returns:
        Bytes written into ``destination``.
    """
    assert plan.codec is not None and plan.spec is not None
    return plan.codec.encode_chunk_into(
        "mamba" if plan.mode == "mamba" else "attention",
        plan.spec.mamba_layout if plan.mode == "mamba" else None,
        plan.spec.tokens_per_block,
        plan.mamba_options,
        raw,
        destination,
        attention_plane_layout=_plane_layout(plan.mode, plan.spec),
    )


class _PlanResolver:
    """Resolves and caches the :class:`_Plan` of every (group, shape, dtype)."""

    def __init__(self, kwargs: Mapping[str, Any]) -> None:
        self._codec_kwargs = self._build_codec_kwargs(kwargs)
        self._mamba_options = MambaCodecOptions.from_kwargs(kwargs)
        self._ratio = float(
            kwargs.get("linear_max_size_ratio", _DEFAULT_LINEAR_MAX_SIZE_RATIO)
        )
        self._attention_enabled = bool(kwargs.get("attention_quant_enabled", True))
        self._split_enabled = bool(kwargs.get("split_attention_quant_enabled", True))
        self._fused_enabled = bool(kwargs.get("fused_attention_quant_enabled", True))
        self._linear_enabled = bool(kwargs.get("linear_quant_enabled", True))
        self._registry = get_serde_group_info_registry()
        # Static groups from the JSON config: ``object_group_id -> spec`` or
        # ``-> [spec per part]``.
        self._static: dict[int, tuple[KVWeaveGroupSpec, ...]] = {
            int(group_id): tuple(
                KVWeaveGroupSpec.from_dict(part)
                for part in (raw if isinstance(raw, list) else [raw])
            )
            for group_id, raw in dict(kwargs.get("groups", {})).items()
        }
        # Spec of any group the config does not describe: plain split-K/V
        # attention using the serde-level head geometry.
        default_raw = dict(kwargs.get("default_group", {}))
        default_raw.setdefault("cache_category", "attention")
        default_raw.setdefault("attention_plane_layout", "split_kv")
        default_raw.setdefault("num_kv_heads", self._codec_kwargs["num_kv_heads"])
        default_raw.setdefault("head_dim", self._codec_kwargs["head_dim"])
        self._default_spec = KVWeaveGroupSpec.from_dict(default_raw)
        self._codecs: dict[tuple[int, int, bool], KVWeaveCodec] = {}
        self._plans: dict[tuple, _Plan] = {}
        self._lock = threading.Lock()
        self._warned: set[tuple] = set()

    @staticmethod
    def _build_codec_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
        qbit = int(kwargs.get("qbit", 4))
        if qbit not in (4, 8):
            raise ValueError(f"kvweave serde: qbit={qbit} is not one of [4, 8]")
        scaling = str(kwargs.get("scaling_method", "per_channel"))
        if scaling not in ("per_tensor", "per_channel", "per_token"):
            raise ValueError(f"kvweave serde: unknown scaling_method {scaling!r}")
        fp8 = kwargs.get("kv_fp8_dtype")
        if isinstance(fp8, str):
            fp8 = getattr(torch, fp8, None)
            if fp8 not in (torch.float8_e4m3fn, torch.float8_e5m2):
                raise ValueError(
                    f"kvweave serde: kv_fp8_dtype={kwargs['kv_fp8_dtype']!r} "
                    "is not float8_e4m3fn / float8_e5m2"
                )
        return dict(
            quantize=True,
            qbit=qbit,
            scaling_method=scaling,
            rh=bool(kwargs.get("rh", True)),
            asym=bool(kwargs.get("asym", True)),
            precond=bool(kwargs.get("precond", True)),
            precond_seed=int(kwargs.get("precond_seed", 42)),
            precond_path=kwargs.get("precond_path") or None,
            num_threads=int(kwargs.get("num_threads", 4)),
            block_size=int(kwargs.get("block_size", 64)),
            num_kv_heads=int(kwargs.get("num_kv_heads", 1)),
            head_dim=int(kwargs.get("head_dim", 0)),
            device=str(kwargs.get("device", "cpu")),
            kv_fp8_dtype=fp8,
            fused_direct_quant=bool(kwargs.get("fused_direct_quant", True)),
            fused_direct_dequant=bool(kwargs.get("fused_direct_dequant", True)),
        )

    # ----- codec cache -----

    def _codec(self, num_kv_heads: int, head_dim: int, rh: bool) -> KVWeaveCodec:
        ident = (num_kv_heads, head_dim, rh)
        with self._lock:
            codec = self._codecs.get(ident)
            if codec is None:
                kwargs = dict(self._codec_kwargs)
                kwargs.update(num_kv_heads=num_kv_heads, head_dim=head_dim, rh=rh)
                codec = KVWeaveCodec(kwargs)
                self._codecs[ident] = codec
            return codec

    # ----- resolution -----

    def spec_for(self, key: ObjectKey | None, part: int) -> KVWeaveGroupSpec:
        """The spec of part ``part`` of ``key``'s object group.

        Falls back to the serde-level default when the group is unknown.
        """
        if key is None:
            return self._default_spec
        info = self._registry.lookup(key.model_name, key.object_group_id, part)
        if info is not None:
            return KVWeaveGroupSpec.from_part_info(info)
        static = self._static.get(key.object_group_id)
        if static is not None and part < len(static):
            return static[part]
        return self._default_spec

    def plan(
        self,
        key: ObjectKey | None,
        part: int,
        shape: Sequence[int],
        dtype: torch.dtype,
    ) -> _Plan:
        """Return the (cached) plan for one part of ``shape``/``dtype``."""
        spec = self.spec_for(key, part)
        ident = (spec, tuple(int(d) for d in shape), dtype)
        with self._lock:
            plan = self._plans.get(ident)
        if plan is None:
            plan = self._build_plan(spec, ident[1], dtype)
            with self._lock:
                self._plans[ident] = plan
        return plan

    def _raw(self, shape: tuple[int, ...], dtype: torch.dtype, why: str) -> _Plan:
        ident = (why, shape, dtype)
        if ident not in self._warned:
            self._warned.add(ident)
            logger.info(
                "kvweave serde: copying shape=%s dtype=%s through unquantized (%s)",
                shape,
                dtype,
                why,
            )
        return _Plan(mode="raw", size=_raw_nbytes(shape, dtype))

    def _build_plan(
        self, spec: KVWeaveGroupSpec, shape: tuple[int, ...], dtype: torch.dtype
    ) -> _Plan:
        if spec.cache_category == "mamba" and self._linear_enabled:
            # vLLM exposes recurrent pages as opaque int8 byte views; the
            # Mamba codec reinterprets them through ``mamba_layout``.
            if dtype not in _SUPPORTED_DTYPES and dtype != torch.int8:
                return self._raw(shape, dtype, f"unsupported dtype {dtype}")
            return self._build_mamba_plan(spec, shape, dtype)
        if dtype not in _SUPPORTED_DTYPES:
            return self._raw(shape, dtype, f"unsupported dtype {dtype}")
        if spec.cache_category == "attention" and self._attention_enabled:
            return self._build_attention_plan(spec, shape, dtype)
        return self._raw(shape, dtype, f"category {spec.cache_category}")

    def _build_attention_plan(
        self, spec: KVWeaveGroupSpec, shape: tuple[int, ...], dtype: torch.dtype
    ) -> _Plan:
        layout = spec.attention_plane_layout
        if layout == AttentionPlaneLayout.SPLIT_KV:
            if not self._split_enabled or len(shape) != 4 or shape[0] != 2:
                return self._raw(shape, dtype, "shape is not split-K/V [2,L,T,H]")
            mode = "split"
        elif layout == AttentionPlaneLayout.FUSED_KV:
            if not self._fused_enabled or len(shape) != 3:
                return self._raw(shape, dtype, "shape is not fused-K/V [L,T,H]")
            mode = "fused"
        else:
            return self._raw(shape, dtype, f"attention layout {layout.value}")
        heads = spec.num_kv_heads or self._codec_kwargs["num_kv_heads"]
        head_dim = spec.head_dim or self._codec_kwargs["head_dim"]
        codec = self._codec(heads, head_dim, self._codec_kwargs["rh"])
        if dtype == torch.uint8 and not (
            mode == "fused" and codec.supports_fp8_attention()
        ):
            return self._raw(
                shape, dtype, "uint8 (fp8) KV needs fused layout + kv_fp8_dtype"
            )
        desc = MemoryLayoutDesc(shapes=[torch.Size(shape)], dtypes=[dtype])
        try:
            if mode == "split":
                if not codec.accepts_attention_hidden(shape[-1]):
                    return self._raw(shape, dtype, "hidden size != heads * head_dim")
                size = codec.estimate_serialized_size(desc)
            else:
                size = codec.estimate_fused_serialized_size(desc)
        except ValueError as exc:
            return self._raw(shape, dtype, f"size estimate rejected shape: {exc}")
        if size >= _raw_nbytes(shape, dtype):
            return self._raw(shape, dtype, "quantized chunk is not smaller")
        codec = self._probe_rh(codec, spec, mode, shape, dtype, heads, head_dim)
        return _Plan(mode=mode, size=size, codec=codec, spec=spec)

    def _probe_rh(
        self,
        codec: KVWeaveCodec,
        spec: KVWeaveGroupSpec,
        mode: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        heads: int,
        head_dim: int,
    ) -> KVWeaveCodec:
        """Return a codec whose RH setting this chunk shape can actually run.

        The native RH kernel needs a power-of-two transform length, which
        depends on shape and scaling method. Encode a zero chunk once; when the
        kernel rejects it, fall back to the same codec without RH (the payload
        size does not depend on RH, and the header records the choice).
        """
        if not codec.rh or codec.scaling_method == "per_tensor":
            return codec
        probe = torch.zeros(shape, dtype=dtype)
        try:
            codec.encode_chunk(
                "mamba" if mode == "mamba" else "attention",
                spec.mamba_layout if mode == "mamba" else None,
                spec.tokens_per_block,
                None,
                probe,
                attention_plane_layout=_plane_layout(mode, spec),
            )
            return codec
        except ValueError as exc:
            if "power of 2" not in str(exc):
                raise
        logger.info(
            "kvweave serde: shape %s cannot run RH (transform length is not a "
            "power of 2); using no-RH codec for it",
            shape,
        )
        return self._codec(heads, head_dim, False)

    def _build_mamba_plan(
        self, spec: KVWeaveGroupSpec, shape: tuple[int, ...], dtype: torch.dtype
    ) -> _Plan:
        if spec.mamba_layout is None or spec.tokens_per_block <= 0:
            return self._raw(shape, dtype, "mamba group has no real layout")
        if len(shape) not in (3, 4):
            return self._raw(shape, dtype, "mamba chunk is not [L,T,H] / [2,L,T,H]")
        split = _engine_conv_split(spec)
        if split is None:
            return self._raw(shape, dtype, "engine reported no conv q/k/v split")
        options = replace(self._mamba_options, conv_qkv_split=split)
        try:
            options = _resolve_mamba_rh(spec, shape, options, split)
            size = KVWeaveCodec.estimate_mamba_serialized_size(
                MemoryLayoutDesc(shapes=[torch.Size(shape)], dtypes=[dtype]),
                spec.mamba_layout,
                spec.tokens_per_block,
                conv_qkv_split=split,
                scaling_methods=(
                    options.conv_scaling_method,
                    options.ssm_scaling_method,
                ),
                qbits=(options.conv_qbit, options.ssm_qbit),
                quant_enabled=(options.conv_quant_enabled, options.ssm_quant_enabled),
            )
        except ValueError as exc:
            return self._raw(shape, dtype, f"mamba size estimate rejected: {exc}")
        if size > _raw_nbytes(shape, dtype) * self._ratio:
            return self._raw(shape, dtype, "quantized mamba chunk is not smaller")
        codec = self._codec(1, 0, False)
        return _Plan(
            mode="mamba", size=size, codec=codec, spec=spec, mamba_options=options
        )


# ============================================================================
# Serializer / Deserializer
# ============================================================================


def _flat_u8(obj: MemoryObj) -> torch.Tensor:
    tensor = obj.tensor
    if tensor is None:
        raise ValueError("kvweave serde requires MemoryObjs that hold a tensor")
    return tensor.view(torch.uint8).view(-1)


def _part_tensor(obj: MemoryObj, part: int) -> torch.Tensor:
    tensor = obj.get_tensor(part)
    if tensor is None:
        raise ValueError("kvweave serde requires MemoryObjs that hold a tensor")
    return tensor


class KVWeaveSerializer(Serializer):
    """Quantize an L1 chunk into the L2 byte buffer (see module docstring).

    A chunk with several parts (one tensor per kernel group) is stored as
    ``[u32 length | payload]`` per part; a single-part chunk is the bare
    payload.
    """

    def __init__(self, resolver: _PlanResolver) -> None:
        self._resolver = resolver

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Size of the chunk under the serde's default group spec."""
        return self.estimate_serialized_size_for_key(layout_desc, None)

    def estimate_serialized_size_for_key(
        self, layout_desc: MemoryLayoutDesc, key: ObjectKey | None
    ) -> int:
        """Exact size of the serialized chunk of ``key``'s object group."""
        total = 0
        parts = list(zip(layout_desc.shapes, layout_desc.dtypes, strict=True))
        for part, (shape, dtype) in enumerate(parts):
            total += self._resolver.plan(key, part, shape, dtype).size
        return total + (_FRAME_BYTES * len(parts) if len(parts) > 1 else 0)

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Encode every part of ``src`` into ``dst``; return the bytes written."""
        out = _flat_u8(dst)
        num_parts = len(src.get_shapes())
        offset = 0
        for part in range(num_parts):
            raw = _part_tensor(src, part)
            plan = self._resolver.plan(key, part, raw.shape, raw.dtype)
            if num_parts > 1:
                offset += _FRAME_BYTES
            written = self._serialize_part(plan, raw, out[offset:])
            if num_parts > 1:
                out[offset - _FRAME_BYTES : offset].copy_(
                    torch.tensor([written], dtype=torch.int32).view(torch.uint8)
                )
            offset += written
        return offset

    @staticmethod
    def _serialize_part(plan: _Plan, raw: torch.Tensor, out: torch.Tensor) -> int:
        if plan.mode == "raw":
            n = raw.numel() * raw.element_size()
            out[:n].copy_(raw.contiguous().view(torch.uint8).view(-1))
            return n
        raw_cpu = raw if raw.device.type == "cpu" else raw.cpu()
        written = encode_chunk_into_with_plan(plan, raw_cpu, out[: plan.size])
        if written > plan.size:
            raise RuntimeError(
                f"kvweave payload ({written} B) exceeds its estimate ({plan.size} B)"
            )
        return int(written)


class KVWeaveDeserializer(Deserializer):
    """Decode an L2 byte buffer back into the L1 chunk."""

    def __init__(self, resolver: _PlanResolver) -> None:
        self._resolver = resolver

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        """Decode every part of the stored buffer into ``dst``'s tensors."""
        payload = _flat_u8(src)
        num_parts = len(dst.get_shapes())
        offset = 0
        for part in range(num_parts):
            out = _part_tensor(dst, part)
            plan = self._resolver.plan(key, part, out.shape, out.dtype)
            if num_parts > 1:
                length = int(payload[offset : offset + _FRAME_BYTES].view(torch.int32))
                offset += _FRAME_BYTES
            else:
                length = payload.numel()
            self._deserialize_part(plan, payload[offset : offset + length], out)
            offset += length

    @staticmethod
    def _deserialize_part(
        plan: _Plan, payload: torch.Tensor, out: torch.Tensor
    ) -> None:
        if plan.mode == "raw":
            n = out.numel() * out.element_size()
            out.view(torch.uint8).view(-1).copy_(payload[:n])
            return
        assert plan.codec is not None and plan.spec is not None
        spec = plan.spec
        plane = spec.attention_plane_layout if plan.mode in ("split", "fused") else None
        plan.codec.decode_chunk(
            "mamba" if plan.mode == "mamba" else "attention",
            spec.mamba_layout if plan.mode == "mamba" else None,
            spec.tokens_per_block,
            out.shape,
            out.dtype,
            payload,
            attention_plane_layout=plane,
            out=out,
        )


def _create_kvweave_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    resolver = _PlanResolver(kwargs)
    return AsyncSerdeProcessor(
        KVWeaveSerializer(resolver),
        KVWeaveDeserializer(resolver),
        max_workers=int(kwargs.get("max_workers", 1)),  # type: ignore[call-overload]
    )


register_serde_factory("kvweave", _create_kvweave_serde)
