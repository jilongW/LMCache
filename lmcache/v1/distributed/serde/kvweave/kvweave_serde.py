"""KVWeave attention and Mamba codec for LMCache L1 quantization."""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Optional

import numpy as np
import torch

from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    AttentionPlaneLayout,
    ConvQKVSplit,
    KVWeaveCodecConfig,
    MambaCodecOptions,
)
from lmcache.v1.multiprocess.group_view import MambaSubStateWireLayout

try:
    from kvweave import kvweave_quant
except ImportError:  # pragma: no cover
    kvweave_quant = None

try:
    from kvweave import kvweave_quant_xpu
except ImportError:  # pragma: no cover
    kvweave_quant_xpu = None

logger = init_logger(__name__)


def _resolve_native(device: str):
    """Return the native quantization module for ``device`` ('cpu' or 'xpu').

    Single canonical native-module resolver used everywhere in this file --
    both modules expose identical kvweave_serialize_chunk*/
    kvweave_dequantize_chunk* signatures (the XPU ones added a native
    ``num_layers`` axis and fused K/V entry points in the SYCL kernels,
    mirroring the CPU wrapper's fused chunk entry points -- see
    quant_sycl.cpp / kvweave_quant_xpu_wrapper.cpp), so every call site here
    just picks a module and calls the same function name/arguments
    regardless of device.
    """
    if device == "xpu":
        if kvweave_quant_xpu is None:
            raise RuntimeError("KVWeave native XPU quantization extension is unavailable")
        return kvweave_quant_xpu
    if kvweave_quant is None:
        raise RuntimeError("KVWeave native quantization extension is unavailable")
    return kvweave_quant


@dataclass
class _KVShape:
    tensor4d: torch.Tensor
    header_shape: tuple[int, ...]
    num_layers: int
    chunk_tokens: int
    hidden_dim: int
    original_ndim: int


@dataclass(frozen=True)
class MambaChunkSplit:
    conv: torch.Tensor
    ssm: torch.Tensor


class _KVWeaveCodec:
    """Self-describing attention and Mamba KVWeave codec."""

    def __init__(self, kwargs: dict[str, object] | None = None):
        kwargs = kwargs or {}
        config = KVWeaveCodecConfig(
            quantize=bool(kwargs.get("quantize", True)),
            qbit=int(kwargs.get("qbit", 4)),
            scaling_method=str(kwargs.get("scaling_method", "per_channel")),
            rh=bool(kwargs.get("rh", True)),
            asym=bool(kwargs.get("asym", True)),
            log=bool(kwargs.get("log", False)),
            precond=bool(kwargs.get("precond", False)),
            num_threads=int(kwargs.get("num_threads", 8)),
            precond_seed=int(kwargs.get("precond_seed", 42)),
            precond_path=kwargs.get("precond_path") or None,
        )
        self._config = config
        self.quantize = config.quantize
        self.qbit = config.qbit
        self.scaling_method = config.scaling_method
        self.rh = config.rh
        self.asym = config.asym
        self.precond = config.precond
        self.num_threads = config.num_threads
        self.block_size = int(kwargs.get("block_size", config.DEFAULT_BLOCK_SIZE))
        self.num_kv_heads = int(kwargs.get("num_kv_heads", kwargs.get("head_num", 1)))
        self.head_dim = int(kwargs.get("head_dim", 0))
        self.device = str(kwargs.get("device", "cpu")).strip().lower()
        if self.device not in {"cpu", "xpu"}:
            raise ValueError(f"device={self.device!r} is not one of ['cpu', 'xpu']")
        if self.device == "xpu" and not torch.xpu.is_available():
            raise RuntimeError("device='xpu' but torch.xpu.is_available() is False")
        self.native = _resolve_native(self.device)

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> bytes:
        return bytes(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())

    @staticmethod
    def _payload_u8(tensor: torch.Tensor) -> torch.Tensor:
        payload = tensor.detach()
        if payload.dtype != torch.uint8:
            raise ValueError("KVWeave payload tensor must have dtype torch.uint8")
        if payload.device.type != "cpu":
            payload = payload.cpu()
        if not payload.is_contiguous():
            payload = payload.contiguous()
        return payload.view(torch.uint8)

    @staticmethod
    def _payload_buffer(payload: torch.Tensor) -> memoryview:
        return memoryview(payload.numpy())

    @staticmethod
    def _payload_tail(payload: torch.Tensor, offset: int, dtype: torch.dtype) -> torch.Tensor:
        if dtype == torch.int8:
            return payload[offset:].view(torch.int8)
        if dtype == torch.uint8:
            return payload[offset:]
        if dtype == torch.int16:
            if offset % 2:
                raise ValueError("KVWeave int16 q_data is not 2-byte aligned")
            return payload[offset:].view(torch.int16)
        raise ValueError(f"unsupported KVWeave payload view dtype: {dtype}")

    def estimate_serialized_size(
        self, layout_desc: MemoryLayoutDesc, scaling_method: Optional[str] = None
    ) -> int:
        """Estimate the upper bound needed for one serialized KV buffer."""
        return sum(
            self._estimate_shape(tuple(int(dim) for dim in shape), dtype, scaling_method)
            for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
        )

    def estimate_fused_serialized_size(
        self, layout_desc: MemoryLayoutDesc, scaling_method: Optional[str] = None
    ) -> int:
        """Estimate the upper bound for one serialized fused-K/V KV buffer."""
        return sum(
            self._estimate_fused_shape(
                tuple(int(dim) for dim in shape), dtype, scaling_method
            )
            for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
        )

    def _estimate_fused_shape(
        self, shape: tuple[int, ...], dtype: torch.dtype, method: str | None
    ) -> int:
        if len(shape) != 3:
            raise ValueError(
                f"KVWeave fused-K/V estimate expects [L, T, H], got {shape}"
            )
        layers, tokens, hidden = shape
        method = method or self.scaling_method
        elements = int(layers) * int(tokens) * int(hidden)
        if not self.quantize:
            return 32 + elements * dtype.itemsize
        scales = self._fused_scale_count(tokens, hidden, layers, method)
        header = 32 + 4 * len(shape)
        return int(
            (header + 4 + scales * 12 + KVWeaveCodecConfig.quantized_bytes(elements, self.qbit))
            * 1.02
        ) + 4096

    def _estimate_shape(self, shape: tuple[int, ...], dtype: torch.dtype, method: str | None) -> int:
        if len(shape) == 3:
            tokens, kv_size, hidden = shape
            layers = 1
        elif len(shape) == 4:
            kv_size, layers, tokens, hidden = shape
        else:
            return 64 + dtype.itemsize * int(torch.tensor(shape).prod().item())
        if kv_size != 2:
            raise ValueError(f"KVWeave expects K/V dimension 2, got {shape}")
        method = method or self.scaling_method
        elements = int(layers) * int(tokens) * int(hidden)
        if not self.quantize:
            return 32 + 2 * elements * dtype.itemsize
        scales = self._scale_count(tokens, hidden, layers, method)
        header = 32 + 4 * len(shape)
        return int((header + 8 + 2 * (4 + scales * 12) + 2 * KVWeaveCodecConfig.quantized_bytes(elements, self.qbit)) * 1.02) + 4096

    def serialize_tensor(self, tensor: torch.Tensor, scaling_method: str | None = None) -> bytes:
        """Normalize a KV tensor and serialize it as raw or 4-bit data."""
        work = tensor.detach().to("cpu").contiguous()
        shape = self._normalize(work)
        method = scaling_method or self.scaling_method
        if not self.quantize:
            return self._raw_payload(work, shape)
        rh, asym, precond = (False, False, False) if method == "per_tensor" else (self.rh, self.asym, self.precond)
        header = self._quant_header(shape, work.dtype, method, rh, asym, precond)
        ids = KVWeaveCodecConfig.next_scale_ids(2)
        payload = bytes(self.native.kvweave_serialize_chunk(
            shape.tensor4d, header, ids[0], ids[1], qbit=self.qbit,
            blocks_num=max(1, shape.chunk_tokens // self.block_size),
            block_size=self.block_size, head_num=self._head_num(shape.hidden_dim),
            head_dim=self._head_dim(shape.hidden_dim), num_layers=shape.num_layers,
            rh=rh, asym=asym, scaling_method=method, num_threads=self.num_threads,
        ))
        raw_bytes = work.numel() * work.element_size()
        logger.debug(
            "KVWeave quantize store shape=%s dtype=%s qbit=%d scaling=%s "
            "raw_bytes=%d payload_bytes=%d ratio=%.4f",
            tuple(work.shape), work.dtype, self.qbit, method, raw_bytes,
            len(payload), len(payload) / raw_bytes if raw_bytes else 0.0,
        )
        return payload

    def deserialize_tensor(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Decode a payload and restore it into the destination KV tensor."""
        payload = self._payload_u8(src)
        parsed = self._read_header(payload)
        if parsed["raw"]:
            data = self._payload_tail(payload, parsed["data_offset"], parsed["dtype"])
            dst.copy_(data.reshape(dst.shape).to(dtype=dst.dtype, device=dst.device))
            return
        shape = parsed["shape4d"]
        q_dtype = torch.int8 if parsed["qbit"] <= 8 else torch.int16
        q = self._payload_tail(payload, parsed["q_offset"], q_dtype)
        kwargs = dict(
            qbit=parsed["qbit"], blocks_num=parsed["blocks_num"], block_size=self.block_size,
            head_num=parsed["head_num"], head_dim=parsed["head_dim"], rh=parsed["rh"],
            asym=parsed["asym"], scaling_method=parsed["scaling"], output_dtype=dst.dtype,
            num_threads=self.num_threads,
        )
        if (
            hasattr(self.native, "kvweave_dequantize_chunk_into_4d")
            and dst.dim() == 4
            and dst.device.type == "cpu"
        ):
            self.native.kvweave_dequantize_chunk_into_4d(
                q, parsed["k_scales"], parsed["v_scales"], dst, shape[1], shape[2], shape[3], **kwargs
            )
            return
        restored = self.native.kvweave_dequantize_chunk(
            q, parsed["k_scales"], parsed["v_scales"], shape[1], shape[2], shape[3], **kwargs
        )
        if dst.dim() == 3:
            restored = restored.squeeze(1).permute(1, 0, 2)
        dst.copy_(restored.to(dtype=dst.dtype, device=dst.device))

    def serialize_fused_tensor(
        self,
        tensor: torch.Tensor,
        scaling_method: str | None = None,
        head_split: torch.Tensor | None = None,
    ) -> bytes:
        """Serialize a fused-K/V attention chunk (no leading K/V axis)."""
        if tensor.dim() != 3:
            raise ValueError(
                f"KVWeave fused-K/V tensor expects [L, T, H], got {tuple(tensor.shape)}"
            )
        work = tensor.detach().to("cpu").contiguous()
        shape = tuple(int(dim) for dim in work.shape)
        layers, tokens, hidden = shape
        method = scaling_method or self.scaling_method
        rh, asym = (False, False) if method == "per_tensor" else (self.rh, self.asym)
        precond = rh and self.precond
        head_num = self._fused_head_num(hidden)
        head_dim = self._fused_head_dim(hidden)
        per_head_scales = method != "per_tensor" and head_num > 1
        flags = (
            (1 if rh else 0)
            | (2 if asym else 0)
            | (4 if precond else 0)
            | (8 if per_head_scales else 0)
        )
        header = self._config.MAGIC_QUANT_FUSED + struct.pack(
            ">BBBBB" + "i" * len(shape),
            self.qbit,
            self._config.DTYPE_TO_CODE.get(work.dtype, 0),
            flags,
            self._config.SCALING_TO_CODE.get(method, 1),
            len(shape),
            *shape,
        )
        blocks_num = max(1, tokens // self.block_size)
        signs = perm = None
        if precond:
            transform_size = self._fused_rh_transform_size(
                tokens, hidden, head_dim, method, per_head_scales
            )
            signs, perm = self._config.mamba_precond_tensors(transform_size)
        native_src = work
        native_layers = layers
        if per_head_scales:
            expected_shape = (layers * head_num, tokens, head_dim)
            if head_split is not None:
                if (
                    tuple(head_split.shape) != expected_shape
                    or head_split.dtype != work.dtype
                    or not head_split.is_contiguous()
                ):
                    raise ValueError("head_split does not match fused attention layout")
                native_src = head_split
            else:
                native_src = self.split_fused_attention_heads(work, head_num, head_dim)
            native_layers *= head_num
        return bytes(
            self.native.kvweave_serialize_chunk_state(
                native_src.view(-1), header, KVWeaveCodecConfig.next_scale_id(),
                qbit=self.qbit, blocks_num=blocks_num,
                block_size=self.block_size, head_num=head_num, head_dim=head_dim,
                num_layers=native_layers, rh=rh, asym=asym, scaling_method=method,
                signs=signs, perm=perm,
                num_threads=self.num_threads,
            )
        )

    @staticmethod
    def split_fused_attention_heads(
        source: torch.Tensor,
        head_num: int,
        head_dim: int,
        out: torch.Tensor | None = None,
        num_threads: int = 0,
    ) -> torch.Tensor:
        """Reorder fused ``[L,T,H*D]`` attention into contiguous head rows."""
        layers, tokens, hidden = map(int, source.shape)
        expected_shape = (layers * head_num, tokens, head_dim)
        if hidden != head_num * head_dim:
            raise ValueError("fused hidden size does not match head layout")
        if out is not None and (
            tuple(out.shape) != expected_shape
            or out.dtype != source.dtype
            or not out.is_contiguous()
        ):
            raise ValueError("out does not match fused attention head layout")
        if kvweave_quant is not None and hasattr(kvweave_quant, "kvweave_split_fused_heads"):
            return kvweave_quant.kvweave_split_fused_heads(
                source.contiguous(), head_num, head_dim, num_threads, out
            )
        if out is None:
            out = torch.empty(expected_shape, dtype=source.dtype)
        out.copy_(
            source.reshape(layers, tokens, head_num, head_dim)
            .permute(0, 2, 1, 3)
            .reshape(expected_shape)
        )
        return out

    def deserialize_fused_tensor(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Decode a fused-K/V payload and restore it into the destination tensor."""
        payload = self._payload_u8(src)
        parsed = self._read_fused_header(payload)
        layers, tokens, hidden = parsed["shape"]
        blocks_num = max(1, tokens // self.block_size)
        q_dtype = torch.int8 if parsed["qbit"] <= 8 else torch.int16
        q = self._payload_tail(payload, parsed["q_offset"], q_dtype)
        signs = perm = None
        if parsed["precond"]:
            transform_size = self._fused_rh_transform_size(
                tokens,
                hidden,
                parsed["head_dim"],
                parsed["scaling"],
                parsed["per_head_scales"],
            )
            signs, perm = self._config.mamba_precond_tensors(transform_size)
        native_layers = layers * parsed["head_num"] if parsed["per_head_scales"] else layers
        native_hidden = parsed["head_dim"] if parsed["per_head_scales"] else hidden
        restored = self.native.kvweave_dequantize_chunk_state(
            q, parsed["scales"], native_layers, tokens, native_hidden,
            qbit=parsed["qbit"], blocks_num=blocks_num,
            block_size=self.block_size, head_num=parsed["head_num"],
            head_dim=parsed["head_dim"], rh=parsed["rh"], asym=parsed["asym"],
            scaling_method=parsed["scaling"], output_dtype=dst.dtype,
            signs=signs, perm=perm,
            num_threads=self.num_threads,
        )
        if parsed["per_head_scales"]:
            if kvweave_quant is not None and hasattr(kvweave_quant, "kvweave_merge_fused_heads"):
                restored = kvweave_quant.kvweave_merge_fused_heads(
                    restored, layers, parsed["head_num"], native_hidden, self.num_threads
                )
            else:
                restored = (
                    restored.reshape(layers, parsed["head_num"], tokens, native_hidden)
                    .permute(0, 2, 1, 3)
                    .reshape(layers, tokens, hidden)
                    .contiguous()
                )
        dst.copy_(restored.reshape(dst.shape).to(dtype=dst.dtype, device=dst.device))

    def _read_fused_header(self, payload: torch.Tensor) -> dict[str, object]:
        """Read a fused-K/V header without copying the quantized payload."""
        raw = self._payload_buffer(payload)
        magic = bytes(raw[:4])
        if magic != self._config.MAGIC_QUANT_FUSED:
            raise ValueError(f"invalid KVWeave fused payload magic: {magic!r}")
        offset = 4
        qbit, dtype_code, flags, scaling_code, ndim = struct.unpack_from(
            ">BBBBB", raw, offset
        )
        offset += 5
        shape = tuple(struct.unpack_from(">" + "i" * ndim, raw, offset))
        offset += 4 * ndim
        if len(shape) != 3:
            raise ValueError(f"KVWeave fused payload expects 3-D shape, got {shape}")
        (scale_len,) = struct.unpack_from(">I", raw, offset)
        offset += 4
        scales = bytes(raw[offset : offset + scale_len])
        offset += scale_len
        method = {
            value: key for key, value in self._config.SCALING_TO_CODE.items()
        }.get(scaling_code, self.scaling_method)
        hidden = shape[2]
        return {
            "shape": shape,
            "qbit": qbit,
            "rh": bool(flags & 1),
            "asym": bool(flags & 2),
            "precond": bool(flags & 4),
            "per_head_scales": bool(flags & 8),
            "scaling": method,
            "dtype": self._config.CODE_TO_DTYPE.get(dtype_code, torch.float16),
            "scales": scales,
            "q_offset": offset,
            "head_num": self._fused_head_num(hidden),
            "head_dim": self._fused_head_dim(hidden),
        }

    @staticmethod
    def _fused_rh_transform_size(
        tokens: int,
        hidden: int,
        head_dim: int,
        method: str,
        per_head_scales: bool = False,
    ) -> int:
        if per_head_scales:
            return head_dim if method == "per_token" else tokens
        if method == "per_token":
            return hidden
        if method == "per_channel":
            if hidden % head_dim:
                raise ValueError("fused hidden size must divide by head_dim")
            return tokens * (hidden // head_dim)
        return tokens * hidden

    def _normalize(self, tensor: torch.Tensor) -> _KVShape:
        """Convert supported 3D/4D KV layouts to canonical 4D metadata."""
        if tensor.dim() == 3:
            tokens, kv_size, hidden = map(int, tensor.shape)
            if kv_size != 2:
                raise ValueError("KVWeave expects [T, 2, H]")
            return _KVShape(tensor.permute(1, 0, 2).unsqueeze(1).contiguous(), tuple(tensor.shape), 1, tokens, hidden, 3)
        if tensor.dim() == 4:
            kv_size, layers, tokens, hidden = map(int, tensor.shape)
            if kv_size != 2:
                raise ValueError("KVWeave expects [2, L, T, H]")
            return _KVShape(tensor, tuple(tensor.shape), layers, tokens, hidden, 4)
        raise ValueError(f"unsupported KVWeave tensor shape: {tuple(tensor.shape)}")

    def _raw_payload(self, tensor: torch.Tensor, shape: _KVShape) -> bytes:
        header = struct.pack(">4sBBB", self._config.MAGIC_RAW, shape.original_ndim, 1, self._config.DTYPE_TO_CODE.get(tensor.dtype, 0))
        return header + struct.pack(">" + "i" * shape.original_ndim, *shape.header_shape) + self._tensor_bytes(tensor)

    def _quant_header(self, shape: _KVShape, dtype: torch.dtype, method: str, rh: bool, asym: bool, precond: bool) -> bytes:
        return struct.pack(">4sBBBBBBBB" + "i" * shape.original_ndim, self._config.MAGIC_QUANT, self.qbit, int(rh), int(asym), int(precond), shape.original_ndim, 1, self._config.SCALING_TO_CODE.get(method, 1), self._config.DTYPE_TO_CODE.get(dtype, 0), *shape.header_shape)

    def _read_header(self, payload: torch.Tensor) -> dict[str, object]:
        """Read the attention payload header without copying q_data."""
        raw = self._payload_buffer(payload)
        magic = bytes(raw[:4])
        offset = 4
        if magic == self._config.MAGIC_RAW:
            ndim, _, dtype_code = struct.unpack_from(">BBB", raw, offset)
            offset += 3
            shape = tuple(struct.unpack_from(">" + "i" * ndim, raw, offset))
            offset += 4 * ndim
            return {"raw": True, "shape": shape, "dtype": self._config.CODE_TO_DTYPE.get(dtype_code, torch.float16), "data_offset": offset}
        if magic != self._config.MAGIC_QUANT:
            raise ValueError(f"invalid KVWeave payload magic: {magic!r}")
        qbit, rh, asym, _, ndim, _, scaling_code, dtype_code = struct.unpack_from(">BBBBBBBB", raw, offset)
        offset += 8
        shape = tuple(struct.unpack_from(">" + "i" * ndim, raw, offset))
        offset += 4 * ndim
        shape4d = (shape[1], 1, shape[0], shape[2]) if ndim == 3 else shape
        scales = []
        for _ in range(2):
            size = struct.unpack_from(">I", raw, offset)[0]
            offset += 4
            scales.append(bytes(raw[offset:offset + size]))
            offset += size
        method = {value: key for key, value in self._config.SCALING_TO_CODE.items()}.get(scaling_code, self.scaling_method)
        return {"raw": False, "shape4d": shape4d, "qbit": qbit, "rh": bool(rh), "asym": bool(asym), "scaling": method, "dtype": self._config.CODE_TO_DTYPE.get(dtype_code, torch.float16), "k_scales": scales[0], "v_scales": scales[1], "q_offset": offset, "blocks_num": max(1, shape4d[2] // self.block_size), "head_num": self._head_num(shape4d[3]), "head_dim": self._head_dim(shape4d[3])}

    def _head_num(self, hidden: int) -> int:
        return self.num_kv_heads if self.num_kv_heads > 1 and hidden % self.num_kv_heads == 0 else 1

    def _head_dim(self, hidden: int) -> int:
        return self.head_dim or hidden // self._head_num(hidden)

    def _fused_head_dim(self, hidden: int) -> int:
        """Width of one K or V plane in a fused ``[K|V]`` content axis."""
        return hidden // self._fused_head_num(hidden)

    def _fused_head_num(self, hidden: int) -> int:
        """Treat each packed K/V half as its own native quantization head.

        vLLM stores fused attention as ``[NH, K|V, HS]`` flattened to a
        trailing ``CS = 2 * HS`` axis. Passing ``NH`` and ``2*HS`` to the
        native codec incorrectly groups each K/V pair under one scale/RH
        transform. ``2*NH`` heads of width ``HS`` preserves the byte order
        while keeping K and V as independent quantization planes.
        """
        heads = self._head_num(hidden)
        if hidden % (2 * heads) == 0:
            return 2 * heads
        return 1

    def _fused_scale_count(
        self, tokens: int, hidden: int, layers: int, method: str
    ) -> int:
        base = (
            tokens
            if method == "per_token"
            else self._fused_head_dim(hidden)
            if method == "per_channel"
            else 1
        )
        if method != "per_tensor":
            layers *= self._fused_head_num(hidden)
        return max(1, base * layers if layers > 1 else base)

    def _scale_count(self, tokens: int, hidden: int, layers: int, method: str) -> int:
        base = tokens if method == "per_token" else self._head_dim(hidden) if method == "per_channel" else 1
        return max(1, base * layers if layers > 1 else base)

    @staticmethod
    def _mamba_layout(substate: str, shape: tuple[int, ...], method: str) -> tuple[int, int, int, int]:
        return KVWeaveCodecConfig.mamba_layout(substate, shape, method)

    @staticmethod
    def _mamba_precond_pair(size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return KVWeaveCodecConfig().mamba_precond_tensors(size)

    @staticmethod
    def split_mamba_chunk(
        raw: torch.Tensor,
        layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout],
        block_size: int,
        out: MambaChunkSplit | None = None,
    ) -> MambaChunkSplit:
        """Recover real conv/ssm bytes from the opaque synthetic page view.

        When ``out`` is provided its contiguous sub-state tensors receive the
        recovered bytes. This lets engine-driven store use pool scratch rather
        than allocate ordinary CPU tensors for Stage 2.
        """
        conv_layout, ssm_layout = layout
        if block_size <= 0 or raw.shape[-2] % block_size:
            raise ValueError(
                f"chunk_tokens ({raw.shape[-2]}) is not a multiple of block_size ({block_size})"
            )
        is_two_plane = raw.dim() == 4 and int(raw.shape[0]) == 2
        if is_two_plane:
            _, layers, tokens, hidden = map(int, raw.shape)
            page_bytes = 2 * block_size * hidden * raw.element_size()
        elif raw.dim() == 3:
            layers, tokens, hidden = map(int, raw.shape)
            page_bytes = block_size * hidden * raw.element_size()
        else:
            raise ValueError(
                f"expected Mamba chunk shape [2,L,T,H] or [L,T,H], got {tuple(raw.shape)}"
            )
        blocks = tokens // block_size
        blocks = tokens // block_size

        def validate_and_target(
            desc: MambaSubStateWireLayout,
            dtype: torch.dtype,
            target: torch.Tensor | None,
        ) -> torch.Tensor:
            end = desc.byte_offset + desc.byte_length
            if desc.byte_offset < 0 or end > page_bytes:
                raise ValueError("Mamba sub-state byte layout exceeds page size")
            expected_shape = (layers, blocks, *desc.shape)
            if target is None:
                return torch.empty(expected_shape, dtype=dtype)
            if (
                target.shape != expected_shape
                or target.dtype != dtype
                or not target.is_contiguous()
            ):
                raise ValueError("Mamba split out tensor does not match real layout")
            return target

        def read(desc: MambaSubStateWireLayout, dtype: torch.dtype, target: torch.Tensor | None) -> torch.Tensor:
            destination = validate_and_target(desc, dtype, target)
            destination_bytes = destination.view(torch.uint8).reshape(layers, blocks, -1)
            end = desc.byte_offset + desc.byte_length
            if not is_two_plane:
                pages = raw.view(torch.uint8).reshape(layers, blocks, page_bytes)
                destination_bytes.copy_(pages[:, :, desc.byte_offset:end])
                return destination

            plane_bytes = block_size * hidden * raw.element_size()
            source_offset = 0
            page_offset = desc.byte_offset
            while source_offset < desc.byte_length:
                plane_index = page_offset // plane_bytes
                plane_offset = page_offset % plane_bytes
                copy_bytes = min(
                    desc.byte_length - source_offset, plane_bytes - plane_offset
                )
                # Vectorized over all (layer, block) pairs at once instead of a
                # per-(layer, block) Python loop -- raw[plane_index] is already
                # contiguous (slicing a contiguous tensor's outermost dim).
                plane_view = raw[plane_index].view(torch.uint8).reshape(
                    layers, blocks, plane_bytes
                )
                destination_bytes[
                    :, :, source_offset : source_offset + copy_bytes
                ] = plane_view[:, :, plane_offset : plane_offset + copy_bytes]
                source_offset += copy_bytes
                page_offset += copy_bytes
            return destination

        conv = read(
            conv_layout,
            KVWeaveCodecConfig.mamba_dtype(conv_layout.dtype_str),
            out.conv if out is not None else None,
        )
        ssm = read(
            ssm_layout,
            KVWeaveCodecConfig.mamba_dtype(ssm_layout.dtype_str),
            out.ssm if out is not None else None,
        )
        return MambaChunkSplit(conv, ssm)

    @staticmethod
    def merge_mamba_chunk(split: MambaChunkSplit, layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout], block_size: int, hidden_dim: int, raw_shape: torch.Size | None = None, raw_dtype: torch.dtype | None = None, out: torch.Tensor | None = None) -> torch.Tensor:
        """Rebuild the opaque page view, optionally in a caller-owned buffer."""
        conv_layout, ssm_layout = layout
        layers, blocks = map(int, split.conv.shape[:2])
        if split.ssm.shape[:2] != (layers, blocks):
            raise ValueError("conv and ssm chunks have different layer/block dimensions")
        output_dtype = raw_dtype or split.conv.dtype
        planes = 1 if raw_shape is not None and len(raw_shape) == 3 else 2
        page_bytes = planes * block_size * hidden_dim * output_dtype.itemsize
        if out is not None:
            if raw_shape is None or raw_dtype is None:
                raise ValueError("raw_shape and raw_dtype are required with out")
            if out.shape != raw_shape or out.dtype != raw_dtype or not out.is_contiguous():
                raise ValueError("out must be a contiguous tensor matching raw_shape/raw_dtype")

            def write_to_out(tensor: torch.Tensor, desc: MambaSubStateWireLayout) -> None:
                raw = tensor.contiguous().view(torch.uint8).reshape(layers, blocks, -1)
                end = desc.byte_offset + desc.byte_length
                if raw.shape[-1] != desc.byte_length or end > page_bytes:
                    raise ValueError("Mamba sub-state tensor does not match byte layout")
                if planes == 1:
                    out.view(torch.uint8).reshape(layers, blocks, page_bytes)[
                        :, :, desc.byte_offset:end
                    ] = raw
                    return
                plane_bytes = block_size * hidden_dim * output_dtype.itemsize
                source_offset = 0
                page_offset = desc.byte_offset
                while source_offset < desc.byte_length:
                    plane_index = page_offset // plane_bytes
                    plane_offset = page_offset % plane_bytes
                    copy_bytes = min(
                        desc.byte_length - source_offset,
                        plane_bytes - plane_offset,
                    )
                    # Vectorized over all (layer, block) pairs at once instead
                    # of a per-block Python loop -- out[plane_index] is
                    # already contiguous (slicing a contiguous tensor's
                    # outermost dim).
                    dest_view = out[plane_index].view(torch.uint8).reshape(
                        layers, blocks, plane_bytes
                    )
                    dest_view[:, :, plane_offset : plane_offset + copy_bytes] = raw[
                        :, :, source_offset : source_offset + copy_bytes
                    ]
                    source_offset += copy_bytes
                    page_offset += copy_bytes

            write_to_out(split.conv, conv_layout)
            write_to_out(split.ssm, ssm_layout)
            return out
        pages = torch.zeros(layers, blocks, page_bytes, dtype=torch.uint8)
        for tensor, desc in ((split.conv, conv_layout), (split.ssm, ssm_layout)):
            raw = tensor.contiguous().view(torch.uint8).reshape(layers, blocks, -1)
            end = desc.byte_offset + desc.byte_length
            if raw.shape[-1] != desc.byte_length or end > page_bytes:
                raise ValueError("Mamba sub-state tensor does not match byte layout")
            pages[:, :, desc.byte_offset:end] = raw
        if raw_shape is not None and len(raw_shape) == 3:
            return pages.view(output_dtype).reshape(raw_shape).contiguous()
        return pages.view(split.conv.dtype).reshape(layers, blocks, 2, block_size, hidden_dim).permute(2, 0, 1, 3, 4).reshape(2, layers, blocks * block_size, hidden_dim).contiguous()

    @staticmethod
    def _quantize_mamba_substate_payload(
        tensor: torch.Tensor,
        *,
        substate: str,
        scaling_method: str,
        rh: bool,
        asym: bool,
        qbit: int = KVWeaveCodecConfig.MAMBA_QBIT,
        device: str = "cpu",
    ) -> bytes:
        """Quantize one real Mamba sub-state with native state kernels."""
        if tensor.dtype not in KVWeaveCodecConfig.DTYPE_TO_CODE:
            raise ValueError(f"unsupported dtype for 4-bit quantization: {tensor.dtype}")
        if substate not in KVWeaveCodecConfig.SUBSTATE_TO_CODE:
            raise ValueError(f"unsupported substate: {substate!r}")
        if scaling_method not in KVWeaveCodecConfig.SCALING_TO_CODE:
            raise ValueError(f"unsupported scaling_method: {scaling_method!r}")
        if tensor.dim() < 2 :
            raise ValueError("invalid Mamba tensor or RH configuration")
        work = tensor.detach().to("cpu").contiguous()
        shape = tuple(int(dim) for dim in work.shape)
        blocks, heads, head_dim, chunks = _KVWeaveCodec._mamba_layout(substate, shape, scaling_method)
        signs = perm = None
        if rh:
            signs, perm = _KVWeaveCodec._mamba_precond_pair(
                max(work.numel() // shape[0] // chunks, 1)
            )
        flags = (KVWeaveCodecConfig.MAMBA_FLAG_RH if rh else 0) | (KVWeaveCodecConfig.MAMBA_FLAG_ASYM if asym else 0)
        header = KVWeaveCodecConfig.MAMBA_MAGIC + struct.pack(">BBBBBB" + "i" * len(shape), qbit, KVWeaveCodecConfig.DTYPE_TO_CODE[work.dtype], flags, KVWeaveCodecConfig.SCALING_TO_CODE[scaling_method], KVWeaveCodecConfig.SUBSTATE_TO_CODE[substate], len(shape), *shape)
        native = _resolve_native(device)
        payload = bytes(native.kvweave_serialize_chunk_state(
            work.view(-1), header, KVWeaveCodecConfig.next_scale_id(),
            qbit=qbit, blocks_num=blocks,
            block_size=1, head_num=heads, head_dim=head_dim,
            num_layers=shape[0], rh=rh, asym=asym,
            scaling_method=scaling_method, signs=signs, perm=perm,
        ))
        raw_bytes = work.numel() * work.element_size()
        logger.debug(
            "Mamba quantize %s shape=%s dtype=%s qbit=%d raw_bytes=%d "
            "payload_bytes=%d ratio=%.4f",
            substate, shape, work.dtype, qbit, raw_bytes, len(payload),
            len(payload) / raw_bytes if raw_bytes else 0.0,
        )
        return payload

    @staticmethod
    def _quantize_mamba_substates_batch_xpu(jobs: list[dict]) -> list[bytes]:
        """Batch >=2 independent Mamba substate quantize jobs into one XPU
        round trip via ``kvweave_quant_xpu.kvweave_serialize_chunk_state_multi``.

        Each ``job`` dict carries the same per-item inputs
        :meth:`_quantize_mamba_substate_payload` takes (``tensor``,
        ``substate``, ``scaling_method``, ``rh``, ``asym``, ``qbit``).
        Produces byte-identical payloads to calling
        ``quantize_mamba_substate_4bit(..., device="xpu")`` once per job --
        this only collapses the redundant per-job H2D upload/D2H download
        into one combined transfer, never changes the wire format. All jobs
        must share one tensor dtype (enforced by the native entry point);
        callers group jobs by dtype first (see :meth:`_quantize_mamba_jobs`).
        """
        native = _resolve_native("xpu")
        works, headers = [], []
        blocks_list, block_sizes, head_nums, head_dims = [], [], [], []
        num_layers_list, rh_list, asym_list, scaling_list = [], [], [], []
        signs_list, perm_list = [], []
        for job in jobs:
            tensor = job["tensor"]
            work = tensor.detach().to("cpu").contiguous()
            shape = tuple(int(dim) for dim in work.shape)
            blocks, heads, head_dim, chunks = _KVWeaveCodec._mamba_layout(
                job["substate"], shape, job["scaling_method"]
            )
            signs = perm = None
            if job["rh"]:
                signs, perm = _KVWeaveCodec._mamba_precond_pair(
                    max(work.numel() // shape[0] // chunks, 1)
                )
            flags = (
                (KVWeaveCodecConfig.MAMBA_FLAG_RH if job["rh"] else 0)
                | (KVWeaveCodecConfig.MAMBA_FLAG_ASYM if job["asym"] else 0)
            )
            header = KVWeaveCodecConfig.MAMBA_MAGIC + struct.pack(
                ">BBBBBB" + "i" * len(shape),
                job["qbit"], KVWeaveCodecConfig.DTYPE_TO_CODE[work.dtype], flags,
                KVWeaveCodecConfig.SCALING_TO_CODE[job["scaling_method"]],
                KVWeaveCodecConfig.SUBSTATE_TO_CODE[job["substate"]], len(shape), *shape,
            )
            works.append(work.view(-1))
            headers.append(header)
            blocks_list.append(blocks)
            block_sizes.append(1)
            head_nums.append(heads)
            head_dims.append(head_dim)
            num_layers_list.append(shape[0])
            rh_list.append(job["rh"])
            asym_list.append(job["asym"])
            scaling_list.append(job["scaling_method"])
            signs_list.append(signs)
            perm_list.append(perm)
        return list(native.kvweave_serialize_chunk_state_multi(
            works, headers, [0] * len(jobs), [job["qbit"] for job in jobs],
            blocks_list, block_sizes, head_nums, head_dims, num_layers_list,
            rh_list, asym_list, scaling_list, signs_list, perm_list,
        ))

    @staticmethod
    def _quantize_mamba_jobs(jobs: list[dict], device: str) -> list[bytes]:
        """Quantize N independent Mamba substate jobs (e.g. conv-query/
        conv-key/conv-value/ssm), batching same-dtype groups into a single
        XPU round trip when ``device == "xpu"``. CPU device, a single job,
        or a dtype-singleton group all fall back to one native call per job
        -- unchanged from the pre-batching behavior, since there is no
        transfer to save in those cases.
        """
        if device != "xpu" or len(jobs) <= 1:
            return [
                _KVWeaveCodec.quantize_mamba_substate_4bit(
                    job["tensor"], substate=job["substate"], scaling_method=job["scaling_method"],
                    rh=job["rh"], asym=job["asym"], qbit=job["qbit"], device=device,
                )
                for job in jobs
            ]
        results: list[bytes] = [b""] * len(jobs)
        groups: dict[torch.dtype, list[int]] = {}
        for idx, job in enumerate(jobs):
            groups.setdefault(job["tensor"].dtype, []).append(idx)
        for indices in groups.values():
            if len(indices) == 1:
                idx = indices[0]
                job = jobs[idx]
                results[idx] = _KVWeaveCodec.quantize_mamba_substate_4bit(
                    job["tensor"], substate=job["substate"], scaling_method=job["scaling_method"],
                    rh=job["rh"], asym=job["asym"], qbit=job["qbit"], device=device,
                )
                continue
            batch_payloads = _KVWeaveCodec._quantize_mamba_substates_batch_xpu(
                [jobs[i] for i in indices]
            )
            for idx, payload in zip(indices, batch_payloads):
                results[idx] = payload
        return results

    @staticmethod
    def _decode_mamba_substate(
        flagged_payload: bytes,
        layout: MambaSubStateWireLayout,
        layers: int,
        blocks: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Decode one sub-state payload, honoring its leading quant-enabled flag.

        DEBUG ONLY dispatch (see ``LMCACHE_MP_KVWEAVE_CONV_QUANT_ENABLED``/
        ``SSM_QUANT_ENABLED``): a leading ``\\x00`` byte means ``encode_chunk``
        skipped 4-bit quantization for this sub-state and wrote its real
        bytes verbatim; ``\\x01`` means the rest is a normal MQ01 payload.
        """
        view = memoryview(flagged_payload)
        flag, payload = view[0], view[1:]
        if flag == 0:
            dtype = KVWeaveCodecConfig.mamba_dtype(layout.dtype_str)
            return torch.frombuffer(payload, dtype=dtype).reshape(
                layers, blocks, *layout.shape
            )
        if flag == 2:
            return torch.frombuffer(payload, dtype=torch.float16).reshape(
                layers, blocks, *layout.shape
            ).to(dtype=KVWeaveCodecConfig.mamba_dtype(layout.dtype_str))
        return _KVWeaveCodec.dequantize_mamba_substate_4bit(payload, device=device)

    @staticmethod
    def _decode_conv_substate(
        flagged_payload: bytes,
        layout: MambaSubStateWireLayout,
        layers: int,
        blocks: int,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Decode conv_state's payload, honoring its leading quant-enabled flag.

        Mirrors :meth:`_decode_mamba_substate`, except the ``\\x01``
        (quantized) branch's payload is a 3-way query/key/value bundle (see
        :meth:`_split_conv_qkv`/:meth:`pack_conv_qkv_payloads`) rather than a
        single MQ01 payload: each sub-tensor is self-describing and decoded
        independently, then concatenated back along the last (``conv_dim``)
        dimension. conv never produces flag ``2`` (that fp16 fallback is
        ssm-only, see ``encode_chunk``), but it's handled the same way as
        ``_decode_mamba_substate`` for symmetry.
        """
        view = memoryview(flagged_payload)
        flag, payload = view[0], view[1:]
        if flag == 0:
            dtype = KVWeaveCodecConfig.mamba_dtype(layout.dtype_str)
            return torch.frombuffer(payload, dtype=dtype).reshape(
                layers, blocks, *layout.shape
            )
        if flag == 2:
            return torch.frombuffer(payload, dtype=torch.float16).reshape(
                layers, blocks, *layout.shape
            ).to(dtype=KVWeaveCodecConfig.mamba_dtype(layout.dtype_str))
        query, key, value = _KVWeaveCodec.unpack_conv_qkv_payloads(payload)
        return torch.cat(
            [
                _KVWeaveCodec.dequantize_mamba_substate_4bit(sub, device=device)
                for sub in (query, key, value)
            ],
            dim=-1,
        )

    @staticmethod
    def _read_mamba_substate_payload(payload) -> dict[str, object]:
        """Read a native Mamba sub-state payload header without copying q_data.

        Split out of :meth:`dequantize_mamba_substate_4bit` so several
        payloads' native calls can be batched into one XPU round trip while
        the metadata decode stays single-sourced between the batched and
        single-item paths.
        """
        view = memoryview(payload)
        magic = bytes(view[:4])
        if magic != KVWeaveCodecConfig.MAMBA_MAGIC:
            raise ValueError(f"unrecognized payload magic: {magic!r}")
        qbit, dtype_code, flags, scaling_code, substate_code, ndim = struct.unpack_from(
            ">BBBBBB", view, 4
        )
        scaling_map = {v: k for k, v in KVWeaveCodecConfig.SCALING_TO_CODE.items()}
        substate_map = {v: k for k, v in KVWeaveCodecConfig.SUBSTATE_TO_CODE.items()}
        if qbit not in {4, 8} or dtype_code not in KVWeaveCodecConfig.CODE_TO_DTYPE:
            raise ValueError("unsupported qbit or dtype in Mamba payload")
        if scaling_code not in scaling_map or substate_code not in substate_map:
            raise ValueError("unsupported scaling method or substate in Mamba payload")
        offset = 10
        shape = tuple(struct.unpack_from(">" + "i" * ndim, view, offset))
        offset += 4 * ndim
        scaling = scaling_map[scaling_code]
        substate = substate_map[substate_code]
        numel = 1
        for dim in shape:
            numel *= dim
        num_layers = shape[0]
        num_blocks = shape[1]
        blocks, heads, head_dim, chunks = _KVWeaveCodec._mamba_layout(substate, shape, scaling)
        (scale_size,) = struct.unpack_from(">I", view, offset)
        offset += 4
        scales = bytes(view[offset:offset + scale_size])
        offset += scale_size
        signs = perm = None
        if flags & KVWeaveCodecConfig.MAMBA_FLAG_RH:
            transform = max(torch.tensor(shape).prod().item() // shape[0] // chunks, 1)
            signs, perm = _KVWeaveCodec._mamba_precond_pair(int(transform))
        output_dtype = KVWeaveCodecConfig.CODE_TO_DTYPE[dtype_code]
        q_data = torch.frombuffer(view[offset:], dtype=torch.int8)
        return {
            "qbit": qbit,
            "blocks_num": blocks,
            "head_num": heads,
            "head_dim": head_dim,
            "num_layers": num_layers,
            "num_blocks": num_blocks,
            "h_merged": numel // (num_layers * num_blocks),
            "rh": bool(flags & KVWeaveCodecConfig.MAMBA_FLAG_RH),
            "asym": bool(flags & KVWeaveCodecConfig.MAMBA_FLAG_ASYM),
            "scaling_method": scaling,
            "output_dtype": output_dtype,
            "signs": signs,
            "perm": perm,
            "q_data": q_data,
            "scales": scales,
            "shape": shape,
        }

    @staticmethod
    def dequantize_mamba_substate_4bit(payload: bytes, device: str = "cpu") -> torch.Tensor:
        """Decode a self-describing native Mamba sub-state payload."""
        p = _KVWeaveCodec._read_mamba_substate_payload(payload)
        native = _resolve_native(device)
        restored = native.kvweave_dequantize_chunk_state(
            p["q_data"], p["scales"], p["num_layers"], p["num_blocks"], p["h_merged"],
            qbit=p["qbit"], blocks_num=p["blocks_num"],
            block_size=1, head_num=p["head_num"], head_dim=p["head_dim"],
            rh=p["rh"], asym=p["asym"],
            scaling_method=p["scaling_method"],
            output_dtype=p["output_dtype"],
            signs=p["signs"], perm=p["perm"],
        )
        return restored.reshape(p["shape"])

    @staticmethod
    def _dequantize_mamba_substates_batch_xpu(parsed_list: list[dict]) -> list[torch.Tensor]:
        """Batch >=2 already-parsed Mamba substate specs into one XPU
        dequantize round trip via
        ``kvweave_quant_xpu.kvweave_dequantize_chunk_state_multi``. Requires
        all items share one output dtype (enforced by the native entry
        point); callers group parsed specs by output dtype first (see
        :meth:`_dequantize_mamba_payloads`).
        """
        native = _resolve_native("xpu")
        results = native.kvweave_dequantize_chunk_state_multi(
            [p["q_data"] for p in parsed_list],
            [p["scales"] for p in parsed_list],
            [p["num_layers"] for p in parsed_list],
            [p["num_blocks"] for p in parsed_list],
            [p["h_merged"] for p in parsed_list],
            [p["qbit"] for p in parsed_list],
            [p["blocks_num"] for p in parsed_list],
            [1] * len(parsed_list),
            [p["head_num"] for p in parsed_list],
            [p["head_dim"] for p in parsed_list],
            [p["rh"] for p in parsed_list],
            [p["asym"] for p in parsed_list],
            [p["scaling_method"] for p in parsed_list],
            [p["output_dtype"] for p in parsed_list],
            [p["signs"] for p in parsed_list],
            [p["perm"] for p in parsed_list],
        )
        return [r.reshape(p["shape"]) for r, p in zip(results, parsed_list)]

    @staticmethod
    def _dequantize_mamba_payloads(payloads: list[bytes], device: str) -> list[torch.Tensor]:
        """Dequantize N independent Mamba substate payloads, batching
        same-output-dtype groups into a single XPU round trip when
        ``device == "xpu"`` (mirrors :meth:`_quantize_mamba_jobs`). CPU
        device, a single payload, or a dtype-singleton group all fall back
        to one native call per payload -- unchanged behavior, since there
        is no transfer to save in those cases.
        """
        if device != "xpu" or len(payloads) <= 1:
            return [
                _KVWeaveCodec.dequantize_mamba_substate_4bit(p, device=device) for p in payloads
            ]
        parsed_list = [_KVWeaveCodec._read_mamba_substate_payload(p) for p in payloads]
        results: list[torch.Tensor] = [None] * len(payloads)  # type: ignore[list-item]
        groups: dict[torch.dtype, list[int]] = {}
        for idx, parsed in enumerate(parsed_list):
            groups.setdefault(parsed["output_dtype"], []).append(idx)
        native = _resolve_native("xpu")
        for indices in groups.values():
            if len(indices) == 1:
                idx = indices[0]
                p = parsed_list[idx]
                restored = native.kvweave_dequantize_chunk_state(
                    p["q_data"], p["scales"], p["num_layers"], p["num_blocks"], p["h_merged"],
                    qbit=p["qbit"], blocks_num=p["blocks_num"],
                    block_size=1, head_num=p["head_num"], head_dim=p["head_dim"],
                    rh=p["rh"], asym=p["asym"], scaling_method=p["scaling_method"],
                    output_dtype=p["output_dtype"], signs=p["signs"], perm=p["perm"],
                )
                results[idx] = restored.reshape(p["shape"])
                continue
            batch_results = _KVWeaveCodec._dequantize_mamba_substates_batch_xpu(
                [parsed_list[i] for i in indices]
            )
            for idx, r in zip(indices, batch_results):
                results[idx] = r
        return results

    @staticmethod
    def _decode_mamba_substates_xpu_batched(
        conv_payload: bytes,
        ssm_payload: bytes,
        conv_layout: MambaSubStateWireLayout,
        ssm_layout: MambaSubStateWireLayout,
        layers: int,
        blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """XPU fast path for :meth:`decode_chunk`'s Mamba branch: collects
        every substate payload that needs a native dequantize call (conv's
        query/key/value when quantized, ssm when quantized) and dequantizes
        them together via :meth:`_dequantize_mamba_payloads`'s batching,
        instead of each substate independently round-tripping to the
        device. Falls back to :meth:`_decode_conv_substate`/
        :meth:`_decode_mamba_substate` for a substate whose flag byte says
        it was never natively quantized (flag 0/2) -- those never touch the
        device either way, so there is nothing to batch there.
        """
        conv_flag = conv_payload[0]
        ssm_flag = ssm_payload[0]

        native_payloads: list[bytes] = []
        conv_indices: tuple[int, int, int] | None = None
        ssm_index: int | None = None
        if conv_flag == 1:
            qkv = _KVWeaveCodec.unpack_conv_qkv_payloads(conv_payload[1:])
            conv_indices = (0, 1, 2)
            native_payloads.extend(qkv)
        if ssm_flag == 1:
            ssm_index = len(native_payloads)
            native_payloads.append(ssm_payload[1:])

        dequantized = (
            _KVWeaveCodec._dequantize_mamba_payloads(native_payloads, device="xpu")
            if native_payloads else []
        )

        if conv_indices is not None:
            conv = torch.cat([dequantized[i] for i in conv_indices], dim=-1)
        else:
            conv = _KVWeaveCodec._decode_conv_substate(
                conv_payload, conv_layout, layers, blocks, device="xpu"
            )

        if ssm_index is not None:
            ssm = dequantized[ssm_index]
        else:
            ssm = _KVWeaveCodec._decode_mamba_substate(
                ssm_payload, ssm_layout, layers, blocks, device="xpu"
            )

        return conv, ssm

    @staticmethod
    def quantize_mamba_substate_4bit(
        tensor: torch.Tensor,
        *,
        substate: str,
        scaling_method: str = "per_tensor",
        rh: bool = False,
        asym: bool = False,
        qbit: int = KVWeaveCodecConfig.MAMBA_QBIT,
        device: str = "cpu",
    ) -> bytes:
        """Quantize a Mamba sub-state using the verified state layout contract."""
        return _KVWeaveCodec._quantize_mamba_substate_payload(
            tensor, substate=substate, scaling_method=scaling_method,
            rh=rh, asym=asym, qbit=qbit, device=device,
        )

    @staticmethod
    def _estimate_substate_quantized_size(
        substate_name: str,
        shape: tuple[int, ...],
        dtype_str: str,
        scaling_method: str,
        qbit: int,
        quant_enabled: bool = True,
    ) -> int:
        """Estimate one sub-state tensor's upper-bound serialized byte size.

        Shared by :meth:`estimate_mamba_serialized_size`'s ssm branch and its
        per-sub-tensor conv q/k/v branch (see :meth:`_split_conv_qkv`) --
        both need the same native-payload-size formula, just applied to a
        differently-shaped tensor.

        Args:
            substate_name: ``"conv"`` or ``"ssm"`` (selects the native
                ``mamba_layout`` grouping rule).
            shape: The full ``(layers, blocks, *tail)`` shape being sized.
            dtype_str: The tensor's real wire dtype, as ``str(torch.dtype)``.
            scaling_method: This sub-state's configured scaling method.
            qbit: This sub-state's configured quantization bit width.
            quant_enabled: This sub-state's resolved
                ``LMCACHE_MP_KVWEAVE_CONV_QUANT_ENABLED``/``SSM_QUANT_ENABLED``
                value (read once via ``MambaCodecOptions.from_env()`` at
                registration and fixed for the group's lifetime -- it is not
                re-read per store call, so ``encode_chunk`` can never switch
                branches after registration). When ``True`` (the default),
                only the quantized-payload upper bound is sized, since
                ``encode_chunk`` will never fall back to raw bytes for this
                sub-state. When ``False``, only the raw byte size is sized.

        Returns:
            The quantized-payload upper bound, or the raw (unquantized)
            byte size when ``quant_enabled`` is ``False`` -- whichever
            branch ``encode_chunk`` will actually take for this sub-state.
        """
        layers, blocks = shape[0], shape[1]
        elements = layers * blocks * max(
            int(np.prod(shape[2:])) if len(shape) > 2 else 1, 1
        )
        raw_size = elements * KVWeaveCodecConfig.mamba_dtype(dtype_str).itemsize
        if not quant_enabled:
            return raw_size
        native_blocks, _, native_head_dim, _ = _KVWeaveCodec._mamba_layout(
            substate_name, shape, scaling_method
        )
        if scaling_method == "per_tensor":
            native_chunks = 1
        elif scaling_method == "per_channel":
            native_chunks = native_head_dim
        else:
            native_chunks = native_blocks
        scale_blob = 4 + layers * (4 + native_chunks * 12)
        quantized_size = (
            elements * 2
            if qbit == 16
            else 10 + 4 * len(shape) + scale_blob + KVWeaveCodecConfig.quantized_bytes(
                elements, qbit
            )
        )
        return quantized_size

    @staticmethod
    def estimate_mamba_serialized_size(
        raw_layout: MemoryLayoutDesc,
        mamba_layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout],
        block_size: int,
        conv_qkv_split: ConvQKVSplit,
        scaling_methods: tuple[str, str] = ("per_channel", "per_channel"),
        qbits: tuple[int, int] = (4, 8),
        quant_enabled: tuple[bool, bool] = (True, True),
    ) -> int:
        """Estimate the upper bound needed for one packed conv+ssm Mamba chunk.

        Unlike :meth:`estimate_serialized_size` (attention K/V pairs),
        this does not assume a per-token-per-head element layout: it derives
        the element count for each sub-state directly from its own
        ``MambaSubStateWireLayout.shape`` and the group's ``layers``/``tokens``
        (recovered from ``raw_layout``), scaled by the native 4-bit packing
        rate. The result sizes whichever branch (quantized or raw)
        ``encode_chunk`` will actually take per ``quant_enabled`` -- these
        flags are resolved once at registration and fixed for the group's
        lifetime, so there is no need to reserve for both branches at once.

        Args:
            conv_qkv_split: This model's ``key_dim``/``value_dim`` boundary.
                conv_state's last dimension is sized as three
                independently-quantized query/key/value payloads (see
                :meth:`_split_conv_qkv`/:meth:`pack_conv_qkv_payloads`) plus
                their 3-way length-prefix framing, matching
                ``encode_chunk``'s actual conv encoding -- ``encode_chunk``
                always splits conv when quantizing, so this must always
                match the split it will actually use.
            quant_enabled: ``(conv_quant_enabled, ssm_quant_enabled)`` --
                see :meth:`_estimate_substate_quantized_size`.
        """
        shape = tuple(int(dim) for dim in raw_layout.shapes[0])
        if len(shape) == 4:
            _, layers, tokens, _ = shape
        elif len(shape) == 3:
            layers, tokens, _ = shape
        else:
            raise ValueError(f"expected Mamba chunk shape [2,L,T,H] or [L,T,H], got {shape}")
        if block_size <= 0 or tokens % block_size:
            raise ValueError(
                f"chunk_tokens ({tokens}) is not a multiple of block_size ({block_size})"
            )
        blocks = tokens // block_size
        total = 8  # pack_mamba_payloads() conv + ssm length-prefix framing
        conv_layout, ssm_layout = mamba_layout
        conv_scaling, ssm_scaling = scaling_methods
        conv_qbit, ssm_qbit = qbits
        conv_quant_enabled, ssm_quant_enabled = quant_enabled

        conv_dim = conv_layout.shape[-1]
        expected_conv_dim = conv_qkv_split.key_dim * 2 + conv_qkv_split.value_dim
        if conv_dim != expected_conv_dim:
            raise ValueError(
                f"conv_state last dim ({conv_dim}) does not match "
                f"key_dim*2 + value_dim ({expected_conv_dim})"
            )
        total += 12  # pack_conv_qkv_payloads() 3-way length-prefix framing
        for sub_dim in (
            conv_qkv_split.key_dim,
            conv_qkv_split.key_dim,
            conv_qkv_split.value_dim,
        ):
            sub_shape = (layers, blocks, *conv_layout.shape[:-1], sub_dim)
            total += 1 + _KVWeaveCodec._estimate_substate_quantized_size(
                "conv", sub_shape, conv_layout.dtype_str, conv_scaling, conv_qbit,
                quant_enabled=conv_quant_enabled,
            )

        ssm_shape = (layers, blocks, *ssm_layout.shape)
        total += 1 + _KVWeaveCodec._estimate_substate_quantized_size(
            "ssm", ssm_shape, ssm_layout.dtype_str, ssm_scaling, ssm_qbit,
            quant_enabled=ssm_quant_enabled,
        )
        return total

    @staticmethod
    def pack_mamba_payloads(conv: bytes, ssm: bytes) -> bytes:
        """Frame conv and ssm payloads into one stored blob.

        Both sub-payloads get an explicit length prefix -- relying on ssm
        being "whatever bytes remain" breaks once the blob is zero-padded to
        a fixed slot size by the transport layer (the storage layout reserves
        ``estimate_mamba_serialized_size()``'s conservative upper bound, not
        the exact encoded length): the padding zeros would then be read back
        as trailing ssm quantized data, corrupting the recurrent state.
        """
        return (
            struct.pack(">I", len(conv))
            + conv
            + struct.pack(">I", len(ssm))
            + ssm
        )

    @staticmethod
    def unpack_mamba_payloads(blob: bytes) -> tuple[bytes, bytes]:
        """Split a framed Mamba blob back into conv and ssm payloads."""
        if len(blob) < 4:
            raise ValueError("Mamba payload bundle is truncated")
        conv_size = struct.unpack(">I", blob[:4])[0]
        if conv_size > len(blob) - 4:
            raise ValueError("Mamba payload bundle is truncated")
        offset = 4 + conv_size
        conv = blob[4:offset]
        if len(blob) - offset < 4:
            raise ValueError("Mamba payload bundle is truncated")
        ssm_size = struct.unpack(">I", blob[offset : offset + 4])[0]
        offset += 4
        if ssm_size > len(blob) - offset:
            raise ValueError("Mamba payload bundle is truncated")
        return conv, blob[offset : offset + ssm_size]

    @staticmethod
    def _split_conv_qkv(
        conv: torch.Tensor, split: ConvQKVSplit
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice a fused conv_state tensor into its query/key/value sub-tensors.

        ``conv``'s last dimension is ``conv_dim = key_dim*2 + value_dim``
        (query and key share ``key_dim``, see ``mamba_conv_ssm_layout_params.md``
        §1's ``torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)``).
        Slicing (not splitting into new storage) keeps this a view until
        ``quantize_mamba_substate_4bit`` makes each sub-tensor contiguous.

        Args:
            conv: Fused conv_state, shape ``(layers, blocks, kernel_hist, conv_dim)``.
            split: This model's ``key_dim``/``value_dim`` boundary.

        Returns:
            ``(query, key, value)`` views, each ``key_dim`` (query/key) or
            ``value_dim`` (value) wide on the last dimension.

        Raises:
            ValueError: If ``conv``'s last dimension does not match
                ``key_dim*2 + value_dim``.
        """
        key_dim, value_dim = split.key_dim, split.value_dim
        conv_dim = conv.shape[-1]
        if conv_dim != key_dim * 2 + value_dim:
            raise ValueError(
                f"conv_state last dim ({conv_dim}) does not match "
                f"key_dim*2 + value_dim ({key_dim * 2 + value_dim})"
            )
        return (
            conv[..., :key_dim],
            conv[..., key_dim : 2 * key_dim],
            conv[..., 2 * key_dim :],
        )

    @staticmethod
    def pack_conv_qkv_payloads(query: bytes, key: bytes, value: bytes) -> bytes:
        """Frame conv query/key/value payloads into one stored blob."""
        return (
            struct.pack(">I", len(query))
            + query
            + struct.pack(">I", len(key))
            + key
            + struct.pack(">I", len(value))
            + value
        )

    @staticmethod
    def unpack_conv_qkv_payloads(blob: bytes) -> tuple[bytes, bytes, bytes]:
        """Split a framed conv Q/K/V blob back into query, key, and value payloads."""
        if len(blob) < 4:
            raise ValueError("Conv QKV payload bundle is truncated")

        query_size = struct.unpack(">I", blob[:4])[0]
        if query_size > len(blob) - 4:
            raise ValueError("Conv QKV payload bundle is truncated")
        offset = 4 + query_size
        query = blob[4:offset]

        if len(blob) - offset < 4:
            raise ValueError("Conv QKV payload bundle is truncated")
        key_size = struct.unpack(">I", blob[offset : offset + 4])[0]
        offset += 4
        if key_size > len(blob) - offset:
            raise ValueError("Conv QKV payload bundle is truncated")
        key = blob[offset : offset + key_size]
        offset += key_size

        if len(blob) - offset < 4:
            raise ValueError("Conv QKV payload bundle is truncated")
        value_size = struct.unpack(">I", blob[offset : offset + 4])[0]
        offset += 4
        if value_size > len(blob) - offset:
            raise ValueError("Conv QKV payload bundle is truncated")
        value = blob[offset : offset + value_size]
        return query, key, value

    _VALID_CACHE_CATEGORIES = frozenset({"attention", "mamba", "unknown"})

    @staticmethod
    def _validate_cache_category_dispatch(
        cache_category: str,
        mamba_layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout] | None,
        attention_plane_layout: AttentionPlaneLayout | None = None,
    ) -> None:
        """Reject any category/layout combination that would mis-dispatch.

        ``cache_category`` must be exactly one of ``"attention"``,
        ``"mamba"``, or ``"unknown"`` (see ``EngineGroupInfo.cache_category``).
        Only ``"mamba"`` may carry a non-``None`` ``mamba_layout``. Only
        ``"attention"`` may carry a non-``None`` ``attention_plane_layout``.
        ``"unknown"`` is rejected unconditionally: it exists so that a
        caller who failed to resolve a group's real category fails loudly
        here instead of silently falling into the attention path (the
        historical incident this guards against: a Mamba group's opaque
        page-view chunk shape happens to be compatible with the attention
        codec's fused K/V shape, so a shape-based dispatch would silently
        corrupt the recurrent state instead of raising -- see
        MIGRATION_PLAN.md R1/R6).

        Args:
            cache_category: The group's declared category.
            mamba_layout: The group's real conv/ssm sub-state layout, or
                ``None`` for a non-Mamba group.
            attention_plane_layout: The group's classified attention K/V
                plane layout, or ``None`` for a non-attention group.

        Raises:
            ValueError: If ``cache_category`` is not one of the three
                valid values, if ``cache_category != "mamba"`` but
                ``mamba_layout`` is provided, if ``cache_category ==
                "mamba"`` but ``mamba_layout`` is missing, if
                ``cache_category == "unknown"``, or if
                ``attention_plane_layout`` is provided but
                ``cache_category != "attention"``.
        """
        if cache_category not in _KVWeaveCodec._VALID_CACHE_CATEGORIES:
            raise ValueError(
                f"Unknown cache_category {cache_category!r}; expected one "
                f"of {sorted(_KVWeaveCodec._VALID_CACHE_CATEGORIES)}"
            )
        if cache_category == "unknown":
            raise ValueError(
                "cache_category='unknown' must never be passed to "
                "encode_chunk/decode_chunk -- quantization dispatch "
                "requires an explicitly classified group (see "
                "EngineGroupInfo.cache_category); resolve the category "
                "before calling"
            )
        if cache_category == "mamba" and mamba_layout is None:
            raise ValueError(
                "cache_category='mamba' requires a non-None mamba_layout "
                "(conv, ssm); the caller must resolve "
                "EngineGroupInfo.mamba_real_layout before calling "
                "encode_chunk/decode_chunk"
            )
        if cache_category != "mamba" and mamba_layout is not None:
            raise ValueError(
                f"mamba_layout was provided but cache_category="
                f"{cache_category!r} is not 'mamba'; this would misroute "
                "a non-Mamba chunk into the Mamba split/merge codec"
            )
        if cache_category != "attention" and attention_plane_layout is not None:
            raise ValueError(
                f"attention_plane_layout was provided but cache_category="
                f"{cache_category!r} is not 'attention'; this would "
                "misroute a non-attention chunk into the fused-K/V codec"
            )

    def encode_chunk(
        self,
        cache_category: str,
        mamba_layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout] | None,
        tokens_per_block: int,
        mamba_options: MambaCodecOptions | None,
        raw_chunk: torch.Tensor,
        attention_plane_layout: AttentionPlaneLayout | None = None,
        mamba_split: MambaChunkSplit | None = None,
        fused_head_split: torch.Tensor | None = None,
    ) -> bytes:
        """Encode one gathered raw chunk into its wire-quantized byte payload.

        Dispatches on ``cache_category``: Mamba groups are split into their
        real ``conv``/``ssm`` sub-states and quantized independently (Phase
        4's dedicated Mamba codec); attention groups are quantized via
        ``self.serialize_fused_tensor`` when ``attention_plane_layout`` is
        ``AttentionPlaneLayout.FUSED_KV`` (K/V packed into one tensor, no
        leading K/V axis), or ``self.serialize_tensor`` (Phase 3's codec)
        otherwise. Applying the attention codec to a Mamba group's opaque
        page-view chunk would silently corrupt its recurrent state -- see
        Phase 6 in MIGRATION_PLAN.md.

        Raises:
            ValueError: If ``cache_category``/``mamba_layout``/
                ``attention_plane_layout`` do not form a valid, unambiguous
                dispatch -- see :meth:`_validate_cache_category_dispatch`.
        """
        self._validate_cache_category_dispatch(
            cache_category, mamba_layout, attention_plane_layout
        )
        if mamba_layout is not None:
            if mamba_options is None:
                raise RuntimeError("Mamba codec options are not initialized")
            split = mamba_split or self.split_mamba_chunk(
                raw_chunk, mamba_layout, tokens_per_block
            )
            conv_enabled = getattr(mamba_options, "conv_quant_enabled", True)
            ssm_enabled = getattr(mamba_options, "ssm_quant_enabled", True)
            ssm_raw_fallback = ssm_enabled and mamba_options.ssm_qbit == 16

            # Collect every substate that needs a native quantize call
            # (conv's query/key/value when enabled, ssm when enabled and not
            # the raw-fp16 fallback) and quantize them together: on
            # device="xpu" this collapses what would otherwise be up to 4
            # independent host<->XPU round trips (one per substate) into as
            # few as 1, since they usually share the KV cache's tensor
            # dtype. See _quantize_mamba_jobs.
            jobs = []
            if conv_enabled:
                query, key, value = self._split_conv_qkv(
                    split.conv, mamba_options.conv_qkv_split
                )
                for sub in (query, key, value):
                    jobs.append(dict(
                        tensor=sub, substate="conv",
                        scaling_method=mamba_options.conv_scaling_method,
                        rh=mamba_options.conv_rh, asym=mamba_options.asym,
                        qbit=mamba_options.conv_qbit,
                    ))
            if ssm_enabled and not ssm_raw_fallback:
                jobs.append(dict(
                    tensor=split.ssm, substate="ssm",
                    scaling_method=mamba_options.ssm_scaling_method,
                    rh=mamba_options.ssm_rh, asym=mamba_options.asym,
                    qbit=mamba_options.ssm_qbit,
                ))

            payloads = iter(self._quantize_mamba_jobs(jobs, self.device))

            if conv_enabled:
                conv_payload = b"\x01" + self.pack_conv_qkv_payloads(
                    *(next(payloads) for _ in range(3))
                )
            else:
                conv_payload = b"\x00" + self._tensor_bytes(split.conv)

            if ssm_enabled:
                if ssm_raw_fallback:
                    ssm_payload = b"\x02" + self._tensor_bytes(
                        split.ssm.to(dtype=torch.float16)
                    )
                else:
                    ssm_payload = b"\x01" + next(payloads)
            else:
                ssm_payload = b"\x00" + self._tensor_bytes(split.ssm)
            return self.pack_mamba_payloads(conv_payload, ssm_payload)
        if attention_plane_layout == AttentionPlaneLayout.FUSED_KV:
            return self.serialize_fused_tensor(raw_chunk, head_split=fused_head_split)
        return self.serialize_tensor(raw_chunk)

    def encode_chunk_into(
        self,
        cache_category: str,
        mamba_layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout] | None,
        tokens_per_block: int,
        mamba_options: MambaCodecOptions | None,
        raw_chunk: torch.Tensor,
        destination: torch.Tensor,
        attention_plane_layout: AttentionPlaneLayout | None = None,
        mamba_split: MambaChunkSplit | None = None,
        fused_head_split: torch.Tensor | None = None,
    ) -> int:
        """Encode a chunk and copy its payload directly into ``destination``.

        ``destination`` is normally the durable quantized SHM slot. The
        native serializer currently returns Python ``bytes``, so this cannot
        eliminate that native result allocation; it does avoid constructing a
        second ``bytearray``/``uint8`` staging tensor before the SHM copy.
        Returns the number of payload bytes written.
        """
        if destination.dtype != torch.uint8 or not destination.is_contiguous():
            raise ValueError("destination must be a contiguous torch.uint8 tensor")
        payload = self.encode_chunk(
            cache_category,
            mamba_layout,
            tokens_per_block,
            mamba_options,
            raw_chunk,
            attention_plane_layout,
            mamba_split,
            fused_head_split,
        )
        if len(payload) > destination.numel():
            raise ValueError(
                f"encoded payload ({len(payload)} bytes) exceeds destination "
                f"capacity ({destination.numel()} bytes)"
            )
        destination.view(-1)[: len(payload)].copy_(
            torch.frombuffer(memoryview(payload), dtype=torch.uint8)
        )
        return len(payload)

    def decode_chunk(
        self,
        cache_category: str,
        mamba_layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout] | None,
        tokens_per_block: int,
        raw_shape: torch.Size,
        raw_dtype: torch.dtype,
        chunk: torch.Tensor,
        attention_plane_layout: AttentionPlaneLayout | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode one retrieved wire-quantized byte chunk back to its raw shape.

        Mirrors :meth:`encode_chunk`'s dispatch. When ``out`` is supplied,
        the decoded raw chunk is written there; otherwise a fresh tensor is
        allocated.

        Raises:
            ValueError: If ``cache_category``/``mamba_layout``/
                ``attention_plane_layout`` do not form a valid, unambiguous
                dispatch -- see :meth:`_validate_cache_category_dispatch`.
        """
        self._validate_cache_category_dispatch(
            cache_category, mamba_layout, attention_plane_layout
        )
        if mamba_layout is not None:
            conv_layout, ssm_layout = mamba_layout
            conv_payload, ssm_payload = self.unpack_mamba_payloads(
                self._tensor_bytes(chunk)
            )
            layers = raw_shape[1] if len(raw_shape) == 4 else raw_shape[0]
            tokens = raw_shape[-2]
            blocks = max(tokens // tokens_per_block, 1)
            if self.device == "xpu":
                conv, ssm = self._decode_mamba_substates_xpu_batched(
                    conv_payload, ssm_payload, conv_layout, ssm_layout, layers, blocks
                )
            else:
                conv = self._decode_conv_substate(conv_payload, conv_layout, layers, blocks, device=self.device)
                ssm = self._decode_mamba_substate(ssm_payload, ssm_layout, layers, blocks, device=self.device)
            hidden_dim = raw_shape[-1]
            merged = self.merge_mamba_chunk(
                MambaChunkSplit(conv, ssm), mamba_layout, tokens_per_block,
                hidden_dim, raw_shape=raw_shape, raw_dtype=raw_dtype, out=out,
            )
            # ``merged`` is an opaque page view. Its bytes already contain
            # each sub-state in its own wire dtype; converting the page tensor
            # numerically would corrupt fp32 SSM bytes when the page dtype is
            # fp16 (the normal Qwen3.5 layout).
            return merged
        destination = out if out is not None else torch.empty(raw_shape, dtype=raw_dtype)
        if destination.shape != raw_shape or destination.dtype != raw_dtype:
            raise ValueError("out must match raw_shape and raw_dtype")
        if attention_plane_layout == AttentionPlaneLayout.FUSED_KV:
            self.deserialize_fused_tensor(chunk, destination)
        else:
            self.deserialize_tensor(chunk, destination)
        return destination


KVWeaveCodec = _KVWeaveCodec
