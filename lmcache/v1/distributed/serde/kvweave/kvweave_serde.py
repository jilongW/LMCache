"""KVWeave 4-bit codec used by the LMCache L1 quantization path."""

from __future__ import annotations

from dataclasses import dataclass
import io
import struct
import threading
from typing import Optional

import torch

from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.distributed.serde.kvweave.kvweave_config import KVWeaveCodecConfig
from lmcache.v1.multiprocess.group_view import MambaSubStateWireLayout

try:
    from kvweave import kvweave_quant
except ImportError:  # pragma: no cover
    kvweave_quant = None

logger = init_logger(__name__)

_MAGIC_RAW = b"KVW0"
_MAGIC_QUANT = b"KVW3"
_DEFAULT_BLOCK_SIZE = 64
_SCALING_TO_INT = {"per_tensor": 0, "per_token": 1, "per_channel": 2}
_INT_TO_SCALING = {value: key for key, value in _SCALING_TO_INT.items()}
_FMT_TO_INT = {
    MemoryFormat.KV_2LTD: 0,
    MemoryFormat.KV_T2D: 1,
    MemoryFormat.KV_2TD: 2,
    MemoryFormat.KV_MLA_FMT: 3,
}
_INT_TO_FMT = {value: key for key, value in _FMT_TO_INT.items()}
_DTYPE_TO_INT = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}
_INT_TO_DTYPE = {value: key for key, value in _DTYPE_TO_INT.items()}
_SCALE_COUNTER = 0
_SCALE_LOCK = threading.Lock()


def _next_scale_ids(count: int) -> list[int]:
    global _SCALE_COUNTER
    with _SCALE_LOCK:
        result = []
        for _ in range(count):
            _SCALE_COUNTER = (_SCALE_COUNTER + 1) & 0xFFFFFFFF
            result.append(_SCALE_COUNTER)
        return result


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _quantized_bytes(elements: int, qbit: int) -> int:
    return (elements + 1) // 2 if qbit == 4 else elements if qbit <= 8 else elements * 2


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return bytes(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


def _copy_bytes_to_tensor(payload: bytes, dst: torch.Tensor) -> None:
    target = dst.contiguous() if not dst.is_contiguous() else dst
    target = target.view(torch.uint8).reshape(-1)
    if len(payload) > target.numel():
        raise ValueError("KVWeave payload exceeds destination capacity")
    target[: len(payload)].copy_(torch.frombuffer(payload, dtype=torch.uint8))


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
    """Self-describing wrapper around the native KVWeave quantizer."""

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
            num_threads=int(kwargs.get("num_threads", 1)),
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
        self.log = config.log
        self.num_threads = config.num_threads
        self.block_size = int(kwargs.get("block_size", _DEFAULT_BLOCK_SIZE))
        self.num_kv_heads = int(kwargs.get("num_kv_heads", kwargs.get("head_num", 1)))
        self.head_dim = int(kwargs.get("head_dim", 0))
        self._quant_mod = None

    def _native(self):
        if self._quant_mod is None:
            try:
                from kvweave import kvweave_quant
            except ImportError as exc:
                raise RuntimeError("KVWeave native quantization extension is unavailable") from exc
            self._quant_mod = kvweave_quant
        return self._quant_mod

    def estimate_serialized_size(
        self,
        layout_desc: MemoryLayoutDesc,
        scaling_method: Optional[str] = None,
    ) -> int:
        return sum(
            self._estimate_shape(tuple(int(dim) for dim in shape), dtype, scaling_method)
            for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
        )

    def _estimate_shape(
        self, shape: tuple[int, ...], dtype: torch.dtype, scaling_method: str | None
    ) -> int:
        if len(shape) == 3:
            tokens, kv_size, hidden = shape
            layers = 1
        elif len(shape) == 4:
            kv_size, layers, tokens, hidden = shape
        else:
            return 64 + _dtype_size(dtype) * int(torch.tensor(shape).prod().item())
        if kv_size != 2:
            raise ValueError(f"KVWeave expects K/V dimension 2, got {shape}")
        method = scaling_method or self.scaling_method
        elements = int(layers) * int(tokens) * int(hidden)
        if not self.quantize:
            return 32 + 2 * elements * _dtype_size(dtype)
        scales = self._scale_count(tokens, hidden, layers, method)
        header = 32 + 4 * len(shape)
        return int((header + 8 + 2 * (4 + scales * 12) + 2 * _quantized_bytes(elements, self.qbit)) * 1.10) + 4096

    def serialize_tensor(self, tensor: torch.Tensor, scaling_method: str | None = None) -> bytes:
        cpu = tensor.detach().to("cpu").contiguous()
        shape = self._normalize(cpu)
        method = scaling_method or self.scaling_method
        if not self.quantize:
            return self._raw_payload(cpu, shape)
        if method == "per_tensor":
            rh, asym, precond = False, False, False
        else:
            rh, asym, precond = self.rh, self.asym, self.precond
        header = self._quant_header(shape, cpu.dtype, method, rh, asym, precond)
        ids = _next_scale_ids(2)
        native = self._native()
        payload = native.kvweave_serialize_chunk(
            shape.tensor4d,
            header,
            ids[0], ids[1],
            qbit=self.qbit,
            blocks_num=max(1, shape.chunk_tokens // self.block_size),
            block_size=self.block_size,
            head_num=self._head_num(shape.hidden_dim),
            head_dim=self._head_dim(shape.hidden_dim),
            num_layers=shape.num_layers,
            rh=rh,
            asym=asym,
            scaling_method=method,
            num_threads=self.num_threads,
        )
        return bytes(payload)

    def deserialize_tensor(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        parsed = self._parse(_tensor_bytes(src))
        if parsed["raw"]:
            data = torch.frombuffer(bytearray(parsed["data"]), dtype=parsed["dtype"])
            dst.copy_(data.reshape(dst.shape).to(dtype=dst.dtype, device=dst.device))
            return
        shape = parsed["shape4d"]
        q = torch.frombuffer(bytearray(parsed["q_data"]), dtype=torch.int8)
        native = self._native()
        if hasattr(native, "kvweave_dequantize_chunk_into_4d") and dst.dim() == 4:
            native.kvweave_dequantize_chunk_into_4d(
                q, parsed["k_scales"], parsed["v_scales"], dst,
                shape[1], shape[2], shape[3],
                qbit=parsed["qbit"], blocks_num=parsed["blocks_num"],
                block_size=self.block_size, head_num=parsed["head_num"],
                head_dim=parsed["head_dim"], rh=parsed["rh"], asym=parsed["asym"],
                scaling_method=parsed["scaling"], output_dtype=dst.dtype,
                num_threads=self.num_threads,
            )
            return
        restored = native.kvweave_dequantize_chunk(
            q, parsed["k_scales"], parsed["v_scales"],
            shape[1], shape[2], shape[3], qbit=parsed["qbit"],
            blocks_num=parsed["blocks_num"], block_size=self.block_size,
            head_num=parsed["head_num"], head_dim=parsed["head_dim"],
            rh=parsed["rh"], asym=parsed["asym"], scaling_method=parsed["scaling"],
            output_dtype=dst.dtype, num_threads=self.num_threads,
        )
        if dst.dim() == 3:
            restored = restored.squeeze(1).permute(1, 0, 2)
        dst.copy_(restored.to(dtype=dst.dtype, device=dst.device))

    def _normalize(self, tensor: torch.Tensor) -> _KVShape:
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
        header = struct.pack(">4sBBB", _MAGIC_RAW, shape.original_ndim, 1, _DTYPE_TO_INT.get(tensor.dtype, 0))
        header += struct.pack(">" + "i" * shape.original_ndim, *shape.header_shape)
        return header + _tensor_bytes(tensor)

    def _quant_header(self, shape: _KVShape, dtype: torch.dtype, method: str, rh: bool, asym: bool, precond: bool) -> bytes:
        return struct.pack(
            ">4sBBBBBBBB" + "i" * shape.original_ndim,
            _MAGIC_QUANT, self.qbit, int(rh), int(asym), int(precond),
            shape.original_ndim, 1, _SCALING_TO_INT.get(method, 1),
            _DTYPE_TO_INT.get(dtype, 0), *shape.header_shape,
        )

    def _parse(self, raw: bytes) -> dict[str, object]:
        stream = io.BytesIO(raw)
        magic = stream.read(4)
        if magic == _MAGIC_RAW:
            ndim, _, dtype_code = struct.unpack(">BBB", stream.read(3))
            shape = tuple(struct.unpack(">" + "i" * ndim, stream.read(4 * ndim)))
            dtype = _INT_TO_DTYPE.get(dtype_code, torch.float16)
            return {"raw": True, "shape": shape, "dtype": dtype, "data": stream.read()}
        if magic != _MAGIC_QUANT:
            raise ValueError(f"invalid KVWeave payload magic: {magic!r}")
        qbit, rh, asym, _, ndim, _, scaling_code, dtype_code = struct.unpack(">BBBBBBBB", stream.read(8))
        shape = tuple(struct.unpack(">" + "i" * ndim, stream.read(4 * ndim)))
        shape4d = (shape[1], 1, shape[0], shape[2]) if ndim == 3 else shape
        scales = []
        for _ in range(2):
            size = struct.unpack(">I", stream.read(4))[0]
            scales.append(stream.read(size))
        elements = shape4d[1] * shape4d[2] * shape4d[3]
        qsize = _quantized_bytes(elements, qbit)
        return {
            "raw": False, "shape4d": shape4d, "qbit": qbit, "rh": bool(rh), "asym": bool(asym),
            "scaling": _INT_TO_SCALING.get(scaling_code, self.scaling_method),
            "dtype": _INT_TO_DTYPE.get(dtype_code, torch.float16),
            "k_scales": scales[0], "v_scales": scales[1], "q_data": stream.read(2 * qsize),
            "blocks_num": max(1, shape4d[2] // self.block_size),
            "head_num": self._head_num(shape4d[3]), "head_dim": self._head_dim(shape4d[3]),
        }

    def _head_num(self, hidden: int) -> int:
        return self.num_kv_heads if self.num_kv_heads > 1 and hidden % self.num_kv_heads == 0 else 1

    def _head_dim(self, hidden: int) -> int:
        return self.head_dim or hidden // self._head_num(hidden)

    def _scale_count(self, tokens: int, hidden: int, layers: int, method: str) -> int:
        base = tokens if method == "per_token" else self._head_dim(hidden) if method == "per_channel" else 1
        return max(1, base * layers if layers > 1 else base)

    @staticmethod
    def _mamba_layout(
        substate: str, shape: tuple[int, ...], method: str
    ) -> tuple[int, int, int, int]:
        return KVWeaveCodecConfig.mamba_layout(substate, shape, method)

    @staticmethod
    def _mamba_precond_pair(size: int) -> tuple[torch.Tensor, torch.Tensor]:
        return KVWeaveCodecConfig().mamba_precond_tensors(size)

    @staticmethod
    def split_mamba_chunk(
        raw: torch.Tensor,
        layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout],
        block_size: int,
    ) -> MambaChunkSplit:
        """Recover real conv/ssm bytes from the opaque synthetic page view."""
        if raw.dim() != 4 or int(raw.shape[0]) != 2:
            raise ValueError(
                f"expected Mamba chunk shape [2,L,T,H], got {tuple(raw.shape)}"
            )
        conv_layout, ssm_layout = layout
        _, layers, tokens, hidden = map(int, raw.shape)
        if block_size <= 0 or tokens % block_size:
            raise ValueError(
                f"chunk_tokens ({tokens}) is not a multiple of "
                f"block_size ({block_size})"
            )
        blocks = tokens // block_size
        page_bytes = 2 * block_size * hidden * raw.element_size()
        pages = (
            raw.permute(1, 0, 2, 3)
            .reshape(layers, 2, blocks, block_size, hidden)
            .permute(0, 2, 1, 3, 4)
            .reshape(layers, blocks, page_bytes // raw.element_size())
            .contiguous()
            .view(torch.uint8)
            .reshape(layers, blocks, page_bytes)
        )

        def read(desc: MambaSubStateWireLayout) -> torch.Tensor:
            end = desc.byte_offset + desc.byte_length
            if desc.byte_offset < 0 or end > page_bytes:
                raise ValueError("Mamba sub-state byte layout exceeds page size")
            dtype = KVWeaveCodecConfig.mamba_dtype(desc.dtype_str)
            return (
                pages[:, :, desc.byte_offset:end]
                .contiguous()
                .view(dtype)
                .reshape(layers, blocks, *desc.shape)
            )

        return MambaChunkSplit(read(conv_layout), read(ssm_layout))

    @staticmethod
    def merge_mamba_chunk(
        split: MambaChunkSplit,
        layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout],
        block_size: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        """Rebuild the opaque page view, zero-filling unused padding bytes."""
        conv_layout, ssm_layout = layout
        layers, blocks = map(int, split.conv.shape[:2])
        if split.ssm.shape[:2] != (layers, blocks):
            raise ValueError(
                "conv and ssm chunks have different layer/block dimensions"
            )
        page_bytes = 2 * block_size * hidden_dim * split.conv.element_size()
        pages = torch.zeros(layers, blocks, page_bytes, dtype=torch.uint8)
        for tensor, desc in (
            (split.conv, conv_layout),
            (split.ssm, ssm_layout),
        ):
            raw = tensor.contiguous().view(torch.uint8).reshape(layers, blocks, -1)
            end = desc.byte_offset + desc.byte_length
            if raw.shape[-1] != desc.byte_length or end > page_bytes:
                raise ValueError("Mamba sub-state tensor does not match byte layout")
            pages[:, :, desc.byte_offset:end] = raw
        return (
            pages.view(split.conv.dtype)
            .reshape(layers, blocks, 2, block_size, hidden_dim)
            .permute(2, 0, 1, 3, 4)
            .reshape(2, layers, blocks * block_size, hidden_dim)
            .contiguous()
        )

    @staticmethod
    def quantize_mamba_substate_4bit(
        tensor: torch.Tensor,
        *,
        substate: str,
        scaling_method: str = "per_tensor",
        rh: bool = False,
        asym: bool = False,
    ) -> bytes:
        """Quantize one real Mamba sub-state with native state kernels."""
        if kvweave_quant is None:
            raise RuntimeError("KVWeave native quantization extension is unavailable")
        if tensor.dtype not in KVWeaveCodecConfig.DTYPE_TO_CODE:
            raise ValueError(
                f"unsupported dtype for 4-bit quantization: {tensor.dtype}"
            )
        if substate not in KVWeaveCodecConfig.SUBSTATE_TO_CODE:
            raise ValueError(f"unsupported substate: {substate!r}")
        if scaling_method not in KVWeaveCodecConfig.SCALING_TO_CODE:
            raise ValueError(f"unsupported scaling_method: {scaling_method!r}")
        if tensor.dim() < 2 or (
            substate == "conv"
            and rh
            and scaling_method in {"per_tensor", "per_channel"}
        ):
            raise ValueError("invalid Mamba tensor or RH configuration")
        cpu = tensor.detach().to("cpu").contiguous()
        shape = tuple(int(dim) for dim in cpu.shape)
        blocks, heads, head_dim, chunks = _KVWeaveCodec._mamba_layout(
            substate, shape, scaling_method
        )
        signs = perm = None
        if rh:
            transform = max(cpu.numel() // shape[0] // chunks, 1)
            signs, perm = _KVWeaveCodec._mamba_precond_pair(transform)
        flags = (
            KVWeaveCodecConfig.MAMBA_FLAG_RH if rh else 0
        ) | (KVWeaveCodecConfig.MAMBA_FLAG_ASYM if asym else 0)
        header = KVWeaveCodecConfig.MAMBA_MAGIC + struct.pack(
            ">BBBBBB" + "i" * len(shape),
            KVWeaveCodecConfig.MAMBA_QBIT,
            KVWeaveCodecConfig.DTYPE_TO_CODE[cpu.dtype],
            flags,
            KVWeaveCodecConfig.SCALING_TO_CODE[scaling_method],
            KVWeaveCodecConfig.SUBSTATE_TO_CODE[substate],
            len(shape),
            *shape,
        )
        return bytes(
            kvweave_quant.kvweave_serialize_chunk_state(
                cpu,
                header,
                KVWeaveCodecConfig.next_scale_id(),
                qbit=KVWeaveCodecConfig.MAMBA_QBIT,
                blocks_num=blocks,
                block_size=1,
                head_num=heads,
                head_dim=head_dim,
                num_layers=shape[0],
                rh=rh,
                asym=asym,
                scaling_method=scaling_method,
                signs=signs,
                perm=perm,
            )
        )

    @staticmethod
    def dequantize_mamba_substate_4bit(payload: bytes) -> torch.Tensor:
        """Decode a self-describing native Mamba sub-state payload."""
        if kvweave_quant is None:
            raise RuntimeError("KVWeave native quantization extension is unavailable")
        if payload[:4] != KVWeaveCodecConfig.MAMBA_MAGIC:
            raise ValueError(f"unrecognized payload magic: {payload[:4]!r}")
        qbit, dtype_code, flags, scaling_code, substate_code, ndim = struct.unpack(
            ">BBBBBB", payload[4:10]
        )
        scaling_map = {
            value: key for key, value in KVWeaveCodecConfig.SCALING_TO_CODE.items()
        }
        substate_map = {
            value: key for key, value in KVWeaveCodecConfig.SUBSTATE_TO_CODE.items()
        }
        if (
            qbit != KVWeaveCodecConfig.MAMBA_QBIT
            or dtype_code not in KVWeaveCodecConfig.CODE_TO_DTYPE
        ):
            raise ValueError("unsupported qbit or dtype in Mamba payload")
        if scaling_code not in scaling_map or substate_code not in substate_map:
            raise ValueError("unsupported scaling method or substate in Mamba payload")
        offset = 10
        shape = tuple(
            struct.unpack(
                ">" + "i" * ndim,
                payload[offset : offset + 4 * ndim],
            )
        )
        offset += 4 * ndim
        scaling = scaling_map[scaling_code]
        substate = substate_map[substate_code]
        blocks, heads, head_dim, chunks = _KVWeaveCodec._mamba_layout(
            substate, shape, scaling
        )
        (scale_size,) = struct.unpack(">I", payload[offset : offset + 4])
        offset += 4
        scales = payload[offset : offset + scale_size]
        offset += scale_size
        q_data = torch.frombuffer(bytearray(payload[offset:]), dtype=torch.int8)
        signs = perm = None
        if flags & KVWeaveCodecConfig.MAMBA_FLAG_RH:
            transform = max(
                int(torch.tensor(shape).prod().item()) // shape[0] // chunks,
                1,
            )
            signs, perm = _KVWeaveCodec._mamba_precond_pair(transform)
        tail_numel = max(int(torch.tensor(shape[2:]).prod().item()), 1)
        restored = kvweave_quant.kvweave_dequantize_chunk_state(
            q_data,
            scales,
            shape[0],
            blocks,
            tail_numel,
            qbit=KVWeaveCodecConfig.MAMBA_QBIT,
            blocks_num=blocks,
            block_size=1,
            head_num=heads,
            head_dim=head_dim,
            rh=bool(flags & KVWeaveCodecConfig.MAMBA_FLAG_RH),
            asym=bool(flags & KVWeaveCodecConfig.MAMBA_FLAG_ASYM),
            scaling_method=scaling,
            output_dtype=KVWeaveCodecConfig.CODE_TO_DTYPE[dtype_code],
            signs=signs,
            perm=perm,
        )
        return restored.reshape(shape)

    @staticmethod
    def pack_mamba_payloads(conv: bytes, ssm: bytes) -> bytes:
        """Frame conv and ssm payloads into one stored blob."""
        return struct.pack(">I", len(conv)) + conv + ssm

    @staticmethod
    def unpack_mamba_payloads(blob: bytes) -> tuple[bytes, bytes]:
        """Split a framed Mamba blob back into conv and ssm payloads."""
        if len(blob) < 4:
            raise ValueError("Mamba payload bundle is truncated")
        size = struct.unpack(">I", blob[:4])[0]
        if size > len(blob) - 4:
            raise ValueError("Mamba payload bundle is truncated")
        return blob[4 : 4 + size], blob[4 + size :]


KVWeaveCodec = _KVWeaveCodec