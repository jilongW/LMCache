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


KVWeaveCodec = _KVWeaveCodec