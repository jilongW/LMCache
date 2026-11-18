"""KVWeave attention codec for LMCache L2 quantization."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any, Optional, cast
import struct

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    AttentionPlaneLayout,
    KVWeaveCodecConfig,
)

try:
    # Third Party
    from kvweave import kvweave_quant
except ImportError:  # pragma: no cover
    kvweave_quant = None

try:
    # Third Party
    from kvweave import kvweave_quant_xpu
except ImportError:  # pragma: no cover
    kvweave_quant_xpu = None

logger = init_logger(__name__)

_BytesLike = bytes | memoryview


def _cpu_thread_kwargs(device: str, num_threads: int) -> dict:
    """``num_threads`` kwarg for the CPU (OpenMP) native module only.

    Without it the CPU kernels use ``omp_get_max_threads()`` (torch's thread
    count), which oversubscribes against torch's own pool; the SYCL module
    has no such argument.
    """
    return {"num_threads": num_threads} if device == "cpu" and num_threads > 0 else {}


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
            raise RuntimeError(
                "KVWeave native XPU quantization extension is unavailable"
            )
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


class _KVWeaveCodec:
    """Self-describing attention KVWeave codec."""

    def __init__(self, kwargs: dict[str, Any] | None = None):
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
        # fp8 KV cache: the engine hands over uint8 tensors; this says which fp8
        # format they hold.
        self.kv_fp8_dtype: torch.dtype | None = kwargs.get("kv_fp8_dtype")  # type: ignore[assignment]
        self.fused_direct_quant = bool(kwargs.get("fused_direct_quant", True))
        self.fused_direct_dequant = bool(kwargs.get("fused_direct_dequant", True))
        self.paged_direct_encode = bool(kwargs.get("paged_direct_encode", True))
        self.paged_direct_decode = bool(kwargs.get("paged_direct_decode", True))

    def accepts_attention_hidden(self, hidden: int) -> bool:
        """Whether a split-K/V plane of width ``hidden`` matches this codec's heads.

        Args:
            hidden: Last-axis width of one K (or V) plane.

        Returns:
            ``True`` if ``hidden == num_kv_heads * head_dim`` (or the codec was
            built without a head width).
        """
        return self._is_attention_shape(hidden)

    def supports_fp8_attention(self) -> bool:
        """Whether this codec can quantize an fp8 KV cache (fused-K/V attention).

        Both the XPU and the CPU kernels read and write fp8 natively.
        """
        return self.kv_fp8_dtype is not None and self.scaling_method in {
            "per_channel",
            "per_token",
        }

    def _kv_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Map the engine's uint8 fp8-KV storage dtype to the configured fp8 format."""
        if dtype == torch.uint8 and self.kv_fp8_dtype is not None:
            return self.kv_fp8_dtype
        return dtype

    def _kv_view(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reinterpret a uint8 fp8-KV tensor as the fp8 dtype (same memory)."""
        if tensor.dtype == torch.uint8 and self.kv_fp8_dtype is not None:
            return tensor.view(self.kv_fp8_dtype)
        return tensor

    def supports_fused_direct_quant(self) -> bool:
        """Whether fused-K/V attention quantizes without a separate head split."""
        return (
            self.fused_direct_quant
            and self.quantize
            and self.scaling_method in {"per_channel", "per_token"}
            and hasattr(self.native, "kvweave_serialize_fused_chunk_direct")
        )

    def fused_quant_uses_direct(
        self, hidden: int, scaling_method: str | None = None
    ) -> bool:
        """Whether ``serialize_fused_tensor`` takes the direct kernel for this
        hidden size, in which case the heads are never split on the host."""
        method = scaling_method or self.scaling_method
        per_head_scales = method != "per_tensor" and self._fused_head_num(hidden) > 1
        return per_head_scales and self.supports_fused_direct_quant()

    def _paged_fused_geometry_ok(self, hidden: int, tokens: int | None = None) -> bool:
        """Whether paged in-place quant/dequant handles this fused width.

        XPU, per_token or per_channel scaling, per-K/V-plane heads, an
        RH-compatible transform length, and a native kernel whose SLM path fits one
        quantization chunk. A chunk is one head row (per_token) or one channel
        column over the chunk's ``tokens`` (per_channel), so per_channel also needs
        the chunk's token count.
        """
        if self.device != "xpu" or self.scaling_method not in {
            "per_token",
            "per_channel",
        }:
            return False
        head_num = self._fused_head_num(hidden)
        head_dim = self._fused_head_dim(hidden)
        if head_num <= 1:
            return False
        if self.scaling_method == "per_channel":
            if tokens is None or tokens <= 0:
                return False
            transform_len = tokens
        else:
            transform_len = head_dim
        if self.rh and transform_len & (transform_len - 1):
            return False
        return bool(self.native.kvweave_fused_paged_supported(transform_len))

    def supports_paged_fused_encode(
        self, hidden: int, tokens: int | None = None
    ) -> bool:
        """Whether a fused-K/V chunk can be quantized straight from paged KV.

        The caller can then skip the gather: only the quantized payload is copied
        to host. Output bytes match :meth:`serialize_fused_tensor` on the gathered
        chunk.
        """
        return (
            self.paged_direct_encode
            and self.quantize
            and hasattr(self.native, "kvweave_serialize_fused_paged_direct")
            and self._paged_fused_geometry_ok(hidden, tokens)
        )

    def supports_paged_fused_decode(
        self, hidden: int, tokens: int | None = None
    ) -> bool:
        """Whether a fused-K/V payload can be dequantized straight into paged KV.

        The caller can then skip the dense raw decode and the scatter.
        """
        return (
            self.paged_direct_decode
            and hasattr(self.native, "kvweave_dequantize_fused_paged_direct")
            and self._paged_fused_geometry_ok(hidden, tokens)
        )

    def serialize_fused_paged(
        self,
        layer_ptrs: torch.Tensor,
        block_ids: torch.Tensor,
        block_size: int,
        block_stride_elems: int,
        num_blocks: int,
        dtype: torch.dtype,
        hidden: int,
        out: torch.Tensor,
    ) -> int:
        """Quantize one fused-K/V chunk straight from paged KV into ``out``.

        Args:
            layer_ptrs: int64 XPU vector, one paged-KV base pointer per layer.
            block_ids: int64 XPU vector of this chunk's block ids.
            block_size: Tokens per paged block.
            block_stride_elems: Elements between consecutive blocks.
            num_blocks: Number of blocks in each paged tensor (id bound).
            dtype: Paged KV dtype.
            hidden: Fused content width (``num_heads * 2 * head_size``).
            out: Contiguous uint8 CPU tensor receiving the payload.

        Returns:
            Number of payload bytes written to ``out``.

        Raises:
            ValueError: If the codec cannot encode paged KV for this ``hidden``.
        """
        tokens = int(block_ids.numel()) * block_size
        if not self.supports_paged_fused_encode(hidden, tokens):
            raise ValueError("paged fused encode is not supported for this codec/shape")
        dtype = self._kv_dtype(dtype)
        shape = (int(layer_ptrs.numel()), tokens, hidden)
        header, rh, asym, signs, perm = self._fused_quant_params(
            dtype, shape, self.scaling_method, per_head_scales=True
        )
        head_num = self._fused_head_num(hidden)
        head_dim = self._fused_head_dim(hidden)
        return int(
            self.native.kvweave_serialize_fused_paged_direct(
                layer_ptrs,
                block_ids,
                block_size,
                block_stride_elems,
                num_blocks,
                dtype,
                header,
                self.qbit,
                head_num,
                head_dim,
                rh=rh,
                asym=asym,
                signs=signs,
                perm=perm,
                out=out,
                **self._paged_scaling_kwargs(),
            )
        )

    def deserialize_fused_paged(
        self,
        src: torch.Tensor,
        layer_ptrs: torch.Tensor,
        block_ids: torch.Tensor,
        block_size: int,
        block_stride_elems: int,
        num_blocks: int,
        skip_blocks: int,
        dtype: torch.dtype,
        hidden: int,
    ) -> None:
        """Decode a fused-K/V payload straight into paged KV.

        Args:
            src: Encoded payload (trailing padding is ignored).
            layer_ptrs: int64 XPU vector, one paged-KV base pointer per layer.
            block_ids: int64 XPU vector of this chunk's destination block ids.
            block_size: Tokens per paged block.
            block_stride_elems: Elements between consecutive blocks.
            num_blocks: Number of blocks in each paged tensor (id bound).
            skip_blocks: Leading blocks of the chunk to leave untouched.
            dtype: Paged KV dtype.
            hidden: Fused content width (``num_heads * 2 * head_size``).

        Raises:
            ValueError: If the payload is not decodable in place for this
                destination (unsupported codec/shape, or a header that does not
                match the paged geometry).
        """
        if not self.supports_paged_fused_decode(
            hidden, int(block_ids.numel()) * block_size
        ):
            raise ValueError("paged fused decode is not supported for this codec/shape")
        dtype = self._kv_dtype(dtype)
        payload = self._payload_u8(src)
        parsed = self._read_fused_header(payload)
        layers, tokens, parsed_hidden = parsed["shape"]
        if (
            not parsed["per_head_scales"]
            or parsed["scaling"] != self.scaling_method
            or parsed["qbit"] > 8
            or parsed_hidden != hidden
            or layers != int(layer_ptrs.numel())
            or tokens != int(block_ids.numel()) * block_size
            or parsed["head_dim"] != self._fused_head_dim(hidden)
            or parsed["head_num"] != self._fused_head_num(hidden)
        ):
            raise ValueError(
                f"fused payload (shape={parsed['shape']}, scaling={parsed['scaling']}) "
                "does not match the paged destination"
            )
        q = self._payload_tail(payload, parsed["q_offset"], torch.int8)
        signs = perm = None
        if parsed["precond"]:
            transform_size = self._fused_rh_transform_size(
                tokens, hidden, parsed["head_dim"], parsed["scaling"], True
            )
            signs, perm = self._config.mamba_precond_tensors(transform_size)
        self.native.kvweave_dequantize_fused_paged_direct(
            q,
            parsed["scales"],
            layer_ptrs,
            block_ids,
            block_size,
            block_stride_elems,
            num_blocks,
            skip_blocks,
            dtype,
            parsed["qbit"],
            parsed["head_num"],
            parsed["head_dim"],
            rh=parsed["rh"],
            asym=parsed["asym"],
            signs=signs,
            perm=perm,
            **self._paged_scaling_kwargs(),
        )

    def _paged_scaling_kwargs(self) -> dict[str, str]:
        """Scaling-method kwarg for the paged native calls (omitted for per_token, the
        native default, so an older extension without the argument keeps working)."""
        if self.scaling_method == "per_token":
            return {}
        return {"scaling_method": self.scaling_method}

    def supports_fused_direct_dequant(self) -> bool:
        """Whether fused-K/V attention decodes straight into the fused layout."""
        # CPU only: on XPU the wide-stride fused writes are slower than dequant +
        # device permute. Measured on XPU: kvweave_dequantize_fused_chunk_direct is
        # ~15-20% slower than the existing kvweave_dequantize_chunk_state_into path
        # for per_channel, roughly a wash (+/-7%) for per_token. Keep this CPU-only
        # until that changes.
        return (
            self.fused_direct_dequant
            and self.device == "cpu"
            and hasattr(self.native, "kvweave_dequantize_fused_chunk_direct")
        )

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> bytes:
        return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

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
    def _payload_tail(
        payload: torch.Tensor, offset: int, dtype: torch.dtype
    ) -> torch.Tensor:
        if dtype == torch.int8:
            return payload[offset:].view(torch.int8)
        if dtype == torch.uint8:
            return payload[offset:]
        if dtype == torch.int16:
            if offset % 2:
                raise ValueError("KVWeave int16 q_data is not 2-byte aligned")
            return payload[offset:].view(torch.int16)
        if dtype in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        }:
            itemsize = dtype.itemsize
            tail = payload[offset:]
            if offset % itemsize:
                return torch.frombuffer(
                    bytearray(tail.contiguous().numpy().tobytes()), dtype=dtype
                )
            return tail.view(dtype)
        raise ValueError(f"unsupported KVWeave payload view dtype: {dtype}")

    def estimate_serialized_size(
        self, layout_desc: MemoryLayoutDesc, scaling_method: Optional[str] = None
    ) -> int:
        """Estimate the upper bound needed for one serialized KV buffer."""
        return sum(
            self._estimate_shape(
                tuple(int(dim) for dim in shape), dtype, scaling_method
            )
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
        """Exact serialized size of one fused-K/V ``[L, T, H]`` chunk.

        Layout: ``magic(4) + 5 flag bytes + 4*ndim shape + 4 scale_len``, then
        the scale blob (``12`` bytes per scale plus ``4`` per native layer),
        then the packed q data. Independent of ``rh``/``asym``/``precond``.
        """
        if len(shape) != 3:
            raise ValueError(
                f"KVWeave fused-K/V estimate expects [L, T, H], got {shape}"
            )
        layers, tokens, hidden = (int(dim) for dim in shape)
        method = method or self.scaling_method
        elements = layers * tokens * hidden
        if not self.quantize:
            return 32 + elements * dtype.itemsize
        scales = self._fused_scale_count(tokens, hidden, layers, method)
        fused_heads = self._fused_head_num(hidden)
        native_layers = (
            layers * fused_heads
            if method != "per_tensor" and fused_heads > 1
            else layers
        )
        header = 4 + 5 + 4 * len(shape) + 4
        return (
            header
            + scales * 12
            + 4 * native_layers
            + KVWeaveCodecConfig.quantized_bytes(elements, self.qbit)
        )

    def _estimate_shape(
        self, shape: tuple[int, ...], dtype: torch.dtype, method: str | None
    ) -> int:
        """Exact serialized size of one split-K/V ``[2, L, T, H]`` chunk.

        Layout: ``magic(4) + 8 flag bytes + 4*ndim shape``, then two scale
        blobs (``4`` length + ``12`` per scale + ``4`` per layer each), then
        the packed q data of both planes. Independent of ``rh``/``asym``.
        """
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
        header = 4 + 8 + 4 * len(shape)
        return (
            header
            + 2 * (4 + scales * 12 + 4 * int(layers))
            + 2 * KVWeaveCodecConfig.quantized_bytes(elements, self.qbit)
        )

    def serialize_tensor(
        self, tensor: torch.Tensor, scaling_method: str | None = None
    ) -> bytes:
        """Normalize a KV tensor and serialize it as raw or 4-bit data."""
        work = self._kv_view(tensor.detach().to("cpu").contiguous())
        shape = self._normalize(work)
        method = scaling_method or self.scaling_method
        if not self.quantize:
            return self._raw_payload(work, shape)
        if not self._is_attention_shape(shape.hidden_dim):
            raise ValueError(
                f"KVWeave attention quantization expects hidden size "
                f"{self.num_kv_heads * self.head_dim}, got {shape.hidden_dim}"
            )
        rh, asym, precond = (
            (False, False, False)
            if method == "per_tensor"
            else (self.rh, self.asym, self.precond)
        )
        header = self._quant_header(shape, work.dtype, method, rh, asym, precond)
        ids = KVWeaveCodecConfig.next_scale_ids(2)
        signs, perm = self._attention_precond(
            method, rh, precond, shape.chunk_tokens, shape.hidden_dim
        )
        payload = bytes(
            self.native.kvweave_serialize_chunk(
                shape.tensor4d,
                header,
                ids[0],
                ids[1],
                qbit=self.qbit,
                blocks_num=max(1, shape.chunk_tokens // self.block_size),
                block_size=self.block_size,
                head_num=self._head_num(shape.hidden_dim),
                head_dim=self._head_dim(shape.hidden_dim),
                num_layers=shape.num_layers,
                rh=rh,
                asym=asym,
                scaling_method=method,
                signs=signs,
                perm=perm,
                num_threads=self.num_threads,
            )
        )
        raw_bytes = work.numel() * work.element_size()
        logger.debug(
            "KVWeave quantize store shape=%s dtype=%s qbit=%d scaling=%s "
            "raw_bytes=%d payload_bytes=%d ratio=%.4f",
            tuple(work.shape),
            work.dtype,
            self.qbit,
            method,
            raw_bytes,
            len(payload),
            len(payload) / raw_bytes if raw_bytes else 0.0,
        )
        return payload

    def deserialize_tensor(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Decode a payload and restore it into the destination KV tensor."""
        dst = self._kv_view(dst)
        payload = self._payload_u8(src)
        parsed = self._read_header(payload)
        if parsed["raw"]:
            data = self._payload_tail(payload, parsed["data_offset"], parsed["dtype"])
            dst.copy_(data.reshape(dst.shape).to(dtype=dst.dtype, device=dst.device))
            return
        shape = parsed["shape4d"]
        q_dtype = torch.int8 if parsed["qbit"] <= 8 else torch.int16
        q = self._payload_tail(payload, parsed["q_offset"], q_dtype)
        signs, perm = self._attention_precond(
            parsed["scaling"], parsed["rh"], parsed["precond"], shape[2], shape[3]
        )
        kwargs = dict(
            qbit=parsed["qbit"],
            blocks_num=parsed["blocks_num"],
            block_size=self.block_size,
            head_num=parsed["head_num"],
            head_dim=parsed["head_dim"],
            rh=parsed["rh"],
            asym=parsed["asym"],
            scaling_method=parsed["scaling"],
            output_dtype=dst.dtype,
            signs=signs,
            perm=perm,
            num_threads=self.num_threads,
        )
        if (
            hasattr(self.native, "kvweave_dequantize_chunk_into_4d")
            and dst.dim() == 4
            and dst.device.type == "cpu"
        ):
            self.native.kvweave_dequantize_chunk_into_4d(
                q,
                parsed["k_scales"],
                parsed["v_scales"],
                dst,
                shape[1],
                shape[2],
                shape[3],
                **kwargs,
            )
            return
        restored = self.native.kvweave_dequantize_chunk(
            q,
            parsed["k_scales"],
            parsed["v_scales"],
            shape[1],
            shape[2],
            shape[3],
            **kwargs,
        )
        if dst.dim() == 3:
            restored = restored.squeeze(1).permute(1, 0, 2)
        dst.copy_(restored.to(dtype=dst.dtype, device=dst.device))

    def _fused_quant_params(
        self,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        method: str,
        per_head_scales: bool,
    ) -> tuple[bytes, bool, bool, torch.Tensor | None, torch.Tensor | None]:
        """Header and RH/asym/precond parameters for quantizing a fused chunk.

        Returns:
            ``(header, rh, asym, signs, perm)``; ``signs``/``perm`` are ``None``
            unless preconditioning is on. Shared by every fused encode entry so
            their payloads cannot drift apart.
        """
        _, tokens, hidden = shape
        rh, asym = (False, False) if method == "per_tensor" else (self.rh, self.asym)
        precond = rh and self.precond
        flags = (
            (1 if rh else 0)
            | (2 if asym else 0)
            | (4 if precond else 0)
            | (8 if per_head_scales else 0)
        )
        header = self._config.MAGIC_QUANT_FUSED + struct.pack(
            ">BBBBB" + "i" * len(shape),
            self.qbit,
            self._config.DTYPE_TO_CODE.get(dtype, 0),
            flags,
            self._config.SCALING_TO_CODE.get(method, 1),
            len(shape),
            *shape,
        )
        signs = perm = None
        if precond:
            transform_size = self._fused_rh_transform_size(
                tokens, hidden, self._fused_head_dim(hidden), method, per_head_scales
            )
            signs, perm = self._config.mamba_precond_tensors(transform_size)
        return header, rh, asym, signs, perm

    def serialize_fused_tensor(
        self,
        tensor: torch.Tensor,
        scaling_method: str | None = None,
        out: torch.Tensor | None = None,
    ) -> bytes | int:
        """Serialize a fused-K/V attention chunk (no leading K/V axis).

        Returns the payload as ``bytes``. If ``out`` (a contiguous ``uint8``
        CPU tensor) is given and the native module can write in place (the
        direct CPU kernel, or ``kvweave_serialize_chunk_state_into`` on XPU),
        the payload is written straight into ``out`` and its length returned.
        """
        if tensor.dim() != 3:
            raise ValueError(
                f"KVWeave fused-K/V tensor expects [L, T, H], got {tuple(tensor.shape)}"
            )
        work = self._kv_view(tensor.detach().to("cpu").contiguous())
        shape = tuple(int(dim) for dim in work.shape)
        layers, tokens, hidden = shape
        method = scaling_method or self.scaling_method
        head_num = self._fused_head_num(hidden)
        head_dim = self._fused_head_dim(hidden)
        per_head_scales = method != "per_tensor" and head_num > 1
        header, rh, asym, signs, perm = self._fused_quant_params(
            work.dtype, shape, method, per_head_scales
        )
        blocks_num = max(1, tokens // self.block_size)
        if self.fused_quant_uses_direct(hidden, method):
            result = self.native.kvweave_serialize_fused_chunk_direct(
                work,
                header,
                self.qbit,
                head_num,
                head_dim,
                rh=rh,
                asym=asym,
                scaling_method=method,
                signs=signs,
                perm=perm,
                num_threads=self.num_threads,
                out=out,
            )
            return result if out is not None else bytes(result)
        native_src = work
        native_layers = layers
        if per_head_scales:
            native_src = self.split_fused_attention_heads(work, head_num, head_dim)
            native_layers *= head_num
        native_kwargs = dict(
            qbit=self.qbit,
            blocks_num=blocks_num,
            block_size=self.block_size,
            head_num=head_num,
            head_dim=head_dim,
            num_layers=native_layers,
            rh=rh,
            asym=asym,
            scaling_method=method,
            signs=signs,
            perm=perm,
            num_threads=self.num_threads,
        )
        if out is not None and hasattr(
            self.native, "kvweave_serialize_chunk_state_into"
        ):
            return int(
                self.native.kvweave_serialize_chunk_state_into(
                    native_src.view(-1),
                    out.view(-1),
                    header,
                    KVWeaveCodecConfig.next_scale_id(),
                    **native_kwargs,
                )
            )
        return bytes(
            self.native.kvweave_serialize_chunk_state(
                native_src.view(-1),
                header,
                KVWeaveCodecConfig.next_scale_id(),
                **native_kwargs,
            )
        )

    @staticmethod
    def split_fused_attention_heads(
        source: torch.Tensor,
        head_num: int,
        head_dim: int,
        num_threads: int = 0,
    ) -> torch.Tensor:
        """Reorder fused ``[L,T,H*D]`` attention into contiguous head rows."""
        layers, tokens, hidden = map(int, source.shape)
        if hidden != head_num * head_dim:
            raise ValueError("fused hidden size does not match head layout")
        if kvweave_quant is not None and hasattr(
            kvweave_quant, "kvweave_split_fused_heads"
        ):
            return kvweave_quant.kvweave_split_fused_heads(
                source.contiguous(), head_num, head_dim, num_threads, None
            )
        return (
            source.reshape(layers, tokens, head_num, head_dim)
            .permute(0, 2, 1, 3)
            .reshape(layers * head_num, tokens, head_dim)
            .contiguous()
        )

    def deserialize_fused_tensor(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Decode a fused-K/V payload and restore it into the destination tensor."""
        dst = self._kv_view(dst)
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
        native_layers = (
            layers * parsed["head_num"] if parsed["per_head_scales"] else layers
        )
        native_hidden = parsed["head_dim"] if parsed["per_head_scales"] else hidden
        if (
            parsed["per_head_scales"]
            and parsed["scaling"] in {"per_channel", "per_token"}
            and parsed["qbit"] <= 8
            and dst.device.type == "cpu"
            and dst.is_contiguous()
            and self.supports_fused_direct_dequant()
        ):
            self.native.kvweave_dequantize_fused_chunk_direct(
                q,
                parsed["scales"],
                dst,
                layers,
                tokens,
                parsed["head_num"],
                parsed["head_dim"],
                parsed["qbit"],
                rh=parsed["rh"],
                asym=parsed["asym"],
                scaling_method=parsed["scaling"],
                signs=signs,
                perm=perm,
                num_threads=self.num_threads,
            )
            return
        native_kwargs = dict(
            qbit=parsed["qbit"],
            blocks_num=blocks_num,
            block_size=self.block_size,
            head_num=parsed["head_num"],
            head_dim=parsed["head_dim"],
            rh=parsed["rh"],
            asym=parsed["asym"],
            scaling_method=parsed["scaling"],
            output_dtype=dst.dtype,
            signs=signs,
            perm=perm,
            num_threads=self.num_threads,
        )
        if (
            hasattr(self.native, "kvweave_dequantize_chunk_state_into")
            and dst.device.type == "cpu"
            and dst.is_contiguous()
        ):
            # XPU module: DMA straight into dst, head merge done on the device.
            self.native.kvweave_dequantize_chunk_state_into(
                q,
                parsed["scales"],
                dst.view(-1),
                native_layers,
                tokens,
                native_hidden,
                merge_head_num=parsed["head_num"] if parsed["per_head_scales"] else 0,
                **native_kwargs,
            )
            return
        restored = self.native.kvweave_dequantize_chunk_state(
            q,
            parsed["scales"],
            native_layers,
            tokens,
            native_hidden,
            **native_kwargs,
        )
        if parsed["per_head_scales"]:
            if kvweave_quant is not None and hasattr(
                kvweave_quant, "kvweave_merge_fused_heads"
            ):
                kvweave_quant.kvweave_merge_fused_heads(
                    restored,
                    layers,
                    parsed["head_num"],
                    native_hidden,
                    self.num_threads,
                    dst.reshape(layers, tokens, hidden),
                )
                return
            else:
                restored = (
                    restored.reshape(layers, parsed["head_num"], tokens, native_hidden)
                    .permute(0, 2, 1, 3)
                    .reshape(layers, tokens, hidden)
                    .contiguous()
                )
        dst.copy_(restored.reshape(dst.shape).to(dtype=dst.dtype, device=dst.device))

    def _read_fused_header(self, payload: torch.Tensor) -> dict[str, Any]:
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
            return _KVShape(
                tensor.permute(1, 0, 2).unsqueeze(1).contiguous(),
                tuple(tensor.shape),
                1,
                tokens,
                hidden,
                3,
            )
        if tensor.dim() == 4:
            kv_size, layers, tokens, hidden = map(int, tensor.shape)
            if kv_size != 2:
                raise ValueError("KVWeave expects [2, L, T, H]")
            return _KVShape(tensor, tuple(tensor.shape), layers, tokens, hidden, 4)
        raise ValueError(f"unsupported KVWeave tensor shape: {tuple(tensor.shape)}")

    def _raw_payload(self, tensor: torch.Tensor, shape: _KVShape) -> bytes:
        header = struct.pack(
            ">4sBBB",
            self._config.MAGIC_RAW,
            shape.original_ndim,
            1,
            self._config.DTYPE_TO_CODE.get(tensor.dtype, 0),
        )
        return (
            header
            + struct.pack(">" + "i" * shape.original_ndim, *shape.header_shape)
            + self._tensor_bytes(tensor)
        )

    def _quant_header(
        self,
        shape: _KVShape,
        dtype: torch.dtype,
        method: str,
        rh: bool,
        asym: bool,
        precond: bool,
    ) -> bytes:
        return struct.pack(
            ">4sBBBBBBBB" + "i" * shape.original_ndim,
            self._config.MAGIC_QUANT,
            self.qbit,
            int(rh),
            int(asym),
            int(precond),
            shape.original_ndim,
            1,
            self._config.SCALING_TO_CODE.get(method, 1),
            self._config.DTYPE_TO_CODE.get(dtype, 0),
            *shape.header_shape,
        )

    def _read_header(self, payload: torch.Tensor) -> dict[str, Any]:
        """Read the attention payload header without copying q_data."""
        raw = self._payload_buffer(payload)
        magic = bytes(raw[:4])
        offset = 4
        if magic == self._config.MAGIC_RAW:
            ndim, _, dtype_code = struct.unpack_from(">BBB", raw, offset)
            offset += 3
            shape = tuple(struct.unpack_from(">" + "i" * ndim, raw, offset))
            offset += 4 * ndim
            return {
                "raw": True,
                "shape": shape,
                "dtype": self._config.CODE_TO_DTYPE.get(dtype_code, torch.float16),
                "data_offset": offset,
            }
        if magic != self._config.MAGIC_QUANT:
            raise ValueError(f"invalid KVWeave payload magic: {magic!r}")
        qbit, rh, asym, precond, ndim, _, scaling_code, dtype_code = struct.unpack_from(
            ">BBBBBBBB", raw, offset
        )
        offset += 8
        shape = tuple(struct.unpack_from(">" + "i" * ndim, raw, offset))
        offset += 4 * ndim
        shape4d = (shape[1], 1, shape[0], shape[2]) if ndim == 3 else shape
        scales = []
        for _ in range(2):
            size = struct.unpack_from(">I", raw, offset)[0]
            offset += 4
            scales.append(bytes(raw[offset : offset + size]))
            offset += size
        method = {
            value: key for key, value in self._config.SCALING_TO_CODE.items()
        }.get(scaling_code, self.scaling_method)
        return {
            "raw": False,
            "shape4d": shape4d,
            "qbit": qbit,
            "rh": bool(rh),
            "asym": bool(asym),
            "precond": bool(precond),
            "scaling": method,
            "dtype": self._config.CODE_TO_DTYPE.get(dtype_code, torch.float16),
            "k_scales": scales[0],
            "v_scales": scales[1],
            "q_offset": offset,
            "blocks_num": max(1, shape4d[2] // self.block_size),
            "head_num": self._head_num(shape4d[3]),
            "head_dim": self._head_dim(shape4d[3]),
        }

    def _attention_precond(
        self, method: str, rh: bool, precond: bool, tokens: int, hidden: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the (signs, perm) P/D pair for the attention path, if any.

        The transform length is the quantization chunk size of one layer plane
        (same rule as the fused-K/V path). Serialize and deserialize must derive
        it identically; the header's precond flag is only set when this applies.
        """
        if not (rh and precond):
            return None, None
        size = self._fused_rh_transform_size(
            tokens, hidden, self._head_dim(hidden), method
        )
        return self._config.mamba_precond_tensors(size)

    def _head_num(self, hidden: int) -> int:
        return (
            self.num_kv_heads
            if self.num_kv_heads > 1 and hidden % self.num_kv_heads == 0
            else 1
        )

    def _head_dim(self, hidden: int) -> int:
        return self.head_dim or hidden // self._head_num(hidden)

    def _is_attention_shape(self, h_merged: int) -> bool:
        expected_hidden = (
            self.num_kv_heads * self.head_dim if self.head_dim > 0 else h_merged
        )
        return h_merged > 0 and h_merged == expected_hidden

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
        base = (
            tokens
            if method == "per_token"
            else self._head_dim(hidden)
            if method == "per_channel"
            else 1
        )
        return max(1, base * layers if layers > 1 else base)

    @staticmethod
    def encode_chunk(
        self,
        raw_chunk: torch.Tensor,
        attention_plane_layout: AttentionPlaneLayout | None = None,
    ) -> bytes:
        """Encode one gathered raw attention chunk into its quantized payload.

        Chunks are quantized via ``self.serialize_fused_tensor`` when
        ``attention_plane_layout`` is ``AttentionPlaneLayout.FUSED_KV`` (K/V
        packed into one tensor, no leading K/V axis), or
        ``self.serialize_tensor`` otherwise.
        """
        if attention_plane_layout == AttentionPlaneLayout.FUSED_KV:
            return cast(bytes, self.serialize_fused_tensor(raw_chunk))
        return self.serialize_tensor(raw_chunk)

    def encode_chunk_into(
        self,
        raw_chunk: torch.Tensor,
        destination: torch.Tensor,
        attention_plane_layout: AttentionPlaneLayout | None = None,
    ) -> int:
        """Encode a chunk and copy its payload directly into ``destination``.

        For fused-K/V attention chunks on a native path that supports it
        (direct CPU kernel or XPU), the payload is written straight into
        ``destination`` with no intermediate ``bytes`` object; every other path
        builds the payload as ``bytes`` and copies it in. Returns the number of
        payload bytes written.
        """
        if destination.dtype != torch.uint8 or not destination.is_contiguous():
            raise ValueError("destination must be a contiguous torch.uint8 tensor")
        if attention_plane_layout == AttentionPlaneLayout.FUSED_KV:
            written = self.serialize_fused_tensor(raw_chunk, out=destination.view(-1))
            if isinstance(written, int):
                return written
            return self._copy_payload_into(written, destination)
        payload = self.encode_chunk(raw_chunk, attention_plane_layout)
        return self._copy_payload_into(payload, destination)

    @staticmethod
    def _copy_payload_into(payload: _BytesLike, destination: torch.Tensor) -> int:
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
        raw_shape: torch.Size,
        raw_dtype: torch.dtype,
        chunk: torch.Tensor,
        attention_plane_layout: AttentionPlaneLayout | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode one retrieved quantized byte chunk back to its raw shape.

        Mirrors :meth:`encode_chunk`. When ``out`` is supplied, the decoded raw
        chunk is written there; otherwise a fresh tensor is allocated.
        """
        destination = (
            out if out is not None else torch.empty(raw_shape, dtype=raw_dtype)
        )
        if destination.shape != raw_shape or destination.dtype != raw_dtype:
            raise ValueError("out must match raw_shape and raw_dtype")
        if attention_plane_layout == AttentionPlaneLayout.FUSED_KV:
            self.deserialize_fused_tensor(chunk, destination)
        else:
            self.deserialize_tensor(chunk, destination)
        return destination


KVWeaveCodec = _KVWeaveCodec
