"""KVWeave attention and Mamba codec for LMCache L1 quantization."""

from __future__ import annotations

from dataclasses import dataclass
import io
import struct
from typing import Optional

import numpy as np
import torch

from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    KVWeaveCodecConfig,
    MambaCodecOptions,
)
from lmcache.v1.multiprocess.group_view import MambaSubStateWireLayout

try:
    from kvweave import kvweave_quant
except ImportError:  # pragma: no cover
    kvweave_quant = None

logger = init_logger(__name__)


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
        self.num_threads = config.num_threads
        self.block_size = int(kwargs.get("block_size", config.DEFAULT_BLOCK_SIZE))
        self.num_kv_heads = int(kwargs.get("num_kv_heads", kwargs.get("head_num", 1)))
        self.head_dim = int(kwargs.get("head_dim", 0))
        self._quant_mod = None

    def _native(self):
        if self._quant_mod is None:
            if kvweave_quant is None:
                raise RuntimeError("KVWeave native quantization extension is unavailable")
            self._quant_mod = kvweave_quant
        return self._quant_mod

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> bytes:
        return bytes(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())

    def estimate_serialized_size(
        self, layout_desc: MemoryLayoutDesc, scaling_method: Optional[str] = None
    ) -> int:
        """Estimate the upper bound needed for one serialized KV buffer."""
        return sum(
            self._estimate_shape(tuple(int(dim) for dim in shape), dtype, scaling_method)
            for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
        )

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
        return int((header + 8 + 2 * (4 + scales * 12) + 2 * KVWeaveCodecConfig.quantized_bytes(elements, self.qbit)) * 1.10) + 4096

    def serialize_tensor(self, tensor: torch.Tensor, scaling_method: str | None = None) -> bytes:
        """Normalize a KV tensor and serialize it as raw or 4-bit data."""
        cpu = tensor.detach().to("cpu").contiguous()
        shape = self._normalize(cpu)
        method = scaling_method or self.scaling_method
        if not self.quantize:
            return self._raw_payload(cpu, shape)
        rh, asym, precond = (False, False, False) if method == "per_tensor" else (self.rh, self.asym, self.precond)
        header = self._quant_header(shape, cpu.dtype, method, rh, asym, precond)
        ids = KVWeaveCodecConfig.next_scale_ids(2)
        payload = self._native().kvweave_serialize_chunk(
            shape.tensor4d, header, ids[0], ids[1], qbit=self.qbit,
            blocks_num=max(1, shape.chunk_tokens // self.block_size),
            block_size=self.block_size, head_num=self._head_num(shape.hidden_dim),
            head_dim=self._head_dim(shape.hidden_dim), num_layers=shape.num_layers,
            rh=rh, asym=asym, scaling_method=method, num_threads=self.num_threads,
        )
        return bytes(payload)

    def deserialize_tensor(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Decode a payload and restore it into the destination KV tensor."""
        parsed = self._parse(self._tensor_bytes(src))
        if parsed["raw"]:
            data = torch.frombuffer(bytearray(parsed["data"]), dtype=parsed["dtype"])
            dst.copy_(data.reshape(dst.shape).to(dtype=dst.dtype, device=dst.device))
            return
        shape = parsed["shape4d"]
        q = torch.frombuffer(bytearray(parsed["q_data"]), dtype=torch.int8)
        native = self._native()
        kwargs = dict(
            qbit=parsed["qbit"], blocks_num=parsed["blocks_num"], block_size=self.block_size,
            head_num=parsed["head_num"], head_dim=parsed["head_dim"], rh=parsed["rh"],
            asym=parsed["asym"], scaling_method=parsed["scaling"], output_dtype=dst.dtype,
            num_threads=self.num_threads,
        )
        if hasattr(native, "kvweave_dequantize_chunk_into_4d") and dst.dim() == 4:
            native.kvweave_dequantize_chunk_into_4d(q, parsed["k_scales"], parsed["v_scales"], dst, shape[1], shape[2], shape[3], **kwargs)
            return
        restored = native.kvweave_dequantize_chunk(q, parsed["k_scales"], parsed["v_scales"], shape[1], shape[2], shape[3], **kwargs)
        if dst.dim() == 3:
            restored = restored.squeeze(1).permute(1, 0, 2)
        dst.copy_(restored.to(dtype=dst.dtype, device=dst.device))

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

    def _parse(self, raw: bytes) -> dict[str, object]:
        """Parse the self-describing header and expose native decode fields."""
        stream = io.BytesIO(raw)
        magic = stream.read(4)
        if magic == self._config.MAGIC_RAW:
            ndim, _, dtype_code = struct.unpack(">BBB", stream.read(3))
            shape = tuple(struct.unpack(">" + "i" * ndim, stream.read(4 * ndim)))
            return {"raw": True, "shape": shape, "dtype": self._config.CODE_TO_DTYPE.get(dtype_code, torch.float16), "data": stream.read()}
        if magic != self._config.MAGIC_QUANT:
            raise ValueError(f"invalid KVWeave payload magic: {magic!r}")
        qbit, rh, asym, _, ndim, _, scaling_code, dtype_code = struct.unpack(">BBBBBBBB", stream.read(8))
        shape = tuple(struct.unpack(">" + "i" * ndim, stream.read(4 * ndim)))
        shape4d = (shape[1], 1, shape[0], shape[2]) if ndim == 3 else shape
        scales = []
        for _ in range(2):
            size = struct.unpack(">I", stream.read(4))[0]
            scales.append(stream.read(size))
        method = {value: key for key, value in self._config.SCALING_TO_CODE.items()}.get(scaling_code, self.scaling_method)
        return {"raw": False, "shape4d": shape4d, "qbit": qbit, "rh": bool(rh), "asym": bool(asym), "scaling": method, "dtype": self._config.CODE_TO_DTYPE.get(dtype_code, torch.float16), "k_scales": scales[0], "v_scales": scales[1], "q_data": stream.read(), "blocks_num": max(1, shape4d[2] // self.block_size), "head_num": self._head_num(shape4d[3]), "head_dim": self._head_dim(shape4d[3])}

    def _head_num(self, hidden: int) -> int:
        return self.num_kv_heads if self.num_kv_heads > 1 and hidden % self.num_kv_heads == 0 else 1

    def _head_dim(self, hidden: int) -> int:
        return self.head_dim or hidden // self._head_num(hidden)

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
    def split_mamba_chunk(raw: torch.Tensor, layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout], block_size: int) -> MambaChunkSplit:
        """Recover real conv/ssm bytes from the opaque synthetic page view."""
        conv_layout, ssm_layout = layout
        if block_size <= 0 or raw.shape[-2] % block_size:
            raise ValueError(
                f"chunk_tokens ({raw.shape[-2]}) is not a multiple of block_size ({block_size})"
            )
        if raw.dim() == 4 and int(raw.shape[0]) == 2:
            _, layers, tokens, hidden = map(int, raw.shape)
            page_bytes = 2 * block_size * hidden * raw.element_size()
            pages = raw.permute(1, 0, 2, 3).reshape(
                layers, 2, tokens // block_size, block_size, hidden
            ).permute(0, 2, 1, 3, 4).reshape(
                layers, tokens // block_size, page_bytes // raw.element_size()
            ).contiguous().view(torch.uint8)
        elif raw.dim() == 3:
            layers, tokens, hidden = map(int, raw.shape)
            page_bytes = block_size * hidden * raw.element_size()
            pages = raw.reshape(
                layers, tokens // block_size, block_size, hidden
            ).contiguous().view(torch.uint8).reshape(layers, tokens // block_size, page_bytes)
        else:
            raise ValueError(
                f"expected Mamba chunk shape [2,L,T,H] or [L,T,H], got {tuple(raw.shape)}"
            )
        blocks = tokens // block_size
        def read(item):
            desc, dtype = item
            end = desc.byte_offset + desc.byte_length
            if desc.byte_offset < 0 or end > page_bytes:
                raise ValueError("Mamba sub-state byte layout exceeds page size")
            return pages[:, :, desc.byte_offset:end].contiguous().view(dtype).reshape(layers, blocks, *desc.shape)
        return MambaChunkSplit(read((conv_layout, KVWeaveCodecConfig.mamba_dtype(conv_layout.dtype_str))), read((ssm_layout, KVWeaveCodecConfig.mamba_dtype(ssm_layout.dtype_str))))

    @staticmethod
    def merge_mamba_chunk(split: MambaChunkSplit, layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout], block_size: int, hidden_dim: int, raw_shape: torch.Size | None = None, raw_dtype: torch.dtype | None = None) -> torch.Tensor:
        """Rebuild the opaque page view, zero-filling unused padding bytes."""
        conv_layout, ssm_layout = layout
        layers, blocks = map(int, split.conv.shape[:2])
        if split.ssm.shape[:2] != (layers, blocks):
            raise ValueError("conv and ssm chunks have different layer/block dimensions")
        output_dtype = raw_dtype or split.conv.dtype
        planes = 1 if raw_shape is not None and len(raw_shape) == 3 else 2
        page_bytes = planes * block_size * hidden_dim * output_dtype.itemsize
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
    ) -> bytes:
        """Quantize one real Mamba sub-state with native state kernels."""
        if kvweave_quant is None:
            raise RuntimeError("KVWeave native quantization extension is unavailable")
        if tensor.dtype not in KVWeaveCodecConfig.DTYPE_TO_CODE:
            raise ValueError(f"unsupported dtype for 4-bit quantization: {tensor.dtype}")
        if substate not in KVWeaveCodecConfig.SUBSTATE_TO_CODE:
            raise ValueError(f"unsupported substate: {substate!r}")
        if scaling_method not in KVWeaveCodecConfig.SCALING_TO_CODE:
            raise ValueError(f"unsupported scaling_method: {scaling_method!r}")
        if tensor.dim() < 2 :
            raise ValueError("invalid Mamba tensor or RH configuration")
        cpu = tensor.detach().to("cpu").contiguous()
        shape = tuple(int(dim) for dim in cpu.shape)
        blocks, heads, head_dim, chunks = _KVWeaveCodec._mamba_layout(substate, shape, scaling_method)
        signs = perm = None
        if rh:
            signs, perm = _KVWeaveCodec._mamba_precond_pair(
                max(cpu.numel() // shape[0] // chunks, 1)
            )
        flags = (KVWeaveCodecConfig.MAMBA_FLAG_RH if rh else 0) | (KVWeaveCodecConfig.MAMBA_FLAG_ASYM if asym else 0)
        header = KVWeaveCodecConfig.MAMBA_MAGIC + struct.pack(">BBBBBB" + "i" * len(shape), qbit, KVWeaveCodecConfig.DTYPE_TO_CODE[cpu.dtype], flags, KVWeaveCodecConfig.SCALING_TO_CODE[scaling_method], KVWeaveCodecConfig.SUBSTATE_TO_CODE[substate], len(shape), *shape)
        return bytes(kvweave_quant.kvweave_serialize_chunk_state(
            cpu.view(-1), header, KVWeaveCodecConfig.next_scale_id(),
            qbit=qbit, blocks_num=blocks,
            block_size=1, head_num=heads, head_dim=head_dim,
            num_layers=shape[0], rh=rh, asym=asym,
            scaling_method=scaling_method, signs=signs, perm=perm,
        ))

    @staticmethod
    def _decode_mamba_substate(
        flagged_payload: bytes,
        layout: MambaSubStateWireLayout,
        layers: int,
        blocks: int,
    ) -> torch.Tensor:
        """Decode one sub-state payload, honoring its leading quant-enabled flag.

        DEBUG ONLY dispatch (see ``LMCACHE_MP_KVWEAVE_CONV_QUANT_ENABLED``/
        ``SSM_QUANT_ENABLED``): a leading ``\\x00`` byte means ``encode_chunk``
        skipped 4-bit quantization for this sub-state and wrote its real
        bytes verbatim; ``\\x01`` means the rest is a normal MQ01 payload.
        """
        flag, payload = flagged_payload[0], flagged_payload[1:]
        if flag == 0:
            dtype = KVWeaveCodecConfig.mamba_dtype(layout.dtype_str)
            return torch.frombuffer(bytearray(payload), dtype=dtype).reshape(
                layers, blocks, *layout.shape
            )
        if flag == 2:
            return torch.frombuffer(bytearray(payload), dtype=torch.float16).reshape(
                layers, blocks, *layout.shape
            ).to(dtype=KVWeaveCodecConfig.mamba_dtype(layout.dtype_str))
        return _KVWeaveCodec.dequantize_mamba_substate_4bit(payload)

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
        scaling_map = {v: k for k, v in KVWeaveCodecConfig.SCALING_TO_CODE.items()}
        substate_map = {v: k for k, v in KVWeaveCodecConfig.SUBSTATE_TO_CODE.items()}
        if qbit not in {4, 8} or dtype_code not in KVWeaveCodecConfig.CODE_TO_DTYPE:
            raise ValueError("unsupported qbit or dtype in Mamba payload")
        if scaling_code not in scaling_map or substate_code not in substate_map:
            raise ValueError("unsupported scaling method or substate in Mamba payload")
        offset = 10
        shape = tuple(struct.unpack(">" + "i" * ndim, payload[offset:offset + 4 * ndim]))
        offset += 4 * ndim
        scaling = scaling_map[scaling_code]
        substate = substate_map[substate_code]
        numel = 1
        for dim in shape:
            numel *= dim
        num_layers = shape[0]
        num_blocks = shape[1]
        blocks, heads, head_dim, chunks = _KVWeaveCodec._mamba_layout(substate, shape, scaling)
        (scale_size,) = struct.unpack(">I", payload[offset:offset + 4])
        offset += 4
        scales = payload[offset:offset + scale_size]
        offset += scale_size
        q_data = torch.frombuffer(bytearray(payload[offset:]), dtype=torch.int8)
        signs = perm = None
        if flags & KVWeaveCodecConfig.MAMBA_FLAG_RH:
            transform = max(torch.tensor(shape).prod().item() // shape[0] // chunks, 1)
            signs, perm = _KVWeaveCodec._mamba_precond_pair(int(transform))
        restored = kvweave_quant.kvweave_dequantize_chunk_state(
            q_data, scales, shape[0], num_blocks,
            numel // (num_layers * num_blocks),
            qbit=qbit, blocks_num=blocks,
            block_size=1, head_num=heads, head_dim=head_dim,
            rh=bool(flags & KVWeaveCodecConfig.MAMBA_FLAG_RH),
            asym=bool(flags & KVWeaveCodecConfig.MAMBA_FLAG_ASYM),
            scaling_method=scaling,
            output_dtype=KVWeaveCodecConfig.CODE_TO_DTYPE[dtype_code],
            signs=signs, perm=perm,
        )
        return restored.reshape(shape)

    @staticmethod
    def quantize_mamba_substate_4bit(
        tensor: torch.Tensor,
        *,
        substate: str,
        scaling_method: str = "per_tensor",
        rh: bool = False,
        asym: bool = False,
        qbit: int = KVWeaveCodecConfig.MAMBA_QBIT,
    ) -> bytes:
        """Quantize a Mamba sub-state using the verified state layout contract."""
        return _KVWeaveCodec._quantize_mamba_substate_payload(
            tensor, substate=substate, scaling_method=scaling_method,
            rh=rh, asym=asym, qbit=qbit,
        )

    @staticmethod
    def estimate_mamba_serialized_size(
        raw_layout: MemoryLayoutDesc,
        mamba_layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout],
        block_size: int,
        scaling_methods: tuple[str, str] = ("per_channel", "per_channel"),
        qbits: tuple[int, int] = (4, 8),
    ) -> int:
        """Estimate the upper bound needed for one packed conv+ssm Mamba chunk.

        Unlike :meth:`estimate_serialized_size` (attention K/V pairs),
        this does not assume a per-token-per-head element layout: it derives
        the element count for each sub-state directly from its own
        ``MambaSubStateWireLayout.shape`` and the group's ``layers``/``tokens``
        (recovered from ``raw_layout``), scaled by the native 4-bit packing
        rate. The result is a conservative upper bound (matching the safety
        margin used by ``_estimate_shape``), since it also sizes the SHM
        slot / pickle chunk buffer the quantized payload is copied into.
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
        for substate_name, substate_layout, scaling_method, qbit in zip(
            ("conv", "ssm"),
            mamba_layout, scaling_methods, qbits, strict=True
        ):
            elements = layers * blocks * max(
                int(np.prod(substate_layout.shape)) if substate_layout.shape else 1, 1
            )
            full_shape = (layers, blocks, *substate_layout.shape)
            native_blocks, _, native_head_dim, _ = _KVWeaveCodec._mamba_layout(
                substate_name, full_shape, scaling_method
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
                else 10 + 4 * len(full_shape) + scale_blob + KVWeaveCodecConfig.quantized_bytes(
                    elements, qbit
                )
            )
            # DEBUG ONLY: LMCACHE_MP_KVWEAVE_CONV_QUANT_ENABLED/SSM_QUANT_ENABLED
            # can make encode_chunk skip quantization and write raw bytes
            # instead -- size the slot for whichever is larger so toggling
            # either off at runtime never overflows the reserved slot.
            raw_size = elements * KVWeaveCodecConfig.mamba_dtype(
                substate_layout.dtype_str
            ).itemsize
            total += 1 + max(quantized_size, raw_size)
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

    def encode_chunk(
        self,
        cache_category: str,
        mamba_layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout] | None,
        tokens_per_block: int,
        mamba_options: MambaCodecOptions | None,
        raw_chunk: torch.Tensor,
    ) -> bytes:
        """Encode one gathered raw chunk into its wire-quantized byte payload.

        Dispatches on ``cache_category``: Mamba groups are split into their
        real ``conv``/``ssm`` sub-states and quantized independently (Phase
        4's dedicated Mamba codec); every other category is quantized as an
        attention K/V tensor via ``self.serialize_tensor`` (Phase 3's
        codec). Applying the attention codec to a Mamba group's opaque
        page-view chunk would silently corrupt its recurrent state -- see
        Phase 6 in MIGRATION_PLAN.md.
        """
        if mamba_layout is not None:
            if mamba_options is None:
                raise RuntimeError("Mamba codec options are not initialized")
            split = self.split_mamba_chunk(raw_chunk, mamba_layout, tokens_per_block)
            if getattr(mamba_options, "conv_quant_enabled", True):
                conv_payload = b"\x01" + self.quantize_mamba_substate_4bit(
                    split.conv,
                    substate="conv",
                    scaling_method=mamba_options.conv_scaling_method,
                    rh=mamba_options.conv_rh,
                    asym=mamba_options.asym,
                    qbit=mamba_options.conv_qbit,
                )
            else:
                conv_payload = b"\x00" + self._tensor_bytes(split.conv)
            if getattr(mamba_options, "ssm_quant_enabled", True):
                if mamba_options.ssm_qbit == 16:
                    ssm_payload = b"\x02" + self._tensor_bytes(
                        split.ssm.to(dtype=torch.float16)
                    )
                else:
                    ssm_payload = b"\x01" + self.quantize_mamba_substate_4bit(
                        split.ssm,
                        substate="ssm",
                        scaling_method=mamba_options.ssm_scaling_method,
                        rh=mamba_options.ssm_rh,
                        asym=mamba_options.asym,
                        qbit=mamba_options.ssm_qbit,
                    )
            else:
                ssm_payload = b"\x00" + self._tensor_bytes(split.ssm)
            return self.pack_mamba_payloads(conv_payload, ssm_payload)
        return self.serialize_tensor(raw_chunk)

    def decode_chunk(
        self,
        cache_category: str,
        mamba_layout: tuple[MambaSubStateWireLayout, MambaSubStateWireLayout] | None,
        tokens_per_block: int,
        raw_shape: torch.Size,
        raw_dtype: torch.dtype,
        chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Decode one retrieved wire-quantized byte chunk back to its raw shape.

        Mirrors :meth:`encode_chunk`'s dispatch. Both branches return a
        freshly allocated tensor (the attention branch's
        ``deserialize_tensor`` writes into a caller-provided destination
        internally, so callers of this method never need to know which
        convention the underlying codec uses).
        """
        if mamba_layout is not None:
            conv_layout, ssm_layout = mamba_layout
            conv_payload, ssm_payload = self.unpack_mamba_payloads(
                self._tensor_bytes(chunk)
            )
            layers = raw_shape[1] if len(raw_shape) == 4 else raw_shape[0]
            tokens = raw_shape[-2]
            blocks = max(tokens // tokens_per_block, 1)
            conv = self._decode_mamba_substate(conv_payload, conv_layout, layers, blocks)
            ssm = self._decode_mamba_substate(ssm_payload, ssm_layout, layers, blocks)
            hidden_dim = raw_shape[-1]
            merged = self.merge_mamba_chunk(
                MambaChunkSplit(conv, ssm), mamba_layout, tokens_per_block,
                hidden_dim, raw_shape=raw_shape, raw_dtype=raw_dtype,
            )
            # ``merged`` is an opaque page view. Its bytes already contain
            # each sub-state in its own wire dtype; converting the page tensor
            # numerically would corrupt fp32 SSM bytes when the page dtype is
            # fp16 (the normal Qwen3.5 layout).
            return merged
        destination = torch.empty(raw_shape, dtype=raw_dtype)
        self.deserialize_tensor(chunk, destination)
        return destination


KVWeaveCodec = _KVWeaveCodec
