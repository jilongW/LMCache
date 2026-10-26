# SPDX-License-Identifier: Apache-2.0
"""Non-GPU context abstractions and utilities for multiprocess transport.

This module provides:
- ``EngineDrivenContextMetadata``: layout metadata dataclass for non-CUDA workers.
- ``EngineDrivenContext``: abstract base class with a two-phase prepare/commit
  interface for CPU-side KV data transfer. Concrete implementations (e.g.
  ``EngineDrivenContextPickle``) each decide *how* data is serialised and transported.
- ``create_engine_driven_context()``: factory that returns the appropriate
  ``EngineDrivenContext`` subclass.
- ``compute_kv_layout``, ``gather_paged_kv_to_cpu``, ``scatter_cpu_to_paged_kv``:
  shared gather/scatter utilities used by all concrete implementations.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
import inspect
import ctypes
import threading

# Third Party
import numpy as np
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transport.base import RequestClient
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    pass

logger = init_logger(__name__)

_XPU_H2D_USM_POOL_LOCK = threading.Lock()
_XPU_H2D_USM_POOL: dict[int, list[tuple[int, object]]] = {}
_XPU_USM_POOL_MAX_PER_SIZE = 1
_XPU_TRANSFER_TENSOR_CACHE_LOCK = threading.Lock()
_XPU_TRANSFER_TENSOR_CACHE: dict[tuple[object, ...], torch.Tensor] = {}
_XPU_TRANSFER_TENSOR_CACHE_LIMIT = 256
def _cached_xpu_long_tensor(
    values: list[int], device: torch.device, kind: str
) -> torch.Tensor:
    """Reuse XPU transfer tensors across requests.

    Layer pointers are immutable for a KV group and are cached by value.
    Block IDs change on every request, so their tensor is pooled by length and
    updated in place instead of allocating a new XPU tensor for each request.
    """
    key = (kind, device.type, device.index, *values)
    with _XPU_TRANSFER_TENSOR_CACHE_LOCK:
        cached = _XPU_TRANSFER_TENSOR_CACHE.get(key)
        if cached is not None:
            return cached
    if kind == "layer_ptrs":
        raw = np.asarray(values, dtype=np.uint64).view(np.int64)
        tensor = torch.from_numpy(raw).to(device=device)
    else:
        tensor = torch.tensor(values, dtype=torch.int64, device=device)
    with _XPU_TRANSFER_TENSOR_CACHE_LOCK:
        if len(_XPU_TRANSFER_TENSOR_CACHE) >= _XPU_TRANSFER_TENSOR_CACHE_LIMIT:
            _XPU_TRANSFER_TENSOR_CACHE.pop(next(iter(_XPU_TRANSFER_TENSOR_CACHE)))
        _XPU_TRANSFER_TENSOR_CACHE[key] = tensor
    return tensor


def _return_xpu_usm_buffer(nbytes: int, ptr: int, buffer: object) -> None:
    """Return a USM buffer to its bounded size bucket or release it."""
    with _XPU_H2D_USM_POOL_LOCK:
        bucket = _XPU_H2D_USM_POOL.setdefault(nbytes, [])
        if len(bucket) < _XPU_USM_POOL_MAX_PER_SIZE:
            bucket.append((ptr, buffer))
            return
    from lmcache import device_ops

    device_ops.free_pinned_ptr(ptr)


def set_xpu_usm_pool_max_per_size(max_buffers: int) -> None:
    """Set the per-size USM pool limit from the transfer worker count."""
    global _XPU_USM_POOL_MAX_PER_SIZE
    with _XPU_H2D_USM_POOL_LOCK:
        _XPU_USM_POOL_MAX_PER_SIZE = max(1, int(max_buffers))


# ---------------------------------------------------------------------------
# Global capability flag: does device_ops.multi_layer_block_kv_transfer accept
# list[torch.Tensor] directly for lmcache_objects_ptrs, or only list[int]?
#
# We inspect the function signature once at import time. If the annotation
# for ``lmcache_objects_ptrs`` includes ``Tensor``, the op can handle tensors
# natively and we pass them through. Otherwise (annotation is list[int], or
# inspect fails entirely) we must convert tensors to data pointers before
# calling.
# ---------------------------------------------------------------------------
def _detect_block_transfer_accepts_tensor() -> bool:
    """Return True if device_ops.multi_layer_block_kv_transfer accepts
    list[torch.Tensor] for its lmcache_objects_ptrs parameter."""
    try:
        # First Party
        from lmcache import device_ops as _device_ops

        fn = _device_ops.multi_layer_block_kv_transfer

        # Attempt: use inspect.signature (works on newer pybind11 builds)
        # Assumptions: if lmcache_objects_ptrs accepts tensors,
        # it's fallback path, and we do not convert tensors to ptrs explicitly.
        # TODO: String matching on annotations is fragile. Wait for device_ops to
        # expose a direct version flag (e.g., device_ops.__version__) or
        # an explicit capability boolean.
        try:
            sig = inspect.signature(fn)
            param = sig.parameters.get("lmcache_objects_ptrs")
            if param is not None and param.annotation is not inspect.Parameter.empty:
                ann_str = str(param.annotation)
                if "Tensor" in ann_str:
                    return True
                # Annotation exists but no Tensor mention → ptr-only
                return False
        except (ValueError, TypeError):
            pass

    except Exception:
        # Import failed or any other error → conservative: assume ptr-only
        pass

    # Default: inspect failed or device_ops not available → assume ptr-only
    return False


_LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR: bool = _detect_block_transfer_accepts_tensor()
"""If True, ``device_ops.multi_layer_block_kv_transfer`` accepts
``list[torch.Tensor]`` directly for ``lmcache_objects_ptrs``.
If False, callers must convert tensors to ``list[int]`` data pointers."""

logger.info(
    "multi_layer_block_kv_transfer mode: %s",
    "tensor" if _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR else "ptr",
)

# Non-flash-infer XPU formats whose native SYCL block-ID kernel has been
# verified correct (D2H/H2D round-trip against the torch reference, multiple
# shapes incl. non-divisible head counts) and is safe to route off the
# torch_ops fallback. Only add a format here after such verification --
# other formats' native XPU kernels may still have unverified layout bugs,
# which is why the fallback below defaults to torch_ops for anything not on
# this list (or not the single flash-infer format checked separately above).
_XPU_NATIVE_VERIFIED_FORMATS: frozenset = frozenset(
    {
        # vLLM non-MLA blocks-first, fused K/V (NHD) -- attention's format.
        lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS,
    }
)


def _multi_layer_block_kv_transfer(
    paged_buffer_ptrs: torch.Tensor | list[torch.Tensor],
    lmcache_objects: list[int] | list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device,
    direction: lmcache_native.TransferDirection,
    shape_desc: object,
    lmcache_chunk_size: int,
    engine_kv_format: lmcache_native.EngineKVFormat,
    skip_prefix_n_blocks: int,
    fallback_paged_buffer: list[torch.Tensor] | None = None,
    fallback_objects: list[torch.Tensor] | None = None,
) -> None:
    """Use the native transfer when its layout coverage is available."""
    from lmcache import device_ops
    from lmcache.v1.platform import torch_ops

    native_xpu_format = (
        lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
    )
    if (
        device.type == "xpu"
        and engine_kv_format != native_xpu_format
        and engine_kv_format not in _XPU_NATIVE_VERIFIED_FORMATS
        and fallback_paged_buffer is not None
        and fallback_objects is not None
    ):
        torch_ops.multi_layer_block_kv_transfer(
            fallback_paged_buffer,
            fallback_objects,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )
        return

    usm_ptrs: list[int] = []
    usm_buffers: list[object] = []
    if (
        device.type == "xpu"
        and direction == lmcache_native.TransferDirection.H2D
        and engine_kv_format == native_xpu_format
        and fallback_objects is not None
        and not _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR
        and not all(source.is_pinned() for source in fallback_objects)
    ):
        try:
            staged_objects: list[int] = []
            pooled_buffers: list[tuple[int, object]] = []
            for source in fallback_objects:
                source_contiguous = source.contiguous()
                nbytes = int(source_contiguous.numel()) * source_contiguous.element_size()
                with _XPU_H2D_USM_POOL_LOCK:
                    bucket = _XPU_H2D_USM_POOL.get(nbytes)
                    pooled = bucket.pop() if bucket else None
                if pooled is None:
                    ptr = device_ops.alloc_pinned_ptr(nbytes)
                    buffer = (ctypes.c_uint8 * nbytes).from_address(ptr)
                else:
                    ptr, buffer = pooled
                ctypes.memmove(ptr, source_contiguous.data_ptr(), nbytes)
                usm_ptrs.append(ptr)
                usm_buffers.append((buffer, source_contiguous))
                pooled_buffers.append((ptr, buffer))
                staged_objects.append(ptr)
            device_ops.multi_layer_block_kv_transfer(
                paged_buffer_ptrs,
                staged_objects,
                block_ids,
                device,
                direction,
                shape_desc,
                lmcache_chunk_size,
                engine_kv_format,
                skip_prefix_n_blocks,
            )
            torch_dev.synchronize()
            return
        finally:
            for nbytes, (ptr, buffer) in zip(
                [int(source.numel()) * source.element_size() for source in fallback_objects],
                pooled_buffers,
                strict=True,
            ):
                _return_xpu_usm_buffer(nbytes, ptr, buffer)
            usm_ptrs.clear()
            usm_buffers.clear()

    try:
        device_ops.multi_layer_block_kv_transfer(
            paged_buffer_ptrs,
            lmcache_objects,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )
    except RuntimeError as exc:
        if "Unsupported" not in str(exc) or "EngineKVFormat" not in str(exc):
            raise
        logger.debug(
            "Falling back to torch KV transfer for unsupported XPU format=%s",
            engine_kv_format,
        )
        if fallback_paged_buffer is not None and fallback_objects is not None:
            paged_buffer_ptrs = fallback_paged_buffer
            lmcache_objects = fallback_objects
        torch_ops.multi_layer_block_kv_transfer(
            paged_buffer_ptrs,
            lmcache_objects,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )


def _tensors_to_ptrs(tensors: list[torch.Tensor]) -> list[int]:
    """Convert a list of tensors to a list of their data_ptr() values."""
    return [t.data_ptr() for t in tensors]


# ---------------------------------------------------------------------------


@dataclass
class EngineDrivenContextMetadata:
    """Non-GPU context layout metadata for non-CUDA workers.

    Attributes:
        layout_desc: Memory layout descriptor used to interpret chunk payloads.
        block_size: Number of tokens per paged block.
        use_mla: Whether the worker KV format is MLA.
    """

    layout_desc: MemoryLayoutDesc
    block_size: int
    use_mla: bool


class EngineDrivenContext(ABC):
    """Abstract base class for CPU-side KV data transfer contexts.

    All concrete implementations share a common request client and
    expose a uniform two-phase ``prepare/commit`` interface so that the
    worker adapter is implementation-agnostic.

    Args:
        metadata: Layout metadata describing the chunk format.
        req_client: Transport-neutral client used for server requests.
        mq_timeout: Timeout in seconds for blocking MQ requests.
    """

    def __init__(
        self,
        metadata: EngineDrivenContextMetadata,
        req_client: RequestClient,
        mq_timeout: float,
    ) -> None:
        self.metadata = metadata
        self.req_client = req_client
        self.mq_timeout = mq_timeout

    @property
    def layout_desc(self) -> MemoryLayoutDesc:
        """The memory layout descriptor for this context."""
        return self.metadata.layout_desc

    @abstractmethod
    def prepare_store(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        """Prepare SHM buffers for a store operation.

        Returns:
            None: pickle mode — no pre-allocated buffers. Caller gathers all
                chunks to CPU itself and sends the serialized data via
                commit_store.
            ([], []): SHM mode but all chunks already cached. Caller should
                skip gather and commit entirely.
            (tensors, chunk_indices): SHM mode with new chunks to write.
                - tensors[i] is a writable SHM-backed buffer for one chunk.
                - chunk_indices[i] is the position of that chunk in the full
                  block_ids sequence (e.g. [0, 2] means only chunks 0 and 2
                  need writing; chunk 1 is already cached).
                Caller gathers only these chunks into the provided tensors,
                then calls commit_store with empty payload.
        """
        ...

    @abstractmethod
    def commit_store(
        self, key: IPCCacheServerKey, instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        """Commit store. Pickle: serialize and send. Shm: notify server."""
        ...

    @abstractmethod
    def prepare_retrieve(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> list[torch.Tensor] | None:
        """Prepare retrieve. Returns chunks or shm views, or None on miss."""
        ...

    @abstractmethod
    def commit_retrieve(self, key: IPCCacheServerKey, instance_id: int) -> bool:
        """Commit retrieve. Pickle: no-op. Shm: release read locks."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this context."""
        ...


def create_engine_driven_context(
    metadata: EngineDrivenContextMetadata,
    req_client: RequestClient,
    mq_timeout: float,
    shm_name: str,
    pool_size: int,
    *,
    use_pickle: bool = False,
    scratch_offset: int = 0,
    scratch_size: int = 0,
) -> EngineDrivenContext:
    """Factory that returns the appropriate :class:`EngineDrivenContext` implementation.

    Returns SHM-based implementation when shared-memory pool information is
    available; otherwise falls back to the pickle-based implementation.
    If SHM initialization fails for any reason (e.g. segment not found,
    permission error), gracefully falls back to pickle transport.

    Args:
        metadata: Layout metadata for the non-GPU context.
        req_client: Transport-neutral client used for server requests.
        mq_timeout: Timeout in seconds for blocking MQ requests.
        shm_name: Shared-memory segment name. Empty values force pickle mode.
        pool_size: Shared-memory pool size in bytes. Non-positive values force
            pickle mode.
        use_pickle: Explicitly use pickle transport even when SHM info is
            available.

    Returns:
        A concrete :class:`EngineDrivenContext` instance.
    """
    if not shm_name or pool_size <= 0:
        use_pickle = True

    if not use_pickle:
        # Local
        from .shm import EngineDrivenContextShm

        try:
            logger.info(
                "Creating EngineDrivenContextShm (shm_name=%s, pool_size=%d)",
                shm_name,
                pool_size,
            )
            return EngineDrivenContextShm(
                metadata,
                req_client,
                mq_timeout,
                shm_name,
                pool_size,
                scratch_offset,
                scratch_size,
            )
        except Exception:
            logger.warning(
                "Failed to initialize SHM context (shm_name=%s), "
                "falling back to pickle transport",
                shm_name,
                exc_info=True,
            )

    # Local
    from .pickle import EngineDrivenContextPickle

    logger.info("Creating EngineDrivenContextPickle (pickle transport)")
    return EngineDrivenContextPickle(metadata, req_client, mq_timeout)


# ---------------------------------------------------------------------------
# Shared gather / scatter utilities
# ---------------------------------------------------------------------------


def compute_kv_layout(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None = None,
) -> tuple[int, int, int, str, "lmcache_native.EngineKVFormat", int]:
    """Compute KV layout metadata from KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        layout_hints: Optional engine layout hints.

    Returns:
        Tuple of ``(block_size, num_layers, hidden_dim_size, dtype_str,``
        ``engine_kv_format, kv_size)``. ``hidden_dim_size`` is the per-token
        object width (``num_heads * head_size``; for fused-K/V formats the
        head size is the packed ``2 * head_size``) and ``kv_size`` is the
        object plane count (1 for MLA and fused-K/V, else 2).

    Raises:
        ValueError: If ``kv_caches`` is empty.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_dtype,
        get_head_size,
        get_kv_size,
        get_num_heads,
        get_num_layers,
        normalize_kv_and_discover_format,
    )

    tensors = list(kv_caches.values())
    if not tensors:
        raise ValueError("kv_caches is empty. Cannot compute KV layout.")

    engine_kv_format, normalized = normalize_kv_and_discover_format(
        tensors, EngineType.VLLM, layout_hints=layout_hints
    )
    block_size = get_block_size(normalized, engine_kv_format)
    num_layers = get_num_layers(normalized, engine_kv_format)
    hidden_dim_size = get_num_heads(normalized, engine_kv_format) * get_head_size(
        normalized, engine_kv_format
    )
    kv_size = get_kv_size(normalized, engine_kv_format)
    dtype_str = str(get_dtype(normalized, engine_kv_format)).replace("torch.", "")
    return (
        block_size,
        num_layers,
        hidden_dim_size,
        dtype_str,
        engine_kv_format,
        kv_size,
    )


def _validate_raw_transfer(
    buffers: list[torch.Tensor],
    expected_numel: int,
    expected_dtype: torch.dtype,
    block_ids: list[int],
    num_blocks: int,
) -> None:
    """Validate raw-pointer transfer extents before any device work is queued."""
    for i, buffer in enumerate(buffers):
        if (
            buffer.device.type != "cpu"
            or not buffer.is_contiguous()
            or buffer.dtype != expected_dtype
            or buffer.numel() != expected_numel
        ):
            raise ValueError(
                f"Raw transfer buffer {i} must be contiguous CPU storage with "
                f"{expected_numel} elements of {expected_dtype} "
                f"({expected_numel * expected_dtype.itemsize} bytes); got "
                f"shape={tuple(buffer.shape)}, dtype={buffer.dtype}, "
                f"device={buffer.device}, contiguous={buffer.is_contiguous()}, "
                f"bytes={buffer.numel() * buffer.element_size()}"
            )
    for i, block_id in enumerate(block_ids):
        if block_id < 0 or block_id >= num_blocks:
            raise ValueError(
                f"Raw transfer block ID {block_id} at index {i} is outside "
                f"[0, {num_blocks})"
            )


@dataclass(frozen=True)
class PagedFusedSource:
    """Paged fused-K/V KV tensors described for in-place quantization.

    Attributes:
        layer_ptrs: int64 vector on the accelerator, one base pointer per layer.
        block_size: Tokens per paged block.
        block_stride_elems: Elements between consecutive blocks.
        num_blocks: Blocks per paged tensor (valid id range ``[0, num_blocks)``).
        dtype: Paged KV dtype.
        hidden: Fused content width (``num_heads * packed_head_size``).
        device: Accelerator holding the paged KV.
    """

    layer_ptrs: torch.Tensor
    block_size: int
    block_stride_elems: int
    num_blocks: int
    dtype: torch.dtype
    hidden: int
    device: torch.device


@dataclass(frozen=True)
class PagedMambaSource:
    """Paged Mamba state pages described for in-place quantization.

    Attributes:
        layer_ptrs: int64 vector on the accelerator, one base pointer per layer.
        page_stride_bytes: Bytes between consecutive pages (blocks).
        page_bytes: Bytes of one page's payload (both planes).
        num_blocks: Pages per layer tensor (valid id range ``[0, num_blocks)``).
        block_size: Tokens per page (the physical block size).
        device: Accelerator holding the paged state.
    """

    layer_ptrs: torch.Tensor
    page_stride_bytes: int
    page_bytes: int
    num_blocks: int
    block_size: int
    device: torch.device


_PAGED_FUSED_SOURCE_CACHE_LOCK = threading.Lock()
_PAGED_FUSED_SOURCE_CACHE: dict[tuple[int, ...], PagedFusedSource | None] = {}


def prepare_paged_fused_source(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None = None,
) -> PagedFusedSource | None:
    """Describe a group's paged fused-K/V KV tensors, or ``None`` if ineligible.

    Eligible only for XPU tensors in the blocks-first fused NHD format
    (``NL_X_NB_BS_NH_CS``), contiguous per layer with identical shape/dtype.
    Results are cached by layer pointers, which are fixed for a registered group.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_group_data_ptrs,
        normalize_kv_and_discover_format,
    )

    tensors = list(kv_caches.values())
    if not tensors or tensors[0].device.type != "xpu":
        return None
    # Pointers alone can be reused by a later allocation with a different
    # geometry, so the key also pins shape, strides and dtype.
    first = tensors[0]
    key = (
        *(int(t.data_ptr()) for t in tensors),
        hash((tuple(first.shape), tuple(first.stride()), first.dtype)),
    )
    with _PAGED_FUSED_SOURCE_CACHE_LOCK:
        if key in _PAGED_FUSED_SOURCE_CACHE:
            return _PAGED_FUSED_SOURCE_CACHE[key]
    result: PagedFusedSource | None = None
    try:
        engine_kv_format, normalized = normalize_kv_and_discover_format(
            tensors, EngineType.VLLM, layout_hints=layout_hints
        )
        if engine_kv_format == lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS:
            # vLLM may hand over a permuted view of NHD pages; only the
            # normalized view has the (block, token, head, cs) shape/strides.
            tensors = list(normalized)
            first = tensors[0]
            num_blocks, block_size, num_heads, cs = (int(d) for d in first.shape)
            expected_stride = (block_size * num_heads * cs, num_heads * cs, cs, 1)
            if all(
                t.dim() == 4
                and t.shape == first.shape
                and t.dtype == first.dtype
                and tuple(t.stride()) == expected_stride
                for t in tensors
            ):
                layer_ptrs = _cached_xpu_long_tensor(
                    get_group_data_ptrs(
                        normalized, engine_kv_format, list(range(len(tensors)))
                    ),
                    first.device,
                    "layer_ptrs",
                )
                result = PagedFusedSource(
                    layer_ptrs=layer_ptrs,
                    block_size=block_size,
                    block_stride_elems=expected_stride[0],
                    num_blocks=num_blocks,
                    dtype=first.dtype,
                    hidden=num_heads * cs,
                    device=first.device,
                )
    except (ValueError, RuntimeError, TypeError, IndexError):
        logger.warning(
            "Could not describe paged fused KV for in-place quantization; "
            "falling back to gather.",
            exc_info=True,
        )
    with _PAGED_FUSED_SOURCE_CACHE_LOCK:
        _PAGED_FUSED_SOURCE_CACHE[key] = result
    return result


def paged_fused_rejection_reason(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None = None,
) -> str:
    """Explain why :func:`prepare_paged_fused_source` rejects a group's KV.

    Returns a short human-readable reason, or ``"layout is eligible"`` when the
    tensors themselves pass (so any rejection comes from the codec's geometry).
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format

    tensors = list(kv_caches.values())
    if not tensors:
        return "no KV tensors"
    first = tensors[0]
    if first.device.type != "xpu":
        return f"KV tensors are on {first.device.type}, not xpu"
    try:
        engine_kv_format, normalized = normalize_kv_and_discover_format(
            tensors, EngineType.VLLM, layout_hints=layout_hints
        )
    except (ValueError, RuntimeError, TypeError, IndexError) as exc:
        return f"KV format discovery failed: {exc!r}"
    if engine_kv_format != lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS:
        return f"KV format is {engine_kv_format}, need NL_X_NB_BS_NH_CS"
    tensors = list(normalized)
    first = tensors[0]
    if first.dim() != 4:
        return f"per-layer KV has {first.dim()} dims, need 4"
    num_blocks, block_size, num_heads, cs = (int(d) for d in first.shape)
    expected = (block_size * num_heads * cs, num_heads * cs, cs, 1)
    for t in tensors:
        if t.shape != first.shape or t.dtype != first.dtype:
            return "layers differ in shape or dtype"
        if tuple(t.stride()) != expected:
            return (
                f"per-layer KV stride {tuple(t.stride())} != contiguous {expected} "
                f"(shape {tuple(t.shape)}, e.g. blocks padded to a shared page size)"
            )
    return "layout is eligible"


_PAGED_MAMBA_SOURCE_CACHE: dict[tuple[int, ...], PagedMambaSource | None] = {}


def prepare_paged_mamba_source(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None = None,
) -> PagedMambaSource | None:
    """Describe a Mamba group's paged state pages, or ``None`` if ineligible.

    Eligible only for XPU tensors in the two-plane page format
    (``NL_X_NB_TWO_BS_NH_HS``) whose pages are contiguous with identical
    geometry on every layer. Results are cached by layer pointers and geometry.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_group_data_ptrs,
        normalize_kv_and_discover_format,
    )

    tensors = list(kv_caches.values())
    if not tensors or tensors[0].device.type != "xpu":
        return None
    first = tensors[0]
    key = (
        *(int(t.data_ptr()) for t in tensors),
        hash((tuple(first.shape), tuple(first.stride()), first.dtype)),
    )
    with _PAGED_FUSED_SOURCE_CACHE_LOCK:
        if key in _PAGED_MAMBA_SOURCE_CACHE:
            return _PAGED_MAMBA_SOURCE_CACHE[key]
    result: PagedMambaSource | None = None
    try:
        engine_kv_format, normalized = normalize_kv_and_discover_format(
            tensors, EngineType.VLLM, layout_hints=layout_hints
        )
        if engine_kv_format == lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS:
            num_blocks, planes, block_size, num_heads, head_size = (
                int(d) for d in first.shape
            )
            row = num_heads * head_size
            inner = (block_size * row, row, head_size, 1)
            if (
                planes == 2
                and all(
                    t.dim() == 5
                    and t.shape == first.shape
                    and t.dtype == first.dtype
                    and tuple(t.stride()[1:]) == inner
                    and t.stride(0) == first.stride(0)
                    and t.stride(0) >= 2 * block_size * row
                    for t in tensors
                )
            ):
                itemsize = first.element_size()
                result = PagedMambaSource(
                    layer_ptrs=_cached_xpu_long_tensor(
                        get_group_data_ptrs(
                            normalized, engine_kv_format, list(range(len(tensors)))
                        ),
                        first.device,
                        "layer_ptrs",
                    ),
                    page_stride_bytes=int(first.stride(0)) * itemsize,
                    page_bytes=2 * block_size * row * itemsize,
                    num_blocks=num_blocks,
                    block_size=block_size,
                    device=first.device,
                )
    except (ValueError, RuntimeError, TypeError, IndexError):
        logger.warning(
            "Could not describe paged Mamba state for in-place quantization; "
            "falling back to gather.",
            exc_info=True,
        )
    with _PAGED_FUSED_SOURCE_CACHE_LOCK:
        _PAGED_MAMBA_SOURCE_CACHE[key] = result
    return result


def paged_block_ids_tensor(block_ids: list[int], device: torch.device) -> torch.Tensor:
    """Upload one chunk's block ids to the accelerator (cached by value)."""
    return _cached_xpu_long_tensor(block_ids, device, "block_ids")


@dataclass(frozen=True)
class _BlockGeometry:
    """Paged-KV geometry shared by gather and scatter."""

    engine_kv_format: "lmcache_native.EngineKVFormat"
    normalized: Any
    block_size: int
    num_layers: int
    num_blocks: int
    content_size: int
    kv_size: int
    dtype: torch.dtype
    shape_desc: Any
    device: torch.device


def _block_geometry(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None,
    engine_kv_format: "lmcache_native.EngineKVFormat" | None,
) -> _BlockGeometry:
    """Detect a group's KV format and derive the geometry block transfers need."""
    # First Party
    from lmcache.v1.gpu_connector.utils import (
        get_block_size,
        get_device,
        get_dtype,
        get_head_size,
        get_kv_size,
        get_num_blocks,
        get_num_heads,
        get_num_layers,
        make_page_buffer_shape_desc,
        normalize_kv_and_discover_format,
    )

    fmt, normalized = normalize_kv_and_discover_format(
        list(kv_caches.values()), EngineType.VLLM, layout_hints=layout_hints
    )
    if engine_kv_format is None:
        engine_kv_format = fmt
    block_size = get_block_size(normalized, engine_kv_format)
    num_layers = get_num_layers(normalized, engine_kv_format)
    num_blocks = get_num_blocks(normalized, engine_kv_format)
    return _BlockGeometry(
        engine_kv_format=engine_kv_format,
        normalized=normalized,
        block_size=block_size,
        num_layers=num_layers,
        num_blocks=num_blocks,
        content_size=get_num_heads(normalized, engine_kv_format)
        * get_head_size(normalized, engine_kv_format),
        kv_size=get_kv_size(normalized, engine_kv_format),
        dtype=get_dtype(normalized, engine_kv_format),
        shape_desc=make_page_buffer_shape_desc(
            normalized,
            engine_kv_format,
            layer_idx=0,
            num_layers_in_group=num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
        ),
        device=get_device(normalized),
    )


def _run_ptr_block_transfer(
    geometry: _BlockGeometry,
    direction: "lmcache_native.TransferDirection",
    chunks: list[torch.Tensor],
    selected_block_ids: list[int],
    blocks_per_chunk: int,
    skip_prefix_n_blocks: int = 0,
) -> None:
    """Run a block transfer through the compiled pointer interface.

    Chunks go through in batches of at most four objects (the CUDA kernel's
    limit); each batch gets its own slice of the block ids and chunks. A block
    prefix to skip applies to the first batch only.
    """
    # First Party
    from lmcache.v1.gpu_connector.utils import get_group_data_ptrs

    device = geometry.device
    layer_ptrs = get_group_data_ptrs(
        geometry.normalized,
        geometry.engine_kv_format,
        list(range(geometry.num_layers)),
    )
    if device.type == "xpu":
        paged_arg = _cached_xpu_long_tensor(layer_ptrs, device, "layer_ptrs")
        # The XPU native binding consumes block IDs on the host and expands them
        # into device slot mappings; uploading them here would only add a
        # device-to-host round trip inside the binding.
        block_ids_arg = torch.as_tensor(selected_block_ids, dtype=torch.int64)
    else:
        ptrs = np.array(layer_ptrs, dtype=np.uint64).view(np.int64)
        paged_arg = torch.from_numpy(ptrs).to(device=device)
        block_ids_arg = torch.tensor(
            selected_block_ids, dtype=torch.int64, device=device
        )
    objs_arg = _tensors_to_ptrs(chunks)
    chunk_tokens = blocks_per_chunk * geometry.block_size
    max_objects = 4
    for i in range(0, len(chunks), max_objects):
        _multi_layer_block_kv_transfer(
            paged_arg,
            objs_arg[i : i + max_objects],
            block_ids_arg[
                i * blocks_per_chunk : min(
                    (i + max_objects) * blocks_per_chunk, len(selected_block_ids)
                )
            ],
            device,
            direction,
            geometry.shape_desc,
            chunk_tokens,
            geometry.engine_kv_format,
            skip_prefix_n_blocks if i == 0 else 0,
            fallback_paged_buffer=geometry.normalized,
            fallback_objects=chunks[i : i + max_objects],
        )


def gather_paged_kv_to_cpu(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    blocks_per_chunk: int,
    layout_hints: LayoutHints | None = None,
    engine_kv_format: "lmcache_native.EngineKVFormat" | None = None,
    out: list[torch.Tensor] | None = None,
    chunk_indices: list[int] | None = None,
) -> list[torch.Tensor]:
    """Gather paged KV blocks into CPU chunk tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping.
        block_ids: Flattened block IDs for all chunks.
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        layout_hints: Optional engine layout hints.
        engine_kv_format: Optional pre-detected KV format.
        out: Optional pre-allocated output tensors.  If provided, length
            must be at least ``len(chunk_indices)`` when ``chunk_indices``
            is given, or the total number of chunks otherwise.  Any extra
            buffers beyond the number of gathered chunks are ignored.
        chunk_indices: Optional list of chunk positions (into the full
            ``block_ids`` sequence) to gather.  When provided together with
            ``out``, only those chunks are gathered and written into
            ``out[i]`` in order.  When ``None``, all chunks are gathered
            (backward-compatible behaviour).

    Returns:
        List of CPU tensors, one per chunk. For split-K/V formats each chunk
        has shape ``[2, num_layers, chunk_tokens, hidden_dim]`` where
        dimension ``0`` stores ``(K, V)``. For single-plane formats
        (``kv_size == 1``: MLA and fused-K/V) each chunk has shape
        ``[num_layers, chunk_tokens, num_heads * head_size]`` — for fused
        formats ``head_size`` is the packed ``2 * head_size``.

    Raises:
        ValueError: If ``out`` has too few buffers, an incompatible dtype,
            size or memory layout, or selected chunks/block IDs are out of
            bounds, or ``blocks_per_chunk`` is not positive.
    """
    # First Party
    from lmcache import device_ops

    if blocks_per_chunk <= 0:
        raise ValueError("blocks_per_chunk must be positive")
    geometry = _block_geometry(kv_caches, layout_hints, engine_kv_format)
    normalized = geometry.normalized
    engine_kv_format = geometry.engine_kv_format
    num_layers = geometry.num_layers
    content_size = geometry.content_size
    num_blocks = geometry.num_blocks
    num_chunks = len(block_ids) // blocks_per_chunk
    chunk_tokens = blocks_per_chunk * geometry.block_size
    shape_desc = geometry.shape_desc

    iter_indices = (
        list(chunk_indices) if chunk_indices is not None else list(range(num_chunks))
    )
    # Require at least one output buffer per gathered chunk. Extra trailing
    # buffers are ignored (see ``chunks = out[: len(iter_indices)]`` below),
    # mirroring the scatter-side length check for consistency.
    if out is not None and len(out) < len(iter_indices):
        raise ValueError(
            f"out length ({len(out)}) must be at least the number of "
            f"gathered chunks ({len(iter_indices)})"
        )

    selected_block_ids: list[int] = []
    for chunk_idx in iter_indices:
        if chunk_idx < 0 or chunk_idx >= num_chunks:
            raise ValueError(
                f"Gather chunk index {chunk_idx} is outside [0, {num_chunks})"
            )
        selected_block_ids.extend(
            block_ids[chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk]
        )
    _validate_raw_transfer(
        out[: len(iter_indices)] if out is not None else [],
        geometry.kv_size * num_layers * chunk_tokens * content_size,
        geometry.dtype,
        selected_block_ids,
        num_blocks,
    )

    # Determine if pinned memory is strictly required
    # (only for the compiled C++ path which does not accept tensor)
    requires_pinned = not _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR
    needs_staging = False
    staged_chunks = []
    staged_pinned_ptrs: list[int] = []
    staged_pinned_buffers: list[object] = []
    staged_pinned_sizes: list[int] = []

    if out is None:
        # One object plane per K/V entry: MLA and fused-K/V formats
        # (kv_size == 1) store a single plane, split formats store K and V.
        if geometry.kv_size == 1:
            chunks = [
                torch.empty(
                    (num_layers, chunk_tokens, content_size),
                    dtype=geometry.dtype,
                    device=torch.device("cpu"),
                    pin_memory=requires_pinned,
                )
                for _ in iter_indices
            ]
        else:
            chunks = [
                torch.empty(
                    (2, num_layers, chunk_tokens, content_size),
                    dtype=geometry.dtype,
                    device=torch.device("cpu"),
                    pin_memory=requires_pinned,
                )
                for _ in iter_indices
            ]
    else:
        _target_out = out[: len(iter_indices)]

        if requires_pinned and not all(t.is_pinned() for t in _target_out):
            # Core fallback: Unpinned memory (e.g., IPC Shared Memory) detected.
            # We cannot dynamically call `.pin_memory()` on `out` because it
            # would allocate new tensors, breaking the caller's expectation
            # of an in-place update. Instead, we allocate a temporary pinned
            # staging buffer for the C++ kernel to write to safely.
            logger.debug(
                "Unpinned memory detected in 'out' during "
                "gather_paged_kv_to_cpu (likely Shared Memory). "
                "Using an internal pinned staging buffer, which "
                "adds a CPU memory copy overhead."
            )
            needs_staging = True
            if torch_device_type == "xpu" and not _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR:
                staged_chunks = []
                for target in _target_out:
                    nbytes = int(target.numel()) * target.element_size()
                    with _XPU_H2D_USM_POOL_LOCK:
                        bucket = _XPU_H2D_USM_POOL.get(nbytes)
                        pooled = bucket.pop() if bucket else None
                    if pooled is None:
                        ptr = device_ops.alloc_pinned_ptr(nbytes)
                        buffer_type = ctypes.c_uint8 * nbytes
                        buffer = buffer_type.from_address(ptr)
                    else:
                        ptr, buffer = pooled
                    staged_pinned_ptrs.append(ptr)
                    staged_pinned_buffers.append(buffer)
                    staged_pinned_sizes.append(nbytes)
                    staged_chunks.append(
                        torch.frombuffer(buffer, dtype=target.dtype).view(target.shape)
                    )
            else:
                staged_chunks = [
                    torch.empty_like(t, pin_memory=True) for t in _target_out
                ]
            chunks = (
                staged_chunks  # Point to the safe staging buffer for the H2D transfer
            )
        else:
            # Ideal case: Memory is pinned, or we are using Python fallback.
            # Ignore any extra trailing buffers beyond what we actually gather so
            # the kernel's ``total_blocks % num_objects`` invariant still holds.
            # Return ``out`` unchanged when no trimming is needed so the in-place
            # fill contract (result is out) is preserved.
            if len(out) == len(iter_indices):
                chunks = out
            else:
                chunks = out[: len(iter_indices)]

    if selected_block_ids:
        if _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR:
            # Python fallback: accepts tensor list directly for all params.
            _multi_layer_block_kv_transfer(
                normalized,
                chunks,
                selected_block_ids,
                geometry.device,
                lmcache_native.TransferDirection.D2H,
                shape_desc,
                chunk_tokens,
                engine_kv_format,
                0,
                fallback_paged_buffer=normalized,
                fallback_objects=chunks,
            )
        else:
            # Compiled C++/CUDA/XPU: pointer interface; chunks are either the
            # caller's pinned buffers or the staged copies above.
            _run_ptr_block_transfer(
                geometry,
                lmcache_native.TransferDirection.D2H,
                chunks,
                selected_block_ids,
                blocks_per_chunk,
            )

    # --- Final reconciliation ---
    # If we used a staging buffer to protect unpinned shared memory,
    # we now copy the gathered data back into the caller's original tensors.
    if needs_staging:
        assert out is not None
        # The CPU MUST block and wait for the GPU ONLY when a temporary
        # staging buffer is used. This is because the CPU needs to immediately
        # read this data for the memory copy below.
        torch_dev.synchronize()

        for dst, src in zip(_target_out, staged_chunks, strict=False):
            dst.copy_(src)  # High-speed CPU-to-CPU memory copy

        if len(out) == len(iter_indices):
            chunks = out
        else:
            chunks = _target_out

        for nbytes, ptr, buffer in zip(
            staged_pinned_sizes,
            staged_pinned_ptrs,
            staged_pinned_buffers,
            strict=True,
        ):
            _return_xpu_usm_buffer(nbytes, ptr, buffer)
        staged_pinned_ptrs.clear()
        staged_pinned_buffers.clear()
        staged_pinned_sizes.clear()
    # Fast path: The async GPU copy might still be in progress.
    # We intentionally omit synchronization here for performance.
    # WARNING: The caller MUST explicitly call `torch_dev.synchronize()`
    # before consuming these chunks to ensure data validity.

    return chunks


def scatter_cpu_to_paged_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: LayoutHints | None = None,
    engine_kv_format: "lmcache_native.EngineKVFormat" | None = None,
) -> None:
    """Scatter CPU chunk tensors back into paged KV tensors.

    Args:
        kv_caches: Per-layer KV tensor mapping to write into.
        block_ids: Flattened destination block IDs for all chunks.  Length
            must be at least ``len(chunks) * blocks_per_chunk``; any extra
            trailing block IDs are ignored.
        chunks: List of CPU chunk tensors (as returned by
            :func:`gather_paged_kv_to_cpu`).
        blocks_per_chunk: Number of paged blocks in one LMCache chunk.
        skip_first_n_tokens: Token prefix to skip when scattering.  Must be a
            multiple of ``block_size``; non-aligned values are rounded down
            to the nearest whole block and an error is logged (matching the
            GPU transfer path).
        layout_hints: Optional engine layout hints.
        engine_kv_format: Optional pre-detected KV format.

    Raises:
        ValueError: If ``block_ids`` is shorter than
            ``len(chunks) * blocks_per_chunk``, a block ID is out of bounds,
            a chunk's dtype, size or memory layout is incompatible, or
            ``blocks_per_chunk`` is not positive.
    """
    if not chunks:
        return
    if blocks_per_chunk <= 0:
        raise ValueError("blocks_per_chunk must be positive")
    # Require enough block IDs to cover every chunk. Extra trailing block IDs
    # are ignored by the per-chunk slicing below, mirroring the gather-side
    # ``out`` length check for consistency.
    if len(block_ids) < len(chunks) * blocks_per_chunk:
        raise ValueError(
            f"block_ids length ({len(block_ids)}) must be at least "
            f"len(chunks) ({len(chunks)}) * blocks_per_chunk "
            f"({blocks_per_chunk})"
        )

    geometry = _block_geometry(kv_caches, layout_hints, engine_kv_format)
    normalized = geometry.normalized
    engine_kv_format = geometry.engine_kv_format
    block_size = geometry.block_size
    num_layers = geometry.num_layers
    num_blocks = geometry.num_blocks
    chunk_tokens = blocks_per_chunk * block_size

    # Block-level transfer can only skip whole blocks. A non-aligned prefix is
    # rounded down to the nearest block (matching the lmcache-driven path in
    # lmcache_driven_transfer.py) rather than raising, so a slightly misaligned skip
    # degrades gracefully instead of failing the whole retrieve.
    if skip_first_n_tokens % block_size != 0:
        logger.error(
            "skip_first_n_tokens (%d) is not block-aligned (block_size=%d); "
            "rounding down to %d blocks",
            skip_first_n_tokens,
            block_size,
            skip_first_n_tokens // block_size,
        )
    skip_prefix_n_blocks = skip_first_n_tokens // block_size

    shape_desc = geometry.shape_desc

    selected_block_ids: list[int] = []
    for chunk_idx in range(len(chunks)):
        selected_block_ids.extend(
            block_ids[chunk_idx * blocks_per_chunk : (chunk_idx + 1) * blocks_per_chunk]
        )

    if not selected_block_ids:
        return

    _validate_raw_transfer(
        chunks,
        geometry.kv_size * num_layers * chunk_tokens * geometry.content_size,
        geometry.dtype,
        selected_block_ids,
        num_blocks,
    )

    # Only the ptr-only branch below pins temporaries; the tensor branch
    # hands torch the chunks directly and keeps them alive itself.
    dynamically_pinned = False
    if _LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR:
        # Python fallback: accepts tensor list directly for all params.
        _multi_layer_block_kv_transfer(
            normalized,
            chunks,
            selected_block_ids,
            geometry.device,
            lmcache_native.TransferDirection.H2D,
            shape_desc,
            chunk_tokens,
            engine_kv_format,
            skip_prefix_n_blocks,
            fallback_paged_buffer=normalized,
            fallback_objects=chunks,
        )
    else:
        # assuming this is c ops path which requires pin memory
        # TODO: may have a better approach here
        # Defensive check: Ensure all incoming CPU chunks are pinned memory.
        # Otherwise, the underlying CUDA kernel may throw an Illegal
        # Memory Access error during H2D transfer.
        dynamically_pinned = not all(chunk.is_pinned() for chunk in chunks)
        if dynamically_pinned:
            logger.debug(
                "Received unpinned CPU tensors in scatter_cpu_to_paged_kv. "
                "Dynamically pinning memory now, which may incur additional"
                "synchronization overhead."
            )
            chunks = [
                chunk.pin_memory() if not chunk.is_pinned() else chunk
                for chunk in chunks
            ]

        _run_ptr_block_transfer(
            geometry,
            lmcache_native.TransferDirection.H2D,
            chunks,
            selected_block_ids,
            blocks_per_chunk,
            skip_prefix_n_blocks,
        )

    # Fast path: The async GPU copy might still be in progress.
    # We intentionally omit synchronization here for performance.
    # WARNING: The caller MUST explicitly call `torch_dev.synchronize()`
    # before consuming these chunks to ensure data validity.

    if dynamically_pinned:
        # The temporaries pinned above are read by the async H2D launches
        # through raw pointers, which torch's stream tracking cannot see.
        # Returning drops the last reference to them, and the caching host
        # allocator then hands that memory to the next caller while the
        # copies are still in flight -- the next pin_memory() overwrites the
        # bytes mid-transfer. The caller-side synchronize documented above
        # cannot cover this: by then the temporaries are already gone.
        # Only the dynamically pinned case pays this; caller-owned pinned
        # chunks keep the fast path.
        torch_dev.synchronize()

    if torch_device_type == "xpu" and not dynamically_pinned:
        torch_dev.synchronize()
