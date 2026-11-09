# SPDX-License-Identifier: Apache-2.0
"""Descriptions of paged KV tensors for in-place KVWeave quantization.

The native codec can quantize straight from paged KV and dequantize straight into
it, which removes the dense gather/scatter and its host staging. It needs the
tensors described as raw device pointers plus block geometry; this module builds
that description. XPU only: the kernels run on the device that holds the KV.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import threading

# Third Party
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_group_data_ptrs,
    normalize_kv_and_discover_format,
)
import lmcache.lmcache_native as lmcache_native

logger = init_logger(__name__)

# Block-id tensors are tiny but uploaded per chunk; reuse them by value.
_BLOCK_IDS_CACHE_LIMIT = 256
_block_ids_cache_lock = threading.Lock()
_block_ids_cache: dict[tuple[object, ...], torch.Tensor] = {}


@dataclass(frozen=True)
class PagedFusedSource:
    """Paged fused-K/V KV tensors described for in-place quantization.

    Attributes:
        layer_ptrs: int64 vector on the accelerator, one base pointer per layer.
        layer_key: The layers' ``data_ptr()`` values at description time, used to
            detect that the engine reallocated its KV tensors.
        block_size: Tokens per paged block.
        block_stride_elems: Elements between consecutive blocks.
        num_blocks: Blocks per paged tensor (valid ids are ``[0, num_blocks)``).
        dtype: Paged KV dtype.
        hidden: Fused content width (``num_heads * packed_head_size``).
        device: Accelerator holding the paged KV.
    """

    layer_ptrs: torch.Tensor
    layer_key: tuple[int, ...]
    block_size: int
    block_stride_elems: int
    num_blocks: int
    dtype: torch.dtype
    hidden: int
    device: torch.device


def prepare_paged_fused_source(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None = None,
) -> PagedFusedSource | None:
    """Describe a group's paged fused-K/V tensors, or ``None`` if ineligible.

    Eligible only for XPU tensors in the blocks-first fused NHD format
    (``NL_X_NB_BS_NH_CS``), contiguous per layer with identical shape and dtype.
    Use :func:`paged_fused_rejection_reason` to learn why a group is rejected.

    Args:
        kv_caches: This group's KV tensors keyed by layer name.
        layout_hints: Optional engine layout hints.

    Returns:
        The description, or ``None`` when the group cannot be handled in place.
    """
    tensors = list(kv_caches.values())
    if not tensors or tensors[0].device.type != "xpu":
        return None
    try:
        engine_kv_format, normalized = normalize_kv_and_discover_format(
            tensors, EngineType.VLLM, layout_hints=layout_hints
        )
    except (ValueError, RuntimeError, TypeError, IndexError):
        logger.warning(
            "Could not describe paged fused KV for in-place quantization; "
            "falling back to gather.",
            exc_info=True,
        )
        return None
    if engine_kv_format != lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS:
        return None
    # vLLM may hand over a permuted view of NHD pages; only the normalized view has
    # the (block, token, head, cs) shape and strides.
    views = list(normalized)
    first = views[0]
    if first.dim() != 4:
        return None
    num_blocks, block_size, num_heads, cs = (int(d) for d in first.shape)
    expected_stride = (block_size * num_heads * cs, num_heads * cs, cs, 1)
    if not all(
        view.shape == first.shape
        and view.dtype == first.dtype
        and tuple(view.stride()) == expected_stride
        for view in views
    ):
        return None
    return PagedFusedSource(
        layer_ptrs=_layer_ptrs_tensor(
            get_group_data_ptrs(
                normalized, engine_kv_format, list(range(len(views)))
            ),
            first.device,
        ),
        layer_key=tuple(int(t.data_ptr()) for t in tensors),
        block_size=block_size,
        block_stride_elems=expected_stride[0],
        num_blocks=num_blocks,
        dtype=first.dtype,
        hidden=num_heads * cs,
        device=first.device,
    )


def paged_fused_rejection_reason(
    kv_caches: dict[str, torch.Tensor],
    layout_hints: LayoutHints | None = None,
) -> str:
    """Explain why :func:`prepare_paged_fused_source` rejects a group's KV.

    Args:
        kv_caches: This group's KV tensors keyed by layer name.
        layout_hints: Optional engine layout hints.

    Returns:
        A short reason, or ``"layout is eligible"`` when the tensors themselves
        pass (any rejection then comes from the codec's geometry check).
    """
    tensors = list(kv_caches.values())
    if not tensors:
        return "no KV tensors"
    if tensors[0].device.type != "xpu":
        return f"KV tensors are on {tensors[0].device.type}, not xpu"
    try:
        engine_kv_format, normalized = normalize_kv_and_discover_format(
            tensors, EngineType.VLLM, layout_hints=layout_hints
        )
    except (ValueError, RuntimeError, TypeError, IndexError) as exc:
        return f"KV format discovery failed: {exc!r}"
    if engine_kv_format != lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_CS:
        return f"KV format is {engine_kv_format}, need NL_X_NB_BS_NH_CS"
    views = list(normalized)
    first = views[0]
    if first.dim() != 4:
        return f"per-layer KV has {first.dim()} dims, need 4"
    num_blocks, block_size, num_heads, cs = (int(d) for d in first.shape)
    expected = (block_size * num_heads * cs, num_heads * cs, cs, 1)
    for view in views:
        if view.shape != first.shape or view.dtype != first.dtype:
            return "layers differ in shape or dtype"
        if tuple(view.stride()) != expected:
            return (
                f"per-layer KV stride {tuple(view.stride())} != contiguous "
                f"{expected} (shape {tuple(view.shape)}, e.g. blocks padded to a "
                "shared page size)"
            )
    return "layout is eligible"


def paged_block_ids_tensor(block_ids: list[int], device: torch.device) -> torch.Tensor:
    """Upload one chunk's block ids to the accelerator, reusing equal uploads.

    Args:
        block_ids: The chunk's engine block ids.
        device: Accelerator the paged KV lives on.

    Returns:
        An int64 tensor on ``device``; do not modify it.
    """
    key = ("block_ids", device.type, device.index, *block_ids)
    with _block_ids_cache_lock:
        cached = _block_ids_cache.get(key)
    if cached is not None:
        return cached
    tensor = torch.tensor(block_ids, dtype=torch.int64, device=device)
    with _block_ids_cache_lock:
        if len(_block_ids_cache) >= _BLOCK_IDS_CACHE_LIMIT:
            _block_ids_cache.pop(next(iter(_block_ids_cache)))
        _block_ids_cache[key] = tensor
    return tensor


def _layer_ptrs_tensor(ptrs: list[int], device: torch.device) -> torch.Tensor:
    """Pack per-layer base pointers into an int64 device vector."""
    raw = np.asarray(ptrs, dtype=np.uint64).view(np.int64)
    return torch.from_numpy(raw).to(device=device)
