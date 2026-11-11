# SPDX-License-Identifier: Apache-2.0
"""Block-transfer helpers for backends whose compiled kernel takes addresses.

CUDA/ROCm (``lmcache.cuda_ops``), XPU (``lmcache.xpu_ops``) and Ascend
(``lmcache_ascend.ops``) ship a ``multi_layer_block_kv_transfer`` kernel that
takes a device tensor of per-layer base addresses, chunk addresses as
``list[int]`` and a device block-ID tensor. Its GPU threads read host chunks
directly, so they must be pinned, and one launch addresses at most four chunks.
Engine-driven gather and scatter keep their operands as tensors and call
:func:`transfer_with_address_kernel` when the resolved block kernel takes
addresses; it builds the kernel's operands and splits the launches.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import TYPE_CHECKING

# Third Party
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import (
    get_device_spec,
    resolve_device_ops,
    synchronize_device,
)
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.common.kv_format.types import DiscoverableKVCache

logger = init_logger(__name__)

#: Chunks one kernel launch can address (``TORCH_CHECK(num_objects <= 4)`` in
#: ``csrc/cuda/mp_mem_kernels.cu``).
MAX_CHUNKS_PER_LAUNCH = 4


def _paged_pointer_table(
    paged_kv: DiscoverableKVCache,
    engine_kv_format: lmcache_native.EngineKVFormat,
    num_layers: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the kernel-ordered per-layer base addresses of ``paged_kv``.

    Args:
        paged_kv: Normalized paged KV tensors.
        engine_kv_format: Layout of ``paged_kv``.
        num_layers: Number of layers to address.
        device: Device the kernel runs on.

    Returns:
        A 1-D int64 tensor on ``device`` with the addresses in the order
        ``KVFormatSpec.data_ptrs`` defines for the format.
    """
    # Imported lazily, not at module scope: the specs package reads
    # ``lmcache.device_ops``, which is resolved only once the backends that
    # import this module have been imported.
    # First Party
    from lmcache.v1.common.kv_format.specs.registry import get_spec

    addresses = get_spec(paged_kv, engine_kv_format).data_ptrs(list(range(num_layers)))
    # Addresses are unsigned; the kernel reads the same bits as int64.
    as_int64 = np.array(addresses, dtype=np.uint64).view(np.int64)
    return torch.from_numpy(as_int64).to(device=device)


#: Alignment passed to ``lmcache_memcpy_async`` for host chunks that are not
#: part of a registered pin pool; large enough that one chunk is one copy.
_HOST_COPY_ALIGNMENT = 1 << 30


def _all_registered(device: torch.device, chunks: list[torch.Tensor]) -> bool:
    """Whether every host chunk lies in memory registered for device DMA.

    Registered memory (XPU host registration) can be copied by the copy engine
    at the full link rate but cannot be addressed by a kernel, so it takes the
    device-scratch route. Pageable memory does not: its copies go through a
    driver bounce buffer and are slower than the pinned-staging fallback.

    Args:
        device: Device that owns the paged KV tensors.
        chunks: Host chunk tensors.

    Returns:
        True when the backend of ``device`` reports every chunk as registered.
    """
    spec = get_device_spec(device.type)
    if spec is None or not chunks:
        return False
    return all(
        spec.is_pinned_range(chunk.data_ptr(), chunk.numel() * chunk.element_size())
        for chunk in chunks
    )


def _transfer_via_device_scratch(
    kernel: Callable[..., None],
    paged_addresses: torch.Tensor,
    chunks: list[torch.Tensor],
    block_id_tensor: torch.Tensor,
    device: torch.device,
    direction: lmcache_native.TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: lmcache_native.EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """Run a block transfer whose host chunks are registered, not addressable.

    XPU host registration (``prepare_for_device_copy``) lets ``queue.memcpy``
    DMA to and from a range such as a shared-memory pool at the full link
    rate, but it does not make the range USM, so the kernel's threads cannot
    touch it. The kernel therefore works on a device scratch buffer and the
    copy engine moves each chunk between the scratch and the host chunk, which
    is about twice as fast as staging through pinned buffers and a CPU copy.

    The scratch holds one launch's chunks and is reused by every launch; the
    stream is in order, so a launch never overwrites chunks still being
    copied. The call synchronizes ``device`` before returning because the host
    chunks are reached through raw addresses that torch cannot track.

    Args:
        kernel: The compiled ``multi_layer_block_kv_transfer``.
        paged_addresses: Per-layer base-address table on ``device``.
        chunks: Host chunk tensors, all the same size, filled in place for
            ``D2H``.
        block_id_tensor: Paged-block IDs in chunk order, on ``device``.
        device: Device that owns the paged KV tensors.
        direction: ``D2H`` or ``H2D``.
        shape_desc: Paged-buffer shape descriptor.
        lmcache_chunk_size: Tokens per chunk.
        engine_kv_format: Engine KV layout.
        skip_prefix_n_blocks: Leading blocks to skip across the whole transfer.
    """
    memcpy_async = resolve_device_ops(device.type).lmcache_memcpy_async
    is_d2h = int(direction) == int(lmcache_native.TransferDirection.D2H)
    chunk_bytes = chunks[0].numel() * chunks[0].element_size()
    scratch = torch.empty(
        MAX_CHUNKS_PER_LAUNCH * chunk_bytes, dtype=torch.uint8, device=device
    )
    blocks_per_launch = MAX_CHUNKS_PER_LAUNCH * lmcache_chunk_size // int(shape_desc.bs)
    for launch, first in enumerate(range(0, len(chunks), MAX_CHUNKS_PER_LAUNCH)):
        batch = chunks[first : first + MAX_CHUNKS_PER_LAUNCH]
        slots = [scratch.data_ptr() + i * chunk_bytes for i in range(len(batch))]
        if not is_d2h:
            for slot, chunk in zip(slots, batch, strict=True):
                memcpy_async(
                    slot,
                    chunk.data_ptr(),
                    chunk_bytes,
                    direction,
                    0,
                    _HOST_COPY_ALIGNMENT,
                )
        start = launch * blocks_per_launch
        launch_block_ids = block_id_tensor[start : start + blocks_per_launch]
        kernel(
            paged_addresses,
            slots,
            launch_block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            max(0, min(skip_prefix_n_blocks - start, len(launch_block_ids))),
        )
        if is_d2h:
            for slot, chunk in zip(slots, batch, strict=True):
                memcpy_async(
                    chunk.data_ptr(),
                    slot,
                    chunk_bytes,
                    direction,
                    0,
                    _HOST_COPY_ALIGNMENT,
                )
    synchronize_device(device)


def transfer_with_address_kernel(
    kernel: Callable[..., None],
    paged_kv: DiscoverableKVCache,
    chunks: list[torch.Tensor],
    block_ids: list[int],
    device: torch.device | str,
    direction: lmcache_native.TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: lmcache_native.EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """Run a tensor block transfer on an address-taking compiled kernel.

    Operands follow ``multi_layer_block_kv_transfer``. The kernel receives the
    per-layer address table, chunk addresses and a device block-ID tensor, in
    launches of at most :data:`MAX_CHUNKS_PER_LAUNCH` chunks. Unpinned host
    chunks are staged through pinned buffers for ``D2H`` and pinned into
    temporaries for ``H2D``; both cases synchronize ``device`` before
    returning, because the kernel reaches them through raw addresses that
    torch cannot track. Pinned chunks keep the asynchronous path.

    Args:
        kernel: The compiled ``multi_layer_block_kv_transfer``.
        paged_kv: Normalized paged KV tensors on ``device``.
        chunks: Host chunk tensors, filled in place for ``D2H``.
        block_ids: Paged-block IDs in chunk order.
        device: Device that owns the paged KV tensors.
        direction: ``D2H`` or ``H2D``.
        shape_desc: Paged-buffer shape descriptor.
        lmcache_chunk_size: Tokens per chunk.
        engine_kv_format: Engine KV layout.
        skip_prefix_n_blocks: Leading blocks to skip across the whole
            transfer; each launch skips the part that falls inside it.
    """
    device = torch.device(device)
    is_d2h = int(direction) == int(lmcache_native.TransferDirection.D2H)
    unpinned = not all(chunk.is_pinned() for chunk in chunks)
    if unpinned and _all_registered(device, chunks):
        _transfer_via_device_scratch(
            kernel,
            _paged_pointer_table(
                paged_kv, engine_kv_format, int(shape_desc.nl), device
            ),
            chunks,
            torch.as_tensor(block_ids, dtype=torch.int64, device=device),
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )
        return
    kernel_chunks = chunks
    if unpinned and is_d2h:
        # Pinning would replace the caller's tensors and break the in-place
        # fill, so the kernel writes pinned staging buffers that are copied
        # back below.
        logger.warning(
            "Unpinned host chunks in a native D2H block transfer (likely "
            "shared memory); staging through pinned buffers adds a host "
            "copy and a device sync."
        )
        kernel_chunks = [torch.empty_like(c, pin_memory=True) for c in chunks]
    elif unpinned:
        logger.warning(
            "Unpinned host chunks in a native H2D block transfer; pinning "
            "temporaries adds a device sync."
        )
        kernel_chunks = [c if c.is_pinned() else c.pin_memory() for c in chunks]

    paged_addresses = _paged_pointer_table(
        paged_kv, engine_kv_format, int(shape_desc.nl), device
    )
    chunk_addresses = [chunk.data_ptr() for chunk in kernel_chunks]
    block_id_tensor = torch.as_tensor(block_ids, dtype=torch.int64, device=device)
    blocks_per_launch = MAX_CHUNKS_PER_LAUNCH * lmcache_chunk_size // int(shape_desc.bs)
    for launch, first in enumerate(
        range(0, len(chunk_addresses), MAX_CHUNKS_PER_LAUNCH)
    ):
        start = launch * blocks_per_launch
        launch_block_ids = block_id_tensor[start : start + blocks_per_launch]
        kernel(
            paged_addresses,
            chunk_addresses[first : first + MAX_CHUNKS_PER_LAUNCH],
            launch_block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            # A skip longer than one launch carries into the next launches.
            max(0, min(skip_prefix_n_blocks - start, len(launch_block_ids))),
        )

    if unpinned:
        # The kernel reaches these buffers through raw addresses that torch's
        # stream tracking cannot see: wait before the host reads the staging
        # copies (D2H) or the pinned temporaries are released (H2D).
        synchronize_device(device)
        if is_d2h:
            for dst, src in zip(chunks, kernel_chunks, strict=True):
                dst.copy_(src)
