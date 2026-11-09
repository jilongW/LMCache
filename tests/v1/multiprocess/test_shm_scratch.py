# SPDX-License-Identifier: Apache-2.0
"""Scratch allocation from the SHM tail of ``EngineDrivenContextShm``."""

# Standard
from multiprocessing import shared_memory
from unittest.mock import MagicMock
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.shm import EngineDrivenContextShm

L1_SIZE = 4096
SCRATCH = 8192
POOL = L1_SIZE + SCRATCH


@pytest.fixture
def segment():
    name = f"lmcache_l1_pool_scratch_test_{os.getpid()}"
    shm = shared_memory.SharedMemory(name=name, create=True, size=POOL)
    shm.buf[:] = bytes(POOL)
    try:
        yield shm
    finally:
        shm.close()
        shm.unlink()


def _context(segment, *, offset=L1_SIZE, size=SCRATCH) -> EngineDrivenContextShm:
    metadata = EngineDrivenContextMetadata(
        layout_desc=MemoryLayoutDesc(
            shapes=[torch.Size([2, 1, 4, 8])], dtypes=[torch.float16]
        ),
        block_size=4,
        use_mla=False,
    )
    return EngineDrivenContextShm(
        metadata,
        MagicMock(),
        1.0,
        segment.name,
        POOL,
        scratch_offset=offset,
        scratch_size=size,
    )


def test_scratch_views_alias_the_shared_tail(segment):
    ctx = _context(segment)
    try:
        assert ctx.has_scratch
        result = ctx.allocate_scratch_tensors(torch.Size([4, 8]), torch.float16, 2)
        assert result is not None
        tensors, allocation = result
        assert [t.shape for t in tensors] == [torch.Size([4, 8])] * 2
        assert all(t.dtype == torch.float16 for t in tensors)

        tensors[0].fill_(1.0)
        tensors[1].fill_(2.0)

        # The writes are visible through the shared segment itself, inside the
        # scratch range and nowhere in the L1 pool.
        seen = torch.frombuffer(segment.buf, dtype=torch.float16)
        assert float(seen[: L1_SIZE // 2].abs().sum()) == 0.0
        assert float(seen.sum()) == 32 * 1.0 + 32 * 2.0
        assert L1_SIZE <= allocation.offset
        assert allocation.offset + allocation.size <= POOL
        # The two tensors do not overlap.
        assert tensors[0].data_ptr() != tensors[1].data_ptr()
        assert abs(tensors[1].data_ptr() - tensors[0].data_ptr()) >= 4 * 8 * 2
    finally:
        ctx.close()


def test_scratch_start_is_aligned_even_for_an_odd_partition(segment):
    ctx = _context(segment, offset=L1_SIZE + 3, size=SCRATCH - 3)
    try:
        tensors, allocation = ctx.allocate_scratch_tensors(
            torch.Size([5]), torch.float32, 1
        )
        assert allocation.offset % 64 == 0
        assert tensors[0].data_ptr() % 4 == 0
        assert allocation.offset + allocation.size <= POOL
    finally:
        ctx.close()


def test_exhausted_scratch_returns_none_until_freed(segment):
    ctx = _context(segment)
    try:
        # One float16 element per 64-byte unit => the whole tail in one go.
        shape = torch.Size([SCRATCH // 2])
        first = ctx.allocate_scratch_tensors(shape, torch.float16, 1, wait=False)
        assert first is not None
        assert ctx.allocate_scratch_tensors(shape, torch.float16, 1, wait=False) is None

        ctx.free_scratch(first[1])
        again = ctx.allocate_scratch_tensors(shape, torch.float16, 1, wait=False)
        assert again is not None
    finally:
        ctx.close()


def test_freed_neighbours_merge_into_one_range(segment):
    ctx = _context(segment)
    try:
        half = torch.Size([SCRATCH // 4])
        a = ctx.allocate_scratch_tensors(half, torch.float16, 1, wait=False)
        b = ctx.allocate_scratch_tensors(half, torch.float16, 1, wait=False)
        assert a is not None and b is not None
        ctx.free_scratch(b[1])
        ctx.free_scratch(a[1])

        whole = ctx.allocate_scratch_tensors(
            torch.Size([SCRATCH // 2]), torch.float16, 1, wait=False
        )
        assert whole is not None
    finally:
        ctx.close()


def test_without_a_partition_there_is_no_scratch(segment):
    ctx = _context(segment, offset=0, size=0)
    try:
        assert not ctx.has_scratch
        assert (
            ctx.allocate_scratch_tensors(torch.Size([4]), torch.float16, 1, wait=False)
            is None
        )
    finally:
        ctx.close()


def test_a_partition_beyond_the_pool_is_rejected(segment):
    with pytest.raises(ValueError, match="exceeds the SHM pool"):
        _context(segment, offset=L1_SIZE, size=SCRATCH + 1)
