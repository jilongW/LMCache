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

POOL = 4096
SCRATCH = 8192


@pytest.fixture
def segments():
    """An L1 pool segment and a separate scratch segment, as the server makes."""
    pool = shared_memory.SharedMemory(
        name=f"lmcache_l1_pool_scratch_test_{os.getpid()}", create=True, size=POOL
    )
    scratch = shared_memory.SharedMemory(
        name=f"lmcache_l1_scratch_test_{os.getpid()}", create=True, size=SCRATCH
    )
    pool.buf[:] = bytes(POOL)
    scratch.buf[:] = bytes(SCRATCH)
    try:
        yield pool, scratch
    finally:
        for segment in (pool, scratch):
            segment.close()
            segment.unlink()


def _context(segments, *, size=SCRATCH, named=True) -> EngineDrivenContextShm:
    pool, scratch = segments
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
        pool.name,
        POOL,
        scratch_shm_name=scratch.name if named else "",
        scratch_size=size,
    )


def test_scratch_views_alias_the_scratch_segment_only(segments):
    ctx = _context(segments)
    try:
        assert ctx.has_scratch
        result = ctx.allocate_scratch_tensors(torch.Size([4, 8]), torch.float16, 2)
        assert result is not None
        tensors, allocation = result
        assert [t.shape for t in tensors] == [torch.Size([4, 8])] * 2
        assert all(t.dtype == torch.float16 for t in tensors)

        tensors[0].fill_(1.0)
        tensors[1].fill_(2.0)

        # The writes land in the scratch segment and never in the L1 pool.
        pool, scratch = segments
        assert float(torch.frombuffer(pool.buf, dtype=torch.float16).abs().sum()) == 0
        seen = torch.frombuffer(scratch.buf, dtype=torch.float16)
        assert float(seen.sum()) == 32 * 1.0 + 32 * 2.0
        assert allocation.offset + allocation.size <= SCRATCH
        # The two tensors do not overlap.
        assert tensors[0].data_ptr() != tensors[1].data_ptr()
        assert abs(tensors[1].data_ptr() - tensors[0].data_ptr()) >= 4 * 8 * 2
    finally:
        ctx.close()


def test_scratch_allocations_are_aligned(segments):
    ctx = _context(segments)
    try:
        tensors, allocation = ctx.allocate_scratch_tensors(
            torch.Size([5]), torch.float32, 1
        )
        assert allocation.offset % 64 == 0
        assert tensors[0].data_ptr() % 4 == 0
        assert allocation.offset + allocation.size <= SCRATCH
    finally:
        ctx.close()


def test_exhausted_scratch_returns_none_until_freed(segments):
    ctx = _context(segments)
    try:
        # One float16 element per 2 bytes => the whole segment in one go.
        shape = torch.Size([SCRATCH // 2])
        first = ctx.allocate_scratch_tensors(shape, torch.float16, 1, wait=False)
        assert first is not None
        assert ctx.allocate_scratch_tensors(shape, torch.float16, 1, wait=False) is None

        ctx.free_scratch(first[1])
        again = ctx.allocate_scratch_tensors(shape, torch.float16, 1, wait=False)
        assert again is not None
    finally:
        ctx.close()


def test_freed_neighbours_merge_into_one_range(segments):
    ctx = _context(segments)
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


def test_without_a_scratch_segment_there_is_no_scratch(segments):
    ctx = _context(segments, size=0, named=False)
    try:
        assert not ctx.has_scratch
        assert (
            ctx.allocate_scratch_tensors(torch.Size([4]), torch.float16, 1, wait=False)
            is None
        )
    finally:
        ctx.close()


def test_a_missing_scratch_segment_degrades_to_no_scratch(segments):
    pool, _ = segments
    metadata = EngineDrivenContextMetadata(
        layout_desc=MemoryLayoutDesc(
            shapes=[torch.Size([2, 1, 4, 8])], dtypes=[torch.float16]
        ),
        block_size=4,
        use_mla=False,
    )
    ctx = EngineDrivenContextShm(
        metadata,
        MagicMock(),
        1.0,
        pool.name,
        POOL,
        scratch_shm_name="lmcache_l1_scratch_does_not_exist",
        scratch_size=SCRATCH,
    )
    try:
        assert not ctx.has_scratch
    finally:
        ctx.close()
