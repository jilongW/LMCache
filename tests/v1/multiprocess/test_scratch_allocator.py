# SPDX-License-Identifier: Apache-2.0
# Standard
import threading
import time

# First Party
from lmcache.v1.multiprocess.scratch_allocator import ScratchAllocator


def test_scratch_allocator_reuses_and_merges_ranges() -> None:
    allocator = ScratchAllocator(100, 30)
    first = allocator.allocate(10)
    second = allocator.allocate(20)
    assert first is not None
    assert second is not None
    assert allocator.allocate(1) is None

    allocator.free(first)
    allocator.free(second)
    merged = allocator.allocate(30)
    assert merged is not None
    assert merged.offset == 100
    assert merged.size == 30


def test_scratch_allocator_waits_for_free_range() -> None:
    allocator = ScratchAllocator(0, 4)
    held = allocator.allocate(4)
    assert held is not None
    result: list[object] = []

    def allocate_after_free() -> None:
        result.append(allocator.allocate(4, wait=True, timeout_s=1.0))

    thread = threading.Thread(target=allocate_after_free)
    thread.start()
    time.sleep(0.02)
    allocator.free(held)
    thread.join(timeout=1.0)
    assert result and result[0] is not None


def test_scratch_allocator_times_out() -> None:
    allocator = ScratchAllocator(0, 4)
    held = allocator.allocate(4)
    assert held is not None
    started = time.monotonic()
    assert allocator.allocate(4, wait=True, max_retries=1, timeout_s=0.02) is None
    assert time.monotonic() - started < 0.2