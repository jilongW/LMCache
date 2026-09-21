# SPDX-License-Identifier: Apache-2.0
"""Process-local allocator for engine-driven scratch memory."""

# Standard
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class ScratchAllocation:
    """One allocation in a scratch byte range."""

    offset: int
    size: int


class ScratchAllocator:
    """Thread-safe first-fit allocator for one process-owned byte range."""

    def __init__(self, offset: int, size: int) -> None:
        if offset < 0 or size < 0:
            raise ValueError("scratch offset and size must be non-negative")
        self._condition = threading.Condition()
        self._free_ranges: list[tuple[int, int]] = [(offset, size)] if size else []

    def allocate(
        self,
        size: int,
        *,
        wait: bool = False,
        max_retries: int = 3,
        timeout_s: float = 0.5,
    ) -> ScratchAllocation | None:
        """Allocate ``size`` bytes, optionally waiting for a free range."""
        if size <= 0:
            raise ValueError("scratch allocation size must be positive")
        deadline = time.monotonic() + max(0.0, timeout_s)
        attempts = 0
        with self._condition:
            while True:
                for index, (offset, free_size) in enumerate(self._free_ranges):
                    if free_size < size:
                        continue
                    allocation = ScratchAllocation(offset, size)
                    if free_size == size:
                        del self._free_ranges[index]
                    else:
                        self._free_ranges[index] = (offset + size, free_size - size)
                    return allocation
                if not wait or attempts >= max_retries:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                attempts += 1
                self._condition.wait(timeout=remaining)

    def free(self, allocation: ScratchAllocation) -> None:
        """Return an allocation and merge adjacent free ranges."""
        if allocation.size <= 0:
            raise ValueError("scratch allocation size must be positive")
        with self._condition:
            ranges = self._free_ranges + [(allocation.offset, allocation.size)]
            ranges.sort()
            merged: list[tuple[int, int]] = []
            for offset, size in ranges:
                if merged and merged[-1][0] + merged[-1][1] == offset:
                    previous_offset, previous_size = merged[-1]
                    merged[-1] = (previous_offset, previous_size + size)
                else:
                    merged.append((offset, size))
            self._free_ranges = merged
            self._condition.notify_all()
