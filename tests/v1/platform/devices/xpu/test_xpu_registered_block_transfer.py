# SPDX-License-Identifier: Apache-2.0
"""Hardware tests: block transfers to and from host-registered XPU memory.

Host registration lets the copy engine reach a shared-memory pool, but the
transfer kernel cannot address it and ``tensor.is_pinned()`` does not report
it. ``gather_paged_kv_to_cpu`` / ``scatter_cpu_to_paged_kv`` must still produce
exactly the bytes the pinned-memory path produces.

Two chunk sizes are covered, and chunk counts that cross the kernel's
four-chunk launch boundary, so the scratch buffer is reused across launches.
"""

# Standard
from dataclasses import dataclass
import ctypes
import logging
import mmap
import random

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.platform.devices.xpu import XpuDeviceSpec

if not (torch_device_type == "xpu" and torch_dev.is_available()):
    pytest.skip("Requires an available XPU runtime", allow_module_level=True)

# First Party
from lmcache.v1.client.transfer_context.base import (  # noqa: E402
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)

pytestmark = pytest.mark.xpu

_DEVICE = torch.device("xpu:0")
_BLOCK_SIZE = 16


@dataclass(frozen=True)
class _Geometry:
    """Paged KV cache and chunk geometry of one test configuration."""

    num_layers: int
    num_blocks: int
    num_heads: int
    head_size: int
    chunk_tokens: int

    @property
    def blocks_per_chunk(self) -> int:
        return self.chunk_tokens // _BLOCK_SIZE

    @property
    def chunk_shape(self) -> list[int]:
        return [2, self.num_layers, self.chunk_tokens, self.num_heads * self.head_size]

    @property
    def chunk_bytes(self) -> int:
        hidden = self.num_heads * self.head_size
        return 2 * self.num_layers * self.chunk_tokens * hidden * 2


# ~98 KB chunks.
_SMALL = _Geometry(
    num_layers=3, num_blocks=64, num_heads=2, head_size=64, chunk_tokens=64
)
# ~8 MiB chunks.
_LARGE = _Geometry(
    num_layers=2, num_blocks=800, num_heads=4, head_size=256, chunk_tokens=1024
)

# (geometry, number of chunks). Five, nine and eleven chunks span several
# four-chunk launches.
_CASES = [
    pytest.param(_SMALL, 1, id="small-1"),
    pytest.param(_SMALL, 4, id="small-4"),
    pytest.param(_SMALL, 5, id="small-5"),
    pytest.param(_SMALL, 9, id="small-9"),
    pytest.param(_LARGE, 1, id="large-1"),
    pytest.param(_LARGE, 9, id="large-9"),
    pytest.param(_LARGE, 11, id="large-11"),
]


def _make_kv_caches(geometry: _Geometry) -> dict[str, torch.Tensor]:
    """Random per-layer paged KV caches on the XPU."""
    torch.manual_seed(5)
    return {
        f"layer.{i}": torch.randn(
            2,
            geometry.num_blocks,
            _BLOCK_SIZE,
            geometry.num_heads,
            geometry.head_size,
            device=_DEVICE,
        ).bfloat16()
        for i in range(geometry.num_layers)
    }


@pytest.fixture
def registered_pool():
    """Yield a factory of chunk tensors backed by one host-registered mapping."""
    spec = XpuDeviceSpec()
    mappings: list[tuple[mmap.mmap, int]] = []

    def make(geometry: _Geometry, num_chunks: int) -> list[torch.Tensor]:
        size = num_chunks * geometry.chunk_bytes
        mapping = mmap.mmap(-1, size)
        base = np.frombuffer(mapping, dtype=np.uint8)
        ptr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        assert spec.pin_memory(ptr, size) is True
        mappings.append((mapping, ptr))
        step = geometry.chunk_bytes
        return [
            torch.from_numpy(base[i * step : (i + 1) * step])
            .view(torch.bfloat16)
            .view(geometry.chunk_shape)
            for i in range(num_chunks)
        ]

    yield make
    for _mapping, ptr in mappings:
        spec.unpin_memory(ptr)


def test_is_pinned_range_tracks_registration() -> None:
    """A range is pinned only inside a registered region, until it is released."""
    spec = XpuDeviceSpec()
    size = 4 * mmap.PAGESIZE
    mapping = mmap.mmap(-1, size)
    ptr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    try:
        assert spec.is_pinned_range(ptr, size) is False
        assert spec.pin_memory(ptr, size) is True
        assert spec.is_pinned_range(ptr, size) is True
        assert spec.is_pinned_range(ptr + mmap.PAGESIZE, mmap.PAGESIZE) is True
        assert spec.is_pinned_range(ptr + size - 1, 2) is False
        assert spec.unpin_memory(ptr) is True
        assert spec.is_pinned_range(ptr, size) is False
    finally:
        mapping.close()


@pytest.mark.parametrize(("geometry", "num_chunks"), _CASES)
def test_gather_into_registered_memory_matches_pinned(
    registered_pool, caplog, geometry: _Geometry, num_chunks: int
) -> None:
    """D2H into registered memory is byte-identical to D2H into pinned memory."""
    kv_caches = _make_kv_caches(geometry)
    blocks_per_chunk = geometry.blocks_per_chunk
    block_ids = random.Random(1).sample(
        range(geometry.num_blocks), num_chunks * blocks_per_chunk
    )
    pinned = [
        torch.empty(geometry.chunk_shape, dtype=torch.bfloat16).pin_memory()
        for _ in range(num_chunks)
    ]
    gather_paged_kv_to_cpu(kv_caches, block_ids, blocks_per_chunk, out=pinned)
    torch_dev.synchronize()

    registered = registered_pool(geometry, num_chunks)
    with caplog.at_level(logging.WARNING):
        gather_paged_kv_to_cpu(kv_caches, block_ids, blocks_per_chunk, out=registered)
    torch_dev.synchronize()
    # Registered memory must not take the slow staging path.
    assert "Unpinned host chunks" not in caplog.text

    for expected, actual in zip(pinned, registered, strict=True):
        assert int(expected.view(torch.int16).abs().sum()) != 0
        assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))


@pytest.mark.parametrize("skip_blocks", [0, 1, "chunk", "chunk_plus_block"])
@pytest.mark.parametrize(("geometry", "num_chunks"), _CASES)
def test_scatter_from_registered_memory_matches_pinned(
    registered_pool, caplog, geometry: _Geometry, num_chunks: int, skip_blocks
) -> None:
    """H2D from registered memory writes the same pages as H2D from pinned memory,
    including a skipped token prefix that spans launches."""
    kv_caches = _make_kv_caches(geometry)
    blocks_per_chunk = geometry.blocks_per_chunk
    skip_tokens = {
        0: 0,
        1: _BLOCK_SIZE,
        "chunk": geometry.chunk_tokens,
        "chunk_plus_block": geometry.chunk_tokens + _BLOCK_SIZE,
    }[skip_blocks]
    block_ids = random.Random(2).sample(
        range(geometry.num_blocks), num_chunks * blocks_per_chunk
    )
    source = [
        torch.randint(-1000, 1000, geometry.chunk_shape, dtype=torch.int16).view(
            torch.bfloat16
        )
        for _ in range(num_chunks)
    ]
    pinned = [chunk.clone().pin_memory() for chunk in source]
    registered = registered_pool(geometry, num_chunks)
    for dst, src in zip(registered, source, strict=True):
        dst.copy_(src)
    before = {name: t.clone() for name, t in kv_caches.items()}

    scatter_cpu_to_paged_kv(
        kv_caches, block_ids, pinned, blocks_per_chunk, skip_first_n_tokens=skip_tokens
    )
    torch_dev.synchronize()
    after_pinned = {name: t.clone() for name, t in kv_caches.items()}
    for name, snapshot in before.items():
        kv_caches[name].copy_(snapshot)

    with caplog.at_level(logging.WARNING):
        scatter_cpu_to_paged_kv(
            kv_caches,
            block_ids,
            registered,
            blocks_per_chunk,
            skip_first_n_tokens=skip_tokens,
        )
    torch_dev.synchronize()
    assert "Unpinned host chunks" not in caplog.text

    for name, expected in after_pinned.items():
        assert torch.equal(
            expected.view(torch.int16), kv_caches[name].view(torch.int16)
        )
    if skip_tokens < num_chunks * geometry.chunk_tokens:
        # Not everything was skipped, so something must have been written.
        assert any(
            not torch.equal(
                after_pinned[name].view(torch.int16), before[name].view(torch.int16)
            )
            for name in before
        )
