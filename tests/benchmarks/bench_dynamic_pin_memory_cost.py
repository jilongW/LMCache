# SPDX-License-Identifier: Apache-2.0
"""Measure the real cost of the "dynamically pinned" H2D fallback
(``scatter_cpu_to_paged_kv``'s ``dynamically_pinned`` branch, see
``lmcache/v1/multiprocess/transfer_context/base.py`` around the
``requires_pinned``/``chunk.pin_memory()`` code) against Qwen3-14B's real
KV-cache geometry, split into 256-token chunks (16 chunks for a 4096-token
retrieve).

Context: ``EngineDrivenContextShm._pin_shm_buffer`` (shm.py) tries to pin the
whole SHM pool once, at registration, via
``current_device_spec.pin_memory_backend``. XPU's ``DeviceSpec`` never
overrides this property, so it falls back to the base ``PinMemoryBackend``,
whose ``pin_memory()`` is a no-op that always returns ``False`` (see
``lmcache/v1/platform/base/pin_memory.py``) -- meaning the SHM pool is
*never* pinned on XPU today. When the native transfer kernel build requires
pinned host memory (``_LMC_OPS_BLOCK_TRANSFER_ACCEPTS_TENSOR is False``,
i.e. "ptr" mode), every unquantized (RAW) retrieve chunk -- which reads
directly from that never-pinned SHM pool -- takes the ``dynamically_pinned``
fallback: ``chunk.pin_memory()`` per chunk, which allocates a *fresh* pinned
buffer and copies the chunk's bytes into it. This script measures exactly
that per-chunk pin+copy cost in isolation, compares it against pinning the
same-shaped *regular* (non-SHM) memory, and reports the effective bandwidth
so it can be compared directly against an external system-memory-bandwidth
benchmark (e.g. ``mlc``/``stream``), which is this investigation's open
question: does one pin_memory() call's throughput match plain memcpy
bandwidth, or is there additional fixed per-call overhead that a pure
bytes/bandwidth model can't explain?

Requires a real XPU device (exits early otherwise; the ``dynamically_pinned``
fallback this script targets is XPU/CUDA/"ptr"-mode specific). Run with:

    python tests/benchmarks/bench_dynamic_pin_memory_cost.py
"""

# Standard
import os
import statistics
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.multiprocess.posix_shm import (
    shm_create_readwrite,
    shm_munmap,
    shm_unlink,
)

# Qwen3-14B real geometry (docs/source/_static/modelconfig.json):
# num_hidden_layers=40, num_key_value_heads=8, head_dim=128, bf16/fp16 (2B).
_NUM_LAYERS = 40
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_HIDDEN_DIM = _NUM_KV_HEADS * _HEAD_DIM
_DTYPE = torch.float16
_ITEMSIZE = 2

_TOKENS_PER_CHUNK = 256
_NUM_CHUNKS = 16  # 4096 tokens / 256 tokens-per-chunk

# Raw chunk shape matches production's split-K/V gather output:
# [2, num_layers, tokens, hidden_dim] -- see _raw_gather_shape in
# worker_transfer.py.
_CHUNK_SHAPE = torch.Size([2, _NUM_LAYERS, _TOKENS_PER_CHUNK, _HIDDEN_DIM])
_CHUNK_BYTES = (
    2 * _NUM_LAYERS * _TOKENS_PER_CHUNK * _HIDDEN_DIM * _ITEMSIZE
)
_TOTAL_BYTES = _CHUNK_BYTES * _NUM_CHUNKS

_NUM_ITERATIONS = 20
_NUM_WARMUP = 3


def _print_stats(label: str, samples_ms: list[float], num_bytes: int) -> None:
    mean_ms = statistics.mean(samples_ms)
    stdev_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    mean_s = mean_ms / 1000.0
    gib_per_s = (num_bytes / (1024**3)) / mean_s if mean_s > 0 else 0.0
    size_mib = num_bytes / (1024**2)
    print(
        f"  {label:<32s} "
        f"mean={mean_ms:8.3f}ms  min={min(samples_ms):8.3f}ms  "
        f"max={max(samples_ms):8.3f}ms  stdev={stdev_ms:7.3f}ms  "
        f"n={len(samples_ms)}  size={size_mib:9.3f}MiB  bw={gib_per_s:7.2f}GiB/s"
    )


def _time_one_pin_per_chunk(chunks: list[torch.Tensor]) -> float:
    """Pin every chunk independently (mirrors the real per-chunk
    ``dynamically_pinned`` fallback: one ``pin_memory()`` call per chunk,
    each allocating its own fresh pinned buffer)."""
    start = time.perf_counter()
    pinned = [chunk.pin_memory() for chunk in chunks]
    elapsed = time.perf_counter() - start
    del pinned
    return elapsed


def _time_one_big_pin(chunks: list[torch.Tensor]) -> float:
    """Pin the same total bytes as a single contiguous tensor -- isolates
    the "N separate calls" effect from the "total bytes" effect."""
    merged = torch.empty(
        (len(chunks), *chunks[0].shape), dtype=chunks[0].dtype
    )
    start = time.perf_counter()
    pinned = merged.pin_memory()
    elapsed = time.perf_counter() - start
    del pinned
    return elapsed


def main() -> int:
    if not (torch_device_type == "xpu" and torch_dev.is_available()):
        print("Skipping: requires an available XPU runtime.")
        return 0

    print(
        f"Qwen3-14B dynamic pin_memory() cost "
        f"({_NUM_LAYERS} layers, {_NUM_KV_HEADS} kv heads x {_HEAD_DIM} head_dim, "
        f"{_NUM_CHUNKS} chunks x {_TOKENS_PER_CHUNK} tokens = 4096 tokens total, "
        f"{_CHUNK_BYTES / 1024**2:.1f} MiB/chunk, "
        f"{_TOTAL_BYTES / 1024**2:.1f} MiB total, "
        f"{_NUM_ITERATIONS} iterations after {_NUM_WARMUP} warmup)\n"
    )

    # --- Case 1: regular (non-SHM) unpinned CPU tensors ---------------------
    regular_chunks = [
        torch.empty(_CHUNK_SHAPE, dtype=_DTYPE) for _ in range(_NUM_CHUNKS)
    ]
    for chunk in regular_chunks:
        assert not chunk.is_pinned()

    print("Case 1: regular (heap) unpinned memory")
    samples = []
    for _ in range(_NUM_WARMUP):
        _time_one_pin_per_chunk(regular_chunks)
    for _ in range(_NUM_ITERATIONS):
        samples.append(_time_one_pin_per_chunk(regular_chunks) * 1000.0)
    _print_stats(f"{_NUM_CHUNKS}x pin_memory() calls", samples, _TOTAL_BYTES)

    samples_one_big = []
    for _ in range(_NUM_WARMUP):
        _time_one_big_pin(regular_chunks)
    for _ in range(_NUM_ITERATIONS):
        samples_one_big.append(_time_one_big_pin(regular_chunks) * 1000.0)
    _print_stats("1x pin_memory() call (merged)", samples_one_big, _TOTAL_BYTES)

    # --- Case 2: real SHM-pool-backed unpinned chunks -----------------------
    # Mirrors production exactly: EngineDrivenContextShm's SHM pool, read as
    # torch.frombuffer views (same mechanism _build_slot_tensors uses), with
    # _pin_shm_buffer never succeeding on XPU (see module docstring).
    shm_name = f"lmcache_bench_pin_{os.getpid()}"
    pool_size = _CHUNK_BYTES * _NUM_CHUNKS
    addr = shm_create_readwrite(shm_name, pool_size)
    try:
        import multiprocessing.shared_memory as shared_memory
        from multiprocessing.resource_tracker import unregister

        shm = shared_memory.SharedMemory(name=shm_name.lstrip("/"), create=False)
        # This process doesn't own the segment (shm_create_readwrite does);
        # unregister so Python's resource tracker doesn't also try to unlink
        # it (see EngineDrivenContextShm.__init__'s identical pattern).
        unregister(f"/{shm.name}", "shared_memory")
        try:
            shm_chunks = [
                torch.frombuffer(
                    shm.buf,
                    dtype=_DTYPE,
                    count=_CHUNK_SHAPE.numel(),
                    offset=i * _CHUNK_BYTES,
                ).view(_CHUNK_SHAPE)
                for i in range(_NUM_CHUNKS)
            ]
            for chunk in shm_chunks:
                assert not chunk.is_pinned()

            print("\nCase 2: real SHM-pool-backed memory (production's actual path)")
            samples_shm = []
            for _ in range(_NUM_WARMUP):
                _time_one_pin_per_chunk(shm_chunks)
            for _ in range(_NUM_ITERATIONS):
                samples_shm.append(_time_one_pin_per_chunk(shm_chunks) * 1000.0)
            _print_stats(
                f"{_NUM_CHUNKS}x pin_memory() calls", samples_shm, _TOTAL_BYTES
            )
        finally:
            shm.close()
    finally:
        shm_munmap(addr, pool_size)
        shm_unlink(shm_name)

    # --- Back out an implied fixed-cost-per-call vs. bytes-proportional
    #     split from Case 1's two measurements (N calls vs. 1 call, same
    #     total bytes) -- answers whether chunking itself adds overhead
    #     beyond the raw bytes/bandwidth prediction.
    print(f"\n{'=' * 70}")
    mean_n_calls_ms = statistics.mean(samples)
    mean_one_call_ms = statistics.mean(samples_one_big)
    per_call_overhead_ms = (mean_n_calls_ms - mean_one_call_ms) / (_NUM_CHUNKS - 1)
    print(
        f"{_NUM_CHUNKS}x separate calls: {mean_n_calls_ms:.3f}ms total  vs.  "
        f"1x merged call: {mean_one_call_ms:.3f}ms total (same {_TOTAL_BYTES / 1024**2:.1f} MiB)\n"
        f"=> implied fixed overhead per extra pin_memory() call: "
        f"~{per_call_overhead_ms:.3f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
