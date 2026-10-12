# SPDX-License-Identifier: Apache-2.0
"""Does a pool of XPU streams give real D2H copy concurrency, or does the
Level-Zero/SYCL backend serialize them anyway?

LMCache's engine-driven store path (``AsyncEngineDrivenTransferContext`` in
lmcache/v1/multiprocess/transfer_context/async_engine_driven.py) funnels every
KV-cache-group gather copy through a single shared ``self._copy_stream``,
which strictly serializes otherwise-independent copies. Before investing in a
stream-pool fix, this benchmark isolates the one empirical question that
determines whether that fix would help at all: does issuing N independent
``torch.xpu.Stream()`` copies actually reduce wall-clock time vs. issuing the
same N copies back-to-back on one stream?

This measures plain ``tensor.to("cpu", non_blocking=True)`` D2H copies only --
no LMCache internals -- so the result isolates stream-level concurrency from
gather-specific overhead (layout hints, format dispatch, quantization).

Run directly for a printed timing table::

    python tests/benchmarks/test_xpu_multistream_copy_benchmark.py
"""

from __future__ import annotations

import argparse
import time

import pytest
import torch

from lmcache import torch_dev, torch_device_type


if not (torch_device_type == "xpu" and torch_dev.is_available()):
    pytest.skip("Requires an available XPU runtime", allow_module_level=True)


# "small": matches a single KV-cache group's size observed in production
# logs (1024 tokens/block x 4 heads x 256 head_dim x 2 bytes fp16 ~= 2MB).
# At this size a single copy completes in ~0.065ms on this machine -- fast
# enough that Python-level stream/context-manager dispatch overhead is the
# same order of magnitude as the copy itself, which can mask a real hardware
# concurrency signal. "large" repeats the same comparison at a size where
# the copy duration (~1-2ms) dominates dispatch overhead, so any genuine
# multi-stream speedup shows up unambiguously; absence of speedup there is
# strong evidence the backend serializes streams, not just benchmark noise.
_SIZES_MB = {
    "small (real group size, ~2MB)": 2,
    "large (signal-clarifying, ~64MB)": 64,
}

_CONCURRENCY_LEVELS = (2, 4, 8)
_WARMUP_ITERS = 2
_MEASURED_ITERS = 10


def _make_tensors(
    n: int, size_mb: float
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Allocate ``n`` device source tensors and pinned CPU destinations."""
    numel = int(size_mb * 1024 * 1024 // 2)  # fp16 = 2 bytes/elem
    sources = [
        torch.randn(numel, dtype=torch.float16, device=torch_device_type)
        for _ in range(n)
    ]
    destinations = [
        torch.empty(numel, dtype=torch.float16, device="cpu", pin_memory=True)
        for _ in range(n)
    ]
    return sources, destinations


def _run_serial_one_stream(
    sources: list[torch.Tensor], destinations: list[torch.Tensor]
) -> float:
    """Issue all copies back-to-back on a single stream; return elapsed ms."""
    stream = torch_dev.Stream()
    torch_dev.synchronize()
    start = time.perf_counter()
    with torch_dev.stream(stream):
        for src, dst in zip(sources, destinations, strict=True):
            dst.copy_(src, non_blocking=True)
    stream.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _run_parallel_n_streams(
    sources: list[torch.Tensor], destinations: list[torch.Tensor]
) -> float:
    """Issue each copy on its own stream; return elapsed ms."""
    streams = [torch_dev.Stream() for _ in sources]
    torch_dev.synchronize()
    start = time.perf_counter()
    for src, dst, stream in zip(sources, destinations, streams, strict=True):
        with torch_dev.stream(stream):
            dst.copy_(src, non_blocking=True)
    for stream in streams:
        stream.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _bench_concurrency_level(n: int, size_mb: float) -> dict[str, float]:
    """Run warmup + measured iterations for one concurrency level."""
    sources, destinations = _make_tensors(n, size_mb)

    for _ in range(_WARMUP_ITERS):
        _run_serial_one_stream(sources, destinations)
        _run_parallel_n_streams(sources, destinations)

    serial_times = [
        _run_serial_one_stream(sources, destinations) for _ in range(_MEASURED_ITERS)
    ]
    parallel_times = [
        _run_parallel_n_streams(sources, destinations) for _ in range(_MEASURED_ITERS)
    ]

    serial_mean = sum(serial_times) / len(serial_times)
    parallel_mean = sum(parallel_times) / len(parallel_times)
    total_mb = n * size_mb
    return {
        "serial_mean_ms": serial_mean,
        "serial_min_ms": min(serial_times),
        "serial_max_ms": max(serial_times),
        "parallel_mean_ms": parallel_mean,
        "parallel_min_ms": min(parallel_times),
        "parallel_max_ms": max(parallel_times),
        "serial_gbps": total_mb / (serial_mean / 1000.0) / 1024.0,
        "parallel_gbps": total_mb / (parallel_mean / 1000.0) / 1024.0,
    }


def test_multistream_copy_not_drastically_slower() -> None:
    """Keep pytest discovery useful: parallel mode should not regress badly."""
    result = _bench_concurrency_level(4, size_mb=64)
    # Sanity check only -- this is not an assertion about speedup direction,
    # just a guard against a broken Stream-pool implementation making things
    # dramatically worse (e.g. 10x), which would indicate a bug, not just
    # "no hardware concurrency benefit."
    assert result["parallel_mean_ms"] < result["serial_mean_ms"] * 10


def _print_report(label: str, results: dict[int, dict[str, float]]) -> None:
    print(f"=== {label} ===")
    print(
        f"{'N':>3}  {'serial_mean':>12}  {'parallel_mean':>14}  "
        f"{'speedup':>8}  {'serial_GB/s':>11}  {'parallel_GB/s':>13}"
    )
    for n, r in results.items():
        speedup = r["serial_mean_ms"] / r["parallel_mean_ms"]
        print(
            f"{n:>3}  {r['serial_mean_ms']:>9.3f}ms  {r['parallel_mean_ms']:>11.3f}ms  "
            f"{speedup:>7.2f}x  {r['serial_gbps']:>9.2f}  {r['parallel_gbps']:>11.2f}"
        )
    print()


def _print_verdict() -> None:
    print(
        "speedup ~= 1.0x across all N (esp. at 'large' size, where dispatch "
        "overhead cannot explain it) => XPU serializes streams regardless of "
        "pool size (stream-pool fix would NOT help).\n"
        "speedup meaningfully > 1.0x (approaching N, until bandwidth "
        "saturates) => real hardware concurrency confirmed (stream-pool fix "
        "worth pursuing)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=list(_CONCURRENCY_LEVELS),
        help="Concurrency levels (number of simultaneous copies) to test.",
    )
    args = parser.parse_args()

    for label, size_mb in _SIZES_MB.items():
        results = {n: _bench_concurrency_level(n, size_mb) for n in args.levels}
        _print_report(label, results)
    _print_verdict()
