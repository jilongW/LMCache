# SPDX-License-Identifier: Apache-2.0
"""Does concurrent prefill-like compute change the EU-gather vs.
memcpy-reshape verdict from test_xpu_eu_vs_memcpy_benchmark.py?

That benchmark measures both copy paths in isolation (GPU otherwise idle) and
finds memcpy-reshape faster there. But it does not model the real production
situation: KV-cache gather for one request's store typically happens while
the GPU is busy running prefill/decode compute for other requests.

This matters because the two paths use different hardware:

- The EU gather kernel (``gather_paged_kv_to_cpu`` ->
  ``multi_layer_kv_transfer``, see csrc/sycl/mem_kernels_sycl.cpp) is a SYCL
  ``parallel_for`` -- a compute kernel that runs on the same Execution Units
  (EU) as prefill/decode matmuls. Even on its own stream, it competes with
  compute kernels for EU issue slots; the GPU cannot run two compute kernels
  on the same EUs at once.
- ``queue.memcpy()`` (``lmcache_memcpy_async``) may be routed by the driver
  to BCS (the fixed-function Blitter/copy-engine DMA queue), which is
  hardware independent of the EUs -- if so, it can genuinely overlap with
  compute instead of competing for the same execution resources.

So the idle-GPU comparison could flip, or at least narrow, once there is
real EU contention: the EU kernel's isolated-case disadvantage could shrink
(both copy and compute now pay EU contention) while memcpy's isolated-case
advantage could grow (if it truly avoids EU contention) -- or the driver
could route memcpy's DMA through the EUs too, in which case nothing changes.
This benchmark isolates that one variable empirically rather than assuming
either outcome.

Method: run a sustained matmul loop on a dedicated "compute" stream to
simulate prefill EU occupancy, then measure, on a separate "copy" stream:
  (a) copy latency with vs. without the concurrent compute load, for both
      copy paths (EU gather vs. memcpy-reshape, same geometry as
      test_xpu_engine_driven_first_transfer.py / test_xpu_eu_vs_memcpy_benchmark.py).
  (b) compute throughput (matmuls/sec) with vs. without each concurrent copy
      path, to see whether the copy steals EU cycles from compute.

Run directly for a printed timing table::

    python tests/benchmarks/test_xpu_copy_under_compute_load_benchmark.py
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass

import pytest
import torch

from lmcache import torch_dev, torch_device_type
from lmcache.lmcache_native import EngineKVFormat
from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu


if not (torch_device_type == "xpu" and torch_dev.is_available()):
    pytest.skip("Requires an available XPU runtime", allow_module_level=True)

import lmcache.xpu_ops as xpu_ops  # noqa: E402


# Geometry matches test_xpu_engine_driven_first_transfer.py /
# test_xpu_eu_vs_memcpy_benchmark.py: 8 layers, 1 block, 1024 tokens/block,
# 4 heads, head_dim 256, fp16.
_NUM_BLOCKS = 1
_BLOCK_SIZE = 1024
_NUM_HEADS = 4
_HEAD_SIZE = 256
_NUM_LAYERS = 8
_BLOCK_IDS = [0]
_BLOCKS_PER_CHUNK = 1

_ATTENTION_FORMAT = EngineKVFormat.NL_X_NB_BS_NH_CS
_MAMBA_FORMAT = EngineKVFormat.NL_X_NB_TWO_BS_NH_HS

_TRANSFER_DIRECTION_D2H = 1  # TransferDirection::D2H, see csrc/kv_transfer_types.h
# See test_xpu_eu_vs_memcpy_benchmark.py for why this must be >= the largest
# single per-call transfer size tested, so each call happens as one
# queue.memcpy() instead of being chopped into many tiny ones.
_HOST_BUFFER_ALIGNMENT = 1 << 26  # 64MB, power of two

_COPY_MODE_EU = "eu"
_COPY_MODE_MEMCPY = "memcpy"

# 4096x4096 fp16 matmul measured at ~3.4ms/call on this hardware -- same
# order of magnitude as the copy durations under test (3-16ms), so overlap
# windows are realistic rather than the compute finishing instantly or
# dwarfing the copy entirely.
_MATMUL_N = 4096

_WARMUP_ITERS = 2
_MEASURED_ITERS = 10


@dataclass(frozen=True)
class _FormatSpec:
    name: str
    engine_kv_format: EngineKVFormat
    kv_size: int  # 1 for fused attention, 2 for split Mamba K/V


_ATTENTION_SPEC = _FormatSpec("attention (fused K/V)", _ATTENTION_FORMAT, kv_size=1)
_MAMBA_SPEC = _FormatSpec("mamba (split K/V)", _MAMBA_FORMAT, kv_size=2)


def _make_source(spec: _FormatSpec) -> dict[str, torch.Tensor]:
    if spec.kv_size == 1:
        shape = (_NUM_BLOCKS, _BLOCK_SIZE, _NUM_HEADS, 2 * _HEAD_SIZE)
    else:
        shape = (_NUM_BLOCKS, spec.kv_size, _BLOCK_SIZE, _NUM_HEADS, _HEAD_SIZE)
    return {
        f"layer_{layer_idx}": torch.zeros(shape, dtype=torch.float16, device="xpu")
        for layer_idx in range(_NUM_LAYERS)
    }


def _make_output(spec: _FormatSpec) -> torch.Tensor:
    if spec.kv_size == 1:
        return torch.empty(
            _NUM_LAYERS,
            _BLOCK_SIZE,
            _NUM_HEADS * 2 * _HEAD_SIZE,
            dtype=torch.float16,
            device="cpu",
        )
    return torch.empty(
        spec.kv_size,
        _NUM_LAYERS,
        _BLOCK_SIZE,
        _NUM_HEADS * _HEAD_SIZE,
        dtype=torch.float16,
        device="cpu",
    )


def _eu_gather(spec: _FormatSpec, source: dict[str, torch.Tensor]) -> torch.Tensor:
    output = [_make_output(spec)]
    gather_paged_kv_to_cpu(
        source,
        _BLOCK_IDS,
        _BLOCKS_PER_CHUNK,
        engine_kv_format=spec.engine_kv_format,
        out=output,
    )
    return output[0]


def _memcpy_reshape_gather(
    spec: _FormatSpec, source: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Reproduce the EU kernel's copy+reshape via one queue.memcpy() per
    contiguous source slice. See test_xpu_eu_vs_memcpy_benchmark.py for why
    a bare single memcpy is not an equivalent comparison.
    """
    output = _make_output(spec)
    output_ptr = output.data_ptr()
    if spec.kv_size == 1:
        layer_stride_bytes = _BLOCK_SIZE * _NUM_HEADS * 2 * _HEAD_SIZE * 2
        for layer_idx, src in enumerate(source.values()):
            xpu_ops.lmcache_memcpy_async(
                output_ptr + layer_idx * layer_stride_bytes,
                src.data_ptr(),
                layer_stride_bytes,
                _TRANSFER_DIRECTION_D2H,
                0,
                _HOST_BUFFER_ALIGNMENT,
            )
    else:
        bytes_per_layer_kv = _BLOCK_SIZE * _NUM_HEADS * _HEAD_SIZE * 2
        layer_stride_bytes = bytes_per_layer_kv
        kv_stride_bytes = _NUM_LAYERS * layer_stride_bytes
        for layer_idx, src in enumerate(source.values()):
            for k_or_v in range(spec.kv_size):
                src_offset_bytes = k_or_v * bytes_per_layer_kv
                dst_offset_bytes = (
                    k_or_v * kv_stride_bytes + layer_idx * layer_stride_bytes
                )
                xpu_ops.lmcache_memcpy_async(
                    output_ptr + dst_offset_bytes,
                    src.data_ptr() + src_offset_bytes,
                    bytes_per_layer_kv,
                    _TRANSFER_DIRECTION_D2H,
                    0,
                    _HOST_BUFFER_ALIGNMENT,
                )
    return output


def _gather(spec: _FormatSpec, source: dict[str, torch.Tensor], copy_mode: str) -> None:
    if copy_mode == _COPY_MODE_MEMCPY:
        _memcpy_reshape_gather(spec, source)
    else:
        _eu_gather(spec, source)


def _check_correctness(spec: _FormatSpec) -> None:
    source = _make_source(spec)
    for layer_idx, src in enumerate(source.values()):
        src.fill_(float(layer_idx + 1))
    eu_out = _eu_gather(spec, source)
    memcpy_out = _memcpy_reshape_gather(spec, source)
    if not torch.equal(eu_out, memcpy_out):
        raise RuntimeError(
            f"[{spec.name}] memcpy-based reshape does not match EU kernel "
            "output -- benchmark comparison would be invalid"
        )


class _ComputeLoadRunner:
    """Runs a sustained matmul loop on a dedicated XPU stream to simulate
    prefill EU occupancy, from a background Python thread so it can overlap
    with copy calls issued from the main thread.
    """

    def __init__(self, matmul_n: int = _MATMUL_N) -> None:
        self._a = torch.randn(matmul_n, matmul_n, dtype=torch.float16, device="xpu")
        self._b = torch.randn(matmul_n, matmul_n, dtype=torch.float16, device="xpu")
        self._stream = torch_dev.Stream()
        self._stop = threading.Event()
        self._iters = 0
        self._thread: threading.Thread | None = None

    def _loop(self) -> None:
        with torch_dev.stream(self._stream):
            while not self._stop.is_set():
                torch.matmul(self._a, self._b)
                self._stream.synchronize()
                self._iters += 1

    def start(self) -> None:
        self._stop.clear()
        self._iters = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop_and_count(self) -> int:
        self._stop.set()
        assert self._thread is not None
        self._thread.join()
        return self._iters


def _time_gather_calls(
    spec: _FormatSpec, copy_mode: str, num_calls: int
) -> list[float]:
    """Issue ``num_calls`` independent gathers back-to-back on their own
    stream; return each call's elapsed ms (submit through completion)."""
    source = _make_source(spec)
    copy_stream = torch_dev.Stream()
    times = []
    for _ in range(num_calls):
        torch_dev.synchronize()
        start = time.perf_counter()
        with torch_dev.stream(copy_stream):
            _gather(spec, source, copy_mode)
        copy_stream.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def _bench_copy_alone(spec: _FormatSpec, copy_mode: str) -> dict[str, float]:
    for _ in range(_WARMUP_ITERS):
        _time_gather_calls(spec, copy_mode, 1)
    times = _time_gather_calls(spec, copy_mode, _MEASURED_ITERS)
    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


_LOAD_BASELINE_SECONDS = 0.5


def _bench_copy_under_load(spec: _FormatSpec, copy_mode: str) -> dict[str, float]:
    """Measure copy latency while a matmul loop keeps the EUs busy, and
    compute throughput with vs. without the concurrent copy."""
    loop = _ComputeLoadRunner()

    # Baseline compute throughput: matmul loop alone, no concurrent copy.
    loop.start()
    time.sleep(_LOAD_BASELINE_SECONDS)
    baseline_iters = loop.stop_and_count()
    baseline_matmul_per_s = baseline_iters / _LOAD_BASELINE_SECONDS

    # Copy latency with the matmul loop running concurrently. Warmup runs
    # happen before starting the iteration counter / wall-clock window so
    # they don't dilute the loaded-throughput measurement below.
    source = _make_source(spec)
    copy_stream = torch_dev.Stream()
    for _ in range(_WARMUP_ITERS):
        with torch_dev.stream(copy_stream):
            _gather(spec, source, copy_mode)
        copy_stream.synchronize()

    loop.start()
    window_start = time.perf_counter()
    copy_times = []
    for _ in range(_MEASURED_ITERS):
        call_start = time.perf_counter()
        with torch_dev.stream(copy_stream):
            _gather(spec, source, copy_mode)
        copy_stream.synchronize()
        copy_times.append((time.perf_counter() - call_start) * 1000.0)
    window_elapsed_s = time.perf_counter() - window_start
    loaded_iters = loop.stop_and_count()

    return {
        "copy_mean_ms": sum(copy_times) / len(copy_times),
        "copy_min_ms": min(copy_times),
        "copy_max_ms": max(copy_times),
        "baseline_matmul_per_s": baseline_matmul_per_s,
        "loaded_matmul_per_s": loaded_iters / window_elapsed_s,
    }


def _bench(spec: _FormatSpec, copy_mode: str) -> dict[str, float]:
    _check_correctness(spec)
    alone = _bench_copy_alone(spec, copy_mode)
    loaded = _bench_copy_under_load(spec, copy_mode)
    slowdown = loaded["copy_mean_ms"] / alone["mean_ms"]
    compute_slowdown = (
        loaded["baseline_matmul_per_s"] / loaded["loaded_matmul_per_s"]
        if loaded["loaded_matmul_per_s"] > 0
        else float("inf")
    )
    return {
        "alone_mean_ms": alone["mean_ms"],
        "alone_min_ms": alone["min_ms"],
        "alone_max_ms": alone["max_ms"],
        "loaded_mean_ms": loaded["copy_mean_ms"],
        "loaded_min_ms": loaded["copy_min_ms"],
        "loaded_max_ms": loaded["copy_max_ms"],
        "copy_slowdown_under_load": slowdown,
        "baseline_matmul_per_s": loaded["baseline_matmul_per_s"],
        "loaded_matmul_per_s": loaded["loaded_matmul_per_s"],
        "compute_slowdown_from_copy": compute_slowdown,
    }


def test_eu_and_memcpy_survive_concurrent_compute() -> None:
    """Keep pytest discovery useful: both copy paths should complete without
    error while a concurrent matmul load is running."""
    for copy_mode in (_COPY_MODE_EU, _COPY_MODE_MEMCPY):
        result = _bench(_ATTENTION_SPEC, copy_mode)
        assert result["alone_mean_ms"] > 0
        assert result["loaded_mean_ms"] > 0


def _print_report(spec: _FormatSpec, copy_mode: str, r: dict[str, float]) -> None:
    print(f"=== {spec.name} -- copy_mode={copy_mode} ===")
    print(
        f"  copy alone:        mean={r['alone_mean_ms']:.3f}ms  "
        f"min={r['alone_min_ms']:.3f}ms  max={r['alone_max_ms']:.3f}ms"
    )
    print(
        f"  copy under load:   mean={r['loaded_mean_ms']:.3f}ms  "
        f"min={r['loaded_min_ms']:.3f}ms  max={r['loaded_max_ms']:.3f}ms"
    )
    print(
        f"  copy slowdown under concurrent compute: "
        f"{r['copy_slowdown_under_load']:.2f}x"
    )
    print(
        f"  matmul/s baseline (no concurrent copy):   "
        f"{r['baseline_matmul_per_s']:.2f}"
    )
    print(
        f"  matmul/s with concurrent copy:            "
        f"{r['loaded_matmul_per_s']:.2f}"
    )
    print(
        f"  compute slowdown from concurrent copy:    "
        f"{r['compute_slowdown_from_copy']:.2f}x"
    )
    print()


def _print_verdict(results: dict[tuple[str, str], dict[str, float]]) -> None:
    print("=== Verdict ===")
    for (spec_name, copy_mode), r in results.items():
        copy_sd = r["copy_slowdown_under_load"]
        compute_sd = r["compute_slowdown_from_copy"]
        print(
            f"{spec_name} / {copy_mode}: copy {copy_sd:.2f}x slower under "
            f"load, compute {compute_sd:.2f}x slower from concurrent copy "
            "-- EU contention present"
            if copy_sd > 1.1 or compute_sd > 1.1
            else f"{spec_name} / {copy_mode}: no significant mutual "
            "slowdown -- this path likely avoids EU contention (e.g. "
            "routed through BCS)"
        )
    print(
        "\nCompare EU vs memcpy per format above: if EU's copy_slowdown and "
        "compute_slowdown are both small while memcpy's are large (or vice "
        "versa), that path is the one genuinely free of EU contention -- "
        "which may reverse the idle-GPU verdict from "
        "test_xpu_eu_vs_memcpy_benchmark.py once real prefill/decode "
        "compute is sharing the GPU."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--copy-modes",
        nargs="+",
        choices=[_COPY_MODE_EU, _COPY_MODE_MEMCPY],
        default=[_COPY_MODE_EU, _COPY_MODE_MEMCPY],
    )
    args = parser.parse_args()

    results: dict[tuple[str, str], dict[str, float]] = {}
    for spec in (_ATTENTION_SPEC, _MAMBA_SPEC):
        for copy_mode in args.copy_modes:
            r = _bench(spec, copy_mode)
            results[(spec.name, copy_mode)] = r
            _print_report(spec, copy_mode, r)
    _print_verdict(results)
