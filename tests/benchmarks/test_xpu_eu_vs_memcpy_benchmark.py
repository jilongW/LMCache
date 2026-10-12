# SPDX-License-Identifier: Apache-2.0
"""Compare LMCache's EU-bound ``parallel_for`` gather kernel against an
equivalent reshape built from SYCL USM ``queue.memcpy()`` calls, for both the
fused-attention and Mamba KV formats, on XPU.

Context: LMCache's attention/paged-KV gather-scatter kernels (dispatched via
``gather_paged_kv_to_cpu`` -> ``lmcache.xpu_ops.multi_layer_kv_transfer``, see
csrc/sycl/mem_kernels_sycl.cpp) are written as explicit SYCL ``parallel_for``
kernels -- these are compute-kernel launches, so they execute on the GPU's EU
(Execution Units), never on BCS (the fixed-function Blitter/copy-engine DMA
queue). Separately, ``lmcache.xpu_ops.lmcache_memcpy_async`` wraps plain
``sycl::queue::memcpy()``, whose EU-vs-BCS routing is an internal driver
decision this codebase has no control over (see the standalone multi-stream
D2H benchmark, test_xpu_multistream_copy_benchmark.py, which found that path
fully serializes across streams at ~53GB/s on this hardware).

The EU kernel does not just copy bytes -- for this single-block test case it
reshapes N separate per-layer page tensors into one contiguous
``[k_or_v, num_layers, tokens, hidden]`` (Mamba) or ``[num_layers, tokens,
hidden]`` (attention) output. A bare ``queue.memcpy()`` of the same total
byte count does strictly less work (one contiguous range, no reshape), so
comparing it directly to the EU kernel is not apples-to-apples. This
benchmark instead compares the EU kernel against the actual alternative
implementation: ``queue.memcpy()`` each contiguous source slice to its
correct offset in a pinned CPU buffer (doing the same reshape via N
placement-copies instead of one fused kernel), separately for:

- attention (``NL_X_NB_BS_NH_CS``, fused K/V, kv_size==1): one memcpy per
  layer (8 total).
- Mamba (``NL_X_NB_TWO_BS_NH_HS``, split K/V, kv_size==2): the source layout
  is ``[block, k_or_v, token, head, head_dim]`` -- for a single block,
  slicing out a fixed ``k_or_v`` still leaves a contiguous
  ``[token, head, head_dim]`` range, so this needs one memcpy per
  (layer, k_or_v) pair (16 total) to land in the kernel's ``[k_or_v, layer,
  token, scalar]`` output layout.

Run directly for a printed timing table::

    python tests/benchmarks/test_xpu_eu_vs_memcpy_benchmark.py
"""

from __future__ import annotations

import argparse
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


# Geometry matches test_xpu_engine_driven_first_transfer.py: 8 layers, 1
# block, 1024 tokens/block, 4 heads, head_dim 256, fp16.
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
# lmcache_memcpy_async (csrc/sycl/mem_kernels_sycl.cpp) splits the transfer
# into queue.memcpy() calls no larger than `host_buffer_alignments` bytes --
# it's designed for writing into aligned slots of a ring-buffer-style pinned
# host buffer, not as a general bulk-copy API. A small alignment here (e.g.
# 64 bytes) silently turns a multi-MB transfer into hundreds of thousands of
# tiny memcpy calls, each paying full SYCL dispatch overhead -- this must be
# >= the largest single per-call transfer size tested so each call happens as
# one queue.memcpy(), which is the fair comparison against the EU kernel.
_HOST_BUFFER_ALIGNMENT = 1 << 26  # 64MB, power of two

_WARMUP_ITERS = 2
_MEASURED_ITERS = 10


@dataclass(frozen=True)
class _FormatSpec:
    """Per-KV-format geometry needed to build sources/outputs and the
    equivalent memcpy-based reshape for that format's EU kernel layout."""

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


def _bytes_per_call(spec: _FormatSpec) -> int:
    # Both formats move the same total volume here (32MB): attention packs
    # K+V into one plane with hidden=2*_HEAD_SIZE (kv_size=1), mamba keeps
    # K/V as two separate kv_size=2 planes of hidden=_HEAD_SIZE each -- do
    # not conflate "kv_size" with a generic multiplier, the two formats
    # reach the same byte count via different dimensions.
    if spec.kv_size == 1:
        hidden = 2 * _HEAD_SIZE
        return _NUM_LAYERS * _BLOCK_SIZE * _NUM_HEADS * hidden * 2  # fp16
    return _NUM_LAYERS * spec.kv_size * _BLOCK_SIZE * _NUM_HEADS * _HEAD_SIZE * 2


def _run_eu_gather(spec: _FormatSpec, source: dict[str, torch.Tensor]) -> float:
    """One EU-kernel gather call (parallel_for path); return elapsed ms."""
    output = [_make_output(spec)]
    torch_dev.synchronize()
    start = time.perf_counter()
    gather_paged_kv_to_cpu(
        source,
        _BLOCK_IDS,
        _BLOCKS_PER_CHUNK,
        engine_kv_format=spec.engine_kv_format,
        out=output,
    )
    torch_dev.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _run_queue_memcpy(src: torch.Tensor, dst: torch.Tensor, nbytes: int) -> float:
    """One queue.memcpy() call moving the same byte volume; return elapsed ms.

    This is NOT doing the same work as the EU kernel (no reshape, one
    contiguous range) -- kept only as a lower-bound reference for "pure bytes,
    no reshape" throughput. See ``_run_memcpy_reshape`` for the fair
    comparison.
    """
    torch_dev.synchronize()
    start = time.perf_counter()
    xpu_ops.lmcache_memcpy_async(
        dst.data_ptr(),
        src.data_ptr(),
        nbytes,
        _TRANSFER_DIRECTION_D2H,
        0,
        _HOST_BUFFER_ALIGNMENT,
    )
    torch_dev.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _run_memcpy_reshape(
    spec: _FormatSpec, source: dict[str, torch.Tensor], output: torch.Tensor
) -> float:
    """Reproduce the EU kernel's copy+reshape via one queue.memcpy() per
    contiguous source slice.

    For a single block, each layer's source tensor is contiguous once
    ``k_or_v`` (if present) is fixed, so the EU kernel's output placement
    (``[k_or_v, layer, token, scalar]`` or ``[layer, token, scalar]``)
    reduces to N memcpy calls into N different offsets of the same
    destination buffer, instead of one fused kernel launch.
    """
    output_ptr = output.data_ptr()
    torch_dev.synchronize()
    start = time.perf_counter()
    if spec.kv_size == 1:
        # attention: source per layer is [1 block, token, head, 2*head_dim]
        # (already fused K/V), output is [layer, token, hidden] contiguous.
        # layer stride = tokens * (num_heads * 2*head_dim) * 2 bytes.
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
        # mamba: source per layer is [1 block, 2(k_or_v), token, head,
        # head_dim] -- fixing k_or_v leaves a contiguous
        # [token, head, head_dim] range. Output is [k_or_v, layer, token,
        # hidden] contiguous, hidden = num_heads*head_dim (no *2, K and V
        # are separate planes here, unlike the attention format above).
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
    torch_dev.synchronize()
    return (time.perf_counter() - start) * 1000.0


def _check_correctness(spec: _FormatSpec) -> None:
    """The memcpy-based reshape must match the EU kernel's output, or the
    "fair comparison" claim for this format is void."""
    source = _make_source(spec)
    for layer_idx, src in enumerate(source.values()):
        src.fill_(float(layer_idx + 1))
    eu_output = [_make_output(spec)]
    gather_paged_kv_to_cpu(
        source,
        _BLOCK_IDS,
        _BLOCKS_PER_CHUNK,
        engine_kv_format=spec.engine_kv_format,
        out=eu_output,
    )
    reshape_output = _make_output(spec)
    _run_memcpy_reshape(spec, source, reshape_output)
    if not torch.equal(eu_output[0], reshape_output):
        raise RuntimeError(
            f"[{spec.name}] memcpy-based reshape does not match EU kernel "
            "output -- benchmark comparison would be invalid"
        )


def _bench(spec: _FormatSpec) -> dict[str, float]:
    _check_correctness(spec)

    nbytes = _bytes_per_call(spec)
    eu_source = _make_source(spec)
    memcpy_src = torch.zeros(nbytes // 2, dtype=torch.float16, device="xpu")
    memcpy_dst = torch.empty(
        nbytes // 2, dtype=torch.float16, device="cpu", pin_memory=True
    )
    reshape_source = _make_source(spec)
    reshape_output = _make_output(spec)

    for _ in range(_WARMUP_ITERS):
        _run_eu_gather(spec, eu_source)
        _run_queue_memcpy(memcpy_src, memcpy_dst, nbytes)
        _run_memcpy_reshape(spec, reshape_source, reshape_output)

    eu_times = [_run_eu_gather(spec, eu_source) for _ in range(_MEASURED_ITERS)]
    memcpy_times = [
        _run_queue_memcpy(memcpy_src, memcpy_dst, nbytes)
        for _ in range(_MEASURED_ITERS)
    ]
    reshape_times = [
        _run_memcpy_reshape(spec, reshape_source, reshape_output)
        for _ in range(_MEASURED_ITERS)
    ]

    eu_mean = sum(eu_times) / len(eu_times)
    memcpy_mean = sum(memcpy_times) / len(memcpy_times)
    reshape_mean = sum(reshape_times) / len(reshape_times)
    mb = nbytes / 1024 / 1024
    return {
        "nbytes": nbytes,
        "eu_mean_ms": eu_mean,
        "eu_min_ms": min(eu_times),
        "eu_max_ms": max(eu_times),
        "eu_gbps": mb / (eu_mean / 1000.0) / 1024.0,
        "memcpy_mean_ms": memcpy_mean,
        "memcpy_min_ms": min(memcpy_times),
        "memcpy_max_ms": max(memcpy_times),
        "memcpy_gbps": mb / (memcpy_mean / 1000.0) / 1024.0,
        "reshape_mean_ms": reshape_mean,
        "reshape_min_ms": min(reshape_times),
        "reshape_max_ms": max(reshape_times),
        "reshape_gbps": mb / (reshape_mean / 1000.0) / 1024.0,
    }


def test_eu_and_memcpy_both_complete_attention() -> None:
    """Keep pytest discovery useful: all paths should run without error."""
    result = _bench(_ATTENTION_SPEC)
    assert result["eu_mean_ms"] > 0
    assert result["memcpy_mean_ms"] > 0
    assert result["reshape_mean_ms"] > 0


def test_eu_and_memcpy_both_complete_mamba() -> None:
    """Keep pytest discovery useful: all paths should run without error."""
    result = _bench(_MAMBA_SPEC)
    assert result["eu_mean_ms"] > 0
    assert result["memcpy_mean_ms"] > 0
    assert result["reshape_mean_ms"] > 0


def _print_report(spec: _FormatSpec, r: dict[str, float]) -> None:
    mb = r["nbytes"] / 1024 / 1024
    print(f"=== {spec.name} ({mb:.1f} MB, {_NUM_LAYERS} layers) ===")
    print(f"{'path':>28}  {'mean':>10}  {'min':>10}  {'max':>10}  {'GB/s':>8}")
    print(
        f"{'EU (parallel_for, fused)':>28}  {r['eu_mean_ms']:>8.3f}ms  "
        f"{r['eu_min_ms']:>8.3f}ms  {r['eu_max_ms']:>8.3f}ms  {r['eu_gbps']:>7.2f}"
    )
    print(
        f"{'memcpy (equiv. reshape)':>28}  {r['reshape_mean_ms']:>8.3f}ms  "
        f"{r['reshape_min_ms']:>8.3f}ms  {r['reshape_max_ms']:>8.3f}ms  "
        f"{r['reshape_gbps']:>7.2f}"
    )
    print(
        f"{'memcpy x1 (no reshape, ref)':>28}  {r['memcpy_mean_ms']:>8.3f}ms  "
        f"{r['memcpy_min_ms']:>8.3f}ms  {r['memcpy_max_ms']:>8.3f}ms  "
        f"{r['memcpy_gbps']:>7.2f}"
    )
    ratio = r["reshape_mean_ms"] / r["eu_mean_ms"]
    if ratio > 1.05:
        print(
            f"-> EU gather (fused copy+reshape) is {ratio:.2f}x FASTER than "
            "doing the same reshape via N queue.memcpy() calls."
        )
    elif ratio < 0.95:
        print(
            f"-> N queue.memcpy() calls (reshape via placement) is "
            f"{1 / ratio:.2f}x FASTER than the EU kernel."
        )
    else:
        print("-> EU gather and the memcpy-based reshape perform about the same.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    for spec in (_ATTENTION_SPEC, _MAMBA_SPEC):
        _print_report(spec, _bench(spec))
    print(
        "(The single-memcpy, no-reshape row is a reference lower bound only "
        "-- it does strictly less work than either of the other two rows.)"
    )
