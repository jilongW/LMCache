# SPDX-License-Identifier: Apache-2.0
"""Measure first-transfer behavior of the XPU engine-driven fallback path.

Run directly for visible timing output::

    python tests/benchmarks/test_xpu_engine_driven_first_transfer.py

Use ``--idle-ms`` to add an idle gap between simulated requests. The source
geometry matches production Qwen3.5-9B hybrid registration: groups 0~2 are
Mamba ``NL_X_NB_TWO_BS_NH_HS`` (8 layers each), and group 3 is fused attention
``NL_X_NB_BS_NH_CS`` (8 layers). Every group contains one 1024-token chunk.
"""

from __future__ import annotations

import argparse
import time

import pytest
import torch

from lmcache import torch_dev, torch_device_type
from lmcache.lmcache_native import EngineKVFormat
from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu


if not (torch_device_type == "xpu" and torch_dev.is_available()):
    pytest.skip("Requires an available XPU runtime", allow_module_level=True)


_NUM_BLOCKS = 1
_BLOCK_SIZE = 1024
_NUM_HEADS = 4
_HEAD_SIZE = 256
_BLOCK_IDS = [0]
_BLOCKS_PER_CHUNK = 1
_NUM_GROUPS = 4
_MAMBA_FORMAT = EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
_ATTENTION_FORMAT = EngineKVFormat.NL_X_NB_BS_NH_CS


def _make_source(engine_kv_format: EngineKVFormat) -> dict[str, torch.Tensor]:
    if engine_kv_format == _MAMBA_FORMAT:
        shape = (
            _NUM_BLOCKS,
            2,
            _BLOCK_SIZE,
            _NUM_HEADS,
            _HEAD_SIZE,
        )
    else:
        shape = (
            _NUM_BLOCKS,
            _BLOCK_SIZE,
            _NUM_HEADS,
            2 * _HEAD_SIZE,
        )
    return {
        f"layer_{layer_idx}": torch.zeros(shape, dtype=torch.float16, device="xpu")
        for layer_idx in range(8)
    }


def _make_output(engine_kv_format: EngineKVFormat) -> list[torch.Tensor]:
    if engine_kv_format != _MAMBA_FORMAT:
        return [
            torch.empty(
                8,
                _BLOCK_SIZE,
                _NUM_HEADS * 2 * _HEAD_SIZE,
                dtype=torch.float16,
                device="cpu",
            )
        ]
    return [
        torch.empty(
            2,
            8,
            _BLOCK_SIZE,
            _NUM_HEADS * _HEAD_SIZE,
            dtype=torch.float16,
            device="cpu",
        )
    ]


def _one_transfer(
    source: dict[str, torch.Tensor], engine_kv_format: EngineKVFormat
) -> tuple[float, float]:
    output = _make_output(engine_kv_format)
    call_start = time.perf_counter()
    gather_paged_kv_to_cpu(
        source,
        _BLOCK_IDS,
        _BLOCKS_PER_CHUNK,
        engine_kv_format=engine_kv_format,
        out=output,
    )
    call_ms = (time.perf_counter() - call_start) * 1000.0
    sync_start = time.perf_counter()
    torch_dev.synchronize()
    sync_ms = (time.perf_counter() - sync_start) * 1000.0
    return call_ms, sync_ms


def _run(num_requests: int, idle_ms: float, warmup: int) -> None:
    groups = [
        (_make_source(_MAMBA_FORMAT), _MAMBA_FORMAT),
        (_make_source(_MAMBA_FORMAT), _MAMBA_FORMAT),
        (_make_source(_MAMBA_FORMAT), _MAMBA_FORMAT),
        (_make_source(_ATTENTION_FORMAT), _ATTENTION_FORMAT),
    ]
    torch_dev.synchronize()

    for warmup_idx in range(warmup):
        for source, engine_kv_format in groups:
            _one_transfer(source, engine_kv_format)
        print(f"warmup={warmup_idx} done", flush=True)

    for request_idx in range(num_requests):
        request_start = time.perf_counter()
        group_timings = []
        for group_idx, (source, engine_kv_format) in enumerate(groups):
            call_ms, sync_ms = _one_transfer(source, engine_kv_format)
            group_timings.append((call_ms, sync_ms))
            print(
                f"request={request_idx} group={group_idx} "
                f"format={engine_kv_format.name} "
                f"transfer_call_ms={call_ms:.3f} sync_ms={sync_ms:.3f}",
                flush=True,
            )
        total_ms = (time.perf_counter() - request_start) * 1000.0
        print(f"request={request_idx} total_ms={total_ms:.3f}", flush=True)
        if request_idx + 1 < num_requests and idle_ms > 0:
            time.sleep(idle_ms / 1000.0)


def test_first_transfer_timing() -> None:
    """Keep pytest discovery useful while preserving direct-run output."""
    _run(num_requests=2, idle_ms=0.0, warmup=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--idle-ms", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0)
    args = parser.parse_args()
    _run(args.requests, args.idle_ms, args.warmup)
