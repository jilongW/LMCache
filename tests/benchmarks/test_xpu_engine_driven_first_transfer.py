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
from importlib.util import module_from_spec, spec_from_file_location
from unittest.mock import MagicMock
import os
import time

import pytest
import torch

from lmcache import torch_dev, torch_device_type
from lmcache.lmcache_native import EngineKVFormat
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, PrepareStoreResponse
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


def _one_engine_driven_request(
    groups: list[tuple[dict[str, torch.Tensor], EngineKVFormat]],
) -> None:
    copy_stream = torch.xpu.Stream()
    pending: list[tuple[int, EngineKVFormat, float, torch.xpu.Event]] = []
    with torch.xpu.stream(copy_stream):
        for group_idx, (source, engine_kv_format) in enumerate(groups):
            output = _make_output(engine_kv_format)
            call_start = time.perf_counter()
            gather_paged_kv_to_cpu(
                source,
                _BLOCK_IDS,
                _BLOCKS_PER_CHUNK,
                engine_kv_format=engine_kv_format,
                out=output,
            )
            submit_ms = (time.perf_counter() - call_start) * 1000.0
            event = torch.xpu.Event()
            event.record(copy_stream)
            pending.append(
                (group_idx, engine_kv_format, submit_ms, event)
            )

    for group_idx, engine_kv_format, submit_ms, event in pending:
        wait_start = time.perf_counter()
        event.synchronize()
        wait_ms = (time.perf_counter() - wait_start) * 1000.0
        print(
            f"engine group={group_idx} format={engine_kv_format.name} "
            f"submit_ms={submit_ms:.3f} event_wait_ms={wait_ms:.3f}",
            flush=True,
        )


def _run_submit_store(
    num_requests: int,
    idle_ms: float,
    warmup: int,
    quantization: bool,
) -> None:
    helper_path = os.path.join(
        os.path.dirname(__file__), "bench_engine_driven_shm_timing.py"
    )
    helper_spec = spec_from_file_location("engine_driven_shm_timing", helper_path)
    if helper_spec is None or helper_spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark helper: {helper_path}")
    helper = module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    from lmcache.v1.multiprocess.posix_shm import (
        shm_create_readwrite,
        shm_munmap,
        shm_unlink,
    )

    if quantization:
        os.environ.setdefault("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")

    kv_caches = helper._qwen35_9b_kv_caches()
    shm_name = f"lmcache_first_transfer_{os.getpid()}"
    shm_addr = shm_create_readwrite(shm_name, helper._SHM_POOL_SIZE)
    try:
        original_groups = helper._qwen35_9b_groups

        def mamba_first_groups():
            groups = original_groups()
            return groups[1:] + groups[:1]

        helper._qwen35_9b_groups = mamba_first_groups
        ctx, shm_ctx = helper._register_shm_context(
            kv_caches, shm_name, helper._SHM_POOL_SIZE
        )
        worker_transfer = __import__(
            "lmcache.v1.multiprocess.transfer_context.worker_transfer",
            fromlist=["gather_paged_kv_to_cpu"],
        )
        original_gather = worker_transfer.gather_paged_kv_to_cpu
        original_sync = worker_transfer.torch_dev.synchronize
        active_request = -1
        gather_index = 0
        sync_index = 0

        def timed_gather(*args, **kwargs):
            nonlocal gather_index
            start = time.perf_counter()
            result = original_gather(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if active_request >= 0:
                engine_format = kwargs.get("engine_kv_format")
                chunk_indices = kwargs.get("chunk_indices")
                print(
                    f"request={active_request} chunk={gather_index} "
                    f"format={engine_format.name if engine_format else 'unknown'} "
                    f"chunk_indices={chunk_indices} "
                    f"gather_call_ms={elapsed_ms:.3f}",
                    flush=True,
                )
            gather_index += 1
            return result

        worker_transfer.gather_paged_kv_to_cpu = timed_gather

        def timed_sync(*args, **kwargs):
            nonlocal sync_index
            start = time.perf_counter()
            result = original_sync(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if active_request >= 0:
                print(
                    f"request={active_request} sync={sync_index} "
                    f"sync_ms={elapsed_ms:.3f}",
                    flush=True,
                )
            sync_index += 1
            return result

        worker_transfer.torch_dev.synchronize = timed_sync
        original_encode = ctx._encode_group_chunks_into_slots

        def timed_encode(plan, gathered, slots, *args, **kwargs):
            start = time.perf_counter()
            result = original_encode(plan, gathered, slots, *args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if active_request >= 0:
                group_id = (
                    plan.group_info.engine_group_id
                    if plan.group_info is not None
                    else -1
                )
                print(
                    f"request={active_request} chunk group={group_id} "
                    f"encode_ms={elapsed_ms:.3f}",
                    flush=True,
                )
            return result

        ctx._encode_group_chunks_into_slots = timed_encode
        slots = []
        offset = 0
        for plan in ctx._group_plans:
            layout = plan.quant_layout_desc if plan.quantized else plan.raw_layout_desc
            assert layout is not None
            shape = list(layout.shapes[0])
            dtype = str(layout.dtypes[0]).removeprefix("torch.")
            itemsize = torch.empty((), dtype=layout.dtypes[0]).element_size()
            num_bytes = int(torch.Size(shape).numel()) * itemsize
            slots.append(
                {
                    "offset": offset,
                    "length": num_bytes,
                    "shape": shape,
                    "dtype": dtype,
                }
            )
            offset += num_bytes

        future = MagicMock()
        future.wait.return_value = True
        future.result.return_value = PrepareStoreResponse(
            context={
                "slots": slots,
                # The response is flat and group-major: one selected chunk
                # for attention followed by one chunk for each Mamba group.
                "chunk_indices": list(range(len(ctx._group_plans))),
            }
        )
        shm_ctx.req_client.prepare_store.return_value = future
        shm_ctx.req_client.commit_store.return_value = future

        block_ids = [[0] for _ in ctx._group_plans]
        token_ids = tuple(range(_BLOCK_SIZE))

        def run_request(request_idx: int) -> None:
            nonlocal active_request, gather_index, sync_index
            active_request = request_idx
            if request_idx == 0:
                gather_index = 0
                sync_index = 0
            key = IPCCacheServerKey.from_token_ids(
                "qwen3.5-9b", 1, 0, token_ids,
                request_id=f"request-{request_idx}",
            )
            start = time.perf_counter()
            result = ctx.submit_store(
                key.request_id,
                key,
                1,
                kv_caches,
                block_ids,
                None,
                _BLOCKS_PER_CHUNK,
            ).result()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(
                f"submit-store request={request_idx} result={result} "
                f"total_ms={elapsed_ms:.3f}",
                flush=True,
            )

        for warmup_idx in range(warmup):
            run_request(-warmup_idx - 1)
            print(f"submit-store warmup={warmup_idx} done", flush=True)
        for request_idx in range(num_requests):
            run_request(request_idx)
            if request_idx + 1 < num_requests and idle_ms > 0:
                time.sleep(idle_ms / 1000.0)
    finally:
        if "worker_transfer" in locals():
            worker_transfer.gather_paged_kv_to_cpu = original_gather
            worker_transfer.torch_dev.synchronize = original_sync
        shm_munmap(shm_addr, helper._SHM_POOL_SIZE)
        shm_unlink(shm_name)


def _run(
    num_requests: int,
    idle_ms: float,
    warmup: int,
    engine_driven: bool = False,
) -> None:
    groups = [
        (_make_source(_MAMBA_FORMAT), _MAMBA_FORMAT),
        (_make_source(_MAMBA_FORMAT), _MAMBA_FORMAT),
        (_make_source(_MAMBA_FORMAT), _MAMBA_FORMAT),
        (_make_source(_ATTENTION_FORMAT), _ATTENTION_FORMAT),
    ]
    torch_dev.synchronize()

    for warmup_idx in range(warmup):
        if engine_driven:
            _one_engine_driven_request(groups)
        else:
            for source, engine_kv_format in groups:
                _one_transfer(source, engine_kv_format)
        print(f"warmup={warmup_idx} done", flush=True)

    for request_idx in range(num_requests):
        request_start = time.perf_counter()
        if engine_driven:
            _one_engine_driven_request(groups)
            total_ms = (time.perf_counter() - request_start) * 1000.0
            print(f"request={request_idx} total_ms={total_ms:.3f}", flush=True)
            if request_idx + 1 < num_requests and idle_ms > 0:
                time.sleep(idle_ms / 1000.0)
            continue

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
    parser.add_argument(
        "--engine-driven",
        action="store_true",
        help="Use a dedicated XPU copy stream and per-group events.",
    )
    parser.add_argument(
        "--submit-store",
        action="store_true",
        help="Run the real SHM EngineDrivenTransferContext.submit_store path.",
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable KVWeave quantization in --submit-store mode.",
    )
    args = parser.parse_args()
    if args.submit_store:
        _run_submit_store(
            args.requests,
            args.idle_ms,
            args.warmup,
            quantization=not args.no_quantization,
        )
    else:
        _run(args.requests, args.idle_ms, args.warmup, args.engine_driven)
