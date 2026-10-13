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
import threading
import time

import pytest
import torch

from lmcache import torch_dev, torch_device_type
from lmcache.lmcache_native import EngineKVFormat
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, PrepareStoreResponse
from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu


if not (torch_device_type == "xpu" and torch_dev.is_available()):
    pytest.skip("Requires an available XPU runtime", allow_module_level=True)

import lmcache.xpu_ops as xpu_ops  # noqa: E402


_NUM_BLOCKS = 1
_BLOCK_SIZE = 1024
_NUM_HEADS = 4
_HEAD_SIZE = 256
_BLOCK_IDS = [0]
_BLOCKS_PER_CHUNK = 1
_NUM_GROUPS = 4
_MAMBA_FORMAT = EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
_ATTENTION_FORMAT = EngineKVFormat.NL_X_NB_BS_NH_CS

_COPY_MODE_EU = "eu"
_COPY_MODE_MEMCPY = "memcpy"

_TRANSFER_DIRECTION_D2H = 1  # TransferDirection::D2H, see csrc/kv_transfer_types.h
# See test_xpu_eu_vs_memcpy_benchmark.py for why this must be >= the largest
# single per-call transfer size tested here, so each call happens as one
# queue.memcpy() instead of being chopped into many tiny ones.
_HOST_BUFFER_ALIGNMENT = 1 << 26  # 64MB, power of two


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


def _memcpy_reshape_gather(
    source: dict[str, torch.Tensor],
    engine_kv_format: EngineKVFormat,
    output: list[torch.Tensor],
) -> None:
    """Reproduce the EU kernel's copy+reshape via one queue.memcpy() per
    contiguous source slice, instead of one fused SYCL parallel_for kernel.

    See test_xpu_eu_vs_memcpy_benchmark.py for why a bare single memcpy is
    not an equivalent comparison -- this must place each layer (and each
    k_or_v plane, for mamba) at its correct offset in the output buffer to
    do the same reshape the EU kernel does.
    """
    output_ptr = output[0].data_ptr()
    if engine_kv_format != _MAMBA_FORMAT:
        # attention: per-layer source is [1 block, token, head, 2*head_dim]
        # (fused K/V), output is [layer, token, hidden] contiguous.
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
        # mamba: per-layer source is [1 block, 2(k_or_v), token, head,
        # head_dim] -- fixing k_or_v leaves a contiguous
        # [token, head, head_dim] range. Output is [k_or_v, layer, token,
        # hidden] contiguous.
        bytes_per_layer_kv = _BLOCK_SIZE * _NUM_HEADS * _HEAD_SIZE * 2
        layer_stride_bytes = bytes_per_layer_kv
        kv_stride_bytes = 8 * layer_stride_bytes
        for layer_idx, src in enumerate(source.values()):
            for k_or_v in range(2):
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


def _do_gather(
    source: dict[str, torch.Tensor],
    engine_kv_format: EngineKVFormat,
    output: list[torch.Tensor],
    copy_mode: str,
) -> None:
    if copy_mode == _COPY_MODE_MEMCPY:
        _memcpy_reshape_gather(source, engine_kv_format, output)
    else:
        gather_paged_kv_to_cpu(
            source,
            _BLOCK_IDS,
            _BLOCKS_PER_CHUNK,
            engine_kv_format=engine_kv_format,
            out=output,
        )


def _one_transfer(
    source: dict[str, torch.Tensor],
    engine_kv_format: EngineKVFormat,
    copy_mode: str = _COPY_MODE_EU,
) -> tuple[float, float]:
    output = _make_output(engine_kv_format)
    call_start = time.perf_counter()
    _do_gather(source, engine_kv_format, output, copy_mode)
    call_ms = (time.perf_counter() - call_start) * 1000.0
    sync_start = time.perf_counter()
    torch_dev.synchronize()
    sync_ms = (time.perf_counter() - sync_start) * 1000.0
    return call_ms, sync_ms


def _one_engine_driven_request(
    groups: list[tuple[dict[str, torch.Tensor], EngineKVFormat]],
    copy_mode: str = _COPY_MODE_EU,
) -> None:
    copy_stream = torch.xpu.Stream()
    pending: list[tuple[int, EngineKVFormat, float, torch.xpu.Event]] = []
    with torch.xpu.stream(copy_stream):
        for group_idx, (source, engine_kv_format) in enumerate(groups):
            output = _make_output(engine_kv_format)
            call_start = time.perf_counter()
            _do_gather(source, engine_kv_format, output, copy_mode)
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
    use_async: bool = False,
    pipeline: bool = False,
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
    # Shared origin so every printed timestamp can be reconstructed as a
    # single wall-clock timeline (offset_ms) when parsing the log later,
    # instead of guessing request/group start times from durations alone.
    t0 = time.perf_counter()

    def offset_ms() -> float:
        return (time.perf_counter() - t0) * 1000.0

    try:
        original_groups = helper._qwen35_9b_groups

        def mamba_first_groups():
            groups = original_groups()
            return groups[1:] + groups[:1]

        helper._qwen35_9b_groups = mamba_first_groups
        ctx, shm_ctx = helper._register_shm_context(
            kv_caches, shm_name, helper._SHM_POOL_SIZE, use_async=use_async
        )
        worker_transfer = __import__(
            "lmcache.v1.multiprocess.transfer_context.worker_transfer",
            fromlist=["gather_paged_kv_to_cpu"],
        )
        async_engine_driven = __import__(
            "lmcache.v1.multiprocess.transfer_context.async_engine_driven",
            fromlist=["gather_paged_kv_to_cpu"],
        )
        original_gather = worker_transfer.gather_paged_kv_to_cpu
        # async_engine_driven.py imports its own independent module-level
        # binding of gather_paged_kv_to_cpu from base.py; patching
        # worker_transfer's binding alone never affects calls made from
        # _launch_shm_quantized_batch (the bounded-pipeline gather path).
        original_async_gather = async_engine_driven.gather_paged_kv_to_cpu
        original_sync = worker_transfer.torch_dev.synchronize
        # In pipeline mode, multiple requests' background gather/split/encode
        # threads run concurrently, so "the active request" is per-thread,
        # not a single shared counter.
        active_request_by_thread: dict[int, int] = {}
        gather_index = 0
        sync_index = 0

        def active_request() -> int:
            return active_request_by_thread.get(threading.get_ident(), -1)

        def _make_timed_gather(original):
            def timed_gather(*args, **kwargs):
                nonlocal gather_index
                start_offset_ms = offset_ms()
                start = time.perf_counter()
                result = original(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if active_request() >= 0:
                    engine_format = kwargs.get("engine_kv_format")
                    chunk_indices = kwargs.get("chunk_indices")
                    print(
                        f"request={active_request()} chunk={gather_index} "
                        f"format={engine_format.name if engine_format else 'unknown'} "
                        f"chunk_indices={chunk_indices} "
                        f"phase=launch start_offset_ms={start_offset_ms:.3f} "
                        f"end_offset_ms={offset_ms():.3f} "
                        f"gather_call_ms={elapsed_ms:.3f}",
                        flush=True,
                    )
                gather_index += 1
                return result

            return timed_gather

        worker_transfer.gather_paged_kv_to_cpu = _make_timed_gather(original_gather)
        async_engine_driven.gather_paged_kv_to_cpu = _make_timed_gather(
            original_async_gather
        )

        def timed_sync(*args, **kwargs):
            nonlocal sync_index
            start_offset_ms = offset_ms()
            start = time.perf_counter()
            result = original_sync(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if active_request() >= 0:
                print(
                    f"request={active_request()} sync={sync_index} "
                    f"phase=wait start_offset_ms={start_offset_ms:.3f} "
                    f"end_offset_ms={offset_ms():.3f} "
                    f"sync_ms={elapsed_ms:.3f}",
                    flush=True,
                )
            sync_index += 1
            return result

        worker_transfer.torch_dev.synchronize = timed_sync

        # Patched at the Event *class* level (not a module-level function
        # binding) so it covers every torch_dev.Event() instance created by
        # _launch_shm_quantized_batch, including the bounded pipeline's
        # per-chunk event.synchronize() -- the actual D2H-wait that overlaps
        # (or fails to overlap) with the previous chunk's split/encode.
        event_cls = type(torch_dev.Event())
        original_event_synchronize = event_cls.synchronize
        event_sync_index = 0

        def timed_event_synchronize(self):
            nonlocal event_sync_index
            start_offset_ms = offset_ms()
            start = time.perf_counter()
            result = original_event_synchronize(self)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if active_request() >= 0:
                print(
                    f"request={active_request()} event_sync={event_sync_index} "
                    f"phase=event_wait start_offset_ms={start_offset_ms:.3f} "
                    f"end_offset_ms={offset_ms():.3f} "
                    f"event_wait_ms={elapsed_ms:.3f}",
                    flush=True,
                )
            event_sync_index += 1
            return result

        event_cls.synchronize = timed_event_synchronize

        kvweave_codec = ctx._kvweave_codec
        original_split_mamba = kvweave_codec.split_mamba_chunk
        original_split_fused = kvweave_codec.split_fused_attention_heads
        split_index = 0

        def timed_split_mamba(*args, **kwargs):
            nonlocal split_index
            start_offset_ms = offset_ms()
            start = time.perf_counter()
            result = original_split_mamba(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if active_request() >= 0:
                print(
                    f"request={active_request()} split={split_index} kind=mamba "
                    f"phase=split start_offset_ms={start_offset_ms:.3f} "
                    f"end_offset_ms={offset_ms():.3f} "
                    f"split_ms={elapsed_ms:.3f}",
                    flush=True,
                )
            split_index += 1
            return result

        def timed_split_fused(*args, **kwargs):
            nonlocal split_index
            start_offset_ms = offset_ms()
            start = time.perf_counter()
            result = original_split_fused(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if active_request() >= 0:
                print(
                    f"request={active_request()} split={split_index} kind=fused_attention "
                    f"phase=split start_offset_ms={start_offset_ms:.3f} "
                    f"end_offset_ms={offset_ms():.3f} "
                    f"split_ms={elapsed_ms:.3f}",
                    flush=True,
                )
            split_index += 1
            return result

        kvweave_codec.split_mamba_chunk = timed_split_mamba
        kvweave_codec.split_fused_attention_heads = timed_split_fused

        original_encode = ctx._encode_group_chunks_into_slots

        def timed_encode(plan, gathered, slots, *args, **kwargs):
            start_offset_ms = offset_ms()
            start = time.perf_counter()
            result = original_encode(plan, gathered, slots, *args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if active_request() >= 0:
                group_id = (
                    plan.group_info.engine_group_id
                    if plan.group_info is not None
                    else -1
                )
                print(
                    f"request={active_request()} chunk group={group_id} "
                    f"phase=encode start_offset_ms={start_offset_ms:.3f} "
                    f"end_offset_ms={offset_ms():.3f} "
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

        if pipeline and not use_async:
            raise ValueError("pipeline mode requires use_async=True")
        if use_async:
            # submit_store's background gather/split/encode work always runs
            # on a commit_executor worker thread, not the thread that called
            # submit_store (true whether or not --pipeline overlaps multiple
            # requests), so tag the worker thread itself -- not the calling
            # thread -- for the duration of that request's task.
            original_commit_submit = ctx._commit_executor.submit

            def tagging_submit(fn, *args, **kwargs):
                request_idx = active_request_by_thread.get(
                    threading.get_ident(), -1
                )

                def tagged_fn(*fn_args, **fn_kwargs):
                    active_request_by_thread[threading.get_ident()] = request_idx
                    try:
                        return fn(*fn_args, **fn_kwargs)
                    finally:
                        del active_request_by_thread[threading.get_ident()]

                return original_commit_submit(tagged_fn, *args, **kwargs)

            ctx._commit_executor.submit = tagging_submit

        def submit_request(request_idx: int):
            nonlocal gather_index, sync_index, split_index
            active_request_by_thread[threading.get_ident()] = request_idx
            if request_idx == 0:
                gather_index = 0
                sync_index = 0
                split_index = 0
            key = IPCCacheServerKey.from_token_ids(
                "qwen3.5-9b", 1, 0, token_ids,
                request_id=f"request-{request_idx}",
            )
            start_offset_ms = offset_ms()
            print(
                f"submit-store request={request_idx} "
                f"start_offset_ms={start_offset_ms:.3f}",
                flush=True,
            )
            # AsyncEngineDrivenTransferContext.submit_store requires a real
            # local ordering event (raises RuntimeError on None); the
            # synchronous context accepts None.
            ordering_event = (
                ctx.create_recorded_event() if use_async else None
            )
            future = ctx.submit_store(
                key.request_id,
                key,
                1,
                kv_caches,
                block_ids,
                ordering_event,
                _BLOCKS_PER_CHUNK,
            )
            return future

        def await_request(request_idx: int, future, start: float) -> None:
            result = future.result()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(
                f"submit-store request={request_idx} result={result} "
                f"end_offset_ms={offset_ms():.3f} "
                f"total_ms={elapsed_ms:.3f}",
                flush=True,
            )

        def run_request(request_idx: int) -> None:
            start = time.perf_counter()
            future = submit_request(request_idx)
            await_request(request_idx, future, start)

        for warmup_idx in range(warmup):
            run_request(-warmup_idx - 1)
            print(f"submit-store warmup={warmup_idx} done", flush=True)

        if pipeline:
            # Fire every request's submit_store from this thread first --
            # each call only does O(1) work before handing off to
            # commit_executor -- then wait on all futures, so requests'
            # background gather/split/encode/commit work overlaps instead of
            # being serialized by a blocking .result() per request.
            starts = []
            futures = []
            for request_idx in range(num_requests):
                starts.append(time.perf_counter())
                futures.append(submit_request(request_idx))
                if request_idx + 1 < num_requests and idle_ms > 0:
                    time.sleep(idle_ms / 1000.0)
            for request_idx, (future, start) in enumerate(zip(futures, starts)):
                await_request(request_idx, future, start)
        else:
            for request_idx in range(num_requests):
                run_request(request_idx)
                if request_idx + 1 < num_requests and idle_ms > 0:
                    time.sleep(idle_ms / 1000.0)
    finally:
        if "original_commit_submit" in locals():
            ctx._commit_executor.submit = original_commit_submit
        if "worker_transfer" in locals():
            worker_transfer.gather_paged_kv_to_cpu = original_gather
            worker_transfer.torch_dev.synchronize = original_sync
        if "async_engine_driven" in locals():
            async_engine_driven.gather_paged_kv_to_cpu = original_async_gather
        if "event_cls" in locals():
            event_cls.synchronize = original_event_synchronize
        if "kvweave_codec" in locals():
            kvweave_codec.split_mamba_chunk = original_split_mamba
            kvweave_codec.split_fused_attention_heads = original_split_fused
        shm_munmap(shm_addr, helper._SHM_POOL_SIZE)
        shm_unlink(shm_name)


def _run(
    num_requests: int,
    idle_ms: float,
    warmup: int,
    engine_driven: bool = False,
    copy_mode: str = _COPY_MODE_EU,
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
            _one_engine_driven_request(groups, copy_mode)
        else:
            for source, engine_kv_format in groups:
                _one_transfer(source, engine_kv_format, copy_mode)
        print(f"warmup={warmup_idx} done", flush=True)

    for request_idx in range(num_requests):
        request_start = time.perf_counter()
        if engine_driven:
            _one_engine_driven_request(groups, copy_mode)
            total_ms = (time.perf_counter() - request_start) * 1000.0
            print(f"request={request_idx} total_ms={total_ms:.3f}", flush=True)
            if request_idx + 1 < num_requests and idle_ms > 0:
                time.sleep(idle_ms / 1000.0)
            continue

        group_timings = []
        for group_idx, (source, engine_kv_format) in enumerate(groups):
            call_ms, sync_ms = _one_transfer(source, engine_kv_format, copy_mode)
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


def test_reverse_group_gathers_keep_block_ids_isolated() -> None:
    """Queued D2H calls must not observe a later group's block-ID mapping."""
    block_size = 256
    mamba_source = torch.empty(
        2,
        2,
        block_size,
        _NUM_HEADS,
        _HEAD_SIZE,
        dtype=torch.float16,
        device="xpu",
    )
    mamba_source[0].fill_(10)
    mamba_source[1].fill_(11)
    attention_source = torch.empty(
        2,
        block_size,
        _NUM_HEADS,
        2 * _HEAD_SIZE,
        dtype=torch.float16,
        device="xpu",
    )
    attention_source[0].fill_(20)
    attention_source[1].fill_(21)
    torch_dev.synchronize()

    mamba_caches = {"layer_0": mamba_source}
    attention_caches = {"layer_0": attention_source}
    attention_out = torch.empty(
        1,
        block_size,
        _NUM_HEADS * 2 * _HEAD_SIZE,
        dtype=torch.float16,
        pin_memory=True,
    )
    mamba_outputs = [
        torch.empty(
            2,
            1,
            block_size,
            _NUM_HEADS * _HEAD_SIZE,
            dtype=torch.float16,
            pin_memory=True,
        )
        for _ in range(3)
    ]

    gather_paged_kv_to_cpu(
        attention_caches,
        [1],
        1,
        engine_kv_format=_ATTENTION_FORMAT,
        out=[attention_out],
    )
    for block_id, output in zip((0, 1, 0), mamba_outputs, strict=True):
        gather_paged_kv_to_cpu(
            mamba_caches,
            [block_id],
            1,
            engine_kv_format=_MAMBA_FORMAT,
            out=[output],
        )
    torch_dev.synchronize()

    assert torch.unique(attention_out).tolist() == [21.0]
    assert [torch.unique(output).tolist() for output in mamba_outputs] == [
        [10.0],
        [11.0],
        [10.0],
    ]


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
    parser.add_argument(
        "--async-engine-driven",
        action="store_true",
        help=(
            "In --submit-store mode, register an "
            "AsyncEngineDrivenTransferContext instead of the synchronous "
            "EngineDrivenTransferContext, so the trace exercises the "
            "cross-group D2H/split/encode pipeline "
            "(async_engine_driven.py) instead of the synchronous bounded "
            "pipeline (worker_transfer.py)."
        ),
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help=(
            "In --submit-store --async-engine-driven mode, submit every "
            "request's submit_store up front and wait on all futures "
            "afterwards, instead of blocking on .result() per request. "
            "Exercises cross-request overlap on the commit_executor thread "
            "pool; requires --async-engine-driven."
        ),
    )
    parser.add_argument(
        "--copy-mode",
        choices=[_COPY_MODE_EU, _COPY_MODE_MEMCPY],
        default=_COPY_MODE_EU,
        help=(
            "Gather implementation for --requests/--engine-driven mode: "
            "'eu' uses the production SYCL parallel_for gather kernel "
            "(gather_paged_kv_to_cpu); 'memcpy' reproduces the same "
            "copy+reshape via per-layer queue.memcpy() calls instead, for "
            "comparison (see test_xpu_eu_vs_memcpy_benchmark.py). Ignored "
            "with --submit-store, which always exercises the real "
            "production path.",
        ),
    )
    args = parser.parse_args()
    if args.pipeline and not args.async_engine_driven:
        parser.error("--pipeline requires --async-engine-driven")
    if args.submit_store:
        _run_submit_store(
            args.requests,
            args.idle_ms,
            args.warmup,
            quantization=not args.no_quantization,
            use_async=args.async_engine_driven,
            pipeline=args.pipeline,
        )
    else:
        _run(
            args.requests,
            args.idle_ms,
            args.warmup,
            args.engine_driven,
            args.copy_mode,
        )
