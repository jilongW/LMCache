# SPDX-License-Identifier: Apache-2.0
"""Single-group (dense attention, non-hybrid) engine-driven store timing.

Qwen3-8B has no Mamba/linear-attention layers, so vLLM's
``kv_cache_config.kv_cache_groups`` reports exactly one ``KVCacheGroupSpec``
covering all 36 attention layers; ``create_engine_group_infos_from_vllm``
(``lmcache/integration/vllm/kv_cache_groups.py``) turns that into a single
explicit ``EngineGroupInfo(cache_category="attention", ...)`` -- NOT an
empty ``engine_group_infos=[]`` list. The empty-list path is reserved for
engines that report no KV cache group metadata at all and collapses to
``group_info=None``, which ``_attention_plane_layout_for_group`` treats as
non-attention and never quantizes -- so this script registers the same
explicit single-group list production actually sends, matching Qwen3.5-9B's
hybrid 4-group registration pattern
(``test_xpu_engine_driven_first_transfer.py``) but with 1 group instead of 4.

This matters for ``_env_num_threads``'s few-groups doubling
(``lmcache/v1/distributed/serde/kvweave/kvweave_config.py``): with only one
group, the bounded store pipeline has no other chunk's D2H gather to overlap
against, so there's no cross-chunk CPU parallelism opportunity within one
``submit_store`` call -- the few-groups threshold exists precisely to let a
single chunk's native OpenMP encode use more cores in this case.

Run directly for visible timing output::

    python tests/benchmarks/bench_qwen3_8b_single_group.py --requests 4 --warmup 1

Use ``--pipeline`` (requires ``--async-engine-driven``) to submit every
request up front and wait on all futures afterwards, exercising
cross-request overlap instead of per-request blocking -- see
``test_xpu_engine_driven_first_transfer.py``'s ``--pipeline`` flag for the
equivalent on the Qwen3.5-9B hybrid geometry.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from unittest.mock import MagicMock

import pytest
import torch

from lmcache import torch_dev, torch_device_type
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, PrepareStoreResponse
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.posix_shm import (
    shm_create_readwrite,
    shm_munmap,
    shm_unlink,
)
from lmcache.v1.multiprocess.protocols.engine import (
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
    AsyncEngineDrivenTransferContext,
)
from lmcache.v1.multiprocess.transfer_context.shm import EngineDrivenContextShm
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)


if not (torch_device_type == "xpu" and torch_dev.is_available()):
    pytest.skip("Requires an available XPU runtime", allow_module_level=True)


# Qwen3-8B real dense-attention geometry (see config.json: num_hidden_layers,
# num_key_value_heads, head_dim). No Mamba/linear-attention layers, so vLLM
# reports exactly one KV cache group covering all layers (see module
# docstring) -- a single explicit EngineGroupInfo, not an empty list.
_NUM_LAYERS = 36
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_BLOCK_SIZE = 1024
_BLOCKS_PER_CHUNK = 1
_SHM_QUANTIZED_REGION_BYTES = 128 * 1024 * 1024
_SHM_RAW_CHUNK_OFFSET = _SHM_QUANTIZED_REGION_BYTES
_SHM_POOL_SIZE = 512 * 1024 * 1024


def _qwen3_8b_groups() -> list[EngineGroupInfo]:
    """The single attention group vLLM reports for a dense (non-hybrid) model."""
    return [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=tuple(range(_NUM_LAYERS)),
            tokens_per_block=_BLOCK_SIZE,
            cache_category="attention",
        ),
    ]


def _qwen3_8b_kv_caches() -> dict[str, torch.Tensor]:
    """Paged fused-K/V attention tensors, one per Qwen3-8B layer."""
    shape = (_BLOCKS_PER_CHUNK, _BLOCK_SIZE, _NUM_KV_HEADS, 2 * _HEAD_DIM)
    return {
        f"attn_layer_{i}": torch.zeros(shape, dtype=torch.float16, device="xpu")
        for i in range(_NUM_LAYERS)
    }


def _register_shm_context(
    kv_caches: dict[str, torch.Tensor],
    shm_name: str,
    pool_size: int,
    use_async: bool,
) -> tuple[EngineDrivenTransferContext, EngineDrivenContextShm]:
    """Register a real ``EngineDrivenContextShm`` with a mocked RPC client.

    Mirrors ``bench_engine_driven_shm_timing._register_shm_context`` but with
    the single explicit ``EngineGroupInfo`` (see ``_qwen3_8b_groups``) a
    dense, non-hybrid model like Qwen3-8B actually registers.
    """
    created_shm_contexts: list[EngineDrivenContextShm] = []

    def _create_shm_context(
        metadata, *_args: object, **_kwargs: object
    ) -> EngineDrivenContextShm:
        shm_context = EngineDrivenContextShm(
            metadata=metadata,
            req_client=MagicMock(),
            mq_timeout=5.0,
            shm_name=shm_name,
            pool_size=pool_size,
            scratch_offset=_SHM_RAW_CHUNK_OFFSET,
            scratch_size=_SHM_POOL_SIZE - _SHM_RAW_CHUNK_OFFSET,
        )
        created_shm_contexts.append(shm_context)
        return shm_context

    original_factory = worker_transfer.create_engine_driven_context
    worker_transfer.create_engine_driven_context = _create_shm_context
    try:
        future = MagicMock()
        future.result.return_value = RegisterEngineDrivenContextResponse(
            shm_name=shm_name,
            pool_size=pool_size,
            scratch_offset=_SHM_RAW_CHUNK_OFFSET,
            scratch_size=_SHM_POOL_SIZE - _SHM_RAW_CHUNK_OFFSET,
        )
        req_client = MagicMock()
        req_client.register_kv_cache_engine_driven_context.return_value = future

        ctx = (
            AsyncEngineDrivenTransferContext()
            if use_async
            else EngineDrivenTransferContext()
        )
        ctx.register(
            instance_id=1,
            kv_caches=kv_caches,
            model_name="qwen3-8b",
            world_size=1,
            blocks_in_chunk=_BLOCKS_PER_CHUNK,
            req_client=req_client,
            mq_timeout=5.0,
            engine_group_infos=_qwen3_8b_groups(),
        )
        return ctx, created_shm_contexts[0]
    finally:
        worker_transfer.create_engine_driven_context = original_factory


def _run_submit_store(
    num_requests: int,
    idle_ms: float,
    warmup: int,
    quantization: bool,
    use_async: bool,
    pipeline: bool,
) -> None:
    if quantization:
        os.environ.setdefault("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    if pipeline and not use_async:
        raise ValueError("pipeline mode requires use_async=True")

    kv_caches = _qwen3_8b_kv_caches()
    shm_name = f"lmcache_qwen3_8b_single_group_{os.getpid()}"
    shm_addr = shm_create_readwrite(shm_name, _SHM_POOL_SIZE)
    t0 = time.perf_counter()

    def offset_ms() -> float:
        return (time.perf_counter() - t0) * 1000.0

    try:
        ctx, shm_ctx = _register_shm_context(
            kv_caches, shm_name, _SHM_POOL_SIZE, use_async=use_async
        )
        print(
            f"registered num_groups={len(ctx._engine_group_infos)} "
            f"num_threads={ctx._kvweave_codec.num_threads}",
            flush=True,
        )

        active_request_by_thread: dict[int, int] = {}

        def active_request() -> int:
            return active_request_by_thread.get(threading.get_ident(), -1)

        original_encode = ctx._encode_group_chunks_into_slots

        def timed_encode(plan, gathered, slots, *args, **kwargs):
            start_offset_ms = offset_ms()
            start = time.perf_counter()
            result = original_encode(plan, gathered, slots, *args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if active_request() >= 0:
                print(
                    f"request={active_request()} phase=encode "
                    f"start_offset_ms={start_offset_ms:.3f} "
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
                {"offset": offset, "length": num_bytes, "shape": shape, "dtype": dtype}
            )
            offset += num_bytes

        future = MagicMock()
        future.wait.return_value = True
        future.result.return_value = PrepareStoreResponse(
            context={
                "slots": slots,
                "chunk_indices": list(range(len(ctx._group_plans))),
            }
        )
        shm_ctx.req_client.prepare_store.return_value = future
        shm_ctx.req_client.commit_store.return_value = future

        block_ids = [[0] for _ in ctx._group_plans]
        token_ids = tuple(range(_BLOCK_SIZE))

        if use_async:
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
            active_request_by_thread[threading.get_ident()] = request_idx
            key = IPCCacheServerKey.from_token_ids(
                "qwen3-8b", 1, 0, token_ids, request_id=f"request-{request_idx}"
            )
            print(
                f"submit-store request={request_idx} "
                f"start_offset_ms={offset_ms():.3f}",
                flush=True,
            )
            ordering_event = ctx.create_recorded_event() if use_async else None
            return ctx.submit_store(
                key.request_id,
                key,
                1,
                kv_caches,
                block_ids,
                ordering_event,
                _BLOCKS_PER_CHUNK,
            )

        def await_request(request_idx: int, future, start: float) -> None:
            result = future.result()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            print(
                f"submit-store request={request_idx} result={result} "
                f"end_offset_ms={offset_ms():.3f} total_ms={elapsed_ms:.3f}",
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
        shm_munmap(shm_addr, _SHM_POOL_SIZE)
        shm_unlink(shm_name)


def test_qwen3_8b_single_group_store_timing() -> None:
    """Keep pytest discovery useful while preserving direct-run output."""
    _run_submit_store(
        num_requests=2, idle_ms=0.0, warmup=0, quantization=True, use_async=False,
        pipeline=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--idle-ms", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no-quantization", action="store_true")
    parser.add_argument("--async-engine-driven", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    if args.pipeline and not args.async_engine_driven:
        parser.error("--pipeline requires --async-engine-driven")
    _run_submit_store(
        args.requests,
        args.idle_ms,
        args.warmup,
        quantization=not args.no_quantization,
        use_async=args.async_engine_driven,
        pipeline=args.pipeline,
    )
