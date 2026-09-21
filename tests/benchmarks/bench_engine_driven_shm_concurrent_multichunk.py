# SPDX-License-Identifier: Apache-2.0
"""Concurrent multi-chunk engine-driven SHM store correctness + timing bench.

Manual timing script (run directly with ``python``, not through pytest):
exercises :class:`AsyncEngineDrivenTransferContext.submit_store` the way real
vLLM usage does -- one non-hybrid, non-quantized attention group registered
once, then many independent requests each calling ``submit_store`` from the
"forward thread" for its own token range. Per
``scratch_shm_raw_buffer_plan.md`` section 4.2: ``submit_store`` only does
O(1) work on the calling thread before handing gather (GPU->CPU) and commit
off to the background ``commit_executor`` (a
:class:`~concurrent.futures.ThreadPoolExecutor`,
``DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS = 4`` by default), so multiple
requests' gather/commit for the *same* registered group genuinely run
concurrently -- this is the scenario this script drives and measures.

Each of ``_NUM_REQUESTS`` simulated requests gets:
  * its own :class:`IPCCacheServerKey` / request id,
  * its own disjoint slice of physical blocks in the (shared, per-layer)
    device KV cache tensors, matching different sequences occupying
    disjoint block ids in a real paged KV cache pool,
  * its own disjoint region of the mocked SHM pool (mocked
    ``prepare_store``/``prepare_retrieve`` hand back a distinct offset per
    request, exactly like the real server would for concurrent, unrelated
    stores), and
  * ``_CHUNKS_PER_REQUEST`` chunks in a single ``submit_store`` call, i.e. a
    long-prefill-style multi-chunk store rather than one chunk at a time.

After every request's store future resolves, this script round-trips each
request's data back out with a synchronous ``submit_retrieve`` (scatter into
a fresh device tensor) and compares it against the original per-block data
written before the store. This is the correctness check: if concurrent
gather/commit ever raced and one request's bytes clobbered another's SHM
region, the corrupted request's retrieved data would no longer match its
own original input.

Also times ``_NUM_REQUESTS`` concurrent ``submit_store`` calls (fire all,
then wait for all futures) against the same ``_NUM_REQUESTS`` stores run
strictly one-at-a-time (submit, wait, submit next), to quantify the overlap
the ``commit_executor`` thread pool provides for multi-chunk stores.

Like ``bench_engine_driven_shm_timing.py``, the whole run above is executed
twice: once with the group unquantized (registration's real, automatic
decision for a non-hybrid group -- always unquantized), and once with
quantization forced on. Forcing it on bypasses the registration-time model
text-config lookup that ``KVWeaveRuntimeConfig.from_env()`` would otherwise
need (this script never talks to a real model): the group's cached
``GroupTransferPlan`` is patched in place to ``quantized=True`` with a
manually built fused-K/V attention codec (mirroring
``bench_engine_driven_shm_timing.py``'s ``_attention_codec()``), so the
*real* ``submit_store``/``submit_retrieve`` quantize/dequantize dispatch
(``EngineDrivenTransferContext._encode_group_chunks_into_slots``/
``_decode_group_chunks``) runs for real under concurrency, not a manually
invoked standalone call. The quantized store path gathers raw chunks into
pool scratch, encodes directly into durable SHM slots, then returns raw
scratch before ``commit_store``; retrieve decodes to raw scratch, releases
the compact read slots, then scatters to device.

Requires a real XPU device (exits early otherwise). Run with:

    python tests/benchmarks/bench_engine_driven_shm_concurrent_multichunk.py
"""

# Standard
from dataclasses import replace
from unittest.mock import MagicMock
import os
import statistics
import threading
import time

# Third Party
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.serde.kvweave.kvweave_config import AttentionPlaneLayout
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import _KVWeaveCodec
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.posix_shm import (
    shm_create_readwrite,
    shm_munmap,
    shm_unlink,
)
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transfer_context import async_engine_driven, worker_transfer
from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
    AsyncEngineDrivenTransferContext,
    DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS,
)
from lmcache.v1.multiprocess.transfer_context.shm import EngineDrivenContextShm
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)

try:
    from kvweave import kvweave_quant  # noqa: F401

    _KVWEAVE_QUANT_AVAILABLE = True
except ImportError:
    _KVWEAVE_QUANT_AVAILABLE = False


class _CompletedFuture:
    """Minimal already-resolved future, matching the mocked-RPC test pattern."""

    def __init__(self, value):
        self._value = value

    def wait(self, timeout=None):  # noqa: ARG002
        return True

    def result(self, timeout=None):  # noqa: ARG002
        return self._value


# Small, deliberately non-quantized single-group attention geometry: one
# real vLLM non-MLA fused-K/V layout ([NB, BS, NH, CS]), same shape family as
# bench_engine_driven_shm_timing.py's ``_attention_kv_caches`` but scaled to
# hold many requests' disjoint block ranges instead of just one chunk.
_NUM_LAYERS = 8
_NUM_KV_HEADS = 4
_HEAD_DIM = 64
_BLOCK_SIZE = 16
_BLOCKS_IN_CHUNK = 1
_CHUNK_SIZE_TOKENS = _BLOCKS_IN_CHUNK * _BLOCK_SIZE

# Concurrency knobs. ``_NUM_REQUESTS`` is deliberately larger than
# ``DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS`` so some requests must queue behind
# the commit_executor's worker threads, matching a bursty real workload.
_NUM_REQUESTS = 2 * DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS
# Chunks in a single submit_store call -- simulates one long-prefill store
# rather than one chunk at a time (see scratch_shm_raw_buffer_plan.md's
# ``chunks_per_call`` discussion).
_CHUNKS_PER_REQUEST = 8
_BLOCKS_PER_REQUEST = _CHUNKS_PER_REQUEST * _BLOCKS_IN_CHUNK
_TOTAL_BLOCKS = _NUM_REQUESTS * _BLOCKS_PER_REQUEST

_NUM_ITERATIONS = 5
_NUM_WARMUP = 1
_MQ_TIMEOUT_S = 5.0
# Real scratch region size for the quantized scenario's Stage 1 gather
# target, per scratch_shm_raw_buffer_plan.md's proposed default budget.
_SCRATCH_SIZE_BYTES = 5 * 1024 * 1024 * 1024


def _attention_codec() -> _KVWeaveCodec:
    """Manually built fused-K/V codec sized to this script's geometry.

    Bypasses ``KVWeaveRuntimeConfig.from_env()``'s model text-config lookup
    (same reason ``bench_engine_driven_shm_timing.py``'s ``_attention_codec()``
    does this) since this script never registers a real model.
    """
    return _KVWeaveCodec(
        {
            "quantize": True,
            "qbit": 4,
            "scaling_method": "per_channel",
            "rh": False,
            "asym": True,
            "block_size": _BLOCK_SIZE,
            "num_kv_heads": _NUM_KV_HEADS,
            "head_dim": _HEAD_DIM,
        }
    )


def _attention_kv_caches() -> dict[str, torch.Tensor]:
    """Real per-layer fused-K/V paged tensors, ``_TOTAL_BLOCKS`` blocks each.

    Big enough to give every simulated request its own disjoint block range
    (see :func:`_request_block_ids`).
    """
    shape = (_TOTAL_BLOCKS, _BLOCK_SIZE, _NUM_KV_HEADS, 2 * _HEAD_DIM)
    return {
        f"attn_layer_{i}": torch.randn(
            shape, dtype=torch.float16, device=torch_device_type
        )
        for i in range(_NUM_LAYERS)
    }


def _request_block_ids(request_idx: int) -> list[int]:
    """This request's disjoint slice of physical blocks."""
    start = request_idx * _BLOCKS_PER_REQUEST
    return list(range(start, start + _BLOCKS_PER_REQUEST))


def _request_key(request_idx: int) -> IPCCacheServerKey:
    """Distinct key per request: distinct token ids and request id."""
    tokens = _CHUNKS_PER_REQUEST * _BLOCK_SIZE
    token_ids = [request_idx + 1] * tokens
    return IPCCacheServerKey.from_token_ids(
        "concurrent-multichunk-model",
        1,
        0,
        token_ids,
        start=0,
        end=tokens,
        request_id=f"req-{request_idx}",
    )


def _enable_quantization(
    ctx: AsyncEngineDrivenTransferContext, attention_codec: _KVWeaveCodec
) -> int:
    """Force the registered (single, non-hybrid) group to be quantized.

    Patches ``ctx._group_plans[0]`` in place to ``quantized=True`` with a
    real ``cache_category="attention"``/``FUSED_KV`` classification, and
    swaps in a manually built codec (see :func:`_attention_codec`) so the
    real direct-slot encode/decode dispatch used by
    ``submit_store``/``submit_retrieve`` actually quantizes -- rather than
    calling the codec by hand outside the real store/retrieve path.

    Returns the quantized wire payload's byte length (from a one-off encode
    of a dummy chunk; deterministic for fixed shape/qbit), used to size the
    mocked SHM slots.
    """
    plan = ctx._group_plans[0]  # noqa: SLF001 -- internal, mirrors other tests
    quantized_group_info = EngineGroupInfo(
        engine_group_id=0,
        layer_indices=tuple(range(_NUM_LAYERS)),
        tokens_per_block=_BLOCK_SIZE,
        cache_category="attention",
    )
    ctx._group_plans[0] = replace(  # noqa: SLF001
        plan,
        group_info=quantized_group_info,
        quantized=True,
        attention_plane_layout=AttentionPlaneLayout.FUSED_KV,
    )
    ctx._kvweave_codec = attention_codec  # noqa: SLF001

    dummy_chunk = torch.zeros(plan.chunk_shape, dtype=plan.raw_layout_desc.dtypes[0])
    payload = attention_codec.encode_chunk(
        "attention", None, _BLOCK_SIZE, None, dummy_chunk, AttentionPlaneLayout.FUSED_KV
    )
    return len(payload)


class _RequestSlots:
    """One request's disjoint SHM byte region, sliced into per-chunk slots."""

    def __init__(self, offset: int, chunk_shape: list[int], dtype_str: str, chunk_bytes: int):
        self.slots = [
            {
                "offset": offset + chunk_idx * chunk_bytes,
                "length": chunk_bytes,
                "shape": chunk_shape,
                "dtype": dtype_str,
            }
            for chunk_idx in range(_CHUNKS_PER_REQUEST)
        ]


def _register_shm_context(
    kv_caches: dict[str, torch.Tensor],
    shm_name: str,
    pool_size: int,
    commit_workers: int,
    *,
    scratch_offset: int = 0,
    scratch_size: int = 0,
) -> tuple[AsyncEngineDrivenTransferContext, EngineDrivenContextShm, MagicMock]:
    """Register a real ``EngineDrivenContextShm`` with a mocked RPC client.

    Only the RPC layer is mocked; the SHM segment, its pinning, and every
    tensor-view read/write are real (same pattern as
    ``bench_engine_driven_shm_timing.py``'s ``_register_shm_context``).
    ``scratch_offset``/``scratch_size`` carve out a real scratch region
    quantized groups' Stage 1 gather can use instead of falling back to
    ``torch.empty`` (see ``scratch_shm_raw_buffer_plan.md``); ``0``/``0``
    (the default) reproduces the "no scratch configured" degraded path.
    Returns the transfer context, the underlying SHM context, and the
    mocked ``req_client`` so the caller can wire per-request
    ``prepare_store``/``prepare_retrieve`` side effects once slot offsets are
    known (they depend on ``plan.chunk_shape``, only available after
    registration).
    """
    created_shm_contexts: list[EngineDrivenContextShm] = []

    def _create_shm_context(
        metadata, passed_req_client, mq_timeout, *_args, **_kwargs
    ) -> EngineDrivenContextShm:
        # Must reuse the same req_client register() was called with (not a
        # fresh MagicMock()), since _wire_request_slots() configures this
        # exact mock's prepare_store/commit_store side effects afterwards.
        shm_context = EngineDrivenContextShm(
            metadata=metadata,
            req_client=passed_req_client,
            mq_timeout=mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
            scratch_offset=scratch_offset,
            scratch_size=scratch_size,
        )
        created_shm_contexts.append(shm_context)
        return shm_context

    original_factory = worker_transfer.create_engine_driven_context
    worker_transfer.create_engine_driven_context = _create_shm_context
    try:
        req_client = MagicMock()
        future = MagicMock()
        future.result.return_value = RegisterEngineDrivenContextResponse(
            shm_name=shm_name,
            pool_size=pool_size,
            scratch_offset=scratch_offset,
            scratch_size=scratch_size,
        )
        req_client.register_kv_cache_engine_driven_context.return_value = future

        ctx = AsyncEngineDrivenTransferContext(commit_workers=commit_workers)
        ctx.register(
            instance_id=1,
            kv_caches=kv_caches,
            model_name="concurrent-multichunk-model",
            world_size=1,
            blocks_in_chunk=_BLOCKS_IN_CHUNK,
            req_client=req_client,
            mq_timeout=_MQ_TIMEOUT_S,
        )
        return ctx, created_shm_contexts[0], req_client
    finally:
        worker_transfer.create_engine_driven_context = original_factory


def _wire_request_slots(
    ctx: EngineDrivenTransferContext,
    req_client: MagicMock,
    *,
    chunk_shape: list[int],
    dtype_str: str,
    chunk_bytes: int,
) -> dict[str, _RequestSlots]:
    """Build each request's disjoint SHM region and wire the mock RPC calls.

    ``prepare_store``/``prepare_retrieve`` return the *same* slot offsets for
    a given request id, since retrieve later must read back exactly what
    store wrote there. Different requests never share an offset, so a race
    that corrupts one request's region cannot be masked by another request's
    legitimate write. ``chunk_shape``/``dtype_str``/``chunk_bytes`` describe
    the *wire* representation of one chunk -- the raw fp16 chunk shape for
    the unquantized scenario, or ``[quantized_bytes]``/``uint8`` for the
    quantized one (see :func:`_enable_quantization`).
    """
    request_slots: dict[str, _RequestSlots] = {}
    for request_idx in range(_NUM_REQUESTS):
        request_id = f"req-{request_idx}"
        offset = request_idx * _CHUNKS_PER_REQUEST * chunk_bytes
        request_slots[request_id] = _RequestSlots(
            offset, chunk_shape, dtype_str, chunk_bytes
        )

    def _prepare_store(key, _instance_id):
        rs = request_slots[key.request_id]
        return _CompletedFuture(
            PrepareStoreResponse(
                context={
                    "slots": rs.slots,
                    "chunk_indices": list(range(_CHUNKS_PER_REQUEST)),
                }
            )
        )

    def _prepare_retrieve(key, _instance_id):
        rs = request_slots[key.request_id]
        return _CompletedFuture(
            PrepareRetrieveResponse(
                success=True, data=b"", context={"slots": rs.slots}
            )
        )

    req_client.prepare_store.side_effect = _prepare_store
    req_client.commit_store.return_value = _CompletedFuture(True)
    req_client.prepare_retrieve.side_effect = _prepare_retrieve
    req_client.commit_retrieve.return_value = _CompletedFuture(True)
    return request_slots


def _required_shm_pool_bytes(chunk_bytes: int) -> int:
    return _NUM_REQUESTS * _CHUNKS_PER_REQUEST * chunk_bytes


def _fill_request_source_data(
    kv_caches: dict[str, torch.Tensor],
) -> dict[int, dict[str, torch.Tensor]]:
    """Write known, per-request-seeded data into each request's block range.

    Returns a snapshot of each request's original per-layer data (still on
    device) for later comparison against the retrieved copy.
    """
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    for request_idx in range(_NUM_REQUESTS):
        block_ids = _request_block_ids(request_idx)
        generator = torch.Generator(device="cpu").manual_seed(request_idx)
        snapshot: dict[str, torch.Tensor] = {}
        for layer_name, tensor in kv_caches.items():
            data = torch.randn(
                (len(block_ids), *tensor.shape[1:]),
                dtype=tensor.dtype,
                generator=generator,
            ).to(tensor.device)
            tensor[block_ids] = data
            snapshot[layer_name] = data.clone()
        snapshots[request_idx] = snapshot
    return snapshots


def _verify_round_trip(
    ctx: EngineDrivenTransferContext,
    kv_caches: dict[str, torch.Tensor],
    snapshots: dict[int, dict[str, torch.Tensor]],
    *,
    quantized: bool,
) -> None:
    """Retrieve each request's stored data back and compare against the original.

    Uses a fresh destination tensor set (zero-initialized) so a passing
    comparison can only mean the retrieved bytes actually came from that
    request's own store, not leftover data already present at those block ids.
    When ``quantized`` is ``True`` the comparison must tolerate real 4-bit
    quantization error (bit-exact equality is not the correctness bar for a
    lossy codec); an isolated race that clobbered another request's region
    would still show up as an outlier far larger than normal quant noise.
    """
    dest_kv_caches = {
        name: torch.zeros_like(tensor) for name, tensor in kv_caches.items()
    }
    mismatches: list[int] = []
    for request_idx in range(_NUM_REQUESTS):
        block_ids = _request_block_ids(request_idx)
        event = ctx.create_recorded_event()
        future = ctx.submit_retrieve(
            f"req-{request_idx}",
            _request_key(request_idx),
            1,
            dest_kv_caches,
            [block_ids],
            event,
            _BLOCKS_IN_CHUNK,
        )
        ok = future.result(timeout=_MQ_TIMEOUT_S)
        if not ok:
            mismatches.append(request_idx)
            continue
        snapshot = snapshots[request_idx]
        for layer_name, expected in snapshot.items():
            actual = dest_kv_caches[layer_name][block_ids]
            matches = (
                torch.allclose(
                    actual.float().cpu(), expected.float().cpu(), atol=0.5, rtol=0.1
                )
                if quantized
                else torch.equal(actual.cpu(), expected.cpu())
            )
            if not matches:
                mismatches.append(request_idx)
                break

    if mismatches:
        raise AssertionError(
            f"Round-trip mismatch for {len(mismatches)}/{_NUM_REQUESTS} "
            f"requests after concurrent multi-chunk store: {mismatches}"
        )
    exactness = "within 4-bit quant tolerance" if quantized else "byte-identical"
    print(
        f"Correctness OK: all {_NUM_REQUESTS} requests' "
        f"{_CHUNKS_PER_REQUEST}-chunk stores round-tripped {exactness} "
        "under concurrency."
    )


def _submit_all_stores(
    ctx: AsyncEngineDrivenTransferContext, kv_caches: dict[str, torch.Tensor]
) -> list[MessagingFuture]:
    """Fire every request's ``submit_store`` from the forward thread.

    Each call only does O(1) work before handing off to the background
    commit_executor (see module docstring), so this loop finishes quickly
    and the returned futures resolve concurrently in the background.
    """
    futures: list[MessagingFuture] = []
    for request_idx in range(_NUM_REQUESTS):
        block_ids = _request_block_ids(request_idx)
        event = ctx.create_recorded_event()
        future = ctx.submit_store(
            f"req-{request_idx}",
            _request_key(request_idx),
            1,
            kv_caches,
            [block_ids],
            event,
            _BLOCKS_IN_CHUNK,
        )
        futures.append(future)
    return futures


def _time_concurrent_stores(
    ctx: AsyncEngineDrivenTransferContext, kv_caches: dict[str, torch.Tensor]
) -> list[float]:
    def _run() -> None:
        futures = _submit_all_stores(ctx, kv_caches)
        for future in futures:
            assert future.result(timeout=_MQ_TIMEOUT_S)

    for _ in range(_NUM_WARMUP):
        _run()
    samples_ms: list[float] = []
    for _ in range(_NUM_ITERATIONS):
        start = time.perf_counter()
        _run()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return samples_ms


def _time_sequential_stores(
    ctx: AsyncEngineDrivenTransferContext, kv_caches: dict[str, torch.Tensor]
) -> list[float]:
    def _run() -> None:
        for request_idx in range(_NUM_REQUESTS):
            block_ids = _request_block_ids(request_idx)
            event = ctx.create_recorded_event()
            future = ctx.submit_store(
                f"req-{request_idx}",
                _request_key(request_idx),
                1,
                kv_caches,
                [block_ids],
                event,
                _BLOCKS_IN_CHUNK,
            )
            assert future.result(timeout=_MQ_TIMEOUT_S)

    for _ in range(_NUM_WARMUP):
        _run()
    samples_ms: list[float] = []
    for _ in range(_NUM_ITERATIONS):
        start = time.perf_counter()
        _run()
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return samples_ms


def _print_stats(label: str, samples_ms: list[float]) -> None:
    if not samples_ms:
        # async_engine_driven.py's Phase 2.5 never calls _encode_group_chunks
        # for an unquantized group (it `continue`s before the call), so this
        # stage genuinely has zero samples rather than being a bug.
        print(f"  {label:<28s} skipped (stage did not run)")
        return
    mean_ms = statistics.mean(samples_ms)
    stdev_ms = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    print(
        f"  {label:<28s} "
        f"mean={mean_ms:8.3f}ms  min={min(samples_ms):8.3f}ms  "
        f"max={max(samples_ms):8.3f}ms  stdev={stdev_ms:7.3f}ms  "
        f"n={len(samples_ms)}"
    )


def _time_stages_concurrent(
    ctx: AsyncEngineDrivenTransferContext,
    kv_caches: dict[str, torch.Tensor],
    shm_ctx: EngineDrivenContextShm,
) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    """Per-stage timing for one request's whole (8-chunk) scratch/gather/quantize/commit/latency.

    Instruments the real call sites used by ``submit_store``'s background
    task -- ``shm_ctx.allocate_scratch_tensors`` (Stage: scratch alloc --
    only called for a quantized group; blocks up to its wait/retry budget
    when the scratch region is too small or unconfigured, see
    ``scratch_shm_raw_buffer_plan.md``), ``async_engine_driven.gather_paged_kv_to_cpu``
    (Stage: gather, GPU->CPU for all 8 chunks in one call),
    ``ctx._encode_group_chunks_into_slots`` (Stage: quantize and direct write
    to the durable SHM slots; raw pool scratch is released immediately after
    it returns), and
    ``shm_ctx.commit_store`` (Stage: commit, notifies the mocked server the
    SHM slots are ready) -- by temporarily wrapping them with timers, since
    all four run inside ``commit_executor`` worker threads concurrently
    across requests.
    """
    lock = threading.Lock()
    scratch_ms: list[float] = []
    gather_ms: list[float] = []
    quantize_ms: list[float] = []
    commit_ms: list[float] = []
    request_ms: list[float] = []

    original_scratch = shm_ctx.allocate_scratch_tensors
    original_gather = async_engine_driven.gather_paged_kv_to_cpu
    original_encode_into_slots = ctx._encode_group_chunks_into_slots  # noqa: SLF001
    original_commit = shm_ctx.commit_store

    def _timed_scratch(*args, **kwargs):
        start = time.perf_counter()
        result = original_scratch(*args, **kwargs)
        with lock:
            scratch_ms.append((time.perf_counter() - start) * 1000.0)
        return result

    def _timed_gather(*args, **kwargs):
        start = time.perf_counter()
        result = original_gather(*args, **kwargs)
        with lock:
            gather_ms.append((time.perf_counter() - start) * 1000.0)
        return result

    def _timed_encode_into_slots(*args, **kwargs):
        start = time.perf_counter()
        result = original_encode_into_slots(*args, **kwargs)
        with lock:
            quantize_ms.append((time.perf_counter() - start) * 1000.0)
        return result

    def _timed_commit(*args, **kwargs):
        start = time.perf_counter()
        result = original_commit(*args, **kwargs)
        with lock:
            commit_ms.append((time.perf_counter() - start) * 1000.0)
        return result

    async_engine_driven.gather_paged_kv_to_cpu = _timed_gather
    shm_ctx.allocate_scratch_tensors = _timed_scratch
    ctx._encode_group_chunks_into_slots = _timed_encode_into_slots  # noqa: SLF001
    shm_ctx.commit_store = _timed_commit
    try:
        for _ in range(_NUM_WARMUP + _NUM_ITERATIONS):
            pending: list[tuple[float, MessagingFuture]] = []
            for request_idx in range(_NUM_REQUESTS):
                block_ids = _request_block_ids(request_idx)
                event = ctx.create_recorded_event()
                start = time.perf_counter()
                future = ctx.submit_store(
                    f"req-{request_idx}",
                    _request_key(request_idx),
                    1,
                    kv_caches,
                    [block_ids],
                    event,
                    _BLOCKS_IN_CHUNK,
                )
                pending.append((start, future))
            for start, future in pending:
                assert future.result(timeout=_MQ_TIMEOUT_S)
                request_ms.append((time.perf_counter() - start) * 1000.0)
    finally:
        async_engine_driven.gather_paged_kv_to_cpu = original_gather
        shm_ctx.allocate_scratch_tensors = original_scratch
        ctx._encode_group_chunks_into_slots = original_encode_into_slots  # noqa: SLF001
        shm_ctx.commit_store = original_commit

    # Drop the warmup iteration's samples (first _NUM_REQUESTS of each list).
    drop = _NUM_REQUESTS
    return (
        scratch_ms[drop:],
        gather_ms[drop:],
        quantize_ms[drop:],
        commit_ms[drop:],
        request_ms[drop:],
    )


def _run_scenario(quantize: bool) -> None:
    kv_caches = _attention_kv_caches()
    shm_name = f"lmcache_bench_concurrent_{os.getpid()}"
    # The SHM segment must already exist (sized big enough) before register()
    # can attach to it, so the per-chunk byte size is derived analytically
    # here to match what register()/compute_kv_layout will compute: one
    # registered "chunk" packs *all* layers together
    # ([num_layers, block_size, num_kv_heads * 2 * head_dim]). Sized off the
    # raw (unquantized) bytes in both scenarios since that's an upper bound
    # -- the quantized wire payload is always smaller.
    itemsize = 2  # fp16
    raw_chunk_bytes = _NUM_LAYERS * _BLOCK_SIZE * _NUM_KV_HEADS * 2 * _HEAD_DIM * itemsize
    slots_region_bytes = _required_shm_pool_bytes(raw_chunk_bytes) + (16 * 1024 * 1024)
    # Only the quantized scenario ever calls allocate_scratch_tensors(), but
    # reserving the region unconditionally keeps both scenarios' pool layout
    # identical for easier comparison.
    scratch_offset = slots_region_bytes
    scratch_size = _SCRATCH_SIZE_BYTES
    pool_size = slots_region_bytes + scratch_size
    addr = shm_create_readwrite(shm_name, pool_size)
    try:
        ctx, shm_ctx, req_client = _register_shm_context(
            kv_caches,
            shm_name,
            pool_size,
            DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS,
            scratch_offset=scratch_offset,
            scratch_size=scratch_size,
        )
        try:
            plan = ctx._group_plans[0]  # noqa: SLF001 -- internal, mirrors other tests
            if quantize:
                attention_codec = _attention_codec()
                quant_bytes = _enable_quantization(ctx, attention_codec)
                _wire_request_slots(
                    ctx,
                    req_client,
                    chunk_shape=[quant_bytes],
                    dtype_str="uint8",
                    chunk_bytes=quant_bytes,
                )
            else:
                dtype_str = str(plan.raw_layout_desc.dtypes[0]).removeprefix("torch.")
                _wire_request_slots(
                    ctx,
                    req_client,
                    chunk_shape=list(plan.chunk_shape),
                    dtype_str=dtype_str,
                    chunk_bytes=raw_chunk_bytes,
                )

            print(
                f"Concurrent multi-chunk engine-driven SHM store bench "
                f"({_NUM_LAYERS} attention layers, {_NUM_REQUESTS} concurrent "
                f"requests x {_CHUNKS_PER_REQUEST} chunks/request "
                f"({_CHUNK_SIZE_TOKENS} tokens/chunk), "
                f"commit_workers={DEFAULT_ENGINE_DRIVEN_COMMIT_WORKERS}, "
                f"quantized={quantize}, "
                f"{_NUM_ITERATIONS} iterations after {_NUM_WARMUP} warmup)"
            )

            print("\n--- correctness: one round of concurrent multi-chunk store + retrieve ---")
            snapshots = _fill_request_source_data(kv_caches)
            futures = _submit_all_stores(ctx, kv_caches)
            for request_idx, future in enumerate(futures):
                if not future.result(timeout=_MQ_TIMEOUT_S):
                    raise AssertionError(
                        f"submit_store failed for request {request_idx}"
                    )
            _verify_round_trip(ctx, kv_caches, snapshots, quantized=quantize)

            print("\n--- timing: concurrent (fire-all, wait-all) submit_store ---")
            samples_ms = _time_concurrent_stores(ctx, kv_caches)
            _print_stats("concurrent", samples_ms)

            print("\n--- timing: sequential (submit, wait, next) submit_store ---")
            samples_ms = _time_sequential_stores(ctx, kv_caches)
            _print_stats("sequential", samples_ms)

            print(
                "\n--- timing: per-stage breakdown under concurrency "
                f"(one request's whole {_CHUNKS_PER_REQUEST}-chunk call) ---"
            )
            scratch_ms, gather_ms, quantize_ms, commit_ms, request_ms = (
                _time_stages_concurrent(ctx, kv_caches, shm_ctx)
            )
            _print_stats("scratch alloc (quantized only)", scratch_ms)
            _print_stats("gather (D2H, all chunks)", gather_ms)
            _print_stats("quantize + direct SHM write", quantize_ms)
            _print_stats("commit (SHM notify)", commit_ms)
            _print_stats("end-to-end request latency", request_ms)
        finally:
            ctx.close()
    finally:
        shm_munmap(addr, pool_size)
        shm_unlink(shm_name)


def main() -> int:
    if not (torch_device_type == "xpu" and torch_dev.is_available()):
        print("Skipping: requires an available XPU runtime.")
        return 0

    scenarios = [
        (False, "Scenario A -- unquantized (real registration default)"),
    ]
    if _KVWEAVE_QUANT_AVAILABLE:
        scenarios.append((True, "Scenario B -- quantized (forced, fused-K/V 4-bit)"))
    else:
        print(
            "Note: native 'kvweave_quant' extension not importable -- "
            "skipping the quantized scenario. Run inside an env with the "
            "'kvweave' package installed (e.g. the 'dev_lmcache' conda env) "
            "to measure it.\n"
        )

    for quantize, label in scenarios:
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
        _run_scenario(quantize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
