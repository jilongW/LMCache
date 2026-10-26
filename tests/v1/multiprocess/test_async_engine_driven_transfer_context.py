# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.transfer_context import (
    async_engine_driven,
    worker_transfer,
)
from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
    AsyncEngineDrivenTransferContext,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_config import AttentionPlaneLayout
from lmcache.v1.multiprocess.group_view import EngineGroupInfo, MambaSubStateWireLayout
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    GroupTransferPlan,
    MemoryLayoutDesc,
    _BoundedStoreBatch,
)


@dataclass
class _FakeStoreContext:
    """Minimal engine-driven context for async store tests."""

    commit_impl: Callable[[list[torch.Tensor]], bool]
    prepare_result: tuple[list[torch.Tensor], list[int]] | None = None
    prepare_impl: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        self.layout_desc = SimpleNamespace(
            shapes=[torch.Size([2, 1, 1, 1])], dtypes=[torch.float32]
        )

    def prepare_store(
        self, _key: object, _instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        if self.prepare_impl is not None:
            self.prepare_impl()
        return self.prepare_result

    def commit_store(
        self, _key: object, _instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        return bool(self.commit_impl(chunks))

    def close(self) -> None:
        return None


class _FakeEvent:
    def __init__(self, gate: threading.Event):
        self._gate = gate

    def record(self, stream: object | None = None) -> None:
        return None

    def wait(self, stream: object | None = None) -> None:
        return None

    def synchronize(self) -> None:
        self._gate.wait(timeout=2)

    def query(self) -> bool:
        return self._gate.is_set()

    def ipc_handle(self) -> object:
        return object()


class _FakeTorchDev:
    def __init__(self, gather_gate: threading.Event):
        self._stream = object()
        self._gather_gate = gather_gate

    def Stream(self) -> object:
        return object()

    def stream(self, stream: object) -> object:
        return nullcontext(stream)

    def current_stream(self) -> object:
        return self._stream

    def Event(self, interprocess: bool = False) -> _FakeEvent:
        return _FakeEvent(self._gather_gate)

    def synchronize(self) -> None:
        pass


def _install_fake_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    def _gather(
        _kv_caches: dict[str, torch.Tensor],
        _block_ids: list[int],
        _blocks_in_chunk: int,
        **kwargs: object,
    ) -> list[torch.Tensor]:
        out = kwargs.get("out")
        if out is None:
            return [torch.ones(1)]
        assert isinstance(out, list)
        for tensor in out:
            tensor.fill_(1.0)
        return out

    # Patch gather in both modules so either path is exercised.
    monkeypatch.setattr(async_engine_driven, "gather_paged_kv_to_cpu", _gather)
    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _gather)


def _new_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gather_gate: threading.Event,
    commit_impl: Callable[[list[torch.Tensor]], bool],
    max_inflight: int = 8,
) -> AsyncEngineDrivenTransferContext:
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    _install_fake_gather(monkeypatch)
    ctx = AsyncEngineDrivenTransferContext(commit_workers=max_inflight)
    ctx._engine_driven_context = (
        _FakeStoreContext(commit_impl=commit_impl)  # type: ignore[assignment]
    )
    return ctx


def test_submit_store_returns_pending_future_until_gather_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()
    ctx = _new_context(
        monkeypatch, gather_gate=gather_gate, commit_impl=lambda _c: True
    )
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    assert not future.query()
    gather_gate.set()
    assert future.result(timeout=1) is True
    ctx.close()


def test_create_recorded_event_uses_local_device_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async engine-driven ordering uses a local, non-IPC event."""
    fake_torch_dev = MagicMock()
    event = fake_torch_dev.Event.return_value
    stream = fake_torch_dev.current_stream.return_value
    monkeypatch.setattr(async_engine_driven, "torch_dev", fake_torch_dev)
    ctx = AsyncEngineDrivenTransferContext()
    ctx._engine_driven_context = MagicMock()

    assert ctx.create_recorded_event() is event
    fake_torch_dev.Event.assert_called_once_with()
    event.record.assert_called_once_with(stream)
    ctx.close()


def test_submit_store_commit_waits_for_gather_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()
    commit_called = threading.Event()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        commit_called.set()
        return True

    ctx = _new_context(monkeypatch, gather_gate=gather_gate, commit_impl=_commit)
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    assert not commit_called.wait(timeout=0.05)
    gather_gate.set()
    assert future.result(timeout=1) is True
    assert commit_called.is_set()
    ctx.close()


def test_close_drains_inflight_async_store(monkeypatch: pytest.MonkeyPatch) -> None:
    gather_gate = threading.Event()
    commit_gate = threading.Event()
    gather_gate.set()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        commit_gate.wait(timeout=2)
        return True

    ctx = _new_context(monkeypatch, gather_gate=gather_gate, commit_impl=_commit)
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    closed = threading.Event()

    def _close() -> None:
        ctx.close()
        closed.set()

    t = threading.Thread(target=_close, daemon=True)
    t.start()
    assert not closed.wait(timeout=0.05)
    commit_gate.set()
    t.join(timeout=1)
    assert closed.is_set()
    assert future.result(timeout=1) is True


def test_commit_failure_sets_false_and_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        raise RuntimeError("commit failed")

    log_exception = MagicMock()
    monkeypatch.setattr(async_engine_driven.logger, "exception", log_exception)
    ctx = _new_context(monkeypatch, gather_gate=gather_gate, commit_impl=_commit)
    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )
    gather_gate.set()
    assert future.result(timeout=1) is False
    log_exception.assert_called_once()
    ctx.close()


def test_flush_inflight_stores_no_inflight_is_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gather_gate = threading.Event()
    gather_gate.set()
    ctx = _new_context(
        monkeypatch, gather_gate=gather_gate, commit_impl=lambda _c: True
    )
    # No in-flight events: flush is a cheap no-op and must not raise.
    ctx.flush_inflight_stores()
    ctx.close()


def test_sync_engine_driven_context_returns_resolved_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RecordingTorchDev:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    fake = _RecordingTorchDev()
    monkeypatch.setattr(worker_transfer, "torch_dev", fake)
    _install_fake_gather(monkeypatch)
    ctx = EngineDrivenTransferContext()
    ctx._engine_driven_context = (
        _FakeStoreContext(commit_impl=lambda _c: True)  # type: ignore[assignment]
    )

    future = ctx.submit_store(
        "r1",
        object(),
        1,
        {"k": torch.zeros(1)},
        [[0]],
        _FakeEvent(threading.Event()),
        1,
    )

    # Sync path resolves inline.
    assert future.query()
    assert future.result(timeout=1) is True
    assert fake.synchronize_calls >= 1
    # flush_inflight_stores is the inherited base no-op; must not raise.
    ctx.flush_inflight_stores()
    ctx.close()


def test_sync_engine_driven_context_has_no_async_resources() -> None:
    ctx = EngineDrivenTransferContext()
    assert not hasattr(ctx, "_copy_stream")
    assert not hasattr(ctx, "_commit_executor")
    assert not hasattr(ctx, "_inflight_semaphore")
    # close() on an unregistered sync context must not raise.
    ctx.close()


def test_build_engine_driven_context_dispatches_on_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker_transfer, "_supports_async_primitives", lambda: True)
    # Avoid touching real stream/event primitives when instantiating the async
    # context returned by the capable branch.
    monkeypatch.setattr(
        async_engine_driven, "torch_dev", _FakeTorchDev(threading.Event())
    )
    capable = worker_transfer._build_engine_driven_context()
    assert isinstance(capable, AsyncEngineDrivenTransferContext)
    capable.close()

    monkeypatch.setattr(worker_transfer, "_supports_async_primitives", lambda: False)
    fallback = worker_transfer._build_engine_driven_context()
    assert isinstance(fallback, EngineDrivenTransferContext)
    assert not isinstance(fallback, AsyncEngineDrivenTransferContext)
    fallback.close()


def test_supports_async_primitives_false_without_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A torch_dev without Stream/Event is not async-capable.
    monkeypatch.setattr(worker_transfer, "torch_dev", object())
    assert worker_transfer._supports_async_primitives() is False


def test_flush_inflight_stores_waits_for_pending_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """flush_inflight_stores must not return before a submitted-but-not-yet-
    launched gather has recorded its CUDA event (preemption race condition)."""
    gather_gate = threading.Event()
    flush_returned = threading.Event()
    gather_started = threading.Event()

    # Gate that blocks gather from completing until we are ready.
    class _BlockingFakeTorchDev(_FakeTorchDev):
        def stream(self, stream: object) -> object:
            gather_started.set()
            # Block here until the test releases the gate — simulating the
            # background thread not yet recording the CUDA event.
            gather_gate.wait(timeout=2)
            return super().stream(stream)

    # Pass gather_gate so _FakeEvent.synchronize() also uses the same gate.
    monkeypatch.setattr(
        async_engine_driven, "torch_dev", _BlockingFakeTorchDev(gather_gate)
    )
    _install_fake_gather(monkeypatch)

    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._engine_driven_context = (
        _FakeStoreContext(commit_impl=lambda _c: True)  # type: ignore[assignment]
    )

    ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
    )

    # Wait until the background thread has started (entered stream()), proving
    # it has not yet recorded its CUDA event.
    assert gather_started.wait(timeout=1), "background thread never started"

    def _flush() -> None:
        ctx.flush_inflight_stores()
        flush_returned.set()

    t = threading.Thread(target=_flush, daemon=True)
    t.start()

    # flush_inflight_stores must NOT return while the gather is still pending.
    assert not flush_returned.wait(timeout=0.05), (
        "flush_inflight_stores returned before gather launched — race condition!"
    )

    # Now let the background gather proceed; flush should complete shortly.
    gather_gate.set()
    t.join(timeout=2)
    assert flush_returned.is_set(), "flush_inflight_stores did not complete"
    ctx.close()


def test_commit_store_serialized_by_commit_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent commit_store calls must be serialized by _commit_lock to
    protect ZMQ socket access."""
    gather_gate = threading.Event()
    gather_gate.set()
    concurrent_commits = threading.Event()
    active_commits = [0]

    def _commit(_chunks: list[torch.Tensor]) -> bool:
        active_commits[0] += 1
        if active_commits[0] > 1:
            concurrent_commits.set()
        # Briefly yield so a second commit could slip in if unlocked.
        time.sleep(0.02)
        active_commits[0] -= 1
        return True

    ctx = _new_context(
        monkeypatch, gather_gate=gather_gate, commit_impl=_commit, max_inflight=4
    )

    futures = [
        ctx.submit_store(
            f"r{i}",
            object(),
            1,
            {"k": torch.zeros(1)},
            [[0]],
            _FakeEvent(gather_gate),
            1,
        )
        for i in range(4)
    ]

    for f in futures:
        assert f.result(timeout=2) is True

    # With _commit_lock, no two commits should have been active simultaneously.
    assert not concurrent_commits.is_set(), (
        "commit_store called concurrently — _commit_lock not working"
    )
    ctx.close()


def test_prepare_store_runs_on_background_thread_not_forward_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """submit_store must return immediately even when prepare_store is slow.

    The forward thread must not be blocked by the prepare_store RPC round-trip.
    Both prepare_store and buffer allocation now run entirely inside the
    background ``_prepare_gather_and_commit`` closure.
    """
    gather_gate = threading.Event()
    gather_gate.set()
    prepare_started = threading.Event()
    prepare_gate = threading.Event()

    def _slow_prepare() -> None:
        prepare_started.set()
        # Block until the test explicitly releases the gate.
        prepare_gate.wait(timeout=2)

    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    _install_fake_gather(monkeypatch)

    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._engine_driven_context = _FakeStoreContext(  # type: ignore[assignment]
        commit_impl=lambda _c: True,
        prepare_impl=_slow_prepare,
    )

    submit_returned = threading.Event()

    def _submit() -> None:
        ctx.submit_store(
            "r1", object(), 1, {"k": torch.zeros(1)}, [[0]], _FakeEvent(gather_gate), 1
        )
        submit_returned.set()

    t = threading.Thread(target=_submit, daemon=True)
    t.start()

    # submit_store must return before prepare_store finishes.
    assert submit_returned.wait(timeout=1), (
        "submit_store blocked the forward thread — prepare_store is not async"
    )

    # Confirm prepare_store was actually reached by the background thread.
    assert prepare_started.wait(timeout=1), (
        "background thread never reached prepare_store"
    )

    # Now release prepare_store and let the background work complete.
    prepare_gate.set()
    t.join(timeout=1)
    ctx.close()


class _FakeShmContext:
    """Minimal stand-in for EngineDrivenContextShm.

    Only implements what the quantized-SHM store path touches:
    prepare_store/commit_store (inherited call shape from
    _FakeStoreContext), plus allocate_scratch_tensors/free_scratch backed by
    plain CPU tensors instead of a real SHM pinned pool.
    """

    def __init__(
        self,
        commit_impl: Callable[[list[torch.Tensor]], bool],
        prepare_result: tuple[list[torch.Tensor], list[int]] | None,
    ) -> None:
        self.commit_impl = commit_impl
        self.prepare_result = prepare_result
        self.layout_desc = SimpleNamespace(
            shapes=[torch.Size([2, 1, 1, 1])], dtypes=[torch.float32]
        )
        self.freed: list[object] = []

    def prepare_store(
        self, _key: object, _instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        return self.prepare_result

    def commit_store(
        self, _key: object, _instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        return bool(self.commit_impl(chunks))

    def allocate_scratch_tensors(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        count: int,
        *,
        wait: bool,
        max_retries: int = 3,
        timeout_s: float = 0.5,
    ) -> tuple[list[torch.Tensor], object] | None:
        tensors = [torch.empty(shape, dtype=dtype) for _ in range(count)]
        allocation = object()
        return tensors, allocation

    def free_scratch(self, allocation: object) -> None:
        self.freed.append(allocation)

    def close(self) -> None:
        return None


def _make_quantized_plan(group_id: int) -> GroupTransferPlan:
    raw_shape = torch.Size([1, 1, 1, 1])
    return GroupTransferPlan(
        group_info=EngineGroupInfo(
            engine_group_id=group_id, cache_category="attention"
        ),
        layer_indices=frozenset(),
        blocks_per_chunk=1,
        chunk_shape=raw_shape,
        engine_kv_format=None,
        attention_plane_layout=None,
        quantized=True,
        raw_layout_desc=MemoryLayoutDesc(shapes=[raw_shape], dtypes=[torch.float32]),
    )


def test_submit_store_pipelines_across_quantized_shm_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Group N+1's launch must be enqueued before group N's encode finishes.

    Regression test for the bug where quantized-SHM groups (Branch A) were
    processed with a fully synchronous outer loop -- each group's
    launch+wait+split+encode ran to completion before the next group's
    launch was even issued, so cross-group D2H/CPU overlap never happened
    in production (4 groups, 1 chunk each).
    """
    gather_gate = threading.Event()
    gather_gate.set()
    events: list[tuple[str, int]] = []

    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    _install_fake_gather(monkeypatch)
    monkeypatch.setattr(async_engine_driven, "EngineDrivenContextShm", _FakeShmContext)

    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._kvweave_enabled = True  # quantized plans imply the master switch is on
    fake_context = _FakeShmContext(
        commit_impl=lambda _c: True,
        prepare_result=([torch.zeros(1, dtype=torch.uint8) for _ in range(2)], [0, 1]),
    )
    ctx._engine_driven_context = fake_context  # type: ignore[assignment]

    plan0 = _make_quantized_plan(0)
    plan1 = _make_quantized_plan(1)
    transfer_groups = [
        (plan0, {"k": torch.zeros(1)}, [0]),
        (plan1, {"k": torch.zeros(1)}, [0]),
    ]
    monkeypatch.setattr(
        ctx, "iter_transfer_groups", lambda *_a, **_kw: iter(transfer_groups)
    )

    _orig_launch = AsyncEngineDrivenTransferContext._launch_shm_quantized_batch

    def _record_launch(
        self: AsyncEngineDrivenTransferContext,
        request_id: str,
        batch: _BoundedStoreBatch,
    ) -> Any:
        events.append(("launch", batch.group_id))
        return _orig_launch(self, request_id, batch)

    monkeypatch.setattr(
        AsyncEngineDrivenTransferContext,
        "_launch_shm_quantized_batch",
        _record_launch,
    )

    def _fake_encode_into_slots(
        self: AsyncEngineDrivenTransferContext,
        plan: GroupTransferPlan,
        gathered: list[torch.Tensor],
        slots: list[torch.Tensor],
        mamba_splits: object = None,
    ) -> None:
        assert plan.group_info is not None
        slots[0].fill_(plan.group_info.engine_group_id + 1)
        events.append(("encode", plan.group_info.engine_group_id))

    monkeypatch.setattr(
        AsyncEngineDrivenTransferContext,
        "_encode_group_chunks_into_slots",
        _fake_encode_into_slots,
    )

    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[0], [0]], _FakeEvent(gather_gate), 1
    )
    assert future.result(timeout=1) is True

    assert [group_id for phase, group_id in events if phase == "launch"] == [0, 1]
    assert [int(slot.item()) for slot in fake_context.prepare_result[0]] == [1, 2]

    # Group 1's launch must happen before group 0's encode --
    # proof the pipeline enqueues the next D2H gather ahead of the current
    # CPU-side split/encode, instead of running group 0 fully to completion
    # first.
    assert events.index(("launch", 1)) < events.index(("encode", 0))
    ctx.close()


def test_run_async_bounded_pipeline_frees_next_batch_scratch_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception while processing batch 0 must not leak batch 1's scratch.

    Batch 1's gather is already launched (and its scratch allocated) before
    batch 0 is processed, per the pipeline's depth-2 lookahead. If
    processing batch 0 raises, the driver's except branch must free batch
    1's already-allocated scratch instead of leaking it.
    """
    gather_gate = threading.Event()
    gather_gate.set()
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    _install_fake_gather(monkeypatch)
    monkeypatch.setattr(async_engine_driven, "EngineDrivenContextShm", _FakeShmContext)

    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    fake_context = _FakeShmContext(commit_impl=lambda _c: True, prepare_result=None)
    ctx._engine_driven_context = fake_context  # type: ignore[assignment]

    plan0 = _make_quantized_plan(0)
    plan1 = _make_quantized_plan(1)
    batches = [
        _BoundedStoreBatch(
            plan=plan0,
            group_kv_caches={"k": torch.zeros(1)},
            group_block_ids=[0],
            group_slots=[torch.zeros(1, dtype=torch.uint8)],
            group_id=0,
            start=0,
            batch_index=0,
        ),
        _BoundedStoreBatch(
            plan=plan1,
            group_kv_caches={"k": torch.zeros(1)},
            group_block_ids=[0],
            group_slots=[torch.zeros(1, dtype=torch.uint8)],
            group_id=1,
            start=0,
            batch_index=0,
        ),
    ]

    def fail_encode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(ctx, "_encode_group_chunks_into_slots", fail_encode)

    with pytest.raises(RuntimeError, match="boom"):
        ctx._run_async_bounded_pipeline("r1", batches)

    # Batch 1's scratch (allocated during its lookahead launch) must have
    # been freed by the except branch even though it was never processed.
    assert len(fake_context.freed) == 2
    ctx.close()


@pytest.mark.parametrize("invalid_slot", ["dtype", "size"])
def test_raw_mamba_copy_failure_releases_scratch_without_committing(
    monkeypatch: pytest.MonkeyPatch, invalid_slot: str
) -> None:
    """A raw-copy error must not encode, commit or leak scratch."""
    gate = threading.Event()
    gate.set()
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gate))
    monkeypatch.setattr(async_engine_driven, "EngineDrivenContextShm", _FakeShmContext)
    _install_fake_gather(monkeypatch)
    commit = MagicMock(return_value=True)
    slot = (
        torch.zeros(1, dtype=torch.float16)
        if invalid_slot == "dtype"
        else torch.zeros(2)
    )
    context = _FakeShmContext(commit_impl=commit, prepare_result=([slot], [0]))
    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._engine_driven_context = context  # type: ignore[assignment]
    ctx._kvweave_enabled = True  # master switch on: raw Mamba stages via scratch
    plan = replace(
        _make_quantized_plan(0),
        quantized=False,
        group_info=EngineGroupInfo(engine_group_id=0, cache_category="mamba"),
    )
    monkeypatch.setattr(
        ctx,
        "iter_transfer_groups",
        lambda *_a, **_kw: iter([(plan, {"k": torch.zeros(1)}, [1])]),
    )
    codec = MagicMock()
    ctx._kvweave_codec = codec
    try:
        future = ctx.submit_store(
            "raw-copy-failure",
            object(),
            1,
            {"k": torch.zeros(1)},
            [[1]],
            _FakeEvent(gate),
            1,
        )
        assert future.result(timeout=1) is False
        commit.assert_not_called()
        assert codec.mock_calls == []
        assert len(context.freed) == 1
        assert torch.count_nonzero(slot) == 0
    finally:
        ctx.close()


def test_raw_mamba_waits_for_gather_before_copy_and_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The raw SHM slot stays untouched until its gather event completes."""
    gate = threading.Event()
    launched = threading.Event()
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gate))
    monkeypatch.setattr(async_engine_driven, "EngineDrivenContextShm", _FakeShmContext)
    slot = torch.zeros(1)
    commit = MagicMock(return_value=True)
    context = _FakeShmContext(commit_impl=commit, prepare_result=([slot], [0]))
    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._engine_driven_context = context  # type: ignore[assignment]
    ctx._kvweave_enabled = True  # master switch on: raw Mamba stages via scratch
    plan = replace(
        _make_quantized_plan(0),
        quantized=False,
        group_info=EngineGroupInfo(engine_group_id=0, cache_category="mamba"),
    )
    monkeypatch.setattr(
        ctx,
        "iter_transfer_groups",
        lambda *_a, **_kw: iter([(plan, {"k": torch.zeros(1)}, [1])]),
    )

    def gather(*args: object, **kwargs: Any) -> list[torch.Tensor]:
        output = kwargs["out"]
        output[0].fill_(7)
        launched.set()
        return output

    monkeypatch.setattr(async_engine_driven, "gather_paged_kv_to_cpu", gather)
    try:
        future = ctx.submit_store(
            "raw-wait", object(), 1, {"k": torch.zeros(1)}, [[1]], _FakeEvent(gate), 1
        )
        assert launched.wait(timeout=1)
        assert not future.query()
        assert slot.item() == 0
        commit.assert_not_called()
        assert context.freed == []
        gate.set()
        assert future.result(timeout=1) is True
        assert slot.item() == 7
        commit.assert_called_once_with([])
        assert len(context.freed) == 1
    finally:
        gate.set()
        ctx.close()


def test_paged_fused_quantized_group_skips_gather_and_encodes_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An eligible fused-K/V group is quantized straight from paged KV.

    No raw gather may run; the codec receives the chunk's own block ids and the
    group's SHM slot, and the payload lands in that slot.
    """
    gather_gate = threading.Event()
    gather_gate.set()
    gather_calls: list[object] = []
    encode_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    monkeypatch.setattr(async_engine_driven, "EngineDrivenContextShm", _FakeShmContext)
    monkeypatch.setattr(
        async_engine_driven,
        "gather_paged_kv_to_cpu",
        lambda *a, **kw: gather_calls.append((a, kw)),
    )
    source = async_engine_driven.PagedFusedSource(
        layer_ptrs=torch.zeros(2, dtype=torch.int64),
        block_size=16,
        block_stride_elems=1024,
        num_blocks=8,
        dtype=torch.bfloat16,
        hidden=64,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        worker_transfer, "prepare_paged_fused_source", lambda *_a, **_kw: source
    )
    monkeypatch.setattr(
        async_engine_driven,
        "paged_block_ids_tensor",
        lambda ids, _device: torch.tensor(ids, dtype=torch.int64),
    )

    class _PagedCodec:
        def supports_paged_fused_encode(self, hidden: int) -> bool:
            return hidden == 64

        def serialize_fused_paged(
            self,
            layer_ptrs: torch.Tensor,
            block_ids: torch.Tensor,
            block_size: int,
            block_stride_elems: int,
            num_blocks: int,
            dtype: torch.dtype,
            hidden: int,
            out: torch.Tensor,
        ) -> int:
            encode_calls.append({"block_ids": block_ids.tolist(), "out": out})
            out.fill_(7)
            return int(out.numel())

    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._kvweave_enabled = True  # quantized plans imply the master switch is on
    ctx._kvweave_codec = _PagedCodec()  # type: ignore[assignment]
    slot = torch.zeros(4, dtype=torch.uint8)
    fake_context = _FakeShmContext(
        commit_impl=lambda _c: True, prepare_result=([slot], [0])
    )
    ctx._engine_driven_context = fake_context  # type: ignore[assignment]

    plan = replace(
        _make_quantized_plan(0),
        blocks_per_chunk=2,
        attention_plane_layout=AttentionPlaneLayout.FUSED_KV,
    )
    monkeypatch.setattr(
        ctx,
        "iter_transfer_groups",
        lambda *_a, **_kw: iter([(plan, {"k": torch.zeros(1)}, [3, 5])]),
    )

    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[3, 5]], _FakeEvent(gather_gate), 2
    )
    assert future.result(timeout=1) is True

    assert gather_calls == []
    assert [call["block_ids"] for call in encode_calls] == [[3, 5]]
    assert slot.tolist() == [7, 7, 7, 7]
    ctx.close()


def test_paged_mamba_quantized_group_skips_gather_and_encodes_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quantized Mamba group on XPU is quantized straight from its pages.

    No raw gather may run; the codec receives the chunk's own block ids and the
    group's real conv/ssm layout, and writes the payload into the group's slot.
    """
    gather_gate = threading.Event()
    gather_gate.set()
    gather_calls: list[object] = []
    encode_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))
    monkeypatch.setattr(async_engine_driven, "EngineDrivenContextShm", _FakeShmContext)
    monkeypatch.setattr(
        async_engine_driven,
        "gather_paged_kv_to_cpu",
        lambda *a, **kw: gather_calls.append((a, kw)),
    )
    layout = (
        MambaSubStateWireLayout(0, 8, "torch.float16", (2, 2)),
        MambaSubStateWireLayout(8, 16, "torch.float32", (2, 2)),
    )
    source = async_engine_driven.PagedMambaSource(
        layer_ptrs=torch.zeros(2, dtype=torch.int64),
        page_stride_bytes=4096,
        page_bytes=1024,
        num_blocks=8,
        block_size=4,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(
        worker_transfer, "prepare_paged_mamba_source", lambda *_a, **_kw: source
    )
    monkeypatch.setattr(
        async_engine_driven,
        "paged_block_ids_tensor",
        lambda ids, _device: torch.tensor(ids, dtype=torch.int64),
    )

    class _MambaPagedCodec:
        def supports_paged_mamba_encode(self, mamba_layout, options, blocks) -> bool:
            return blocks == 2 and mamba_layout is layout

        def serialize_mamba_paged_into(
            self,
            layer_ptrs: torch.Tensor,
            block_ids: torch.Tensor,
            num_blocks: int,
            page_stride_bytes: int,
            mamba_layout: object,
            options: object,
            out: torch.Tensor,
        ) -> int:
            encode_calls.append({"block_ids": block_ids.tolist(), "layout": mamba_layout})
            out.fill_(9)
            return int(out.numel())

    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx._kvweave_enabled = True  # quantized plans imply the master switch is on
    ctx._kvweave_codec = _MambaPagedCodec()  # type: ignore[assignment]
    slot = torch.zeros(4, dtype=torch.uint8)
    ctx._engine_driven_context = _FakeShmContext(  # type: ignore[assignment]
        commit_impl=lambda _c: True, prepare_result=([slot], [0])
    )
    plan = replace(
        _make_quantized_plan(0),
        blocks_per_chunk=2,
        group_info=EngineGroupInfo(
            engine_group_id=0,
            cache_category="mamba",
            tokens_per_block=4,
            mamba_real_layout=layout,
        ),
        mamba_options=MagicMock(),
    )
    monkeypatch.setattr(
        ctx,
        "iter_transfer_groups",
        lambda *_a, **_kw: iter([(plan, {"k": torch.zeros(1)}, [3, 5])]),
    )

    future = ctx.submit_store(
        "r1", object(), 1, {"k": torch.zeros(1)}, [[3, 5]], _FakeEvent(gather_gate), 2
    )
    assert future.result(timeout=1) is True

    assert gather_calls == []
    assert [call["block_ids"] for call in encode_calls] == [[3, 5]]
    assert encode_calls[0]["layout"] is layout
    assert slot.tolist() == [9, 9, 9, 9]
    ctx.close()
