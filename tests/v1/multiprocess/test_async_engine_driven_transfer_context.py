# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable
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
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
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


class _FakeStoreRetrieveContext(_FakeStoreContext):
    """``_FakeStoreContext`` plus the retrieve half, for round-trip tests."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.retrieve_chunks: list[torch.Tensor] | None = None

    def prepare_retrieve(
        self, _key: object, _instance_id: int
    ) -> list[torch.Tensor] | None:
        return self.retrieve_chunks

    def commit_retrieve(self, _key: object, _instance_id: int) -> bool:
        return True


def test_async_submit_store_quantizes_then_submit_retrieve_dequantizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With LMCACHE_MP_L1_KVWEAVE_QUANT=1, the async store path's own gather
    loop (which does not go through the base class's submit_store) must
    still quantize each eligible group's chunk through the KVWeave codec,
    and the inherited (synchronous) submit_retrieve must dequantize those
    bytes back before scattering.

    Uses real gather/scatter (no _install_fake_gather) and a large enough
    chunk for KVWeave's fixed ~4KB header overhead to pay off, so the
    codec path in AsyncEngineDrivenTransferContext.submit_store is actually
    exercised rather than falling back to the raw layout.
    """
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    gather_gate = threading.Event()
    gather_gate.set()
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeTorchDev(gather_gate))

    committed: dict[str, list[torch.Tensor]] = {}
    fake_context = _FakeStoreRetrieveContext(
        commit_impl=lambda chunks: committed.update(chunks=chunks) or True
    )

    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    kv_caches = {f"layer_{i}": torch.randn(2, 64, 1, 1, 128) for i in range(2)}
    future = MagicMock()
    future.result.return_value = MagicMock(shm_name="", pool_size=0)
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        lambda *a, **k: fake_context,
    )
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=64,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
        engine_group_infos=[],
    )
    # _FakeStoreContext hardcodes a [2,1,1,1] layout_desc that doesn't match
    # this test's real KV shape; group_chunk_shape() reads this for the
    # pinned-staging allocation path, so it must reflect the real chunk.
    fake_context.layout_desc = SimpleNamespace(
        shapes=[torch.Size([2, 2, 64, 128])], dtypes=[torch.float32]
    )

    store_future = ctx.submit_store(
        "r1",
        object(),
        1,
        kv_caches,
        [list(range(64))],
        _FakeEvent(gather_gate),
        64,
    )
    assert store_future.result(timeout=1) is True
    assert "chunks" in committed
    raw_nbytes = kv_caches["layer_0"].numel() * kv_caches["layer_0"].element_size()
    for chunk in committed["chunks"]:
        assert chunk.dtype == torch.uint8
        assert chunk.numel() < raw_nbytes

    fake_context.retrieve_chunks = committed["chunks"]
    destination = {name: torch.zeros_like(tensor) for name, tensor in kv_caches.items()}
    retrieve_future = ctx.submit_retrieve(
        "r1", object(), 1, destination, [list(range(64))], _FakeEvent(gather_gate), 64
    )
    assert retrieve_future.result(timeout=1) is True
    for name in kv_caches:
        assert torch.max(torch.abs(destination[name] - kv_caches[name])) < 1.0
    ctx.close()
    ctx.close()
