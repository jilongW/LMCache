# SPDX-License-Identifier: Apache-2.0
"""Worker-side KVWeave wiring: registration, and a store -> retrieve round trip."""

# Standard
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.custom_types import (
    RegisterEngineDrivenContextPayload,
    RegisterEngineDrivenContextResponse,
    deserialize_memory_layout_desc,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context import (
    EngineDrivenTransferContext,
    worker_transfer,
)
from lmcache.v1.multiprocess.transfer_context.inline import EngineDrivenContextInline

NUM_LAYERS = 4
NUM_BLOCKS = 40
BLOCK_SIZE = 16
NUM_HEADS = 8
HEAD_SIZE = 64
BLOCKS_IN_CHUNK = 16
NUM_CHUNKS = 2


@pytest.fixture
def quant_env(monkeypatch):
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_QUANT_DEVICE", "cpu")


def _kv_caches() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        f"layer_{i}": torch.randn(
            2, NUM_BLOCKS, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, dtype=torch.float16
        )
        for i in range(NUM_LAYERS)
    }


def _group_info() -> EngineGroupInfo:
    return EngineGroupInfo(
        engine_group_id=0,
        layer_indices=tuple(range(NUM_LAYERS)),
        tokens_per_block=BLOCK_SIZE,
        cache_category="attention",
    )


class _FakeShmContext:
    """Duck-typed SHM context that keeps slots in plain host tensors.

    Mirrors the slice of ``EngineDrivenContextShm`` the multi-group worker uses.
    Store slots come out sized by the layouts the worker registered, so a
    quantized group's slots are ``uint8`` byte buffers, exactly as the server
    would hand out.
    """

    def __init__(self, layout_by_group: dict[int, tuple[torch.Size, torch.dtype]]):
        self._layouts = layout_by_group
        self.stored: dict[tuple[int, int], torch.Tensor] = {}
        self.committed = 0

    @contextmanager
    def transfer_guard(self):
        yield

    def prepare_store_grouped(self, key, instance_id, skipped=()):
        tensors, chunks, groups = [], [], []
        for gid, (shape, dtype) in self._layouts.items():
            for chunk in range(NUM_CHUNKS):
                slot = torch.zeros(shape, dtype=dtype)
                self.stored[(gid, chunk)] = slot
                tensors.append(slot)
                chunks.append(chunk)
                groups.append(gid)
        return tensors, chunks, groups

    def commit_store(self, key, instance_id, chunks):
        self.committed += 1
        return True

    def prepare_retrieve_grouped(self, key, instance_id):
        keys = sorted(self.stored)
        return (
            [self.stored[k] for k in keys],
            [k[0] for k in keys],
            [k[1] for k in keys],
        )

    def commit_retrieve(self, key, instance_id):
        return True

    def close(self):
        pass


def _register(monkeypatch, *, response=None, ctx=None):
    sent: list[RegisterEngineDrivenContextPayload] = []

    def _submit(payload):
        sent.append(payload)
        future = MagicMock()
        future.result.return_value = response or RegisterEngineDrivenContextResponse(
            shm_name="lmcache_l1_pool_t", pool_size=1 << 20
        )
        return future

    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.side_effect = _submit
    holder: dict[str, Any] = {}
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        lambda *_a, **_k: holder.setdefault("ctx", ctx or MagicMock()),
    )
    context = EngineDrivenTransferContext(1, req_client)
    context.register(
        kv_caches=_kv_caches(),
        model_name="m",
        world_size=1,
        blocks_in_chunk=BLOCKS_IN_CHUNK,
        mq_timeout=1.0,
        engine_group_infos=[_group_info()],
        layout_hints={"kv_layout": "NHD"},
    )
    return context, sent


def test_register_quantizes_a_plain_attention_group(monkeypatch, quant_env):
    context, sent = _register(monkeypatch)

    payload = sent[0]
    assert payload.enable_l1_kvweave_quant is True
    (group,) = payload.group_layouts
    assert group.layout_override is not None
    override = deserialize_memory_layout_desc(group.layout_override)
    tokens = BLOCKS_IN_CHUNK * BLOCK_SIZE
    raw_bytes = 2 * NUM_LAYERS * tokens * NUM_HEADS * HEAD_SIZE * 2
    assert override.dtypes == [torch.uint8]
    assert 0 < override.shapes[0].numel() < raw_bytes
    (state,) = context._group_states
    assert state.quantized
    # The worker keeps the raw layout for retrieve; only the wire layout differs.
    assert state.layout_desc.dtypes == [torch.float16]
    assert context._kvweave_codec is not None


def test_register_without_kvweave_keeps_the_upstream_payload(monkeypatch):
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "0")
    context, sent = _register(monkeypatch)

    assert sent[0].enable_l1_kvweave_quant is False
    # No override and (for one plain group) the legacy single-group path.
    assert all(g.layout_override is None for g in sent[0].group_layouts)
    assert context._kvweave_codec is None
    assert context._group_states == []


def test_register_raises_when_the_server_rejects_quantization(monkeypatch, quant_env):
    rejection = RegisterEngineDrivenContextResponse(error="needs variable-size L1")

    with pytest.raises(RuntimeError, match="needs variable-size L1"):
        _register(monkeypatch, response=rejection)


def test_register_refuses_the_inline_fallback_for_quantized_groups(
    monkeypatch, quant_env
):
    inline = MagicMock(spec=EngineDrivenContextInline)

    with pytest.raises(RuntimeError, match="needs the SHM transport"):
        _register(monkeypatch, ctx=inline)

    inline.close.assert_called_once_with()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


def test_store_then_retrieve_round_trips_through_quantized_slots(
    monkeypatch, quant_env
):
    context, sent = _register(monkeypatch)
    (state,) = context._group_states
    assert state.quantized
    assert state.plan is not None and state.plan.quant_layout_desc is not None
    quant_shape = state.plan.quant_layout_desc.shapes[0]
    fake = _FakeShmContext({0: (quant_shape, torch.uint8)})
    context._engine_driven_context = fake

    kv_caches = _kv_caches()
    original = {name: t.clone() for name, t in kv_caches.items()}
    # Block 0 is the null block, so use 1..N; one list per group.
    block_ids = [list(range(1, 1 + NUM_CHUNKS * BLOCKS_IN_CHUNK))]
    key = MagicMock()

    stored_ok = context.submit_store(
        "r", key, kv_caches, block_ids, None, BLOCKS_IN_CHUNK
    ).result()
    assert stored_ok
    # Slots now hold encoded bytes -- not the raw KV.
    stored = fake.stored[(0, 0)]
    assert stored.dtype == torch.uint8
    assert stored.numel() == quant_shape.numel()
    assert stored.abs().sum() > 0

    # Wipe the cache, then retrieve it back from the quantized slots.
    used = torch.tensor(block_ids[0])
    for tensor in kv_caches.values():
        tensor[:, used] = 0
    assert context.submit_retrieve(
        "r", key, kv_caches, block_ids, None, BLOCKS_IN_CHUNK
    ).result()

    for name, tensor in kv_caches.items():
        restored = tensor[:, used]
        assert restored.abs().sum() > 0, f"{name} was not restored"
        similarity = _cosine(restored, original[name][:, used])
        assert similarity > 0.95, f"{name}: cosine {similarity:.4f}"
    # Blocks outside the transfer are untouched.
    used_set = set(block_ids[0])
    untouched = torch.tensor([b for b in range(NUM_BLOCKS) if b not in used_set])
    for name, tensor in kv_caches.items():
        assert torch.equal(tensor[:, untouched], original[name][:, untouched])


class _ScratchShmContext(_FakeShmContext):
    """Fake SHM context that also serves scratch from a host arena."""

    def __init__(self, layouts, scratch_bytes: int):
        super().__init__(layouts)
        # First Party
        from lmcache.v1.multiprocess.scratch_allocator import ScratchAllocator

        self.scratch_bytes = scratch_bytes
        self.allocator = ScratchAllocator(0, scratch_bytes)
        self.arena = torch.zeros(scratch_bytes, dtype=torch.uint8)
        self.requested: list[int] = []

    def allocate_scratch_tensors(self, shape, dtype, count, **_kwargs):
        item = shape.numel() * torch.empty((), dtype=dtype).element_size()
        self.requested.append(item * count)
        allocation = self.allocator.allocate(item * count, wait=False)
        if allocation is None:
            return None
        views = []
        for i in range(count):
            start = allocation.offset + i * item
            views.append(self.arena[start : start + item].view(dtype).view(shape))
        return views, allocation

    def free_scratch(self, allocation):
        self.allocator.free(allocation)

    def fully_free(self) -> bool:
        return self.allocator._free_ranges == [(0, self.scratch_bytes)]


def _spy_gather(monkeypatch) -> list[Any]:
    """Record the ``out`` buffers every gather was given."""
    outs: list[Any] = []
    real = worker_transfer.gather_paged_kv_to_cpu

    def _gather(*args, **kwargs):
        outs.append(kwargs.get("out"))
        return real(*args, **kwargs)

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _gather)
    return outs


def _store_roundtrip_setup(monkeypatch, scratch_bytes: int):
    context, _ = _register(monkeypatch)
    (state,) = context._group_states
    quant_shape = state.plan.quant_layout_desc.shapes[0]
    fake = _ScratchShmContext({0: (quant_shape, torch.uint8)}, scratch_bytes)
    context._engine_driven_context = fake
    block_ids = [list(range(1, 1 + NUM_CHUNKS * BLOCKS_IN_CHUNK))]
    return context, fake, block_ids


def _raw_chunk_bytes() -> int:
    tokens = BLOCKS_IN_CHUNK * BLOCK_SIZE
    return 2 * NUM_LAYERS * tokens * NUM_HEADS * HEAD_SIZE * 2


def test_quantized_group_gathers_into_scratch_and_releases_it(monkeypatch, quant_env):
    context, fake, block_ids = _store_roundtrip_setup(
        monkeypatch, scratch_bytes=4 * _raw_chunk_bytes()
    )
    outs = _spy_gather(monkeypatch)
    kv_caches = _kv_caches()

    ok = context.submit_store(
        "r", MagicMock(), kv_caches, block_ids, None, BLOCKS_IN_CHUNK
    ).result()

    assert ok
    # One request for the whole group, sized for both raw chunks ...
    assert fake.requested == [NUM_CHUNKS * _raw_chunk_bytes()]
    # ... gather wrote into those scratch views, not into the encoded slots ...
    (out,) = outs
    assert out is not None and len(out) == NUM_CHUNKS
    arena_ptr = fake.arena.data_ptr()
    assert all(
        arena_ptr <= t.data_ptr() < arena_ptr + fake.scratch_bytes for t in out
    )
    # ... and the scratch is back once the store committed.
    assert fake.fully_free()


def test_full_scratch_falls_back_to_plain_gather_buffers(monkeypatch, quant_env):
    context, fake, block_ids = _store_roundtrip_setup(
        monkeypatch, scratch_bytes=_raw_chunk_bytes()  # one chunk, two needed
    )
    outs = _spy_gather(monkeypatch)
    kv_caches = _kv_caches()
    original = {n: t.clone() for n, t in kv_caches.items()}

    assert context.submit_store(
        "r", MagicMock(), kv_caches, block_ids, None, BLOCKS_IN_CHUNK
    ).result()

    # Too big for the scratch: gather allocated its own CPU tensors instead.
    assert outs == [None]
    assert fake.fully_free()
    # The stored bytes are still good.
    used = torch.tensor(block_ids[0])
    for tensor in kv_caches.values():
        tensor[:, used] = 0
    assert context.submit_retrieve(
        "r", MagicMock(), kv_caches, block_ids, None, BLOCKS_IN_CHUNK
    ).result()
    for name, tensor in kv_caches.items():
        assert _cosine(tensor[:, used], original[name][:, used]) > 0.95


def test_scratch_is_released_when_encoding_fails(monkeypatch, quant_env):
    context, fake, block_ids = _store_roundtrip_setup(
        monkeypatch, scratch_bytes=4 * _raw_chunk_bytes()
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("encode failed")

    monkeypatch.setattr(context._kvweave_codec, "encode_chunk_into", _boom)

    with pytest.raises(RuntimeError, match="encode failed"):
        context.submit_store(
            "r", MagicMock(), _kv_caches(), block_ids, None, BLOCKS_IN_CHUNK
        )

    assert fake.fully_free()


def test_scratch_is_released_when_gather_fails(monkeypatch, quant_env):
    context, fake, block_ids = _store_roundtrip_setup(
        monkeypatch, scratch_bytes=4 * _raw_chunk_bytes()
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("gather failed")

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", _boom)

    with pytest.raises(RuntimeError, match="gather failed"):
        context.submit_store(
            "r", MagicMock(), _kv_caches(), block_ids, None, BLOCKS_IN_CHUNK
        )

    assert fake.fully_free()


def test_async_context_falls_back_to_the_reusable_pinned_pool():
    # First Party
    from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
        AsyncEngineDrivenTransferContext,
    )

    context = AsyncEngineDrivenTransferContext(1, MagicMock())
    try:
        allocate = context._raw_staging_alloc()
        assert allocate is not None
        shape = torch.Size([4, 8])
        first = allocate(shape, torch.float16, 2)
        assert [t.shape for t in first] == [shape, shape]

        context._release_raw_staging(first)
        again = allocate(shape, torch.float16, 2)

        # Returned tensors are reused, not reallocated.
        assert {t.data_ptr() for t in again} == {t.data_ptr() for t in first}
    finally:
        context.close()
