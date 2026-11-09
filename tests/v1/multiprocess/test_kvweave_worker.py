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
