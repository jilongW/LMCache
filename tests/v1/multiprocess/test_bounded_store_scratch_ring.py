# SPDX-License-Identifier: Apache-2.0
"""Coverage for the per-group scratch ring in the bounded engine-driven
store pipeline (``EngineDrivenTransferContext._run_bounded_store_pipeline``).

Each staged SHM group's raw-gather (depth 2) and quantized split (depth 1) scratch
is reserved once per ``submit_store`` call and reused across every chunk of
that group, instead of round-tripping the shared ``ScratchAllocator`` on
every chunk. These tests exercise that reuse against a *real*
``EngineDrivenContextShm`` (real POSIX SHM segment, real pinned scratch
pool, real device gather/codec) rather than the lightweight fakes used
elsewhere in this package, since the ring's correctness hinges on genuine
concurrency between the gather executor thread and the calling thread.
"""

# Standard
from unittest.mock import MagicMock
import os

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave.kvweave_config import AttentionPlaneLayout
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import KVWeaveCodec
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
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
from lmcache.v1.multiprocess.transfer_context import (
    async_engine_driven,
    worker_transfer,
)
from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
    AsyncEngineDrivenTransferContext,
)
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transfer_context.shm import EngineDrivenContextShm
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    GroupTransferPlan,
)


class _CompletedFuture:
    """A resolved ``MessagingFuture`` stand-in (mirrors the pattern already
    used in ``test_engine_driven_transfer.py``)."""

    def __init__(self, value: object) -> None:
        self._value = value

    def wait(self, timeout: float | None = None) -> bool:  # noqa: ARG002
        return True

    def result(self, timeout: float | None = None) -> object:  # noqa: ARG002
        return self._value


def _make_kv_caches(
    num_layers: int, num_blocks: int, block_size: int, num_heads: int, head_size: int
) -> dict[str, torch.Tensor]:
    """Small real-device KV caches, matching production's split K/V layout."""
    return {
        f"layer_{i}": torch.randn(
            2, num_blocks, block_size, num_heads, head_size, device=torch_device_type
        )
        for i in range(num_layers)
    }


@pytest.fixture
def shm_context_factory(monkeypatch: pytest.MonkeyPatch):
    """Yield a builder for a real ``EngineDrivenContextShm`` backed by a
    throwaway POSIX SHM segment, monkeypatched into
    ``worker_transfer.create_engine_driven_context`` so ``register()``
    returns it. Cleans up the SHM segment on teardown."""
    created: list[tuple[int, str, int]] = []

    def _build(pool_size: int, scratch_size: int) -> EngineDrivenContextShm:
        shm_name = f"lmcache_test_ring_{os.getpid()}_{len(created)}"
        total = pool_size + scratch_size
        addr = shm_create_readwrite(shm_name, total)
        created.append((addr, shm_name, total))
        context_holder: list[EngineDrivenContextShm] = []

        def _factory(metadata, req_client, mq_timeout, **kwargs):  # noqa: ANN001
            context = EngineDrivenContextShm(
                metadata=metadata,
                req_client=req_client,
                mq_timeout=mq_timeout,
                shm_name=shm_name,
                pool_size=pool_size,
                scratch_offset=pool_size,
                scratch_size=scratch_size,
            )
            context_holder.append(context)
            return context

        monkeypatch.setattr(worker_transfer, "create_engine_driven_context", _factory)
        return context_holder, shm_name

    yield _build

    for addr, shm_name, total in created:
        shm_munmap(addr, total)
        try:
            shm_unlink(shm_name)
        except FileNotFoundError:
            pass


def _register(
    shm_context_factory,
    kv_caches: dict[str, torch.Tensor],
    engine_group_infos: list[EngineGroupInfo],
    blocks_in_chunk: int,
    pool_size: int,
    scratch_size: int,
    context_type: type[EngineDrivenTransferContext] = EngineDrivenTransferContext,
) -> tuple[EngineDrivenTransferContext, MagicMock, EngineDrivenContextShm]:
    context_holder, shm_name = shm_context_factory(pool_size, scratch_size)
    req_client = MagicMock()
    register_future = MagicMock()
    register_future.result.return_value = RegisterEngineDrivenContextResponse(
        shm_name=f"lmcache_l1_pool_{shm_name}",
        pool_size=pool_size,
        scratch_offset=pool_size,
        scratch_size=scratch_size,
    )
    req_client.register_kv_cache_engine_driven_context.return_value = register_future

    ctx = context_type()
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=blocks_in_chunk,
        req_client=req_client,
        mq_timeout=1.0,
        engine_group_infos=engine_group_infos,
    )
    return ctx, req_client, context_holder[0]


def _force_quantized_split_kv(
    ctx: EngineDrivenTransferContext, group_index: int, slot_bytes: int
) -> GroupTransferPlan:
    """Overwrite a registered group's plan to force quantized SPLIT_KV
    dispatch, bypassing ``_decide_group_quantization``'s size heuristic
    (irrelevant to testing the ring itself)."""
    plan = ctx._group_plans[group_index]  # noqa: SLF001
    forced = GroupTransferPlan(
        group_info=plan.group_info,
        layer_indices=plan.layer_indices,
        blocks_per_chunk=plan.blocks_per_chunk,
        chunk_shape=plan.chunk_shape,
        engine_kv_format=plan.engine_kv_format,
        attention_plane_layout=AttentionPlaneLayout.SPLIT_KV,
        quantized=True,
        raw_layout_desc=MemoryLayoutDesc(
            shapes=[plan.chunk_shape], dtypes=[torch.float32]
        ),
        quant_layout_desc=MemoryLayoutDesc(
            shapes=[torch.Size([slot_bytes])], dtypes=[torch.uint8]
        ),
    )
    ctx._group_plans[group_index] = forced  # noqa: SLF001
    return forced


def _free_ranges(context: EngineDrivenContextShm) -> list[tuple[int, int]]:
    return context._scratch_allocator._free_ranges  # noqa: SLF001


@pytest.mark.parametrize(
    "context_type", [EngineDrivenTransferContext, AsyncEngineDrivenTransferContext]
)
@pytest.mark.parametrize("global_quant", ["0", "1"])
def test_raw_mamba_uses_single_chunk_scratch_without_encoding(
    shm_context_factory,
    monkeypatch: pytest.MonkeyPatch,
    context_type: type[EngineDrivenTransferContext],
    global_quant: str,
) -> None:
    """Selected raw pages reach SHM byte-for-byte through scratch, not a codec."""
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_LINEAR_QUANT_ENABLED", "0")
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", global_quant)
    source = {
        f"layer_{i}": (
            torch.arange(10 * 2 * 4 * 4 * 32, device=torch_device_type)
            .reshape(10, 2, 4, 4, 32)
            .remainder(101)
            + i
        ).to(torch.bfloat16)
        for i in range(2)
    }
    group = EngineGroupInfo(
        engine_group_id=0,
        layer_indices=(0, 1),
        tokens_per_block=4,
        cache_category="mamba",
        recurrent_state=True,
    )
    ctx, req_client, context = _register(
        shm_context_factory,
        source,
        [group],
        2,
        pool_size=64,
        scratch_size=65536,
        context_type=context_type,
    )
    baseline_free_ranges = list(_free_ranges(context))
    slots = [torch.zeros(2, 2, 8, 128, dtype=torch.bfloat16) for _ in range(2)]
    context._build_slot_tensors = lambda _slots: slots  # noqa: SLF001
    req_client.prepare_store.return_value = _CompletedFuture(
        PrepareStoreResponse(context={"slots": [{}, {}], "chunk_indices": [0, 2]})
    )
    req_client.commit_store.return_value = _CompletedFuture(True)
    codec = MagicMock()
    ctx._kvweave_codec = codec  # noqa: SLF001
    selections: list[list[int]] = []

    def gather(*args, **kwargs) -> list[torch.Tensor]:
        assert len(kwargs["out"]) == 1
        assert kwargs["out"][0].data_ptr() not in {slot.data_ptr() for slot in slots}
        selections.append(kwargs["chunk_indices"])
        return gather_paged_kv_to_cpu(*args, **kwargs)

    monkeypatch.setattr(worker_transfer, "gather_paged_kv_to_cpu", gather)
    monkeypatch.setattr(async_engine_driven, "gather_paged_kv_to_cpu", gather)
    event = torch_dev.Event()
    event.record()
    try:
        result = ctx.submit_store(
            "raw-mamba",
            IPCCacheServerKey.from_token_ids("m", 1, 0, list(range(24))),
            1,
            source,
            [[1, 2, 3, 4, 5, 6]],
            event,
            2,
        )
        assert result.result(timeout=10) is True
        assert selections == [[0], [2]]
        for slot, blocks in zip(slots, ([1, 2], [5, 6]), strict=True):
            for i, tensor in enumerate(source.values()):
                expected = tensor[blocks].permute(1, 0, 2, 3, 4).reshape(2, 8, 128)
                assert torch.equal(slot[:, i].cpu(), expected.cpu())
        assert codec.mock_calls == []
        req_client.commit_store.assert_called_once()
        assert _free_ranges(context) == baseline_free_ranges
    finally:
        ctx.close()


@pytest.mark.parametrize(
    "context_type", [EngineDrivenTransferContext, AsyncEngineDrivenTransferContext]
)
@pytest.mark.parametrize("skip_tokens", [0, 4, 8])
@pytest.mark.parametrize("masked_prefix", [False, True])
@pytest.mark.parametrize("failure", [None, "copy", "scatter"])
def test_raw_mamba_retrieve_stages_each_chunk_and_preserves_read_locks(
    shm_context_factory,
    monkeypatch: pytest.MonkeyPatch,
    context_type: type[EngineDrivenTransferContext],
    skip_tokens: int,
    masked_prefix: bool,
    failure: str | None,
) -> None:
    """Raw H2D uses one scratch slot, no decoder, and commits only after H2D."""
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_LINEAR_QUANT_ENABLED", "0")
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    cpu_source = (
        torch.arange(8 * 2 * 4 * 4 * 32)
        .reshape(8, 2, 4, 4, 32)
        .remainder(101)
        .to(torch.bfloat16)
    )
    destination = {
        f"layer_{i}": torch.zeros_like(cpu_source, device=torch_device_type)
        for i in range(2)
    }
    group = EngineGroupInfo(
        engine_group_id=0,
        layer_indices=(0, 1),
        tokens_per_block=4,
        cache_category="mamba",
        recurrent_state=True,
    )
    ctx, req_client, context = _register(
        shm_context_factory,
        destination,
        [group],
        2,
        pool_size=64,
        scratch_size=65536,
        context_type=context_type,
    )
    baseline_free_ranges = list(_free_ranges(context))
    slots = [
        torch.stack(
            [
                (cpu_source[blocks] + i).permute(1, 0, 2, 3, 4).reshape(2, 8, 128)
                for i in range(2)
            ],
            dim=1,
        ).contiguous()
        for blocks in ([1, 2], [5, 6])
    ]
    if masked_prefix:
        slots = slots[1:]
    if failure == "copy":
        slots[0] = torch.zeros(1, dtype=torch.bfloat16)
    context._build_slot_tensors = lambda _slots: slots  # noqa: SLF001
    req_client.prepare_retrieve.return_value = _CompletedFuture(
        PrepareRetrieveResponse(
            success=True, data=b"", context={"slots": [{}] * len(slots)}
        )
    )
    codec = MagicMock()
    ctx._kvweave_codec = codec  # noqa: SLF001
    calls: list[list[int]] = []

    def scatter(caches, block_ids, chunks, *args, **kwargs) -> None:
        req_client.commit_retrieve.assert_not_called()
        assert len(chunks) == 1
        assert chunks[0].data_ptr() not in {slot.data_ptr() for slot in slots}
        calls.append(block_ids)
        scatter_cpu_to_paged_kv(caches, block_ids, chunks, *args, **kwargs)
        if failure == "scatter":
            raise RuntimeError("injected scatter failure after H2D")

    monkeypatch.setattr(worker_transfer, "scatter_cpu_to_paged_kv", scatter)

    def commit(*args: object) -> _CompletedFuture:
        assert _free_ranges(context) == baseline_free_ranges
        for i, tensor in enumerate(destination.values()):
            actual = tensor.cpu()
            expected = torch.zeros_like(cpu_source)
            if not masked_prefix and skip_tokens < 4:
                expected[3] = cpu_source[1] + i
            if not masked_prefix and skip_tokens < 8:
                expected[4] = cpu_source[2] + i
            expected[5:7] = cpu_source[5:7] + i
            if failure is None:
                assert torch.equal(actual, expected)
        for slot in slots:
            slot.zero_()
        return _CompletedFuture(True)

    req_client.commit_retrieve.side_effect = commit
    try:
        result = ctx.submit_retrieve(
            "raw-retrieve",
            IPCCacheServerKey.from_token_ids("m", 1, 0, list(range(16))),
            1,
            destination,
            [[0, 0, 5, 6]] if masked_prefix else [[3, 4, 5, 6]],
            None,
            2,
            skip_first_n_tokens=skip_tokens,
        )
        # A fully skipped prefix never attempts its deliberately invalid copy.
        copy_skipped = failure == "copy" and not masked_prefix and skip_tokens == 8
        assert result.result(timeout=10) is (failure is None or copy_skipped)
        expected_calls = [[5, 6]]
        if not masked_prefix and skip_tokens < 8:
            expected_calls.insert(0, [3, 4] if skip_tokens == 0 else [4])
        if failure == "copy" and not copy_skipped:
            expected_calls = []
        elif failure == "scatter":
            expected_calls = expected_calls[:1]
        assert calls == expected_calls
        assert codec.mock_calls == []
        req_client.commit_retrieve.assert_called_once()
        assert _free_ranges(context) == baseline_free_ranges
    finally:
        ctx.close()


def test_ring_reused_across_multiple_chunks_in_one_group(
    shm_context_factory,
) -> None:
    """Three chunks of one quantized group, in one ``submit_store`` call,
    must each round-trip correctly through the reused ring slots, and the
    scratch pool must be fully reclaimed (single free range spanning the
    whole pool) once the call returns -- proving the ring was closed, not
    leaked per chunk."""
    num_layers, num_blocks, block_size, num_heads, head_size = 2, 12, 2, 1, 4
    blocks_in_chunk = 2
    kv_caches = _make_kv_caches(
        num_layers, num_blocks, block_size, num_heads, head_size
    )
    group_info = EngineGroupInfo(
        engine_group_id=0,
        layer_indices=tuple(range(num_layers)),
        tokens_per_block=block_size,
        cache_category="attention",
    )
    ctx, req_client, context = _register(
        shm_context_factory,
        kv_caches,
        [group_info],
        blocks_in_chunk,
        pool_size=64,
        scratch_size=8192,
    )
    ctx._kvweave_codec = KVWeaveCodec(  # noqa: SLF001
        {"num_kv_heads": num_heads, "head_dim": head_size}
    )
    slot_bytes = 512
    forced_plan = _force_quantized_split_kv(ctx, 0, slot_bytes)
    baseline_free_ranges = list(_free_ranges(context))

    num_chunks = 3
    slots = [
        torch.zeros(slot_bytes, dtype=torch.uint8, device="cpu")
        for _ in range(num_chunks)
    ]
    # Patch _build_slot_tensors so the SHM-pool slot lookup returns our own
    # CPU tensors directly (pool_size=0 here -- every byte of the segment is
    # reserved for scratch, so there is no real pool to carve slots from).
    context._build_slot_tensors = lambda _slots: slots  # noqa: SLF001
    req_client.prepare_store.return_value = _CompletedFuture(
        PrepareStoreResponse(
            context={
                "slots": [{}] * num_chunks,
                "chunk_indices": list(range(num_chunks)),
            }
        )
    )
    req_client.commit_store.return_value = _CompletedFuture(True)

    block_ids = [list(range(blocks_in_chunk * num_chunks))]
    future = ctx.submit_store(
        "req", MagicMock(), 1, kv_caches, block_ids, MagicMock(), blocks_in_chunk
    )
    assert future.result() is True

    # Independently gather each chunk the same way the pipeline did, and
    # compare each slot's decoded bytes against it -- this is the check that
    # would catch ring-slot reuse corrupting an earlier chunk's data.
    for chunk_idx in range(num_chunks):
        expected = gather_paged_kv_to_cpu(
            kv_caches,
            block_ids[0],
            forced_plan.blocks_per_chunk,
            layout_hints=None,
            engine_kv_format=forced_plan.engine_kv_format,
            chunk_indices=[chunk_idx],
        )[0]
        decoded = ctx._kvweave_codec.decode_chunk(  # noqa: SLF001
            "attention",
            None,
            block_size,
            forced_plan.chunk_shape,
            torch.float32,
            slots[chunk_idx],
            AttentionPlaneLayout.SPLIT_KV,
        )
        assert torch.allclose(decoded, expected, atol=1.0)

    assert _free_ranges(context) == baseline_free_ranges


def test_ring_isolated_across_group_handoff(shm_context_factory) -> None:
    """Two quantized groups with different layer counts (and therefore
    different raw-chunk byte sizes) in one ``submit_store`` call must each
    round-trip through their own, independently-sized ring without either
    group's slots bleeding into the other's."""
    block_size, num_heads, head_size = 2, 1, 4
    blocks_in_chunk = 2
    kv_caches_a = _make_kv_caches(2, 8, block_size, num_heads, head_size)
    kv_caches_b = _make_kv_caches(3, 8, block_size, num_heads, head_size)
    kv_caches = {**kv_caches_a, **{f"b_{k}": v for k, v in kv_caches_b.items()}}
    group_a = EngineGroupInfo(
        engine_group_id=0,
        layer_indices=(0, 1),
        tokens_per_block=block_size,
        cache_category="attention",
    )
    group_b = EngineGroupInfo(
        engine_group_id=1,
        layer_indices=(2, 3, 4),
        tokens_per_block=block_size,
        cache_category="attention",
    )
    ctx, req_client, context = _register(
        shm_context_factory,
        kv_caches,
        [group_a, group_b],
        blocks_in_chunk,
        pool_size=64,
        scratch_size=16384,
    )
    ctx._kvweave_codec = KVWeaveCodec(  # noqa: SLF001
        {"num_kv_heads": num_heads, "head_dim": head_size}
    )
    slot_bytes = 512
    plan_a = _force_quantized_split_kv(ctx, 0, slot_bytes)
    plan_b = _force_quantized_split_kv(ctx, 1, slot_bytes)
    baseline_free_ranges = list(_free_ranges(context))

    # One chunk for group A, two chunks for group B.
    slots = [torch.zeros(slot_bytes, dtype=torch.uint8) for _ in range(3)]
    context._build_slot_tensors = lambda _slots: slots  # noqa: SLF001
    req_client.prepare_store.return_value = _CompletedFuture(
        PrepareStoreResponse(context={"slots": [{}] * 3, "chunk_indices": [0, 1, 2]})
    )
    req_client.commit_store.return_value = _CompletedFuture(True)

    block_ids_a = list(range(blocks_in_chunk))
    block_ids_b = list(range(blocks_in_chunk * 2))
    future = ctx.submit_store(
        "req",
        MagicMock(),
        1,
        kv_caches,
        [block_ids_a, block_ids_b],
        MagicMock(),
        blocks_in_chunk,
    )
    assert future.result() is True

    expected_a = gather_paged_kv_to_cpu(
        plan_a.select_kv_caches(kv_caches),
        block_ids_a,
        plan_a.blocks_per_chunk,
        layout_hints=None,
        engine_kv_format=plan_a.engine_kv_format,
        chunk_indices=[0],
    )[0]
    decoded_a = ctx._kvweave_codec.decode_chunk(  # noqa: SLF001
        "attention",
        None,
        block_size,
        plan_a.chunk_shape,
        torch.float32,
        slots[0],
        AttentionPlaneLayout.SPLIT_KV,
    )
    assert torch.allclose(decoded_a, expected_a, atol=1.0)

    for i in range(2):
        expected_b = gather_paged_kv_to_cpu(
            plan_b.select_kv_caches(kv_caches),
            block_ids_b,
            plan_b.blocks_per_chunk,
            layout_hints=None,
            engine_kv_format=plan_b.engine_kv_format,
            chunk_indices=[i],
        )[0]
        decoded_b = ctx._kvweave_codec.decode_chunk(  # noqa: SLF001
            "attention",
            None,
            block_size,
            plan_b.chunk_shape,
            torch.float32,
            slots[1 + i],
            AttentionPlaneLayout.SPLIT_KV,
        )
        assert torch.allclose(decoded_b, expected_b, atol=1.0)

    assert _free_ranges(context) == baseline_free_ranges


def test_ring_closed_after_encode_failure(shm_context_factory) -> None:
    """If encode raises mid-pipeline, the exception must propagate (no new
    swallowing) AND the group's ring must still be closed -- a subsequent,
    independent ``submit_store`` call against the same scratch pool must
    still succeed, which would fail with a pool-exhaustion error if the
    failed call had leaked its ring."""
    num_layers, block_size, num_heads, head_size = 2, 2, 1, 4
    blocks_in_chunk = 2
    kv_caches = _make_kv_caches(num_layers, 8, block_size, num_heads, head_size)
    group_info = EngineGroupInfo(
        engine_group_id=0,
        layer_indices=(0, 1),
        tokens_per_block=block_size,
        cache_category="attention",
    )
    ctx, req_client, context = _register(
        shm_context_factory,
        kv_caches,
        [group_info],
        blocks_in_chunk,
        pool_size=64,
        scratch_size=4096,
    )
    ctx._kvweave_codec = KVWeaveCodec(  # noqa: SLF001
        {"num_kv_heads": num_heads, "head_dim": head_size}
    )
    slot_bytes = 512
    _force_quantized_split_kv(ctx, 0, slot_bytes)
    baseline_free_ranges = list(_free_ranges(context))

    slots = [torch.zeros(slot_bytes, dtype=torch.uint8)]
    context._build_slot_tensors = lambda _slots: slots  # noqa: SLF001
    req_client.prepare_store.return_value = _CompletedFuture(
        PrepareStoreResponse(context={"slots": [{}], "chunk_indices": [0]})
    )
    req_client.commit_store.return_value = _CompletedFuture(True)

    original_encode = ctx._encode_group_chunks_into_slots  # noqa: SLF001

    def _raise(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("boom: simulated encode failure")

    ctx._encode_group_chunks_into_slots = _raise  # noqa: SLF001
    block_ids = [list(range(blocks_in_chunk))]
    with pytest.raises(RuntimeError, match="boom"):
        ctx.submit_store(
            "req", MagicMock(), 1, kv_caches, block_ids, MagicMock(), blocks_in_chunk
        )

    assert _free_ranges(context) == baseline_free_ranges

    # Restore the real encode path and confirm the pool still works cleanly.
    ctx._encode_group_chunks_into_slots = original_encode  # noqa: SLF001
    slots[0].zero_()
    future = ctx.submit_store(
        "req2", MagicMock(), 1, kv_caches, block_ids, MagicMock(), blocks_in_chunk
    )
    assert future.result() is True
    assert _free_ranges(context) == baseline_free_ranges
