# SPDX-License-Identifier: Apache-2.0
"""Multi-group (hybrid Mamba+attention) coverage for the engine-driven
transfer path: per-group gather/scatter helpers and store/retrieve
concatenation across LMCache groups.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)


def _make_kv_caches(
    num_layers: int,
    num_blocks: int = 6,
    block_size: int = 4,
    num_heads: int = 2,
    head_size: int = 8,
    prefix: str = "layer_",
) -> dict[str, torch.Tensor]:
    return {
        f"{prefix}{i}": torch.randn(2, num_blocks, block_size, num_heads, head_size)
        for i in range(num_layers)
    }


class TestKvCachesForGroup:
    def test_returns_all_caches_for_none_group(self) -> None:
        kv_caches = _make_kv_caches(4)
        assert worker_transfer._kv_caches_for_group(kv_caches, None) is kv_caches

    def test_filters_by_layer_indices(self) -> None:
        kv_caches = _make_kv_caches(4)
        group = EngineGroupInfo(engine_group_id=0, layer_indices=(1, 3))
        filtered = worker_transfer._kv_caches_for_group(kv_caches, group)
        assert list(filtered.keys()) == ["layer_1", "layer_3"]


class TestBlocksPerChunkForGroup:
    def test_none_group_returns_default(self) -> None:
        assert worker_transfer._blocks_per_chunk_for_group(None, 4, 16) == 4

    def test_group_without_tokens_per_block_returns_default(self) -> None:
        group = EngineGroupInfo(engine_group_id=0, tokens_per_block=0)
        assert worker_transfer._blocks_per_chunk_for_group(group, 4, 16) == 4

    def test_smaller_tokens_per_block_yields_more_blocks(self) -> None:
        """A Mamba group with tokens_per_block=1 needs one block per token,
        so the same 64-token chunk needs 64 blocks instead of 4."""
        group = EngineGroupInfo(engine_group_id=1, tokens_per_block=1)
        assert worker_transfer._blocks_per_chunk_for_group(group, 4, 16) == 64

    def test_misaligned_tokens_per_block_raises(self) -> None:
        group = EngineGroupInfo(engine_group_id=1, tokens_per_block=5)
        with pytest.raises(ValueError, match="must be a multiple of"):
            worker_transfer._blocks_per_chunk_for_group(group, 4, 16)


class TestGroupChunkShape:
    def test_none_group_returns_default_shape(self) -> None:
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([2, 4, 64, 16])], dtypes=[torch.float16]
        )
        shape = worker_transfer._group_chunk_shape(None, layout_desc, 4)
        assert shape == torch.Size([2, 4, 64, 16])

    def test_substitutes_group_layer_count(self) -> None:
        """A group with 6 of the 24 registered layers gets a chunk shape
        scaled to 6 layers, other dims unchanged."""
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([2, 24, 64, 16])], dtypes=[torch.float16]
        )
        group = EngineGroupInfo(engine_group_id=0, layer_indices=tuple(range(6)))
        shape = worker_transfer._group_chunk_shape(group, layout_desc, 24)
        assert shape == torch.Size([2, 6, 64, 16])

    def test_substitutes_group_layer_count_mla_shape(self) -> None:
        """MLA/fused-K/V layouts have no leading kv-plane dim; layer count is
        dim 0, not dim 1."""
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([24, 64, 16])], dtypes=[torch.float16]
        )
        group = EngineGroupInfo(engine_group_id=1, layer_indices=tuple(range(6, 24)))
        shape = worker_transfer._group_chunk_shape(group, layout_desc, 24)
        assert shape == torch.Size([18, 64, 16])


class TestIterTransferGroups:
    def test_empty_groups_yields_single_group_fallback(self) -> None:
        kv_caches = _make_kv_caches(2)
        result = list(
            worker_transfer._iter_transfer_groups([], kv_caches, [[0, 1]], 2, 4)
        )
        assert len(result) == 1
        group_info, group_kv_caches, block_ids, blocks_per_chunk = result[0]
        assert group_info is None
        assert group_kv_caches is kv_caches
        assert block_ids == [0, 1]
        assert blocks_per_chunk == 2

    def test_empty_groups_rejects_multi_group_block_ids(self) -> None:
        kv_caches = _make_kv_caches(2)
        with pytest.raises(RuntimeError, match="does not support hybrid"):
            list(worker_transfer._iter_transfer_groups([], kv_caches, [[0], [1]], 2, 4))

    def test_yields_one_entry_per_group_in_order(self) -> None:
        kv_caches = _make_kv_caches(4)
        groups = [
            EngineGroupInfo(
                engine_group_id=0, layer_indices=(0, 1), tokens_per_block=4
            ),
            EngineGroupInfo(
                engine_group_id=1, layer_indices=(2, 3), tokens_per_block=1
            ),
        ]
        result = list(
            worker_transfer._iter_transfer_groups(
                groups, kv_caches, [[0, 1], list(range(8))], 2, 4
            )
        )
        assert len(result) == 2
        assert list(result[0][1].keys()) == ["layer_0", "layer_1"]
        assert result[0][2] == [0, 1]
        assert result[0][3] == 2
        assert list(result[1][1].keys()) == ["layer_2", "layer_3"]
        assert result[1][2] == list(range(8))
        assert result[1][3] == 8

    def test_rejects_block_id_group_count_mismatch(self) -> None:
        kv_caches = _make_kv_caches(2)
        groups = [
            EngineGroupInfo(engine_group_id=0, layer_indices=(0,)),
            EngineGroupInfo(engine_group_id=1, layer_indices=(1,)),
        ]
        with pytest.raises(ValueError, match="Expected 2 block-id groups"):
            list(
                worker_transfer._iter_transfer_groups(groups, kv_caches, [[0, 1]], 2, 4)
            )


class _FakeEngineDrivenContext:
    """Minimal engine-driven context for multi-group submit_store/retrieve
    tests: pickle transport, always signals new chunks to gather."""

    def __init__(self) -> None:
        self.layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32]
        )
        self.committed_chunks: list[torch.Tensor] | None = None
        self.retrieve_chunks: list[torch.Tensor] | None = None

    def prepare_store(self, _key: object, _instance_id: int):
        return None

    def commit_store(
        self, _key: object, _instance_id: int, chunks: list[torch.Tensor]
    ) -> bool:
        self.committed_chunks = chunks
        return True

    def prepare_retrieve(self, _key: object, _instance_id: int):
        return self.retrieve_chunks

    def commit_retrieve(self, _key: object, _instance_id: int) -> bool:
        return True

    def close(self) -> None:
        return None


def _register_two_group_context(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[EngineDrivenTransferContext, _FakeEngineDrivenContext]:
    """Register an EngineDrivenTransferContext with two same-shape groups
    (2 layers each) so gather/scatter round-trips are easy to assert on."""
    fake_context = _FakeEngineDrivenContext()
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = MagicMock(shm_name="", pool_size=0)

    ctx = EngineDrivenTransferContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    groups = [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1), tokens_per_block=4),
        EngineGroupInfo(engine_group_id=1, layer_indices=(2, 3), tokens_per_block=4),
    ]
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
        engine_group_infos=groups,
    )
    return ctx, fake_context


def test_submit_store_concatenates_chunks_group_major(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two same-shape groups each contribute one chunk; commit_store must
    receive both, group 0's chunk first."""
    ctx, fake_context = _register_two_group_context(monkeypatch)
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)

    future = ctx.submit_store(
        "req",
        MagicMock(),
        1,
        kv_caches,
        [[0, 1], [4, 5]],
        MagicMock(),
        blocks_in_chunk=2,
    )

    assert future.result() is True
    assert fake_context.committed_chunks is not None
    assert len(fake_context.committed_chunks) == 2
    for chunk in fake_context.committed_chunks:
        assert tuple(chunk.shape) == (2, 2, 8, 16)


def test_submit_retrieve_scatters_chunks_group_major(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two groups' chunks (already gathered by a prior store) scatter back
    into the correct group's layers only."""
    ctx, fake_context = _register_two_group_context(monkeypatch)
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)

    # First populate fake_context.retrieve_chunks with a real gather so the
    # scatter step has valid data to round-trip.
    store_future = ctx.submit_store(
        "req", MagicMock(), 1, kv_caches, [[0, 1], [4, 5]], MagicMock(), 2
    )
    assert store_future.result() is True
    fake_context.retrieve_chunks = fake_context.committed_chunks

    destination = {name: torch.zeros_like(tensor) for name, tensor in kv_caches.items()}
    retrieve_future = ctx.submit_retrieve(
        "req", MagicMock(), 1, destination, [[2, 3], [6, 7]], MagicMock(), 2
    )

    assert retrieve_future.result() is True
    # group 0 (layer_0, layer_1) stored from source blocks [0, 1] and must
    # retrieve into destination blocks [2, 3] with the same values.
    assert torch.allclose(destination["layer_0"][:, 2], kv_caches["layer_0"][:, 0])
    assert torch.allclose(destination["layer_0"][:, 3], kv_caches["layer_0"][:, 1])
    # group 1 (layer_2, layer_3) stored from source blocks [4, 5] and must
    # retrieve into destination blocks [6, 7] with the same values.
    assert torch.allclose(destination["layer_2"][:, 6], kv_caches["layer_2"][:, 4])
    assert torch.allclose(destination["layer_2"][:, 7], kv_caches["layer_2"][:, 5])


def test_submit_store_narrows_shm_out_buffers_to_group_chunk_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SHM mode's prepare_store returns one flat out_buffers/chunk_indices
    list spanning all groups; each group's gather must only touch its own
    slice, not the other group's buffers."""

    class _ShmFakeContext(_FakeEngineDrivenContext):
        def __init__(self) -> None:
            super().__init__()
            self.out_buffers = [
                torch.zeros(2, 2, 8, 16),
                torch.zeros(2, 2, 8, 16),
            ]

        def prepare_store(self, _key: object, _instance_id: int):
            return self.out_buffers, [0, 1]

    fake_context = _ShmFakeContext()
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = MagicMock(shm_name="pool", pool_size=4096)

    ctx = EngineDrivenTransferContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    groups = [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1), tokens_per_block=4),
        EngineGroupInfo(engine_group_id=1, layer_indices=(2, 3), tokens_per_block=4),
    ]
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
        engine_group_infos=groups,
    )

    result = ctx.submit_store(
        "req", MagicMock(), 1, kv_caches, [[0, 1], [4, 5]], MagicMock(), 2
    )

    assert result.result() is True
    # Each group's gather wrote into its own out buffer (not left at the
    # all-zeros initial value, and not aliasing the other group's buffer).
    assert not torch.all(fake_context.out_buffers[0] == 0)
    assert not torch.all(fake_context.out_buffers[1] == 0)
    assert not torch.allclose(fake_context.out_buffers[0], fake_context.out_buffers[1])
    # The independently-gathered flat chunks (all group0.0, group0.1
    # unavailable since chunk_indices selects both, so both groups gather
    # fully) must exactly equal what a direct, non-SHM gather over the same
    # group's layers/block-ids would produce.
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu

    expected_group0 = gather_paged_kv_to_cpu(
        {"layer_0": kv_caches["layer_0"], "layer_1": kv_caches["layer_1"]}, [0, 1], 2
    )[0]
    expected_group1 = gather_paged_kv_to_cpu(
        {"layer_2": kv_caches["layer_2"], "layer_3": kv_caches["layer_3"]}, [4, 5], 2
    )[0]
    assert torch.allclose(fake_context.out_buffers[0], expected_group0)
    assert torch.allclose(fake_context.out_buffers[1], expected_group1)
