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
from lmcache.v1.distributed.serde.kvweave.kvweave_config import MambaCodecOptions
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import (
    MambaChunkSplit,
    _KVWeaveCodec,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo, MambaSubStateWireLayout
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
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


class TestMambaRhStartupFallback:
    def test_fallback_disables_conv_rh_when_transform_len_not_pow2(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        options = MambaCodecOptions(
            conv_scaling_method="per_channel",
            conv_rh=True,
            ssm_scaling_method="per_channel",
            ssm_rh=False,
            asym=True,
        )
        raw_layout = MemoryLayoutDesc(
            shapes=[torch.Size([2, 3, 4, 3072])], dtypes=[torch.float16]
        )
        conv_layout = MambaSubStateWireLayout(
            byte_offset=0,
            byte_length=0,
            dtype_str="torch.float16",
            shape=(8, 128, 3),
        )
        ssm_layout = MambaSubStateWireLayout(
            byte_offset=0,
            byte_length=0,
            dtype_str="torch.float16",
            shape=(4, 4, 4),
        )

        resolved = worker_transfer._maybe_fallback_invalid_mamba_rh(
            options,
            linear_quant_enabled=True,
            group_cache_categories=["mamba"],
            group_layouts=[raw_layout],
            group_mamba_layouts=[(conv_layout, ssm_layout)],
            group_tokens_per_block=[1],
        )

        assert resolved is not None
        assert resolved.conv_rh is False
        assert "disabling conv RH at startup" in caplog.text

    def test_fallback_keeps_conv_rh_when_transform_len_is_pow2(self) -> None:
        options = MambaCodecOptions(
            conv_scaling_method="per_channel",
            conv_rh=True,
            ssm_scaling_method="per_channel",
            ssm_rh=False,
            asym=True,
        )
        raw_layout = MemoryLayoutDesc(
            shapes=[torch.Size([2, 3, 4, 4096])], dtypes=[torch.float16]
        )
        conv_layout = MambaSubStateWireLayout(
            byte_offset=0,
            byte_length=0,
            dtype_str="torch.float16",
            shape=(8, 128, 4),
        )
        ssm_layout = MambaSubStateWireLayout(
            byte_offset=0,
            byte_length=0,
            dtype_str="torch.float16",
            shape=(4, 4, 4),
        )

        resolved = worker_transfer._maybe_fallback_invalid_mamba_rh(
            options,
            linear_quant_enabled=True,
            group_cache_categories=["mamba"],
            group_layouts=[raw_layout],
            group_mamba_layouts=[(conv_layout, ssm_layout)],
            group_tokens_per_block=[1],
        )

        assert resolved is not None
        assert resolved.conv_rh is True


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


def test_submit_retrieve_skips_all_null_destination_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mamba null pages must not receive retrieved data or shift later groups."""
    ctx, fake_context = _register_two_group_context(monkeypatch)
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)

    store_future = ctx.submit_store(
        "req", MagicMock(), 1, kv_caches, [[1, 2], [4, 5]], MagicMock(), 2
    )
    assert store_future.result() is True
    fake_context.retrieve_chunks = fake_context.committed_chunks

    destination = {name: torch.zeros_like(tensor) for name, tensor in kv_caches.items()}
    retrieve_future = ctx.submit_retrieve(
        "req", MagicMock(), 1, destination, [[0, 0], [6, 7]], MagicMock(), 2
    )

    assert retrieve_future.result() is True
    assert torch.count_nonzero(destination["layer_0"]) == 0
    assert torch.count_nonzero(destination["layer_1"]) == 0
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


def test_submit_store_retrieve_round_trips_kvweave_quantized_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With LMCACHE_MP_L1_KVWEAVE_QUANT=1, submit_store must serialize each
    quantized group's chunk through the KVWeave codec (not commit raw
    bytes), and submit_retrieve must dequantize those bytes back into the
    real KV shape before scattering — regression test for a store/retrieve
    dtype mismatch where retrieve scattered still-encoded uint8 bytes
    straight into the paged KV cache.

    Uses a larger chunk (64 blocks x 128 hidden) than the other tests in
    this module: KVWeave's fixed ~4KB header overhead only pays off once the
    raw tensor is big enough, so register() would otherwise fall back to
    the raw (non-quantized) layout and the codec path would go untested.
    """
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    fake_context = _FakeEngineDrivenContext()
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = MagicMock(shm_name="", pool_size=0)

    ctx = EngineDrivenTransferContext()
    kv_caches = _make_kv_caches(
        2, num_blocks=64, block_size=1, num_heads=1, head_size=128
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

    store_future = ctx.submit_store(
        "req", MagicMock(), 1, kv_caches, [list(range(64))], MagicMock(), 64
    )
    assert store_future.result() is True
    assert fake_context.committed_chunks is not None
    # The quantized group must not commit a raw fp32 chunk: KVWeave's 4-bit
    # payload is far smaller than the raw (2, 2, 64, 128) float32 chunk.
    raw_nbytes = kv_caches["layer_0"].numel() * kv_caches["layer_0"].element_size()
    for chunk in fake_context.committed_chunks:
        assert chunk.dtype == torch.uint8
        assert chunk.numel() < raw_nbytes

    fake_context.retrieve_chunks = fake_context.committed_chunks
    destination = {name: torch.zeros_like(tensor) for name, tensor in kv_caches.items()}
    retrieve_future = ctx.submit_retrieve(
        "req", MagicMock(), 1, destination, [list(range(64))], MagicMock(), 64
    )

    assert retrieve_future.result() is True
    # Dequantized values must be close to the originals (4-bit quant has
    # bounded error, not bit-exact).
    assert torch.max(
        torch.abs(destination["layer_0"] - kv_caches["layer_0"])
    ) < 1.0
    assert torch.max(
        torch.abs(destination["layer_1"] - kv_caches["layer_1"])
    ) < 1.0


def _mamba_group_layout() -> tuple[MambaSubStateWireLayout, MambaSubStateWireLayout]:
    """Conv/ssm byte layout for a 128-hidden, fp32, single-token-per-block
    Mamba page (page_bytes = 2*1*128*4 = 1024): conv takes the first 128
    bytes (32 fp32 elements), ssm the remaining 896 (224 fp32 elements)."""
    return (
        MambaSubStateWireLayout(0, 128, "torch.float32", (32,)),
        MambaSubStateWireLayout(128, 896, "torch.float32", (224,)),
    )


def test_submit_store_retrieve_round_trips_mamba_group_without_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the bug where a Mamba group's opaque page-view
    chunk was quantized through the attention KVWeave codec (silently
    corrupting conv/ssm state) instead of Phase 4's dedicated Mamba codec.

    The primary assertion is behavioral (which codec method gets called for
    the Mamba group), not a numeric error threshold: a per-channel attention
    codec quantizes finely enough that a same-process quantize-then-
    dequantize round trip can still land within a numeric tolerance even
    when it is semantically the wrong codec (tried several adversarial
    magnitude constructions and none reliably discriminated correct from
    buggy dispatch without becoming coupled to unrelated codec internals
    like per-channel scale granularity or randomized-Hadamard preconditioning
    precision floors — see the investigation notes in MIGRATION_PLAN.md
    Phase 6). Asserting the call site directly is the robust, deterministic
    way to verify this dispatch decision, consistent with this repo's
    existing convention of preferring observable-behavior assertions over
    accessing private state (see Phase 1's test notes)."""
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    fake_context = _FakeEngineDrivenContext()
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = MagicMock(shm_name="", pool_size=0)

    split_spy = MagicMock(wraps=_KVWeaveCodec.split_mamba_chunk)
    merge_spy = MagicMock(wraps=_KVWeaveCodec.merge_mamba_chunk)
    serialize_spy = MagicMock(wraps=_KVWeaveCodec.serialize_tensor)
    monkeypatch.setattr(_KVWeaveCodec, "split_mamba_chunk", split_spy)
    monkeypatch.setattr(_KVWeaveCodec, "merge_mamba_chunk", merge_spy)
    monkeypatch.setattr(_KVWeaveCodec, "serialize_tensor", serialize_spy)

    ctx = EngineDrivenTransferContext()
    kv_caches = _make_kv_caches(
        2, num_blocks=65, block_size=1, num_heads=1, head_size=128
    )
    mamba_layout = _mamba_group_layout()
    groups = [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0, 1),
            tokens_per_block=1,
            cache_category="mamba",
            mamba_real_layout=mamba_layout,
        )
    ]
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=64,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
        engine_group_infos=groups,
    )

    store_future = ctx.submit_store(
        "req", MagicMock(), 1, kv_caches, [list(range(1, 65))], MagicMock(), 64
    )
    assert store_future.result() is True
    assert fake_context.committed_chunks is not None
    raw_nbytes = kv_caches["layer_0"].numel() * kv_caches["layer_0"].element_size()
    for chunk in fake_context.committed_chunks:
        assert chunk.dtype == torch.uint8
        conv_payload, ssm_payload = _KVWeaveCodec.unpack_mamba_payloads(
            _KVWeaveCodec._tensor_bytes(chunk)
        )
        assert 8 + len(conv_payload) + len(ssm_payload) < raw_nbytes

    fake_context.retrieve_chunks = fake_context.committed_chunks
    destination = {name: torch.zeros_like(tensor) for name, tensor in kv_caches.items()}
    retrieve_future = ctx.submit_retrieve(
        "req", MagicMock(), 1, destination, [list(range(1, 65))], MagicMock(), 64
    )
    assert retrieve_future.result() is True

    # The Mamba group must go through Phase 4's dedicated split/merge codec,
    # never through the generic attention serialize_tensor -- this is what
    # the pre-fix code got backwards (it called serialize_tensor on every
    # quantized group regardless of cache_category).
    assert split_spy.call_count >= 1
    assert merge_spy.call_count >= 1
    assert serialize_spy.call_count == 0

    # Sanity check: the correct codec's round trip is still numerically
    # faithful (bounded 4-bit quantization error), not just "some codec ran".
    retrieved_raw_chunks = gather_paged_kv_to_cpu(destination, list(range(1, 65)), 64)
    original_raw_chunks = gather_paged_kv_to_cpu(kv_caches, list(range(1, 65)), 64)
    retrieved_split = _KVWeaveCodec.split_mamba_chunk(
        retrieved_raw_chunks[0], mamba_layout, block_size=1
    )
    original_split = _KVWeaveCodec.split_mamba_chunk(
        original_raw_chunks[0], mamba_layout, block_size=1
    )
    assert torch.max(torch.abs(retrieved_split.conv - original_split.conv)) < 1.0
    assert torch.max(torch.abs(retrieved_split.ssm - original_split.ssm)) < 1.0


def test_submit_retrieve_mamba_skip_first_n_tokens_survives_leading_null_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: a leading all-null Mamba chunk must not desync
    ``skip_first_n_tokens`` from the real chunk that follows it.

    ``submit_retrieve`` used to drop leading all-null chunks from
    ``group_block_ids``/``group_chunks`` before handing them to
    ``_scatter_non_null_mamba_chunks``, but that helper's internal
    ``block_ordinal`` counter (compared against
    ``skip_first_n_tokens // block_size``) assumes it is walking the
    *original*, uncompacted block sequence. Dropping the leading null chunk
    shifted every later block's ordinal down, so a nonzero
    ``skip_first_n_tokens`` sized to skip only the null chunk would also
    skip the first real block after it -- silently leaving that Mamba
    state block never written (stale/garbage device memory) instead of
    corrupting it with a small numeric error.
    """
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    fake_context = _FakeEngineDrivenContext()
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = MagicMock(shm_name="", pool_size=0)

    ctx = EngineDrivenTransferContext()
    kv_caches = _make_kv_caches(
        2, num_blocks=8, block_size=1, num_heads=1, head_size=128
    )
    mamba_layout = _mamba_group_layout()
    groups = [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0, 1),
            tokens_per_block=1,
            cache_category="mamba",
            mamba_real_layout=mamba_layout,
        )
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

    # Store real state for blocks 5 and 6 (chunk 0 = [0, 0] null padding,
    # chunk 1 = [5, 6] real blocks).
    store_future = ctx.submit_store(
        "req", MagicMock(), 1, kv_caches, [[0, 0, 5, 6]], MagicMock(), 2
    )
    assert store_future.result() is True
    fake_context.retrieve_chunks = fake_context.committed_chunks

    # Retrieve with skip_first_n_tokens sized to skip exactly the leading
    # null chunk (2 tokens = 1 block_size each): the real blocks 5 and 6
    # must still be written, not skipped as a side effect of the null
    # chunk being dropped before the ordinal count.
    destination = {name: torch.zeros_like(tensor) for name, tensor in kv_caches.items()}
    retrieve_future = ctx.submit_retrieve(
        "req", MagicMock(), 1, destination, [[0, 0, 5, 6]], MagicMock(), 2,
        skip_first_n_tokens=2,
    )
    assert retrieve_future.result() is True

    for name in kv_caches:
        assert torch.count_nonzero(destination[name][:, 5]) > 0, (
            f"{name} block 5 was never written back (skip desync bug)"
        )
        assert torch.count_nonzero(destination[name][:, 6]) > 0, (
            f"{name} block 6 was never written back (skip desync bug)"
        )
        assert torch.max(
            torch.abs(destination[name][:, 5] - kv_caches[name][:, 5])
        ) < 1.0
        assert torch.max(
            torch.abs(destination[name][:, 6] - kv_caches[name][:, 6])
        ) < 1.0


def test_register_leaves_mamba_group_unquantized_without_real_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Mamba group with cache_category='mamba' but no mamba_real_layout
    (e.g. the engine could not compute it) must never be quantized -- there
    is no safe way to split it into conv/ssm, so silently falling back to
    the attention codec (the original bug) or crashing are both wrong."""
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    fake_context = _FakeEngineDrivenContext()
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = MagicMock(shm_name="", pool_size=0)

    ctx = EngineDrivenTransferContext()
    kv_caches = _make_kv_caches(
        2, num_blocks=64, block_size=1, num_heads=1, head_size=128
    )
    groups = [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0, 1),
            tokens_per_block=1,
            cache_category="mamba",
            mamba_real_layout=None,
        )
    ]
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=64,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
        engine_group_infos=groups,
    )

    assert ctx._group_layout_descs[0].dtypes != [torch.uint8]


def test_register_sends_group_layouts_when_all_groups_are_raw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_LINEAR_QUANT_ENABLED", "0")
    fake_context = _FakeEngineDrivenContext()
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = MagicMock(shm_name="", pool_size=0)
    send_request = MagicMock(return_value=future)
    kv_caches = _make_kv_caches(
        4, num_blocks=64, block_size=1, num_heads=1, head_size=128
    )
    groups = [
        EngineGroupInfo(
            engine_group_id=group_id,
            layer_indices=layer_indices,
            tokens_per_block=1,
            cache_category="mamba",
            mamba_real_layout=_mamba_group_layout(),
        )
        for group_id, layer_indices in enumerate(((0, 1), (2, 3)))
    ]

    ctx = EngineDrivenTransferContext()
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=64,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=send_request,
        engine_group_infos=groups,
    )

    payload = send_request.call_args.args[2][0]
    assert payload.enable_l1_kvweave_quant is False
    assert payload.group_layout_descs is not None
    assert len(payload.group_layout_descs) == 2


def test_submit_store_hybrid_attention_and_mamba_groups_independently_quantized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hybrid registration with one attention group and one Mamba group:
    each group's quantize-or-not decision and codec must be independent --
    neither group's classification should affect the other's."""
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    fake_context = _FakeEngineDrivenContext()
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = MagicMock(shm_name="", pool_size=0)

    ctx = EngineDrivenTransferContext()
    kv_caches = _make_kv_caches(
        4, num_blocks=65, block_size=1, num_heads=1, head_size=128
    )
    mamba_layout = _mamba_group_layout()
    groups = [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0, 1),
            tokens_per_block=1,
            cache_category="attention",
        ),
        EngineGroupInfo(
            engine_group_id=1,
            layer_indices=(2, 3),
            tokens_per_block=1,
            cache_category="mamba",
            mamba_real_layout=mamba_layout,
        ),
    ]
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=64,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=MagicMock(return_value=future),
        engine_group_infos=groups,
    )
    assert ctx._group_layout_descs[0].dtypes == [torch.uint8]
    assert ctx._group_layout_descs[1].dtypes == [torch.uint8]

    store_future = ctx.submit_store(
        "req", MagicMock(), 1, kv_caches, [list(range(64)), list(range(1, 65))],
        MagicMock(), 64,
    )
    assert store_future.result() is True
    assert fake_context.committed_chunks is not None
    for chunk in fake_context.committed_chunks:
        assert chunk.dtype == torch.uint8

    fake_context.retrieve_chunks = fake_context.committed_chunks
    destination = {name: torch.zeros_like(tensor) for name, tensor in kv_caches.items()}
    retrieve_future = ctx.submit_retrieve(
        "req", MagicMock(), 1, destination, [list(range(64)), list(range(1, 65))],
        MagicMock(), 64,
    )
    assert retrieve_future.result() is True
    for name in kv_caches:
        if name in {"layer_2", "layer_3"}:
            assert torch.isfinite(destination[name][1:]).all()
        else:
            assert torch.isfinite(destination[name][:64]).all()
