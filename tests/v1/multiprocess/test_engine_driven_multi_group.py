# SPDX-License-Identifier: Apache-2.0
"""Multi-group (hybrid Mamba+attention) coverage for the engine-driven
transfer path: per-group gather/scatter helpers and store/retrieve
concatenation across LMCache groups.
"""

# Standard
from contextlib import nullcontext
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
import lmcache.lmcache_native as lmcache_native
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    AttentionPlaneLayout,
    KVWeaveRuntimeConfig,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import KVWeaveCodec
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.engine import (
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transfer_context import (
    async_engine_driven,
    worker_transfer,
)
from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
    AsyncEngineDrivenTransferContext,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    null_chunk_mask_from_groups,
)


def _disabled_kvweave_config() -> KVWeaveRuntimeConfig:
    """A KVWeaveRuntimeConfig with quantization off, for plan-building tests
    that are not exercising the quantization decision itself."""
    return KVWeaveRuntimeConfig(
        enabled=False, linear_quant_enabled=False, linear_max_size_ratio=1.2
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


class TestSelectGroupChunks:
    """Rebasing the server's flat, group-major chunk selection onto one group.

    Shared by the sync and async store paths, so a regression here silently
    misroutes one group's chunks into another group's buffers.
    """

    def test_none_selection_selects_every_chunk_in_order(self) -> None:
        selection = worker_transfer._select_group_chunks(None, 4, 3)
        assert selection.num_group_chunks == 3
        assert selection.chunk_indices is None
        assert selection.out_indices == [0, 1, 2]
        # A None selection means "store everything", never skip the group.
        assert not selection.is_empty

    def test_rebases_second_group_indices_to_group_local(self) -> None:
        """Group 1 owns flat chunks [2, 4); its own indices must be [0, 2)."""
        selection = worker_transfer._select_group_chunks([0, 1, 2, 3], 2, 2)
        assert selection.chunk_indices == [0, 1]
        # out_indices stay flat: they index the server's out_buffers list.
        assert selection.out_indices == [2, 3]

    def test_excludes_other_groups_chunks(self) -> None:
        selection = worker_transfer._select_group_chunks([0, 3], 2, 2)
        assert selection.chunk_indices == [1]
        assert selection.out_indices == [1]

    def test_group_with_nothing_selected_is_empty(self) -> None:
        """Group 0 fully cached: it must report empty so callers skip it,
        while still advancing the offset by its own chunk count."""
        selection = worker_transfer._select_group_chunks([1], 0, 1)
        assert selection.chunk_indices == []
        assert selection.out_indices == []
        assert selection.is_empty
        assert selection.num_group_chunks == 1

    def test_out_indices_track_flat_position_not_group_position(self) -> None:
        """A sparse selection must keep out_indices aligned with the flat
        out_buffers list, not renumber them per group."""
        selection = worker_transfer._select_group_chunks([1, 5, 6], 5, 3)
        assert selection.chunk_indices == [0, 1]
        assert selection.out_indices == [1, 2]


class TestGroupChunkShape:
    def test_none_group_returns_default_shape(self) -> None:
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([2, 4, 64, 16])], dtypes=[torch.float16]
        )
        shape, group_use_mla = worker_transfer._group_chunk_shape(
            None, layout_desc, 4, 16, {}, None
        )
        assert shape == torch.Size([2, 4, 64, 16])
        assert group_use_mla is None

    def test_substitutes_group_layer_count(self) -> None:
        """A group with 6 of the 24 registered layers gets a chunk shape
        scaled to 6 layers, other dims unchanged."""
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([2, 24, 64, 16])], dtypes=[torch.float16]
        )
        group = EngineGroupInfo(
            engine_group_id=0, layer_indices=tuple(range(6)), recurrent_state=True
        )
        shape, group_use_mla = worker_transfer._group_chunk_shape(
            group, layout_desc, 24, 16, {}, None
        )
        assert shape == torch.Size([2, 6, 64, 16])
        assert group_use_mla is None

    def test_substitutes_group_layer_count_mla_shape(self) -> None:
        """MLA/fused-K/V layouts have no leading kv-plane dim; layer count is
        dim 0, not dim 1."""
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([24, 64, 16])], dtypes=[torch.float16]
        )
        group = EngineGroupInfo(
            engine_group_id=1,
            layer_indices=tuple(range(6, 24)),
            recurrent_state=True,
        )
        shape, group_use_mla = worker_transfer._group_chunk_shape(
            group, layout_desc, 24, 16, {}, None
        )
        assert shape == torch.Size([18, 64, 16])
        assert group_use_mla is None


class TestBuildGroupTransferPlans:
    """The registration-time per-group plan cache.

    These constants are resolved once at ``register()`` and reused by every
    transfer, so a wrong plan silently misroutes or mis-shapes every
    subsequent store / retrieve rather than failing once.
    """

    def test_empty_groups_builds_single_collapsed_plan(self) -> None:
        kv_caches = _make_kv_caches(2)
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32]
        )
        plans = worker_transfer._build_group_transfer_plans(
            [],
            kv_caches,
            2,
            4,
            layout_desc,
            2,
            16,
            None,
            False,
            _disabled_kvweave_config(),
            KVWeaveCodec(),
        )
        assert len(plans) == 1
        assert plans[0].group_info is None
        # An empty selection means "every layer": the fallback must pass the
        # caller's mapping through untouched.
        assert plans[0].layer_indices == frozenset()
        assert plans[0].select_kv_caches(kv_caches) is kv_caches
        assert plans[0].blocks_per_chunk == 2
        assert plans[0].chunk_shape == torch.Size([2, 2, 8, 16])

    def test_builds_one_plan_per_group_in_order(self) -> None:
        kv_caches = _make_kv_caches(4)
        groups = [
            EngineGroupInfo(
                engine_group_id=0, layer_indices=(0, 1), tokens_per_block=4
            ),
            EngineGroupInfo(
                engine_group_id=1, layer_indices=(2, 3), tokens_per_block=1
            ),
        ]
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([2, 4, 8, 16])], dtypes=[torch.float32]
        )
        plans = worker_transfer._build_group_transfer_plans(
            groups,
            kv_caches,
            2,
            4,
            layout_desc,
            4,
            16,
            None,
            False,
            _disabled_kvweave_config(),
            KVWeaveCodec(),
        )
        assert len(plans) == 2
        assert list(plans[0].select_kv_caches(kv_caches).keys()) == [
            "layer_0",
            "layer_1",
        ]
        assert plans[0].blocks_per_chunk == 2
        # Group 1 has tokens_per_block=1, so the same 8-token chunk needs 8
        # blocks instead of 2.
        assert list(plans[1].select_kv_caches(kv_caches).keys()) == [
            "layer_2",
            "layer_3",
        ]
        assert plans[1].blocks_per_chunk == 8
        # Each group holds 2 of the 4 registered layers.
        assert plans[0].chunk_shape == torch.Size([2, 2, 8, 16])
        assert plans[1].chunk_shape == torch.Size([2, 2, 8, 16])

    def test_caches_a_detected_kv_format_per_group(self) -> None:
        """The cached format is what gather / scatter receive instead of
        re-detecting per transfer."""
        kv_caches = _make_kv_caches(4)
        groups = _two_groups()
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([2, 4, 8, 16])], dtypes=[torch.float32]
        )
        plans = worker_transfer._build_group_transfer_plans(
            groups,
            kv_caches,
            2,
            4,
            layout_desc,
            4,
            16,
            None,
            False,
            _disabled_kvweave_config(),
            KVWeaveCodec(),
        )
        assert [p.engine_kv_format for p in plans] == [
            worker_transfer._detect_group_kv_format(p.select_kv_caches(kv_caches), None)
            for p in plans
        ]
        assert all(p.engine_kv_format is not None for p in plans)

    def test_classifies_attention_plane_layout_from_detected_format(self) -> None:
        group = EngineGroupInfo(
            engine_group_id=0, layer_indices=(0, 1), cache_category="attention"
        )

        assert (
            worker_transfer._attention_plane_layout_for_group(
                group, lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
            )
            == AttentionPlaneLayout.SPLIT_KV
        )
        assert (
            worker_transfer._attention_plane_layout_for_group(
                group, lmcache_native.EngineKVFormat.NL_X_NB_BS_NH_TWO_HS
            )
            == AttentionPlaneLayout.FUSED_KV
        )
        assert (
            worker_transfer._attention_plane_layout_for_group(
                group, lmcache_native.EngineKVFormat.NL_X_NB_BS_HS
            )
            == AttentionPlaneLayout.MLA
        )

    def test_undetectable_group_falls_back_to_per_transfer_detection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A group whose format cannot be pre-resolved caches ``None``, which
        makes gather / scatter detect it per transfer as before."""

        def _boom(*_args: object, **_kwargs: object) -> object:
            raise ValueError("unsupported kv_caches structure")

        monkeypatch.setattr(
            "lmcache.v1.gpu_connector.utils.normalize_kv_and_discover_format", _boom
        )
        assert worker_transfer._detect_group_kv_format(_make_kv_caches(2), None) is None

    def test_misaligned_tokens_per_block_raises(self) -> None:
        kv_caches = _make_kv_caches(2)
        groups = [
            EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1), tokens_per_block=5)
        ]
        layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32]
        )
        with pytest.raises(ValueError, match="must be a multiple of"):
            worker_transfer._build_group_transfer_plans(
                groups,
                kv_caches,
                4,
                16,
                layout_desc,
                2,
                16,
                None,
                False,
                _disabled_kvweave_config(),
                KVWeaveCodec(),
            )


def _mamba_layouts():
    # First Party
    from lmcache.v1.multiprocess.group_view import MambaSubStateWireLayout

    return (
        MambaSubStateWireLayout(0, 16, "torch.float32", (2, 2)),
        MambaSubStateWireLayout(16, 48, "torch.float32", (3, 4)),
    )


def _enabled_kvweave_config(**overrides: object) -> KVWeaveRuntimeConfig:
    kwargs = {
        "enabled": True,
        "linear_quant_enabled": True,
        "linear_max_size_ratio": 1.20,
    }
    kwargs.update(overrides)
    return KVWeaveRuntimeConfig(**kwargs)


class TestDecideGroupQuantization:
    """The per-group quantization decision made once at register() time.

    A wrong decision here either silently ships unquantized chunks (no
    savings) or corrupts data (see MIGRATION_PLAN.md R1/R6): codec dispatch
    must be driven only by ``cache_category``, never tensor shape.
    """

    def test_disabled_config_never_quantizes(self) -> None:
        group = EngineGroupInfo(
            engine_group_id=0, layer_indices=(0, 1), cache_category="attention"
        )
        quantized, quant_layout, mamba_options = (
            worker_transfer._decide_group_quantization(
                group,
                torch.Size([2, 2, 8, 16]),
                torch.float32,
                4,
                AttentionPlaneLayout.SPLIT_KV,
                _disabled_kvweave_config(),
                KVWeaveCodec(),
            )
        )
        assert quantized is False
        assert quant_layout is None
        assert mamba_options is None

    def test_unknown_category_never_quantizes(self) -> None:
        """A group whose category was never resolved must never be
        quantized, regardless of what its tensor shape looks like."""
        group = EngineGroupInfo(
            engine_group_id=0, layer_indices=(0, 1), cache_category="unknown"
        )
        quantized, _, _ = worker_transfer._decide_group_quantization(
            group,
            torch.Size([2, 2, 8, 16]),
            torch.float32,
            4,
            None,
            _enabled_kvweave_config(),
            KVWeaveCodec(),
        )
        assert quantized is False

    def test_mla_attention_group_never_quantizes(self) -> None:
        """MLA formats are excluded from the attention quantization branch."""
        group = EngineGroupInfo(
            engine_group_id=0, layer_indices=(0, 1), cache_category="attention"
        )
        quantized, _, _ = worker_transfer._decide_group_quantization(
            group,
            torch.Size([2, 8, 16]),
            torch.float32,
            4,
            AttentionPlaneLayout.MLA,
            _enabled_kvweave_config(),
            KVWeaveCodec(),
        )
        assert quantized is False

    def test_attention_group_quantizes_when_estimate_is_smaller(self) -> None:
        """A large-enough attention chunk's 4-bit estimate must beat the
        fp32 raw size, and the resulting quant_layout_desc is a uint8 byte
        count, not the original shape."""
        group = EngineGroupInfo(
            engine_group_id=0, layer_indices=(0, 1), cache_category="attention"
        )
        # Large enough that per-channel scale overhead is amortized.
        raw_shape = torch.Size([2, 2, 4096, 16])
        codec = KVWeaveCodec({"num_kv_heads": 2, "head_dim": 8})
        quantized, quant_layout, mamba_options = (
            worker_transfer._decide_group_quantization(
                group,
                raw_shape,
                torch.float32,
                4,
                AttentionPlaneLayout.SPLIT_KV,
                _enabled_kvweave_config(),
                codec,
            )
        )
        assert quantized is True
        assert mamba_options is None
        assert quant_layout is not None
        assert quant_layout.dtypes[0] == torch.uint8
        raw_size = 2 * 2 * 4096 * 16 * 4
        assert quant_layout.shapes[0][0] < raw_size

    def test_fused_attention_group_quantizes_when_estimate_is_smaller(self) -> None:
        group = EngineGroupInfo(
            engine_group_id=0, layer_indices=(0, 1), cache_category="attention"
        )
        raw_shape = torch.Size([2, 4096, 32])
        codec = KVWeaveCodec({"num_kv_heads": 2, "head_dim": 8})
        quantized, quant_layout, mamba_options = (
            worker_transfer._decide_group_quantization(
                group,
                raw_shape,
                torch.float32,
                4,
                AttentionPlaneLayout.FUSED_KV,
                _enabled_kvweave_config(),
                codec,
            )
        )

        assert quantized is True
        assert mamba_options is None
        assert quant_layout is not None
        assert quant_layout.dtypes[0] == torch.uint8
        assert quant_layout.shapes[0][0] < raw_shape.numel() * 4

    @pytest.mark.parametrize(
        ("layout", "config_override"),
        [
            (AttentionPlaneLayout.SPLIT_KV, "split_attention_quant_enabled"),
            (AttentionPlaneLayout.FUSED_KV, "fused_attention_quant_enabled"),
        ],
    )
    def test_attention_layout_debug_switch_disables_only_selected_layout(
        self, layout: AttentionPlaneLayout, config_override: str
    ) -> None:
        group = EngineGroupInfo(
            engine_group_id=0, layer_indices=(0, 1), cache_category="attention"
        )
        raw_shape = (
            torch.Size([2, 2, 4096, 16])
            if layout == AttentionPlaneLayout.SPLIT_KV
            else torch.Size([2, 4096, 32])
        )
        config = _enabled_kvweave_config(**{config_override: False})

        quantized, quant_layout, _ = worker_transfer._decide_group_quantization(
            group,
            raw_shape,
            torch.float32,
            4,
            layout,
            config,
            KVWeaveCodec({"num_kv_heads": 2, "head_dim": 8}),
        )

        assert quantized is False
        assert quant_layout is None

    def test_mamba_group_without_real_layout_falls_back_unquantized(self) -> None:
        """A Mamba group missing mamba_real_layout must safely fall back to
        unquantized transfer, not raise (MIGRATION_PLAN.md Phase D item 2)."""
        group = EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0, 1),
            cache_category="mamba",
            mamba_real_layout=None,
        )
        quantized, quant_layout, mamba_options = (
            worker_transfer._decide_group_quantization(
                group,
                torch.Size([2, 2, 8, 16]),
                torch.float32,
                4,
                None,
                _enabled_kvweave_config(),
                KVWeaveCodec(),
            )
        )
        assert quantized is False
        assert quant_layout is None
        assert mamba_options is None

    def test_mamba_group_disabled_by_linear_quant_enabled_flag(self) -> None:
        """``linear_quant_enabled=False`` disables Mamba quantization
        independently of the overall ``enabled`` switch."""
        group = EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0, 1),
            cache_category="mamba",
            mamba_real_layout=_mamba_layouts(),
        )
        quantized, _, _ = worker_transfer._decide_group_quantization(
            group,
            torch.Size([2, 2, 8, 16]),
            torch.float32,
            4,
            None,
            _enabled_kvweave_config(linear_quant_enabled=False),
            KVWeaveCodec(),
        )
        assert quantized is False

    def test_none_group_info_never_quantizes(self) -> None:
        """The single-group (no cache_category) fallback predates the
        cache_category field and must never be quantized."""
        quantized, _, _ = worker_transfer._decide_group_quantization(
            None,
            torch.Size([2, 2, 8, 16]),
            torch.float32,
            4,
            None,
            _enabled_kvweave_config(),
            KVWeaveCodec(),
        )
        assert quantized is False


class TestIterTransferGroups:
    """Pairing the cached plans with a request's block IDs."""

    def _single_group_ctx(
        self, monkeypatch: pytest.MonkeyPatch, kv_caches: dict[str, torch.Tensor]
    ) -> EngineDrivenTransferContext:
        monkeypatch.setattr(
            worker_transfer,
            "create_engine_driven_context",
            lambda *a, **k: _FakeEngineDrivenContext(),
        )
        future = MagicMock()
        future.result.return_value = RegisterEngineDrivenContextResponse()
        req_client = MagicMock()
        req_client.register_kv_cache_engine_driven_context.return_value = future
        ctx = EngineDrivenTransferContext()
        ctx.register(
            instance_id=1,
            kv_caches=kv_caches,
            model_name="m",
            world_size=1,
            blocks_in_chunk=2,
            req_client=req_client,
            mq_timeout=1.0,
        )
        return ctx

    def test_without_cached_plans_derives_single_group_fallback(self) -> None:
        """A context wired up without register() (as some transport tests do)
        must still transfer via the single-group fallback."""
        ctx = EngineDrivenTransferContext()
        ctx._engine_driven_context = _FakeEngineDrivenContext()  # type: ignore[assignment]
        kv_caches = _make_kv_caches(2)
        result = list(ctx.iter_transfer_groups(kv_caches, [[0, 1]], 2))
        assert len(result) == 1
        plan, group_kv_caches, group_block_ids = result[0]
        assert plan.group_info is None
        assert group_kv_caches is kv_caches
        assert plan.blocks_per_chunk == 2
        assert group_block_ids == [0, 1]

    def test_empty_groups_yields_single_group_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kv_caches = _make_kv_caches(2, num_blocks=8)
        ctx = self._single_group_ctx(monkeypatch, kv_caches)
        result = list(ctx.iter_transfer_groups(kv_caches, [[0, 1]], 2))
        assert len(result) == 1
        plan, group_kv_caches, group_block_ids = result[0]
        assert plan.group_info is None
        assert group_kv_caches is kv_caches
        assert plan.blocks_per_chunk == 2
        assert group_block_ids == [0, 1]

    def test_empty_groups_rejects_multi_group_block_ids(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kv_caches = _make_kv_caches(2, num_blocks=8)
        ctx = self._single_group_ctx(monkeypatch, kv_caches)
        with pytest.raises(RuntimeError, match="does not support hybrid"):
            list(ctx.iter_transfer_groups(kv_caches, [[0], [1]], 2))

    def test_yields_one_entry_per_group_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kv_caches = _make_kv_caches(4, num_blocks=8)
        ctx = _register_context(monkeypatch, _FakeEngineDrivenContext(), kv_caches)
        result = list(ctx.iter_transfer_groups(kv_caches, [[0, 1], [4, 5]], 2))
        assert len(result) == 2
        assert list(result[0][1].keys()) == ["layer_0", "layer_1"]
        assert result[0][2] == [0, 1]
        assert list(result[1][1].keys()) == ["layer_2", "layer_3"]
        assert result[1][2] == [4, 5]

    def test_subsets_come_from_the_caller_not_registration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retrieve scatters into destination tensors, not the registered
        ones, so the yielded subsets must alias the caller's mapping."""
        kv_caches = _make_kv_caches(4, num_blocks=8)
        ctx = _register_context(monkeypatch, _FakeEngineDrivenContext(), kv_caches)
        destination = {
            name: torch.zeros_like(tensor) for name, tensor in kv_caches.items()
        }
        result = list(ctx.iter_transfer_groups(destination, [[0, 1], [4, 5]], 2))
        assert result[0][1]["layer_0"] is destination["layer_0"]
        assert result[1][1]["layer_2"] is destination["layer_2"]

    def test_rejects_block_id_group_count_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kv_caches = _make_kv_caches(4, num_blocks=8)
        ctx = _register_context(monkeypatch, _FakeEngineDrivenContext(), kv_caches)
        with pytest.raises(ValueError, match="Expected 2 block-id groups"):
            list(ctx.iter_transfer_groups(kv_caches, [[0, 1]], 2))


class _FakeEngineDrivenContext:
    """Minimal engine-driven context for multi-group submit_store/retrieve
    tests: pickle transport, always signals new chunks to gather."""

    def __init__(self) -> None:
        self.layout_desc = MemoryLayoutDesc(
            shapes=[torch.Size([2, 2, 8, 16])], dtypes=[torch.float32]
        )
        self.committed_chunks: list[torch.Tensor] | None = None
        self.retrieve_chunks: list[torch.Tensor] | None = None

    def prepare_store(self, _key: object, _instance_id: int) -> None:
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


def _two_groups() -> list[EngineGroupInfo]:
    return [
        EngineGroupInfo(engine_group_id=0, layer_indices=(0, 1), tokens_per_block=4),
        EngineGroupInfo(engine_group_id=1, layer_indices=(2, 3), tokens_per_block=4),
    ]


def _register_context(
    monkeypatch: pytest.MonkeyPatch,
    fake_context: _FakeEngineDrivenContext,
    kv_caches: dict[str, torch.Tensor],
    shm_name: str = "",
    pool_size: int = 0,
) -> EngineDrivenTransferContext:
    """Register an EngineDrivenTransferContext with two same-shape groups
    (2 layers each) so gather/scatter round-trips are easy to assert on."""
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse(
        shm_name=shm_name, pool_size=pool_size
    )
    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.return_value = future

    ctx = EngineDrivenTransferContext()
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        req_client=req_client,
        mq_timeout=1.0,
        engine_group_infos=_two_groups(),
    )
    return ctx


def test_register_forwards_engine_group_infos_to_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registration payload must carry the group metadata so the server
    can build one layout per group."""
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        lambda *a, **k: _FakeEngineDrivenContext(),
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.return_value = future

    EngineDrivenTransferContext().register(
        instance_id=1,
        kv_caches=_make_kv_caches(4, num_blocks=8),
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        req_client=req_client,
        mq_timeout=1.0,
        engine_group_infos=_two_groups(),
    )

    payload = req_client.register_kv_cache_engine_driven_context.call_args.args[0]
    assert payload.engine_group_infos == _two_groups()


def test_register_without_groups_sends_empty_group_infos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-hybrid registration must still send an empty list, which the
    server reads as the single-group fallback."""
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        lambda *a, **k: _FakeEngineDrivenContext(),
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.return_value = future

    EngineDrivenTransferContext().register(
        instance_id=1,
        kv_caches=_make_kv_caches(4, num_blocks=8),
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        req_client=req_client,
        mq_timeout=1.0,
    )

    payload = req_client.register_kv_cache_engine_driven_context.call_args.args[0]
    assert payload.engine_group_infos == []


def test_submit_store_concatenates_chunks_group_major(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two same-shape groups each contribute one chunk; commit_store must
    receive both, group 0's chunk first."""
    fake_context = _FakeEngineDrivenContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    ctx = _register_context(monkeypatch, fake_context, kv_caches)

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


def _recurrent_and_attention_groups() -> list[EngineGroupInfo]:
    """Group 0 is align-mode recurrent state; group 1 is full attention."""
    return [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0, 1),
            tokens_per_block=4,
            recurrent_state=True,
        ),
        EngineGroupInfo(engine_group_id=1, layer_indices=(2, 3), tokens_per_block=4),
    ]


def test_submit_store_attaches_null_chunk_mask_for_recurrent_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recurrent group's null (block id 0) chunks must be marked in the
    key sent to prepare_store/commit_store, and non-null chunks left alone.
    """

    class _KeyCapturingContext(_FakeEngineDrivenContext):
        def __init__(self) -> None:
            super().__init__()
            self.prepare_store_key: object = None
            self.commit_store_key: object = None

        def prepare_store(self, key: object, _instance_id: int) -> None:
            self.prepare_store_key = key
            return None

        def commit_store(
            self, key: object, instance_id: int, chunks: list[torch.Tensor]
        ) -> bool:
            self.commit_store_key = key
            return super().commit_store(key, instance_id, chunks)

    fake_context = _KeyCapturingContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.return_value = future
    ctx = EngineDrivenTransferContext()
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        req_client=req_client,
        mq_timeout=1.0,
        engine_group_infos=_recurrent_and_attention_groups(),
    )

    key = IPCCacheServerKey.from_token_ids(
        "m", 1, 0, list(range(16)), start=0, end=16, request_id="req"
    )
    # Group 0 (recurrent): chunk 0 is all null (both blocks 0), chunk 1 has
    # the one live snapshot (block 7). Group 1 (attention): both chunks real.
    future_result = ctx.submit_store(
        "req", key, 1, kv_caches, [[0, 0, 0, 7], [4, 5, 6, 7]], MagicMock(), 2
    )

    assert future_result.result() is True
    sent_mask = fake_context.prepare_store_key.null_chunk_mask
    # Group 1 (full attention: block ids never null) computes an all-False
    # mask -- every group's chunks are checked the same, unconditional way.
    assert sent_mask == ((True, False), (False, False))
    assert fake_context.commit_store_key.null_chunk_mask == sent_mask
    # Original key's identity fields must be untouched by the copy.
    assert fake_context.prepare_store_key.request_id == "req"
    assert fake_context.prepare_store_key.token_ids == tuple(range(16))
    # The fake server context ignores chunk_indices (ok=None -> gather
    # everything); the real server excludes masked chunks from
    # chunk_indices instead, so the worker never gathers them for a real
    # server (see test_server_prepare_store_excludes_null_masked_chunk).
    assert fake_context.committed_chunks is not None
    assert len(fake_context.committed_chunks) == 4


def test_submit_store_omits_null_chunk_mask_without_recurrent_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No recurrent group registered means the key is sent unmodified."""
    fake_context = _FakeEngineDrivenContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    ctx = _register_context(monkeypatch, fake_context, kv_caches)
    key = IPCCacheServerKey.from_token_ids(
        "m", 1, 0, list(range(16)), start=0, end=16, request_id="req"
    )

    future = ctx.submit_store(
        "req", key, 1, kv_caches, [[0, 1], [4, 5]], MagicMock(), blocks_in_chunk=2
    )

    assert future.result() is True
    assert key.null_chunk_mask is None


def test_submit_store_attaches_null_chunk_mask_for_sliding_window_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine sliding-window attention group (non-recurrent) must gate
    the mask computation too -- vLLM nulls its out-of-window prefix the same
    way it does a recurrent group's (see ``SlidingWindowManager``)."""

    class _KeyCapturingContext(_FakeEngineDrivenContext):
        def __init__(self) -> None:
            super().__init__()
            self.prepare_store_key: object = None

        def prepare_store(self, key: object, _instance_id: int) -> None:
            self.prepare_store_key = key
            return None

    fake_context = _KeyCapturingContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.return_value = future
    ctx = EngineDrivenTransferContext()
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        req_client=req_client,
        mq_timeout=1.0,
        engine_group_infos=[
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0, 1),
                tokens_per_block=4,
                sw_size_tokens=8,
            ),
            EngineGroupInfo(
                engine_group_id=1, layer_indices=(2, 3), tokens_per_block=4
            ),
        ],
    )
    key = IPCCacheServerKey.from_token_ids(
        "m", 1, 0, list(range(16)), start=0, end=16, request_id="req"
    )
    # Group 0 (sliding window): chunk 0 nulled by vLLM (both blocks 0), chunk
    # 1 in-window (block 7). Group 1 (full attention): both chunks real.
    future_result = ctx.submit_store(
        "req", key, 1, kv_caches, [[0, 0, 0, 7], [4, 5, 6, 7]], MagicMock(), 2
    )

    assert future_result.result() is True
    assert fake_context.prepare_store_key.null_chunk_mask == (
        (True, False),
        (False, False),
    )


class TestNullChunkMaskFromGroups:
    """Unit coverage for the pure mask-computation helper, isolated from the
    submit_store plumbing above."""

    def test_marks_all_null_chunks_for_recurrent_group(self) -> None:
        plan = worker_transfer.GroupTransferPlan(
            group_info=EngineGroupInfo(
                engine_group_id=0, tokens_per_block=4, recurrent_state=True
            ),
            layer_indices=frozenset(),
            blocks_per_chunk=2,
            chunk_shape=torch.Size([1]),
            engine_kv_format=None,
        )
        # 3 chunks: null, null, live (block 9 in the last position).
        transfer_groups = [(plan, {}, [0, 0, 0, 0, 0, 9])]

        assert null_chunk_mask_from_groups(transfer_groups) == ((True, True, False),)

    def test_marks_all_null_chunks_for_sliding_window_attention_group(self) -> None:
        """Genuine sliding-window attention groups (non-recurrent) also null
        out their out-of-window prefix (see ``SlidingWindowManager``'s
        ``remove_skipped_blocks``), so they must be masked the same way as
        recurrent groups."""
        plan = worker_transfer.GroupTransferPlan(
            group_info=EngineGroupInfo(
                engine_group_id=0, tokens_per_block=4, sw_size_tokens=8
            ),
            layer_indices=frozenset(),
            blocks_per_chunk=2,
            chunk_shape=torch.Size([1]),
            engine_kv_format=None,
        )
        # 3 chunks: null, null, live (block 9 in the last position) -- same
        # shape as a recurrent group's mask.
        transfer_groups = [(plan, {}, [0, 0, 0, 0, 0, 9])]

        assert null_chunk_mask_from_groups(transfer_groups) == ((True, True, False),)

    def test_full_attention_group_contributes_all_false_mask(self) -> None:
        """A full-attention group's blocks are never nulled by vLLM, so every
        chunk position resolves to False -- distinct from ``None``/empty,
        which mean 'not sent'/'not covered', not 'checked and clean'."""
        plan = worker_transfer.GroupTransferPlan(
            group_info=EngineGroupInfo(engine_group_id=0, tokens_per_block=4),
            layer_indices=frozenset(),
            blocks_per_chunk=2,
            chunk_shape=torch.Size([1]),
            engine_kv_format=None,
        )
        transfer_groups = [(plan, {}, [1, 2, 3, 4])]

        assert null_chunk_mask_from_groups(transfer_groups) == ((False, False),)

    def test_single_group_fallback_computes_from_block_ids_alone(self) -> None:
        """group_info=None (single-group fallback) must not crash; the mask
        is still computed from the raw block ids, same as any other group."""
        plan = worker_transfer.GroupTransferPlan(
            group_info=None,
            layer_indices=frozenset(),
            blocks_per_chunk=2,
            chunk_shape=torch.Size([1]),
            engine_kv_format=None,
        )
        transfer_groups = [(plan, {}, [1, 2, 3, 4])]

        assert null_chunk_mask_from_groups(transfer_groups) == ((False, False),)


def test_submit_retrieve_scatters_chunks_group_major(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two groups' chunks (already gathered by a prior store) scatter back
    into the correct group's layers only."""
    fake_context = _FakeEngineDrivenContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    ctx = _register_context(monkeypatch, fake_context, kv_caches)

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
    # Layers of the other group must not have been written by this group's
    # scatter (block 2/3 of group 1's layers stay zero).
    assert torch.all(destination["layer_2"][:, 2] == 0)


def test_submit_retrieve_attaches_null_chunk_mask_for_recurrent_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recurrent group's null (block id 0) chunks must be marked in the
    key sent to prepare_retrieve, mirroring submit_store, so the server never
    looks up keys for chunks that were never stored.
    """

    class _KeyCapturingContext(_FakeEngineDrivenContext):
        def __init__(self) -> None:
            super().__init__()
            self.prepare_retrieve_key: object = None

        def prepare_retrieve(self, key: object, _instance_id: int):
            self.prepare_retrieve_key = key
            return self.retrieve_chunks

    fake_context = _KeyCapturingContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.return_value = future
    ctx = EngineDrivenTransferContext()
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        req_client=req_client,
        mq_timeout=1.0,
        engine_group_infos=_recurrent_and_attention_groups(),
    )

    key = IPCCacheServerKey.from_token_ids(
        "m", 1, 0, list(range(16)), start=0, end=16, request_id="req"
    )
    # Group 0 (recurrent): chunk 0 is all null (both blocks 0), chunk 1 has
    # the one live snapshot (block 7). Group 1 (attention): both chunks real.
    # The server only ever gathers the live chunk (1 recurrent + 2 attention
    # = 3 chunks), matching a masked prepare_store.
    fake_context.retrieve_chunks = [torch.zeros(2, 2, 8, 16) for _ in range(3)]
    destination = {name: torch.zeros_like(tensor) for name, tensor in kv_caches.items()}

    future_result = ctx.submit_retrieve(
        "req", key, 1, destination, [[0, 0, 0, 7], [4, 5, 6, 7]], MagicMock(), 2
    )

    assert future_result.result() is True
    sent_mask = fake_context.prepare_retrieve_key.null_chunk_mask
    assert sent_mask == ((True, False), (False, False))
    # Original key's identity fields must be untouched by the copy.
    assert fake_context.prepare_retrieve_key.request_id == "req"
    assert fake_context.prepare_retrieve_key.token_ids == tuple(range(16))


def test_submit_retrieve_skips_null_chunk_when_scattering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker must scatter the server's shortened (null-chunk-excluded)
    chunk list into only the recurrent group's live block, not its null
    blocks -- a length mismatch here previously masked itself as a
    server-side cache miss (result=False) on every masked retrieve."""
    fake_context = _FakeEngineDrivenContext()
    kv_caches = _make_kv_caches(4, num_blocks=10, block_size=4, num_heads=2, head_size=8)
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.return_value = future
    ctx = EngineDrivenTransferContext()
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        req_client=req_client,
        mq_timeout=1.0,
        engine_group_infos=_recurrent_and_attention_groups(),
    )

    # Group 0 (recurrent, 3 chunks of 2 blocks): chunks 0-1 are all-null
    # (block id 0) and excluded server-side; chunk 2 holds the one live
    # snapshot (block 9). Group 1 (attention, 1 chunk) is never masked. The
    # server's flat, group-major result is therefore 2 chunks (1 live
    # recurrent + 1 attention), not the 4 a naive (unmasked) group_offset
    # walk would expect.
    live_group0_chunk = torch.full((2, 2, 8, 16), 7.0)
    group1_chunk = torch.full((2, 2, 8, 16), 1.0)
    fake_context.retrieve_chunks = [live_group0_chunk, group1_chunk]
    destination = {name: torch.zeros_like(tensor) for name, tensor in kv_caches.items()}
    key = IPCCacheServerKey.from_token_ids(
        "m", 1, 0, list(range(16)), start=0, end=16, request_id="req"
    )

    retrieve_future = ctx.submit_retrieve(
        "req", key, 1, destination, [[0, 0, 0, 0, 0, 9], [4, 5]], MagicMock(), 2
    )

    assert retrieve_future.result() is True
    # Group 0's all-null chunks (0-1, block ids [0,0,0,0]) are excluded
    # server-side, so the scatter must only consume the 1 live chunk the
    # server actually returned and land it on chunk 2's blocks (the null
    # sentinel block 0 and the live block 9). Block 0 is a shared sentinel
    # (vLLM's ``null_block``) whose contents are never read by any request,
    # so it being overwritten here is immaterial.
    assert not torch.all(destination["layer_0"][:, 9] == 0)
    # Group 1 (attention) scatters its chunk normally into blocks 4-5,
    # unaffected by group 0's mask.
    assert not torch.all(destination["layer_2"][:, 4] == 0)
    assert not torch.all(destination["layer_2"][:, 5] == 0)


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
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    ctx = _register_context(
        monkeypatch, fake_context, kv_caches, shm_name="pool", pool_size=4096
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
    # Each flat slot must exactly equal what a direct, non-SHM gather over
    # that group's own layers/block-ids would produce.
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


def test_submit_store_skips_group_with_no_selected_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When chunk_indices selects only group 1's chunk, group 0 must be
    skipped entirely and group 1's gather must land in the single buffer."""

    class _PartialShmFakeContext(_FakeEngineDrivenContext):
        def __init__(self) -> None:
            super().__init__()
            self.out_buffers = [torch.zeros(2, 2, 8, 16)]

        def prepare_store(self, _key: object, _instance_id: int):
            # Flat chunk index 1 == group 1's only chunk (group 0 cached).
            return self.out_buffers, [1]

    fake_context = _PartialShmFakeContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    ctx = _register_context(
        monkeypatch, fake_context, kv_caches, shm_name="pool", pool_size=4096
    )

    result = ctx.submit_store(
        "req", MagicMock(), 1, kv_caches, [[0, 1], [4, 5]], MagicMock(), 2
    )

    assert result.result() is True
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu

    expected_group1 = gather_paged_kv_to_cpu(
        {"layer_2": kv_caches["layer_2"], "layer_3": kv_caches["layer_3"]}, [4, 5], 2
    )[0]
    assert torch.allclose(fake_context.out_buffers[0], expected_group1)


class _SpyCodec:
    """Records every encode_chunk/decode_chunk call's chunk count, so tests
    can assert quantization only ran on the selection actually passed in
    (MIGRATION_PLAN.md R2/V3), without depending on native quantization."""

    def __init__(self) -> None:
        self.encode_calls: list[torch.Tensor] = []
        self.decode_calls: list[torch.Tensor] = []
        self.encode_layouts: list[AttentionPlaneLayout | None] = []
        self.decode_layouts: list[AttentionPlaneLayout | None] = []

    def encode_chunk(
        self,
        cache_category,
        mamba_layout,
        tokens_per_block,
        mamba_options,
        raw_chunk,
        attention_plane_layout=None,
    ) -> bytes:
        self.encode_calls.append(raw_chunk)
        self.encode_layouts.append(attention_plane_layout)
        return raw_chunk.numpy().tobytes()

    def decode_chunk(
        self,
        cache_category,
        mamba_layout,
        tokens_per_block,
        raw_shape,
        raw_dtype,
        chunk,
        attention_plane_layout=None,
    ) -> torch.Tensor:
        self.decode_calls.append(chunk)
        self.decode_layouts.append(attention_plane_layout)
        flat = torch.frombuffer(bytearray(chunk.numpy().tobytes()), dtype=raw_dtype)
        return flat.view(raw_shape)


def test_submit_store_quantizes_only_selected_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quantized group's server-filtered-out chunks (already cached, or
    null-masked -- SHM mode's ``chunk_indices`` excludes both the same way)
    must never reach encode_chunk: the quantization loop must reuse the
    same ``selection`` the gather loop used, not re-enumerate chunks
    (MIGRATION_PLAN.md R2/V3)."""

    # A quantized group's real SHM slot is a flat uint8 buffer sized to
    # quant_layout_desc (see _decide_group_quantization), not the raw KV
    # dtype/shape -- large enough here for _SpyCodec's 1:1 byte encoding.
    quant_slot = torch.zeros(2048, dtype=torch.uint8)

    class _ShmFakeContext(_FakeEngineDrivenContext):
        def __init__(self) -> None:
            super().__init__()
            # Only flat chunk index 1 (group 0's second chunk) is selected;
            # group 0's first chunk and group 1's chunk are already cached.
            self.out_buffers = [quant_slot]

        def prepare_store(self, _key: object, _instance_id: int):
            return self.out_buffers, [1]

    fake_context = _ShmFakeContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    ctx = _register_context(
        monkeypatch, fake_context, kv_caches, shm_name="pool", pool_size=4096
    )

    spy_codec = _SpyCodec()
    ctx._kvweave_codec = spy_codec  # noqa: SLF001
    # Group 0: quantized. Group 1: unquantized.
    quantized_group0 = worker_transfer.GroupTransferPlan(
        group_info=EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0, 1),
            tokens_per_block=4,
            cache_category="mamba",
        ),
        layer_indices=frozenset({0, 1}),
        blocks_per_chunk=ctx._group_plans[0].blocks_per_chunk,  # noqa: SLF001
        chunk_shape=ctx._group_plans[0].chunk_shape,  # noqa: SLF001
        engine_kv_format=ctx._group_plans[0].engine_kv_format,  # noqa: SLF001
        quantized=True,
        raw_layout_desc=MemoryLayoutDesc(
            shapes=[ctx._group_plans[0].chunk_shape],  # noqa: SLF001
            dtypes=[torch.float32],
        ),
        quant_layout_desc=MemoryLayoutDesc(
            shapes=[torch.Size([2048])], dtypes=[torch.uint8]
        ),
    )
    ctx._group_plans[0] = quantized_group0  # noqa: SLF001

    # Group 0 has 2 chunks ([0,1] and [4,5]); only flat index 1 (its second
    # chunk) is selected. Group 1 is not quantized, so it is irrelevant to
    # the selected chunk_indices=[1] here (index 1 falls in group 0's range).
    result = ctx.submit_store(
        "req", MagicMock(), 1, kv_caches, [[0, 1, 4, 5], [8, 9]], MagicMock(), 2
    )

    assert result.result() is True
    # Only the one selected chunk was encoded.
    assert len(spy_codec.encode_calls) == 1
    # SHM mode: EngineDrivenContextShm.commit_store never transmits chunks
    # (data must already be in the slot), so the encoded bytes must have
    # been copy_'d into the group's real SHM slot instead of committed --
    # the quantized group's chunk never reaches the (empty) commit list.
    assert fake_context.committed_chunks == []
    expected = torch.frombuffer(
        bytearray(spy_codec.encode_calls[0].numpy().tobytes()), dtype=torch.uint8
    )
    assert torch.equal(quant_slot[: expected.numel()], expected)


def test_submit_retrieve_decodes_only_live_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quantized group's decode must run on exactly the live (non-null)
    chunks the server returned, symmetric with the store-side selection."""
    fake_context = _FakeEngineDrivenContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    ctx = _register_context(monkeypatch, fake_context, kv_caches)

    spy_codec = _SpyCodec()
    ctx._kvweave_codec = spy_codec  # noqa: SLF001
    raw_shape = ctx._group_plans[0].chunk_shape  # noqa: SLF001
    quantized_group0 = worker_transfer.GroupTransferPlan(
        group_info=EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0, 1),
            tokens_per_block=4,
            recurrent_state=True,
            cache_category="mamba",
        ),
        layer_indices=frozenset({0, 1}),
        blocks_per_chunk=ctx._group_plans[0].blocks_per_chunk,  # noqa: SLF001
        chunk_shape=raw_shape,
        engine_kv_format=ctx._group_plans[0].engine_kv_format,  # noqa: SLF001
        quantized=True,
        raw_layout_desc=MemoryLayoutDesc(shapes=[raw_shape], dtypes=[torch.float32]),
    )
    ctx._group_plans[0] = quantized_group0  # noqa: SLF001

    key = IPCCacheServerKey.from_token_ids(
        "m", 1, 0, list(range(16)), start=0, end=16, request_id="req"
    )
    # Group 0 (recurrent): one live chunk only (server already dropped the
    # null one from src_buffers). Group 1: both chunks real.
    live_encoded_chunk = torch.zeros(raw_shape, dtype=torch.float32).view(torch.uint8)
    fake_context.retrieve_chunks = [
        live_encoded_chunk,
        torch.zeros(2, 2, 8, 16),
        torch.zeros(2, 2, 8, 16),
    ]

    result = ctx.submit_retrieve(
        "req", key, 1, kv_caches, [[0, 0, 0, 7], [4, 5, 6, 7]], MagicMock(), 2
    )

    assert result.result() is True
    assert len(spy_codec.decode_calls) == 1
    assert spy_codec.decode_calls[0] is live_encoded_chunk


class _FakeAsyncEvent:
    """Device event stub for the async store path (no real stream)."""

    def record(self, stream: object | None = None) -> None:
        return None

    def wait(self, stream: object | None = None) -> None:
        return None

    def synchronize(self) -> None:
        return None


class _FakeAsyncTorchDev:
    def Stream(self) -> object:
        return object()

    def stream(self, stream: object) -> object:
        return nullcontext(stream)

    def current_stream(self) -> object:
        return object()

    def Event(self, interprocess: bool = False) -> _FakeAsyncEvent:
        return _FakeAsyncEvent()

    def synchronize(self) -> None:
        return None


def _new_async_context(
    monkeypatch: pytest.MonkeyPatch,
    fake_context: _FakeEngineDrivenContext,
    kv_caches: dict[str, torch.Tensor],
) -> AsyncEngineDrivenTransferContext:
    """Build a registered async context over two same-shape groups."""
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeAsyncTorchDev())
    monkeypatch.setattr(worker_transfer, "torch_dev", _FakeAsyncTorchDev())
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.return_value = future

    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        req_client=req_client,
        mq_timeout=1.0,
        engine_group_infos=_two_groups(),
    )
    return ctx


def test_async_submit_store_gathers_every_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async store path must gather all groups group-major, matching the
    synchronous path's flat chunk ordering."""
    fake_context = _FakeEngineDrivenContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    ctx = _new_async_context(monkeypatch, fake_context, kv_caches)

    try:
        future = ctx.submit_store(
            "req", MagicMock(), 1, kv_caches, [[0, 1], [4, 5]], _FakeAsyncEvent(), 2
        )
        assert future.result(timeout=10) is True
    finally:
        ctx.close()

    assert fake_context.committed_chunks is not None
    assert len(fake_context.committed_chunks) == 2
    # First Party
    from lmcache.v1.multiprocess.transfer_context.base import gather_paged_kv_to_cpu

    expected_group0 = gather_paged_kv_to_cpu(
        {"layer_0": kv_caches["layer_0"], "layer_1": kv_caches["layer_1"]}, [0, 1], 2
    )[0]
    expected_group1 = gather_paged_kv_to_cpu(
        {"layer_2": kv_caches["layer_2"], "layer_3": kv_caches["layer_3"]}, [4, 5], 2
    )[0]
    assert torch.allclose(fake_context.committed_chunks[0], expected_group0)
    assert torch.allclose(fake_context.committed_chunks[1], expected_group1)


def test_async_submit_store_quantizes_only_selected_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async store path's quantization must reuse the exact same
    ``_encode_group_chunks`` method the sync path calls, and must only
    encode the server-selected chunks (MIGRATION_PLAN.md R2/R3)."""

    # A quantized group's real SHM slot is a flat uint8 buffer sized to
    # quant_layout_desc (see _decide_group_quantization), not the raw KV
    # dtype/shape -- large enough here for _SpyCodec's 1:1 byte encoding.
    quant_slot = torch.zeros(2048, dtype=torch.uint8)

    class _ShmFakeContext(_FakeEngineDrivenContext):
        def __init__(self) -> None:
            super().__init__()
            self.out_buffers = [quant_slot]

        def prepare_store(self, _key: object, _instance_id: int):
            return self.out_buffers, [1]

    fake_context = _ShmFakeContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeAsyncTorchDev())
    monkeypatch.setattr(worker_transfer, "torch_dev", _FakeAsyncTorchDev())
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse(
        shm_name="pool", pool_size=4096
    )
    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.return_value = future

    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        req_client=req_client,
        mq_timeout=1.0,
        engine_group_infos=_two_groups(),
    )
    spy_codec = _SpyCodec()
    ctx._kvweave_codec = spy_codec  # noqa: SLF001
    quantized_group0 = worker_transfer.GroupTransferPlan(
        group_info=EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0, 1),
            tokens_per_block=4,
            cache_category="mamba",
        ),
        layer_indices=frozenset({0, 1}),
        blocks_per_chunk=ctx._group_plans[0].blocks_per_chunk,  # noqa: SLF001
        chunk_shape=ctx._group_plans[0].chunk_shape,  # noqa: SLF001
        engine_kv_format=ctx._group_plans[0].engine_kv_format,  # noqa: SLF001
        quantized=True,
        raw_layout_desc=MemoryLayoutDesc(
            shapes=[ctx._group_plans[0].chunk_shape],  # noqa: SLF001
            dtypes=[torch.float32],
        ),
        quant_layout_desc=MemoryLayoutDesc(
            shapes=[torch.Size([2048])], dtypes=[torch.uint8]
        ),
    )
    ctx._group_plans[0] = quantized_group0  # noqa: SLF001

    try:
        future_result = ctx.submit_store(
            "req",
            MagicMock(),
            1,
            kv_caches,
            [[0, 1, 4, 5], [8, 9]],
            _FakeAsyncEvent(),
            2,
        )
        assert future_result.result(timeout=10) is True
    finally:
        ctx.close()

    assert len(spy_codec.encode_calls) == 1
    # SHM mode: EngineDrivenContextShm.commit_store never transmits chunks
    # (data must already be in the slot), so the encoded bytes must have
    # been copy_'d into the group's real SHM slot instead of committed --
    # the quantized group's chunk never reaches the (empty) commit list.
    assert fake_context.committed_chunks == []
    expected = torch.frombuffer(
        bytearray(spy_codec.encode_calls[0].numpy().tobytes()), dtype=torch.uint8
    )
    assert torch.equal(quant_slot[: expected.numel()], expected)


def test_async_submit_store_attaches_null_chunk_mask_for_recurrent_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async store path must compute and attach the null-chunk mask on
    the forward thread, the same way the synchronous path does."""

    class _KeyCapturingContext(_FakeEngineDrivenContext):
        def __init__(self) -> None:
            super().__init__()
            self.prepare_store_key: object = None

        def prepare_store(self, key: object, _instance_id: int) -> None:
            self.prepare_store_key = key
            return None

    fake_context = _KeyCapturingContext()
    kv_caches = _make_kv_caches(4, num_blocks=8, block_size=4, num_heads=2, head_size=8)
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeAsyncTorchDev())
    monkeypatch.setattr(worker_transfer, "torch_dev", _FakeAsyncTorchDev())
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: fake_context
    )
    future = MagicMock()
    future.result.return_value = RegisterEngineDrivenContextResponse()
    req_client = MagicMock()
    req_client.register_kv_cache_engine_driven_context.return_value = future

    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    ctx.register(
        instance_id=1,
        kv_caches=kv_caches,
        model_name="m",
        world_size=1,
        blocks_in_chunk=2,
        req_client=req_client,
        mq_timeout=1.0,
        engine_group_infos=_recurrent_and_attention_groups(),
    )
    key = IPCCacheServerKey.from_token_ids(
        "m", 1, 0, list(range(16)), start=0, end=16, request_id="req"
    )

    try:
        future_result = ctx.submit_store(
            "req", key, 1, kv_caches, [[0, 0, 0, 7], [4, 5, 6, 7]], _FakeAsyncEvent(), 2
        )
        assert future_result.result(timeout=10) is True
    finally:
        ctx.close()

    assert fake_context.prepare_store_key.null_chunk_mask == (
        (True, False),
        (False, False),
    )
    # The original key passed in must be left untouched (a new copy carries
    # the mask); async callers may still hold a reference to it.
    assert key.null_chunk_mask is None


def test_release_staging_buckets_mixed_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hybrid store stages one buffer set per group; groups with differing
    layer counts have differing chunk shapes, which must be filed under
    their own pool keys rather than all under the first chunk's shape."""
    monkeypatch.setattr(async_engine_driven, "torch_dev", _FakeAsyncTorchDev())
    ctx = AsyncEngineDrivenTransferContext(commit_workers=1)
    try:
        small = torch.zeros(2, 2, 8, 16)
        large = torch.zeros(2, 6, 8, 16)
        ctx._release_staging([small, large])  # noqa: SLF001

        pool = ctx._staging_pool  # noqa: SLF001
        assert pool[(tuple(small.shape), small.dtype)] == [small]
        assert pool[(tuple(large.shape), large.dtype)] == [large]
    finally:
        ctx.close()
