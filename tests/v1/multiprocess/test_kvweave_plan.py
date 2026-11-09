# SPDX-License-Identifier: Apache-2.0
"""Per-group KVWeave planning: which groups quantize, and the geometry probe."""

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    AttentionPlaneLayout,
    KVWeaveRuntimeConfig,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import KVWeaveCodec
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    MambaSubStateWireLayout,
)
from lmcache.v1.multiprocess.transfer_context.kvweave_plan import (
    build_group_transfer_plan,
    local_kvweave_geometry,
)

SPLIT_FORMAT = EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
MLA_FORMAT = EngineKVFormat.NL_X_NB_BS_HS
BLOCK = 16
# A split-K/V chunk [2 planes, layers, tokens, hidden]; compressible well below raw.
SPLIT_SHAPE = torch.Size([2, 4, 256, 512])


@pytest.fixture
def quant_on(monkeypatch):
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_QUANT_DEVICE", "cpu")
    cfg = KVWeaveRuntimeConfig.from_env(1)
    return cfg, KVWeaveCodec(cfg.attention_codec_kwargs)


def _attention_info(group_id: int = 0) -> EngineGroupInfo:
    return EngineGroupInfo(
        engine_group_id=group_id,
        layer_indices=(0, 1, 2, 3),
        tokens_per_block=BLOCK,
        cache_category="attention",
    )


def _plan(cfg, codec, *, info, shape=SPLIT_SHAPE, kv_format=SPLIT_FORMAT, dtype=None):
    return build_group_transfer_plan(
        group_info=info,
        layer_indices=(0, 1, 2, 3),
        blocks_per_chunk=shape[-2] // BLOCK,
        chunk_shape=shape,
        dtype=dtype or torch.float16,
        engine_kv_format=kv_format,
        block_size=BLOCK,
        kvweave_config=cfg,
        codec=codec,
    )


def test_split_attention_group_is_quantized_with_a_smaller_wire_layout(quant_on):
    cfg, codec = quant_on
    plan = _plan(cfg, codec, info=_attention_info())

    assert plan.quantized
    assert plan.attention_plane_layout == AttentionPlaneLayout.SPLIT_KV
    assert plan.mamba_options is None
    raw_bytes = SPLIT_SHAPE.numel() * 2
    assert plan.quant_layout_desc is not None
    quant_shape = plan.quant_layout_desc.shapes[0]
    assert plan.quant_layout_desc.dtypes == [torch.uint8]
    assert 0 < quant_shape[0] < raw_bytes
    # The raw layout is always kept: it is the decode target on retrieve.
    assert plan.raw_layout_desc is not None
    assert plan.raw_layout_desc.shapes == [SPLIT_SHAPE]
    assert plan.layer_indices == frozenset({0, 1, 2, 3})
    assert plan.blocks_per_chunk == SPLIT_SHAPE[-2] // BLOCK


def test_master_switch_off_never_quantizes(monkeypatch):
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "0")
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_QUANT_DEVICE", "cpu")
    cfg = KVWeaveRuntimeConfig.from_env(1)
    plan = _plan(cfg, KVWeaveCodec(cfg.attention_codec_kwargs), info=_attention_info())

    assert not plan.quantized
    assert plan.quant_layout_desc is None


def test_mla_group_is_classified_and_left_raw(quant_on):
    cfg, codec = quant_on
    plan = _plan(
        cfg,
        codec,
        info=_attention_info(),
        shape=torch.Size([4, 256, 512]),
        kv_format=MLA_FORMAT,
    )

    assert plan.attention_plane_layout == AttentionPlaneLayout.MLA
    assert not plan.quantized


@pytest.mark.parametrize("info", [None, "unknown"])
def test_unclassified_group_is_never_quantized(quant_on, info):
    cfg, codec = quant_on
    group_info = (
        None
        if info is None
        else EngineGroupInfo(
            engine_group_id=0,
            layer_indices=(0, 1, 2, 3),
            tokens_per_block=BLOCK,
            cache_category="unknown",
        )
    )

    plan = _plan(cfg, codec, info=group_info)

    assert not plan.quantized
    assert plan.attention_plane_layout is None


def test_mamba_group_without_real_layout_falls_back_to_raw(quant_on):
    cfg, codec = quant_on
    info = EngineGroupInfo(
        engine_group_id=1,
        layer_indices=(0, 1, 2, 3),
        tokens_per_block=BLOCK,
        cache_category="mamba",
        mamba_real_layout=None,
    )

    plan = _plan(cfg, codec, info=info, shape=torch.Size([4, 16, 1024]))

    assert not plan.quantized
    assert plan.mamba_options is None


def _mamba_info(conv_dim: int) -> EngineGroupInfo:
    conv = MambaSubStateWireLayout(
        byte_offset=0,
        byte_length=3 * conv_dim * 2,
        dtype_str="torch.float16",
        shape=(3, conv_dim),
    )
    ssm = MambaSubStateWireLayout(
        byte_offset=conv.byte_length,
        byte_length=8 * 8 * 4,
        dtype_str="torch.float32",
        shape=(8, 8),
    )
    return EngineGroupInfo(
        engine_group_id=1,
        layer_indices=(4, 5),
        tokens_per_block=BLOCK,
        cache_category="mamba",
        mamba_real_layout=(conv, ssm),
    )


def test_geometry_reads_split_fused_and_mamba_widths():
    split = (_attention_info(), SPLIT_SHAPE, SPLIT_FORMAT)
    mamba = (_mamba_info(conv_dim=96), torch.Size([2, 16, 256]), None)

    assert local_kvweave_geometry([split, mamba]) == (512, 96)
    # No attention group: only the conv width is known.
    assert local_kvweave_geometry([mamba]) == (0, 96)
    # No Mamba group: only the attention plane width is known.
    assert local_kvweave_geometry([split]) == (512, 0)
    assert local_kvweave_geometry([]) == (0, 0)


def test_geometry_halves_the_fused_trailing_axis():
    fused_format = EngineKVFormat.NL_X_NB_BS_NH_CS
    fused = (_attention_info(), torch.Size([4, 256, 1024]), fused_format)

    assert local_kvweave_geometry([fused]) == (512, 0)


def test_geometry_uses_the_first_group_of_each_kind():
    wide = (_attention_info(0), torch.Size([2, 4, 256, 512]), SPLIT_FORMAT)
    narrow = (_attention_info(1), torch.Size([2, 4, 256, 64]), SPLIT_FORMAT)
    first_mamba = (_mamba_info(96), torch.Size([2, 16, 256]), None)
    second_mamba = (_mamba_info(32), torch.Size([2, 16, 256]), None)

    assert local_kvweave_geometry([wide, narrow, first_mamba, second_mamba]) == (
        512,
        96,
    )
