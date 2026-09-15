# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    ConvQKVSplit,
    KVWeaveCodecConfig,
    KVWeaveRuntimeConfig,
    MambaCodecOptions,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import _KVWeaveCodec
from lmcache.v1.multiprocess.group_view import MambaSubStateWireLayout


def _codec(**kwargs: object) -> _KVWeaveCodec:
    settings = {
        "quantize": True,
        "qbit": 4,
        "scaling_method": "per_channel",
        "rh": False,
        "asym": True,
        "block_size": 64,
        "num_kv_heads": 1,
        "head_dim": 8,
    }
    settings.update(kwargs)
    return _KVWeaveCodec(settings)


def test_config_generates_deterministic_preconditioner():
    first = KVWeaveCodecConfig(precond_seed=7).get_pd_matrix(8)
    second = KVWeaveCodecConfig(precond_seed=7).get_pd_matrix(8)

    assert first is not None
    assert second is not None
    assert (first[0] == second[0]).all()
    assert (first[1] == second[1]).all()


def test_quantized_four_dimensional_round_trip():
    codec = _codec()
    source = torch.randn(2, 1, 64, 8, dtype=torch.float16)
    payload = codec.serialize_tensor(source)
    restored = torch.empty_like(source)

    codec.deserialize_tensor(torch.tensor(list(payload), dtype=torch.uint8), restored)

    assert payload[:4] == b"KVW3"
    assert restored.shape == source.shape
    assert torch.max(torch.abs(source.float() - restored.float())) < 0.5


def test_quantized_three_dimensional_round_trip():
    codec = _codec()
    source = torch.randn(64, 2, 8, dtype=torch.float16)
    payload = codec.serialize_tensor(source)
    restored = torch.empty_like(source)

    codec.deserialize_tensor(torch.tensor(list(payload), dtype=torch.uint8), restored)

    assert restored.shape == source.shape
    assert torch.max(torch.abs(source.float() - restored.float())) < 0.5


def test_raw_mode_round_trip():
    codec = _codec(quantize=False)
    source = torch.randn(2, 1, 64, 8, dtype=torch.float16)
    payload = codec.serialize_tensor(source)
    restored = torch.empty_like(source)

    codec.deserialize_tensor(torch.tensor(list(payload), dtype=torch.uint8), restored)

    assert payload[:4] == b"KVW0"
    assert torch.equal(source, restored)


def test_estimate_serialized_size_is_an_upper_bound():
    codec = _codec()
    source = torch.randn(2, 1, 64, 8, dtype=torch.float16)
    layout = MemoryLayoutDesc([source.shape], [source.dtype])

    assert codec.estimate_serialized_size(layout) >= len(codec.serialize_tensor(source))


def test_rejects_non_kv_shape():
    with pytest.raises(ValueError, match="KVWeave"):
        _codec().serialize_tensor(torch.randn(1, 64, 8))


def test_runtime_config_resolves_environment(monkeypatch, tmp_path):
    model_dir = tmp_path / "SomeModel"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "head_dim": 128,
                    "num_key_value_heads": 8,
                    "linear_key_head_dim": 64,
                    "linear_num_key_heads": 4,
                    "linear_value_head_dim": 64,
                    "linear_num_value_heads": 4,
                }
            }
        )
    )
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("MODEL", "SomeModel")
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "true")
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_LINEAR_QUANT_ENABLED", "false")
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_LINEAR_MAX_SIZE_RATIO", "1.5")
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_PRECOND", "1")
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_CONV_SCALING_METHOD", "per_token")
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_CONV_RH", "true")

    config = KVWeaveRuntimeConfig.from_env()

    assert config.enabled
    assert not config.linear_quant_enabled
    assert config.linear_max_size_ratio == 1.5
    assert config.attention_codec_kwargs["num_kv_heads"] == 8
    assert config.attention_codec_kwargs["head_dim"] == 128
    assert config.attention_codec_kwargs["precond"]
    assert config.mamba_options == MambaCodecOptions(
        conv_scaling_method="per_token",
        conv_rh=True,
        ssm_scaling_method="per_channel",
        ssm_rh=True,
        asym=True,
        ssm_qbit=4,
        conv_qkv_split=ConvQKVSplit(key_dim=256, value_dim=256),
    )


def test_runtime_config_falls_back_to_qwen35_9b_defaults(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL", raising=False)

    config = KVWeaveRuntimeConfig.from_env()

    assert config.attention_codec_kwargs["num_kv_heads"] == 4
    assert config.attention_codec_kwargs["head_dim"] == 256
    assert config.mamba_options.conv_qkv_split == ConvQKVSplit(
        key_dim=2048, value_dim=4096
    )


def test_runtime_config_falls_back_when_config_json_missing_fields(
    monkeypatch, tmp_path
):
    model_dir = tmp_path / "PartialModel"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"text_config": {}}))
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("MODEL", "PartialModel")

    config = KVWeaveRuntimeConfig.from_env()

    assert config.attention_codec_kwargs["num_kv_heads"] == 4
    assert config.attention_codec_kwargs["head_dim"] == 256
    assert config.mamba_options.conv_qkv_split == ConvQKVSplit(
        key_dim=2048, value_dim=4096
    )


def test_codec_chunk_methods_dispatch_attention():
    codec = _codec()
    source = torch.randn(2, 1, 64, 8, dtype=torch.float16)

    payload = codec.encode_chunk("attention", None, 64, None, source)
    restored = codec.decode_chunk(
        "attention",
        None,
        64,
        source.shape,
        source.dtype,
        torch.tensor(list(payload), dtype=torch.uint8),
    )

    assert restored.shape == source.shape
    assert torch.max(torch.abs(source.float() - restored.float())) < 0.5


def _mamba_layouts() -> tuple[MambaSubStateWireLayout, MambaSubStateWireLayout]:
    return (
        MambaSubStateWireLayout(0, 16, "torch.float32", (2, 2)),
        MambaSubStateWireLayout(16, 48, "torch.float32", (3, 4)),
    )


def test_estimate_mamba_serialized_size_is_positive_and_scales_with_layers():
    small_layout = MemoryLayoutDesc([torch.Size([2, 2, 8, 8])], [torch.float32])
    large_layout = MemoryLayoutDesc([torch.Size([2, 8, 8, 8])], [torch.float32])

    small = _KVWeaveCodec.estimate_mamba_serialized_size(
        small_layout, _mamba_layouts(), block_size=2
    )
    large = _KVWeaveCodec.estimate_mamba_serialized_size(
        large_layout, _mamba_layouts(), block_size=2
    )

    assert small > 0
    assert large > small


def test_estimate_mamba_serialized_size_rejects_misaligned_block_size():
    layout = MemoryLayoutDesc([torch.Size([2, 2, 7, 8])], [torch.float32])
    with pytest.raises(ValueError, match="chunk_tokens"):
        _KVWeaveCodec.estimate_mamba_serialized_size(
            layout, _mamba_layouts(), block_size=2
        )