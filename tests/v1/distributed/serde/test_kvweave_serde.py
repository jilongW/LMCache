# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch

from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    AttentionPlaneLayout,
    ConvQKVSplit,
    KVWeaveCodecConfig,
    KVWeaveRuntimeConfig,
    MambaCodecOptions,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import (
    MambaChunkSplit,
    _KVWeaveCodec,
)
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


def test_qwen35_fused_attention_keeps_kv_planes_separate():
    codec = _codec(num_kv_heads=4, head_dim=256)
    hidden = 4 * 2 * 256

    assert codec._fused_head_num(hidden) == 8
    assert codec._fused_head_dim(hidden) == 256


def test_qwen35_fused_attention_estimate_covers_payload():
    codec = _codec(num_kv_heads=4, head_dim=256)
    source = torch.randn(8, 64, 2048, dtype=torch.float16)
    layout = MemoryLayoutDesc([source.shape], [source.dtype])

    payload = codec.serialize_fused_tensor(source)

    assert payload[:4] == b"KVW4"
    assert codec.estimate_fused_serialized_size(layout) >= len(payload)


def test_qwen35_fused_attention_preconditioned_round_trip():
    codec = _codec(
        num_kv_heads=4,
        head_dim=256,
        block_size=64,
        rh=True,
        asym=True,
        precond=True,
    )
    source = torch.randn(8, 64, 2048, dtype=torch.float16)

    payload = codec.encode_chunk(
        "attention",
        None,
        64,
        None,
        source,
        AttentionPlaneLayout.FUSED_KV,
    )
    restored = codec.decode_chunk(
        "attention",
        None,
        64,
        source.shape,
        source.dtype,
        torch.frombuffer(bytearray(payload), dtype=torch.uint8),
        AttentionPlaneLayout.FUSED_KV,
    )

    assert torch.isfinite(restored).all()
    assert torch.max(torch.abs(source.float() - restored.float())) < 2.0


@pytest.mark.parametrize("scaling_method", ["per_channel", "per_token"])
def test_qwen35_fused_attention_uses_independent_head_scales(scaling_method):
    codec = _codec(
        num_kv_heads=2,
        head_dim=4,
        block_size=8,
        scaling_method=scaling_method,
        rh=False,
        asym=True,
    )
    source = torch.randn(2, 8, 16, dtype=torch.float16)
    payload = codec.serialize_fused_tensor(source)
    parsed = codec._parse_fused(payload)
    restored = torch.empty_like(source)
    codec.deserialize_fused_tensor(
        torch.frombuffer(bytearray(payload), dtype=torch.uint8), restored
    )

    assert parsed["per_head_scales"]
    assert torch.max(torch.abs(source.float() - restored.float())) < 0.5


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
    assert config.attention_codec_kwargs["qbit"] == 4
    assert config.attention_codec_kwargs["scaling_method"] == "per_token"
    assert config.attention_codec_kwargs["precond"]
    assert config.mamba_options == MambaCodecOptions(
        conv_scaling_method="per_token",
        conv_rh=True,
        ssm_scaling_method="per_token",
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
    assert config.attention_codec_kwargs["scaling_method"] == "per_token"
    assert config.mamba_options.conv_scaling_method == "per_token"
    assert config.mamba_options.ssm_scaling_method == "per_token"
    assert config.mamba_options.conv_qkv_split == ConvQKVSplit(
        key_dim=2048, value_dim=4096
    )


def test_runtime_config_accepts_attention_qbit_4(monkeypatch):
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_QBIT", "4")

    config = KVWeaveRuntimeConfig.from_env()

    assert config.attention_codec_kwargs["qbit"] == 4


def test_runtime_config_rejects_invalid_attention_qbit(monkeypatch):
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_QBIT", "16")

    with pytest.raises(ValueError, match="LMCACHE_MP_KVWEAVE_QBIT"):
        KVWeaveRuntimeConfig.from_env()


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


def test_codec_decode_chunk_writes_attention_into_caller_buffer():
    codec = _codec()
    source = torch.randn(2, 1, 64, 8, dtype=torch.float16)
    payload = codec.encode_chunk("attention", None, 64, None, source)
    destination = torch.empty_like(source)

    restored = codec.decode_chunk(
        "attention",
        None,
        64,
        source.shape,
        source.dtype,
        torch.tensor(list(payload), dtype=torch.uint8),
        out=destination,
    )

    assert restored.data_ptr() == destination.data_ptr()
    assert torch.max(torch.abs(source.float() - restored.float())) < 0.5


def _mamba_layouts() -> tuple[MambaSubStateWireLayout, MambaSubStateWireLayout]:
    return (
        MambaSubStateWireLayout(0, 16, "torch.float32", (2, 2)),
        MambaSubStateWireLayout(16, 48, "torch.float32", (3, 4)),
    )


@pytest.mark.parametrize(
    "cache_category,mamba_layout",
    [
        ("mamba", None),
        ("attention", _mamba_layouts()),
        ("unknown", None),
        ("unknown", _mamba_layouts()),
        ("bogus", None),
    ],
)
def test_encode_decode_chunk_reject_invalid_category_dispatch(
    cache_category, mamba_layout
):
    """Mamba data fed to the attention path (or vice versa) must raise.

    Guards MIGRATION_PLAN.md R1/R6: dispatch is based solely on the
    explicit ``cache_category``/``mamba_layout`` pair, never on tensor
    shape, and "unknown" is never silently treated as attention.
    """
    codec = _codec()
    source = torch.randn(2, 1, 64, 8, dtype=torch.float16)

    with pytest.raises(ValueError):
        codec.encode_chunk(cache_category, mamba_layout, 64, None, source)

    with pytest.raises(ValueError):
        codec.decode_chunk(
            cache_category,
            mamba_layout,
            64,
            source.shape,
            source.dtype,
            torch.zeros(1, dtype=torch.uint8),
        )


def test_estimate_mamba_serialized_size_is_positive_and_scales_with_layers():
    small_layout = MemoryLayoutDesc([torch.Size([2, 2, 8, 8])], [torch.float32])
    large_layout = MemoryLayoutDesc([torch.Size([2, 8, 8, 8])], [torch.float32])
    # _mamba_layouts()'s conv sub-state's last dim (conv_dim) is 2; split as
    # key_dim*2 + value_dim = 1*2 + 0 = 2.
    conv_qkv_split = ConvQKVSplit(key_dim=1, value_dim=0)

    small = _KVWeaveCodec.estimate_mamba_serialized_size(
        small_layout, _mamba_layouts(), block_size=2, conv_qkv_split=conv_qkv_split
    )
    large = _KVWeaveCodec.estimate_mamba_serialized_size(
        large_layout, _mamba_layouts(), block_size=2, conv_qkv_split=conv_qkv_split
    )

    assert small > 0
    assert large > small


def test_estimate_mamba_serialized_size_rejects_misaligned_block_size():
    layout = MemoryLayoutDesc([torch.Size([2, 2, 7, 8])], [torch.float32])
    with pytest.raises(ValueError, match="chunk_tokens"):
        _KVWeaveCodec.estimate_mamba_serialized_size(
            layout,
            _mamba_layouts(),
            block_size=2,
            conv_qkv_split=ConvQKVSplit(key_dim=1, value_dim=0),
        )


def test_estimate_mamba_serialized_size_rejects_mismatched_conv_qkv_split():
    """conv_qkv_split must actually describe the real conv_state layout --
    a silently-accepted mismatch would size the SHM slot for the wrong
    number of query/key/value bytes (see worker_transfer.py's
    _resolve_mamba_options_for_group, which raises the same way)."""
    layout = MemoryLayoutDesc([torch.Size([2, 2, 8, 8])], [torch.float32])
    with pytest.raises(ValueError, match="conv_state last dim"):
        _KVWeaveCodec.estimate_mamba_serialized_size(
            layout,
            _mamba_layouts(),
            block_size=2,
            # _mamba_layouts()'s conv last dim is 2; this split implies 3.
            conv_qkv_split=ConvQKVSplit(key_dim=1, value_dim=1),
        )


def test_estimate_mamba_serialized_size_is_a_real_upper_bound_for_split_conv():
    """The estimate must be large enough to hold the real encode_chunk()
    output once conv_state is split into three independently-quantized
    query/key/value payloads (see MIGRATION_PLAN.md's estimate-is-an-
    upper-bound invariant, now extended to the split conv path)."""
    torch.manual_seed(3)
    block_size = 64
    layers = 2
    conv_shape = (3, 6144)  # Qwen3.5-0.8B: non-power-of-2 fused conv_dim
    ssm_shape = (16, 128, 128)
    conv_dtype = torch.float32
    ssm_dtype = torch.float32
    conv_bytes = torch.Size(conv_shape).numel() * conv_dtype.itemsize
    ssm_bytes = torch.Size(ssm_shape).numel() * ssm_dtype.itemsize
    hidden_dim = (conv_bytes + ssm_bytes) // (block_size * conv_dtype.itemsize)
    layouts = (
        MambaSubStateWireLayout(0, conv_bytes, str(conv_dtype), conv_shape),
        MambaSubStateWireLayout(conv_bytes, ssm_bytes, str(ssm_dtype), ssm_shape),
    )
    conv = torch.randn(layers, 1, *conv_shape, dtype=conv_dtype)
    ssm = torch.randn(layers, 1, *ssm_shape, dtype=ssm_dtype)
    raw = _KVWeaveCodec.merge_mamba_chunk(
        MambaChunkSplit(conv, ssm), layouts, block_size, hidden_dim,
        raw_shape=torch.Size([layers, block_size, hidden_dim]), raw_dtype=conv_dtype,
    )
    options = MambaCodecOptions(
        conv_scaling_method="per_channel", conv_rh=False,
        ssm_scaling_method="per_channel", ssm_rh=False,
        asym=True, conv_qkv_split=ConvQKVSplit(key_dim=2048, value_dim=2048),
    )
    codec = _KVWeaveCodec()

    payload = codec.encode_chunk("mamba", layouts, block_size, options, raw)

    estimate = _KVWeaveCodec.estimate_mamba_serialized_size(
        MemoryLayoutDesc([raw.shape], [raw.dtype]),
        layouts,
        block_size,
        conv_qkv_split=options.conv_qkv_split,
        scaling_methods=(options.conv_scaling_method, options.ssm_scaling_method),
        qbits=(options.conv_qbit, options.ssm_qbit),
    )

    assert len(payload) <= estimate


def test_estimate_mamba_serialized_size_compression_ratio_qwen35_9b():
    """Real Qwen3.5-9B GDN geometry should compress meaningfully under
    KVWeave's mamba estimate.

    Regression test for MIGRATION_PLAN.md's 2026-09-17 L1 usage
    investigation: ``_estimate_substate_quantized_size`` used to return
    ``max(quantized_size, raw_size)`` unconditionally, to cover the DEBUG
    ``*_QUANT_ENABLED=0`` raw-fallback toggle. Since that toggle is resolved
    once at registration (not re-read per store call), the estimate now
    takes the resolved ``conv_quant_enabled``/``ssm_quant_enabled`` flags
    and only sizes the branch that will actually run -- ssm's raw fp32
    content (16.78MB) previously dominated the estimate even though its
    quantized payload is only ~2.1MB.
    """
    layers = 8
    blocks = 1
    block_size = 1024
    tokens = blocks * block_size
    conv_shape = (3, 8192)  # (kernel_history, conv_dim=key_dim*2+value_dim)
    conv_dtype = torch.float16
    ssm_shape = (32, 128, 128)  # (num_heads, head_dim, state_size)
    ssm_dtype = torch.float32  # ssm recurrent state is kept in fp32
    conv_qkv_split = ConvQKVSplit(key_dim=2048, value_dim=4096)
    layouts = (
        MambaSubStateWireLayout(0, 0, str(conv_dtype), conv_shape),
        MambaSubStateWireLayout(0, 0, str(ssm_dtype), ssm_shape),
    )
    raw_layout = MemoryLayoutDesc([torch.Size([layers, tokens, 1])], [conv_dtype])

    true_content_size = layers * blocks * (
        torch.Size(conv_shape).numel() * conv_dtype.itemsize
        + torch.Size(ssm_shape).numel() * ssm_dtype.itemsize
    )
    # Sanity-check against the real (engine_group_id=0) registration log from
    # the two-waves smoke test this geometry was reverse-engineered from.
    assert true_content_size == 17_170_432

    estimate = _KVWeaveCodec.estimate_mamba_serialized_size(
        raw_layout,
        layouts,
        block_size,
        conv_qkv_split=conv_qkv_split,
        scaling_methods=("per_channel", "per_channel"),
        qbits=(4, 4),
    )
    estimate_ssm_per_tensor = _KVWeaveCodec.estimate_mamba_serialized_size(
        raw_layout,
        layouts,
        block_size,
        conv_qkv_split=conv_qkv_split,
        scaling_methods=("per_channel", "per_tensor"),
        qbits=(4, 4),
    )
    estimate_conv_per_tensor = _KVWeaveCodec.estimate_mamba_serialized_size(
        raw_layout,
        layouts,
        block_size,
        conv_qkv_split=conv_qkv_split,
        scaling_methods=("per_tensor", "per_channel"),
        qbits=(4, 4),
    )

    print(
        f"\n[Qwen3.5-9B mamba compression] true_content_size={true_content_size} "
        f"quant_estimate(ssm=per_channel)={estimate} "
        f"ratio={estimate / true_content_size:.4f}\n"
        f"[Qwen3.5-9B mamba compression] quant_estimate(ssm=per_tensor)="
        f"{estimate_ssm_per_tensor} "
        f"ratio={estimate_ssm_per_tensor / true_content_size:.4f}\n"
        f"[Qwen3.5-9B mamba compression] quant_estimate(conv=per_tensor)="
        f"{estimate_conv_per_tensor} "
        f"ratio={estimate_conv_per_tensor / true_content_size:.4f}\n"
        f"(<1.0 means smaller than unquantized content)"
    )

    # Per-substate breakdown: shows what the *old* max(quantized, raw)
    # formula would have picked for each substate, for context on why ssm
    # used to dominate the (now-fixed) estimate.
    def _debug_substate_sizes(shape, dtype_str, scaling_method, qbit):
        layers_, blocks_ = shape[0], shape[1]
        tail = shape[2:]
        n = 1
        for d in tail:
            n *= int(d)
        elements = layers_ * blocks_ * max(n, 1)
        _, _, native_head_dim, native_blocks = KVWeaveCodecConfig.mamba_layout(
            "conv", shape, scaling_method
        )
        if scaling_method == "per_tensor":
            native_chunks = 1
        elif scaling_method == "per_channel":
            native_chunks = native_head_dim
        else:
            native_chunks = native_blocks
        scale_blob = 4 + layers_ * (4 + native_chunks * 12)
        q_bytes = KVWeaveCodecConfig.quantized_bytes(elements, qbit)
        quantized_size = (
            elements * 2 if qbit == 16 else 10 + 4 * len(shape) + scale_blob + q_bytes
        )
        raw_size = elements * KVWeaveCodecConfig.mamba_dtype(dtype_str).itemsize
        return quantized_size, raw_size

    for name, shape, dtype_str, scaling in (
        ("conv.query", (layers, blocks, 3, conv_qkv_split.key_dim), str(conv_dtype), "per_channel"),
        ("conv.key", (layers, blocks, 3, conv_qkv_split.key_dim), str(conv_dtype), "per_channel"),
        ("conv.value", (layers, blocks, 3, conv_qkv_split.value_dim), str(conv_dtype), "per_channel"),
        ("ssm", (layers, blocks, *ssm_shape), str(ssm_dtype), "per_channel"),
    ):
        q, r = _debug_substate_sizes(shape, dtype_str, scaling, 4)
        print(
            f"[substate breakdown] {name} ({scaling}): "
            f"quantized_size={q} raw_size={r} max_picks={'quantized' if q >= r else 'raw'}"
        )

    # With the fix, quant_enabled defaults to (True, True), so the estimate
    # is sized off the real quantized payload -- should land well under
    # true_content_size (observed ~0.17x; 0.3x leaves headroom for future
    # header-format changes without making this test flaky).
    assert estimate < true_content_size * 0.3, (
        f"KVWeave mamba estimate ({estimate}) is not meaningfully smaller "
        f"than the true unquantized content ({true_content_size})"
    )
