# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lmcache.v1.multiprocess.group_view import MambaSubStateWireLayout
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import (
    MambaChunkSplit,
    _KVWeaveCodec,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    ConvQKVSplit,
    MambaCodecOptions,
)

try:
    from kvweave import kvweave_quant_xpu
except ImportError:  # pragma: no cover
    kvweave_quant_xpu = None

requires_xpu = pytest.mark.skipif(
    not (kvweave_quant_xpu is not None and torch.xpu.is_available()),
    reason="kvweave_quant_xpu extension or an XPU device is not available",
)


def _layouts() -> tuple[MambaSubStateWireLayout, MambaSubStateWireLayout]:
    return (
        MambaSubStateWireLayout(0, 16, "torch.float32", (2, 2)),
        MambaSubStateWireLayout(16, 48, "torch.float32", (3, 4)),
    )


def test_split_and_merge_preserve_real_substate_bytes():
    raw = torch.randn(2, 2, 8, 8, dtype=torch.float32)
    split = _KVWeaveCodec.split_mamba_chunk(raw, _layouts(), block_size=2)
    merged = _KVWeaveCodec.merge_mamba_chunk(split, _layouts(), block_size=2, hidden_dim=8)
    split_again = _KVWeaveCodec.split_mamba_chunk(merged, _layouts(), block_size=2)

    assert split.conv.shape == (2, 4, 2, 2)
    assert split.ssm.shape == (2, 4, 3, 4)
    assert torch.equal(split.conv, split_again.conv)
    assert torch.equal(split.ssm, split_again.ssm)


def test_unified_page_view_round_trip_preserves_padding_bytes():
    raw = torch.randn(2, 8, 8, dtype=torch.float32)
    split = _KVWeaveCodec.split_mamba_chunk(raw, _layouts(), block_size=2)
    merged = _KVWeaveCodec.merge_mamba_chunk(
        split,
        _layouts(),
        block_size=2,
        hidden_dim=8,
        raw_shape=raw.shape,
        raw_dtype=raw.dtype,
    )

    assert merged.shape == raw.shape
    assert torch.equal(raw.view(torch.uint8), merged.view(torch.uint8))


def test_merge_writes_real_mamba_substates_into_caller_buffer():
    raw = torch.randn(2, 2, 8, 8, dtype=torch.float32)
    split = _KVWeaveCodec.split_mamba_chunk(raw, _layouts(), block_size=2)
    destination = torch.empty_like(raw)

    merged = _KVWeaveCodec.merge_mamba_chunk(
        split,
        _layouts(),
        block_size=2,
        hidden_dim=8,
        raw_shape=raw.shape,
        raw_dtype=raw.dtype,
        out=destination,
    )
    restored = _KVWeaveCodec.split_mamba_chunk(merged, _layouts(), block_size=2)

    assert merged.data_ptr() == destination.data_ptr()
    assert torch.equal(split.conv, restored.conv)
    assert torch.equal(split.ssm, restored.ssm)


def test_qwen35_mamba_page_quant_round_trip_is_deterministic():
    """Exercise the real Qwen3.5 mixed-dtype Mamba page contract.

    This test intentionally uses the production state shapes and includes
    page padding. It isolates codec precision from vLLM scheduling, block
    tables, and the transport server.
    """
    torch.manual_seed(7)
    block_size = 64
    conv_shape = (3, 8192)
    ssm_shape = (32, 128, 128)
    conv_dtype = torch.float16
    ssm_dtype = torch.float32
    conv_bytes = torch.Size(conv_shape).numel() * conv_dtype.itemsize
    ssm_bytes = torch.Size(ssm_shape).numel() * ssm_dtype.itemsize
    page_bytes = conv_bytes + ssm_bytes + 4096
    hidden_dim = page_bytes // (block_size * conv_dtype.itemsize)
    layouts = (
        MambaSubStateWireLayout(0, conv_bytes, str(conv_dtype), conv_shape),
        MambaSubStateWireLayout(
            conv_bytes, ssm_bytes, str(ssm_dtype), ssm_shape
        ),
    )
    conv = torch.randn(1, 1, *conv_shape, dtype=conv_dtype)
    ssm = torch.randn(1, 1, *ssm_shape, dtype=ssm_dtype)
    raw = _KVWeaveCodec.merge_mamba_chunk(
        MambaChunkSplit(conv, ssm),
        layouts,
        block_size,
        hidden_dim,
        raw_shape=torch.Size([1, block_size, hidden_dim]),
        raw_dtype=conv_dtype,
    )
    codec = _KVWeaveCodec()
    options = MambaCodecOptions(
        conv_scaling_method="per_channel",
        conv_rh=False,
        ssm_scaling_method="per_channel",
        ssm_rh=False,
        asym=True,
    )

    payload = codec.encode_chunk(
        "mamba", layouts, block_size, options, raw
    )
    restored = codec.decode_chunk(
        "mamba", layouts, block_size, raw.shape, raw.dtype,
        torch.frombuffer(bytearray(payload), dtype=torch.uint8),
    )
    original = _KVWeaveCodec.split_mamba_chunk(raw, layouts, block_size)
    decoded = _KVWeaveCodec.split_mamba_chunk(restored, layouts, block_size)

    assert torch.isfinite(decoded.conv).all()
    assert torch.isfinite(decoded.ssm).all()
    assert torch.max(torch.abs(decoded.conv - original.conv)) < 2.0
    assert torch.max(torch.abs(decoded.ssm - original.ssm)) < 2.0


def test_conv_qkv_split_respects_non_default_key_value_dims():
    """Qwen3.5-0.8B's conv_dim (6144) splits unevenly from the 9B default.

    ``key_dim == value_dim == 2048`` here (unlike 9B's
    ``key_dim=2048, value_dim=4096``) -- passing a non-default
    ``ConvQKVSplit`` must change where query/key/value are cut, not just be
    accepted and ignored.
    """
    torch.manual_seed(11)
    block_size = 64
    conv_shape = (3, 6144)
    ssm_shape = (16, 128, 128)
    conv_dtype = torch.float16
    ssm_dtype = torch.float32
    conv_bytes = torch.Size(conv_shape).numel() * conv_dtype.itemsize
    ssm_bytes = torch.Size(ssm_shape).numel() * ssm_dtype.itemsize
    page_bytes = conv_bytes + ssm_bytes
    hidden_dim = page_bytes // (block_size * conv_dtype.itemsize)
    layouts = (
        MambaSubStateWireLayout(0, conv_bytes, str(conv_dtype), conv_shape),
        MambaSubStateWireLayout(conv_bytes, ssm_bytes, str(ssm_dtype), ssm_shape),
    )
    conv = torch.randn(1, 1, *conv_shape, dtype=conv_dtype)
    ssm = torch.randn(1, 1, *ssm_shape, dtype=ssm_dtype)
    raw = _KVWeaveCodec.merge_mamba_chunk(
        MambaChunkSplit(conv, ssm),
        layouts,
        block_size,
        hidden_dim,
        raw_shape=torch.Size([1, block_size, hidden_dim]),
        raw_dtype=conv_dtype,
    )
    codec = _KVWeaveCodec()
    options = MambaCodecOptions(
        conv_scaling_method="per_channel",
        conv_rh=False,
        ssm_scaling_method="per_channel",
        ssm_rh=False,
        asym=True,
        conv_qkv_split=ConvQKVSplit(key_dim=2048, value_dim=2048),
    )

    payload = codec.encode_chunk("mamba", layouts, block_size, options, raw)
    restored = codec.decode_chunk(
        "mamba", layouts, block_size, raw.shape, raw.dtype,
        torch.frombuffer(bytearray(payload), dtype=torch.uint8),
    )
    original = _KVWeaveCodec.split_mamba_chunk(raw, layouts, block_size)
    decoded = _KVWeaveCodec.split_mamba_chunk(restored, layouts, block_size)

    assert torch.isfinite(decoded.conv).all()
    conv_mean_abs_err = torch.mean(torch.abs(decoded.conv - original.conv))
    # per_channel+rh=False on real Qwen3.5-0.8B conv_state measures ~0.005
    # mean abs error; generous margin for this random tensor's own scale.
    assert conv_mean_abs_err < 0.5


def test_encode_chunk_rejects_mismatched_conv_qkv_split():
    """A conv_qkv_split that disagrees with the real conv_dim must raise,
    not silently mis-slice query/key/value (which would corrupt the
    recurrent state)."""
    block_size = 64
    conv_shape = (3, 6144)
    ssm_shape = (16, 128, 128)
    conv_bytes = torch.Size(conv_shape).numel() * 4
    ssm_bytes = torch.Size(ssm_shape).numel() * 4
    layouts = (
        MambaSubStateWireLayout(0, conv_bytes, "torch.float32", conv_shape),
        MambaSubStateWireLayout(conv_bytes, ssm_bytes, "torch.float32", ssm_shape),
    )
    raw = torch.randn(1, block_size, (conv_bytes + ssm_bytes) // (block_size * 4))
    options = MambaCodecOptions(
        conv_scaling_method="per_channel",
        conv_rh=False,
        ssm_scaling_method="per_channel",
        ssm_rh=False,
        asym=True,
        # 6144's real split is key_dim=2048, value_dim=2048; this implies 8192.
        conv_qkv_split=ConvQKVSplit(key_dim=2048, value_dim=4096),
    )

    with pytest.raises(ValueError, match="conv_state last dim"):
        _KVWeaveCodec().encode_chunk("mamba", layouts, block_size, options, raw)


def test_split_accepts_noncontiguous_chunks():
    raw = torch.randn(2, 2, 16, 8, dtype=torch.float32)[:, :, ::2, :]
    assert not raw.is_contiguous()

    split = _KVWeaveCodec.split_mamba_chunk(raw, _layouts(), block_size=2)

    assert split.conv.shape == (2, 4, 2, 2)


def test_merge_preserves_mixed_conv_ssm_dtypes():
    """Opaque page reconstruction must preserve SSM bytes across dtypes."""
    layouts = (
        MambaSubStateWireLayout(0, 8, "torch.float16", (2, 2)),
        MambaSubStateWireLayout(8, 48, "torch.float32", (3, 4)),
    )
    raw = torch.randn(2, 2, 8, 14, dtype=torch.float16)
    split = _KVWeaveCodec.split_mamba_chunk(raw, layouts, block_size=2)
    assert split.conv.dtype == torch.float16
    assert split.ssm.dtype == torch.float32
    merged = _KVWeaveCodec.merge_mamba_chunk(split, layouts, 2, 14)
    split_again = _KVWeaveCodec.split_mamba_chunk(merged, layouts, 2)
    assert torch.equal(split.conv, split_again.conv)
    assert torch.equal(split.ssm, split_again.ssm)


def test_split_rejects_misaligned_chunk_tokens():
    with pytest.raises(ValueError, match="chunk_tokens"):
        _KVWeaveCodec.split_mamba_chunk(torch.randn(2, 2, 7, 8), _layouts(), block_size=2)


def test_payload_bundle_round_trip():
    blob = _KVWeaveCodec.pack_mamba_payloads(b"conv", b"ssm")

    assert _KVWeaveCodec.unpack_mamba_payloads(blob) == (b"conv", b"ssm")


def test_payload_bundle_survives_zero_padding_to_a_larger_slot():
    """Regression test: the transport layer zero-pads the encoded blob to a
    fixed slot size (``estimate_mamba_serialized_size()``'s conservative
    upper bound, not the exact encoded length -- see
    ``_copy_bytes_to_tensor`` in worker_transfer.py). Before ``ssm`` got its
    own length prefix, ``unpack_mamba_payloads`` treated "every byte after
    conv" as the ssm payload, so this padding was silently read back as
    trailing ssm quantized data -- corrupting the recurrent state on
    dequantize while conv (which does have a length prefix) stayed correct.
    """
    blob = _KVWeaveCodec.pack_mamba_payloads(b"conv", b"ssm")
    padded = blob + b"\x00" * 128

    assert _KVWeaveCodec.unpack_mamba_payloads(padded) == (b"conv", b"ssm")


def test_conv_qkv_payload_bundle_round_trip():
    blob = _KVWeaveCodec.pack_conv_qkv_payloads(b"query", b"key", b"value")

    assert _KVWeaveCodec.unpack_conv_qkv_payloads(blob) == (
        b"query", b"key", b"value",
    )


def test_conv_qkv_payload_bundle_survives_zero_padding_to_a_larger_slot():
    blob = _KVWeaveCodec.pack_conv_qkv_payloads(b"query", b"key", b"value")
    padded = blob + b"\x00" * 128

    assert _KVWeaveCodec.unpack_conv_qkv_payloads(padded) == (
        b"query", b"key", b"value",
    )


@pytest.mark.parametrize("substate", ["conv", "ssm"])
def test_native_substate_quant_round_trip(substate: str):
    tensor = torch.randn(2, 4, 8, dtype=torch.float32)
    payload = _KVWeaveCodec.quantize_mamba_substate_4bit(
        tensor, substate=substate, scaling_method="per_tensor"
    )
    restored = _KVWeaveCodec.dequantize_mamba_substate_4bit(payload)

    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    assert torch.max(torch.abs(restored - tensor)) < 1.0


@pytest.mark.parametrize("substate", ["conv", "ssm"])
@pytest.mark.parametrize("asym,rh", [(True, False), (False, True), (True, True)])
def test_sensitive_quantization_round_trip(
    substate: str, asym: bool, rh: bool
):
    """Asymmetry and RH round-trip for both Mamba substates."""
    tensor = torch.randn(2, 4, 4, 8, 16, dtype=torch.float32)
    scaling_method = "per_token" if substate == "conv" and rh else "per_channel"
    payload = _KVWeaveCodec.quantize_mamba_substate_4bit(
        tensor,
        substate=substate,
        scaling_method=scaling_method,
        rh=rh,
        asym=asym,
    )

    # MQ01 header: magic, qbit, dtype, flags, scaling code.
    assert payload[:4] == b"MQ01"
    assert payload[7] == (1 if scaling_method == "per_token" else 2)
    restored = _KVWeaveCodec.dequantize_mamba_substate_4bit(payload)
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    assert torch.isfinite(restored).all()


@pytest.mark.parametrize("substate", ["conv", "ssm"])
@pytest.mark.parametrize("scaling_method", ["per_tensor", "per_channel", "per_token"])
def test_asymmetric_quantization_preserves_requested_scaling_method(
    substate: str, scaling_method: str
):
    """Asymmetric quantization must retain the requested scaling mode."""
    tensor = torch.randn(2, 3, 4, 8, 16, dtype=torch.float32)
    payload = _KVWeaveCodec.quantize_mamba_substate_4bit(
        tensor,
        substate=substate,
        scaling_method=scaling_method,
        asym=True,
        rh=False,
    )
    assert payload[:4] == b"MQ01"
    assert payload[7] == {"per_tensor": 0, "per_token": 1, "per_channel": 2}[scaling_method]
    restored = _KVWeaveCodec.dequantize_mamba_substate_4bit(payload)
    assert restored.shape == tensor.shape
    assert torch.isfinite(restored).all()


@requires_xpu
@pytest.mark.parametrize("substate", ["conv", "ssm"])
def test_xpu_native_substate_quant_round_trip(substate: str):
    """device='xpu' mirror of test_native_substate_quant_round_trip.

    Uses layers=3 (not 1) specifically to catch the class of bug where the
    XPU path's per-layer native calls (kvweave_quant_xpu has no num_layers
    concept, unlike the CPU wrapper's fused entry points) produce a payload
    whose size depends on how many layers are packed -- a single-layer shape
    can't distinguish a broken per-layer framing scheme from a correct one.
    """
    tensor = torch.randn(3, 4, 8, dtype=torch.float32)
    cpu_payload = _KVWeaveCodec.quantize_mamba_substate_4bit(
        tensor, substate=substate, scaling_method="per_tensor", device="cpu"
    )
    xpu_payload = _KVWeaveCodec.quantize_mamba_substate_4bit(
        tensor, substate=substate, scaling_method="per_tensor", device="xpu"
    )
    restored = _KVWeaveCodec.dequantize_mamba_substate_4bit(xpu_payload, device="xpu")

    assert len(xpu_payload) == len(cpu_payload)
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    assert torch.max(torch.abs(restored - tensor)) < 1.0


@requires_xpu
@pytest.mark.parametrize("substate", ["conv", "ssm"])
@pytest.mark.parametrize("asym,rh", [(True, False), (False, True), (True, True)])
def test_xpu_sensitive_quantization_round_trip(substate: str, asym: bool, rh: bool):
    """device='xpu' mirror of test_sensitive_quantization_round_trip."""
    tensor = torch.randn(2, 4, 4, 8, 16, dtype=torch.float32)
    scaling_method = "per_token" if substate == "conv" and rh else "per_channel"
    payload = _KVWeaveCodec.quantize_mamba_substate_4bit(
        tensor,
        substate=substate,
        scaling_method=scaling_method,
        rh=rh,
        asym=asym,
        device="xpu",
    )

    assert payload[:4] == b"MQ01"
    assert payload[7] == (1 if scaling_method == "per_token" else 2)
    restored = _KVWeaveCodec.dequantize_mamba_substate_4bit(payload, device="xpu")
    assert restored.shape == tensor.shape
    assert restored.dtype == tensor.dtype
    assert torch.isfinite(restored).all()


def test_native_substate_quant_rejects_invalid_substate():
    with pytest.raises(ValueError, match="substate"):
        _KVWeaveCodec.quantize_mamba_substate_4bit(
            torch.randn(2, 4, 8), substate="invalid"
        )


@pytest.mark.parametrize("scaling_method", ["per_tensor", "per_channel"])
def test_mamba_options_preserve_explicit_conv_rh_true(
    monkeypatch: pytest.MonkeyPatch,
    scaling_method: str,
):
    """Explicit conv RH setting should not be silently downgraded."""
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_CONV_SCALING_METHOD", scaling_method)
    monkeypatch.setenv("LMCACHE_MP_KVWEAVE_CONV_RH", "true")

    options = MambaCodecOptions.from_env()

    assert options.conv_scaling_method == scaling_method
    assert options.conv_rh is True


def test_resolve_mamba_options_downgrades_rh_for_non_power_of_two_conv_dim():
    """A group whose real conv_state shape (Qwen3.5-0.8B's per-block
    ``(3, 6144)``, i.e. ``kernel_history=3`` x ``conv_dim=6144``) yields a
    non-power-of-2 RH transform length under per_channel scaling (see
    ``mamba_conv_ssm_layout_params.md`` §3.1: real shape ``[6,1,3,6144]`` ->
    transform length 3) must have conv_rh safely downgraded to False at
    registration time, not crash on the first store with ``rh`` now
    defaulting to True (see ``env_vars.md``'s
    ``LMCACHE_MP_KVWEAVE_LINEAR_RH``/``CONV_RH`` note)."""
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        _resolve_mamba_options_for_group,
    )

    layers = 6
    block_size = 64
    conv_shape = (3, 6144)  # per_channel's RH transform length here is 3
    ssm_shape = (128, 128)  # per_channel's RH transform length here is 2048
    layouts = (
        MambaSubStateWireLayout(0, 3 * 6144 * 4, "torch.float32", conv_shape),
        MambaSubStateWireLayout(
            3 * 6144 * 4, 128 * 128 * 4, "torch.float32", ssm_shape
        ),
    )
    group_info = EngineGroupInfo(
        engine_group_id=0,
        layer_indices=tuple(range(layers)),
        tokens_per_block=block_size,
        cache_category="mamba",
        mamba_real_layout=layouts,
    )
    options = MambaCodecOptions(
        conv_scaling_method="per_channel",
        conv_rh=True,
        ssm_scaling_method="per_channel",
        ssm_rh=True,
        asym=True,
        # Qwen3.5-0.8B: key_dim=value_dim=2048, so conv_dim=2048*2+2048=6144,
        # matching this test's conv_shape.
        conv_qkv_split=ConvQKVSplit(key_dim=2048, value_dim=2048),
    )
    # [L, T, H]; layers and tokens (= one block) are what the resolver reads.
    chunk_shape = torch.Size([layers, block_size, 1])

    resolved = _resolve_mamba_options_for_group(
        group_info, chunk_shape, block_size, options
    )

    assert resolved.conv_rh is False
    assert resolved.ssm_rh is True
    # Untouched fields (including the input's own conv_rh) must not mutate.
    assert options.conv_rh is True


def test_resolve_mamba_options_keeps_rh_for_power_of_two_shapes():
    """A group whose real conv/ssm shapes both yield a power-of-2 RH
    transform length must keep the environment-resolved rh=True untouched.

    Per ``mamba_conv_ssm_layout_params.md`` §3.1, ``per_channel`` conv_state
    (the default scaling method) yields transform length = kernel_history
    (here 3), which is *never* a power of 2 regardless of model size -- so
    this uses ``per_token`` for conv (the one combination the doc's real
    9B measurement, shape ``[6,1,3,8192]``, confirms passes: transform
    length 8192 = 2^13) to exercise the "stays True" branch honestly. ssm's
    ``per_channel`` (§3.2) is a power of 2 on both 0.8B and 9B, so it keeps
    the default scaling method.
    """
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        _resolve_mamba_options_for_group,
    )

    layers = 6
    block_size = 64
    conv_shape = (3, 8192)  # per_token's RH transform length here is 8192
    ssm_shape = (128, 128)  # per_channel's RH transform length here is 4096
    layouts = (
        MambaSubStateWireLayout(0, 3 * 8192 * 4, "torch.float32", conv_shape),
        MambaSubStateWireLayout(
            3 * 8192 * 4, 128 * 128 * 4, "torch.float32", ssm_shape
        ),
    )
    group_info = EngineGroupInfo(
        engine_group_id=0,
        layer_indices=tuple(range(layers)),
        tokens_per_block=block_size,
        cache_category="mamba",
        mamba_real_layout=layouts,
    )
    options = MambaCodecOptions(
        conv_scaling_method="per_token",
        conv_rh=True,
        ssm_scaling_method="per_channel",
        ssm_rh=True,
        asym=True,
        # Qwen3.5-9B: key_dim=2048, value_dim=4096, so conv_dim=8192,
        # matching this test's conv_shape.
        conv_qkv_split=ConvQKVSplit(key_dim=2048, value_dim=4096),
    )
    chunk_shape = torch.Size([layers, block_size, 1])

    resolved = _resolve_mamba_options_for_group(
        group_info, chunk_shape, block_size, options
    )

    assert resolved is options
    assert resolved.conv_rh is True
    assert resolved.ssm_rh is True


@requires_xpu
def test_xpu_quantize_mamba_jobs_matches_per_item_for_homogeneous_dtype():
    """``_quantize_mamba_jobs``'s XPU batched path (>=2 same-dtype jobs)
    must produce byte-identical payloads to calling
    ``quantize_mamba_substate_4bit`` once per job -- batching only removes
    the redundant per-job host<->XPU round trip, never changes the wire
    format."""
    torch.manual_seed(21)
    jobs = [
        dict(tensor=torch.randn(2, 4, 8, 16).to(torch.float16),
             substate="conv", scaling_method="per_channel", rh=True, asym=True, qbit=4),
        dict(tensor=torch.randn(2, 4, 8, 16).to(torch.float16),
             substate="conv", scaling_method="per_channel", rh=True, asym=True, qbit=4),
        dict(tensor=torch.randn(2, 4, 8, 24).to(torch.float16),
             substate="conv", scaling_method="per_channel", rh=False, asym=True, qbit=4),
        dict(tensor=torch.randn(2, 4, 4, 32).to(torch.float16),
             substate="ssm", scaling_method="per_token", rh=True, asym=False, qbit=8),
    ]
    batched = _KVWeaveCodec._quantize_mamba_jobs(jobs, "xpu")
    per_item = [
        _KVWeaveCodec.quantize_mamba_substate_4bit(
            job["tensor"], substate=job["substate"], scaling_method=job["scaling_method"],
            rh=job["rh"], asym=job["asym"], qbit=job["qbit"], device="xpu",
        )
        for job in jobs
    ]
    assert batched == per_item


@requires_xpu
def test_xpu_quantize_mamba_jobs_handles_mixed_dtype_groups():
    """A dtype-heterogeneous job list (e.g. Qwen3.5's fp16 conv + fp32 ssm)
    must still round-trip correctly: the fp16 jobs batch together, the fp32
    job falls back to its own single-item call (see ``_quantize_mamba_jobs``'s
    per-dtype grouping)."""
    torch.manual_seed(22)
    jobs = [
        dict(tensor=torch.randn(2, 4, 8, 16).to(torch.float16),
             substate="conv", scaling_method="per_channel", rh=True, asym=True, qbit=4),
        dict(tensor=torch.randn(2, 4, 8, 16).to(torch.float16),
             substate="conv", scaling_method="per_channel", rh=True, asym=True, qbit=4),
        dict(tensor=torch.randn(2, 4, 8, 24).to(torch.float16),
             substate="conv", scaling_method="per_channel", rh=False, asym=True, qbit=4),
        dict(tensor=torch.randn(2, 4, 4, 32, dtype=torch.float32),
             substate="ssm", scaling_method="per_token", rh=True, asym=False, qbit=8),
    ]
    payloads = _KVWeaveCodec._quantize_mamba_jobs(jobs, "xpu")
    for job, payload in zip(jobs, payloads):
        restored = _KVWeaveCodec.dequantize_mamba_substate_4bit(payload, device="xpu")
        assert restored.shape == job["tensor"].shape
        assert restored.dtype == job["tensor"].dtype
        assert torch.isfinite(restored).all()


@requires_xpu
def test_xpu_dequantize_mamba_payloads_matches_per_item_dispatch():
    """``_dequantize_mamba_payloads``'s XPU batched path (>=2 same-output-
    dtype payloads) must produce the exact same tensors as dequantizing
    each payload individually."""
    torch.manual_seed(23)
    jobs = [
        dict(tensor=torch.randn(2, 4, 8, 16).to(torch.float16),
             substate="conv", scaling_method="per_channel", rh=True, asym=True, qbit=4),
        dict(tensor=torch.randn(2, 4, 8, 16).to(torch.float16),
             substate="conv", scaling_method="per_channel", rh=True, asym=True, qbit=4),
        dict(tensor=torch.randn(2, 4, 4, 32).to(torch.float16),
             substate="ssm", scaling_method="per_token", rh=True, asym=False, qbit=8),
    ]
    payloads = [
        _KVWeaveCodec.quantize_mamba_substate_4bit(
            job["tensor"], substate=job["substate"], scaling_method=job["scaling_method"],
            rh=job["rh"], asym=job["asym"], qbit=job["qbit"], device="xpu",
        )
        for job in jobs
    ]
    batched = _KVWeaveCodec._dequantize_mamba_payloads(payloads, "xpu")
    per_item = [
        _KVWeaveCodec.dequantize_mamba_substate_4bit(p, device="xpu") for p in payloads
    ]
    for b, p in zip(batched, per_item):
        assert torch.equal(b, p)


@requires_xpu
def test_xpu_encode_decode_chunk_mamba_batches_homogeneous_dtype_round_trip():
    """End-to-end ``encode_chunk``/``decode_chunk`` on ``device="xpu"`` with
    conv and ssm sharing one dtype (the common case): all 4 substates
    (conv query/key/value + ssm) batch into a single native round trip on
    both encode and decode -- see ``_quantize_mamba_jobs``/
    ``_decode_mamba_substates_xpu_batched``.
    """
    torch.manual_seed(24)
    block_size = 4
    conv_shape = (4, 96)  # key_dim=32, value_dim=32 -> conv_dim=96
    ssm_shape = (8, 64)
    conv_dtype = torch.float16
    conv_bytes = torch.Size(conv_shape).numel() * conv_dtype.itemsize
    ssm_bytes = torch.Size(ssm_shape).numel() * conv_dtype.itemsize
    hidden_dim = (conv_bytes + ssm_bytes) // (block_size * conv_dtype.itemsize)
    layouts = (
        MambaSubStateWireLayout(0, conv_bytes, str(conv_dtype), conv_shape),
        MambaSubStateWireLayout(conv_bytes, ssm_bytes, str(conv_dtype), ssm_shape),
    )
    conv = torch.randn(2, 2, *conv_shape, dtype=conv_dtype)
    ssm = torch.randn(2, 2, *ssm_shape, dtype=conv_dtype)
    raw = _KVWeaveCodec.merge_mamba_chunk(
        MambaChunkSplit(conv, ssm), layouts, block_size, hidden_dim,
        raw_shape=torch.Size([2, block_size * 2, hidden_dim]), raw_dtype=conv_dtype,
    )
    codec = _KVWeaveCodec({"device": "xpu"})
    options = MambaCodecOptions(
        conv_scaling_method="per_channel", conv_rh=True,
        ssm_scaling_method="per_channel", ssm_rh=True,
        asym=True, conv_qkv_split=ConvQKVSplit(key_dim=32, value_dim=32),
    )

    payload = codec.encode_chunk("mamba", layouts, block_size, options, raw)
    restored = codec.decode_chunk(
        "mamba", layouts, block_size, raw.shape, raw.dtype,
        torch.frombuffer(bytearray(payload), dtype=torch.uint8),
    )
    original = _KVWeaveCodec.split_mamba_chunk(raw, layouts, block_size)
    decoded = _KVWeaveCodec.split_mamba_chunk(restored, layouts, block_size)

    assert torch.isfinite(decoded.conv).all()
    assert torch.isfinite(decoded.ssm).all()
    assert torch.max(torch.abs(decoded.conv - original.conv)) < 2.0
    assert torch.max(torch.abs(decoded.ssm - original.ssm)) < 2.0


@requires_xpu
def test_xpu_encode_decode_chunk_mamba_mixed_dtype_round_trip():
    """End-to-end ``encode_chunk``/``decode_chunk`` on ``device="xpu"`` with
    conv fp16 + ssm fp32 (Qwen3.5's real mixed-dtype contract): conv's
    query/key/value batch together, ssm falls back to its own single-item
    native call (see ``_quantize_mamba_jobs``/``_dequantize_mamba_payloads``'s
    per-dtype grouping).
    """
    torch.manual_seed(25)
    block_size = 4
    conv_shape = (4, 96)  # key_dim=32, value_dim=32 -> conv_dim=96
    ssm_shape = (8, 64)
    conv_dtype = torch.float16
    ssm_dtype = torch.float32
    conv_bytes = torch.Size(conv_shape).numel() * conv_dtype.itemsize
    ssm_bytes = torch.Size(ssm_shape).numel() * ssm_dtype.itemsize
    hidden_dim = (conv_bytes + ssm_bytes) // (block_size * conv_dtype.itemsize)
    layouts = (
        MambaSubStateWireLayout(0, conv_bytes, str(conv_dtype), conv_shape),
        MambaSubStateWireLayout(conv_bytes, ssm_bytes, str(ssm_dtype), ssm_shape),
    )
    conv = torch.randn(2, 2, *conv_shape, dtype=conv_dtype)
    ssm = torch.randn(2, 2, *ssm_shape, dtype=ssm_dtype)
    raw = _KVWeaveCodec.merge_mamba_chunk(
        MambaChunkSplit(conv, ssm), layouts, block_size, hidden_dim,
        raw_shape=torch.Size([2, block_size * 2, hidden_dim]), raw_dtype=conv_dtype,
    )
    codec = _KVWeaveCodec({"device": "xpu"})
    options = MambaCodecOptions(
        conv_scaling_method="per_channel", conv_rh=True,
        ssm_scaling_method="per_channel", ssm_rh=True,
        asym=True, conv_qkv_split=ConvQKVSplit(key_dim=32, value_dim=32),
    )

    payload = codec.encode_chunk("mamba", layouts, block_size, options, raw)
    restored = codec.decode_chunk(
        "mamba", layouts, block_size, raw.shape, raw.dtype,
        torch.frombuffer(bytearray(payload), dtype=torch.uint8),
    )
    original = _KVWeaveCodec.split_mamba_chunk(raw, layouts, block_size)
    decoded = _KVWeaveCodec.split_mamba_chunk(restored, layouts, block_size)

    assert torch.isfinite(decoded.conv).all()
    assert torch.isfinite(decoded.ssm).all()
    assert torch.max(torch.abs(decoded.conv - original.conv)) < 2.0
    assert torch.max(torch.abs(decoded.ssm - original.ssm)) < 2.0
