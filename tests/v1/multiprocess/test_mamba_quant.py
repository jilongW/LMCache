# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lmcache.v1.multiprocess.group_view import MambaSubStateWireLayout
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import (
    MambaChunkSplit,
    _KVWeaveCodec,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_config import MambaCodecOptions


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


def test_native_substate_quant_rejects_invalid_substate():
    with pytest.raises(ValueError, match="substate"):
        _KVWeaveCodec.quantize_mamba_substate_4bit(
            torch.randn(2, 4, 8), substate="invalid"
        )


