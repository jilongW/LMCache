# SPDX-License-Identifier: Apache-2.0
"""Bit-exact compatibility of fused Mamba kernels with the contiguous codec."""

import struct
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch

from lmcache.v1.distributed.serde.kvweave.kvweave_serde import (
    KVWeaveCodec,
    MambaChunkSplit,
)
from lmcache.v1.multiprocess.group_view import MambaSubStateWireLayout
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    ConvQKVSplit,
    MambaCodecOptions,
)


def _case(dtype, planes, layers=3, blocks=2, head_dim=128, production=False, rows=5):
    conv_shape = (3, 8192) if production else (2, 16)
    ssm_shape = (32, 128, head_dim) if production else (rows, head_dim)
    conv_bytes = torch.Size(conv_shape).numel() * 2
    ssm_bytes = torch.Size(ssm_shape).numel() * dtype.itemsize
    page_bytes = 4 * 1024 * 1024 if production else conv_bytes + ssm_bytes + 32
    page_bytes = (page_bytes + 7) // 8 * 8
    layouts = (
        MambaSubStateWireLayout(0, conv_bytes, "torch.float16", conv_shape),
        MambaSubStateWireLayout(conv_bytes, ssm_bytes, str(dtype), ssm_shape),
    )
    generator = torch.Generator().manual_seed(79)
    ssm = torch.randn(layers, blocks, *ssm_shape, generator=generator).to(dtype)
    ssm.reshape(-1, head_dim)[0] = 0
    ssm.reshape(-1, head_dim)[1] = 1
    split = MambaChunkSplit(
        torch.randn(layers, blocks, *conv_shape, generator=generator).half(), ssm
    )
    hidden = page_bytes // (planes * 2)
    shape = (layers, blocks, hidden) if planes == 1 else (2, layers, blocks, hidden)
    raw = KVWeaveCodec.merge_mamba_chunk(
        split,
        layouts,
        1,
        hidden,
        raw_shape=torch.Size(shape),
        raw_dtype=torch.float16,
    )
    return raw, layouts, ssm


def _scale_entries(payload, layers):
    entries = []
    cursor = 0
    for _ in range(layers):
        count = struct.unpack_from("=I", payload, cursor)[0]
        cursor += 4
        layer = dict(
            (index, (scale, zp))
            for index, scale, zp in struct.iter_unpack(
                "=Iff", payload[cursor : cursor + count * 12]
            )
        )
        entries.append(layer)
        cursor += count * 12
    assert cursor == len(payload)
    return entries


def _check_quantization_error(original, restored, reference):
    original, restored, reference = (
        tensor.float() for tensor in (original, restored, reference)
    )
    assert torch.isfinite(restored).all()
    error = (restored - original).square().mean().sqrt()
    baseline_error = (reference - original).square().mean().sqrt()
    # A quality regression budget, independent of bit-exact compatibility.
    assert error <= baseline_error * 1.01 + 1e-6


def _check(raw, layouts, ssm, qbit, asym, method="per_token", rh=True):
    codec = KVWeaveCodec()
    layers, blocks = ssm.shape[:2]
    old = codec.quantize_mamba_substate_4bit(
        ssm,
        substate="ssm",
        scaling_method=method,
        rh=rh,
        asym=asym,
        qbit=qbit,
    )
    new = codec._quantize_mamba_ssm_direct(
        raw,
        layouts[1],
        1,
        layers,
        blocks,
        scaling_method=method,
        rh=rh,
        asym=asym,
        qbit=qbit,
    )
    old_parsed = codec._read_mamba_substate_payload(old)
    new_parsed = codec._read_mamba_substate_payload(new)
    assert torch.equal(old_parsed["q_data"], new_parsed["q_data"])
    assert _scale_entries(old_parsed["scales"], layers) == _scale_entries(
        new_parsed["scales"], layers
    )
    reference = codec.dequantize_mamba_substate_4bit(old)
    _check_quantization_error(ssm, codec.dequantize_mamba_substate_4bit(new), reference)
    # Old consumers must also decode the new payload without format changes.
    assert torch.equal(
        reference.view(torch.uint8),
        codec.dequantize_mamba_substate_4bit(new).view(torch.uint8),
    )
    expected = raw.clone()
    codec.write_mamba_substate_into(
        reference,
        layouts[1],
        expected,
        layers,
        blocks,
        1,
        raw.shape[-1],
        2 if raw.dim() == 4 else 1,
        raw.dtype,
    )
    for payload in (old, new):
        destination = raw.clone()
        codec._dequantize_mamba_ssm_direct_into(payload, layouts[1], 1, destination)
        restored = codec.split_mamba_chunk(destination, layouts, 1).ssm
        _check_quantization_error(ssm, restored, reference)
        # Check the whole opaque page, including untouched conv and padding.
        assert torch.equal(destination.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("planes", [1, 2])
@pytest.mark.parametrize("qbit", [4, 8])
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("head_dim", [2, 8, 128, 256])
def test_row_fused_rh_matches_contiguous(dtype, planes, qbit, asym, head_dim):
    _check(*_case(dtype, planes, head_dim=head_dim), qbit, asym)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("planes", [1, 2])
@pytest.mark.parametrize("qbit", [4, 8])
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize(
    "method,rh", [("per_token", False), ("per_channel", False), ("per_channel", True)]
)
@pytest.mark.parametrize("head_dim", [1, 2, 3, 8, 34, 128])
def test_other_fused_ssm_configurations(
    dtype, planes, qbit, asym, method, rh, head_dim
):
    # Per-channel RH operates along blocks*rows, not head_dim.
    raw, layouts, ssm = _case(dtype, planes, head_dim=head_dim, rows=4 if rh else 5)
    _check(raw, layouts, ssm, qbit, asym, method, rh)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("planes", [1, 2])
@pytest.mark.parametrize("qbit", [4, 8])
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("method", ["per_token", "per_channel"])
@pytest.mark.parametrize("rh", [False, True])
def test_fused_conv_qkv_configurations(dtype, planes, qbit, asym, method, rh):
    codec = KVWeaveCodec()
    layers, blocks = 3, 2
    shape = (4, 16)
    conv = torch.randn(layers, blocks, *shape).to(dtype)
    conv[0, 0] = 0
    conv_bytes = conv[0, 0].numel() * dtype.itemsize
    layouts = (
        MambaSubStateWireLayout(0, conv_bytes, str(dtype), shape),
        MambaSubStateWireLayout(conv_bytes, 32, "torch.float32", (2, 4)),
    )
    hidden = (conv_bytes + 64) // dtype.itemsize
    raw_shape = (layers, blocks, hidden) if planes == 1 else (2, layers, blocks, hidden)
    raw = codec.merge_mamba_chunk(
        MambaChunkSplit(conv, torch.randn(layers, blocks, 2, 4)),
        layouts,
        1,
        hidden,
        raw_shape=torch.Size(raw_shape),
        raw_dtype=dtype,
    )
    qkv = ConvQKVSplit(key_dim=4, value_dim=8)
    new = codec._quantize_mamba_conv_qkv_direct(
        raw,
        layouts[0],
        qkv,
        1,
        layers,
        blocks,
        scaling_method=method,
        rh=rh,
        asym=asym,
        qbit=qbit,
    )
    old_parts = [
        codec.quantize_mamba_substate_4bit(
            segment.contiguous(),
            substate="conv",
            scaling_method=method,
            rh=rh,
            asym=asym,
            qbit=qbit,
        )
        for segment in torch.split(conv, [4, 4, 8], dim=-1)
    ]
    new_parts = codec.unpack_conv_qkv_payloads(new)
    for old_part, new_part in zip(old_parts, new_parts, strict=True):
        old_p = codec._read_mamba_substate_payload(old_part)
        new_p = codec._read_mamba_substate_payload(new_part)
        assert torch.equal(old_p["q_data"], new_p["q_data"])
        assert _scale_entries(old_p["scales"], layers) == _scale_entries(
            new_p["scales"], layers
        )
        assert torch.equal(
            codec.dequantize_mamba_substate_4bit(old_part).view(torch.uint8),
            codec.dequantize_mamba_substate_4bit(new_part).view(torch.uint8),
        )
    reference = torch.cat(
        [codec.dequantize_mamba_substate_4bit(p) for p in old_parts], dim=-1
    )
    _check_quantization_error(
        conv,
        torch.cat([codec.dequantize_mamba_substate_4bit(p) for p in new_parts], dim=-1),
        reference,
    )
    expected = raw.clone()
    codec.write_mamba_substate_into(
        reference,
        layouts[0],
        expected,
        layers,
        blocks,
        1,
        hidden,
        planes,
        raw.dtype,
    )
    for payload in (new, codec.pack_conv_qkv_payloads(*old_parts)):
        target = raw.clone()
        codec._dequantize_mamba_conv_qkv_direct_into(payload, layouts[0], 1, target)
        assert torch.equal(target.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("qbit", [4, 8])
@pytest.mark.parametrize("asym", [False, True])
@pytest.mark.parametrize("method", ["per_token", "per_channel"])
@pytest.mark.parametrize("rh", [False, True])
def test_row_fused_rh_production_eight_layer_page(qbit, asym, method, rh):
    _check(
        *_case(torch.float32, 2, layers=8, blocks=1, production=True),
        qbit,
        asym,
        method,
        rh,
    )


@pytest.mark.parametrize("qbit", [4, 8])
@pytest.mark.parametrize(
    "conv_method,conv_rh",
    [("per_token", False), ("per_token", True), ("per_channel", False)],
)
@pytest.mark.parametrize("ssm_method", ["per_token", "per_channel"])
@pytest.mark.parametrize("ssm_rh", [False, True])
def test_production_codec_all_valid_configurations(
    qbit,
    conv_method,
    conv_rh,
    ssm_method,
    ssm_rh,
    monkeypatch,
):
    raw, layouts, _ = _case(torch.float32, 2, layers=8, blocks=1, production=True)
    codec = KVWeaveCodec()
    options = MambaCodecOptions(
        conv_scaling_method=conv_method,
        conv_rh=conv_rh,
        ssm_scaling_method=ssm_method,
        ssm_rh=ssm_rh,
        asym=True,
        conv_qbit=qbit,
        ssm_qbit=qbit,
    )
    old = codec.encode_chunk(
        "mamba",
        layouts,
        1,
        replace(
            options, conv_direct_quant_enabled=False, ssm_direct_quant_enabled=False
        ),
        raw,
    )
    chunk = torch.frombuffer(bytearray(old), dtype=torch.uint8)
    with monkeypatch.context() as patch:
        patch.delattr(codec.native, "kvweave_dequantize_mamba_page_substate_direct")
        expected = codec.decode_chunk(
            "mamba",
            layouts,
            1,
            raw.shape,
            raw.dtype,
            chunk,
            out=torch.zeros_like(raw),
        )

    def unexpected_split(*args, **kwargs):
        pytest.fail("all supported configurations must bypass Python split/merge")

    with monkeypatch.context() as patch:
        patch.setattr(codec, "split_mamba_chunk", unexpected_split)
        patch.setattr(codec, "merge_mamba_chunk", unexpected_split)
        new = codec.encode_chunk("mamba", layouts, 1, options, raw)
        for payload in (old, new):
            chunk = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            actual = codec.decode_chunk(
                "mamba",
                layouts,
                1,
                raw.shape,
                raw.dtype,
                chunk,
                out=torch.zeros_like(raw),
            )
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def test_row_fused_rh_concurrent_calls_keep_scales_independent():
    def check(index):
        dtype = (torch.float32, torch.float16, torch.bfloat16)[index % 3]
        _check(
            *_case(dtype, 1 + index % 2, layers=1 + index % 3, rows=4),
            4 if index % 2 else 8,
            bool(index % 3),
            "per_token" if index % 2 else "per_channel",
            bool(index % 3),
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(check, range(6)))


@pytest.mark.parametrize("method", ["per_token", "per_channel"])
@pytest.mark.parametrize("rh", [False, True])
def test_row_fused_rh_does_not_allocate_full_substate_scratch(method, rh):
    raw, layouts, ssm = _case(torch.float32, 2, layers=8, blocks=1, production=True)
    codec = KVWeaveCodec()
    codec._mamba_precond_pair(128)
    destination = torch.empty_like(raw)
    full_substate_bytes = ssm.numel() * ssm.element_size()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
    ) as encode_profile:
        payload = codec._quantize_mamba_ssm_direct(
            raw,
            layouts[1],
            1,
            8,
            1,
            scaling_method=method,
            rh=rh,
            asym=True,
            qbit=4,
        )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
    ) as decode_profile:
        codec._dequantize_mamba_ssm_direct_into(payload, layouts[1], 1, destination)
    for profile in (encode_profile, decode_profile):
        allocations = [
            event.cpu_memory_usage
            for event in profile.events()
            if event.name == "aten::empty"
        ]
        assert max(allocations, default=0) < full_substate_bytes


@pytest.mark.parametrize(
    "corruption", ["truncated", "duplicate", "out_of_range", "count"]
)
@pytest.mark.parametrize("method", ["per_token", "per_channel"])
@pytest.mark.parametrize("rh", [False, True])
def test_row_fused_rh_rejects_malformed_scales(corruption, method, rh):
    raw, layouts, _ = _case(torch.float32, 2, layers=1, blocks=1, rows=4)
    codec = KVWeaveCodec()
    payload = codec._quantize_mamba_ssm_direct(
        raw,
        layouts[1],
        1,
        1,
        1,
        scaling_method=method,
        rh=rh,
        asym=True,
        qbit=4,
    )
    parsed = codec._read_mamba_substate_payload(payload)
    scales = bytearray(parsed["scales"])
    if corruption == "truncated":
        scales.pop()
    elif corruption == "duplicate":
        struct.pack_into("=I", scales, 4 + 12, 0)
    elif corruption == "out_of_range":
        struct.pack_into("=I", scales, 4, 128)
    else:
        struct.pack_into("=I", scales, 0, 0)
    parsed["scales"] = bytes(scales)
    destination = raw.clone()
    with pytest.raises(ValueError, match="Mamba scale"):
        codec._dequantize_mamba_ssm_direct_into_parsed(
            parsed, layouts[1], 1, destination
        )
    assert torch.equal(destination.view(torch.uint8), raw.view(torch.uint8))
