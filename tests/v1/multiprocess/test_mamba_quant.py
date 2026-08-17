# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lmcache.v1.distributed.serde.kvweave.kvweave_serde import _KVWeaveCodec
from lmcache.v1.multiprocess.group_view import MambaSubStateWireLayout


def _layouts() -> tuple[MambaSubStateWireLayout, MambaSubStateWireLayout]:
    return (
        MambaSubStateWireLayout(0, 16, "torch.float32", (2, 2)),
        MambaSubStateWireLayout(16, 48, "torch.float32", (3, 4)),
    )


def test_split_and_merge_preserve_real_substate_bytes():
    raw = torch.randn(2, 2, 8, 8, dtype=torch.float32)
    split = _KVWeaveCodec.split_mamba_chunk(raw, _layouts(), block_size=2)
    merged = _KVWeaveCodec.merge_mamba_chunk(
        split, _layouts(), block_size=2, hidden_dim=8
    )
    split_again = _KVWeaveCodec.split_mamba_chunk(
        merged, _layouts(), block_size=2
    )

    assert split.conv.shape == (2, 4, 2, 2)
    assert split.ssm.shape == (2, 4, 3, 4)
    assert torch.equal(split.conv, split_again.conv)
    assert torch.equal(split.ssm, split_again.ssm)


def test_split_accepts_noncontiguous_chunks():
    raw = torch.randn(2, 2, 16, 8, dtype=torch.float32)[:, :, ::2, :]
    assert not raw.is_contiguous()

    split = _KVWeaveCodec.split_mamba_chunk(raw, _layouts(), block_size=2)

    assert split.conv.shape == (2, 4, 2, 2)


def test_split_rejects_misaligned_chunk_tokens():
    with pytest.raises(ValueError, match="chunk_tokens"):
        _KVWeaveCodec.split_mamba_chunk(
            torch.randn(2, 2, 7, 8), _layouts(), block_size=2
        )


def test_payload_bundle_round_trip():
    blob = _KVWeaveCodec.pack_mamba_payloads(b"conv", b"ssm")

    assert _KVWeaveCodec.unpack_mamba_payloads(blob) == (b"conv", b"ssm")


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


def test_native_substate_quant_rejects_invalid_substate():
    with pytest.raises(ValueError, match="substate"):
        _KVWeaveCodec.quantize_mamba_substate_4bit(
            torch.randn(2, 4, 8), substate="invalid"
        )
