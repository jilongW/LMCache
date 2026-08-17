# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave.kvweave_config import KVWeaveCodecConfig
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import _KVWeaveCodec


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