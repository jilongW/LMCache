# SPDX-License-Identifier: Apache-2.0
"""Paged (in-place) Mamba encode: bytes must equal the gather + encode path."""

# Standard
from dataclasses import replace
import itertools

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    ConvQKVSplit,
    MambaCodecOptions,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import KVWeaveCodec
from lmcache.v1.multiprocess.group_view import MambaSubStateWireLayout

if not (torch_device_type == "xpu" and torch_dev.is_available()):
    pytest.skip("Requires an available XPU runtime", allow_module_level=True)

# First Party
from lmcache.v1.multiprocess.transfer_context.base import (  # noqa: E402
    gather_paged_kv_to_cpu,
    paged_block_ids_tensor,
    prepare_paged_mamba_source,
    scatter_cpu_to_paged_kv,
)

try:
    # Third Party
    from kvweave import kvweave_quant_xpu
except ImportError:  # pragma: no cover
    kvweave_quant_xpu = None

if kvweave_quant_xpu is None or not hasattr(
    kvweave_quant_xpu, "kvweave_serialize_mamba_state_paged"
):
    pytest.skip("kvweave XPU build lacks paged Mamba state", allow_module_level=True)

_BLOCK = 8
_LAYERS = 3
_NUM_BLOCKS = 6
_CONV = (3, 64)  # (kernel_hist, conv_dim); Q/K/V = 16 / 16 / 32
_SSM = (4, 16, 16)
_SPLIT = ConvQKVSplit(key_dim=16, value_dim=32)


def _build_pages() -> tuple[dict[str, torch.Tensor], tuple[MambaSubStateWireLayout, ...]]:
    conv_bytes = int(np.prod(_CONV)) * 2
    ssm_bytes = int(np.prod(_SSM)) * 4
    page_bytes = conv_bytes + ssm_bytes + 64
    head_size = -(-page_bytes // (4 * _BLOCK))  # two fp16 planes of _BLOCK x head_size
    layout = (
        MambaSubStateWireLayout(0, conv_bytes, str(torch.float16), _CONV),
        MambaSubStateWireLayout(conv_bytes, ssm_bytes, str(torch.float32), _SSM),
    )
    torch.manual_seed(11)
    pages = {
        f"layer_{i}": torch.zeros(
            _NUM_BLOCKS, 2, _BLOCK, 1, head_size, dtype=torch.float16, device="xpu"
        )
        for i in range(_LAYERS)
    }
    for tensor in pages.values():
        flat = tensor.view(torch.uint8).view(_NUM_BLOCKS, -1)
        for block in range(_NUM_BLOCKS):
            conv = torch.randn(_CONV, dtype=torch.float16)
            ssm = torch.randn(_SSM, dtype=torch.float32)
            flat[block, :conv_bytes] = conv.view(torch.uint8).flatten().to("xpu")
            flat[block, conv_bytes : conv_bytes + ssm_bytes] = (
                ssm.view(torch.uint8).flatten().to("xpu")
            )
    return pages, layout


@pytest.mark.parametrize("blocks", [1, 2])
@pytest.mark.parametrize(
    ("conv_method", "ssm_method", "ssm_rh"),
    [
        *itertools.product(["per_token", "per_channel"], ["per_token", "per_channel"], [False, True]),
    ],
)
def test_paged_mamba_encode_matches_gather_then_encode(
    blocks: int, conv_method: str, ssm_method: str, ssm_rh: bool
) -> None:
    pages, layout = _build_pages()
    options = MambaCodecOptions(
        conv_scaling_method=conv_method,
        conv_rh=False,
        ssm_scaling_method=ssm_method,
        ssm_rh=ssm_rh,
        asym=True,
        conv_qkv_split=_SPLIT,
    )
    codec = KVWeaveCodec(
        {"device": "xpu", "num_kv_heads": 1, "head_dim": 8, "qbit": 4,
         "paged_direct_encode": True}
    )
    source = prepare_paged_mamba_source(pages)
    assert source is not None
    if not codec.supports_paged_mamba_encode(layout, options, blocks):
        pytest.skip("combination is not eligible for the paged path")

    ids = [4, 1][:blocks]
    chunk = torch.empty(2, _LAYERS, blocks * _BLOCK, pages["layer_0"].shape[-1],
                        dtype=torch.float16).pin_memory()
    gather_paged_kv_to_cpu(pages, ids, blocks, out=[chunk], chunk_indices=[0])
    torch_dev.synchronize()
    expected = codec.encode_chunk("mamba", layout, _BLOCK, options, chunk)

    out = torch.empty(1 << 20, dtype=torch.uint8)
    written = codec.serialize_mamba_paged_into(
        source.layer_ptrs,
        paged_block_ids_tensor(ids, source.device),
        source.num_blocks,
        source.page_stride_bytes,
        layout,
        options,
        out,
    )
    assert bytes(expected) == bytes(out[:written].numpy())


def test_paged_mamba_is_ineligible_when_a_substate_is_not_quantized() -> None:
    _, layout = _build_pages()
    codec = KVWeaveCodec(
        {"device": "xpu", "num_kv_heads": 1, "head_dim": 8, "qbit": 4,
         "paged_direct_encode": True}
    )
    base = MambaCodecOptions(
        conv_scaling_method="per_token",
        conv_rh=False,
        ssm_scaling_method="per_token",
        ssm_rh=False,
        asym=True,
        conv_qkv_split=_SPLIT,
    )
    assert codec.supports_paged_mamba_encode(layout, base, 1)
    assert not codec.supports_paged_mamba_encode(
        layout, replace(base, conv_quant_enabled=False), 1
    )
    assert not codec.supports_paged_mamba_encode(
        layout, replace(base, ssm_quant_enabled=False), 1
    )
    assert not codec.supports_paged_mamba_encode(layout, replace(base, ssm_qbit=16), 1)
    assert not codec.supports_paged_mamba_encode(
        layout, replace(base, ssm_scaling_method="per_tensor"), 1
    )
    off = KVWeaveCodec(
        {
            "device": "xpu",
            "num_kv_heads": 1,
            "head_dim": 8,
            "qbit": 4,
            "paged_direct_encode": False,
        }
    )
    assert not off.supports_paged_mamba_encode(layout, base, 1)


def _substate_bytes(
    pages: dict[str, torch.Tensor], layout, block: int
) -> list[torch.Tensor]:
    """conv and ssm bytes of one page of every layer, as CPU uint8 tensors."""
    conv, ssm = layout
    out = []
    for tensor in pages.values():
        flat = tensor.view(torch.uint8).view(_NUM_BLOCKS, -1)[block].cpu()
        out.append(flat[conv.byte_offset : conv.byte_offset + conv.byte_length])
        out.append(flat[ssm.byte_offset : ssm.byte_offset + ssm.byte_length])
    return out


@pytest.mark.parametrize("blocks", [1, 2])
@pytest.mark.parametrize("skip_blocks", [0, 1])
@pytest.mark.parametrize(
    ("conv_method", "ssm_method"),
    [("per_token", "per_token"), ("per_channel", "per_channel"), ("per_channel", "per_token")],
)
def test_paged_mamba_decode_matches_dense_decode_and_keeps_other_bytes(
    blocks: int, skip_blocks: int, conv_method: str, ssm_method: str
) -> None:
    if skip_blocks >= blocks:
        pytest.skip("nothing left to write")
    source_pages, layout = _build_pages()
    options = MambaCodecOptions(
        conv_scaling_method=conv_method,
        conv_rh=False,
        ssm_scaling_method=ssm_method,
        ssm_rh=False,
        asym=True,
        conv_qkv_split=_SPLIT,
    )
    codec = KVWeaveCodec(
        {"device": "xpu", "num_kv_heads": 1, "head_dim": 8, "qbit": 4,
         "paged_direct_decode": True}
    )
    assert codec.supports_paged_mamba_decode(layout, options, blocks)
    ids = [4, 1][:blocks]
    hidden = source_pages["layer_0"].shape[-1]
    chunk = torch.empty(2, _LAYERS, blocks * _BLOCK, hidden, dtype=torch.float16).pin_memory()
    gather_paged_kv_to_cpu(source_pages, ids, blocks, out=[chunk], chunk_indices=[0])
    torch_dev.synchronize()
    payload = torch.frombuffer(
        bytearray(codec.encode_chunk("mamba", layout, _BLOCK, options, chunk)),
        dtype=torch.uint8,
    )

    def blank() -> dict[str, torch.Tensor]:
        # A recognizable value everywhere, so untouched bytes are detectable.
        return {
            name: torch.full_like(t, 0.3333) for name, t in source_pages.items()
        }

    # Reference: dense decode of the whole chunk, then the baseline scatter.
    reference = blank()
    dense = codec.decode_chunk("mamba", layout, _BLOCK, chunk.shape, chunk.dtype, payload)
    scatter_cpu_to_paged_kv(reference, ids, [dense], blocks)
    torch_dev.synchronize()

    target = blank()
    pristine = blank()
    paged = prepare_paged_mamba_source(target)
    assert paged is not None
    codec.deserialize_mamba_paged(
        payload,
        paged.layer_ptrs,
        paged_block_ids_tensor(ids, paged.device),
        paged.num_blocks,
        paged.page_stride_bytes,
        layout,
        skip_blocks,
    )
    torch_dev.synchronize()

    for index, block in enumerate(ids):
        got = _substate_bytes(target, layout, block)
        if index < skip_blocks:
            assert all(
                torch.equal(a, b) for a, b in zip(got, _substate_bytes(pristine, layout, block))
            ), "a skipped block was written"
        else:
            assert all(
                torch.equal(a, b)
                for a, b in zip(got, _substate_bytes(reference, layout, block))
            ), "paged decode differs from dense decode + scatter"
    # Everything outside the written sub-states (padding, other blocks) is untouched.
    conv, ssm = layout
    for name in target:
        got = target[name].view(torch.uint8).view(_NUM_BLOCKS, -1).cpu()
        want = pristine[name].view(torch.uint8).view(_NUM_BLOCKS, -1).cpu()
        for block in range(_NUM_BLOCKS):
            if block in ids[skip_blocks:]:
                mask = torch.ones(got.shape[1], dtype=torch.bool)
                mask[conv.byte_offset : conv.byte_offset + conv.byte_length] = False
                mask[ssm.byte_offset : ssm.byte_offset + ssm.byte_length] = False
                assert torch.equal(got[block][mask], want[block][mask])
            else:
                assert torch.equal(got[block], want[block])
