# SPDX-License-Identifier: Apache-2.0
"""Paged (in-place) Mamba encode / decode with the spectral (svd) ssm codec: bytes must equal the gather + encode path,
and decoded pages must equal dense decode + scatter."""

# Standard
from dataclasses import replace

# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    ConvQKVSplit,
    MambaCodecOptions,
    SvdOptions,
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

if kvweave_quant_xpu is None or not hasattr(kvweave_quant_xpu, "spectral_decode_chunk_paged"):
    pytest.skip("kvweave XPU build lacks the paged spectral encoder / decoder", allow_module_level=True)

_BLOCK = 8
_LAYERS = 3
_NUM_BLOCKS = 6
_CONV = (3, 64)
_SSM = (4, 128, 128)  # (heads, v, k): the spectral kernels are specialised for 128 x 128 heads
_SPLIT = ConvQKVSplit(key_dim=16, value_dim=32)


def _build_pages(ssm_dtype: torch.dtype):
    conv_bytes = int(np.prod(_CONV)) * 2
    ssm_bytes = int(np.prod(_SSM)) * torch.empty((), dtype=ssm_dtype).element_size()
    page_bytes = conv_bytes + ssm_bytes + 64
    head_size = -(-page_bytes // (4 * _BLOCK))
    layout = (
        MambaSubStateWireLayout(0, conv_bytes, str(torch.float16), _CONV),
        MambaSubStateWireLayout(conv_bytes, ssm_bytes, str(ssm_dtype), _SSM),
    )
    torch.manual_seed(5)
    pages = {
        f"layer_{i}": torch.zeros(_NUM_BLOCKS, 2, _BLOCK, 1, head_size, dtype=torch.float16, device="xpu")
        for i in range(_LAYERS)
    }
    sv = torch.pow(0.93, torch.arange(128, dtype=torch.float32))
    for tensor in pages.values():
        flat = tensor.view(torch.uint8).view(_NUM_BLOCKS, -1)
        for block in range(_NUM_BLOCKS):
            conv = torch.randn(_CONV, dtype=torch.float16)
            ssm = (torch.randn(_SSM) * sv) @ torch.randn(_SSM) / 128**0.5  # low-rank-ish like a GDN state
            flat[block, :conv_bytes] = conv.view(torch.uint8).flatten().to("xpu")
            flat[block, conv_bytes : conv_bytes + ssm_bytes] = (
                ssm.to(ssm_dtype).view(torch.uint8).flatten().to("xpu")
            )
    return pages, layout


def _options() -> MambaCodecOptions:
    return MambaCodecOptions(
        conv_scaling_method="per_token",
        conv_rh=False,
        ssm_scaling_method="per_token",
        ssm_rh=False,
        asym=False,
        conv_qkv_split=_SPLIT,
        ssm_codec="svd",
        ssm_svd=SvdOptions(budget_bits=1.5, r_max=96, weights="none"),
        ssm_layer_ids=tuple(range(_LAYERS)),
    )


def _codec(**extra) -> KVWeaveCodec:
    return KVWeaveCodec(
        {"device": "xpu", "num_kv_heads": 1, "head_dim": 8, "qbit": 4, "paged_direct_encode": True, **extra}
    )


@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("blocks", [1, 2])
def test_paged_svd_encode_matches_gather_then_encode(ssm_dtype: torch.dtype, blocks: int) -> None:
    pages, layout = _build_pages(ssm_dtype)
    options = _options()
    codec = _codec()
    source = prepare_paged_mamba_source(pages)
    assert source is not None
    assert codec.supports_paged_mamba_encode(layout, options, blocks)

    ids = [4, 1][:blocks]
    chunk = torch.empty(2, _LAYERS, blocks * _BLOCK, pages["layer_0"].shape[-1], dtype=torch.float16).pin_memory()
    gather_paged_kv_to_cpu(pages, ids, blocks, out=[chunk], chunk_indices=[0])
    torch_dev.synchronize()
    expected = codec.encode_chunk("mamba", layout, _BLOCK, options, chunk)

    out = torch.empty(1 << 22, dtype=torch.uint8)
    written = codec.serialize_mamba_paged_into(
        source.layer_ptrs, paged_block_ids_tensor(ids, source.device), source.num_blocks, source.page_stride_bytes,
        layout, options, out,
    )
    assert bytes(expected) == bytes(out[:written].numpy())


def test_svd_is_paged_encodable_and_decodable() -> None:
    _, layout = _build_pages(torch.float32)
    options = _options()
    codec = _codec(paged_direct_decode=True)
    assert codec.supports_paged_mamba_encode(layout, options, 1)
    assert codec.supports_paged_mamba_decode(layout, options, 1)
    # a head that is not 128 x 128, or a disabled ssm quantizer, keeps the gather / scatter path
    small = (layout[0], MambaSubStateWireLayout(layout[1].byte_offset, 4 * 16 * 16 * 4, str(torch.float32), (4, 16, 16)))
    for supports in (codec.supports_paged_mamba_encode, codec.supports_paged_mamba_decode):
        assert not supports(small, options, 1)
        assert not supports(layout, replace(options, ssm_quant_enabled=False), 1)
    assert not _codec(paged_direct_decode=False).supports_paged_mamba_decode(layout, options, 1)


def _substate_bytes(pages: dict[str, torch.Tensor], layout, block: int) -> list[torch.Tensor]:
    """conv and ssm bytes of one page of every layer, as CPU uint8 tensors."""
    out = []
    for tensor in pages.values():
        flat = tensor.view(torch.uint8).view(_NUM_BLOCKS, -1)[block].cpu()
        for sub in layout:
            out.append(flat[sub.byte_offset : sub.byte_offset + sub.byte_length])
    return out


@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("blocks", "skip_blocks"), [(1, 0), (2, 0), (2, 1)])
def test_paged_svd_decode_matches_dense_decode_and_keeps_other_bytes(
    ssm_dtype: torch.dtype, blocks: int, skip_blocks: int
) -> None:
    source_pages, layout = _build_pages(ssm_dtype)
    options = _options()
    codec = _codec(paged_direct_decode=True)
    assert codec.supports_paged_mamba_decode(layout, options, blocks)
    ids = [4, 1][:blocks]
    hidden = source_pages["layer_0"].shape[-1]
    chunk = torch.empty(2, _LAYERS, blocks * _BLOCK, hidden, dtype=torch.float16).pin_memory()
    gather_paged_kv_to_cpu(source_pages, ids, blocks, out=[chunk], chunk_indices=[0])
    torch_dev.synchronize()
    payload = torch.frombuffer(
        bytearray(codec.encode_chunk("mamba", layout, _BLOCK, options, chunk)), dtype=torch.uint8
    )
    assert payload.numel() > 0

    def blank() -> dict[str, torch.Tensor]:
        return {name: torch.full_like(t, 0.3333) for name, t in source_pages.items()}   # recognizable everywhere

    reference = blank()   # dense decode of the whole chunk, then the baseline scatter
    dense = codec.decode_chunk("mamba", layout, _BLOCK, chunk.shape, chunk.dtype, payload)
    scatter_cpu_to_paged_kv(reference, ids, [dense], blocks)
    torch_dev.synchronize()

    target, pristine = blank(), blank()
    paged = prepare_paged_mamba_source(target)
    assert paged is not None
    codec.deserialize_mamba_paged(
        payload, paged.layer_ptrs, paged_block_ids_tensor(ids, paged.device), paged.num_blocks,
        paged.page_stride_bytes, layout, skip_blocks,
    )
    torch_dev.synchronize()

    for index, block in enumerate(ids):
        got = _substate_bytes(target, layout, block)
        want = _substate_bytes(pristine if index < skip_blocks else reference, layout, block)
        assert all(torch.equal(a, b) for a, b in zip(got, want)), (
            "a skipped block was written" if index < skip_blocks else "paged decode differs from dense decode + scatter"
        )
    # everything outside the written sub-states (padding, other blocks) is untouched
    for name in target:
        got = target[name].view(torch.uint8).view(_NUM_BLOCKS, -1).cpu()
        want = pristine[name].view(torch.uint8).view(_NUM_BLOCKS, -1).cpu()
        for block in range(_NUM_BLOCKS):
            if block in ids[skip_blocks:]:
                mask = torch.ones(got.shape[1], dtype=torch.bool)
                for sub in layout:
                    mask[sub.byte_offset : sub.byte_offset + sub.byte_length] = False
                assert torch.equal(got[block][mask], want[block][mask])
            else:
                assert torch.equal(got[block], want[block])
