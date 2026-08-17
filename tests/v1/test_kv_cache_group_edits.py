# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass

# Third Party
from vllm.v1.kv_cache_interface import MambaSpec
import pytest
import torch

# First Party
from lmcache.integration.vllm.kv_cache_group_edits import _MambaPageViewEdit


@dataclass
class _InconsistentMambaSpec:
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    page_size_bytes: int


def test_real_layout_reports_conv_and_ssm_byte_extents():
    spec = MambaSpec(
        shapes=((4, 8), (2, 3, 5)),
        dtypes=(torch.float16, torch.float32),
        block_size=1,
    )

    layout = _MambaPageViewEdit().real_layout(spec)

    conv_bytes = 4 * 8 * torch.empty((), dtype=torch.float16).element_size()
    ssm_bytes = 2 * 3 * 5 * torch.empty((), dtype=torch.float32).element_size()
    assert layout.conv.byte_offset == 0
    assert layout.conv.byte_length == conv_bytes
    assert layout.conv.dtype == torch.float16
    assert layout.conv.shape == (4, 8)
    assert layout.ssm.byte_offset == conv_bytes
    assert layout.ssm.byte_length == ssm_bytes
    assert layout.ssm.dtype == torch.float32
    assert layout.ssm.shape == (2, 3, 5)
    assert layout.page_size_bytes == spec.page_size_bytes


def test_real_layout_reports_trailing_padding():
    spec = MambaSpec(
        shapes=((4, 8), (2, 3, 5)),
        dtypes=(torch.float16, torch.float32),
        block_size=1,
        page_size_padded=256,
    )

    layout = _MambaPageViewEdit().real_layout(spec)

    assert layout.pad_byte_offset == layout.ssm.byte_offset + layout.ssm.byte_length
    assert layout.pad_byte_length == 256 - layout.pad_byte_offset


def test_real_layout_rejects_wrong_number_of_substates():
    spec = MambaSpec(
        shapes=((4, 8),),
        dtypes=(torch.float16,),
        block_size=1,
    )

    with pytest.raises(ValueError, match="exactly 2 sub-states"):
        _MambaPageViewEdit().real_layout(spec)


def test_real_layout_rejects_substates_larger_than_page():
    spec = _InconsistentMambaSpec(
        shapes=((4, 8), (2, 3, 5)),
        dtypes=(torch.float16, torch.float32),
        page_size_bytes=16,
    )

    with pytest.raises(ValueError, match="exceed the page size"):
        _MambaPageViewEdit().real_layout(spec)
