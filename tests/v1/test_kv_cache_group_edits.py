# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.kv_cache_group_edits import _MambaPageViewEdit


@dataclass
class FakeMambaSpec:
    shapes: tuple
    dtypes: tuple
    page_size_bytes: int


def _qwen35_0_8b_spec(*, page_size_bytes: "int | None" = None) -> FakeMambaSpec:
    """Real Qwen3.5-0.8B conv_state/ssm_state shapes and dtypes.

    See ``kvweave/mamba_conv_ssm_layout_params.md``: conv_state is
    ``[3, 6144]`` fp16, ssm_state is ``[16, 128, 128]`` fp32.
    """
    conv_shape, ssm_shape = (3, 6144), (16, 128, 128)
    conv_bytes = 3 * 6144 * 2
    ssm_bytes = 16 * 128 * 128 * 4
    return FakeMambaSpec(
        shapes=(conv_shape, ssm_shape),
        dtypes=(torch.float16, torch.float32),
        page_size_bytes=page_size_bytes
        if page_size_bytes is not None
        else conv_bytes + ssm_bytes,
    )


def test_real_layout_splits_conv_and_ssm_by_own_dtype():
    """conv and ssm keep their own dtype/shape even though the page-view
    edit's ``apply()`` would read the whole page as one (conv's) dtype."""
    edit = _MambaPageViewEdit()
    layout = edit.real_layout(_qwen35_0_8b_spec())

    assert layout.conv.byte_offset == 0
    assert layout.conv.byte_length == 3 * 6144 * 2
    assert layout.conv.dtype == torch.float16
    assert layout.conv.shape == (3, 6144)

    assert layout.ssm.byte_offset == layout.conv.byte_length
    assert layout.ssm.byte_length == 16 * 128 * 128 * 4
    assert layout.ssm.dtype == torch.float32
    assert layout.ssm.shape == (16, 128, 128)


def test_real_layout_pad_covers_remaining_page_bytes():
    """conv + ssm + pad must sum exactly to the page size."""
    edit = _MambaPageViewEdit()
    spec = _qwen35_0_8b_spec(page_size_bytes=3 * 6144 * 2 + 16 * 128 * 128 * 4 + 64)
    layout = edit.real_layout(spec)

    assert layout.pad_byte_offset == layout.ssm.byte_offset + layout.ssm.byte_length
    assert layout.pad_byte_length == 64
    assert (
        layout.conv.byte_length + layout.ssm.byte_length + layout.pad_byte_length
        == spec.page_size_bytes
    )


def test_real_layout_page_aligned_has_no_pad():
    edit = _MambaPageViewEdit()
    layout = edit.real_layout(_qwen35_0_8b_spec())

    assert layout.pad_byte_length == 0


def test_real_layout_rejects_wrong_substate_count():
    edit = _MambaPageViewEdit()
    bad_spec = FakeMambaSpec(
        shapes=((3, 6144),), dtypes=(torch.float16,), page_size_bytes=100
    )

    with pytest.raises(ValueError, match="exactly 2 sub-states"):
        edit.real_layout(bad_spec)


def test_real_layout_rejects_bytes_exceeding_page_size():
    edit = _MambaPageViewEdit()
    bad_spec = _qwen35_0_8b_spec(page_size_bytes=10)

    with pytest.raises(ValueError, match="exceed the page size"):
        edit.real_layout(bad_spec)
