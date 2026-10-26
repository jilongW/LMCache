# SPDX-License-Identifier: Apache-2.0
"""Retrieve-side in-place (paged) decode of quantized fused-K/V groups."""

# Standard
from dataclasses import replace
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.serde.kvweave.kvweave_config import AttentionPlaneLayout
from lmcache.v1.multiprocess.group_view import EngineGroupInfo, MambaSubStateWireLayout
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.base import PagedFusedSource, PagedMambaSource
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
    GroupTransferPlan,
)
from lmcache.v1.distributed.api import MemoryLayoutDesc

_BLOCK_SIZE = 16


def _plan(blocks_per_chunk: int = 4, quantized: bool = True) -> GroupTransferPlan:
    shape = torch.Size([2, blocks_per_chunk * _BLOCK_SIZE, 64])
    return GroupTransferPlan(
        group_info=EngineGroupInfo(engine_group_id=0, cache_category="attention"),
        layer_indices=frozenset(),
        blocks_per_chunk=blocks_per_chunk,
        chunk_shape=shape,
        engine_kv_format=None,
        attention_plane_layout=AttentionPlaneLayout.FUSED_KV,
        quantized=quantized,
        raw_layout_desc=MemoryLayoutDesc(shapes=[shape], dtypes=[torch.bfloat16]),
    )


def _source() -> PagedFusedSource:
    return PagedFusedSource(
        layer_ptrs=torch.zeros(2, dtype=torch.int64),
        block_size=_BLOCK_SIZE,
        block_stride_elems=_BLOCK_SIZE * 64,
        num_blocks=32,
        dtype=torch.bfloat16,
        hidden=64,
        device=torch.device("cpu"),
    )


class _Codec:
    def __init__(self, supported: bool = True) -> None:
        self.supported = supported
        self.calls: list[dict[str, Any]] = []

    def supports_paged_fused_decode(self, hidden: int) -> bool:
        return self.supported and hidden == 64

    def deserialize_fused_paged(
        self,
        src: torch.Tensor,
        layer_ptrs: torch.Tensor,
        block_ids: torch.Tensor,
        block_size: int,
        block_stride_elems: int,
        num_blocks: int,
        skip_blocks: int,
        dtype: torch.dtype,
        hidden: int,
    ) -> None:
        self.calls.append(
            {"src": src, "block_ids": block_ids.tolist(), "skip_blocks": skip_blocks}
        )


@pytest.fixture
def ctx(monkeypatch: pytest.MonkeyPatch) -> EngineDrivenTransferContext:
    monkeypatch.setattr(
        worker_transfer,
        "paged_block_ids_tensor",
        lambda ids, _device: torch.tensor(ids, dtype=torch.int64),
    )
    context = EngineDrivenTransferContext()
    context._kvweave_codec = _Codec()  # type: ignore[assignment]
    yield context
    context.close()


def test_each_chunk_gets_its_own_block_ids_and_no_skip(ctx) -> None:
    chunks = [torch.zeros(8, dtype=torch.uint8), torch.ones(8, dtype=torch.uint8)]
    ctx._decode_group_into_paged_kv(  # noqa: SLF001
        _plan(), _source(), [3, 5, 7, 9, 11, 13, 15, 17], chunks, 0
    )
    calls = ctx._kvweave_codec.calls  # type: ignore[union-attr]
    assert [c["block_ids"] for c in calls] == [[3, 5, 7, 9], [11, 13, 15, 17]]
    assert [c["skip_blocks"] for c in calls] == [0, 0]
    assert calls[0]["src"] is chunks[0] and calls[1]["src"] is chunks[1]


@pytest.mark.parametrize(
    ("skip_blocks_total", "expected_calls"),
    [
        (2, [(0, 2), (1, 0)]),
        (4, [(1, 0)]),  # whole first chunk skipped: no native call for it
        (6, [(1, 2)]),
        (8, []),
    ],
)
def test_global_prefix_skip_is_split_across_chunks(
    ctx, skip_blocks_total: int, expected_calls: list[tuple[int, int]]
) -> None:
    chunks = [torch.zeros(8, dtype=torch.uint8), torch.ones(8, dtype=torch.uint8)]
    ctx._decode_group_into_paged_kv(  # noqa: SLF001
        _plan(),
        _source(),
        list(range(8)),
        chunks,
        skip_blocks_total * _BLOCK_SIZE,
    )
    calls = ctx._kvweave_codec.calls  # type: ignore[union-attr]
    got = [(0 if c["src"] is chunks[0] else 1, c["skip_blocks"]) for c in calls]
    assert got == expected_calls


def test_out_of_range_block_id_is_rejected_before_any_native_call(ctx) -> None:
    with pytest.raises(ValueError, match="outside"):
        ctx._decode_group_into_paged_kv(  # noqa: SLF001
            _plan(), _source(), [0, 1, 2, 99], [torch.zeros(8, dtype=torch.uint8)], 0
        )
    assert ctx._kvweave_codec.calls == []  # type: ignore[union-attr]


def test_short_block_list_is_rejected(ctx) -> None:
    with pytest.raises(ValueError, match="one block list per"):
        ctx._decode_group_into_paged_kv(  # noqa: SLF001
            _plan(), _source(), [0, 1], [torch.zeros(8, dtype=torch.uint8)], 0
        )


@pytest.mark.parametrize(
    ("plan_edit", "supported", "expect_source"),
    [
        ({}, True, True),
        ({}, False, False),
        ({"quantized": False}, True, False),
        ({"attention_plane_layout": AttentionPlaneLayout.SPLIT_KV}, True, False),
    ],
)
def test_eligibility(
    ctx, monkeypatch: pytest.MonkeyPatch, plan_edit, supported, expect_source
) -> None:
    ctx._kvweave_codec = _Codec(supported)  # type: ignore[assignment]
    monkeypatch.setattr(
        worker_transfer, "prepare_paged_fused_source", lambda *_a, **_k: _source()
    )
    plan = replace(_plan(), **plan_edit)
    source = ctx._paged_fused_decode_source(plan, {"k": torch.zeros(1)})  # noqa: SLF001
    assert (source is not None) is expect_source


_MAMBA_LAYOUT = (
    MambaSubStateWireLayout(0, 8, "torch.float16", (2, 2)),
    MambaSubStateWireLayout(8, 16, "torch.float32", (2, 2)),
)


def _mamba_plan(blocks_per_chunk: int = 2, quantized: bool = True) -> GroupTransferPlan:
    return replace(
        _plan(blocks_per_chunk, quantized),
        attention_plane_layout=None,
        group_info=EngineGroupInfo(
            engine_group_id=0,
            cache_category="mamba",
            tokens_per_block=_BLOCK_SIZE,
            mamba_real_layout=_MAMBA_LAYOUT,
        ),
        mamba_options=object(),
    )


def _mamba_source() -> PagedMambaSource:
    return PagedMambaSource(
        layer_ptrs=torch.zeros(2, dtype=torch.int64),
        page_stride_bytes=4096,
        page_bytes=1024,
        num_blocks=32,
        block_size=_BLOCK_SIZE,
        device=torch.device("cpu"),
    )


class _MambaCodec:
    def __init__(self, supported: bool = True) -> None:
        self.supported = supported
        self.calls: list[dict[str, Any]] = []

    def supports_paged_mamba_decode(self, layout, options, blocks: int) -> bool:
        return self.supported and layout is _MAMBA_LAYOUT and blocks == 2

    def deserialize_mamba_paged(
        self,
        src: torch.Tensor,
        layer_ptrs: torch.Tensor,
        block_ids: torch.Tensor,
        num_blocks: int,
        page_stride_bytes: int,
        layout: object,
        skip_blocks: int,
    ) -> None:
        self.calls.append(
            {
                "src": src,
                "block_ids": block_ids.tolist(),
                "skip_blocks": skip_blocks,
                "layout": layout,
                "page_stride_bytes": page_stride_bytes,
            }
        )


def test_mamba_chunks_decode_in_place_with_per_chunk_skip(ctx) -> None:
    ctx._kvweave_codec = _MambaCodec()  # type: ignore[assignment]
    chunks = [torch.zeros(8, dtype=torch.uint8), torch.ones(8, dtype=torch.uint8)]
    ctx._decode_group_into_paged_kv(  # noqa: SLF001
        _mamba_plan(), _mamba_source(), [3, 5, 7, 9], chunks, 3 * _BLOCK_SIZE
    )
    calls = ctx._kvweave_codec.calls  # type: ignore[union-attr]
    # The first chunk (2 blocks) lies wholly inside the 3-block skipped prefix, so
    # it is not decoded at all; the second skips only its first block.
    assert [c["block_ids"] for c in calls] == [[7, 9]]
    assert [c["skip_blocks"] for c in calls] == [1]
    assert all(c["layout"] is _MAMBA_LAYOUT and c["page_stride_bytes"] == 4096 for c in calls)


@pytest.mark.parametrize(
    ("quantized", "supported", "expect_source"),
    [(True, True, True), (True, False, False), (False, True, False)],
)
def test_mamba_decode_eligibility(
    ctx, monkeypatch: pytest.MonkeyPatch, quantized, supported, expect_source
) -> None:
    ctx._kvweave_codec = _MambaCodec(supported)  # type: ignore[assignment]
    monkeypatch.setattr(
        worker_transfer, "prepare_paged_mamba_source", lambda *_a, **_k: _mamba_source()
    )
    source = ctx._paged_decode_source(  # noqa: SLF001
        _mamba_plan(quantized=quantized), {"k": torch.zeros(1)}
    )
    assert (source is not None) is expect_source


def test_mamba_page_layout_must_fit_the_page(ctx, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx._kvweave_codec = _MambaCodec()  # type: ignore[assignment]
    small = replace(_mamba_source(), page_bytes=16)  # ssm ends at byte 24
    monkeypatch.setattr(
        worker_transfer, "prepare_paged_mamba_source", lambda *_a, **_k: small
    )
    assert ctx._paged_decode_source(_mamba_plan(), {"k": torch.zeros(1)}) is None  # noqa: SLF001
