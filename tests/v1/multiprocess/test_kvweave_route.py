# SPDX-License-Identifier: Apache-2.0
"""Decision table for the store/retrieve routing levels."""

# Standard
from dataclasses import replace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave.kvweave_config import AttentionPlaneLayout
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.kvweave_route import (
    GroupKind,
    QuantDevice,
    RetrieveRoute,
    StoreRoute,
    decide_retrieve_route,
    decide_store_route,
    group_kind,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import GroupTransferPlan

_SHAPE = torch.Size([2, 16, 64])


def _plan(kind: GroupKind, quantized: bool) -> GroupTransferPlan:
    category = "mamba" if kind is GroupKind.MAMBA else "attention"
    layout = {
        GroupKind.FUSED_ATTENTION: AttentionPlaneLayout.FUSED_KV,
        GroupKind.OTHER: AttentionPlaneLayout.SPLIT_KV,
        GroupKind.MAMBA: None,
    }[kind]
    return GroupTransferPlan(
        group_info=EngineGroupInfo(engine_group_id=0, cache_category=category),
        layer_indices=frozenset(),
        blocks_per_chunk=1,
        chunk_shape=_SHAPE,
        engine_kv_format=None,
        attention_plane_layout=layout,
        quantized=quantized,
        raw_layout_desc=MemoryLayoutDesc(shapes=[_SHAPE], dtypes=[torch.bfloat16]),
    )


ALL_KINDS = list(GroupKind)


@pytest.mark.parametrize("device", list(QuantDevice))
@pytest.mark.parametrize("kind", ALL_KINDS)
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("paged_ok", [False, True])
def test_master_switch_off_is_baseline_for_every_group(
    device: QuantDevice, kind: GroupKind, quantized: bool, paged_ok: bool
) -> None:
    plan = _plan(kind, quantized)
    kwargs = {"master_on": False, "device": device, "plan": plan, "paged_ok": paged_ok}
    assert decide_store_route(**kwargs) is StoreRoute.DIRECT
    assert decide_retrieve_route(**kwargs) is RetrieveRoute.SCATTER


@pytest.mark.parametrize(
    ("kind", "quantized", "paged_ok", "expected"),
    [
        (GroupKind.MAMBA, False, False, StoreRoute.STAGED),
        (GroupKind.MAMBA, True, False, StoreRoute.STAGED),
        (GroupKind.MAMBA, True, True, StoreRoute.PAGED),
        (GroupKind.MAMBA, False, True, StoreRoute.STAGED),  # raw Mamba is never encoded
        (GroupKind.FUSED_ATTENTION, True, True, StoreRoute.PAGED),
        (GroupKind.FUSED_ATTENTION, True, False, StoreRoute.STAGED),
        (GroupKind.FUSED_ATTENTION, False, True, StoreRoute.DIRECT),
        (GroupKind.OTHER, True, True, StoreRoute.STAGED),
        (GroupKind.OTHER, False, False, StoreRoute.DIRECT),
    ],
)
def test_store_route_xpu(kind, quantized, paged_ok, expected) -> None:
    got = decide_store_route(
        master_on=True,
        device=QuantDevice.XPU,
        plan=_plan(kind, quantized),
        paged_ok=paged_ok,
    )
    assert got is expected


@pytest.mark.parametrize("paged_ok", [False, True])
@pytest.mark.parametrize(
    ("kind", "quantized", "expected"),
    [
        (GroupKind.MAMBA, False, StoreRoute.STAGED),
        (GroupKind.MAMBA, True, StoreRoute.STAGED),
        (GroupKind.FUSED_ATTENTION, True, StoreRoute.STAGED),  # never PAGED on CPU
        (GroupKind.OTHER, True, StoreRoute.STAGED),
        (GroupKind.FUSED_ATTENTION, False, StoreRoute.DIRECT),
        (GroupKind.OTHER, False, StoreRoute.DIRECT),
    ],
)
def test_store_route_cpu_never_paged(kind, quantized, expected, paged_ok) -> None:
    got = decide_store_route(
        master_on=True,
        device=QuantDevice.CPU,
        plan=_plan(kind, quantized),
        paged_ok=paged_ok,
    )
    assert got is expected


@pytest.mark.parametrize(
    ("kind", "quantized", "paged_ok", "expected"),
    [
        (GroupKind.MAMBA, False, False, RetrieveRoute.STAGED_RAW),
        (GroupKind.MAMBA, True, False, RetrieveRoute.SCATTER),
        (GroupKind.MAMBA, True, True, RetrieveRoute.PAGED),
        (GroupKind.MAMBA, False, True, RetrieveRoute.STAGED_RAW),  # raw state, no codec
        (GroupKind.FUSED_ATTENTION, True, True, RetrieveRoute.PAGED),
        (GroupKind.FUSED_ATTENTION, True, False, RetrieveRoute.SCATTER),
        (GroupKind.FUSED_ATTENTION, False, True, RetrieveRoute.SCATTER),
        (GroupKind.OTHER, True, True, RetrieveRoute.SCATTER),
    ],
)
def test_retrieve_route_xpu(kind, quantized, paged_ok, expected) -> None:
    got = decide_retrieve_route(
        master_on=True,
        device=QuantDevice.XPU,
        plan=_plan(kind, quantized),
        paged_ok=paged_ok,
    )
    assert got is expected


@pytest.mark.parametrize("paged_ok", [False, True])
def test_retrieve_route_cpu_never_paged(paged_ok: bool) -> None:
    for kind in ALL_KINDS:
        for quantized in (False, True):
            got = decide_retrieve_route(
                master_on=True,
                device=QuantDevice.CPU,
                plan=_plan(kind, quantized),
                paged_ok=paged_ok,
            )
            raw_mamba = kind is GroupKind.MAMBA and not quantized
            assert got is (
                RetrieveRoute.STAGED_RAW if raw_mamba else RetrieveRoute.SCATTER
            )


def test_group_kind_classification() -> None:
    assert group_kind(_plan(GroupKind.MAMBA, False)) is GroupKind.MAMBA
    assert group_kind(_plan(GroupKind.FUSED_ATTENTION, True)) is GroupKind.FUSED_ATTENTION
    assert group_kind(_plan(GroupKind.OTHER, True)) is GroupKind.OTHER
    unclassified = replace(_plan(GroupKind.OTHER, False), group_info=None)
    assert group_kind(unclassified) is GroupKind.OTHER
