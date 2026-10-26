# SPDX-License-Identifier: Apache-2.0
"""Store / retrieve routing for the engine-driven transfer contexts.

A group's path is decided in three ordered levels, each only consulted when the
previous one allows it:

1. **Master switch** (``LMCACHE_MP_L1_KVWEAVE_QUANT``). Off: every group, of
   any type, takes the baseline direct gather/scatter -- the code as it was
   before KVWeave existed.
2. **Quantization device** (``LMCACHE_MP_KVWEAVE_QUANT_DEVICE``). XPU and CPU
   follow different routes: only XPU can quantize straight from / dequantize
   straight into paged KV, while CPU relies on its direct codec kernels.
3. **Group type**: Mamba (linear attention), fused-K/V attention, or any other
   attention layout.
4. **That type's quantization switch**: ``LINEAR_QUANT_ENABLED`` for Mamba,
   ``FULL_ATTENTION_QUANT_ENABLED`` plus ``FUSED_`` / ``SPLIT_ATTENTION_QUANT_ENABLED``
   for attention. Resolved once at ``register()`` into ``plan.quantized``, which
   also requires the quantized payload to be smaller than the raw chunk; a group
   whose switch is on but fails that size check is transferred raw.

The functions here are pure so the whole decision table is unit-testable.
"""

# Future
from __future__ import annotations

# Standard
from enum import Enum
from typing import TYPE_CHECKING

# First Party
from lmcache.v1.distributed.serde.kvweave.kvweave_config import AttentionPlaneLayout

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        GroupTransferPlan,
    )


class QuantDevice(str, Enum):
    """Device that runs KVWeave quantization and dequantization."""

    XPU = "xpu"
    CPU = "cpu"


class GroupKind(str, Enum):
    """Coarse group type used for the third routing level."""

    MAMBA = "mamba"
    FUSED_ATTENTION = "fused_attention"
    OTHER = "other"


class StoreRoute(str, Enum):
    """How one group's chunks reach the server on store.

    Attributes:
        DIRECT: Gather straight into the SHM slot (or pinned staging in pickle
            mode); no codec involved.
        STAGED: Gather raw data into scratch, then split/encode it (or copy raw
            Mamba state) into the SHM slot.
        PAGED: Quantize straight from paged KV; only the payload is copied out.
            XPU only.
    """

    DIRECT = "direct"
    STAGED = "staged"
    PAGED = "paged"


class RetrieveRoute(str, Enum):
    """How one group's chunks land in paged KV on retrieve.

    Attributes:
        SCATTER: Decode into scratch if quantized, then scatter raw data.
        STAGED_RAW: Raw Mamba state, staged one scratch chunk at a time with a
            per-chunk skip.
        PAGED: Dequantize straight into paged KV. XPU only.
    """

    SCATTER = "scatter"
    STAGED_RAW = "staged_raw"
    PAGED = "paged"


def group_kind(plan: GroupTransferPlan) -> GroupKind:
    """Classify a group for the third routing level."""
    if plan.group_info is not None and plan.group_info.cache_category == "mamba":
        return GroupKind.MAMBA
    if plan.attention_plane_layout == AttentionPlaneLayout.FUSED_KV:
        return GroupKind.FUSED_ATTENTION
    return GroupKind.OTHER


def decide_store_route(
    *,
    master_on: bool,
    device: QuantDevice,
    plan: GroupTransferPlan,
    paged_ok: bool,
) -> StoreRoute:
    """Route one group's store.

    Args:
        master_on: Whether the KVWeave master switch is on.
        device: Quantization device.
        plan: The group's registration-time plan; ``plan.quantized`` is the
            fourth level (this group type's quantization switch is on).
        paged_ok: Whether the codec can quantize this group straight from paged
            KV. Only consulted on XPU when a Mamba or fused-K/V group's switch is
            on (other attention layouts have no in-place kernel).

    Returns:
        The route for this group.
    """
    if not master_on:
        return StoreRoute.DIRECT
    if device is QuantDevice.XPU:
        return _store_route_xpu(plan, paged_ok)
    return _store_route_cpu(plan)


def _store_route_xpu(plan: GroupTransferPlan, paged_ok: bool) -> StoreRoute:
    kind = group_kind(plan)
    if plan.quantized and paged_ok and kind is not GroupKind.OTHER:
        return StoreRoute.PAGED
    if plan.quantized or kind is GroupKind.MAMBA:
        return StoreRoute.STAGED
    return StoreRoute.DIRECT


def _store_route_cpu(plan: GroupTransferPlan) -> StoreRoute:
    if plan.quantized or group_kind(plan) is GroupKind.MAMBA:
        return StoreRoute.STAGED
    return StoreRoute.DIRECT


def decide_retrieve_route(
    *,
    master_on: bool,
    device: QuantDevice,
    plan: GroupTransferPlan,
    paged_ok: bool,
) -> RetrieveRoute:
    """Route one group's retrieve.

    Args:
        master_on: Whether the KVWeave master switch is on.
        device: Quantization device.
        plan: The group's registration-time plan; ``plan.quantized`` is the
            fourth level (this group type's quantization switch is on).
        paged_ok: Whether the codec can dequantize this group straight into
            paged KV (and nothing else rules it out, e.g. a null-chunk mask).
            Only consulted on XPU when a Mamba or fused-K/V group's switch is on
            (other attention layouts have no in-place kernel).

    Returns:
        The route for this group.
    """
    if not master_on:
        return RetrieveRoute.SCATTER
    if device is QuantDevice.XPU:
        return _retrieve_route_xpu(plan, paged_ok)
    return _retrieve_route_cpu(plan)


def _retrieve_route_xpu(plan: GroupTransferPlan, paged_ok: bool) -> RetrieveRoute:
    kind = group_kind(plan)
    if plan.quantized and paged_ok and kind is not GroupKind.OTHER:
        return RetrieveRoute.PAGED
    if kind is GroupKind.MAMBA and not plan.quantized:
        return RetrieveRoute.STAGED_RAW
    return RetrieveRoute.SCATTER


def _retrieve_route_cpu(plan: GroupTransferPlan) -> RetrieveRoute:
    if group_kind(plan) is GroupKind.MAMBA and not plan.quantized:
        return RetrieveRoute.STAGED_RAW
    return RetrieveRoute.SCATTER
