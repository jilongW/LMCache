# SPDX-License-Identifier: Apache-2.0
"""Per-group KVWeave planning for the engine-driven worker.

Resolved once at ``register()`` time: for every LMCache group, whether its
chunks are KVWeave-quantized on store and decoded on retrieve, with which codec
options and which wire layout. Everything here is a pure function of the group's
shape, dtype and metadata -- no device access -- so the decision table is unit
testable without a server or a GPU.

The group's raw shape, dtype, KV format and layer membership come from the
worker's ``KVLayerGroupsManager`` kernel groups (see
``EngineDrivenTransferContext.register``); the Mamba metadata comes from the
matching ``EngineGroupInfo`` (kernel group ``i`` corresponds to
``engine_group_infos[i]``).
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

# Third Party
import msgspec
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    AttentionPlaneLayout,
    KVWeaveCodecConfig,
    KVWeaveRuntimeConfig,
    MambaCodecOptions,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import KVWeaveCodec
from lmcache.v1.gpu_connector.kv_format.specs.registry import get_spec_class
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    MambaSubStateWireLayout,
)

logger = init_logger(__name__)


def select_kv_caches_by_index(
    kv_caches: dict[str, torch.Tensor],
    layer_indices: frozenset[int],
) -> dict[str, torch.Tensor]:
    """Return the subset of ``kv_caches`` at positions in ``layer_indices``.

    Args:
        kv_caches: Worker KV-cache tensors keyed by layer name, in
            registration order.
        layer_indices: Registered KV tensor positions to keep. Empty selects
            every layer (the single-group fallback).

    Returns:
        Only the layers at positions in ``layer_indices``, or ``kv_caches``
        itself when ``layer_indices`` is empty.
    """
    if not layer_indices:
        return kv_caches
    return {
        name: tensor
        for idx, (name, tensor) in enumerate(kv_caches.items())
        if idx in layer_indices
    }



@dataclass(frozen=True)
class GroupTransferPlan:
    """One LMCache group's registration-time constants for gather / scatter.

    Resolved once at ``register`` time instead of recomputed per transfer,
    which was ``O(num_groups * num_layers^2)`` plus a format-detection call
    per group. KV tensors are deliberately not cached here, since retrieve
    may scatter into a different mapping than the one registered --
    :meth:`select_kv_caches` applies the cached layer selection to whatever
    mapping the caller passes.

    Attributes:
        group_info: The LMCache group this plan describes, or ``None`` for
            the single-group (non-hybrid) fallback.
        layer_indices: Registered KV tensor positions this group owns, as a
            set for O(1) membership. Empty selects every layer (the
            single-group fallback).
        blocks_per_chunk: Paged blocks of this group's own block size needed
            to cover one LMCache chunk.
        chunk_shape: One chunk's tensor shape for this group's layer count.
        engine_kv_format: This group's pre-detected KV format, passed to
            gather / scatter so they skip re-detection. ``None`` when it
            could not be resolved up front, which makes them detect it per
            transfer as before.
        attention_plane_layout: Classified attention K/V object-plane layout
            for this group, or ``None`` for non-attention groups.
        quantized: Whether ``register()`` decided to KVWeave-quantize this
            group's chunks on store and decode them on retrieve. ``False``
            for every group when quantization is disabled or was not
            worthwhile for this group (see ``_build_group_transfer_plans``).
        raw_layout_desc: This group's un-quantized chunk layout (matches
            ``chunk_shape``/the KV cache dtype). Always populated -- used as
            the ``decode_chunk`` target shape/dtype regardless of whether
            this group is quantized, and sent to the server to size SHM
            slots when this group is not quantized.
        quant_layout_desc: This group's quantized wire layout (``uint8``,
            sized by ``estimate_serialized_size``/``estimate_mamba_serialized_size``).
            ``None`` unless ``quantized`` is ``True``. Sent to the server as
            this group's ``group_layout_descs`` override so its SHM chunks
            are allocated for the quantized payload instead of the raw KV
            tensor.
        mamba_options: This group's resolved Mamba conv/ssm quantization
            options, when ``quantized`` is ``True`` and the group's
            ``cache_category`` is ``"mamba"``. ``None`` otherwise.
    """

    group_info: EngineGroupInfo | None
    layer_indices: frozenset[int]
    blocks_per_chunk: int
    chunk_shape: torch.Size
    engine_kv_format: Any
    attention_plane_layout: AttentionPlaneLayout | None = None
    quantized: bool = False
    raw_layout_desc: MemoryLayoutDesc | None = None
    quant_layout_desc: MemoryLayoutDesc | None = None
    mamba_options: MambaCodecOptions | None = None

    def select_kv_caches(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return this group's subset of ``kv_caches``.

        Args:
            kv_caches: Worker KV-cache tensors keyed by layer name, in the
                same registration order the plan was built from. Retrieve
                may pass a different mapping than ``register`` did, so the
                selection is applied to the argument rather than to a cached
                subset.

        Returns:
            Only the layers this group owns, or ``kv_caches`` itself for the
            single-group fallback.
        """
        return select_kv_caches_by_index(kv_caches, self.layer_indices)



def _attention_plane_layout_for_group(
    group_info: EngineGroupInfo | None,
    engine_kv_format: Any,
) -> AttentionPlaneLayout | None:
    """Classify an attention group's physical K/V plane layout.

    Returns ``None`` for non-attention groups. Attention groups with an
    undetected or unknown format are classified as ``UNKNOWN`` so quantized
    dispatch never guesses from tensor shape.
    """
    if group_info is None or group_info.cache_category != "attention":
        return None
    if engine_kv_format is None:
        return AttentionPlaneLayout.UNKNOWN
    try:
        spec = get_spec_class(engine_kv_format)
    except ValueError:
        logger.warning(
            "Could not classify attention plane layout for KV format %s; "
            "falling back to unquantized transfer for this group",
            engine_kv_format,
            exc_info=True,
        )
        return AttentionPlaneLayout.UNKNOWN
    if spec.is_mla:
        return AttentionPlaneLayout.MLA
    if spec.is_fused_packed:
        return AttentionPlaneLayout.FUSED_KV
    return AttentionPlaneLayout.SPLIT_KV



def _raw_chunk_byte_size(shape: torch.Size, dtype: torch.dtype) -> int:
    """Return the byte size of one un-quantized chunk tensor.

    Args:
        shape: The chunk's tensor shape.
        dtype: The chunk's tensor dtype.

    Returns:
        ``numel(shape) * itemsize(dtype)``.
    """
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * torch.empty((), dtype=dtype).element_size()


def _mamba_rh_transform_length(
    substate: str,
    sub_layout: MambaSubStateWireLayout,
    layers: int,
    blocks: int,
    scaling_method: str,
) -> int:
    """Compute the RH (randomized Hadamard) transform length for one Mamba
    sub-state, without needing the real tensor's data.

    Mirrors ``_KVWeaveCodec._quantize_mamba_substate_payload``'s
    ``max(cpu.numel() // shape[0] // chunks, 1)``: ``cpu.numel() // shape[0]``
    is ``blocks * prod(sub_layout.shape)`` (one layer's worth of elements),
    and ``chunks`` comes from the same ``mamba_layout()`` grouping the actual
    quantization call uses -- computable purely from shape/scaling_method,
    which is already known at registration time.

    Args:
        substate: ``"conv"`` or ``"ssm"``.
        sub_layout: This sub-state's real per-block byte layout.
        layers: Number of layers in this group.
        blocks: Number of paged blocks in one chunk.
        scaling_method: This sub-state's configured scaling method.

    Returns:
        The transform length ``mamba_precond_tensors()`` would be called
        with for this sub-state.
    """
    numel_per_layer = blocks
    for dim in sub_layout.shape:
        numel_per_layer *= int(dim)
    shape = (layers, blocks, *sub_layout.shape)
    _, _, _, chunks = KVWeaveCodecConfig.mamba_layout(substate, shape, scaling_method)
    return max(numel_per_layer // chunks, 1)


def _resolve_mamba_options_for_group(
    group_info: EngineGroupInfo,
    chunk_shape: torch.Size,
    block_size: int,
    mamba_options: MambaCodecOptions,
) -> MambaCodecOptions:
    """Downgrade ``conv_rh``/``ssm_rh`` to ``False`` when this group's real
    shape can't satisfy RH's power-of-2 transform-length requirement.

    ``mamba_precond_tensors()`` hard-``raise``s on a non-power-of-2 length
    (see ``KVWeaveCodecConfig.mamba_precond_tensors``); with ``rh`` now
    defaulting to ``True`` (see ``env_vars.md``), a model whose conv/ssm
    shapes don't happen to produce a power-of-2 transform length would fail
    every store. Checking here, once at registration time when the real
    ``mamba_real_layout`` is known, turns that into a safe per-group,
    per-substate downgrade with a log line instead of a runtime crash.

    Args:
        group_info: The Mamba group being decided (must have
            ``mamba_real_layout`` set by the caller).
        chunk_shape: This group's raw chunk shape (``[2,L,T,H]`` or ``[L,T,H]``).
        block_size: This group's tokens-per-block.
        mamba_options: The environment-resolved options to downgrade from.

    Returns:
        ``mamba_options``, or a copy with ``conv_rh``/``ssm_rh`` cleared for
        whichever sub-state(s) fail the power-of-2 check.
    """
    assert group_info.mamba_real_layout is not None
    conv_layout, ssm_layout = group_info.mamba_real_layout
    tokens = int(chunk_shape[-2])
    layers = int(chunk_shape[1]) if len(chunk_shape) == 4 else int(chunk_shape[0])
    blocks = max(tokens // block_size, 1)

    updates: dict[str, bool] = {}

    if mamba_options.conv_rh:
        # conv_state is quantized as three independent query/key/value
        # sub-tensors (see kvweave_serde.py's _split_conv_qkv), each with
        # its own RH transform length -- check all three, since encode_chunk
        # applies the same conv_rh flag to every sub-tensor and any one of
        # them failing the power-of-2 check would crash that sub-tensor's
        # quantize call.
        split = mamba_options.conv_qkv_split
        conv_dim = conv_layout.shape[-1]
        expected_conv_dim = split.key_dim * 2 + split.value_dim
        if conv_dim != expected_conv_dim:
            raise ValueError(
                f"conv_state last dim ({conv_dim}) does not match "
                f"conv_qkv_split's key_dim*2 + value_dim ({expected_conv_dim}) "
                f"for group (engine_group_id={group_info.engine_group_id}); "
                "MODEL_PATH/MODEL's config.json disagrees with the real "
                "registered conv_state layout"
            )
        for sub_name, sub_dim in (
            ("conv.query", split.key_dim),
            ("conv.key", split.key_dim),
            ("conv.value", split.value_dim),
        ):
            sub_layout = msgspec.structs.replace(
                conv_layout, shape=(*conv_layout.shape[:-1], sub_dim)
            )
            length = _mamba_rh_transform_length(
                "conv", sub_layout, layers, blocks, mamba_options.conv_scaling_method
            )
            if length <= 0 or length & (length - 1):
                logger.info(
                    "Disabling conv RH for group (engine_group_id=%s): %s "
                    "transform length %d is not a power of 2 "
                    "(scaling_method=%s, conv_dim=%d)",
                    group_info.engine_group_id,
                    sub_name,
                    length,
                    mamba_options.conv_scaling_method,
                    conv_dim,
                )
                updates["conv_rh"] = False
                break

    if mamba_options.ssm_rh:
        length = _mamba_rh_transform_length(
            "ssm", ssm_layout, layers, blocks, mamba_options.ssm_scaling_method
        )
        if length <= 0 or length & (length - 1):
            logger.info(
                "Disabling ssm RH for group (engine_group_id=%s): transform "
                "length %d is not a power of 2 (scaling_method=%s)",
                group_info.engine_group_id,
                length,
                mamba_options.ssm_scaling_method,
            )
            updates["ssm_rh"] = False

    if not updates:
        return mamba_options
    return replace(mamba_options, **updates)


def _decide_group_quantization(
    group_info: EngineGroupInfo | None,
    chunk_shape: torch.Size,
    dtype: torch.dtype,
    block_size: int,
    attention_plane_layout: AttentionPlaneLayout | None,
    kvweave_config: KVWeaveRuntimeConfig,
    codec: KVWeaveCodec,
) -> tuple[bool, MemoryLayoutDesc | None, MambaCodecOptions | None]:
    """Decide whether to KVWeave-quantize one group's chunks.

    Attention groups are quantized when
    ``full_attention_quant_enabled`` is set and the codec's conservative size
    estimate is smaller than the raw chunk; MLA is excluded (single-plane
    formats are not supported by the attention codec's fused K/V layout).
    Mamba groups are quantized when the estimate does not exceed
    ``linear_max_size_ratio`` times the raw size; a group missing
    ``mamba_real_layout`` safely falls back to unquantized transfer rather
    than failing registration. Groups with ``cache_category == "unknown"``
    are never quantized (see MIGRATION_PLAN.md R1/R6).

    Args:
        group_info: The LMCache group being decided, or ``None`` for the
            single-group fallback (treated as an unclassified attention
            group -- never quantized, since it predates ``cache_category``).
        chunk_shape: This group's raw (un-quantized) chunk shape.
        dtype: This group's raw chunk dtype.
        block_size: This group's tokens-per-block, for Mamba size estimation.
        attention_plane_layout: This group's classified attention K/V plane
            layout. ``FUSED_KV`` uses the fused single-tensor codec;
            ``SPLIT_KV`` uses the standard split K/V codec; ``MLA`` and
            ``UNKNOWN`` are never quantized.
        kvweave_config: Resolved runtime configuration (call
            ``KVWeaveRuntimeConfig.from_env()`` once per registration, not
            per group).
        codec: A ``KVWeaveCodec`` instance used only for size estimation
            here; the same instance is reused by ``submit_store``/
            ``submit_retrieve`` for the actual encode/decode.

    Returns:
        ``(quantized, quant_layout_desc, mamba_options)``. ``quant_layout_desc``
        and ``mamba_options`` are ``None`` when ``quantized`` is ``False``.
    """
    if not kvweave_config.enabled:
        return False, None, None
    category = group_info.cache_category if group_info is not None else "unknown"
    raw_layout_desc = MemoryLayoutDesc(shapes=[chunk_shape], dtypes=[dtype])
    raw_size = _raw_chunk_byte_size(chunk_shape, dtype)

    if category == "attention":
        if not kvweave_config.full_attention_quant_enabled:
            return False, None, None
        if dtype == torch.uint8 and not (
            codec.supports_fp8_attention()
            and attention_plane_layout == AttentionPlaneLayout.FUSED_KV
        ):
            # An fp8 KV cache reaches LMCache as uint8 bytes. Only the XPU fused-K/V
            # kernels understand fp8, and they need to know the format
            # (LMCACHE_MP_KVWEAVE_KV_CACHE_DTYPE); quantizing the bytes as numbers
            # would corrupt the cache, so this group is transferred raw instead.
            logger.warning(
                "KVWeave cannot quantize this uint8 (fp8) attention group "
                "(engine_group_id=%s, plane_layout=%s, device=%s, fp8 format=%s); "
                "falling back to unquantized transfer for this group",
                group_info.engine_group_id if group_info is not None else None,
                attention_plane_layout,
                codec.device,
                codec.kv_fp8_dtype,
            )
            return False, None, None
        try:
            if attention_plane_layout == AttentionPlaneLayout.SPLIT_KV:
                if not kvweave_config.split_attention_quant_enabled:
                    return False, None, None
                quant_size = codec.estimate_serialized_size(raw_layout_desc)
            elif attention_plane_layout == AttentionPlaneLayout.FUSED_KV:
                if not kvweave_config.fused_attention_quant_enabled:
                    return False, None, None
                quant_size = codec.estimate_fused_serialized_size(raw_layout_desc)
            else:
                return False, None, None
        except ValueError:
            # The group's classified plane layout (SPLIT_KV/FUSED_KV) does not
            # match the actual chunk_shape rank -- a KV-format misdetection
            # for this model/attention backend. Fall back to unquantized
            # transfer for this group instead of failing engine startup.
            logger.warning(
                "KVWeave size estimate rejected chunk_shape=%s for "
                "attention_plane_layout=%s (engine_group_id=%s); falling "
                "back to unquantized transfer for this group",
                tuple(chunk_shape),
                attention_plane_layout,
                group_info.engine_group_id if group_info is not None else None,
                exc_info=True,
            )
            return False, None, None
        if quant_size < raw_size:
            quant_layout_desc = MemoryLayoutDesc(
                shapes=[torch.Size([quant_size])], dtypes=[torch.uint8]
            )
            return True, quant_layout_desc, None
        return False, None, None

    if category == "mamba" and kvweave_config.linear_quant_enabled:
        mamba_layout = group_info.mamba_real_layout if group_info is not None else None
        if mamba_layout is None:
            logger.warning(
                "Mamba group (engine_group_id=%s) has no mamba_real_layout; "
                "falling back to unquantized transfer for this group",
                group_info.engine_group_id if group_info is not None else None,
            )
            return False, None, None
        try:
            mamba_options = _resolve_mamba_options_for_group(
                cast(EngineGroupInfo, group_info),
                chunk_shape,
                block_size,
                kvweave_config.mamba_options,
            )
            quant_size = KVWeaveCodec.estimate_mamba_serialized_size(
                raw_layout_desc,
                mamba_layout,
                block_size,
                scaling_methods=(
                    mamba_options.conv_scaling_method,
                    mamba_options.ssm_scaling_method,
                ),
                qbits=(mamba_options.conv_qbit, mamba_options.ssm_qbit),
                conv_qkv_split=mamba_options.conv_qkv_split,
                quant_enabled=(
                    mamba_options.conv_quant_enabled,
                    mamba_options.ssm_quant_enabled,
                ),
            )
        except ValueError:
            # The configured conv Q/K/V split does not match the registered
            # conv_state (e.g. config.json disagrees with the real layout, or
            # a TP shard that localized_to_shard could not rescale). Transfer
            # this group raw instead of failing engine startup, like the
            # attention branch above.
            logger.warning(
                "KVWeave Mamba size estimate rejected chunk_shape=%s "
                "(engine_group_id=%s); falling back to unquantized transfer "
                "for this group",
                tuple(chunk_shape),
                group_info.engine_group_id if group_info is not None else None,
                exc_info=True,
            )
            return False, None, None
        if quant_size <= raw_size * kvweave_config.linear_max_size_ratio:
            quant_layout_desc = MemoryLayoutDesc(
                shapes=[torch.Size([quant_size])], dtypes=[torch.uint8]
            )
            return True, quant_layout_desc, mamba_options
        return False, None, None

    # "unknown" (or MLA/unknown attention layout, or "mamba" with linear
    # quantization disabled): never quantized.
    return False, None, None


def build_group_transfer_plan(
    *,
    group_info: EngineGroupInfo | None,
    layer_indices: Sequence[int],
    blocks_per_chunk: int,
    chunk_shape: torch.Size,
    dtype: torch.dtype,
    engine_kv_format: Any,
    block_size: int,
    kvweave_config: KVWeaveRuntimeConfig,
    codec: KVWeaveCodec,
) -> GroupTransferPlan:
    """Build one group's :class:`GroupTransferPlan` from its kernel-group facts.

    Args:
        group_info: The group's ``EngineGroupInfo`` (carries ``cache_category``
            and ``mamba_real_layout``), or ``None`` when the worker registered
            without group infos (never quantized).
        layer_indices: Registered KV tensor positions of this group's layers.
        blocks_per_chunk: Paged blocks of this group per LMCache chunk.
        chunk_shape: The group's raw chunk shape as it is stored (after
            sliding-window trimming), ``[2, L, T, H]`` or ``[L, T, H]``.
        dtype: The group's raw chunk dtype.
        engine_kv_format: The group's detected KV format.
        block_size: The group's tokens per paged block.
        kvweave_config: Resolved KVWeave runtime configuration, read once per
            registration.
        codec: Codec used for size estimation here and reused for the actual
            encode/decode.

    Returns:
        The group's plan, with ``quantized`` and the quantized wire layout
        decided.
    """
    attention_plane_layout = _attention_plane_layout_for_group(
        group_info, engine_kv_format
    )
    quantized, quant_layout_desc, mamba_options = _decide_group_quantization(
        group_info,
        chunk_shape,
        dtype,
        block_size,
        attention_plane_layout,
        kvweave_config,
        codec,
    )
    return GroupTransferPlan(
        group_info=group_info,
        layer_indices=frozenset(layer_indices),
        blocks_per_chunk=blocks_per_chunk,
        chunk_shape=chunk_shape,
        engine_kv_format=engine_kv_format,
        attention_plane_layout=attention_plane_layout,
        quantized=quantized,
        raw_layout_desc=MemoryLayoutDesc(shapes=[chunk_shape], dtypes=[dtype]),
        quant_layout_desc=quant_layout_desc,
        mamba_options=mamba_options,
    )


def local_kvweave_geometry(
    groups: Sequence[tuple[EngineGroupInfo | None, torch.Size, Any]],
) -> tuple[int, int]:
    """Measure this worker's KV widths for ``KVWeaveRuntimeConfig.localized_to_shard``.

    The registered tensors are the only reliable source of the per-worker head
    shard: ``config.json`` describes the whole model, and the ``world_size``
    given to ``register`` is the KV world size (it folds in PP, MLA replication
    and the server count), not the tensor-parallel degree.

    Args:
        groups: One ``(group_info, chunk_shape, engine_kv_format)`` per LMCache
            group, in protocol order.

    Returns:
        ``(attention_plane_hidden, conv_dim)``: the width of one K/V plane of
        the first split- or fused-K/V attention group, and the last dimension of
        the first Mamba group's ``conv_state``. Either is ``0`` when no such
        group exists.
    """
    attention_plane_hidden = 0
    conv_dim = 0
    for group_info, chunk_shape, engine_kv_format in groups:
        if group_info is not None and group_info.cache_category == "mamba":
            if conv_dim == 0 and group_info.mamba_real_layout is not None:
                conv_dim = int(group_info.mamba_real_layout[0].shape[-1])
            continue
        if attention_plane_hidden:
            continue
        plane_layout = _attention_plane_layout_for_group(group_info, engine_kv_format)
        if plane_layout == AttentionPlaneLayout.SPLIT_KV:
            attention_plane_hidden = int(chunk_shape[-1])
        elif plane_layout == AttentionPlaneLayout.FUSED_KV:
            # Fused chunks pack ``[K|V]`` per head on the trailing axis.
            attention_plane_hidden = int(chunk_shape[-1]) // 2
    return attention_plane_hidden, conv_dim
