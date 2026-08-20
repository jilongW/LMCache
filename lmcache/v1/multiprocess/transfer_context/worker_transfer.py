# SPDX-License-Identifier: Apache-2.0
"""Transfer context abstractions for LMCache multiprocess worker adapters."""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any, Callable, Protocol
import os

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.utils import EngineType, init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave import (
    KVWeaveCodec,
    KVWeaveRuntimeConfig,
    MambaCodecOptions,
)
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    RegisterEngineDrivenContextPayload,
    serialize_memory_layout_desc,
)
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo, MambaSubStateWireLayout
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.engine import RegisterEngineDrivenContextResponse
from lmcache.v1.multiprocess.transfer_context.base import (
    EngineDrivenContext,
    EngineDrivenContextMetadata,
    compute_kv_layout,
    create_engine_driven_context,
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.platform import get_device_spec, resolve_kv_wrapper_factory
from lmcache.v1.platform.base.event_ipc import (
    EventIPCBackend,
    get_event_ipc_backend,
)
from lmcache.v1.platform.kv_wrap import wrap_kv_caches

logger = init_logger(__name__)


def _copy_bytes_to_tensor(payload: bytes, destination: torch.Tensor) -> None:
    """Copy a serialized payload into a byte-addressable storage slot."""
    target = destination.view(torch.uint8).flatten()
    if len(payload) > target.numel():
        raise ValueError(
            f"serialized payload ({len(payload)} bytes) exceeds slot "
            f"({target.numel()} bytes)"
        )
    target.zero_()
    target[: len(payload)].copy_(
        torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    )


# Environment variable that lets the user override the default routing
# performed by :func:`create_transfer_context`. Accepted values match the
# string values of :class:`MPTransferMode` (``auto`` / ``engine_driven`` /
# ``lmcache_driven``); ``auto`` reproduces the historical device-type-based
# dispatch.
ENV_MP_TRANSFER_MODE = "LMCACHE_MP_TRANSFER_MODE"


# Helper functions
def _supports_async_primitives() -> bool:
    """Probe whether the worker device supports the async store primitives.

    The async engine-driven store path needs a stream, an event exposing
    ``record``/``synchronize``/``wait``, and pinned (page-locked) host memory.
    When any of these is unavailable (e.g. a CPU-only backend), the factory
    falls back to the synchronous :class:`EngineDrivenTransferContext`. This
    dispatch is internal and capability-based; there is no user-facing
    async/sync flag.

    Returns:
        True if all required async primitives are available, else False.
    """
    if not hasattr(torch_dev, "Stream") or not hasattr(torch_dev, "Event"):
        return False
    # CPU-only stub exposes Stream/Event but has no real async capability.
    if hasattr(torch_dev, "is_available") and not torch_dev.is_available():
        return False
    try:
        stream = torch_dev.Stream()
        event = torch_dev.Event()
    except Exception:
        return False
    for attr in ("record", "synchronize", "wait"):
        if not callable(getattr(event, attr, None)):
            del stream, event
            return False
    del stream, event
    try:
        probe = torch.empty(1, dtype=torch.uint8, device="cpu", pin_memory=True)
        del probe
    except (RuntimeError, TypeError):
        return False
    return True


def _build_engine_driven_context() -> "TransferContext":
    """Build the engine-driven context, async when device-capable else sync.

    Routes the ``ENGINE_DRIVEN`` and AUTO branches through a single capability
    check. ``AsyncEngineDrivenTransferContext`` is imported lazily to avoid an
    import cycle and to keep the synchronous path free of stream/event
    dependencies.

    Returns:
        ``AsyncEngineDrivenTransferContext`` when async primitives are
        available, otherwise ``EngineDrivenTransferContext``.
    """
    if _supports_async_primitives():
        # First Party
        from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
            AsyncEngineDrivenTransferContext,
        )

        logger.info("Using AsyncEngineDrivenTransferContext for store path")
        return AsyncEngineDrivenTransferContext()

    logger.info("Using EngineDrivenTransferContext (sync) for store path")
    return EngineDrivenTransferContext()


class MPTransferMode(str, Enum):
    """Routing mode used by :func:`create_transfer_context`.

    * ``AUTO``: dispatch by ``tensor.device.type`` (CUDA -> lmcache-driven,
      others -> engine-driven). Preserves the historical behaviour.
    * ``ENGINE_DRIVEN``: force :class:`EngineDrivenTransferContext`
      (worker-side gather / scatter copy path).
    * ``LMCACHE_DRIVEN``: force :class:`LMCacheDrivenTransferContext`
      (IPC / SHM zero-copy path). Requires a registered KV-wrapper factory
      for the device.
    """

    AUTO = "auto"
    ENGINE_DRIVEN = "engine_driven"
    LMCACHE_DRIVEN = "lmcache_driven"


def _resolve_mode(mode: "str | MPTransferMode | None") -> MPTransferMode:
    """Coerce ``mode`` into :class:`MPTransferMode`, falling back to env."""
    raw = (
        mode
        if mode is not None
        else os.environ.get(ENV_MP_TRANSFER_MODE, MPTransferMode.AUTO.value)
    )
    if isinstance(raw, MPTransferMode):
        return raw
    try:
        return MPTransferMode(str(raw).lower())
    except ValueError as exc:
        valid = ", ".join(m.value for m in MPTransferMode)
        raise ValueError(
            "Invalid MP transfer mode %r (valid: %s)" % (raw, valid)
        ) from exc


def _build_lmcache_driven_context(device_type: str) -> "TransferContext":
    """Build a :class:`LMCacheDrivenTransferContext` after capability check."""
    try:
        resolve_kv_wrapper_factory(device_type)
    except ValueError as exc:
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not supported for device type "
            "%r: no KV-cache wrapper factory is registered. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        ) from exc
    device_spec = get_device_spec(device_type)
    if device_spec and not device_spec.is_handle_transfer_available():
        raise ValueError(
            "MP transfer mode 'lmcache_driven' is not available for device type "
            "%r: required platform capability checks failed. "
            "Use mode 'engine_driven' or 'auto' instead." % device_type
        )
    return LMCacheDrivenTransferContext()


class IPCEvent(Protocol):
    """Protocol for device events used by transport operations."""

    def wait(self, stream: object | None = None) -> None:
        """Make ``stream`` wait for this event (async ordering primitive)."""


SendRequest = Callable[[MessageQueueClient, RequestType, list[object]], MessagingFuture]


def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Return the flat block-id list for transports without HMA support."""
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
    return block_ids[0]


def _safe_gather_block_ids(block_ids: list[int]) -> tuple[list[int], int | None]:
    """Replace null Mamba block IDs for a read, preserving their positions."""
    safe_id = next((block_id for block_id in block_ids if block_id != 0), None)
    if safe_id is None:
        return list(block_ids), None
    return [safe_id if block_id == 0 else block_id for block_id in block_ids], safe_id


def _scatter_non_null_mamba_chunks(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[int],
    chunks: list[torch.Tensor],
    blocks_per_chunk: int,
    block_size: int,
    layout_hints: LayoutHints | None,
    skip_first_n_tokens: int,
) -> None:
    """Scatter only real Mamba pages; null IDs never reach the H2D kernel."""
    skip_blocks = skip_first_n_tokens // block_size
    block_ordinal = 0
    for chunk_index, chunk in enumerate(chunks):
        ids = block_ids[
            chunk_index * blocks_per_chunk : (chunk_index + 1) * blocks_per_chunk
        ]
        for block_index, block_id in enumerate(ids):
            if block_id == 0:
                block_ordinal += 1
                continue
            written = block_ordinal >= skip_blocks
            if written:
                start = block_index * block_size
                end = start + block_size
                block_chunk = (
                    chunk[:, :, start:end, :]
                    if chunk.dim() == 4
                    else chunk[:, start:end, :]
                ).contiguous()
                scatter_cpu_to_paged_kv(
                    kv_caches,
                    [block_id],
                    [block_chunk],
                    1,
                    layout_hints=layout_hints,
                    engine_kv_format=None,
                )
            block_ordinal += 1


def _kv_caches_for_group(
    kv_caches: dict[str, torch.Tensor],
    group_info: EngineGroupInfo | None,
) -> dict[str, torch.Tensor]:
    """Return the subset of ``kv_caches`` registered for one LMCache group.

    Args:
        kv_caches: Worker KV-cache tensors keyed by layer name, in
            registration order.
        group_info: The LMCache group to filter for, or ``None`` for the
            single-group (non-hybrid) fallback, which returns ``kv_caches``
            unchanged.

    Returns:
        A dict containing only the layers in ``group_info.layer_indices``,
        or all of ``kv_caches`` when ``group_info`` is ``None``.
    """
    if group_info is None:
        return kv_caches
    kv_cache_items = list(kv_caches.items())
    return {
        name: tensor
        for idx, (name, tensor) in enumerate(kv_cache_items)
        if idx in group_info.layer_indices
    }


def _blocks_per_chunk_for_group(
    group_info: EngineGroupInfo | None,
    default_blocks_in_chunk: int,
    default_block_size: int,
) -> int:
    """Return one LMCache chunk's block count for a group's own block size.

    Args:
        group_info: The LMCache group, or ``None`` for the single-group
            fallback, which returns ``default_blocks_in_chunk`` unchanged.
        default_blocks_in_chunk: Blocks per LMCache chunk for the default
            (single-group) block size.
        default_block_size: Tokens per paged block for the default
            (single-group) layout, used to recover the LMCache chunk size in
            tokens (``default_blocks_in_chunk * default_block_size``).

    Returns:
        ``default_blocks_in_chunk`` when ``group_info`` is ``None`` or does
        not report ``tokens_per_block``; otherwise the number of this
        group's own paged blocks needed to cover one LMCache chunk.

    Raises:
        ValueError: If the LMCache chunk size in tokens is not a multiple of
            the group's ``tokens_per_block``.
    """
    if group_info is None or group_info.tokens_per_block <= 0:
        return default_blocks_in_chunk
    chunk_size_tokens = default_blocks_in_chunk * default_block_size
    if chunk_size_tokens % group_info.tokens_per_block != 0:
        raise ValueError(
            f"LMCache chunk size {chunk_size_tokens} must be a multiple of "
            f"group tokens_per_block {group_info.tokens_per_block}"
        )
    return chunk_size_tokens // group_info.tokens_per_block


def _group_chunk_shape(
    group_info: EngineGroupInfo | None,
    default_layout_desc: MemoryLayoutDesc,
    default_num_layers: int,
) -> torch.Size:
    """Return one chunk's tensor shape for a group's own layer count.

    Mirrors the shape formula ``register()`` uses to build
    ``default_layout_desc`` (``[2, num_layers, chunk_tokens, hidden_dim]``, or
    without the leading ``2`` for MLA/fused-K/V), substituting the group's own
    layer count for ``num_layers``. The chunk-token and hidden-dim extents are
    shared across every group by construction (one LMCache chunk always spans
    the same token count; hidden dim does not vary per group in Phase 2).

    Args:
        group_info: The LMCache group, or ``None`` for the single-group
            fallback, which returns ``default_layout_desc.shapes[0]`` unchanged.
        default_layout_desc: The default (single-group) layout descriptor
            computed at ``register()`` time.
        default_num_layers: The default (single-group) layer count that
            ``default_layout_desc.shapes[0]`` was built from.

    Returns:
        This group's chunk shape, with its own layer count substituted in.
    """
    default_shape = default_layout_desc.shapes[0]
    if group_info is None or not group_info.layer_indices:
        return default_shape
    num_layers = len(group_info.layer_indices)
    if num_layers == default_num_layers:
        return default_shape
    layer_dim = 0 if len(default_shape) == 3 else 1
    dims = list(default_shape)
    dims[layer_dim] = num_layers
    return torch.Size(dims)


def _iter_transfer_groups(
    engine_group_infos: Sequence[EngineGroupInfo],
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    blocks_in_chunk: int,
    block_size: int,
):
    """Yield one (group_info, kv_caches, block_ids, blocks_per_chunk) tuple
    per LMCache group to transfer.

    Args:
        engine_group_infos: The worker's registered LMCache groups, in
            protocol order. Empty means a single non-hybrid group.
        kv_caches: Worker KV-cache tensors keyed by layer name.
        block_ids: Engine block IDs indexed by LMCache group id.
        blocks_in_chunk: Blocks per LMCache chunk for the default
            (single-group) block size.
        block_size: Tokens per paged block for the default (single-group)
            layout.

    Yields:
        For each group: its :class:`EngineGroupInfo` (``None`` for the
        single-group fallback), that group's KV-cache subset, its flat
        block-id list, and its own blocks-per-chunk count.

    Raises:
        RuntimeError: If ``engine_group_infos`` is empty and ``block_ids``
            does not carry exactly one group (see :func:`_single_group_block_ids`).
        ValueError: If ``engine_group_infos`` is non-empty and ``block_ids``
            does not carry exactly one entry per group.
    """
    if not engine_group_infos:
        yield None, kv_caches, _single_group_block_ids(block_ids), blocks_in_chunk
        return
    if len(block_ids) != len(engine_group_infos):
        raise ValueError(
            f"Expected {len(engine_group_infos)} block-id groups, "
            f"got {len(block_ids)}"
        )
    for group_info, group_block_ids in zip(engine_group_infos, block_ids, strict=True):
        yield (
            group_info,
            _kv_caches_for_group(kv_caches, group_info),
            group_block_ids,
            _blocks_per_chunk_for_group(group_info, blocks_in_chunk, block_size),
        )


def _get_kv_device(kv_caches: dict[str, torch.Tensor]) -> torch.device:
    """Return the device shared by a non-empty KV-cache mapping.

    Args:
        kv_caches: Worker KV-cache tensors keyed by layer name.

    Returns:
        The device of the first KV-cache tensor.

    Raises:
        ValueError: If ``kv_caches`` is empty.
    """
    if not kv_caches:
        raise ValueError("LMCache-driven transfer requires at least one KV cache")
    return next(iter(kv_caches.values())).device


class TransferContext(ABC):
    """Abstract transport layer for worker-side KV transfer.

    Concrete implementations encapsulate how worker-side store/retrieve
    operations are transmitted to the multiprocess server. Device-handle paths
    return event-aware futures backed by MQ requests, while CPU paths may perform
    gather/scatter synchronously and return already-resolved futures.
    """

    @abstractmethod
    def register(
        self,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register KV caches with the server and wait for ACK.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            model_name: Model name used by cache keys.
            world_size: KV world size.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            mq_client: Message queue client used to communicate with server.
            mq_timeout: Timeout in seconds for synchronous request wait.
            send_request: Request sender callable used to issue MQ requests.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.
            engine_type: Serving engine that produced the caches. Only
                consumed by the handle path; adapters should pass their
                own :class:`EngineType` so this transport stays engine-
                neutral. Defaults to :attr:`EngineType.VLLM` for
                backwards compatibility.

        Raises:
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """

    def register_q(
        self,
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        """Register the paged Q ring with the server under the same worker
        instance_id but different model_name (model_name##query).

        Args:
            instance_id: Worker process instance identifier.
            q_caches: Worker Q cache tensors keyed by layer name.
            model_name: Model name used by cache keys (model_name##query).
            world_size: KV world size.
            blocks_in_chunk: Number of Q ring blocks per LMCache chunk.
            mq_client: Message queue client used to communicate with server.
            mq_timeout: Timeout in seconds for synchronous request wait.
            send_request: Request sender callable used to issue MQ requests.
            layout_hints: Optional inference-engine-provided layout hints.
            engine_group_infos: LMCache-owned engine KV cache group metadata.

        Raises:
            NotImplementedError: If the concrete transport does not support the
                Q ring (now only lmcache-driven).
            TimeoutError: If server registration does not complete before
                ``mq_timeout``.
            RuntimeError: If a concrete context cannot initialize.
        """
        raise NotImplementedError(
            "Q ring registration is not supported by this transfer context"
        )

    def submit_q_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a Q ring store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key for the Q store range (query-specific model_name).
            instance_id: Worker process instance identifier (shared with KV).
            q_caches: Q ring tensors keyed by layer name.
            block_ids: Q ring block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of Q ring blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            NotImplementedError: If the concrete transport does not support the
                Q ring (only the lmcache-driven path does).
            RuntimeError: If register_q() was not called first.
        """
        raise NotImplementedError(
            "Q ring store is not supported by this transfer context"
        )

    @abstractmethod
    def submit_store(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a store request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the store range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to store, indexed by LMCache KV group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def submit_retrieve(
        self,
        request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a retrieve request and return a completion future.

        Args:
            request_id: External request identifier.
            key: LMCache key object for the retrieve range.
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV cache tensors keyed by layer name.
            block_ids: vLLM block IDs to retrieve into, indexed by LMCache KV
                group id.
            event: Synchronization event object.
            blocks_in_chunk: Number of vLLM blocks per LMCache chunk.
            skip_first_n_tokens: Number of initial tokens to skip when writing.

        Returns:
            A future compatible with adapter-side ``query()``/``result()`` flow.

        Raises:
            RuntimeError: If register() was not called first.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this context."""

    @abstractmethod
    def flush_inflight_stores(self) -> None:
        """Synchronize any in-flight gather operations.

        Subclasses must implement this method. Contexts with no deferred
        operations should implement it as a no-op. Async contexts that
        defer GPU->CPU gather work must block until all in-flight stores
        have completed, so that vLLM cannot overwrite paged KV blocks
        before they are read.
        """


class LMCacheDrivenTransferContext(TransferContext):
    """LMCache-driven IPC + MQ future transport context.

    In this mode the serving engine provides device handles (accelerator IPC,
    or SHM wrappers for CPU with IPC-like semantics) and the LMCache server
    performs direct device-side data transfer.
    """

    def __init__(self) -> None:
        self._mq_client: MessageQueueClient | None = None
        self._send_request: SendRequest | None = None
        self._device: torch.device | None = None
        self._event_backend: EventIPCBackend | None = None

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register the worker KV cache with the LMCache server.

        Args:
            instance_id: Worker process instance identifier.
            kv_caches: Worker KV-cache tensors keyed by layer name.
            model_name: Model identifier used by the server.
            world_size: Tensor-parallel world size.
            _blocks_in_chunk: Engine blocks per LMCache chunk.
            mq_client: Message-queue client used for requests.
            mq_timeout: Timeout for the registration response.
            send_request: Request sender used by this context.
            layout_hints: Optional KV-layout metadata.
            engine_group_infos: Optional engine KV-group metadata.
            engine_type: Serving engine that produced the caches.

        Raises:
            RuntimeError: If event IPC is unsupported for the KV-cache device.
            ValueError: If ``kv_caches`` is empty.
        """
        device = _get_kv_device(kv_caches)
        event_backend = get_event_ipc_backend(device)
        event_backend.check_event_support(device)

        self._mq_client = mq_client
        self._send_request = send_request
        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                instance_id,
                wrap_kv_caches(kv_caches),
                model_name,
                world_size,
                engine_type,
                layout_hints,
                list(engine_group_infos),
            ],
        )
        future.result(timeout=mq_timeout)
        self._device = device
        self._event_backend = event_backend

    def register_q(
        self,
        instance_id: int,
        q_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        _blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        self._mq_client = mq_client
        self._send_request = send_request
        future = send_request(
            mq_client,
            RequestType.REGISTER_Q_CACHE,
            [
                instance_id,
                wrap_kv_caches(q_caches),
                model_name,
                world_size,
                EngineType.VLLM,
                layout_hints,
                list(engine_group_infos),
            ],
        )
        future.result(timeout=mq_timeout)

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        """Submit a handle-based store ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the store range.
            instance_id: Worker process instance identifier.
            _kv_caches: Worker KV-cache tensors accepted for interface
                consistency; the registered device is reused.
            block_ids: Engine block IDs indexed by LMCache KV group.
            event: Producer event that orders reads of the engine KV cache.
            _blocks_in_chunk: Engine blocks per chunk (unused by this transport).

        Returns:
            A device-event-aware future for the server response.

        Raises:
            RuntimeError: If the context is not registered or event IPC is
                unsupported.
        """
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.STORE,
            [key, instance_id, block_ids, event_ipc_handle],
        ).to_device_future(device=self._device)

    def submit_q_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _q_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
    ) -> MessagingFuture:
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_q_store()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.STORE_Q,
            [key, instance_id, block_ids, event_ipc_handle],
        ).to_device_future(device=self._device)

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        event: IPCEvent,
        _blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        """Submit a handle-based retrieve ordered by ``event``.

        Args:
            _request_id: External request identifier (unused by this transport).
            key: LMCache key for the retrieve range.
            instance_id: Worker process instance identifier.
            _kv_caches: Worker KV-cache tensors accepted for interface
                consistency; the registered device is reused.
            block_ids: Engine block IDs indexed by LMCache KV group.
            event: Producer event that orders writes to the engine KV cache.
            _blocks_in_chunk: Engine blocks per chunk (unused by this transport).
            skip_first_n_tokens: Initial tokens the server must not overwrite.

        Returns:
            A device-event-aware future for the server response.

        Raises:
            RuntimeError: If the context is not registered or event IPC is
                unsupported.
        """
        if (
            self._mq_client is None
            or self._send_request is None
            or self._device is None
            or self._event_backend is None
        ):
            raise RuntimeError(
                "LMCache-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )
        event_ipc_handle = self._event_backend.export_event(event, self._device)
        return self._send_request(
            self._mq_client,
            RequestType.RETRIEVE,
            [key, instance_id, block_ids, event_ipc_handle, skip_first_n_tokens],
        ).to_device_future(device=self._device)

    def close(self) -> None:
        """Release the message queue and cached event-backend state."""
        self._mq_client = None
        self._send_request = None
        self._device = None
        self._event_backend = None

    def flush_inflight_stores(self) -> None:
        pass


class EngineDrivenTransferContext(TransferContext):
    """Engine-driven transfer context for non-CUDA workers.

    In this mode the engine (worker side) owns the data movement: the
    worker adapter gathers/packs KV into CPU buffers, commits via
    message-queue, and the server side persists/rehydrates from storage.
    """

    def __init__(self) -> None:
        self._engine_driven_context: EngineDrivenContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._engine_kv_format: Any = None
        self._engine_group_infos: list[EngineGroupInfo] = []
        self._block_size: int = 0
        self._num_layers: int = 0
        self._kvweave_codec: KVWeaveCodec | None = None
        self._kvweave_quant_enabled = False
        self._group_layout_descs: list[MemoryLayoutDesc] = []
        self._group_raw_layout_descs: list[MemoryLayoutDesc] = []
        self._group_cache_categories: list[str] = []
        self._group_mamba_layouts: list[
            tuple[MambaSubStateWireLayout, MambaSubStateWireLayout] | None
        ] = []
        self._group_tokens_per_block: list[int] = []
        self._mamba_codec_options: MambaCodecOptions | None = None

    @property
    def engine_driven_context(self) -> EngineDrivenContext:
        """Return the underlying SHM/pickle context created by ``register``.

        Raises:
            RuntimeError: If accessed before ``register`` has run.
        """
        if self._engine_driven_context is None:
            raise RuntimeError(
                "EngineDrivenTransferContext is not registered, call register() first."
            )
        return self._engine_driven_context

    def iter_transfer_groups(
        self,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        blocks_in_chunk: int,
    ):
        """Yield this context's registered LMCache groups for one transfer.

        Thin wrapper around :func:`_iter_transfer_groups` binding the
        registered ``engine_group_infos``/``block_size``, for subclasses
        (e.g. :class:`AsyncEngineDrivenTransferContext`) that need per-group
        iteration without reaching into private attributes.

        Args:
            kv_caches: Worker KV-cache tensors keyed by layer name.
            block_ids: Engine block IDs indexed by LMCache group id.
            blocks_in_chunk: Blocks per LMCache chunk for the default
                (single-group) block size.

        Yields:
            See :func:`_iter_transfer_groups`.
        """
        yield from _iter_transfer_groups(
            self._engine_group_infos,
            kv_caches,
            block_ids,
            blocks_in_chunk,
            self._block_size,
        )

    def group_chunk_shape(self, group_info: EngineGroupInfo | None) -> torch.Size:
        """Return one chunk's tensor shape for a group's own layer count.

        Thin wrapper around :func:`_group_chunk_shape` binding this context's
        registered layout descriptor and default layer count.

        Args:
            group_info: The LMCache group, or ``None`` for the single-group
                fallback.

        Returns:
            This group's chunk shape (see :func:`_group_chunk_shape`).
        """
        if group_info is not None and group_info in self._engine_group_infos:
            group_index = self._engine_group_infos.index(group_info)
            if group_index < len(self._group_raw_layout_descs):
                return self._group_raw_layout_descs[group_index].shapes[0]
        return _group_chunk_shape(
            group_info, self.engine_driven_context.layout_desc, self._num_layers
        )

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
        engine_type: EngineType = EngineType.VLLM,
    ) -> None:
        """Register KV caches with the non-GPU context server.

        ``engine_group_infos`` is used to split worker-side gather/scatter by
        LMCache group at store/retrieve time (see ``_iter_transfer_groups``),
        matching the CUDA transfer path's per-group block-id addressing.
        """
        del engine_type  # unused on the engine-driven path
        (
            block_size,
            num_layers,
            hidden_dim_size,
            dtype_str,
            engine_kv_format,
            kv_size,
        ) = compute_kv_layout(kv_caches, layout_hints=layout_hints)
        self._layout_hints = layout_hints
        self._engine_kv_format = engine_kv_format
        self._engine_group_infos = list(engine_group_infos)
        self._block_size = block_size
        self._num_layers = num_layers

        # The wire field is named use_mla but only drives the object plane
        # count: single-plane (kv_size == 1) covers MLA and fused-K/V formats.
        use_mla_flag = kv_size == 1
        shape = (
            torch.Size([num_layers, blocks_in_chunk * block_size, hidden_dim_size])
            if use_mla_flag
            else torch.Size(
                [2, num_layers, blocks_in_chunk * block_size, hidden_dim_size]
            )
        )
        dtype = getattr(torch, dtype_str)
        layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
        self._group_raw_layout_descs = []
        group_layouts = []
        group_cache_categories: list[str] = []
        group_mamba_layouts: list[
            tuple[MambaSubStateWireLayout, MambaSubStateWireLayout] | None
        ] = []
        group_tokens_per_block: list[int] = []
        group_kv_sizes: list[int] = []
        group_infos = self._engine_group_infos or [None]
        for group_info in group_infos:
            group_caches = (
                _kv_caches_for_group(kv_caches, group_info)
                if group_info is not None
                else kv_caches
            )
            (
                group_block_size,
                group_num_layers,
                group_hidden_dim,
                group_dtype_str,
                _group_format,
                group_kv_size,
            ) = compute_kv_layout(group_caches, layout_hints=layout_hints)
            group_tokens = (
                group_info.tokens_per_block if group_info is not None else block_size
            ) or block_size
            group_blocks = (
                _blocks_per_chunk_for_group(
                    group_info, blocks_in_chunk, group_block_size
                )
                if group_info is not None
                else blocks_in_chunk
            )
            group_layer_count = (
                len(group_info.layer_indices)
                if group_info is not None and group_info.layer_indices
                else group_num_layers
            )
            if group_kv_size == 1:
                group_shape = torch.Size(
                    [group_layer_count, group_blocks * group_tokens, group_hidden_dim]
                )
            else:
                group_shape = torch.Size(
                    [
                        2,
                        group_layer_count,
                        group_blocks * group_tokens,
                        group_hidden_dim,
                    ]
                )
            group_category = (
                group_info.cache_category if group_info is not None else "attention"
            )
            group_mamba_layout = (
                group_info.mamba_real_layout if group_info is not None else None
            )
            group_dtype = getattr(torch, group_dtype_str)
            if group_category == "mamba" and group_mamba_layout is not None:
                # Mamba opaque page view must use conv_state dtype for byte
                # addressing. vLLM version changes may desync spec dtype vs
                # actual page view dtype; pin to real conv dtype to keep
                # split/merge byte math consistent.
                conv_layout, ssm_layout = group_mamba_layout
                conv_dtype = getattr(
                    torch, conv_layout.dtype_str.removeprefix("torch.")
                )
                if group_dtype != conv_dtype:
                    logger.warning(
                        "Mamba group %d page-view dtype mismatch: "
                        "compute_kv_layout=%s, conv_layout=%s; "
                        "using conv_layout dtype for raw page view",
                        len(group_layouts),
                        group_dtype,
                        conv_dtype,
                    )
                group_dtype = conv_dtype

                planes = 1 if group_kv_size == 1 else 2
                page_bytes = (
                    planes
                    * group_tokens
                    * group_hidden_dim
                    * group_dtype.itemsize
                )
                mamba_bytes = conv_layout.byte_length + ssm_layout.byte_length
                if mamba_bytes > page_bytes:
                    raise ValueError(
                        "Mamba real layout exceeds page bytes: "
                        f"group={len(group_layouts)} mamba_bytes={mamba_bytes} "
                        f"page_bytes={page_bytes}"
                    )
            group_layout = MemoryLayoutDesc(
                shapes=[group_shape],
                dtypes=[group_dtype],
            )
            self._group_raw_layout_descs.append(group_layout)
            group_layouts.append(group_layout)
            group_cache_categories.append(group_category)
            group_mamba_layouts.append(group_mamba_layout)
            group_tokens_per_block.append(group_tokens)
            group_kv_sizes.append(group_kv_size)
        self._group_cache_categories = group_cache_categories
        self._group_mamba_layouts = group_mamba_layouts
        self._group_tokens_per_block = group_tokens_per_block

        runtime_config = KVWeaveRuntimeConfig.from_env()
        if runtime_config.enabled:
            self._kvweave_codec = KVWeaveCodec(runtime_config.attention_codec_kwargs)
            self._mamba_codec_options = runtime_config.mamba_options
            quantized_layouts = []
            for group_index, raw_layout in enumerate(group_layouts):
                category = group_cache_categories[group_index]
                raw_size = sum(
                    int(torch.tensor(shape).prod().item()) * itemsize
                    for shape, dtype_item in zip(
                        raw_layout.shapes, raw_layout.dtypes, strict=True
                    )
                    for itemsize in [dtype_item.itemsize]
                )
                if category == "mamba":
                    mamba_layout = group_mamba_layouts[group_index]
                    if not runtime_config.linear_quant_enabled or mamba_layout is None:
                        quantized_layouts.append(raw_layout)
                        continue
                    serialized_size = KVWeaveCodec.estimate_mamba_serialized_size(
                        raw_layout,
                        mamba_layout,
                        group_tokens_per_block[group_index],
                        (
                            self._mamba_codec_options.conv_scaling_method,
                            self._mamba_codec_options.ssm_scaling_method,
                        ),
                        (
                            self._mamba_codec_options.conv_qbit,
                            self._mamba_codec_options.ssm_qbit,
                        ),
                    )
                    threshold = raw_size * runtime_config.linear_max_size_ratio
                elif (
                    category in {"attention", "unknown"}
                    and group_kv_sizes[group_index] == 2
                ):
                    serialized_size = self._kvweave_codec.estimate_serialized_size(
                        raw_layout
                    )
                    threshold = raw_size
                else:
                    # MLA attention (single-plane) is not supported by the
                    # attention codec; leave the group unquantized.
                    quantized_layouts.append(raw_layout)
                    continue
                if serialized_size < threshold:
                    quantized_layouts.append(
                        MemoryLayoutDesc(
                            shapes=[torch.Size([serialized_size])],
                            dtypes=[torch.uint8],
                        )
                    )
                else:
                    quantized_layouts.append(raw_layout)
            self._group_layout_descs = quantized_layouts
            self._kvweave_quant_enabled = any(
                layout.dtypes == [torch.uint8]
                for layout in quantized_layouts
            )
        else:
            self._group_layout_descs = group_layouts
            self._kvweave_quant_enabled = False
        group_layout_descs = (
            [serialize_memory_layout_desc(layout) for layout in self._group_layout_descs]
            if engine_group_infos
            else None
        )
        enable_l1_kvweave_quant = self._kvweave_quant_enabled

        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
            [
                RegisterEngineDrivenContextPayload(
                    instance_id=instance_id,
                    model_name=model_name,
                    world_size=world_size,
                    block_size=block_size,
                    num_layers=num_layers,
                    hidden_dim_size=hidden_dim_size,
                    dtype_str=dtype_str,
                    use_mla=use_mla_flag,
                    engine_group_infos=list(engine_group_infos),
                    group_layout_descs=group_layout_descs,
                    enable_l1_kvweave_quant=enable_l1_kvweave_quant,
                )
            ],
        )
        response = future.result(timeout=mq_timeout)
        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size

        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=block_size,
            use_mla=use_mla_flag,
        )
        self._engine_driven_context = create_engine_driven_context(
            metadata,
            mq_client,
            mq_timeout,
            shm_name=shm_name,
            pool_size=pool_size,
        )
        supported_transfer_mode = "SHM" if shm_name and pool_size > 0 else "pickle"
        logger.info(
            "Worker non-GPU transfer context registered (instance_id=%d, mode=%s)",
            instance_id,
            supported_transfer_mode,
        )

    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_store()."
            )

        torch_dev.synchronize()
        result = self._engine_driven_context.prepare_store(key, instance_id)
        out_buffers, chunk_indices = result if result is not None else (None, None)
        # All chunks already in cache — nothing to gather or commit.
        if chunk_indices is not None and len(chunk_indices) == 0:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(True)
            return future

        cpu_chunks: list[torch.Tensor] = []
        # ``out_buffers``/``chunk_indices`` (when present) are flat over the
        # whole multi-group chunk sequence (group 0's chunks first, then
        # group 1's, ...), matching the order the server concatenated them in
        # (see EngineDrivenTransferModule.prepare_store). Each group's own
        # chunk-count range is sliced out before gathering that group.
        group_offset = 0
        for group_index, (
            _group_info,
            group_kv_caches,
            group_block_ids,
            group_blocks_in_chunk,
        ) in enumerate(
            self.iter_transfer_groups(kv_caches, block_ids, blocks_in_chunk)
        ):
            num_group_chunks = len(group_block_ids) // group_blocks_in_chunk
            if chunk_indices is None:
                group_chunk_indices = None
                group_out_buffers = None
            else:
                selected = [
                    (out_idx, chunk_idx - group_offset)
                    for out_idx, chunk_idx in enumerate(chunk_indices)
                    if group_offset <= chunk_idx < group_offset + num_group_chunks
                ]
                group_chunk_indices = [chunk_idx for _, chunk_idx in selected]
                group_out_buffers = (
                    [out_buffers[out_idx] for out_idx, _ in selected]
                    if out_buffers is not None
                    else None
                )
            if chunk_indices is None or group_chunk_indices:
                is_mamba_group = (
                    self._group_is_quantized(group_index)
                    and
                    _group_info is not None and _group_info.cache_category == "mamba"
                )
                if not is_mamba_group:
                    raw_chunks = gather_paged_kv_to_cpu(
                        group_kv_caches,
                        group_block_ids,
                        group_blocks_in_chunk,
                        layout_hints=self._layout_hints,
                        engine_kv_format=None,
                        out=None if self._group_is_quantized(group_index) else group_out_buffers,
                        chunk_indices=group_chunk_indices,
                    )
                else:
                    selected_group_block_ids = (
                        group_block_ids
                        if group_chunk_indices is None
                        else [
                            block_id
                            for chunk_index in group_chunk_indices
                            for block_id in group_block_ids[
                                chunk_index * group_blocks_in_chunk :
                                (chunk_index + 1) * group_blocks_in_chunk
                            ]
                        ]
                    )
                    gather_block_ids, safe_block_id = _safe_gather_block_ids(
                        selected_group_block_ids
                    )
                    if safe_block_id is None:
                        raw_shape = self.group_chunk_shape(_group_info)
                        raw_chunks = [
                            torch.zeros(
                                raw_shape,
                                dtype=next(iter(group_kv_caches.values())).dtype,
                            )
                            for _ in range(
                                len(group_chunk_indices)
                                if group_chunk_indices is not None
                                else len(group_block_ids) // group_blocks_in_chunk
                            )
                        ]
                    else:
                        raw_chunks = gather_paged_kv_to_cpu(
                            group_kv_caches,
                            gather_block_ids,
                            group_blocks_in_chunk,
                            layout_hints=self._layout_hints,
                            engine_kv_format=None,
                            out=None,
                            chunk_indices=None,
                        )
                if self._group_is_quantized(group_index):
                    if self._kvweave_codec is None:
                        raise RuntimeError("KVWeave codec is not initialized")
                    category = self._group_cache_categories[group_index]
                    mamba_layout = self._group_mamba_layouts[group_index]
                    tokens_per_block = self._group_tokens_per_block[group_index]
                    if group_out_buffers is not None:
                        if len(raw_chunks) != len(group_out_buffers):
                            raise ValueError("quantized SHM slot count mismatch")
                        for raw_chunk, destination in zip(
                            raw_chunks, group_out_buffers, strict=True
                        ):
                            payload = self._kvweave_codec.encode_chunk(
                                category, mamba_layout, tokens_per_block,
                                self._mamba_codec_options, raw_chunk,
                            )
                            _copy_bytes_to_tensor(payload, destination)
                    else:
                        for raw_chunk in raw_chunks:
                            payload = self._kvweave_codec.encode_chunk(
                                category, mamba_layout, tokens_per_block,
                                self._mamba_codec_options, raw_chunk,
                            )
                            destination = torch.empty(
                                (self._group_layout_descs[group_index].shapes[0][0],),
                                dtype=torch.uint8,
                            )
                            _copy_bytes_to_tensor(payload, destination)
                            cpu_chunks.append(destination)
                else:
                    cpu_chunks.extend(raw_chunks)
            group_offset += num_group_chunks

        if out_buffers is not None:
            # SHM path uses async device->CPU copies; complete them before commit.
            torch_dev.synchronize()
        ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)

        future = MessagingFuture()
        future.set_result(ok)
        return future

    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError(
                "Engine-driven transfer context is not registered. "
                "Call register() before submit_retrieve()."
            )

        src_buffers = self._engine_driven_context.prepare_retrieve(key, instance_id)
        ok = src_buffers is not None
        if src_buffers is not None:
            try:
                # ``src_buffers`` is flat over the whole multi-group chunk
                # sequence, in the same group-major order submit_store wrote
                # it in (see the group_offset bookkeeping there).
                group_offset = 0
                for group_index, (
                    _group_info,
                    group_kv_caches,
                    group_block_ids,
                    group_blocks_in_chunk,
                ) in enumerate(
                    self.iter_transfer_groups(kv_caches, block_ids, blocks_in_chunk)
                ):
                    num_group_chunks = len(group_block_ids) // group_blocks_in_chunk
                    group_chunks = src_buffers[
                        group_offset : group_offset + num_group_chunks
                    ]
                    # ``_scatter_non_null_mamba_chunks`` walks ``block_ids``
                    # in original, uncompacted order to keep its internal
                    # ``block_ordinal`` (used to compare against
                    # ``skip_first_n_tokens // block_size``) in the same
                    # reference frame as the caller's global skip count.
                    # Dropping leading all-null chunks before that call would
                    # shift ``block_ordinal`` relative to ``skip_first_n_tokens``
                    # and desync which block gets skipped -- so only compact
                    # for the non-mamba (attention) scatter path below.
                    is_mamba_group = (
                        _group_info is not None
                        and _group_info.cache_category == "mamba"
                    )
                    if not is_mamba_group:
                        active_chunk_indices = [
                            chunk_index
                            for chunk_index in range(num_group_chunks)
                            if any(
                                group_block_ids[
                                    chunk_index
                                    * group_blocks_in_chunk : (chunk_index + 1)
                                    * group_blocks_in_chunk
                                ]
                            )
                        ]
                        group_chunks = [
                            group_chunks[chunk_index]
                            for chunk_index in active_chunk_indices
                        ]
                        group_block_ids = [
                            block_id
                            for chunk_index in active_chunk_indices
                            for block_id in group_block_ids[
                                chunk_index
                                * group_blocks_in_chunk : (chunk_index + 1)
                                * group_blocks_in_chunk
                            ]
                        ]
                    if self._group_is_quantized(group_index):
                        if self._kvweave_codec is None:
                            raise RuntimeError("KVWeave codec is not initialized")
                        category = self._group_cache_categories[group_index]
                        mamba_layout = self._group_mamba_layouts[group_index]
                        tokens_per_block = self._group_tokens_per_block[group_index]
                        raw_shape = self._group_raw_layout_descs[group_index].shapes[0]
                        raw_dtype = self._group_raw_layout_descs[group_index].dtypes[0]
                        group_chunks = [
                            self._kvweave_codec.decode_chunk(
                                category, mamba_layout, tokens_per_block,
                                raw_shape, raw_dtype, chunk,
                            )
                            for chunk in group_chunks
                        ]
                    has_null_mamba_block = (
                        self._group_is_quantized(group_index)
                        and is_mamba_group
                        and any(block_id == 0 for block_id in group_block_ids)
                    )
                    if has_null_mamba_block:
                        _scatter_non_null_mamba_chunks(
                            group_kv_caches,
                            group_block_ids,
                            group_chunks,
                            group_blocks_in_chunk,
                            self._group_tokens_per_block[group_index],
                            self._layout_hints,
                            skip_first_n_tokens,
                        )
                    else:
                        scatter_cpu_to_paged_kv(
                            group_kv_caches,
                            group_block_ids,
                            group_chunks,
                            group_blocks_in_chunk,
                            skip_first_n_tokens=skip_first_n_tokens,
                            layout_hints=self._layout_hints,
                            engine_kv_format=None,
                        )
                    group_offset += num_group_chunks
            except (RuntimeError, ValueError, TypeError, IndexError):
                logger.exception("Failed to scatter retrieved CPU context chunks")
                ok = False
            # SHM path: ensure all device writes are complete before releasing
            # the SHM slot (server may immediately reuse it after commit_retrieve).
            torch_dev.synchronize()
        self._engine_driven_context.commit_retrieve(key, instance_id)

        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(ok)
        return future

    def _group_is_quantized(self, group_index: int) -> bool:
        return (
            group_index < len(self._group_layout_descs)
            and self._group_layout_descs[group_index].dtypes == [torch.uint8]
        )

    def close(self) -> None:
        if self._engine_driven_context is not None:
            self._engine_driven_context.close()
            self._engine_driven_context = None

    def flush_inflight_stores(self) -> None:
        pass


def create_transfer_context(
    kv_caches: dict[str, torch.Tensor],
    mode: "str | MPTransferMode | None" = None,
    **_kwargs: Any,
) -> TransferContext:
    """Create a transfer context from KV cache device type.

    The device check is intentionally centralized here. Routing can be
    overridden via the ``mode`` argument or the ``LMCACHE_MP_TRANSFER_MODE``
    environment variable; see :class:`MPTransferMode` for accepted values.

    Args:
        kv_caches: Worker KV cache tensors keyed by layer name.
        mode: Optional routing override. When ``None`` the value of
            ``LMCACHE_MP_TRANSFER_MODE`` is consulted, defaulting to
            :attr:`MPTransferMode.AUTO`.
        **kwargs: Unused placeholder for forward-compatible factory extension.

    Returns:
        A concrete :class:`TransferContext` implementation.

    Raises:
        ValueError: If ``kv_caches`` is empty, has mixed device types, the
            requested mode string is unknown, or the requested mode is not
            supported for the worker device.
    """
    if not kv_caches:
        raise ValueError("kv_caches is empty")
    device_types = {tensor.device.type for tensor in kv_caches.values()}
    if len(device_types) != 1:
        raise ValueError(
            f"All KV cache tensors must share one device type, got {device_types}"
        )
    device_type = next(iter(device_types))
    resolved_mode = _resolve_mode(mode)
    logger.info(
        "Creating transfer context (device_type=%s, mode=%s)",
        device_type,
        resolved_mode.value,
    )
    if resolved_mode is MPTransferMode.LMCACHE_DRIVEN:
        return _build_lmcache_driven_context(device_type)
    if resolved_mode is MPTransferMode.ENGINE_DRIVEN:
        return _build_engine_driven_context()
    # AUTO: dispatch by device type (CUDA -> handle path, else -> data path).
    if device_type == "cuda":
        return LMCacheDrivenTransferContext()
    return _build_engine_driven_context()
