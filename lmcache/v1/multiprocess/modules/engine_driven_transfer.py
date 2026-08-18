# SPDX-License-Identifier: Apache-2.0
"""Engine-driven KV cache transfer operations for the MPCacheServer."""

# Standard
from dataclasses import dataclass, field
from functools import partial
import pickle
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
)
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
    deserialize_memory_layout_desc,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext, ShmPoolInfo
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    InstanceLivenessTarget,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata

# Local
from .server_transfer import (
    TransferStrategy,
    create_transfer_strategy,
)

logger = init_logger(__name__)


@dataclass
class EngineDrivenContextEntry:
    """Registered non-GPU context metadata for a single worker instance.

    Attributes:
        metadata: Layout metadata describing the non-CUDA chunk format.
        model_name: The name of the model associated with this context.
        world_size: The world size associated with this context.
        last_seen: ``time.monotonic()`` of the most recent activity from this
            instance (register, PING, prepare/commit). Drives reaping.
        has_liveness_signal: True once the instance has sent at least one
            PING. Selects the reap window. Latched only by PING.
        metadata_by_group: Per-LMCache-group layout metadata, in protocol
            order. Empty for a single non-hybrid group, in which case
            ``metadata`` above is used as-is.
    """

    metadata: EngineDrivenContextMetadata
    model_name: str
    world_size: int
    last_seen: float = 0.0
    has_liveness_signal: bool = False
    metadata_by_group: list[EngineDrivenContextMetadata] = field(default_factory=list)


class EngineDrivenTransferModule(InstanceLivenessTarget):
    """Handles Engine-driven KV cache transfer operations.

    Owns non-GPU context registrations and provides handlers for
    register, unregister, prepare/commit store, and prepare/commit retrieve
    of CPU-serialized KV caches.

    Args:
        ctx: The shared engine context.
    """

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._engine_driven_contexts: dict[int, EngineDrivenContextEntry] = {}
        self._strategies: dict[int, TransferStrategy] = {}
        # Guards _engine_driven_contexts and _strategies together (the reaper
        # mutates them off the MQ main loop). Leaf lock, never held with
        # _pending_shm_lock.
        self._lock = threading.Lock()
        self._pending_shm_writes: dict[
            tuple[int, IPCCacheServerKey], list[ObjectKey]
        ] = {}
        self._pending_shm_reads: dict[
            tuple[int, IPCCacheServerKey], list[ObjectKey]
        ] = {}
        self._pending_shm_lock = threading.Lock()
        self._shm_pool_info: ShmPoolInfo = self._ctx.shm_pool_info

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handler specs for all request types this module serves.

        Returns:
            A list of HandlerSpec entries mapping request types to
            their handler callables and thread pool assignments.
        """
        return [
            HandlerSpec(
                RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
                self.register_kv_cache_engine_driven_context,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
                self.unregister_kv_cache,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.PREPARE_STORE,
                self.prepare_store,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.COMMIT_STORE,
                self.commit_store,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.PREPARE_RETRIEVE,
                self.prepare_retrieve,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.COMMIT_RETRIEVE,
                self.commit_retrieve,
                ThreadPoolType.AFFINITY,
            ),
        ]

    def report_status(self) -> dict:
        """Return non-GPU transfer module status information.

        Returns:
            A dict containing registered non-CUDA instance IDs and
            per-instance context metadata.
        """
        registered_non_cuda_ids: list[int] = []
        non_cuda_context_meta: dict[str, dict] = {}

        with self._lock:
            entries = dict(self._engine_driven_contexts)
        for instance_id, entry in entries.items():
            registered_non_cuda_ids.append(instance_id)
            non_cuda_context_meta[str(instance_id)] = {
                "model_name": entry.model_name,
                "world_size": entry.world_size,
                "block_size": entry.metadata.block_size,
                "use_mla": entry.metadata.use_mla,
            }

        return {
            "registered_non_cuda_instance_ids": registered_non_cuda_ids,
            "non_cuda_context_meta": non_cuda_context_meta,
        }

    def close(self) -> None:
        """Release resources owned by this module."""
        with self._lock:
            self._engine_driven_contexts.clear()
            self._strategies.clear()

    def touch_instance(self, instance_id: int) -> None:
        """Refresh the worker's last-seen time and mark it ping-proven.

        A no-op if the instance is not tracked.

        Args:
            instance_id: The worker instance ID.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._engine_driven_contexts.get(instance_id)
            if entry is not None:
                entry.last_seen = now
                entry.has_liveness_signal = True

    def tracked_instance_count(self) -> int:
        """Return the number of currently registered non-GPU instances."""
        with self._lock:
            return len(self._engine_driven_contexts)

    def reap_stale_instances(
        self, reap_timeout_s: float, registration_grace_s: float
    ) -> list[int]:
        """Reap non-GPU registrations that have gone silent.

        A ping-proven instance is judged against ``reap_timeout_s``; one that
        has never pinged against the larger ``registration_grace_s``.

        Args:
            reap_timeout_s: Silence budget for ping-proven instances.
            registration_grace_s: Silence budget for never-pinged instances.

        Returns:
            The instance IDs reaped this scan.
        """
        now = time.monotonic()
        reaped: list[tuple[int, EngineDrivenContextEntry]] = []
        with self._lock:
            stale_ids = [
                iid
                for iid, entry in self._engine_driven_contexts.items()
                if now - entry.last_seen
                > (
                    reap_timeout_s
                    if entry.has_liveness_signal
                    else registration_grace_s
                )
            ]
            for iid in stale_ids:
                entry = self._engine_driven_contexts.pop(iid)
                self._strategies.pop(iid, None)
                reaped.append((iid, entry))
        for iid, entry in reaped:
            self._release_entry(iid, entry)
            logger.warning(
                "Reaped non-GPU instance %d: silent for %.1fs (pinged=%s)",
                iid,
                now - entry.last_seen,
                entry.has_liveness_signal,
            )
        return [iid for iid, _ in reaped]

    def _resolve_for_transfer(
        self, instance_id: int
    ) -> tuple[EngineDrivenContextEntry, TransferStrategy]:
        """Return (entry, strategy) for a transfer, refreshing last_seen.

        Pair-atomicity guarantees the entry exists whenever the strategy
        does. Refreshes last_seen (no latch) so an active worker is not
        reaped mid-transfer.

        Args:
            instance_id: The worker instance ID.

        Returns:
            The entry and its transfer strategy.

        Raises:
            ValueError: If the instance is not registered (or was reaped).
        """
        now = time.monotonic()
        with self._lock:
            entry = self._engine_driven_contexts.get(instance_id)
            strategy = self._strategies.get(instance_id)
            if entry is None or strategy is None:
                raise ValueError(
                    "non-GPU context not registered (or reaped) for "
                    f"instance ID {instance_id}"
                )
            entry.last_seen = now
            return entry, strategy

    def _release_entry(self, instance_id: int, entry: EngineDrivenContextEntry) -> None:
        """Release resources for a popped entry (run outside the lock).

        Sweeps the instance's pending SHM transfers and unregisters its
        layout descriptor.

        Args:
            instance_id: The popped instance ID.
            entry: The popped entry.
        """
        with self._pending_shm_lock:
            stale_writes = [k for k in self._pending_shm_writes if k[0] == instance_id]
            stale_reads = [k for k in self._pending_shm_reads if k[0] == instance_id]
            write_obj_keys = [self._pending_shm_writes.pop(k) for k in stale_writes]
            read_obj_keys = [self._pending_shm_reads.pop(k) for k in stale_reads]

        for obj_keys in write_obj_keys:
            if obj_keys:
                self._ctx.storage_manager.finish_write(obj_keys)
        for obj_keys in read_obj_keys:
            if obj_keys:
                self._ctx.storage_manager.finish_read_prefetched(obj_keys)

        self._ctx.layout_desc_registry.unregister(entry.model_name, entry.world_size)

    @staticmethod
    def _make_transfer_key(
        key: IPCCacheServerKey, instance_id: int
    ) -> tuple[int, IPCCacheServerKey]:
        return (instance_id, key)

    def _resolve_obj_keys(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> list[ObjectKey]:
        """Resolve object keys for every LMCache group registered for
        ``instance_id``, flattened group-major.

        Group-major flattening matches the order the worker concatenates
        gathered chunks in (see ``EngineDrivenTransferContext.submit_store``),
        so a positional zip between this flat list and the worker's flat
        chunk list still lines up per group. Each returned key retains its
        own ``object_group_id``, so downstream storage lookups still resolve
        per-group even though the list itself is flat.

        Args:
            key: Cache key for the token range.
            instance_id: Worker instance identifier, used to look up the
                registered group count.

        Returns:
            Object keys for all registered groups, flattened group-major.
        """
        entry, _ = self._resolve_for_transfer(instance_id)
        num_groups = max(1, len(entry.metadata_by_group))
        per_group_keys = self._ctx.resolve_obj_keys(key, list(range(num_groups)))
        return [obj_key for group_keys in per_group_keys for obj_key in group_keys]

    def _resolve_group_obj_keys(
        self, key: IPCCacheServerKey, object_group_id: int, num_groups: int
    ) -> list[ObjectKey]:
        """Resolve object keys for one LMCache group only.

        Args:
            key: Cache key for the token range.
            object_group_id: The group to resolve keys for.
            num_groups: Total number of registered groups (``1`` for the
                single-group fallback), used to size the resolver call.

        Returns:
            Object keys for the requested group only.
        """
        return self._ctx.resolve_obj_keys(key, list(range(num_groups)))[
            object_group_id
        ]

    @staticmethod
    def _make_group_layout_desc(
        num_layers: int,
        chunk_size: int,
        hidden_dim_size: int,
        dtype: torch.dtype,
        use_mla: bool,
    ) -> MemoryLayoutDesc:
        """Build one LMCache group's chunk layout descriptor.

        Args:
            num_layers: Number of layers carried by this group.
            chunk_size: Tokens per LMCache chunk.
            hidden_dim_size: Flattened hidden dimension per token.
            dtype: Torch dtype of the chunk tensor.
            use_mla: Whether the worker KV format is MLA (single-plane).

        Returns:
            A :class:`MemoryLayoutDesc` describing one chunk for this group.
        """
        shape = (
            torch.Size([num_layers, chunk_size, hidden_dim_size])
            if use_mla
            else torch.Size([2, num_layers, chunk_size, hidden_dim_size])
        )
        return MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])

    def register_kv_cache_engine_driven_context(
        self,
        payload: RegisterEngineDrivenContextPayload,
    ) -> RegisterEngineDrivenContextResponse:
        """Register non-CUDA KV layout metadata for non-GPU context mode.

        Args:
            payload: Struct containing all registration fields
                (instance_id, model_name, world_size, block_size,
                num_layers, hidden_dim_size, dtype_str, use_mla,
                engine_group_infos).

        Raises:
            ValueError: If ``payload.dtype_str`` is not a valid torch dtype name.
        """
        shm_name = self._shm_pool_info["shm_name"]
        pool_size = self._shm_pool_info["pool_size"]

        now = time.monotonic()
        with self._lock:
            existing = self._engine_driven_contexts.get(payload.instance_id)
            if existing is not None:
                existing.last_seen = now
                logger.info(
                    "Instance %d already registered (non-GPU); refreshing liveness",
                    payload.instance_id,
                )
                return RegisterEngineDrivenContextResponse(
                    shm_name=shm_name, pool_size=pool_size
                )

        dtype = getattr(torch, payload.dtype_str, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Invalid dtype_str '{payload.dtype_str}': must be a valid torch dtype "
                "attribute name (e.g. 'float16' for torch.float16, "
                "'bfloat16' for torch.bfloat16, 'float32' for torch.float32)."
            )

        if (
            payload.enable_l1_kvweave_quant
            and not self._ctx.storage_manager.is_l1_variable_size()
        ):
            raise ValueError(
                "L1 KVWeave quantization requires a variable-size L1 allocator"
            )

        layout_desc = self._make_group_layout_desc(
            payload.num_layers,
            self._ctx.chunk_size,
            payload.hidden_dim_size,
            dtype,
            payload.use_mla,
        )
        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=payload.block_size,
            use_mla=payload.use_mla,
        )
        if payload.group_layout_descs is not None:
            expected = max(1, len(payload.engine_group_infos))
            if len(payload.group_layout_descs) != expected:
                raise ValueError(
                    "group_layout_descs length mismatch: "
                    f"expected {expected}, got {len(payload.group_layout_descs)}"
                )
            metadata_by_group = [
                EngineDrivenContextMetadata(
                    layout_desc=deserialize_memory_layout_desc(encoded),
                    block_size=(
                        payload.engine_group_infos[idx].tokens_per_block
                        if payload.engine_group_infos
                        else payload.block_size
                    )
                    or payload.block_size,
                    use_mla=payload.use_mla,
                )
                for idx, encoded in enumerate(payload.group_layout_descs)
            ]
        else:
            metadata_by_group = [
                EngineDrivenContextMetadata(
                    layout_desc=self._make_group_layout_desc(
                        len(group_info.layer_indices),
                        self._ctx.chunk_size,
                        payload.hidden_dim_size,
                        dtype,
                        payload.use_mla,
                    ),
                    block_size=group_info.tokens_per_block or payload.block_size,
                    use_mla=payload.use_mla,
                )
                for group_info in payload.engine_group_infos
            ]
        # Build the entry and strategy outside the lock, then insert the pair
        # atomically so a concurrent reap can never strand one without the
        # other. REGISTER is SYNC-serialized, so it is the sole inserter.
        entry = EngineDrivenContextEntry(
            metadata=metadata,
            model_name=payload.model_name,
            world_size=payload.world_size,
            last_seen=now,
            has_liveness_signal=False,
            metadata_by_group=metadata_by_group,
        )
        strategy: TransferStrategy = create_transfer_strategy(
            self._ctx.storage_manager,
            shm_name=shm_name,
            pool_size=pool_size,
            pending_writes=self._pending_shm_writes,
            pending_reads=self._pending_shm_reads,
            pending_lock=self._pending_shm_lock,
            transfer_key_factory=self._make_transfer_key,
        )
        with self._lock:
            self._engine_driven_contexts[payload.instance_id] = entry
            self._strategies[payload.instance_id] = strategy

        logger.info(
            "Registered non-GPU context for instance %d (model=%s, world_size=%d, "
            "num_groups=%d)",
            payload.instance_id,
            payload.model_name,
            payload.world_size,
            max(1, len(metadata_by_group)),
        )

        # Multi-group registrations key each group's layout by object_group_id
        # (hybrid models: attention/mamba groups have distinct chunk shapes);
        # single-group registrations fall back to one shared layout_desc for
        # every object group, matching pre-hybrid behavior exactly.
        group_layout_descs = (
            {idx: m.layout_desc for idx, m in enumerate(metadata_by_group)}
            if metadata_by_group
            else None
        )
        attn_desc = AttnWindowDesc(
            num_chunks_in_sw=[-1] * max(1, len(metadata_by_group))
        )
        self._ctx.layout_desc_registry.register(
            payload.model_name,
            payload.world_size,
            layout_desc,
            attn_desc=attn_desc,
            group_layout_descs=group_layout_descs,
        )
        return RegisterEngineDrivenContextResponse(
            shm_name=shm_name, pool_size=pool_size
        )

    def unregister_kv_cache(self, instance_id: int) -> None:
        """Unregister a non-GPU KV cache context for the given instance ID.

        Args:
            instance_id: The worker instance identifier.
        """
        with self._lock:
            entry = self._engine_driven_contexts.pop(instance_id, None)
            if entry is not None:
                self._strategies.pop(instance_id, None)
        if entry is None:
            logger.warning(
                "No registered non-GPU context found for instance ID %d",
                instance_id,
            )
            return

        self._release_entry(instance_id, entry)
        logger.info("Unregistered non-CUDA context for instance ID %d", instance_id)

    @_lmcache_nvtx_annotate
    def prepare_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareStoreResponse:
        """Prepare a store operation.

        Loops once per registered LMCache group (metadata_by_group), since
        each group may have its own chunk layout (hybrid models); a single
        non-hybrid group loops exactly once over ``entry.metadata``. Per-group
        ``slots``/``chunk_indices`` are concatenated group-major, with
        ``chunk_indices`` offset by the chunk counts of earlier groups, so the
        worker's flat gather/commit sequencing lines up (see
        ``EngineDrivenTransferContext.submit_store``).

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.

        Returns:
            PrepareStoreResponse with empty slots for pickle mode.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        contexts = entry.metadata_by_group or [entry.metadata]
        num_groups = len(contexts)
        all_slots: list[dict] = []
        all_chunk_indices: list[int] = []
        chunk_offset = 0
        for group_id, context in enumerate(contexts):
            group_response = strategy.prepare_store(
                key=key,
                instance_id=instance_id,
                context=context,
                resolve_obj_keys=partial(
                    self._resolve_group_obj_keys,
                    object_group_id=group_id,
                    num_groups=num_groups,
                ),
            )
            group_slots = group_response.context.get("slots", [])
            group_chunk_indices = group_response.context.get("chunk_indices", [])
            all_slots.extend(group_slots)
            all_chunk_indices.extend(idx + chunk_offset for idx in group_chunk_indices)
            group_obj_keys = self._resolve_group_obj_keys(key, group_id, num_groups)
            chunk_offset += len(group_obj_keys)

        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras["store_start_time"] = time.perf_counter()
        return PrepareStoreResponse(
            context={"slots": all_slots, "chunk_indices": all_chunk_indices}
        )

    @_lmcache_nvtx_annotate
    def commit_store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        cpu_data: bytes,
    ) -> bool:
        """Commit serialized CPU chunks to storage.

        For SHM mode, ``cpu_data`` is empty (``b""``): every group's write
        locks were already accumulated under the same transfer key by
        ``ShmTransferStrategy.prepare_store`` (see its per-group accumulation
        note), so a single ``commit_store`` call releases all of them in one
        pop; calling it once per group would incorrectly pop (and thus lose
        track of) the shared entry after the first group. Pickle mode's
        ``cpu_data`` decodes to the worker's flat, group-major chunk list
        (see ``EngineDrivenTransferContext.submit_store``); since
        ``PickleTransferStrategy.commit_store`` unpickles ``cpu_data`` and
        zips it positionally against ``resolve_obj_keys(key)``, each group's
        call must receive only that group's own re-serialized chunk slice
        (not the shared flat payload), or positions would misalign across
        groups -- so pickle mode still loops once per group.

        Args:
            key: Cache key for the token range to store.
            instance_id: Worker instance identifier.
            cpu_data: Pickled list of CPU tensors produced by the worker, or
                ``b""`` for SHM mode.

        Returns:
            ``True`` when all reserved objects are written, otherwise ``False``.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        contexts = entry.metadata_by_group or [entry.metadata]
        num_groups = len(contexts)
        session = self._ctx.session_manager.get_or_create(key.request_id)
        st = session.extras.pop("store_start_time", None)

        if not cpu_data:
            ok = strategy.commit_store(
                key=key,
                instance_id=instance_id,
                cpu_data=cpu_data,
                context=contexts[0],
                resolve_obj_keys=partial(
                    self._resolve_obj_keys, instance_id=instance_id
                ),
            )
        else:
            all_chunks: list[torch.Tensor] = pickle.loads(cpu_data)
            ok = True
            chunk_offset = 0
            for group_id, context in enumerate(contexts):
                group_obj_keys = self._resolve_group_obj_keys(
                    key, group_id, num_groups
                )
                group_chunks = all_chunks[
                    chunk_offset : chunk_offset + len(group_obj_keys)
                ]
                group_cpu_data = pickle.dumps(group_chunks)
                chunk_offset += len(group_obj_keys)
                ok = (
                    strategy.commit_store(
                        key=key,
                        instance_id=instance_id,
                        cpu_data=group_cpu_data,
                        context=context,
                        resolve_obj_keys=partial(
                            self._resolve_group_obj_keys,
                            object_group_id=group_id,
                            num_groups=num_groups,
                        ),
                    )
                    and ok
                )

        if st is not None and ok:
            num_tokens = (
                len(self._resolve_obj_keys(key, instance_id)) * self._ctx.chunk_size
            )
            logger.info(
                "Stored %d tokens in %.3f seconds",
                num_tokens,
                time.perf_counter() - st,
            )
        return ok

    @_lmcache_nvtx_annotate
    def prepare_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> PrepareRetrieveResponse:
        """Retrieve prefetched chunks and return serialized CPU tensors.

        Loops once per registered LMCache group, concatenating per-group
        results group-major to match the worker's flat gather order (see
        ``prepare_store``): SHM mode's ``slots`` lists concatenate directly;
        pickle mode's per-group ``data`` payloads are each unpickled back to
        their chunk list and re-concatenated into one flat, group-major
        ``list[Tensor]`` before being re-pickled for the worker.

        Args:
            key: Cache key for the token range to retrieve.
            instance_id: Worker instance identifier.

        Returns:
            PrepareRetrieveResponse with serialized data on hit.

        Raises:
            ValueError: If no non-GPU context is registered for the given
                instance ID.
        """
        entry, strategy = self._resolve_for_transfer(instance_id)
        num_groups = max(1, len(entry.metadata_by_group))
        all_slots: list[dict] = []
        all_chunks: list[torch.Tensor] = []
        success = True
        for group_id in range(num_groups):
            group_response = strategy.prepare_retrieve(
                key=key,
                instance_id=instance_id,
                resolve_obj_keys=partial(
                    self._resolve_group_obj_keys,
                    object_group_id=group_id,
                    num_groups=num_groups,
                ),
            )
            success = success and group_response.success
            if not success:
                break
            all_slots.extend(group_response.context.get("slots", []))
            if group_response.data:
                all_chunks.extend(pickle.loads(group_response.data))

        session = self._ctx.session_manager.get_or_create(key.request_id)
        session.extras["retrieve_start_time"] = time.perf_counter()
        data = pickle.dumps(all_chunks) if all_chunks else b""
        return PrepareRetrieveResponse(
            success=success, data=data, context={"slots": all_slots}
        )

    @_lmcache_nvtx_annotate
    def commit_retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
    ) -> bool:
        """Finalize a retrieve operation.

        Every group's SHM read locks were accumulated under the same
        transfer key by ``ShmTransferStrategy.prepare_retrieve`` (see its
        per-group accumulation note), so a single ``commit_retrieve`` call
        releases all of them in one pop.

        Args:
            key: Cache key (unused for pickle).
            instance_id: Worker instance identifier (unused for pickle).

        Returns:
            ``True`` when the retrieve finalizes successfully.
        """
        _, strategy = self._resolve_for_transfer(instance_id)
        session = self._ctx.session_manager.get_or_create(key.request_id)
        st = session.extras.pop("retrieve_start_time", None)
        result = strategy.commit_retrieve(key=key, instance_id=instance_id)
        if st is not None:
            num_tokens = (
                len(self._resolve_obj_keys(key, instance_id)) * self._ctx.chunk_size
            )
            logger.info(
                "Retrieved %d tokens in %.3f seconds",
                num_tokens,
                time.perf_counter() - st,
            )
        return result
