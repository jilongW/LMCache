# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

# Third Party
import msgspec
import torch

# First Party
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.platform.base.ipc_wrapper import (  # noqa: E402,F401
    DeviceIPCWrapper,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc

"""
Defines the types and the customized encoder/decoders for inter-process
communications.

Key Types:
- IPCCacheServerKey: Token-based cache key
  - Contains token_ids, start, end, request_id (all required)
  - Converted to ObjectKey for storage operations via ipc_key_to_object_keys()
"""


@dataclass(order=True, frozen=True)
class IPCCacheServerKey:
    """Cache key for the IPC (multiprocess) protocol.

    This key type is sent by the client over ZMQ (serialized via msgspec).

    The client sends token_ids, start, end, and request_id (all required).
    The server computes chunk hashes via TokenHasher and converts to
    ObjectKey for storage operations using ipc_key_to_object_keys().

    The request_id field is for session tracking and is NOT included
    in equality/hash comparisons (two keys with same content but different
    request_ids are considered equal for cache purposes).
    """

    model_name: str
    world_size: int
    worker_id: int | None

    token_ids: tuple[int, ...]  # frozen tuple for hashability
    start: int
    end: int

    # === Session tracking (not part of cache identity) ===
    request_id: str = field(compare=False)

    # === Per-user isolation salt (part of cache identity) ===
    # msgspec encodes dataclasses as maps, so forward wire compatibility
    # works by field name: an old payload without ``cache_salt`` decodes
    # on new code using the default "". Placing the field last is a style
    # choice — all defaulted fields must come after non-defaulted ones.
    #
    # Invariant: must not contain ``@``, ``/``, ``\``, or NUL, and
    # must be <= 128 chars — same rationale as ObjectKey (see
    # ObjectKey.cache_salt). Validated in __post_init__.
    cache_salt: str = ""

    # Request-scoped LMCache configuration passed across the IPC boundary.
    # It is metadata, not part of cache identity.
    request_configs: dict[str, Any] | None = field(default=None, compare=False)

    # Number of workers that retrieve this key's object; the server reserves
    # that many read locks (see ``require_num_kv_readers``). 0 = not sent;
    # lookups reject it.
    num_kv_readers: int = field(default=0, compare=False)

    # ``[group_id][chunk_position]``; ``True`` marks a chunk backed entirely
    # by vLLM's null block, which the server skips on store (``None`` = all
    # chunks real).
    null_chunk_mask: tuple[tuple[bool, ...], ...] | None = field(
        default=None, compare=False
    )

    # Duplicated from ObjectKey — cannot import ObjectKey here due to
    # circular dependency (api.py imports IPCCacheServerKey).
    _SALT_FORBIDDEN_CHARS = frozenset("@/\\\x00")
    _SALT_MAX_LEN = 128

    def __post_init__(self) -> None:
        bad = self._SALT_FORBIDDEN_CHARS & set(self.cache_salt)
        if bad:
            raise ValueError(
                f"cache_salt must not contain {bad!r} (got {self.cache_salt!r})"
            )
        if len(self.cache_salt) > self._SALT_MAX_LEN:
            raise ValueError(
                f"cache_salt exceeds max length {self._SALT_MAX_LEN} "
                f"(got {len(self.cache_salt)})"
            )

    # Helper function for unit tests only
    @classmethod
    def from_token_ids(
        cls,
        model_name: str,
        world_size: int,
        worker_id: int | None,
        token_ids: list[int],
        start: int = 0,
        end: int = 0,
        request_id: str = "",
        cache_salt: str = "",
        num_kv_readers: int = 1,
        request_configs: dict[str, Any] | None = None,
    ) -> "IPCCacheServerKey":
        """Create a key from token ids. Only used by the tests."""
        return cls(
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            num_kv_readers=num_kv_readers,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
            request_configs=request_configs,
        )

    def require_num_kv_readers(self) -> int:
        """Declared reader count; rejects keys from pre-field clients.

        Each reader's retrieve releases one read lock, so the count must
        be exact: under-counting unpins an object mid-copy; over-counting
        only holds it to the TTL. 0 means the field was never sent --
        rejected, not guessed.
        """
        if self.num_kv_readers < 1:
            raise ValueError(
                f"num_kv_readers={self.num_kv_readers}: this server "
                "requires clients that send "
                "IPCCacheServerKey.num_kv_readers. Upgrade the LMCache "
                "client."
            )
        return self.num_kv_readers

    def no_worker_id_version(self) -> "IPCCacheServerKey":
        """Create a copy with worker_id=None for lookup requests."""
        return IPCCacheServerKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            num_kv_readers=self.num_kv_readers,
            token_ids=self.token_ids,
            start=self.start,
            end=self.end,
            request_id=self.request_id,
            cache_salt=self.cache_salt,
            request_configs=self.request_configs,
        )


# Type exports
KVCache = list[DeviceIPCWrapper]


class SerializedMemoryLayoutDesc(msgspec.Struct, frozen=True):
    """Message-pack-safe mirror of ``MemoryLayoutDesc``.

    ``MemoryLayoutDesc`` (``lmcache.v1.distributed.api``) holds real
    ``torch.Size``/``torch.dtype`` objects, which msgspec cannot encode when
    nested inside another struct (the ``torch.dtype``/``torch.Size``
    encode/decode hooks in this module only apply when ``MemoryLayoutDesc``
    is itself the top-level payload/response class for an RPC, not when it
    is a field of another struct -- see ``mq.py``'s
    ``_SPECIAL_ENCODER_DECODERS``). This struct carries the same
    information with wire-safe primitives so it can be embedded in
    ``RegisterEngineDrivenContextPayload.group_layout_descs``.

    Attributes:
        shapes: One entry per tensor in the described layout, as plain
            ``list[int]`` (mirrors ``MemoryLayoutDesc.shapes``).
        dtypes: One entry per tensor, as ``str(torch.dtype)`` (mirrors
            ``MemoryLayoutDesc.dtypes``).
    """

    shapes: list[list[int]]
    dtypes: list[str]


def serialize_memory_layout_desc(
    layout_desc: "MemoryLayoutDesc",
) -> SerializedMemoryLayoutDesc:
    """Encode a memory layout for the engine-driven registration payload.

    Args:
        layout_desc: The layout to encode.

    Returns:
        A wire-safe ``SerializedMemoryLayoutDesc`` with the same shapes and
        dtypes.
    """
    return SerializedMemoryLayoutDesc(
        shapes=[list(shape) for shape in layout_desc.shapes],
        dtypes=[str(dtype) for dtype in layout_desc.dtypes],
    )


def deserialize_memory_layout_desc(
    payload: SerializedMemoryLayoutDesc,
) -> "MemoryLayoutDesc":
    """Decode a serialized layout descriptor from an IPC payload.

    Args:
        payload: The wire-safe struct produced by
            ``serialize_memory_layout_desc``.

    Returns:
        The equivalent ``MemoryLayoutDesc`` with real ``torch.Size``/
        ``torch.dtype`` values.

    Raises:
        ValueError: If any entry in ``payload.dtypes`` does not name a
            valid ``torch.dtype``.
    """
    from lmcache.v1.distributed.api import MemoryLayoutDesc

    dtypes: list[torch.dtype] = []
    for dtype_name in payload.dtypes:
        dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported torch dtype in payload: {dtype_name}")
        dtypes.append(dtype)
    return MemoryLayoutDesc(
        shapes=[torch.Size(shape) for shape in payload.shapes], dtypes=dtypes
    )


class RegisterEngineDrivenContextPayload(msgspec.Struct):
    """Payload for the REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT protocol message.

    Attributes:
        instance_id: Worker instance identifier (typically PID).
        model_name: Model name associated with this worker.
        world_size: Worker world size used in cache keys.
        block_size: Tokens per paged block.
        num_layers: Number of model layers.
        hidden_dim_size: Flattened hidden dimension per token.
        dtype_str: Torch dtype name (e.g. ``"float16"``).
        use_mla: Whether the worker KV format is MLA.
        num_physical_slots: Number of physical KV slots gathered into one
            LMCache chunk. ``None`` accepts the legacy protocol, where the
            server assumed one physical slot per logical token.
        engine_group_infos: One entry per KV cache group, in protocol order,
            giving each group its own layout for hybrid models. Empty means
            a single non-hybrid group.
        group_hidden_dim_sizes: Per-group override of ``hidden_dim_size``,
            aligned with ``engine_group_infos``. Missing entries fall back
            to the shared ``hidden_dim_size``.
        group_layout_descs: Per-group override of the layout descriptor the
            server would otherwise derive from ``hidden_dim_size``/
            ``group_hidden_dim_sizes``, aligned with ``engine_group_infos``.
            An entry is non-``None`` only for a group the worker has
            decided to quantize (KVWeave), where it carries the encoded
            ``uint8`` byte layout so the server allocates SHM chunks sized
            for the quantized payload rather than the raw KV tensor.
            ``None`` (whole field or a given entry) means the server
            derives the layout as before.
        enable_l1_kvweave_quant: Worker's declared intent to quantize at
            least one group's chunks for this registration. The server is
            the sole authority on whether this is actually permitted --
            see ``StorageManager.is_l1_variable_size()`` -- and rejects
            registration via ``RegisterEngineDrivenContextResponse.error``
            when it is not.
    """

    instance_id: int
    model_name: str
    world_size: int
    block_size: int
    num_layers: int
    hidden_dim_size: int
    dtype_str: str
    use_mla: bool
    num_physical_slots: int | None = None
    engine_group_infos: list[EngineGroupInfo] = msgspec.field(default_factory=list)
    group_hidden_dim_sizes: list[int] | None = None
    group_layout_descs: list[SerializedMemoryLayoutDesc | None] | None = None
    enable_l1_kvweave_quant: bool = False


@dataclass
class RegisterEngineDrivenContextResponse:
    """Shared response for engine-driven context registration.

    Attributes:
        shm_name: Name of the shared-memory pool to attach to. Only
            meaningful when ``error`` is ``None``.
        pool_size: Size in bytes of the shared-memory pool. Only
            meaningful when ``error`` is ``None``.
        error: Human-readable rejection reason set when the server declines
            this registration (e.g. quantization requested on a
            fixed-size L1). ``None`` on success. Callers must check this
            field before trusting ``shm_name``/``pool_size``.
    """

    shm_name: str = ""
    pool_size: int = 0
    scratch_offset: int = 0
    scratch_size: int = 0
    error: str | None = None


@dataclass
class PrepareStoreResponse:
    """Shared response for an engine-driven store preparation."""

    context: dict = field(default_factory=dict)


@dataclass
class PrepareRetrieveResponse:
    """Shared response for an engine-driven retrieve preparation."""

    success: bool
    data: bytes = b""
    context: dict = field(default_factory=dict)


@dataclass
class CustomizedSerdeConfig:
    serializer: Callable[[Any], bytes]
    deserializer: Callable[[bytes], Any]
    code: int


@dataclass
class BlockAllocationRecord:
    """A single per-request GPU block allocation delta from vLLM."""

    req_id: str
    new_block_ids: list[int]
    new_token_ids: list[int]


@dataclass
class CBMatchResult:
    """Result of a sub-sequence match from BlendTokenRangeMatcher.

    Attributes:
        old_st: Start position in the originally registered (stored) sequence.
        old_ed: End position in the originally registered (stored) sequence.
        cur_st: Start position in the query sequence where the match was found.
        cur_ed: End position in the query sequence where the match was found.
        hash: Token hash bytes (from registration) used as the storage key.
    """

    old_st: int
    old_ed: int
    cur_st: int
    cur_ed: int
    hash: bytes


@dataclass
class CBUnifiedLookupResult:
    """Resolved payload of ``CB_UNIFIED_LOOKUP``: prefix lookup + non-prefix
    fingerprint match, reconciled in one RPC. The RPC returns ``None`` (not this)
    while either leg's KV is still loading into L1; this type is sent only once
    both are resident.

    Attributes:
        prefix_coverage_tokens: Contiguous prefix-cache coverage (L1+L2) in
            tokens — what the standard LOOKUP would report.
        non_prefix_segments: Fingerprint matches outside the prefix coverage
            (cur_st order), each carrying ``(old_st, old_ed, cur_st, cur_ed,
            hash)``. Already sparse-prefetched, so the retrieve set equals the
            prefetched set. Includes fleet-coordinator (shared-L2) matches:
            those are merged in before the sparse prefetch -- prefix-covered and
            locally-duplicated ones dropped -- so they ride the identical
            prefetch + retrieve path and need no separate handling.
        segmented_prefix_segments: Post-gap chunks retained by the
            ``SEGMENTED_PREFIX`` prefix leg (beyond ``count_leading_ones``) — at
            their original positions (``old_st == cur_st``), so the connector
            tags them ``prefix`` (pure load, no recompute) and only the gap is
            recomputed. Sourced from the prefix bitmap, not the fingerprint
            matcher; empty when ``SEGMENTED_PREFIX`` is off.
    """

    prefix_coverage_tokens: int
    non_prefix_segments: list[CBMatchResult]
    segmented_prefix_segments: list[CBMatchResult] = field(default_factory=list)


_CUSTOMERIZED_SERIALIZERS = {
    DeviceIPCWrapper: CustomizedSerdeConfig(
        serializer=DeviceIPCWrapper.Serialize,
        deserializer=DeviceIPCWrapper.Deserialize,
        code=1,
    ),
}


def get_customized_encoder(type: Any) -> msgspec.msgpack.Encoder:
    # TODO: `type` is not used here
    def enc_hook(obj: Any) -> Any:
        for supported_type, cfg in _CUSTOMERIZED_SERIALIZERS.items():
            if isinstance(obj, supported_type):
                data = cfg.serializer(obj)
                return msgspec.msgpack.Ext(cfg.code, data)
        if isinstance(obj, torch.dtype):
            return str(obj).removeprefix("torch.")
        if isinstance(obj, torch.Size):
            return list(obj)
        raise TypeError(f"Unsupported type for serialization: {type(obj)}")

    return msgspec.msgpack.Encoder(enc_hook=enc_hook)


def get_customized_decoder(type: Any) -> msgspec.msgpack.Decoder:
    def ext_hook(code: int, data: bytes) -> Any:
        for cfg in _CUSTOMERIZED_SERIALIZERS.values():
            if cfg.code == code:
                return cfg.deserializer(data)
        raise TypeError(f"Unsupported ext code for deserialization: {code}")

    def dec_hook(expected_type: type, obj: Any) -> Any:
        if expected_type is torch.dtype:
            return getattr(torch, obj)
        if expected_type is torch.Size:
            return torch.Size(obj)
        if isinstance(obj, expected_type):
            return obj
        raise NotImplementedError(
            f"Unsupported type for deserialization: {expected_type}"
        )

    return msgspec.msgpack.Decoder(ext_hook=ext_hook, dec_hook=dec_hook, type=type)
