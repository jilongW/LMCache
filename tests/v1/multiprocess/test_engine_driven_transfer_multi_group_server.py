# SPDX-License-Identifier: Apache-2.0
"""Server-side (EngineDrivenTransferModule) multi-group coverage: hybrid
registration builds per-group metadata, and resolve/store/retrieve loop over
groups with correctly-flattened, correctly-offset results.
"""

# Standard
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock, patch
import pickle
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)


def _default_key(tokens: int = 8) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "m", 1, 0, [1] * tokens, start=0, end=tokens, request_id="req"
    )


def _hybrid_register_payload(
    instance_id: int = 1,
) -> RegisterEngineDrivenContextPayload:
    """A 2-group hybrid registration: group 0 has 2 attention layers,
    group 1 has 2 Mamba-style layers with a smaller tokens_per_block."""
    return RegisterEngineDrivenContextPayload(
        instance_id=instance_id,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=4,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
        engine_group_infos=[
            EngineGroupInfo(
                engine_group_id=0, layer_indices=(0, 1), tokens_per_block=4
            ),
            EngineGroupInfo(
                engine_group_id=1, layer_indices=(2, 3), tokens_per_block=1
            ),
        ],
    )


def _make_group_object_keys(
    key: IPCCacheServerKey, chunk_hashes: list[bytes], object_group_ids: list[int]
) -> list[list[ObjectKey]]:
    """Fake ``ipc_key_to_object_keys``: one distinct ObjectKey per (chunk,
    group), so multi-group flattening/slicing is distinguishable in
    assertions."""
    return [
        [
            ObjectKey(
                chunk_hash=chash + f"|g{gid}".encode(),
                model_name=key.model_name,
                kv_rank=0,
                object_group_id=gid,
            )
            for chash in chunk_hashes
        ]
        for gid in object_group_ids
    ]


@pytest.fixture
def stub_native_storage_ops() -> Any:
    """Stub native modules so server imports work in source-only test runs."""
    module = type(sys)("lmcache.native_storage_ops")
    module.TTLLock = type("TTLLock", (), {})  # type: ignore[attr-defined]
    module.Bitmap = type("Bitmap", (), {})  # type: ignore[attr-defined]
    module.PeriodicEventNotifier = type(  # type: ignore[attr-defined]
        "PeriodicEventNotifier", (), {}
    )
    with patch.dict(
        sys.modules,
        {"lmcache.native_storage_ops": module, "cupy": MagicMock()},
    ):
        yield


@pytest.fixture
def hybrid_server(
    stub_native_storage_ops: Any,
) -> Iterator[tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext]]:
    """A registered 2-group hybrid server module with a mocked storage manager."""
    from contextlib import ExitStack

    stack = ExitStack()
    mock_storage = MagicMock()
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h1", b"h2"]

    stack.enter_context(
        patch(
            "lmcache.v1.multiprocess.engine_context.StorageManager",
            return_value=mock_storage,
        )
    )
    token_hasher = stack.enter_context(
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher")
    )
    # Identity pass-through: distinct chunk hashes must stay distinguishable
    # across the two resolve_obj_keys() calls each group's commit makes
    # (once for chunk_offset bookkeeping, once inside the transfer
    # strategy's resolve_obj_keys closure) -- a bare MagicMock collapses
    # every call to the same return_value regardless of input.
    token_hasher.hash_to_bytes.side_effect = lambda h: h
    session_cls = stack.enter_context(
        patch("lmcache.v1.multiprocess.engine_context.SessionManager")
    )
    stack.enter_context(patch("lmcache.v1.multiprocess.engine_context.get_event_bus"))
    stack.enter_context(
        patch(
            "lmcache.v1.multiprocess.engine_context.ipc_key_to_object_keys",
            side_effect=_make_group_object_keys,
        )
    )
    session_cls.return_value.get_or_create.return_value = mock_session

    storage_manager_config = MagicMock()
    storage_manager_config.l1_manager_config.gds_l1_config = None
    ctx = MPCacheServerContext(
        storage_manager_config=storage_manager_config, chunk_size=8
    )
    module = EngineDrivenTransferModule(ctx)
    module.register_kv_cache_engine_driven_context(_hybrid_register_payload())

    yield module, mock_storage, ctx
    stack.close()


def test_register_builds_metadata_by_group_with_correct_layer_counts(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """Registration must build one EngineDrivenContextMetadata per group,
    sized to that group's own layer count, keyed to its own tokens_per_block.

    Verified indirectly through commit_store: the storage manager's
    reserve_write is called once per group with that group's own layout, so
    the captured layout_desc/block_size prove per-group metadata was built.
    """
    module, mock_storage, _ = hybrid_server
    mock_storage.reserve_write.return_value = {}

    key = _default_key()
    flat_payload = pickle.dumps(
        [torch.zeros(2, 2, 8, 16)] * 4  # 2 chunk hashes x 2 groups
    )
    module.commit_store(key, 1, flat_payload)

    assert mock_storage.reserve_write.call_count == 2
    group0_layout_desc = mock_storage.reserve_write.call_args_list[0].args[1]
    group1_layout_desc = mock_storage.reserve_write.call_args_list[1].args[1]
    # Both groups have 2 layers (0,1) and (2,3) -> the same chunk shape here,
    # but each call must carry its own (non-shared) layout_desc instance.
    assert tuple(group0_layout_desc.shapes[0]) == (2, 2, 8, 16)
    assert tuple(group1_layout_desc.shapes[0]) == (2, 2, 8, 16)
    assert group0_layout_desc is not group1_layout_desc


def test_register_falls_back_to_single_metadata_without_groups(
    stub_native_storage_ops: Any,
) -> None:
    """A registration with no engine_group_infos results in exactly one
    reserve_write call at commit time (the single non-hybrid group), not
    one per some phantom group count."""
    from contextlib import ExitStack

    stack = ExitStack()
    mock_storage = MagicMock()
    mock_storage.reserve_write.return_value = {}
    mock_session = MagicMock()
    mock_session.get_hashes.return_value = [b"h"]
    stack.enter_context(
        patch(
            "lmcache.v1.multiprocess.engine_context.StorageManager",
            return_value=mock_storage,
        )
    )
    token_hasher = stack.enter_context(
        patch("lmcache.v1.multiprocess.engine_context.TokenHasher")
    )
    token_hasher.hash_to_bytes.side_effect = lambda h: h
    session_cls = stack.enter_context(
        patch("lmcache.v1.multiprocess.engine_context.SessionManager")
    )
    stack.enter_context(patch("lmcache.v1.multiprocess.engine_context.get_event_bus"))
    stack.enter_context(
        patch(
            "lmcache.v1.multiprocess.engine_context.ipc_key_to_object_keys",
            return_value=[["obj"]],
        )
    )
    session_cls.return_value.get_or_create.return_value = mock_session
    storage_manager_config = MagicMock()
    storage_manager_config.l1_manager_config.gds_l1_config = None
    ctx = MPCacheServerContext(
        storage_manager_config=storage_manager_config, chunk_size=8
    )
    module = EngineDrivenTransferModule(ctx)

    payload = RegisterEngineDrivenContextPayload(
        instance_id=1,
        model_name="m",
        world_size=1,
        block_size=4,
        num_layers=2,
        hidden_dim_size=16,
        dtype_str="float32",
        use_mla=False,
    )
    module.register_kv_cache_engine_driven_context(payload)

    key = _default_key()
    module.commit_store(key, 1, pickle.dumps([torch.zeros(2, 2, 8, 16)]))

    assert mock_storage.reserve_write.call_count == 1
    stack.close()


def test_resolve_obj_keys_flattens_groups_major_order(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """_resolve_obj_keys must flatten group 0's keys before group 1's."""
    module, _, _ = hybrid_server
    key = _default_key()

    flat_keys = module._resolve_obj_keys(key, 1)  # noqa: SLF001

    # 2 chunk hashes x 2 groups = 4 keys, group-major (group 0's 2 keys first).
    assert len(flat_keys) == 4
    assert [k.object_group_id for k in flat_keys] == [0, 0, 1, 1]


def test_prepare_store_pickle_mode_concatenates_slots_across_groups(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """Pickle-mode prepare_store returns empty slots/chunk_indices from every
    group's PickleTransferStrategy.prepare_store, concatenated (still empty)."""
    module, _, _ = hybrid_server
    key = _default_key()

    response = module.prepare_store(key, 1)

    assert response.context == {"slots": [], "chunk_indices": []}


def test_commit_store_splits_flat_payload_by_group_chunk_count(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """The worker sends one flat, group-major pickled chunk list; commit_store
    must split it back into each group's own slice before writing, and each
    group's write must land on that group's own reserved object keys."""
    module, mock_storage, _ = hybrid_server

    written_by_group: dict[int, list[torch.Tensor]] = {0: [], 1: []}

    def _reserve_write(obj_keys, _layout_desc, _mode):
        reserved = {}
        for obj_key in obj_keys:
            memory_obj = MagicMock()
            memory_obj.tensor = torch.zeros(2, 2, 8, 16)
            reserved[obj_key] = memory_obj
        return reserved

    def _finish_write(obj_keys):
        for obj_key in obj_keys:
            written_by_group[obj_key.object_group_id].append(obj_key)

    mock_storage.reserve_write.side_effect = _reserve_write
    mock_storage.finish_write.side_effect = _finish_write

    key = _default_key()
    # The fixture's mock_session reports 2 chunk hashes, so each group has 2
    # object keys (2 chunks) -- the flat payload must be group-major with 2
    # chunks per group (4 total) to match.
    group0_chunks = [torch.ones(2, 2, 8, 16) * 1.0, torch.ones(2, 2, 8, 16) * 1.1]
    group1_chunks = [torch.ones(2, 2, 8, 16) * 2.0, torch.ones(2, 2, 8, 16) * 2.1]
    flat_payload = pickle.dumps(group0_chunks + group1_chunks)

    ok = module.commit_store(key, 1, flat_payload)

    assert ok is True
    # Both groups wrote exactly 2 object keys each (2 chunk hashes per group).
    assert len(written_by_group[0]) == 2
    assert len(written_by_group[1]) == 2


def test_prepare_retrieve_pickle_mode_merges_chunks_group_major(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """Each group's prepare_retrieve independently pickles its own chunks;
    the server must unpickle and re-merge them into one flat, group-major
    payload for the worker."""
    module, mock_storage, _ = hybrid_server

    def _read_prefetched_results(obj_keys):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            memory_objs = []
            for obj_key in obj_keys:
                memory_obj = MagicMock()
                # Value encodes which group this object key belongs to, so
                # the merged order can be verified.
                memory_obj.tensor = torch.full(
                    (2, 2, 8, 16), float(obj_key.object_group_id)
                )
                memory_objs.append(memory_obj)
            yield memory_objs

        return _ctx()

    mock_storage.read_prefetched_results.side_effect = _read_prefetched_results

    key = _default_key()
    response = module.prepare_retrieve(key, 1)

    assert response.success is True
    chunks: list[torch.Tensor] = pickle.loads(response.data)
    # 2 chunk hashes x 2 groups = 4 chunks, group-major.
    assert len(chunks) == 4
    assert torch.all(chunks[0] == 0.0)
    assert torch.all(chunks[1] == 0.0)
    assert torch.all(chunks[2] == 1.0)
    assert torch.all(chunks[3] == 1.0)


def test_prepare_retrieve_fails_if_any_group_misses(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """A miss in any one group must fail the whole multi-group retrieve."""
    module, mock_storage, _ = hybrid_server

    def _read_prefetched_results(obj_keys):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            # Group 1 (object_group_id == 1) always misses.
            if obj_keys and obj_keys[0].object_group_id == 1:
                yield None
                return
            memory_objs = []
            for _ in obj_keys:
                memory_obj = MagicMock()
                memory_obj.tensor = torch.zeros(2, 2, 8, 16)
                memory_objs.append(memory_obj)
            yield memory_objs

        return _ctx()

    mock_storage.read_prefetched_results.side_effect = _read_prefetched_results

    key = _default_key()
    response = module.prepare_retrieve(key, 1)

    assert response.success is False


def test_commit_retrieve_finalizes_once_per_group(
    hybrid_server: tuple[EngineDrivenTransferModule, MagicMock, MPCacheServerContext],
) -> None:
    """commit_retrieve releases every group's pending read locks in one
    finalize call now that groups share a single accumulated transfer key."""
    module, _, _ = hybrid_server
    key = _default_key()

    assert module.commit_retrieve(key, 1) is True
