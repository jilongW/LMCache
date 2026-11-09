# SPDX-License-Identifier: Apache-2.0
"""Engine-driven SHM scratch reservation in ``MPCacheServerContext``."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.multiprocess import engine_context
from lmcache.v1.multiprocess.engine_context import (
    MPCacheServerContext,
    _engine_driven_scratch_size_bytes,
)

GIB = 1 << 30


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LMCACHE_MP_ENGINE_DRIVEN_SCRATCH_GB", raising=False)
    monkeypatch.delenv("LMCACHE_MP_L1_KVWEAVE_QUANT", raising=False)


def _shm_free(monkeypatch, free_bytes: int) -> None:
    monkeypatch.setattr(
        engine_context.os,
        "statvfs",
        lambda _path: SimpleNamespace(f_bavail=free_bytes // 4096, f_frsize=4096),
    )


def test_no_scratch_by_default(monkeypatch):
    """Without KVWeave a deployment keeps exactly the upstream SHM pool."""
    _shm_free(monkeypatch, 100 * GIB)
    assert _engine_driven_scratch_size_bytes(5 * GIB, "lmcache_l1_pool_x") == 0


def test_master_switch_enables_default_scratch(monkeypatch):
    _shm_free(monkeypatch, 100 * GIB)
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    assert _engine_driven_scratch_size_bytes(5 * GIB, "lmcache_l1_pool_x") == 5 * GIB


def test_explicit_size_overrides_master_switch(monkeypatch):
    _shm_free(monkeypatch, 100 * GIB)
    monkeypatch.setenv("LMCACHE_MP_L1_KVWEAVE_QUANT", "1")
    monkeypatch.setenv("LMCACHE_MP_ENGINE_DRIVEN_SCRATCH_GB", "0")
    assert _engine_driven_scratch_size_bytes(5 * GIB, "lmcache_l1_pool_x") == 0
    monkeypatch.setenv("LMCACHE_MP_ENGINE_DRIVEN_SCRATCH_GB", "2")
    assert _engine_driven_scratch_size_bytes(5 * GIB, "lmcache_l1_pool_x") == 2 * GIB


def test_explicit_size_works_without_master_switch(monkeypatch):
    _shm_free(monkeypatch, 100 * GIB)
    monkeypatch.setenv("LMCACHE_MP_ENGINE_DRIVEN_SCRATCH_GB", "1")
    assert _engine_driven_scratch_size_bytes(5 * GIB, "lmcache_l1_pool_x") == GIB


def test_invalid_size_disables_scratch(monkeypatch):
    _shm_free(monkeypatch, 100 * GIB)
    monkeypatch.setenv("LMCACHE_MP_ENGINE_DRIVEN_SCRATCH_GB", "lots")
    assert _engine_driven_scratch_size_bytes(5 * GIB, "lmcache_l1_pool_x") == 0


def test_scratch_is_capped_by_free_shm_after_l1(monkeypatch):
    _shm_free(monkeypatch, 6 * GIB)
    monkeypatch.setenv("LMCACHE_MP_ENGINE_DRIVEN_SCRATCH_GB", "5")
    # 6 GiB free, 5 GiB goes to L1, so only 1 GiB is left for scratch.
    assert _engine_driven_scratch_size_bytes(5 * GIB, "lmcache_l1_pool_x") == GIB


def test_no_scratch_without_shm(monkeypatch):
    _shm_free(monkeypatch, 100 * GIB)
    monkeypatch.setenv("LMCACHE_MP_ENGINE_DRIVEN_SCRATCH_GB", "5")
    assert _engine_driven_scratch_size_bytes(5 * GIB, "") == 0


def _storage_config(**memory_kwargs):
    memory = L1MemoryManagerConfig(
        size_in_bytes=5 * GIB, use_lazy=False, **memory_kwargs
    )
    return SimpleNamespace(
        l1_manager_configs=[SimpleNamespace(memory_config=memory)],
        l1_manager_config=SimpleNamespace(memory_config=memory),
    )


def test_pool_info_keeps_upstream_shape_without_scratch(monkeypatch):
    _shm_free(monkeypatch, 100 * GIB)
    info = MPCacheServerContext._compute_shm_pool_info(
        _storage_config(shm_name="lmcache_l1_pool_t")
    )
    assert info == {"shm_name": "lmcache_l1_pool_t", "pool_size": 5 * GIB}


def test_pool_info_reports_scratch_tail(monkeypatch):
    _shm_free(monkeypatch, 100 * GIB)
    monkeypatch.setenv("LMCACHE_MP_ENGINE_DRIVEN_SCRATCH_GB", "2")
    info = MPCacheServerContext._compute_shm_pool_info(
        _storage_config(shm_name="lmcache_l1_pool_t")
    )
    assert info == {
        "shm_name": "lmcache_l1_pool_t",
        "pool_size": 7 * GIB,
        "scratch_offset": 5 * GIB,
        "scratch_size": 2 * GIB,
    }


def test_pool_info_has_no_scratch_without_shm(monkeypatch):
    _shm_free(monkeypatch, 100 * GIB)
    monkeypatch.setenv("LMCACHE_MP_ENGINE_DRIVEN_SCRATCH_GB", "2")
    info = MPCacheServerContext._compute_shm_pool_info(_storage_config(shm_name=""))
    assert info == {"shm_name": "", "pool_size": 0}
