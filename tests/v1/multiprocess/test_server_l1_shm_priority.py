# SPDX-License-Identifier: Apache-2.0
"""On XPU the SHM transport wins over lazy L1 allocation."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.server import _prefer_shm_over_lazy_l1

_SIZE = 15 << 30


def _cfg(*, use_lazy: bool, shm_name: str) -> L1MemoryManagerConfig:
    cfg = L1MemoryManagerConfig(
        size_in_bytes=_SIZE, use_lazy=False, shm_name=shm_name
    )
    # Set after construction: validation may already have turned lazy off on
    # platforms that cannot pin host memory, which is not what is under test.
    cfg.use_lazy = use_lazy
    return cfg


def _pool(cfg: L1MemoryManagerConfig) -> dict:
    manager = SimpleNamespace(memory_config=cfg)
    storage = SimpleNamespace(l1_manager_config=manager)
    return MPCacheServerContext._compute_shm_pool_info(storage)  # noqa: SLF001


def test_xpu_with_shm_turns_lazy_off_and_keeps_the_shm_pool() -> None:
    cfg = _cfg(use_lazy=True, shm_name="lmcache_l1_pool_server")
    assert _pool(cfg)["shm_name"] == "", "lazy L1 silently disables SHM"

    _prefer_shm_over_lazy_l1(cfg, "xpu")

    assert cfg.use_lazy is False
    info = _pool(cfg)
    assert info["shm_name"] == "lmcache_l1_pool_server"
    assert info["pool_size"] >= _SIZE
    assert info["scratch_offset"] == _SIZE


def test_xpu_without_shm_keeps_lazy() -> None:
    cfg = _cfg(use_lazy=True, shm_name="")
    _prefer_shm_over_lazy_l1(cfg, "xpu")
    assert cfg.use_lazy is True


def test_xpu_already_eager_is_untouched() -> None:
    cfg = _cfg(use_lazy=False, shm_name="lmcache_l1_pool_server")
    _prefer_shm_over_lazy_l1(cfg, "xpu")
    assert cfg.use_lazy is False


@pytest.mark.parametrize("device_type", ["cuda", "cpu", "npu", "hpu"])
def test_other_platforms_keep_their_lazy_setting(device_type: str) -> None:
    cfg = _cfg(use_lazy=True, shm_name="lmcache_l1_pool_server")
    _prefer_shm_over_lazy_l1(cfg, device_type)
    assert cfg.use_lazy is True
