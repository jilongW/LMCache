# SPDX-License-Identifier: Apache-2.0
"""Who pins the SHM-backed L1 pool, and when its pages become resident."""

# Standard
import os
import subprocess
import sys
import textwrap

# Third Party
import pytest

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.xpu import XpuDeviceSpec


def test_default_spec_pins_the_pool_in_the_server() -> None:
    assert DeviceSpec.pins_shm_pool_in_server.fget(object()) is True


def test_xpu_spec_leaves_the_pool_to_the_workers() -> None:
    assert XpuDeviceSpec().pins_shm_pool_in_server is False


@pytest.mark.skipif(
    not (torch_device_type == "xpu" and torch_dev.is_available()),
    reason="Requires an available XPU runtime",
)
@pytest.mark.parametrize("precommit_gb", ["0", "0.03125"])
def test_xpu_server_commits_only_the_precommit_prefix_until_a_worker_pins(
    precommit_gb: str,
) -> None:
    """The server commits just its precommit prefix; a worker's pin commits the rest."""
    script = textwrap.dedent(
        """
        import ctypes, os, sys
        from multiprocessing import shared_memory
        from lmcache.v1.platform import current_device_spec, torch_ops

        size = 64 << 20
        name = "lmcache_l1_pool_pin_test"
        ptr = torch_ops.alloc_shm_pinned_ptr(size, name)
        committed = lambda: os.stat("/dev/shm/" + name).st_blocks * 512
        server = committed()

        # A worker maps the same segment and pins its own mapping.
        shm = shared_memory.SharedMemory(name=name, create=False)
        wptr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
        assert current_device_spec.pin_memory(wptr, size)
        worker = committed()
        current_device_spec.unpin_memory(wptr)
        del wptr
        shm.close()
        torch_ops.free_shm_pinned_ptr(ptr, size, name)
        print(server, worker, size)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "LMCACHE_MP_SHM_PRECOMMIT_GB": precommit_gb},
    ).stdout.split()[-3:]
    server, worker, size = (int(x) for x in out)
    precommit = int(float(precommit_gb) * (1 << 30))
    assert precommit <= server < precommit + size // 8, (
        "server committed more or less than its precommit prefix"
    )
    assert worker >= size * 7 // 8, "worker pin did not commit the pool"
    assert not os.path.exists("/dev/shm/lmcache_l1_pool_pin_test")
