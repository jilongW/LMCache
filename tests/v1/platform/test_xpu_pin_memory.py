# SPDX-License-Identifier: Apache-2.0
"""Hardware tests for XPU SYCL host-memory registration."""

# Standard
import ctypes
import mmap

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type

if not (torch_device_type == "xpu" and torch_dev.is_available()):
    pytest.skip("Requires an available XPU runtime", allow_module_level=True)

# First Party
from lmcache.v1.platform.xpu import XpuDeviceSpec  # noqa: E402
from lmcache.v1.platform.xpu.pin_memory import XpuPinMemoryBackend  # noqa: E402

_SIZE = 8 << 20


def _region() -> tuple[mmap.mmap, torch.Tensor, int]:
    region = mmap.mmap(-1, _SIZE)
    host = torch.frombuffer(region, dtype=torch.uint8)
    ptr = ctypes.addressof(ctypes.c_char.from_buffer(region))
    return region, host, ptr


def test_device_spec_selects_native_backend() -> None:
    assert XpuDeviceSpec().pin_memory_backend is XpuPinMemoryBackend


def test_register_round_trip_and_unregister() -> None:
    backend = XpuPinMemoryBackend()
    assert backend.is_pin_supported
    region, host, ptr = _region()
    host.copy_(torch.randint(0, 255, (_SIZE,), dtype=torch.uint8))
    try:
        assert backend.pin_memory(ptr, _SIZE)
        device = host.to(torch_device_type)
        assert torch.equal(device.cpu(), host)
        back = torch.zeros(_SIZE, dtype=torch.uint8)
        # D2H into a registered range must also be correct.
        host.zero_()
        host.copy_(device)
        torch_dev.synchronize()
        assert torch.equal(host, device.cpu())
        del back
    finally:
        assert backend.unpin_memory(ptr)
        del host
        region.close()


@pytest.mark.parametrize("ptr,size", [(0, 4096), (4096, 0), (4096, -1)])
def test_invalid_ranges_return_false(ptr: int, size: int) -> None:
    assert XpuPinMemoryBackend().pin_memory(ptr, size) is False


def test_unpin_null_pointer_returns_false() -> None:
    assert XpuPinMemoryBackend().unpin_memory(0) is False
