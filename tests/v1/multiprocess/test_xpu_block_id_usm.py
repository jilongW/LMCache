# SPDX-License-Identifier: Apache-2.0
"""Focused XPU test for the compact block-ID USM transfer path.

Run directly for timing output::

    python tests/v1/multiprocess/test_xpu_block_id_usm.py

This exercises the native XPU binding without starting vLLM or EngineCore.
"""

from __future__ import annotations

import time

import pytest
import torch

from lmcache import torch_dev, torch_device_type
from lmcache.lmcache_native import EngineKVFormat
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
)


if not (torch_device_type == "xpu" and torch_dev.is_available()):
    pytest.skip("Requires an available XPU runtime", allow_module_level=True)


_NUM_BLOCKS = 16
_NUM_LAYERS = 8
_BLOCK_SIZE = 1024
_NUM_HEADS = 4
_HEAD_SIZE = 256
_BLOCKS_PER_CHUNK = 1
_FORMAT = EngineKVFormat.NL_X_NB_TWO_BS_NH_HS


def _make_source() -> dict[str, torch.Tensor]:
    source: dict[str, torch.Tensor] = {}
    for layer_idx in range(_NUM_LAYERS):
        tensor = torch.empty(
            _NUM_BLOCKS,
            2,
            _BLOCK_SIZE,
            _NUM_HEADS,
            _HEAD_SIZE,
            dtype=torch.float16,
            device="xpu",
        )
        for block_idx in range(_NUM_BLOCKS):
            tensor[block_idx].fill_(block_idx + layer_idx / 100.0)
        source[f"layer_{layer_idx}"] = tensor
    torch_dev.synchronize()
    return source


def _gather(
    source: dict[str, torch.Tensor], block_id: int
) -> tuple[float, torch.Tensor]:
    output = torch.empty(
        2,
        _NUM_LAYERS,
        _BLOCK_SIZE,
        _NUM_HEADS * _HEAD_SIZE,
        dtype=torch.float16,
        device="cpu",
    )
    start = time.perf_counter()
    gather_paged_kv_to_cpu(
        source,
        [block_id],
        _BLOCKS_PER_CHUNK,
        engine_kv_format=_FORMAT,
        out=[output],
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, output


def test_compact_block_ids_usm_round_trip() -> None:
    source = _make_source()
    _gather(source, 0)

    elapsed_ms, output = _gather(source, 7)
    for layer_idx in range(_NUM_LAYERS):
        expected = torch.full_like(
            output[:, layer_idx], 7 + layer_idx / 100.0
        )
        assert torch.equal(output[:, layer_idx], expected)

    print(f"steady_state_block_id_transfer_ms={elapsed_ms:.3f}", flush=True)


if __name__ == "__main__":
    source = _make_source()
    for block_id in (0, 7, 13, 3):
        elapsed_ms, _ = _gather(source, block_id)
        print(f"block_id={block_id} transfer_ms={elapsed_ms:.3f}", flush=True)