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
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.serde.kvweave.kvweave_config import KVWeaveRuntimeConfig
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import KVWeaveCodec
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.transfer_context.base import (
    gather_paged_kv_to_cpu,
    scatter_cpu_to_paged_kv,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    _build_group_transfer_plans,
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


def test_raw_recurrent_page_round_trip_uses_group_layout() -> None:
    """Native D2H/H2D must use the recurrent page's width, not attention's."""
    source = {
        f"layer_{i}": (
            torch.arange(6 * 2 * 4 * 4 * 32, device="xpu")
            .reshape(6, 2, 4, 4, 32)
            .remainder(101)
            + i
        ).to(torch.bfloat16)
        for i in range(2)
    }
    group = EngineGroupInfo(
        engine_group_id=0,
        layer_indices=(0, 1),
        tokens_per_block=4,
        recurrent_state=True,
        cache_category="mamba",
    )
    plan = _build_group_transfer_plans(
        [group],
        source,
        2,
        4,
        MemoryLayoutDesc(shapes=[torch.Size([4, 8, 16])], dtypes=[torch.float32]),
        4,
        16,
        {"kv_layout": "NHD"},
        True,
        KVWeaveRuntimeConfig(
            enabled=False, linear_quant_enabled=False, linear_max_size_ratio=1.2
        ),
        KVWeaveCodec(),
    )[0]
    assert plan.chunk_shape == torch.Size([2, 2, 8, 128])
    output = torch.empty(plan.chunk_shape, dtype=torch.bfloat16)
    gathered = gather_paged_kv_to_cpu(
        source, [0, 1], 2, engine_kv_format=plan.engine_kv_format, out=[output]
    )
    torch_dev.synchronize()
    for i, tensor in enumerate(source.values()):
        expected = tensor[:2].permute(1, 0, 2, 3, 4).reshape(2, 8, 128).cpu()
        assert torch.equal(output[:, i], expected)
    destination = {key: torch.zeros_like(value) for key, value in source.items()}
    scatter_cpu_to_paged_kv(
        destination, [3, 4], gathered, 2, engine_kv_format=plan.engine_kv_format
    )
    torch_dev.synchronize()
    for key in source:
        assert torch.equal(destination[key][[3, 4]], source[key][[0, 1]])


if __name__ == "__main__":
    source = _make_source()
    for block_id in (0, 7, 13, 3):
        elapsed_ms, _ = _gather(source, block_id)
        print(f"block_id={block_id} transfer_ms={elapsed_ms:.3f}", flush=True)