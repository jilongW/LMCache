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
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    AttentionPlaneLayout,
    KVWeaveRuntimeConfig,
)
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


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.float16])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_raw_page_transfer_preserves_byte_dtype_and_physical_stride(
    dtype: torch.dtype, padded: bool, fused: bool
) -> None:
    """Opaque byte pages and shared-pool strides must round-trip exactly."""
    num_blocks, block_size, width = 8, 4, 32
    page_shape = (block_size, 1, width * 2) if fused else (2, block_size, 1, width)
    page_elements = torch.Size(page_shape).numel()
    stride = page_elements * (3 if padded else 1)
    source = {}
    destination = {}
    backing_tensors = []
    for i in range(2):
        # Extra backing keeps the old byte-as-fp16 bug inside these test
        # allocations, so the regression reports corruption, not device loss.
        backing = torch.full((num_blocks * stride * 2,), 101, dtype=dtype, device="xpu")
        view = backing[: num_blocks * stride].view(num_blocks, stride)
        view = view[:, :page_elements].view(num_blocks, *page_shape)
        for block in range(num_blocks):
            view[block].fill_(block * 7 + i)
        source[f"layer_{i}"] = view
        target_backing = torch.full_like(backing, 103)
        target = target_backing[: num_blocks * stride].view(num_blocks, stride)
        destination[f"layer_{i}"] = target[:, :page_elements].view(
            num_blocks, *page_shape
        )
        backing_tensors.append(target_backing)
    fmt = (
        EngineKVFormat.NL_X_NB_BS_NH_CS
        if fused
        else EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
    )
    shape = (2, 8, width * 2) if fused else (2, 2, 8, width)
    numel = torch.Size(shape).numel()
    output_backing = torch.full((numel * 2,), 107, dtype=dtype)
    output = output_backing[:numel].view(shape)
    chunks = gather_paged_kv_to_cpu(
        source, [1, 2], 2, engine_kv_format=fmt, out=[output]
    )
    torch_dev.synchronize()
    assert torch.all(output_backing[numel:] == 107), "Gather overwrote chunk boundary"
    for i, tensor in enumerate(source.values()):
        selected = tensor[[1, 2]].cpu()
        wanted = (
            selected.reshape(8, width * 2)
            if fused
            else selected.permute(1, 0, 2, 3, 4).reshape(2, 8, width)
        )
        assert torch.equal(output[i] if fused else output[:, i], wanted)
    scatter_cpu_to_paged_kv(destination, [3, 4], chunks, 2, engine_kv_format=fmt)
    torch_dev.synchronize()
    for i, key in enumerate(source):
        assert torch.equal(destination[key][[3, 4]], source[key][[1, 2]])
        backing = backing_tensors[i].view(num_blocks * 2, stride)
        assert torch.all(backing[[0, 1, 2, 5, 6, 7]] == 103)
        if padded:
            assert torch.all(backing[:, page_elements:] == 103)


def test_vllm_unified_mamba_byte_views_round_trip() -> None:
    """Use vLLM's own byte/stride views, not a float approximation of them."""
    interface = pytest.importorskip("vllm.v1.kv_cache_interface")
    layouts = pytest.importorskip("vllm.v1.kv_cache_layout")
    if not hasattr(interface, "create_kv_cache_views"):
        pytest.skip("Requires the unified vLLM cache allocator")
    # First Party
    from lmcache.integration.vllm.kv_cache_group_edits import apply_kv_cache_group_edits

    spec = interface.MambaSpec(
        block_size=4,
        shapes=((4, 8), (4, 8)),
        dtypes=(torch.float16, torch.float32),
        page_size_padded=256,
        mamba_cache_mode="align",
    )
    placement = interface.KVCacheTensor(
        size=4096,
        layers=["layer0", "layer1"],
        layer_stride=256,
        block_stride=512,
    )
    config = interface.KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[placement],
        kv_cache_groups=[
            interface.KVCacheGroupSpec(
                layer_names=placement.layers, kv_cache_spec=spec
            )
        ],
    )

    def make_views(backing: torch.Tensor) -> dict[str, torch.Tensor]:
        views = interface.create_kv_cache_views(
            backing, spec, 8, layouts.KVCacheLayout.BLNHC, placement
        )
        return apply_kv_cache_group_edits(
            config, dict(zip(placement.layers, views, strict=True)), {"kv_layout": "NHD"}
        )

    source_backing = (
        torch.arange(4096, device="xpu").remainder(127).to(torch.int8)
    )
    destination_backing = torch.full_like(source_backing, -1)
    source = make_views(source_backing)
    destination = make_views(destination_backing)
    output = torch.empty(2, 8, 48, dtype=torch.int8)
    fmt = EngineKVFormat.NL_X_NB_BS_NH_CS
    chunks = gather_paged_kv_to_cpu(
        source, [1, 2], 2, engine_kv_format=fmt, out=[output]
    )
    torch_dev.synchronize()
    for i, tensor in enumerate(source.values()):
        assert tensor.dtype == torch.int8
        assert tensor.stride(0) == 512
        assert torch.equal(output[i], tensor[[1, 2]].reshape(8, 48).cpu())
    scatter_cpu_to_paged_kv(destination, [3, 4], chunks, 2, engine_kv_format=fmt)
    torch_dev.synchronize()
    expected_backing = torch.full_like(destination_backing, -1)
    expected_views = make_views(expected_backing)
    for key, tensor in source.items():
        expected_views[key][[3, 4]] = tensor[[1, 2]]
    assert torch.equal(destination_backing, expected_backing)


@pytest.mark.parametrize("layout_name", ["LBNHC", "LBHNC", "BLNHC", "BLHNC"])
def test_vllm_full_attention_allocator_quantized_round_trip(
    layout_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check real attention allocator views separately from Mamba transfers."""
    interface = pytest.importorskip("vllm.v1.kv_cache_interface")
    layouts = pytest.importorskip("vllm.v1.kv_cache_layout")
    if not hasattr(interface, "create_kv_cache_views"):
        pytest.skip("Requires the unified vLLM cache allocator")
    # First Party
    from lmcache.integration.vllm.utils import translate_vllm_kv_cache_layout
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import compute_kv_layout

    for name, value in {
        "SCALING_METHOD": "per_token", "RH": "1", "ASYM": "1", "PRECOND": "1"
    }.items():
        monkeypatch.setenv(f"LMCACHE_MP_KVWEAVE_{name}", value)
    spec = interface.FullAttentionSpec(
        block_size=16, num_kv_heads=4, head_size=256, dtype=torch.float16
    )
    layout = layouts.KVCacheLayout[layout_name]
    page = spec.page_size_bytes
    placement = interface.KVCacheTensor(
        size=page * 8 * 2,
        layers=["layer0", "layer1"],
        layer_stride=page * 8 if layout.is_layer_compact else page,
        block_stride=page if layout.is_layer_compact else page * 2,
    )
    hints = {"kv_layout": translate_vllm_kv_cache_layout(layout_name)}

    def make_views(backing: torch.Tensor) -> dict[str, torch.Tensor]:
        views = interface.create_kv_cache_views(backing, spec, 8, layout, placement)
        return dict(zip(placement.layers, views, strict=True))

    raw_source = torch.zeros(placement.size, dtype=torch.int8, device="xpu")
    source = make_views(raw_source)
    for tensor in source.values():
        tensor.copy_(torch.randn_like(tensor))
    torch_dev.synchronize()
    block_size, layers, hidden, _, fmt, _ = compute_kv_layout(source, hints)
    shape = torch.Size((layers, block_size * 2, hidden))
    output = torch.empty(shape, dtype=torch.float16)
    chunks = gather_paged_kv_to_cpu(
        source, [1, 2], 2, layout_hints=hints, engine_kv_format=fmt, out=[output]
    )
    torch_dev.synchronize()
    expected = torch.stack(
        [
            tensor[[1, 2]].permute(0, 2, 1, 3).reshape(block_size * 2, hidden).cpu()
            for tensor in source.values()
        ]
    )
    assert torch.equal(output, expected), "Attention gather differs before encoding"
    codec = KVWeaveCodec({"num_kv_heads": 4, "head_dim": 256})
    payload = codec.encode_chunk(
        "attention", None, 16, None, chunks[0], AttentionPlaneLayout.FUSED_KV
    )
    wire = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    decoded = codec.decode_chunk(
        "attention", None, 16, shape, torch.float16, wire,
        AttentionPlaneLayout.FUSED_KV,
    )
    relative_error = (decoded.float() - expected.float()).norm() / expected.float().norm()
    assert relative_error < 0.2
    raw_destination = torch.zeros_like(raw_source)
    destination = make_views(raw_destination)
    scatter_cpu_to_paged_kv(
        destination, [3, 4], [decoded], 2, layout_hints=hints, engine_kv_format=fmt
    )
    torch_dev.synchronize()
    expected_backing = torch.zeros_like(raw_destination)
    expected_views = make_views(expected_backing)
    for i, tensor in enumerate(expected_views.values()):
        tensor[[3, 4]] = decoded[i].reshape(2, 16, 4, 512).permute(0, 2, 1, 3).to("xpu")
    assert torch.equal(raw_destination, expected_backing), "Attention scatter changed data"


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