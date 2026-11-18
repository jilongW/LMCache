# SPDX-License-Identifier: Apache-2.0
"""
Tests for the ``kvweave`` L2 serde.

The serde is driven only through its public surface: the factory-built
``SerdeProcessor`` (submit / query / size estimate) and, end to end, the
``StorageManager`` L1 -> L2 -> L1 path. Engine knowledge arrives the way the
engine registration delivers it, via the serde group-info registry.

The native kvweave extension is required; CPU is always exercised and XPU is
added when an XPU device is present.
"""

# Standard
from dataclasses import dataclass
import time

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.common.group_view import MambaSubStateWireLayout
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.serde import (
    SerdeConfig,
    create_serde_processor,
    get_registered_serde_types,
)
from lmcache.v1.distributed.serde.group_info import (
    SerdePartInfo,
    get_serde_group_info_registry,
)

kvweave = pytest.importorskip("kvweave")

_DEVICES = ["cpu"] + (
    ["xpu"] if hasattr(torch, "xpu") and torch.xpu.is_available() else []
)

_MODEL = "kvweave-test-model"
_BLOCK = 64
_HEADS, _HEAD_DIM = 4, 64
_HIDDEN = _HEADS * _HEAD_DIM


def _key(group: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=b"\x01" * 32, model_name=_MODEL, kv_rank=0, object_group_id=group
    )


@dataclass
class _Obj:
    """MemoryObj stand-in exposing the accessors the serde uses."""

    parts: list[torch.Tensor]
    used: int | None = None

    @property
    def tensor(self) -> torch.Tensor:
        return self.parts[0]

    def get_tensor(self, index: int) -> torch.Tensor:
        return self.parts[index]

    def get_shapes(self) -> list[torch.Size]:
        return [p.shape for p in self.parts]

    def get_dtypes(self) -> list[torch.dtype]:
        return [p.dtype for p in self.parts]

    def set_used_size(self, n: int) -> None:
        self.used = n


def _wait(processor, kind: str, task_id: int) -> bool:
    query = getattr(processor, f"query_{kind}_result")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        result = query(task_id)
        if result is not None:
            return result
        time.sleep(0.005)
    raise TimeoutError(kind)


def _round_trip(processor, src: _Obj, group: int) -> tuple[_Obj, _Obj]:
    """Serialize ``src`` then deserialize into a fresh object; return both."""
    key = _key(group)
    layout = MemoryLayoutDesc(shapes=src.get_shapes(), dtypes=src.get_dtypes())
    size = processor.estimate_serialized_size_for_key(layout, key)
    stored = _Obj([torch.zeros(size, dtype=torch.uint8)])
    assert _wait(
        processor, "serialize", processor.submit_serialize([src], [stored], [key])
    )
    # What the L2 adapter persists is the narrowed buffer.
    assert stored.used is not None
    loaded = _Obj([torch.zeros(size, dtype=torch.uint8)])
    loaded.parts[0].copy_(stored.parts[0])
    out = _Obj([torch.zeros_like(p) for p in src.parts])
    assert _wait(
        processor, "deserialize", processor.submit_deserialize([loaded], [out], [key])
    )
    return stored, out


def _mamba_page() -> tuple[
    tuple[MambaSubStateWireLayout, MambaSubStateWireLayout], int
]:
    """A Qwen3.5-like page: fp16 conv (3 x 8192), fp32 ssm (32 x 128 x 128), pad."""
    conv_shape, ssm_shape = (3, 8192), (32, 128, 128)
    conv_bytes, ssm_bytes = 3 * 8192 * 2, 32 * 128 * 128 * 4
    hidden = (conv_bytes + ssm_bytes + 4096) // (_BLOCK * 2)
    layout = (
        MambaSubStateWireLayout(
            0,
            conv_bytes,
            "torch.float16",
            conv_shape,
            channel_segments=(2048, 2048, 4096),
        ),
        MambaSubStateWireLayout(conv_bytes, ssm_bytes, "torch.float32", ssm_shape),
    )
    return layout, hidden


@pytest.fixture
def registry():
    reg = get_serde_group_info_registry()
    reg.clear()
    yield reg
    reg.clear()


def _processor(device: str, **extra):
    kwargs = dict(
        qbit=4,
        device=device,
        num_threads=2,
        num_kv_heads=_HEADS,
        head_dim=_HEAD_DIM,
        conv_scaling_method="per_channel",
        ssm_scaling_method="per_channel",
    )
    kwargs.update(extra)
    return create_serde_processor(SerdeConfig(type="kvweave", kwargs=kwargs))


def test_kvweave_is_registered() -> None:
    assert "kvweave" in get_registered_serde_types()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("qbit,max_ratio", [(4, 0.30), (8, 0.55)])
def test_split_attention_round_trip(registry, device, qbit, max_ratio) -> None:
    """[2, L, T, H] chunks shrink, stay close, and the estimate is exact."""
    registry.register(
        _MODEL,
        0,
        [SerdePartInfo("attention", _BLOCK, _HEADS, _HEAD_DIM, False, False)],
    )
    processor = _processor(device, qbit=qbit)
    try:
        torch.manual_seed(0)
        x = torch.randn(2, 4, 128, _HIDDEN).half()
        stored, out = _round_trip(processor, _Obj([x]), 0)
        assert stored.used == stored.parts[0].numel()  # estimate == actual size
        assert stored.used / (x.numel() * 2) < max_ratio
        assert (out.tensor.float() - x.float()).abs().mean() < 0.15
    finally:
        processor.close()


@pytest.mark.parametrize("device", _DEVICES)
def test_fused_attention_round_trip(registry, device) -> None:
    """[L, T, H] chunks of a fused-packed format use the fused codec."""
    registry.register(
        _MODEL,
        1,
        [SerdePartInfo("attention", _BLOCK, _HEADS, _HEAD_DIM, False, True)],
    )
    processor = _processor(device)
    try:
        torch.manual_seed(1)
        x = torch.randn(4, 128, 2 * _HIDDEN).half()
        stored, out = _round_trip(processor, _Obj([x]), 1)
        assert stored.used == stored.parts[0].numel()
        assert stored.used < x.numel() * 2 * 0.30
        assert (out.tensor.float() - x.float()).abs().mean() < 0.15
    finally:
        processor.close()


@pytest.mark.parametrize(
    "info",
    [
        SerdePartInfo("attention", _BLOCK, 1, 128, True, False),  # MLA latent plane
        SerdePartInfo("unknown", _BLOCK),  # unclassified group
    ],
)
def test_unquantizable_groups_pass_through_bit_exact(registry, info) -> None:
    """Groups the codec must not touch are stored and restored byte for byte."""
    registry.register(_MODEL, 2, [info])
    processor = _processor("cpu")
    try:
        x = torch.randn(4, 128, 128).half()
        stored, out = _round_trip(processor, _Obj([x]), 2)
        assert stored.used == x.numel() * 2
        assert torch.equal(out.tensor.view(torch.uint8), x.view(torch.uint8))
    finally:
        processor.close()


def test_unregistered_group_without_defaults_passes_through(registry) -> None:
    """With no facts and no head geometry, a chunk is never mis-quantized."""
    processor = create_serde_processor(
        SerdeConfig(type="kvweave", kwargs={"device": "cpu"})
    )
    try:
        x = torch.randn(4, 128, 128).half()  # rank 3: not split-K/V
        _, out = _round_trip(processor, _Obj([x]), 7)
        assert torch.equal(out.tensor.view(torch.uint8), x.view(torch.uint8))
    finally:
        processor.close()


@pytest.mark.parametrize("device", _DEVICES)
def test_mamba_without_engine_conv_segments_is_stored_raw(registry, device) -> None:
    """The conv q/k/v split comes only from the engine; a group whose engine did
    not report it (e.g. a non-GDN Mamba kind) is stored unquantized."""
    layout, hidden = _mamba_page()
    conv, ssm = layout
    registry.register(
        _MODEL,
        3,
        [
            SerdePartInfo(
                "mamba",
                _BLOCK,
                mamba_real_layout=(
                    msgspec.structs.replace(conv, channel_segments=()),
                    ssm,
                ),
            )
        ],
    )
    processor = _processor(device)
    try:
        desc = MemoryLayoutDesc(
            shapes=[torch.Size([2, _BLOCK * 2, hidden])], dtypes=[torch.float16]
        )
        raw_bytes = 2 * _BLOCK * 2 * hidden * 2
        assert processor.estimate_serialized_size_for_key(desc, _key(3)) == raw_bytes
    finally:
        processor.close()


@pytest.mark.parametrize("device", _DEVICES)
def test_mamba_round_trip_keeps_each_substate_in_its_own_dtype(
    registry, device
) -> None:
    """conv (fp16) and ssm (fp32) are quantized separately and restored in place."""
    # First Party
    from lmcache.v1.distributed.serde.kvweave import KVWeaveCodec, MambaChunkSplit

    layout, hidden = _mamba_page()
    registry.register(
        _MODEL, 3, [SerdePartInfo("mamba", _BLOCK, mamba_real_layout=layout)]
    )
    processor = _processor(device)
    try:
        torch.manual_seed(2)
        layers, blocks = 2, 2
        split = MambaChunkSplit(
            torch.randn(layers, blocks, 3, 8192).half(),
            torch.randn(layers, blocks, 32, 128, 128),
        )
        raw = KVWeaveCodec.merge_mamba_chunk(
            split,
            layout,
            _BLOCK,
            hidden,
            raw_shape=torch.Size([layers, _BLOCK * blocks, hidden]),
            raw_dtype=torch.float16,
        )
        stored, out = _round_trip(processor, _Obj([raw]), 3)
        assert stored.used == stored.parts[0].numel()
        assert stored.used < raw.numel() * 2 * 0.25
        restored = KVWeaveCodec.split_mamba_chunk(out.tensor, layout, _BLOCK)
        assert restored.conv.dtype == torch.float16
        assert restored.ssm.dtype == torch.float32
        assert (restored.conv.float() - split.conv.float()).abs().max() < 1.0
        assert (restored.ssm - split.ssm).abs().max() < 1.0
    finally:
        processor.close()


@pytest.mark.parametrize("device", _DEVICES)
def test_multi_part_object_round_trips_every_part(registry, device) -> None:
    """An object holding one tensor per kernel group is framed and restored per part."""
    layout, hidden = _mamba_page()
    registry.register(
        _MODEL,
        4,
        [
            SerdePartInfo("attention", _BLOCK, _HEADS, _HEAD_DIM, False, False),
            SerdePartInfo("attention", _BLOCK, 1, 128, True, False),
            SerdePartInfo("mamba", _BLOCK, mamba_real_layout=layout),
        ],
    )
    processor = _processor(device)
    try:
        torch.manual_seed(3)
        parts = [
            torch.randn(2, 2, 128, _HIDDEN).half(),
            torch.randn(2, 128, 128).half(),
            torch.randn(1, 2 * _BLOCK, hidden).half(),
        ]
        stored, out = _round_trip(processor, _Obj(parts), 4)
        assert stored.used == stored.parts[0].numel()
        assert (out.parts[0].float() - parts[0].float()).abs().mean() < 0.15
        assert torch.equal(out.parts[1].view(torch.uint8), parts[1].view(torch.uint8))
        assert out.parts[2].shape == parts[2].shape
    finally:
        processor.close()


def test_estimate_depends_on_the_objects_group(registry) -> None:
    """Same shape, different group: the buffer is sized for that key's group."""
    registry.register(
        _MODEL, 0, [SerdePartInfo("attention", _BLOCK, _HEADS, _HEAD_DIM, False, False)]
    )
    registry.register(_MODEL, 1, [SerdePartInfo("unknown", _BLOCK)])
    processor = _processor("cpu")
    try:
        layout = MemoryLayoutDesc(
            shapes=[torch.Size([2, 4, 128, _HIDDEN])], dtypes=[torch.float16]
        )
        quantized = processor.estimate_serialized_size_for_key(layout, _key(0))
        raw = processor.estimate_serialized_size_for_key(layout, _key(1))
        assert raw == 2 * 4 * 128 * _HIDDEN * 2
        assert quantized < raw
    finally:
        processor.close()


def test_corrupt_payload_fails_instead_of_returning_garbage(registry) -> None:
    """A buffer that is not a kvweave payload makes the deserialize task fail."""
    registry.register(
        _MODEL, 0, [SerdePartInfo("attention", _BLOCK, _HEADS, _HEAD_DIM, False, False)]
    )
    processor = _processor("cpu")
    try:
        x = torch.randn(2, 4, 128, _HIDDEN).half()
        layout = MemoryLayoutDesc(shapes=[x.shape], dtypes=[x.dtype])
        size = processor.estimate_serialized_size_for_key(layout, _key(0))
        junk = _Obj([torch.full((size,), 0x5A, dtype=torch.uint8)])
        out = _Obj([torch.zeros_like(x)])
        task = processor.submit_deserialize([junk], [out], [_key(0)])
        assert _wait(processor, "deserialize", task) is False
    finally:
        processor.close()


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="qbit"):
        create_serde_processor(SerdeConfig(type="kvweave", kwargs={"qbit": 3}))
    with pytest.raises(ValueError, match="scaling_method"):
        create_serde_processor(
            SerdeConfig(type="kvweave", kwargs={"scaling_method": "per_row"})
        )
