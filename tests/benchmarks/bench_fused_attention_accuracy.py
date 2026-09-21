# SPDX-License-Identifier: Apache-2.0
"""Full-attention and Mamba KVWeave round-trip accuracy matrices.

Manual benchmark; run directly rather than through pytest:

    python tests/benchmarks/bench_fused_attention_accuracy.py

Measures the production fused-attention codec on Qwen3.5-9B's logical
``[layers, tokens, 2 * num_kv_heads * head_dim]`` shape. The matrix spans
qbit, scale granularity, RH, asymmetric quantization, and P/D
preconditioning. For per-channel/per-token, the codec uses independent
scales for every packed K/V head.

Also measures Qwen3.5-9B-shaped Mamba ``conv_state`` Q/K/V substates and
``ssm_state`` across their native quantization configurations.
"""

# Standard
import itertools

# Third Party
import torch

# First Party
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    AttentionPlaneLayout,
    ConvQKVSplit,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import _KVWeaveCodec


_SEED = 20260921
_NUM_LAYERS = 8
_TOKENS = 1024
_NUM_KV_HEADS = 4
_HEAD_DIM = 256
_BLOCK_SIZE = 1024
_MAMBA_CONV_SPLIT = ConvQKVSplit(key_dim=2048, value_dim=4096)


def _codec(
    qbit: int,
    scaling_method: str,
    rh: bool,
    asym: bool,
    precond: bool,
) -> _KVWeaveCodec:
    return _KVWeaveCodec(
        {
            "quantize": True,
            "qbit": qbit,
            "scaling_method": scaling_method,
            "rh": rh,
            "asym": asym,
            "precond": precond,
            "block_size": _BLOCK_SIZE,
            "num_kv_heads": _NUM_KV_HEADS,
            "head_dim": _HEAD_DIM,
        }
    )


def _print_mamba_metrics(
    label: str,
    payload_bytes: int,
    source: torch.Tensor,
    restored: torch.Tensor,
) -> None:
    error = restored.float() - source.float()
    rmse = error.square().mean().sqrt().item()
    source_rms = source.float().square().mean().sqrt().item()
    print(
        f"{label},{payload_bytes / (1024**2):.3f},"
        f"{error.abs().mean().item():.7f},{rmse:.7f},"
        f"{error.abs().max().item():.7f},{100 * rmse / source_rms:.5f}"
    )


def _run_mamba_conv_matrix(source: torch.Tensor) -> None:
    """Measure production conv Q/K/V split quantization combinations."""
    print("\n# mamba_conv_state")
    print("qbit,scaling,rh,asym,payload_mib,mae,rmse,max_abs,relative_rmse_pct")
    query, key, value = _KVWeaveCodec._split_conv_qkv(
        source, _MAMBA_CONV_SPLIT
    )
    for qbit, scaling_method, rh, asym in itertools.product(
        (4, 8), ("per_tensor", "per_channel", "per_token"), (False, True), (False, True)
    ):
        try:
            payloads = [
                _KVWeaveCodec.quantize_mamba_substate_4bit(
                    tensor,
                    substate="conv",
                    scaling_method=scaling_method,
                    rh=rh,
                    asym=asym,
                    qbit=qbit,
                )
                for tensor in (query, key, value)
            ]
            restored = torch.cat(
                [
                    _KVWeaveCodec.dequantize_mamba_substate_4bit(payload)
                    for payload in payloads
                ],
                dim=-1,
            )
        except ValueError as error:
            print(f"{qbit},{scaling_method},{int(rh)},{int(asym)},skipped,{error}")
            continue
        _print_mamba_metrics(
            f"{qbit},{scaling_method},{int(rh)},{int(asym)}",
            len(_KVWeaveCodec.pack_conv_qkv_payloads(*payloads)),
            source,
            restored,
        )


def _run_mamba_ssm_matrix(source: torch.Tensor) -> None:
    """Measure production SSM substate quantization combinations."""
    print("\n# mamba_ssm_state")
    print("qbit,scaling,rh,asym,payload_mib,mae,rmse,max_abs,relative_rmse_pct")
    for qbit, scaling_method, rh, asym in itertools.product(
        (4, 8), ("per_tensor", "per_channel", "per_token"), (False, True), (False, True)
    ):
        try:
            payload = _KVWeaveCodec.quantize_mamba_substate_4bit(
                source,
                substate="ssm",
                scaling_method=scaling_method,
                rh=rh,
                asym=asym,
                qbit=qbit,
            )
            restored = _KVWeaveCodec.dequantize_mamba_substate_4bit(payload)
        except ValueError as error:
            print(f"{qbit},{scaling_method},{int(rh)},{int(asym)},skipped,{error}")
            continue
        _print_mamba_metrics(
            f"{qbit},{scaling_method},{int(rh)},{int(asym)}",
            len(payload),
            source,
            restored,
        )


def main() -> int:
    torch.manual_seed(_SEED)
    source = torch.randn(
        _NUM_LAYERS,
        _TOKENS,
        2 * _NUM_KV_HEADS * _HEAD_DIM,
        dtype=torch.float16,
    )
    source_rms = source.float().square().mean().sqrt().item()

    print(
        "qbit,scaling,rh,asym,precond,payload_mib,mae,rmse,"
        "max_abs,relative_rmse_pct"
    )
    for qbit, scaling_method, rh, asym, precond in itertools.product(
        (4, 8),
        ("per_tensor", "per_channel", "per_token"),
        (False, True),
        (False, True),
        (False, True),
    ):
        codec = _codec(qbit, scaling_method, rh, asym, precond)
        payload = codec.encode_chunk(
            "attention",
            None,
            _BLOCK_SIZE,
            None,
            source,
            AttentionPlaneLayout.FUSED_KV,
        )
        restored = codec.decode_chunk(
            "attention",
            None,
            _BLOCK_SIZE,
            source.shape,
            source.dtype,
            torch.frombuffer(bytearray(payload), dtype=torch.uint8),
            AttentionPlaneLayout.FUSED_KV,
        )
        error = restored.float() - source.float()
        rmse = error.square().mean().sqrt().item()
        print(
            f"{qbit},{scaling_method},{int(rh)},{int(asym)},{int(precond)},"
            f"{len(payload) / (1024**2):.3f},{error.abs().mean().item():.7f},"
            f"{rmse:.7f},{error.abs().max().item():.7f},"
            f"{100 * rmse / source_rms:.5f}"
        )
    _run_mamba_conv_matrix(
        torch.randn(1, 1, 3, 8192, dtype=torch.float16)
    )
    _run_mamba_ssm_matrix(
        torch.randn(1, 1, 32, 128, 128, dtype=torch.float32)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())