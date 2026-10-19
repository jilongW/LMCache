# SPDX-License-Identifier: Apache-2.0
"""Compare Mamba direct builds on the production two-plane page geometry.

Run the same command before and after rebuilding KVWeave. Timings include
native payload serialization / scale parsing, not just the arithmetic kernel.
The end-to-end measurements include both conv and SSM codec work.

    OMP_NUM_THREADS=8 python tests/benchmarks/bench_mamba_ssm_rh_direct.py
    OMP_NUM_THREADS=8 python tests/benchmarks/bench_mamba_ssm_rh_direct.py --matrix
"""

import argparse
import json
import statistics
import time

import torch

from kvweave import kvweave_quant
from lmcache.v1.distributed.serde.kvweave.kvweave_config import (
    ConvQKVSplit,
    MambaCodecOptions,
)
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import (
    KVWeaveCodec,
    MambaChunkSplit,
)
from lmcache.v1.multiprocess.group_view import MambaSubStateWireLayout


def error_metrics(original, restored):
    original = original.double().reshape(-1)
    restored = restored.double().reshape(-1)
    delta = restored - original
    return {
        "rmse": delta.square().mean().sqrt().item(),
        "relative_l2": (delta.norm() / original.norm()).item(),
        "max_abs": delta.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            original, restored, dim=0
        ).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--qbits", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Measure all valid production conv/SSM scaling/RH pairs",
    )
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 0 or args.threads < 1:
        parser.error("iters/threads must be positive and warmup nonnegative")
    torch.set_num_threads(args.threads)

    block_size, layers, blocks, hidden = 1024, 8, 1, 1024
    conv_shape, ssm_shape = (3, 8192), (32, 128, 128)
    conv_bytes = torch.Size(conv_shape).numel() * 2
    ssm_bytes = torch.Size(ssm_shape).numel() * 4
    layouts = (
        MambaSubStateWireLayout(0, conv_bytes, "torch.float16", conv_shape),
        MambaSubStateWireLayout(conv_bytes, ssm_bytes, "torch.float32", ssm_shape),
    )
    gen = torch.Generator().manual_seed(123)
    split = MambaChunkSplit(
        torch.randn(layers, blocks, *conv_shape, generator=gen).half(),
        torch.randn(layers, blocks, *ssm_shape, generator=gen),
    )
    raw = KVWeaveCodec.merge_mamba_chunk(
        split,
        layouts,
        block_size,
        hidden,
        raw_shape=torch.Size((2, layers, block_size, hidden)),
        raw_dtype=torch.float16,
    )
    codec = KVWeaveCodec()
    target = torch.empty_like(raw)
    page_bytes = 2 * block_size * hidden * raw.element_size()

    configurations = (
        [
            (conv_method, conv_rh, ssm_method, ssm_rh)
            for conv_method, conv_rh in (
                ("per_token", True),
                ("per_token", False),
                ("per_channel", False),
            )
            for ssm_method in ("per_token", "per_channel")
            for ssm_rh in (False, True)
        ]
        if args.matrix
        else [("per_token", True, "per_token", True)]
    )
    for qbit, configuration in (
        (qbit, configuration) for qbit in args.qbits for configuration in configurations
    ):
        conv_method, conv_rh, ssm_method, ssm_rh = configuration
        options = MambaCodecOptions(
            conv_scaling_method=conv_method,
            conv_rh=conv_rh,
            ssm_scaling_method=ssm_method,
            ssm_rh=ssm_rh,
            asym=True,
            conv_qbit=qbit,
            ssm_qbit=qbit,
            conv_qkv_split=ConvQKVSplit(key_dim=2048, value_dim=4096),
        )
        transform_signs, transform_perm = (
            codec._mamba_precond_pair(128 if ssm_method == "per_token" else 4096)
            if ssm_rh
            else (None, None)
        )
        kwargs = dict(
            qbit=qbit,
            blocks_num=4096,
            block_size=1,
            head_num=1,
            head_dim=128,
            asym=True,
            scaling_method=ssm_method,
            substate_dtype=torch.float32,
            rh=ssm_rh,
            signs=transform_signs,
            perm=transform_perm,
            num_threads=args.threads,
        )

        def native_encode(kwargs=kwargs):
            return kvweave_quant.kvweave_serialize_mamba_page_substate_direct(
                raw,
                page_bytes,
                conv_bytes,
                128,
                4096,
                layers,
                blocks,
                True,
                b"",
                123,
                **kwargs,
            )

        # Older builds cannot directly handle no-RH per-token two-plane pages.
        native_supported = not (
            ssm_method == "per_token"
            and not ssm_rh
            and not getattr(kvweave_quant, "MAMBA_FUSED_SCALING", False)
        )
        if native_supported:
            payload = bytes(native_encode())
            scale_len = int.from_bytes(payload[:4], "big")
            scales = payload[4 : 4 + scale_len]
            q_data = torch.frombuffer(
                bytearray(payload[4 + scale_len :]), dtype=torch.int8
            )

            def native_decode(q_data=q_data, scales=scales, kwargs=kwargs):
                kvweave_quant.kvweave_dequantize_mamba_page_substate_direct(
                    q_data,
                    scales,
                    target,
                    page_bytes,
                    conv_bytes,
                    128,
                    4096,
                    layers,
                    blocks,
                    True,
                    **kwargs,
                )

        chunk = torch.frombuffer(
            bytearray(codec.encode_chunk("mamba", layouts, block_size, options, raw)),
            dtype=torch.uint8,
        )
        operations = {
            "codec_encode_ms": lambda options=options: codec.encode_chunk(
                "mamba", layouts, block_size, options, raw
            ),
            "codec_decode_ms": lambda chunk=chunk: codec.decode_chunk(
                "mamba", layouts, block_size, raw.shape, raw.dtype, chunk, out=target
            ),
        }
        if native_supported:
            operations = {
                "ssm_encode_ms": native_encode,
                "ssm_decode_ms": native_decode,
                **operations,
            }
        samples = {name: [] for name in operations}
        for _ in range(args.warmup):
            for fn in operations.values():
                fn()
        for iteration in range(args.iters):
            names = list(operations)
            if iteration % 2:
                names.reverse()
            for name in names:
                start = time.perf_counter()
                operations[name]()
                samples[name].append((time.perf_counter() - start) * 1000)
        codec.decode_chunk(
            "mamba", layouts, block_size, raw.shape, raw.dtype, chunk, out=target
        )
        restored = codec.split_mamba_chunk(target, layouts, block_size)
        print(
            json.dumps(
                {
                    "shape": list(raw.shape),
                    "threads": args.threads,
                    "qbit": qbit,
                    "iters": args.iters,
                    "build": kvweave_quant.build_info(),
                    "conv_scaling": conv_method,
                    "conv_rh": conv_rh,
                    "ssm_scaling": ssm_method,
                    "ssm_rh": ssm_rh,
                    "conv_error": error_metrics(split.conv, restored.conv),
                    "ssm_error": error_metrics(split.ssm, restored.ssm),
                    **{
                        name: round(statistics.median(values), 4)
                        for name, values in samples.items()
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
