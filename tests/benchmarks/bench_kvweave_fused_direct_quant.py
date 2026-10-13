# SPDX-License-Identifier: Apache-2.0
"""Fused-K/V attention: split + per-layer quantize vs direct kernel, and the
matching decode (dequantize + head merge vs direct dequantize).

Manual script (run with ``python``, not pytest). For each config and device it
times the existing encode path (``kvweave_split_fused_heads`` +
``kvweave_serialize_chunk_state``) against the direct kernel
(``kvweave_serialize_fused_chunk_direct``), and the existing decode path
(``deserialize_fused_tensor`` with the direct dequant disabled: split-plane
dequantize + head merge) against ``kvweave_dequantize_fused_chunk_direct``. It
checks that both decodes are identical and reconstruct the original input within
a per-qbit error bound. On XPU the old encode is CPU head split + XPU
``serialize_chunk_state``; the old decode dequantizes on device, merges heads on
device, then D2H. per_tensor has no head split and no direct kernels, so only
its existing path is timed/checked.

    python tests/benchmarks/bench_kvweave_fused_direct_quant.py
    python tests/benchmarks/bench_kvweave_fused_direct_quant.py --devices cpu,xpu \
        --dtypes float16,bfloat16 --methods per_channel,per_token --rh 1,0 \
        --asym 1,0 --precond 1,0 --qbits 4,8 --threads 1,4,8
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
import time

import torch

from kvweave import kvweave_quant
from lmcache.v1.distributed.serde.kvweave.kvweave_config import KVWeaveCodecConfig
from lmcache.v1.distributed.serde.kvweave.kvweave_serde import KVWeaveCodec

try:
    from kvweave import kvweave_quant_xpu
except ImportError:
    kvweave_quant_xpu = None


def _make_chunk(layers: int, tokens: int, hidden: int, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1234)
    x = torch.randn(layers, tokens, hidden, generator=gen)
    channel_scale = torch.rand(1, 1, hidden, generator=gen) * 4 + 0.05
    x = x * channel_scale
    # Outlier channels, and an all-zero padded tail, as seen in real KV chunks.
    x[:, :, ::97] *= 20
    x[:, tokens - tokens // 8 :, :] = 0
    return x.to(dtype).contiguous()


# Loose regression bounds on relative MSE vs the original input; the synthetic
# chunk has x20 outlier channels, so no-RH per_token / per_tensor 4-bit sit ~0.1-0.2.
_MAX_REL_MSE = {4: 0.25, 8: 0.02}


def _accuracy(raw: torch.Tensor, decoded: torch.Tensor, qbit: int) -> dict:
    ref = raw.float()
    out = decoded.float()
    err = out - ref
    signal = ref.pow(2).mean().item()
    rel_mse = err.pow(2).mean().item() / signal
    cos = torch.nn.functional.cosine_similarity(
        out.double().flatten(), ref.double().flatten(), dim=0
    ).item()
    finite = bool(torch.isfinite(out).all())
    return dict(
        rel_mse=rel_mse,
        snr_db=-10 * math.log10(rel_mse) if rel_mse > 0 else float("inf"),
        cos=cos,
        rel_max_err=err.abs().max().item() / ref.abs().max().item(),
        acc_ok=finite and rel_mse < _MAX_REL_MSE.get(qbit, 0.25),
    )


def _acc_str(acc: dict) -> str:
    return (
        f"rel_mse={acc['rel_mse']:.4e} snr={acc['snr_db']:5.1f}dB cos={acc['cos']:.5f} "
        f"rel_max_err={acc['rel_max_err']:.4f} acc={'OK' if acc['acc_ok'] else 'BAD'}"
    )


def _decode(codec, payload: bytes, like: torch.Tensor) -> torch.Tensor:
    decoded = torch.empty_like(like)
    codec.deserialize_fused_tensor(torch.frombuffer(bytearray(payload), dtype=torch.uint8), decoded)
    return decoded


def _sizes(raw: torch.Tensor, old_bytes: int, new_bytes: int | None) -> dict:
    raw_bytes = raw.numel() * raw.element_size()
    return dict(
        raw_bytes=raw_bytes,
        old_bytes=old_bytes,
        new_bytes=new_bytes,
        old_ratio=raw_bytes / old_bytes,
        new_ratio=None if new_bytes is None else raw_bytes / new_bytes,
    )


def _size_str(s: dict) -> str:
    new = "n/a" if s["new_bytes"] is None else (
        f"{s['new_bytes'] / 2**20:.3f}MB ({s['new_ratio']:.2f}x)"
    )
    return (
        f"raw={s['raw_bytes'] / 2**20:.3f}MB old={s['old_bytes'] / 2**20:.3f}MB "
        f"({s['old_ratio']:.2f}x) new={new}"
    )


def _median_ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def _xpu_available() -> bool:
    return (
        kvweave_quant_xpu is not None
        and hasattr(kvweave_quant_xpu, "kvweave_serialize_fused_chunk_direct")
        and hasattr(torch, "xpu")
        and torch.xpu.is_available()
    )


def _run_per_tensor(
    args, device, dtype_name, codec, native, raw, header, qbit, threads,
) -> dict:
    """per_tensor has no head split and no direct kernel: time the codec's only path."""
    layers, tokens, hidden = raw.shape
    head_num = codec._fused_head_num(hidden)
    head_dim = codec._fused_head_dim(hidden)

    def quant():
        return native.kvweave_serialize_chunk_state(
            raw.view(-1), header, KVWeaveCodecConfig.next_scale_id(),
            qbit=qbit, blocks_num=max(1, tokens // codec.block_size),
            block_size=codec.block_size, head_num=head_num, head_dim=head_dim,
            num_layers=layers, rh=False, asym=False, scaling_method="per_tensor",
            signs=None, perm=None, num_threads=threads,
        )

    acc = _accuracy(raw, _decode(codec, bytes(quant()), raw), qbit)
    sizes = _sizes(raw, len(bytes(quant())), None)
    t_quant = _median_ms(quant, args.iters, args.warmup)
    payload = torch.frombuffer(bytearray(bytes(quant())), dtype=torch.uint8)
    dst = torch.empty_like(raw)
    t_dq_old = _median_ms(lambda: codec.deserialize_fused_tensor(payload, dst), args.iters, args.warmup)
    print(
        f"{device:<3} {dtype_name:<8} {'per_tensor':<11} rh=0 asym=0 precond=0 "
        f"qbit={qbit} thr={threads:<2} | old split=  0.000 quant={t_quant:8.3f} "
        f"total={t_quant:8.3f} ms | new=    n/a (no head split, direct kernel N/A) "
        f"| decode old={t_dq_old:7.3f} ms new=n/a "
        f"| {_size_str(sizes)} | {_acc_str(acc)}",
        flush=True,
    )
    return dict(
        device=device, dtype=dtype_name, method="per_tensor", rh=False, asym=False,
        precond=False, qbit=qbit, threads=threads, t_split=0.0, t_quant=t_quant,
        t_old=t_quant, t_new=None, speedup=None, t_dq_old=t_dq_old, t_dq_new=None,
        dq_speedup=None, dq_equal=None, **sizes, **acc,
    )


def run_case(
    args, device: str, dtype_name: str, method: str, rh: bool, asym: bool,
    precond: bool, qbit: int, threads: int,
) -> dict:
    layers, tokens = args.layers, args.tokens
    hidden = 2 * args.kv_heads * args.head_dim
    dtype = getattr(torch, dtype_name)
    native = kvweave_quant if device == "cpu" else kvweave_quant_xpu
    settings = dict(
        qbit=qbit, scaling_method=method, rh=rh, asym=asym, precond=precond,
        num_threads=threads, num_kv_heads=args.kv_heads, head_dim=args.head_dim,
        device=device,
    )
    codec = KVWeaveCodec(settings)
    old_codec = KVWeaveCodec(dict(settings, fused_direct_dequant=False))
    raw = _make_chunk(layers, tokens, hidden, dtype)
    head_num = codec._fused_head_num(hidden)
    head_dim = codec._fused_head_dim(hidden)
    signs = perm = None
    if rh and precond:
        size = codec._fused_rh_transform_size(tokens, hidden, head_dim, method, True)
        signs, perm = codec._config.mamba_precond_tensors(size)

    reference = codec.serialize_fused_tensor(raw)
    header_len = len(KVWeaveCodecConfig.MAGIC_QUANT_FUSED) + 5 + 4 * raw.dim()
    header = reference[:header_len]
    if method == "per_tensor":
        return _run_per_tensor(args, device, dtype_name, codec, native, raw, header, qbit, threads)
    split_out = torch.empty(layers * head_num, tokens, head_dim, dtype=dtype)
    out_buf = torch.empty(len(reference) + 4096, dtype=torch.uint8)

    def old_split():
        kvweave_quant.kvweave_split_fused_heads(raw, head_num, head_dim, threads, split_out)

    def old_quant():
        return native.kvweave_serialize_chunk_state(
            split_out.view(-1), header, KVWeaveCodecConfig.next_scale_id(),
            qbit=qbit, blocks_num=max(1, tokens // codec.block_size),
            block_size=codec.block_size, head_num=head_num, head_dim=head_dim,
            num_layers=layers * head_num, rh=rh, asym=asym, scaling_method=method,
            signs=signs, perm=perm, num_threads=threads,
        )

    def new_direct(out=None):
        return native.kvweave_serialize_fused_chunk_direct(
            raw, header, qbit, head_num, head_dim, rh=rh, asym=asym,
            scaling_method=method, signs=signs, perm=perm, num_threads=threads, out=out,
        )

    old_split()
    written = new_direct(out_buf)
    payload = torch.frombuffer(bytearray(bytes(out_buf[:written].numpy())), dtype=torch.uint8)
    decoded_new = torch.empty_like(raw)
    decoded_old = torch.empty_like(raw)
    codec.deserialize_fused_tensor(payload, decoded_new)
    old_codec.deserialize_fused_tensor(payload, decoded_old)
    dq_equal = torch.equal(decoded_new, decoded_old)
    acc = _accuracy(raw, decoded_new, qbit)
    sizes = _sizes(raw, len(bytes(old_quant())), int(written))

    t_split = _median_ms(old_split, args.iters, args.warmup)
    t_quant = _median_ms(old_quant, args.iters, args.warmup)
    t_new = _median_ms(lambda: new_direct(out_buf), args.iters, args.warmup)
    t_old = t_split + t_quant
    t_dq_old = _median_ms(
        lambda: old_codec.deserialize_fused_tensor(payload, decoded_old), args.iters, args.warmup)
    t_dq_new = _median_ms(
        lambda: codec.deserialize_fused_tensor(payload, decoded_new), args.iters, args.warmup)
    print(
        f"{device:<3} {dtype_name:<8} {method:<11} rh={int(rh)} asym={int(asym)} "
        f"precond={int(precond)} qbit={qbit} thr={threads:<2} | "
        f"old split={t_split:7.3f} quant={t_quant:8.3f} total={t_old:8.3f} ms | "
        f"new={t_new:7.3f} ms | speedup={t_old / t_new:5.2f}x | "
        f"decode old={t_dq_old:7.3f} new={t_dq_new:7.3f} ms speedup={t_dq_old / t_dq_new:5.2f}x "
        f"same={'YES' if dq_equal else 'NO'} | {_size_str(sizes)} | new {_acc_str(acc)}",
        flush=True,
    )
    return dict(
        device=device, dtype=dtype_name, method=method, rh=rh, asym=asym,
        precond=precond, qbit=qbit, threads=threads, t_split=t_split,
        t_quant=t_quant, t_old=t_old, t_new=t_new, speedup=t_old / t_new,
        t_dq_old=t_dq_old, t_dq_new=t_dq_new, dq_speedup=t_dq_old / t_dq_new,
        dq_equal=dq_equal, **sizes, **acc,
    )


def _print_summary(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n=== summary (enc/dec old = split path, new = fused direct; "
          "accuracy = new decode vs original input) ===")
    print(
        f"{'dev':<4}{'dtype':<10}{'method':<12}{'rh':>3}{'asym':>5}{'pc':>4}{'qbit':>5}"
        f"{'thr':>5}{'enc old':>9}{'enc new':>9}{'enc x':>8}"
        f"{'dec old':>9}{'dec new':>9}{'dec x':>8}{'same':>6}"
        f"{'raw MB':>9}{'old MB':>9}{'new MB':>9}{'ratio':>7}"
        f"{'rel_mse':>12}{'snr dB':>8}{'cos':>9}{'acc':>5}"
    )
    for r in rows:
        new = "n/a" if r["t_new"] is None else f"{r['t_new']:.3f}"
        speedup = "n/a" if r["speedup"] is None else f"{r['speedup']:.2f}x"
        dq_new = "n/a" if r["t_dq_new"] is None else f"{r['t_dq_new']:.3f}"
        dq_speedup = "n/a" if r["dq_speedup"] is None else f"{r['dq_speedup']:.2f}x"
        same = "-" if r["dq_equal"] is None else ("YES" if r["dq_equal"] else "NO")
        new_mb = "n/a" if r["new_bytes"] is None else f"{r['new_bytes'] / 2**20:.3f}"
        ratio = r["new_ratio"] if r["new_ratio"] is not None else r["old_ratio"]
        print(
            f"{r['device']:<4}{r['dtype']:<10}{r['method']:<12}{int(r['rh']):>3}"
            f"{int(r['asym']):>5}{int(r['precond']):>4}{r['qbit']:>5}{r['threads']:>5}"
            f"{r['t_old']:>9.3f}{new:>9}{speedup:>8}"
            f"{r['t_dq_old']:>9.3f}{dq_new:>9}{dq_speedup:>8}{same:>6}"
            f"{r['raw_bytes'] / 2**20:>9.3f}{r['old_bytes'] / 2**20:>9.3f}{new_mb:>9}"
            f"{ratio:>6.2f}x"
            f"{r['rel_mse']:>12.4e}{r['snr_db']:>8.1f}{r['cos']:>9.5f}"
            f"{'OK' if r['acc_ok'] else 'BAD':>5}"
        )
    for device in sorted({r["device"] for r in rows}):
        for label, key in (("encode", "speedup"), ("decode", "dq_speedup")):
            speedups = [r[key] for r in rows if r["device"] == device and r[key]]
            if not speedups:
                continue
            geo = statistics.geometric_mean(speedups)
            print(
                f"{device} {label}: {len(speedups)} configs, speedup geomean={geo:.2f}x "
                f"min={min(speedups):.2f}x max={max(speedups):.2f}x"
            )
    mismatched = [r for r in rows if r["dq_equal"] is False]
    print(f"decode new vs old identical: {sum(r['dq_equal'] is True for r in rows)} YES, "
          f"{len(mismatched)} NO")
    bad = [r for r in rows if not r["acc_ok"]]
    print(
        f"accuracy (rel_mse < {_MAX_REL_MSE}): {len(bad)} BAD of {len(rows)}, "
        f"worst rel_mse per qbit: "
        + ", ".join(
            f"{q}bit={max(r['rel_mse'] for r in rows if r['qbit'] == q):.4e}"
            for q in sorted({r['qbit'] for r in rows})
        )
    )

    # Same config on CPU and XPU should land at about the same reconstruction error.
    by_cfg: dict[tuple, dict[str, float]] = {}
    for r in rows:
        key = (r["dtype"], r["method"], r["rh"], r["asym"], r["precond"], r["qbit"])
        by_cfg.setdefault(key, {})[r["device"]] = r["rel_mse"]
    pairs = [(k, v) for k, v in by_cfg.items() if {"cpu", "xpu"} <= v.keys()]
    if pairs:
        worst = max(pairs, key=lambda kv: abs(kv[1]["xpu"] / kv[1]["cpu"] - 1))
        ratio = worst[1]["xpu"] / worst[1]["cpu"]
        print(
            f"cpu vs xpu rel_mse: {len(pairs)} configs, worst xpu/cpu ratio={ratio:.4f} "
            f"at {worst[0]}"
        )


def _csv(value: str, cast=str) -> list:
    return [cast(v) for v in value.split(",") if v]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--devices", default="cpu,xpu")
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--methods", default="per_channel,per_token,per_tensor")
    parser.add_argument("--rh", default="1,0")
    parser.add_argument("--asym", default="1,0")
    parser.add_argument("--precond", default="1,0")
    parser.add_argument("--qbits", default="4,8")
    parser.add_argument("--threads", default="4,8")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    devices = _csv(args.devices)
    if "xpu" in devices and not _xpu_available():
        print("xpu: kvweave_quant_xpu direct kernel or XPU device unavailable, skipping")
        devices = [d for d in devices if d != "xpu"]
    print(
        f"shape L={args.layers} T={args.tokens} kv_heads={args.kv_heads} "
        f"head_dim={args.head_dim}"
    )
    rows = []
    seen_per_tensor = set()
    for device, dtype_name, method, rh, asym, precond, qbit, threads in itertools.product(
        devices,
        _csv(args.dtypes),
        _csv(args.methods),
        [bool(int(v)) for v in _csv(args.rh)],
        [bool(int(v)) for v in _csv(args.asym)],
        [bool(int(v)) for v in _csv(args.precond)],
        _csv(args.qbits, int),
        _csv(args.threads, int),
    ):
        # precond only exists on top of RH.
        if precond and not rh:
            continue
        # The codec forces rh/asym/precond off for per_tensor: run it once per combo.
        if method == "per_tensor":
            key = (device, dtype_name, qbit, threads)
            if key in seen_per_tensor:
                continue
            seen_per_tensor.add(key)
            rh = asym = precond = False
        rows.append(run_case(args, device, dtype_name, method, rh, asym, precond, qbit, threads))
    _print_summary(rows)


if __name__ == "__main__":
    main()
