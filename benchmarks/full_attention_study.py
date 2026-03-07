"""
End-to-end benchmark + profiling study for baseline vs fused attention kernels.

Outputs:
  - Latency and throughput sweep for selected sequence lengths
  - Precision trade-off metrics (fp32 vs bf16)
  - Neuron profile artifacts (.neff/.ntff)
  - Profile summaries (DMA queue/event/cycle counts)
  - JSON + CSV reports under profiling/reports/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neuronxcc.nki import benchmark
from neuronxcc import nki

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.baseline.attention import baseline_attention, numpy_attention
from kernels.fused.attention import fused_attention


def configure_neuron_target(target: str) -> None:
    if target == "auto":
        return
    target_flag = f"--target={target}"
    existing = os.environ.get("NEURON_CC_FLAGS", "").strip()
    tokens = existing.split()
    if target_flag not in tokens:
        os.environ["NEURON_CC_FLAGS"] = (
            f"{existing} {target_flag}".strip() if existing else target_flag
        )


def ensure_neuron_tools_available() -> None:
    neuron_bin = "/opt/aws/neuron/bin"
    path_entries = os.environ.get("PATH", "").split(":")
    if neuron_bin not in path_entries and Path(neuron_bin).exists():
        os.environ["PATH"] = f"{neuron_bin}:{os.environ.get('PATH', '')}"

    # nki.profile may require 2 logical NCs for this workload.
    os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")


def available_dtypes() -> dict[str, np.dtype]:
    out: dict[str, np.dtype] = {"fp32": np.float32}
    try:
        import ml_dtypes

        out["bf16"] = ml_dtypes.bfloat16
    except Exception:
        pass
    return out


def make_inputs(d_head: int, seqlen: int, dtype: np.dtype, seed: int = 42):
    rng = np.random.default_rng(seed)
    q_f32 = ((rng.random((d_head, seqlen)) - 0.5) * 2).astype(np.float32)
    k_f32 = ((rng.random((d_head, seqlen)) - 0.5) * 2).astype(np.float32)
    v_f32 = ((rng.random((d_head, seqlen)) - 0.5) * 2).astype(np.float32)
    q = q_f32.astype(dtype)
    k = k_f32.astype(dtype)
    v = v_f32.astype(dtype)
    return q, k, v, q_f32, k_f32, v_f32


def approx_attention_flops(d_head: int, seqlen: int) -> float:
    # Two GEMMs: Q^T K and P V^T. Using 2 * m*n*k for each GEMM.
    return float(4 * d_head * seqlen * seqlen)


def hbm_model_bytes(d_head: int, seqlen: int, input_dtype: np.dtype) -> dict[str, int]:
    dtype_bytes = np.dtype(input_dtype).itemsize
    input_bytes = 3 * d_head * seqlen * dtype_bytes
    output_bytes = seqlen * d_head * dtype_bytes
    # Baseline has HBM round-trips on score/weight matrix (float32 intermediates).
    score_matrix_bytes = seqlen * seqlen * np.dtype(np.float32).itemsize
    baseline_bytes = input_bytes + output_bytes + (4 * score_matrix_bytes)
    fused_bytes = input_bytes + output_bytes
    return {
        "baseline_hbm_bytes_model": int(baseline_bytes),
        "fused_hbm_bytes_model": int(fused_bytes),
    }


def run_latency_benchmark(
    d_head: int,
    seqlen: int,
    dtype_name: str,
    dtype: np.dtype,
    warmup: int,
    iters: int,
) -> list[dict]:
    q, k, v, _, _, _ = make_inputs(d_head, seqlen, dtype)
    flops = approx_attention_flops(d_head, seqlen)
    results = []
    for kernel_name, kernel in [("baseline", baseline_attention), ("fused", fused_attention)]:
        bench_fn = benchmark(warmup=warmup, iters=iters)(kernel)
        bench_fn(q, k, v)
        latency = bench_fn.benchmark_result.nc_latency
        p50_us = float(latency.get_latency_percentile(50))
        p99_us = float(latency.get_latency_percentile(99))
        seq_per_s = 1_000_000.0 / p50_us
        tok_per_s = seq_per_s * seqlen
        est_tflops = flops / (p50_us * 1e-6) / 1e12
        row = {
            "dtype": dtype_name,
            "kernel": kernel_name,
            "seqlen": seqlen,
            "d_head": d_head,
            "p50_us": p50_us,
            "p99_us": p99_us,
            "seq_per_s_p50": seq_per_s,
            "tok_per_s_p50": tok_per_s,
            "est_tflops_p50": est_tflops,
        }
        row.update(hbm_model_bytes(d_head, seqlen, dtype))
        results.append(row)
    return results


def run_precision_check(
    d_head: int, seqlen: int, dtype_name: str, dtype: np.dtype
) -> list[dict]:
    q, k, v, q_f32, k_f32, v_f32 = make_inputs(d_head, seqlen, dtype)
    ref = numpy_attention(q_f32, k_f32, v_f32).astype(np.float32)
    results = []
    for kernel_name, kernel in [("baseline", baseline_attention), ("fused", fused_attention)]:
        out = kernel(q, k, v).astype(np.float32)
        abs_err = np.abs(out - ref)
        row = {
            "dtype": dtype_name,
            "kernel": kernel_name,
            "seqlen": seqlen,
            "d_head": d_head,
            "max_abs_err": float(np.max(abs_err)),
            "mean_abs_err": float(np.mean(abs_err)),
            "p99_abs_err": float(np.percentile(abs_err, 99)),
            "allclose_atol_1e2": bool(np.allclose(out, ref, atol=1e-2)),
        }
        results.append(row)
    return results


def newest_ntff(out_dir: Path) -> Path:
    ntffs = sorted(out_dir.glob("*.ntff"), key=lambda p: p.stat().st_mtime)
    if not ntffs:
        raise FileNotFoundError(f"No .ntff generated in {out_dir}")
    return ntffs[-1]


def parse_show_session_summary(text: str) -> dict[str, int]:
    cycles = 0
    events = 0
    dma_queues = 0
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "NODE ID" in line or "---" in line:
            continue
        # Columns:
        # | NODE ID | EXEC TIME (NS) | EXECUTOR | NAME | TRACE | CYCLES | EVENTS | ...
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 13:
            continue
        try:
            cycles += int(parts[6])
            events += int(parts[7])
            dma_queues += int(parts[12])
        except ValueError:
            continue
    return {
        "cycles_sum": cycles,
        "events_sum": events,
        "dma_queues_sum": dma_queues,
    }


def run_profile_and_summarize(
    d_head: int, seqlen: int, dtype_name: str, dtype: np.dtype, out_root: Path
) -> list[dict]:
    q, k, v, _, _, _ = make_inputs(d_head, seqlen, dtype)
    results = []
    for kernel_name, kernel in [("baseline", baseline_attention), ("fused", fused_attention)]:
        out_dir = out_root / f"{kernel_name}_{dtype_name}_s{seqlen}"
        out_dir.mkdir(parents=True, exist_ok=True)
        profile_fn = nki.profile(
            working_directory=str(out_dir),
            save_neff_name=f"{kernel_name}.neff",
            save_trace_name=f"{kernel_name}.ntff",
            profile_nth=2,
        )(kernel)
        profile_fn(q, k, v)

        neff_path = out_dir / f"{kernel_name}.neff"
        ntff_path = newest_ntff(out_dir)
        cmd = [
            "/opt/aws/neuron/bin/neuron-profile",
            "show-session",
            "-s",
            str(ntff_path),
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        summary = parse_show_session_summary(proc.stdout)
        summary_row = {
            "kernel": kernel_name,
            "dtype": dtype_name,
            "seqlen": seqlen,
            "d_head": d_head,
            "neff_path": str(neff_path),
            "ntff_path": str(ntff_path),
        }
        summary_row.update(summary)
        results.append(summary_row)
    return results


def print_latency_table(rows: list[dict]) -> None:
    print("\nLatency / Throughput")
    print("-" * 110)
    print(
        f"{'dtype':>6} {'kernel':>9} {'seqlen':>8} {'p50_us':>10} {'p99_us':>10} "
        f"{'seq/s':>12} {'tok/s':>14} {'tflops':>10}"
    )
    for r in rows:
        print(
            f"{r['dtype']:>6} {r['kernel']:>9} {r['seqlen']:>8d} "
            f"{r['p50_us']:>10.1f} {r['p99_us']:>10.1f} "
            f"{r['seq_per_s_p50']:>12.2f} {r['tok_per_s_p50']:>14.2f} {r['est_tflops_p50']:>10.2f}"
        )


def print_precision_table(rows: list[dict]) -> None:
    print("\nPrecision Trade-offs (vs fp32 NumPy reference)")
    print("-" * 110)
    print(
        f"{'dtype':>6} {'kernel':>9} {'seqlen':>8} {'max_abs_err':>14} "
        f"{'mean_abs_err':>14} {'p99_abs_err':>14} {'allclose':>9}"
    )
    for r in rows:
        print(
            f"{r['dtype']:>6} {r['kernel']:>9} {r['seqlen']:>8d} "
            f"{r['max_abs_err']:>14.6e} {r['mean_abs_err']:>14.6e} {r['p99_abs_err']:>14.6e} "
            f"{str(r['allclose_atol_1e2']):>9}"
        )


def print_hbm_proxy(rows: list[dict], profile_rows: list[dict]) -> None:
    print("\nHBM Traffic")
    print("-" * 110)
    print("Modeled HBM bytes (from kernel structure):")
    for r in rows:
        if r["seqlen"] == rows[0]["seqlen"] and r["dtype"] == rows[0]["dtype"]:
            print(
                f"  {r['kernel']:>9} dtype={r['dtype']} seqlen={r['seqlen']}: "
                f"{r[f'{r['kernel']}_hbm_bytes_model'] / (1024**2):.2f} MiB"
            )
    if profile_rows:
        print("Profiler DMA queue count proxy (from neuron-profile show-session):")
        for p in profile_rows:
            print(
                f"  {p['kernel']:>9} dtype={p['dtype']} seqlen={p['seqlen']}: "
                f"dma_queues_sum={p['dma_queues_sum']} events_sum={p['events_sum']} "
                f"cycles_sum={p['cycles_sum']}"
            )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Full attention benchmarking/profiling study")
    parser.add_argument("--target", default="trn2", choices=["auto", "trn1", "trn2"])
    parser.add_argument("--d_head", type=int, default=128)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--profile-seqlen", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtypes", nargs="+", default=["fp32", "bf16"])
    args = parser.parse_args()

    ensure_neuron_tools_available()
    configure_neuron_target(args.target)

    dtypes_map = available_dtypes()
    selected = []
    for dname in args.dtypes:
        if dname not in dtypes_map:
            print(f"[warn] dtype '{dname}' not available in this environment. Skipping.")
            continue
        selected.append((dname, dtypes_map[dname]))
    if not selected:
        raise RuntimeError("No valid dtypes selected.")

    reports_dir = PROJECT_ROOT / "profiling" / "reports"
    artifacts_dir = PROJECT_ROOT / "profiling" / "artifacts"
    reports_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    latency_rows: list[dict] = []
    precision_rows: list[dict] = []
    profile_rows: list[dict] = []

    for dname, dtype in selected:
        for seqlen in args.seqlens:
            latency_rows.extend(
                run_latency_benchmark(
                    d_head=args.d_head,
                    seqlen=seqlen,
                    dtype_name=dname,
                    dtype=dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                )
            )
            precision_rows.extend(
                run_precision_check(
                    d_head=args.d_head,
                    seqlen=seqlen,
                    dtype_name=dname,
                    dtype=dtype,
                )
            )

        profile_rows.extend(
            run_profile_and_summarize(
                d_head=args.d_head,
                seqlen=args.profile_seqlen,
                dtype_name=dname,
                dtype=dtype,
                out_root=artifacts_dir,
            )
        )

    # Speedup convenience fields
    idx = {(r["dtype"], r["seqlen"], r["kernel"]): r for r in latency_rows}
    speedup_rows = []
    for dname, _ in selected:
        for seqlen in args.seqlens:
            b = idx[(dname, seqlen, "baseline")]
            f = idx[(dname, seqlen, "fused")]
            speedup_rows.append(
                {
                    "dtype": dname,
                    "seqlen": seqlen,
                    "p50_speedup_x": b["p50_us"] / f["p50_us"],
                    "tok_per_s_speedup_x": f["tok_per_s_p50"] / b["tok_per_s_p50"],
                    "hbm_model_reduction_x": b["baseline_hbm_bytes_model"] / f["fused_hbm_bytes_model"],
                }
            )

    summary = {
        "generated_at_utc": ts,
        "config": {
            "target": args.target,
            "d_head": args.d_head,
            "seqlens": args.seqlens,
            "profile_seqlen": args.profile_seqlen,
            "warmup": args.warmup,
            "iters": args.iters,
            "dtypes": [d for d, _ in selected],
        },
        "latency_throughput": latency_rows,
        "precision_tradeoffs": precision_rows,
        "speedups": speedup_rows,
        "profile_summaries": profile_rows,
    }

    write_csv(reports_dir / f"latency_throughput_{ts}.csv", latency_rows)
    write_csv(reports_dir / f"precision_tradeoffs_{ts}.csv", precision_rows)
    write_csv(reports_dir / f"profile_summaries_{ts}.csv", profile_rows)
    write_csv(reports_dir / f"speedups_{ts}.csv", speedup_rows)
    with (reports_dir / f"summary_{ts}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_latency_table(latency_rows)
    print_precision_table(precision_rows)
    print("\nSpeedups")
    print("-" * 110)
    for s in speedup_rows:
        print(
            f"dtype={s['dtype']} seqlen={s['seqlen']} "
            f"p50_speedup={s['p50_speedup_x']:.2f}x "
            f"tok/s_speedup={s['tok_per_s_speedup_x']:.2f}x "
            f"hbm_model_reduction={s['hbm_model_reduction_x']:.2f}x"
        )
    print_hbm_proxy(latency_rows, profile_rows)
    print(f"\nReports written to: {reports_dir}")
    print(f"Profile artifacts written to: {artifacts_dir}")


if __name__ == "__main__":
    main()
