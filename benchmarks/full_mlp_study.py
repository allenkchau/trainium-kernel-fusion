"""
End-to-end benchmark + profiling study for baseline vs fused MLP kernels.

Outputs:
  - Latency and throughput sweep for selected (M, K, N) configurations
  - Precision trade-off metrics (fp32 vs bf16, PWL vs Taylor GELU)
  - Neuron profile artifacts (.neff/.ntff)
  - Profile summaries (DMA queue/event/cycle counts)
  - JSON + CSV reports under profiling/reports/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neuronxcc.nki import benchmark
from neuronxcc import nki

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.baseline.mlp_gemm import (
    mlp_baseline, mlp_reference, gemm_kernel, gelu_kernel,
)
from kernels.fused.mlp import mlp_fused, fused_mlp_pwl, fused_mlp_taylor

# Default (M, K, N) configs — all tile-aligned
DEFAULT_CONFIGS = [
    (128,  128,  512),
    (256,  512,  1024),
    (512,  1024, 2048),
    (1024, 2048, 4096),
    (2048, 4096, 4096),
]


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
    os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")


def available_dtypes() -> dict[str, np.dtype]:
    out: dict[str, np.dtype] = {"fp32": np.float32}
    try:
        import ml_dtypes
        out["bf16"] = ml_dtypes.bfloat16
    except Exception:
        pass
    return out


def make_inputs(M: int, K: int, N: int, dtype: np.dtype, seed: int = 42):
    rng = np.random.default_rng(seed)
    X_f32 = rng.normal(size=(M, K)).astype(np.float32)
    W_f32 = rng.normal(size=(K, N)).astype(np.float32)
    bias_f32 = rng.normal(size=(N,)).astype(np.float32)
    X = X_f32.astype(dtype)
    W = W_f32.astype(dtype)
    bias = bias_f32.astype(dtype)
    return X, W, bias, X_f32, W_f32, bias_f32


def config_label(M: int, K: int, N: int) -> str:
    return f"{M}x{K}x{N}"


def approx_mlp_flops(M: int, K: int, N: int) -> float:
    """2*M*N*K for GEMM + ~8*M*N for GELU element-wise ops."""
    return float(2 * M * N * K + 8 * M * N)


def hbm_model_bytes(M: int, K: int, N: int, input_dtype: np.dtype) -> dict[str, int]:
    dtype_bytes = np.dtype(input_dtype).itemsize
    # Reads: X[M,K] + W[K,N] + bias[N]
    input_bytes = (M * K + K * N + N) * dtype_bytes
    # Writes: Z[M,N]
    output_bytes = M * N * dtype_bytes
    # Baseline extra: intermediate Y[M,N] written then read (FP32 accum cast back)
    intermediate_bytes = 2 * M * N * dtype_bytes
    return {
        "baseline_hbm_bytes_model": int(input_bytes + output_bytes + intermediate_bytes),
        "fused_hbm_bytes_model": int(input_bytes + output_bytes),
    }


def bench_kernel(name: str, X, W, bias, warmup: int, iters: int):
    """Return (p50_us, p99_us) for a kernel."""
    lhsT = X.T.copy()
    bias_2d = bias.reshape(1, -1)

    if name == "baseline":
        gemm_bench = benchmark(warmup=warmup, iters=iters)(gemm_kernel)
        gemm_bench(lhsT, W, bias_2d)
        gp50 = gemm_bench.benchmark_result.nc_latency.get_latency_percentile(50)
        gp99 = gemm_bench.benchmark_result.nc_latency.get_latency_percentile(99)
        Y = gemm_kernel(lhsT, W, bias_2d)
        gelu_bench = benchmark(warmup=warmup, iters=iters)(gelu_kernel)
        gelu_bench(Y)
        ep50 = gelu_bench.benchmark_result.nc_latency.get_latency_percentile(50)
        ep99 = gelu_bench.benchmark_result.nc_latency.get_latency_percentile(99)
        return gp50 + ep50, gp99 + ep99
    else:
        kernel_fn = fused_mlp_pwl if name == "fused_pwl" else fused_mlp_taylor
        bench_fn = benchmark(warmup=warmup, iters=iters)(kernel_fn)
        bench_fn(lhsT, W, bias_2d)
        lat = bench_fn.benchmark_result.nc_latency
        return (lat.get_latency_percentile(50), lat.get_latency_percentile(99))


def run_kernel(name: str, X, W, bias):
    if name == "baseline":
        return mlp_baseline(X, W, bias)
    elif name == "fused_pwl":
        return mlp_fused(X, W, bias, mode="pwl")
    elif name == "fused_taylor":
        return mlp_fused(X, W, bias, mode="taylor")
    raise ValueError(name)


KERNEL_NAMES = ["baseline", "fused_pwl", "fused_taylor"]


def run_latency_benchmark(
    M: int, K: int, N: int,
    dtype_name: str, dtype: np.dtype,
    warmup: int, iters: int,
) -> list[dict]:
    X, W, bias, _, _, _ = make_inputs(M, K, N, dtype)
    flops = approx_mlp_flops(M, K, N)
    results = []
    for kernel_name in KERNEL_NAMES:
        p50_us, p99_us = bench_kernel(kernel_name, X, W, bias, warmup, iters)
        seq_per_s = 1_000_000.0 / p50_us
        est_tflops = flops / (p50_us * 1e-6) / 1e12
        row = {
            "dtype": dtype_name,
            "kernel": kernel_name,
            "M": M, "K": K, "N": N,
            "config": config_label(M, K, N),
            "p50_us": float(p50_us),
            "p99_us": float(p99_us),
            "inferences_per_s_p50": seq_per_s,
            "est_tflops_p50": est_tflops,
        }
        row.update(hbm_model_bytes(M, K, N, dtype))
        results.append(row)
    return results


def run_precision_check(
    M: int, K: int, N: int,
    dtype_name: str, dtype: np.dtype,
) -> list[dict]:
    X, W, bias, X_f32, W_f32, bias_f32 = make_inputs(M, K, N, dtype)
    ref = mlp_reference(X_f32, W_f32, bias_f32).astype(np.float32)
    results = []
    for kernel_name in KERNEL_NAMES:
        out = run_kernel(kernel_name, X, W, bias).astype(np.float32)
        abs_err = np.abs(out - ref)
        row = {
            "dtype": dtype_name,
            "kernel": kernel_name,
            "M": M, "K": K, "N": N,
            "config": config_label(M, K, N),
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
    M: int, K: int, N: int,
    dtype_name: str, dtype: np.dtype,
    out_root: Path,
) -> list[dict]:
    X, W, bias, _, _, _ = make_inputs(M, K, N, dtype)
    lhsT = X.T.copy()
    bias_2d = bias.reshape(1, -1)

    profile_targets = [
        ("baseline_gemm", gemm_kernel, (lhsT, W, bias_2d)),
        ("fused_pwl",     fused_mlp_pwl, (lhsT, W, bias_2d)),
        ("fused_taylor",  fused_mlp_taylor, (lhsT, W, bias_2d)),
    ]

    results = []
    for kernel_name, kernel, args in profile_targets:
        out_dir = out_root / f"mlp_{kernel_name}_{dtype_name}_m{M}_k{K}_n{N}"
        out_dir.mkdir(parents=True, exist_ok=True)
        profile_fn = nki.profile(
            working_directory=str(out_dir),
            save_neff_name=f"{kernel_name}.neff",
            save_trace_name=f"{kernel_name}.ntff",
            profile_nth=2,
        )(kernel)
        profile_fn(*args)

        neff_path = out_dir / f"{kernel_name}.neff"
        ntff_path = newest_ntff(out_dir)
        cmd = [
            "/opt/aws/neuron/bin/neuron-profile",
            "show-session", "-s", str(ntff_path),
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        summary = parse_show_session_summary(proc.stdout)
        summary_row = {
            "kernel": kernel_name,
            "dtype": dtype_name,
            "M": M, "K": K, "N": N,
            "config": config_label(M, K, N),
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
        f"{'dtype':>6} {'kernel':>14} {'config':>20} {'p50_us':>10} {'p99_us':>10} "
        f"{'inf/s':>12} {'tflops':>10}"
    )
    for r in rows:
        print(
            f"{r['dtype']:>6} {r['kernel']:>14} {r['config']:>20} "
            f"{r['p50_us']:>10.1f} {r['p99_us']:>10.1f} "
            f"{r['inferences_per_s_p50']:>12.2f} {r['est_tflops_p50']:>10.2f}"
        )


def print_precision_table(rows: list[dict]) -> None:
    print("\nPrecision Trade-offs (vs fp32 NumPy reference)")
    print("-" * 110)
    print(
        f"{'dtype':>6} {'kernel':>14} {'config':>20} {'max_abs_err':>14} "
        f"{'mean_abs_err':>14} {'p99_abs_err':>14} {'allclose':>9}"
    )
    for r in rows:
        print(
            f"{r['dtype']:>6} {r['kernel']:>14} {r['config']:>20} "
            f"{r['max_abs_err']:>14.6e} {r['mean_abs_err']:>14.6e} "
            f"{r['p99_abs_err']:>14.6e} {str(r['allclose_atol_1e2']):>9}"
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_config(s: str) -> tuple[int, int, int]:
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected M,K,N but got: {s}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Full MLP benchmarking/profiling study")
    parser.add_argument("--target", default="trn2", choices=["auto", "trn1", "trn2"])
    parser.add_argument(
        "--configs", nargs="+", type=parse_config,
        default=DEFAULT_CONFIGS,
        help="M,K,N triples (e.g. 256,512,1024 512,1024,2048)",
    )
    parser.add_argument("--profile-config", type=parse_config, default=(256, 512, 1024),
                        help="M,K,N config for profiling (default: 256,512,1024)")
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
            print(f"[warn] dtype '{dname}' not available. Skipping.")
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
        for M, K, N in args.configs:
            print(f"\n[run] dtype={dname} config={config_label(M, K, N)}")
            latency_rows.extend(
                run_latency_benchmark(M, K, N, dname, dtype, args.warmup, args.iters)
            )
            precision_rows.extend(
                run_precision_check(M, K, N, dname, dtype)
            )

        pM, pK, pN = args.profile_config
        print(f"\n[profile] dtype={dname} config={config_label(pM, pK, pN)}")
        profile_rows.extend(
            run_profile_and_summarize(pM, pK, pN, dname, dtype, artifacts_dir)
        )

    # Speedup rows
    idx = {(r["dtype"], r["config"], r["kernel"]): r for r in latency_rows}
    speedup_rows = []
    for dname, _ in selected:
        for M, K, N in args.configs:
            label = config_label(M, K, N)
            b = idx[(dname, label, "baseline")]
            for variant in ["fused_pwl", "fused_taylor"]:
                f = idx[(dname, label, variant)]
                speedup_rows.append({
                    "dtype": dname,
                    "config": label,
                    "M": M, "K": K, "N": N,
                    "variant": variant,
                    "p50_speedup_x": b["p50_us"] / f["p50_us"],
                    "hbm_model_reduction_x": (
                        b["baseline_hbm_bytes_model"] / f["fused_hbm_bytes_model"]
                    ),
                })

    summary = {
        "generated_at_utc": ts,
        "config": {
            "target": args.target,
            "configs": [list(c) for c in args.configs],
            "profile_config": list(args.profile_config),
            "warmup": args.warmup,
            "iters": args.iters,
            "dtypes": [d for d, _ in selected],
        },
        "latency_throughput": latency_rows,
        "precision_tradeoffs": precision_rows,
        "speedups": speedup_rows,
        "profile_summaries": profile_rows,
    }

    write_csv(reports_dir / f"mlp_latency_throughput_{ts}.csv", latency_rows)
    write_csv(reports_dir / f"mlp_precision_tradeoffs_{ts}.csv", precision_rows)
    write_csv(reports_dir / f"mlp_profile_summaries_{ts}.csv", profile_rows)
    write_csv(reports_dir / f"mlp_speedups_{ts}.csv", speedup_rows)
    with (reports_dir / f"mlp_summary_{ts}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_latency_table(latency_rows)
    print_precision_table(precision_rows)
    print("\nSpeedups")
    print("-" * 110)
    for s in speedup_rows:
        print(
            f"dtype={s['dtype']} config={s['config']} variant={s['variant']} "
            f"p50_speedup={s['p50_speedup_x']:.2f}x "
            f"hbm_model_reduction={s['hbm_model_reduction_x']:.2f}x"
        )
    print(f"\nReports written to: {reports_dir}")
    print(f"Profile artifacts written to: {artifacts_dir}")


if __name__ == "__main__":
    main()
