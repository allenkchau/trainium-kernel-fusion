"""
Benchmark and profile baseline vs fused MLP kernels on Trainium.

Modes:
  accuracy  — verify all kernels match NumPy reference
  benchmark — measure latency (p50, p99) via nki.benchmark
  profile   — generate neuron-profile traces for HBM traffic analysis
  sweep     — latency sweep across multiple (M, K, N) configurations
  all       — run everything

Usage (on a Trainium instance):
  python mlp_benchmark.py --mode accuracy
  python mlp_benchmark.py --mode benchmark --M 512 --K 1024 --N 2048
  python mlp_benchmark.py --mode sweep
  python mlp_benchmark.py --mode all
"""

import argparse
import os
import sys
import numpy as np

from neuronxcc import nki
from neuronxcc.nki import benchmark

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from kernels.baseline.mlp_gemm import mlp_baseline, mlp_reference, gemm_kernel, gelu_kernel
from kernels.fused.mlp import mlp_fused, fused_mlp_pwl, fused_mlp_taylor

PROFILE_DIR = os.path.join(PROJECT_ROOT, "profiling")

# Default sweep configurations: (M, K, N)
# All dimensions must be multiples of tile sizes (M%128==0, K%128==0, N%512==0)
DEFAULT_CONFIGS = [
    (128,  128,  512),
    (256,  512,  1024),
    (512,  1024, 2048),
    (1024, 2048, 4096),
    (2048, 4096, 4096),
]

KERNELS = [
    ("baseline",     "pwl"),
    ("fused_pwl",    "pwl"),
    ("fused_taylor", "taylor"),
]


def configure_neuron_target(target):
    if target == "auto":
        return
    target_flag = f"--target={target}"
    existing = os.environ.get("NEURON_CC_FLAGS", "").strip()
    tokens = existing.split()
    if target_flag not in tokens:
        os.environ["NEURON_CC_FLAGS"] = (
            f"{existing} {target_flag}".strip() if existing else target_flag
        )
    print(f"[config] NEURON_CC_FLAGS={os.environ['NEURON_CC_FLAGS']}")


def make_inputs(M, K, N, dtype=np.float32):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(M, K)).astype(dtype)
    W = rng.normal(size=(K, N)).astype(dtype)
    bias = rng.normal(size=(N,)).astype(dtype)
    return X, W, bias


def run_kernel(name, X, W, bias):
    if name == "baseline":
        return mlp_baseline(X, W, bias)
    elif name == "fused_pwl":
        return mlp_fused(X, W, bias, mode="pwl")
    elif name == "fused_taylor":
        return mlp_fused(X, W, bias, mode="taylor")
    raise ValueError(f"Unknown kernel: {name}")


def bench_kernel(name, X, W, bias, warmup, iters):
    """Benchmark a kernel and return (p50_us, p99_us)."""
    lhsT = X.T.copy()
    bias_2d = bias.reshape(1, -1)

    if name == "baseline":
        # Benchmark the two separate launches together
        # Benchmark GEMM
        gemm_bench = benchmark(warmup=warmup, iters=iters)(gemm_kernel)
        gemm_bench(lhsT, W, bias_2d)
        gemm_p50 = gemm_bench.benchmark_result.nc_latency.get_latency_percentile(50)
        gemm_p99 = gemm_bench.benchmark_result.nc_latency.get_latency_percentile(99)
        # Benchmark GELU (need intermediate Y)
        Y = gemm_kernel(lhsT, W, bias_2d)
        gelu_bench = benchmark(warmup=warmup, iters=iters)(gelu_kernel)
        gelu_bench(Y)
        gelu_p50 = gelu_bench.benchmark_result.nc_latency.get_latency_percentile(50)
        gelu_p99 = gelu_bench.benchmark_result.nc_latency.get_latency_percentile(99)
        return gemm_p50 + gelu_p50, gemm_p99 + gelu_p99
    else:
        kernel_fn = fused_mlp_pwl if name == "fused_pwl" else fused_mlp_taylor
        bench_fn = benchmark(warmup=warmup, iters=iters)(kernel_fn)
        bench_fn(lhsT, W, bias_2d)
        latency = bench_fn.benchmark_result.nc_latency
        return (latency.get_latency_percentile(50),
                latency.get_latency_percentile(99))


# -----------------------------------------------------------------------
# Accuracy
# -----------------------------------------------------------------------
def run_accuracy(M, K, N):
    print(f"\n{'='*60}")
    print(f"  ACCURACY TEST   M={M}  K={K}  N={N}")
    print(f"{'='*60}")

    X, W, bias = make_inputs(M, K, N)
    ref = mlp_reference(X, W, bias)

    for name, _ in KERNELS:
        out = run_kernel(name, X, W, bias)
        max_err = np.max(np.abs(out.astype(np.float32) - ref))
        ok = np.allclose(out.astype(np.float32), ref, atol=1e-2)
        status = "PASS" if ok else "FAIL"
        print(f"  {name:14s}  max_abs_err={max_err:.6e}  [{status}]")


# -----------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------
def run_benchmark(M, K, N, warmup=5, iters=20):
    print(f"\n{'='*60}")
    print(f"  BENCHMARK   M={M}  K={K}  N={N}")
    print(f"  warmup={warmup}  iters={iters}")
    print(f"{'='*60}")

    X, W, bias = make_inputs(M, K, N)

    for name, _ in KERNELS:
        p50, p99 = bench_kernel(name, X, W, bias, warmup, iters)
        print(f"  {name:14s}  p50={p50:>8.1f} us   p99={p99:>8.1f} us")


# -----------------------------------------------------------------------
# Profile
# -----------------------------------------------------------------------
def run_profile(M, K, N):
    print(f"\n{'='*60}")
    print(f"  PROFILE   M={M}  K={K}  N={N}")
    print(f"{'='*60}")

    X, W, bias = make_inputs(M, K, N)
    lhsT = X.T.copy()
    bias_2d = bias.reshape(1, -1)

    profile_targets = [
        ("baseline_gemm", gemm_kernel, (lhsT, W, bias_2d)),
        ("fused_pwl",     fused_mlp_pwl, (lhsT, W, bias_2d)),
        ("fused_taylor",  fused_mlp_taylor, (lhsT, W, bias_2d)),
    ]

    for name, kernel, args in profile_targets:
        out_dir = os.path.join(PROFILE_DIR, f"{name}_mlp")
        os.makedirs(out_dir, exist_ok=True)

        profile_fn = nki.profile(
            working_directory=out_dir,
            save_neff_name=f"{name}.neff",
            save_trace_name=f"{name}.ntff",
            profile_nth=2,
        )(kernel)

        profile_fn(*args)
        print(f"  {name:14s}  traces saved to {out_dir}/")


# -----------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------
def run_sweep(warmup=5, iters=20):
    print(f"\n{'='*60}")
    print(f"  LATENCY SWEEP")
    print(f"{'='*60}")
    print(f"  {'config':>20s}  {'baseline (us)':>14s}  {'fused_pwl (us)':>14s}  "
          f"{'fused_taylor (us)':>17s}  {'pwl spdup':>9s}  {'taylor spdup':>12s}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*14}  {'-'*17}  {'-'*9}  {'-'*12}")

    for M, K, N in DEFAULT_CONFIGS:
        X, W, bias = make_inputs(M, K, N)
        latencies = {}
        for name, _ in KERNELS:
            p50, _ = bench_kernel(name, X, W, bias, warmup, iters)
            latencies[name] = p50

        spd_pwl = latencies["baseline"] / latencies["fused_pwl"]
        spd_tay = latencies["baseline"] / latencies["fused_taylor"]
        label = f"{M}x{K}x{N}"
        print(f"  {label:>20s}  {latencies['baseline']:>14.1f}  "
              f"{latencies['fused_pwl']:>14.1f}  {latencies['fused_taylor']:>17.1f}  "
              f"{spd_pwl:>8.2f}x  {spd_tay:>11.2f}x")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark baseline vs fused MLP on Trainium")
    parser.add_argument("--mode", default="all",
                        choices=["accuracy", "benchmark", "profile",
                                 "sweep", "all"])
    parser.add_argument("--M", type=int, default=256)
    parser.add_argument("--K", type=int, default=512)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--target", default="auto",
                        choices=["auto", "trn1", "trn2"])
    args = parser.parse_args()
    configure_neuron_target(args.target)

    if args.mode in ("accuracy", "all"):
        run_accuracy(args.M, args.K, args.N)

    if args.mode in ("benchmark", "all"):
        run_benchmark(args.M, args.K, args.N, args.warmup, args.iters)

    if args.mode in ("profile", "all"):
        run_profile(args.M, args.K, args.N)

    if args.mode in ("sweep", "all"):
        run_sweep(args.warmup, args.iters)
