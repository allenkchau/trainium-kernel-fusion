"""
Benchmark and profile baseline vs fused attention kernels on Trainium.

Three modes:
  accuracy  — verify both kernels match NumPy reference
  benchmark — measure latency (p50, p99) via nki.benchmark
  profile   — generate neuron-profile traces for HBM traffic analysis

Usage (on a Trainium instance, trn1 or trn2):
  python attention_benchmark.py --mode accuracy
  python attention_benchmark.py --mode benchmark
  python attention_benchmark.py --mode profile
  python attention_benchmark.py --mode all
"""

import argparse
import os
import sys
import numpy as np

from neuronxcc import nki
from neuronxcc.nki import benchmark, baremetal
import neuronxcc.nki.language as nl

# Add project root to path so we can import kernels
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from kernels.baseline.attention import baseline_attention, numpy_attention
from kernels.fused.attention import fused_attention

PROFILE_DIR = os.path.join(PROJECT_ROOT, "profiling")


def configure_neuron_target(target):
    """
    Optionally pin Neuron compiler target via NEURON_CC_FLAGS.

    NKI JIT compilation will respect this environment variable.
    """
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


def make_inputs(d_head, seqlen, dtype=np.float32):
    """Generate random Q, K, V inputs in (d_head, seqlen) layout."""
    rng = np.random.default_rng(42)
    q = ((rng.random((d_head, seqlen)) - 0.5) * 2).astype(dtype)
    k = ((rng.random((d_head, seqlen)) - 0.5) * 2).astype(dtype)
    v = ((rng.random((d_head, seqlen)) - 0.5) * 2).astype(dtype)
    return q, k, v


# -----------------------------------------------------------------------
# Accuracy
# -----------------------------------------------------------------------
def run_accuracy(d_head, seqlen):
    print(f"\n{'='*60}")
    print(f"  ACCURACY TEST   d_head={d_head}  seqlen={seqlen}")
    print(f"{'='*60}")

    q, k, v = make_inputs(d_head, seqlen)
    ref = numpy_attention(q, k, v)

    for name, kernel in [("baseline", baseline_attention),
                         ("fused",    fused_attention)]:
        out = kernel(q, k, v)
        max_err = np.max(np.abs(out - ref))
        ok = np.allclose(out, ref, atol=1e-2)
        status = "PASS" if ok else "FAIL"
        print(f"  {name:12s}  max_abs_err={max_err:.6e}  [{status}]")


# -----------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------
def run_benchmark(d_head, seqlen, warmup=5, iters=20):
    print(f"\n{'='*60}")
    print(f"  BENCHMARK   d_head={d_head}  seqlen={seqlen}")
    print(f"  warmup={warmup}  iters={iters}")
    print(f"{'='*60}")

    q, k, v = make_inputs(d_head, seqlen)

    for name, kernel in [("baseline", baseline_attention),
                         ("fused",    fused_attention)]:
        bench_fn = benchmark(warmup=warmup, iters=iters)(kernel)
        bench_fn(q, k, v)
        latency = bench_fn.benchmark_result.nc_latency
        p50 = latency.get_latency_percentile(50)
        p99 = latency.get_latency_percentile(99)
        print(f"  {name:12s}  p50={p50:>8.1f} us   p99={p99:>8.1f} us")


# -----------------------------------------------------------------------
# Profile (generates traces for neuron-profile viewer)
# -----------------------------------------------------------------------
def run_profile(d_head, seqlen):
    print(f"\n{'='*60}")
    print(f"  PROFILE   d_head={d_head}  seqlen={seqlen}")
    print(f"{'='*60}")

    q, k, v = make_inputs(d_head, seqlen)

    for name, kernel in [("baseline", baseline_attention),
                         ("fused",    fused_attention)]:
        out_dir = os.path.join(PROFILE_DIR, f"{name}_attention")
        os.makedirs(out_dir, exist_ok=True)

        profile_fn = nki.profile(
            working_directory=out_dir,
            save_neff_name=f"{name}.neff",
            save_trace_name=f"{name}.ntff",
            profile_nth=2,
        )(kernel)

        profile_fn(q, k, v)
        print(f"  {name:12s}  traces saved to {out_dir}/")

    print(f"\nView traces with:")
    print(f"  neuron-profile view -n profiling/baseline_attention/baseline.neff "
          f"-s profiling/baseline_attention/baseline.ntff")
    print(f"  neuron-profile view -n profiling/fused_attention/fused.neff "
          f"-s profiling/fused_attention/fused.ntff")


# -----------------------------------------------------------------------
# Sweep multiple sequence lengths
# -----------------------------------------------------------------------
def run_sweep(d_head, warmup=5, iters=20):
    seqlens = [512, 1024, 2048, 4096]

    print(f"\n{'='*60}")
    print(f"  LATENCY SWEEP   d_head={d_head}")
    print(f"{'='*60}")
    print(f"  {'seqlen':>8s}  {'baseline (us)':>14s}  {'fused (us)':>14s}  {'speedup':>8s}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*8}")

    for seqlen in seqlens:
        q, k, v = make_inputs(d_head, seqlen)
        latencies = {}
        for name, kernel in [("baseline", baseline_attention),
                             ("fused",    fused_attention)]:
            bench_fn = benchmark(warmup=warmup, iters=iters)(kernel)
            bench_fn(q, k, v)
            latencies[name] = bench_fn.benchmark_result.nc_latency \
                                      .get_latency_percentile(50)

        speedup = latencies["baseline"] / latencies["fused"]
        print(f"  {seqlen:>8d}  {latencies['baseline']:>14.1f}  "
              f"{latencies['fused']:>14.1f}  {speedup:>7.2f}x")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark baseline vs fused attention on Trainium")
    parser.add_argument("--mode", default="all",
                        choices=["accuracy", "benchmark", "profile",
                                 "sweep", "all"])
    parser.add_argument("--d_head", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--target",
        default="auto",
        choices=["auto", "trn1", "trn2"],
        help="Optional Neuron compiler target override.",
    )
    args = parser.parse_args()
    configure_neuron_target(args.target)

    if args.mode in ("accuracy", "all"):
        run_accuracy(args.d_head, args.seqlen)

    if args.mode in ("benchmark", "all"):
        run_benchmark(args.d_head, args.seqlen, args.warmup, args.iters)

    if args.mode in ("profile", "all"):
        run_profile(args.d_head, args.seqlen)

    if args.mode in ("sweep", "all"):
        run_sweep(args.d_head, args.warmup, args.iters)
