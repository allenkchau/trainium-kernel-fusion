"""
Hardware design insight study: guiding next-gen Trainium chip design.

Experiments:
  1. SBUF Breakeven — fine-grained sweep to find the intermediate tensor
     size where fusion benefit vanishes. Maps to minimum SBUF capacity.
  2. Roofline Analysis — achieved GFLOP/s vs arithmetic intensity to
     locate the memory-compute ridge point on Trainium2.
  3. BF16 vs FP32 — how halving memory traffic shifts the fusion crossover.

Outputs JSON + publication-quality plots under profiling/reports/.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from neuronxcc.nki import benchmark

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.baseline.mlp_gemm import gemm_kernel, gelu_kernel
from kernels.fused.mlp import fused_mlp_pwl


# ── Trainium2 specs (from public docs) ──────────────────────────────
TRN2_PEAK_TFLOPS_FP32 = 47.5     # per NeuronCore, FP32
TRN2_PEAK_TFLOPS_BF16 = 190.0    # per NeuronCore, BF16
TRN2_HBM_BW_GB_S = 820.0         # per chip, GB/s (HBM3)
TRN2_SBUF_KB = 24 * 1024         # 24 MiB SBUF per NeuronCore


def configure_target():
    flag = "--target=trn2"
    existing = os.environ.get("NEURON_CC_FLAGS", "").strip()
    if flag not in existing.split():
        os.environ["NEURON_CC_FLAGS"] = f"{existing} {flag}".strip() if existing else flag


def make_inputs(M, K, N, dtype=np.float32):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(M, K)).astype(dtype)
    W = rng.normal(size=(K, N)).astype(dtype)
    bias = rng.normal(size=(N,)).astype(dtype)
    return X, W, bias


def bench_baseline(X, W, bias, warmup=3, iters=10):
    lhsT = X.T.copy()
    bias_2d = bias.reshape(1, -1)
    gemm_b = benchmark(warmup=warmup, iters=iters)(gemm_kernel)
    gemm_b(lhsT, W, bias_2d)
    gp50 = gemm_b.benchmark_result.nc_latency.get_latency_percentile(50)
    Y = gemm_kernel(lhsT, W, bias_2d)
    gelu_b = benchmark(warmup=warmup, iters=iters)(gelu_kernel)
    gelu_b(Y)
    ep50 = gelu_b.benchmark_result.nc_latency.get_latency_percentile(50)
    return float(gp50 + ep50)


def bench_fused(X, W, bias, warmup=3, iters=10):
    lhsT = X.T.copy()
    bias_2d = bias.reshape(1, -1)
    b = benchmark(warmup=warmup, iters=iters)(fused_mlp_pwl)
    b(lhsT, W, bias_2d)
    return float(b.benchmark_result.nc_latency.get_latency_percentile(50))


def mlp_flops(M, K, N):
    return float(2 * M * K * N)


def mlp_hbm_bytes(M, K, N, dtype_bytes):
    """Total HBM traffic for baseline (reads + writes including intermediate)."""
    input_bytes = (M * K + K * N + N) * dtype_bytes
    intermediate_rw = 2 * M * N * dtype_bytes  # write Y then read Y
    output_bytes = M * N * dtype_bytes
    return float(input_bytes + intermediate_rw + output_bytes)


def mlp_hbm_bytes_fused(M, K, N, dtype_bytes):
    """Total HBM traffic for fused (no intermediate)."""
    input_bytes = (M * K + K * N + N) * dtype_bytes
    output_bytes = M * N * dtype_bytes
    return float(input_bytes + output_bytes)


def intermediate_size_kb(M, N, dtype_bytes):
    """Size of the intermediate Y tensor in KB."""
    return M * N * dtype_bytes / 1024.0


# =====================================================================
# Experiment 1: SBUF Breakeven — sweep M with fixed K, N
# =====================================================================
def run_sbuf_breakeven(dtype=np.float32):
    dtype_bytes = np.dtype(dtype).itemsize
    dtype_name = "fp32" if dtype == np.float32 else "bf16"
    K, N = 512, 2048
    # M values: 128 to 4096 stepping by 128 (tile-aligned)
    M_values = list(range(128, 4096 + 1, 128))

    print(f"\n{'='*70}")
    print(f"  SBUF BREAKEVEN STUDY  ({dtype_name})  K={K} N={N}")
    print(f"{'='*70}")
    print(f"  {'M':>6} {'intermed_KB':>12} {'baseline_us':>12} {'fused_us':>10} {'speedup':>8}")

    rows = []
    for M in M_values:
        X, W, bias = make_inputs(M, K, N, dtype)
        bl = bench_baseline(X, W, bias)
        fu = bench_fused(X, W, bias)
        speedup = bl / fu
        imm_kb = intermediate_size_kb(M, N, dtype_bytes)

        flops = mlp_flops(M, K, N)
        hbm_bl = mlp_hbm_bytes(M, K, N, dtype_bytes)
        hbm_fu = mlp_hbm_bytes_fused(M, K, N, dtype_bytes)
        arith_intensity_bl = flops / hbm_bl
        arith_intensity_fu = flops / hbm_fu
        gflops_bl = flops / (bl * 1e-6) / 1e9
        gflops_fu = flops / (fu * 1e-6) / 1e9

        row = {
            "dtype": dtype_name, "M": M, "K": K, "N": N,
            "intermediate_KB": imm_kb,
            "baseline_us": bl, "fused_us": fu, "speedup": speedup,
            "flops": flops,
            "hbm_bytes_baseline": hbm_bl, "hbm_bytes_fused": hbm_fu,
            "arith_intensity_baseline": arith_intensity_bl,
            "arith_intensity_fused": arith_intensity_fu,
            "gflops_baseline": gflops_bl, "gflops_fused": gflops_fu,
        }
        rows.append(row)
        print(f"  {M:>6} {imm_kb:>12.0f} {bl:>12.1f} {fu:>10.1f} {speedup:>7.2f}x")

    return rows


# =====================================================================
# Experiment 2: Roofline — vary problem size to sweep arith intensity
# =====================================================================
def run_roofline_sweep(dtype=np.float32):
    dtype_bytes = np.dtype(dtype).itemsize
    dtype_name = "fp32" if dtype == np.float32 else "bf16"

    # Configs that span a wide range of arithmetic intensities
    # (small K = low intensity / memory-bound, large K = high intensity / compute-bound)
    configs = [
        # M,    K,    N       -- arith intensity ≈ 2*K / (overhead per element)
        (128,  128,  512),
        (256,  128,  512),
        (128,  256,  512),
        (256,  256,  1024),
        (256,  512,  1024),
        (512,  512,  2048),
        (512,  1024, 2048),
        (1024, 1024, 2048),
        (1024, 2048, 4096),
        (2048, 2048, 4096),
        (2048, 4096, 4096),
    ]

    print(f"\n{'='*70}")
    print(f"  ROOFLINE SWEEP  ({dtype_name})")
    print(f"{'='*70}")

    rows = []
    for M, K, N in configs:
        X, W, bias = make_inputs(M, K, N, dtype)
        bl = bench_baseline(X, W, bias)
        fu = bench_fused(X, W, bias)
        flops = mlp_flops(M, K, N)
        hbm_bl = mlp_hbm_bytes(M, K, N, dtype_bytes)
        hbm_fu = mlp_hbm_bytes_fused(M, K, N, dtype_bytes)
        rows.append({
            "dtype": dtype_name, "M": M, "K": K, "N": N,
            "config": f"{M}x{K}x{N}",
            "arith_intensity_baseline": flops / hbm_bl,
            "arith_intensity_fused": flops / hbm_fu,
            "gflops_baseline": flops / (bl * 1e-6) / 1e9,
            "gflops_fused": flops / (fu * 1e-6) / 1e9,
            "baseline_us": bl, "fused_us": fu,
            "speedup": bl / fu,
        })
        print(f"  {M:>5}x{K:>4}x{N:>4}  AI_bl={flops/hbm_bl:>6.1f}  "
              f"bl={bl:>8.1f}us  fu={fu:>8.1f}us  "
              f"{bl/fu:>5.2f}x  {flops/(fu*1e-6)/1e9:>7.1f} GFLOP/s")

    return rows


# =====================================================================
# Plotting
# =====================================================================
def generate_plots(sbuf_fp32, sbuf_bf16, roofline_fp32, roofline_bf16, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Plot 1: Fusion Speedup vs Intermediate Size ──────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, rows, color in [("FP32", sbuf_fp32, "#2563eb"),
                                ("BF16", sbuf_bf16, "#dc2626")]:
        if not rows:
            continue
        x = [r["intermediate_KB"] for r in rows]
        y = [r["speedup"] for r in rows]
        ax.plot(x, y, marker="o", markersize=4, label=label, color=color)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.6, label="Breakeven")
    ax.axvline(x=TRN2_SBUF_KB, color="green", linestyle=":", alpha=0.6,
               label=f"SBUF capacity ({TRN2_SBUF_KB/1024:.0f} MiB)")
    ax.set_xlabel("Intermediate Tensor Size (KB)", fontsize=12)
    ax.set_ylabel("Fused / Baseline Speedup (x)", fontsize=12)
    ax.set_title("Kernel Fusion Benefit vs On-Chip Buffer Demand", fontsize=14)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "sbuf_breakeven.png", dpi=180)
    plt.close(fig)
    print(f"  Saved sbuf_breakeven.png")

    # ── Plot 2: Roofline ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # Theoretical roofline lines
    ai_range = np.logspace(-1, 3, 200)
    for peak_tflops, bw, dtype_label, ls in [
        (TRN2_PEAK_TFLOPS_FP32, TRN2_HBM_BW_GB_S, "FP32 roof", "-"),
        (TRN2_PEAK_TFLOPS_BF16, TRN2_HBM_BW_GB_S, "BF16 roof", "--"),
    ]:
        roof = np.minimum(peak_tflops * 1e3, ai_range * bw)  # GFLOP/s
        ax.plot(ai_range, roof, color="gray", linestyle=ls, alpha=0.5, label=dtype_label)

    for label, rows, marker, color in [
        ("FP32 baseline", roofline_fp32, "o", "#93c5fd"),
        ("FP32 fused",    roofline_fp32, "s", "#2563eb"),
        ("BF16 baseline", roofline_bf16, "o", "#fca5a5"),
        ("BF16 fused",    roofline_bf16, "s", "#dc2626"),
    ]:
        if not rows:
            continue
        key = "baseline" if "baseline" in label else "fused"
        x = [r[f"arith_intensity_{key}"] for r in rows]
        y = [r[f"gflops_{key}"] for r in rows]
        ax.scatter(x, y, marker=marker, s=60, label=label, color=color, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
    ax.set_ylabel("Achieved Throughput (GFLOP/s)", fontsize=12)
    ax.set_title("Roofline Model — Trainium2 NeuronCore", fontsize=14)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "roofline.png", dpi=180)
    plt.close(fig)
    print(f"  Saved roofline.png")

    # ── Plot 3: BF16 vs FP32 speedup comparison ─────────────────
    if sbuf_bf16:
        fig, ax = plt.subplots(figsize=(10, 6))
        for label, rows, color in [("FP32", sbuf_fp32, "#2563eb"),
                                    ("BF16", sbuf_bf16, "#dc2626")]:
            x = [r["M"] for r in rows]
            y = [r["speedup"] for r in rows]
            ax.plot(x, y, marker="o", markersize=4, label=label, color=color)

        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.6)
        ax.set_xlabel("M (batch/token dimension)", fontsize=12)
        ax.set_ylabel("Fusion Speedup (x)", fontsize=12)
        ax.set_title("FP32 vs BF16: How Precision Shifts the Fusion Crossover\n"
                      "(K=512, N=2048)", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "bf16_vs_fp32_crossover.png", dpi=180)
        plt.close(fig)
        print(f"  Saved bf16_vs_fp32_crossover.png")

    # ── Plot 4: SBUF sizing recommendation ───────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    # Real-world MLP hidden dims (from known models)
    model_layers = {
        "GPT-2 (768)":   768 * 3072,     # h=768, 4h=3072
        "LLaMA-7B (4096)": 4096 * 11008,
        "LLaMA-13B (5120)": 5120 * 13824,
        "LLaMA-70B (8192)": 8192 * 28672,
    }

    for label, rows, color in [("FP32", sbuf_fp32, "#2563eb"),
                                ("BF16", sbuf_bf16, "#dc2626")]:
        if not rows:
            continue
        x = [r["intermediate_KB"] / 1024 for r in rows]  # MiB
        y = [r["speedup"] for r in rows]
        ax.plot(x, y, marker="o", markersize=4, label=label, color=color)

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    # Mark SBUF size
    ax.axvline(x=TRN2_SBUF_KB / 1024, color="green", linestyle=":",
               linewidth=2, label=f"Current SBUF ({TRN2_SBUF_KB/1024:.0f} MiB)")

    # Mark model intermediate sizes for a single tile-row (M=128 tokens)
    for name, hidden_size in model_layers.items():
        intermediate_mib = 128 * (hidden_size // 128) * 4 / (1024**2)  # FP32
        if intermediate_mib < 200:
            ax.axvline(x=intermediate_mib, color="orange", linestyle="--",
                       alpha=0.4, linewidth=1)
            ax.text(intermediate_mib * 1.05, 1.6, name, rotation=90,
                    fontsize=8, color="orange", alpha=0.7)

    ax.set_xlabel("Intermediate Tensor Size (MiB)", fontsize=12)
    ax.set_ylabel("Fusion Speedup (x)", fontsize=12)
    ax.set_title("SBUF Sizing for Next-Gen Trainium:\nFusion Benefit vs On-Chip Memory Demand",
                 fontsize=13)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "sbuf_sizing_recommendation.png", dpi=180)
    plt.close(fig)
    print(f"  Saved sbuf_sizing_recommendation.png")


# =====================================================================
# Main
# =====================================================================
def main():
    configure_target()

    reports_dir = PROJECT_ROOT / "profiling" / "reports"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = reports_dir / "figures" / f"hw_design_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check BF16 availability
    bf16_dtype = None
    try:
        import ml_dtypes
        bf16_dtype = ml_dtypes.bfloat16
    except Exception:
        print("[warn] ml_dtypes not available, skipping BF16 experiments")

    # Run experiments
    print("\n" + "=" * 70)
    print("  HARDWARE DESIGN INSIGHT STUDY — Trainium2")
    print("=" * 70)

    sbuf_fp32 = run_sbuf_breakeven(np.float32)
    sbuf_bf16 = run_sbuf_breakeven(bf16_dtype) if bf16_dtype else []

    roofline_fp32 = run_roofline_sweep(np.float32)
    roofline_bf16 = run_roofline_sweep(bf16_dtype) if bf16_dtype else []

    # Save results
    summary = {
        "generated_at_utc": ts,
        "hardware": {
            "target": "trn2",
            "peak_tflops_fp32": TRN2_PEAK_TFLOPS_FP32,
            "peak_tflops_bf16": TRN2_PEAK_TFLOPS_BF16,
            "hbm_bw_gb_s": TRN2_HBM_BW_GB_S,
            "sbuf_kb": TRN2_SBUF_KB,
        },
        "sbuf_breakeven_fp32": sbuf_fp32,
        "sbuf_breakeven_bf16": sbuf_bf16,
        "roofline_fp32": roofline_fp32,
        "roofline_bf16": roofline_bf16,
    }

    json_path = reports_dir / f"hw_design_summary_{ts}.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Analysis: find crossover point
    print("\n" + "=" * 70)
    print("  ANALYSIS & RECOMMENDATIONS")
    print("=" * 70)

    for label, rows in [("FP32", sbuf_fp32), ("BF16", sbuf_bf16)]:
        if not rows:
            continue
        # Find largest M where speedup > 1.05 (meaningful benefit)
        beneficial = [r for r in rows if r["speedup"] > 1.05]
        if beneficial:
            last = beneficial[-1]
            print(f"\n  [{label}] Fusion beneficial (>1.05x) up to:")
            print(f"    M={last['M']}, intermediate={last['intermediate_KB']:.0f} KB "
                  f"({last['intermediate_KB']/1024:.1f} MiB)")
            print(f"    Speedup at crossover: {last['speedup']:.2f}x")
            peak = max(rows, key=lambda r: r["speedup"])
            print(f"    Peak speedup: {peak['speedup']:.2f}x at M={peak['M']} "
                  f"({peak['intermediate_KB']:.0f} KB)")
        else:
            print(f"\n  [{label}] No meaningful fusion benefit observed")

    # Roofline ridge point
    for label, rows in [("FP32", roofline_fp32), ("BF16", roofline_bf16)]:
        if not rows:
            continue
        best_fused = max(rows, key=lambda r: r["gflops_fused"])
        print(f"\n  [{label}] Peak achieved throughput (fused): "
              f"{best_fused['gflops_fused']:.1f} GFLOP/s at {best_fused['config']}")
        peak = TRN2_PEAK_TFLOPS_FP32 * 1e3 if label == "FP32" else TRN2_PEAK_TFLOPS_BF16 * 1e3
        print(f"    Utilization: {best_fused['gflops_fused']/peak*100:.1f}% of peak")

    # Generate plots
    print(f"\nGenerating plots in {out_dir}/")
    generate_plots(sbuf_fp32, sbuf_bf16, roofline_fp32, roofline_bf16, out_dir)
    print(f"\nDone! All outputs in {out_dir}")


if __name__ == "__main__":
    main()
