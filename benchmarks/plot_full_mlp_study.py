"""
Plot benchmark outputs from benchmarks/full_mlp_study.py.

Input:
  - profiling/reports/mlp_summary_*.json

Output:
  - PNG figures under profiling/reports/figures/<summary-stem>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pick_latest_summary(reports_dir: Path) -> Path:
    candidates = sorted(
        reports_dir.glob("mlp_summary_*.json"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(f"No mlp_summary_*.json found in {reports_dir}")
    return candidates[-1]


def group_by(rows, *keys):
    out = {}
    for row in rows:
        k = tuple(row[key] for key in keys)
        out.setdefault(k, []).append(row)
    return out


def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: "
            "python -m pip install matplotlib"
        ) from exc


def plot_speedup(speedups, out_dir: Path):
    import matplotlib.pyplot as plt

    by_group = group_by(speedups, "dtype", "variant")
    fig, ax = plt.subplots(figsize=(10, 6))
    for (dtype, variant), rows in sorted(by_group.items()):
        rows = sorted(rows, key=lambda r: r["M"] * r["N"])
        x = [r["config"] for r in rows]
        y = [r["p50_speedup_x"] for r in rows]
        ax.plot(x, y, marker="o", label=f"{dtype}-{variant}")
    ax.set_title("Fused vs Baseline Speedup (p50 latency)")
    ax.set_xlabel("Configuration (MxKxN)")
    ax.set_ylabel("Speedup (x)")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "mlp_speedup_p50.png", dpi=160)
    plt.close(fig)


def plot_throughput(latency_rows, out_dir: Path):
    import matplotlib.pyplot as plt

    groups = group_by(latency_rows, "dtype", "kernel")
    fig, ax = plt.subplots(figsize=(10, 6))
    for (dtype, kernel), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["M"] * r["N"])
        x = [r["config"] for r in rows]
        y = [r["est_tflops_p50"] for r in rows]
        ax.plot(x, y, marker="o", label=f"{dtype}-{kernel}")
    ax.set_title("Estimated TFLOPS vs Configuration")
    ax.set_xlabel("Configuration (MxKxN)")
    ax.set_ylabel("TFLOPS (p50)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "mlp_throughput_tflops.png", dpi=160)
    plt.close(fig)


def plot_precision(precision_rows, out_dir: Path):
    import matplotlib.pyplot as plt

    groups = group_by(precision_rows, "dtype", "kernel")
    fig, ax = plt.subplots(figsize=(10, 6))
    for (dtype, kernel), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["M"] * r["N"])
        x = [r["config"] for r in rows]
        y = [r["max_abs_err"] for r in rows]
        ax.plot(x, y, marker="o", label=f"{dtype}-{kernel}")
    ax.set_title("Max Absolute Error vs Configuration")
    ax.set_xlabel("Configuration (MxKxN)")
    ax.set_ylabel("Max abs error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "mlp_precision_max_abs_err.png", dpi=160)
    plt.close(fig)


def plot_dma_proxy(profile_rows, out_dir: Path):
    import matplotlib.pyplot as plt

    if not profile_rows:
        return

    groups = group_by(profile_rows, "dtype", "kernel")
    fig, ax = plt.subplots(figsize=(10, 6))
    for (dtype, kernel), rows in sorted(groups.items()):
        x = [r["config"] for r in rows]
        y = [r["dma_queues_sum"] for r in rows]
        ax.bar(
            [f"{dtype}-{kernel}\n{xi}" for xi in x],
            y, label=f"{dtype}-{kernel}",
        )
    ax.set_title("DMA Queue Count Proxy from neuron-profile")
    ax.set_ylabel("DMA queue count (sum)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mlp_dma_queue_proxy.png", dpi=160)
    plt.close(fig)


def plot_hbm_reduction(speedups, out_dir: Path):
    import matplotlib.pyplot as plt

    by_group = group_by(speedups, "dtype", "variant")
    fig, ax = plt.subplots(figsize=(10, 6))
    for (dtype, variant), rows in sorted(by_group.items()):
        rows = sorted(rows, key=lambda r: r["M"] * r["N"])
        x = [r["config"] for r in rows]
        y = [r["hbm_model_reduction_x"] for r in rows]
        ax.plot(x, y, marker="s", label=f"{dtype}-{variant}")
    ax.set_title("Modeled HBM Traffic Reduction")
    ax.set_xlabel("Configuration (MxKxN)")
    ax.set_ylabel("HBM reduction (x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "mlp_hbm_traffic_reduction.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot full MLP study outputs")
    parser.add_argument(
        "--summary-json", default="latest",
        help="Path to mlp_summary_*.json, or 'latest' (default).",
    )
    parser.add_argument(
        "--reports-dir", default="profiling/reports",
        help="Reports directory for summary discovery.",
    )
    args = parser.parse_args()

    reports_dir = Path(args.reports_dir).resolve()
    summary_path = (
        pick_latest_summary(reports_dir)
        if args.summary_json == "latest"
        else Path(args.summary_json).resolve()
    )

    ensure_matplotlib()

    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = reports_dir / "figures" / summary_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_speedup(data["speedups"], out_dir)
    plot_throughput(data["latency_throughput"], out_dir)
    plot_precision(data["precision_tradeoffs"], out_dir)
    plot_dma_proxy(data["profile_summaries"], out_dir)
    plot_hbm_reduction(data["speedups"], out_dir)

    print(f"Generated figures in: {out_dir}")


if __name__ == "__main__":
    main()
