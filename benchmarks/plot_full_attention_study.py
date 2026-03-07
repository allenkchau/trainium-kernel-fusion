"""
Plot benchmark outputs from benchmarks/full_attention_study.py.

Input:
  - profiling/reports/summary_*.json

Output:
  - PNG figures under profiling/reports/figures/<summary-stem>/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pick_latest_summary(reports_dir: Path) -> Path:
    candidates = sorted(reports_dir.glob("summary_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No summary_*.json found in {reports_dir}")
    return candidates[-1]


def group_by(rows, key):
    out = {}
    for row in rows:
        out.setdefault(row[key], []).append(row)
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

    by_dtype = group_by(speedups, "dtype")
    fig, ax = plt.subplots(figsize=(8, 5))
    for dtype, rows in by_dtype.items():
        rows = sorted(rows, key=lambda r: r["seqlen"])
        x = [r["seqlen"] for r in rows]
        y = [r["p50_speedup_x"] for r in rows]
        ax.plot(x, y, marker="o", label=dtype)
    ax.set_title("Fused vs Baseline Speedup (p50 latency)")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Speedup (x)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "speedup_p50.png", dpi=160)
    plt.close(fig)


def plot_throughput(latency_rows, out_dir: Path):
    import matplotlib.pyplot as plt

    # Separate lines by dtype+kernel
    groups = {}
    for r in latency_rows:
        groups.setdefault((r["dtype"], r["kernel"]), []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (dtype, kernel), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["seqlen"])
        x = [r["seqlen"] for r in rows]
        y = [r["tok_per_s_p50"] / 1e6 for r in rows]
        ax.plot(x, y, marker="o", label=f"{dtype}-{kernel}")
    ax.set_title("Token Throughput vs Sequence Length")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Tokens/s (millions, p50)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "throughput_tokens_per_s.png", dpi=160)
    plt.close(fig)


def plot_precision(precision_rows, out_dir: Path):
    import matplotlib.pyplot as plt

    groups = {}
    for r in precision_rows:
        groups.setdefault((r["dtype"], r["kernel"]), []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (dtype, kernel), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["seqlen"])
        x = [r["seqlen"] for r in rows]
        y = [r["max_abs_err"] for r in rows]
        ax.plot(x, y, marker="o", label=f"{dtype}-{kernel}")
    ax.set_title("Max Absolute Error vs Sequence Length")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Max abs error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "precision_max_abs_err.png", dpi=160)
    plt.close(fig)


def plot_dma_proxy(profile_rows, out_dir: Path):
    import matplotlib.pyplot as plt

    groups = {}
    for r in profile_rows:
        groups.setdefault((r["dtype"], r["kernel"]), []).append(r)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (dtype, kernel), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["seqlen"])
        x = [r["seqlen"] for r in rows]
        y = [r["dma_queues_sum"] for r in rows]
        ax.plot(x, y, marker="o", label=f"{dtype}-{kernel}")
    ax.set_title("DMA Queue Count Proxy from neuron-profile")
    ax.set_xlabel("Profiled Sequence Length")
    ax.set_ylabel("DMA queue count (sum)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "dma_queue_proxy.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot full attention study outputs")
    parser.add_argument(
        "--summary-json",
        default="latest",
        help="Path to summary_*.json, or 'latest' (default).",
    )
    parser.add_argument(
        "--reports-dir",
        default="profiling/reports",
        help="Reports directory for summary discovery (default: profiling/reports).",
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

    print(f"Generated figures in: {out_dir}")


if __name__ == "__main__":
    main()
