#!/usr/bin/env python3
"""Plot reduce bench results from the CSV emitted by BenchBase.

Usage:
    python3 kernels/reduce/scripts/plot_one_line.py
    # defaults (csv in/out under kernels/reduce/figures/) resolve relative to
    # the script, so it works from any CWD. Override only if needed:
    python3 kernels/reduce/scripts/plot_one_line.py --metric bandwidth

CSV columns (written by common/bench_framework.h):
    kernel,size,grid_x,grid_y,grid_z,block_x,block_y,block_z,correct,time_ms

`size` is BenchInfo(), e.g. "1x4096" -> n=1, m=4096. For reduce each element of
the [n, m] tensor is read once, so:
    bandwidth_gbs = n * m * 4 bytes / (time_ms / 1000) / 1e9
(sgemv etc. would use a different byte count -- copy and adjust if needed.)
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# defaults resolve relative to this script so it runs from any CWD:
#   scripts/plot_one_line.py -> ../figures/...
_HERE = Path(__file__).resolve().parent
_FIGURES = _HERE.parent / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(_FIGURES / "reduce_one_line.csv"))
    ap.add_argument("--out", default=str(_FIGURES / "reduce_one_line.png"))
    ap.add_argument("--metric", choices=["time", "bandwidth"], default="time")
    ap.add_argument("--x", default="m", help="x-axis: n or m")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        print("empty csv", file=sys.stderr)
        sys.exit(1)

    # parse "SxK" -> n, m
    split = df["size"].str.split("x", expand=True)
    df["n"] = split[0].astype(int)
    df["m"] = split[1].astype(int)
    # rows without a measured time (launch failure) can't be plotted
    df = df.dropna(subset=["time_ms"])

    if args.metric == "bandwidth":
        df["y"] = df["n"] * df["m"] * 4 / (df["time_ms"] / 1000.0) / 1e9
        ylabel = "Bandwidth (GB/s)"
    else:
        df["y"] = df["time_ms"]
        ylabel = "Time (ms)"

    x = args.x
    fig, ax = plt.subplots(figsize=(9, 6))
    for kernel, g in df.groupby("kernel"):
        g = g.sort_values(x)
        ax.plot(g[x], g["y"], marker="o", label=kernel)

    # annotate each measured x value once: place the label above the highest
    # point at that x so overlapping curves don't stack 5 labels on one spot.
    label_col = "m" if x == "m" else "n"
    for xv, sub in df.groupby(x):
        ymax = sub["y"].max()
        ax.annotate(f"{label_col}={int(xv)}",
                    xy=(xv, ymax),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=7, ha="center")

    ax.set_xscale("log")
    if args.metric == "time":
        ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(f"reduce bench: {ylabel} vs {x}")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
