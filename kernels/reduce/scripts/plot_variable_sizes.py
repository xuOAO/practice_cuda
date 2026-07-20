#!/usr/bin/env python3
"""Plot test_variable_sizes() results.

test_variable_sizes() fixes K (=m=8192) and sweeps S (=n) over
{1,2,4,...,16384}, running reduce_warp and reduce_block<512> once per S.
Each row in the CSV is one (kernel, S) config.

This script plots time (ms) vs n (=S) on a log axis, with each measured
S annotated next to its marker.

Usage:
    python3 kernels/reduce/scripts/plot_variable_sizes.py
    # defaults (csv in/out under kernels/reduce/figures/) resolve relative to
    # the script, so it works from any CWD. Override only if needed:
    python3 kernels/reduce/scripts/plot_variable_sizes.py \
        --csv <path>.csv --out <path>.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# defaults resolve relative to this script so it runs from any CWD:
#   scripts/plot_variable_sizes.py -> ../figures/...
_HERE = Path(__file__).resolve().parent
_FIGURES = _HERE.parent / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(_FIGURES / "reduce_variable_sizes_results.csv"))
    ap.add_argument("--out", default=str(_FIGURES / "reduce_variable_sizes.png"))
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
    if df.empty:
        print("no rows with time_ms after dropna", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(9, 6))

    for kernel, g in df.groupby("kernel"):
        g = g.sort_values("n")
        ax.plot(g["n"], g["time_ms"], marker="o", label=kernel)

    # annotate each measured S once: place the label above the highest point at
    # that S so the two curves don't stack two labels on the same spot.
    for s, sub in df.groupby("n"):
        ymax = sub["time_ms"].max()
        ax.annotate(f"S={int(s)}",
                    xy=(s, ymax),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=7, ha="center")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (=S),  m fixed at 8192")
    ax.set_ylabel("Time (ms)")
    ax.set_title("reduce variable-size sweep")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
