"""
visualize_profile.py
====================
Performance profile visualizer for GE-CSG parser benchmarks.

Reads outputs/performance_profile/profile_data.csv and produces a set of
analysis plots saved to outputs/performance_profile/.

Plots generated
---------------
  01_elapsed_vs_codons.png      scatter + mean trend per grammar
  02_boxplot_per_length.png     box plot of elapsed_ms for each n_codons
  03_mean_elapsed_per_grammar.png   line chart comparing grammar types
  04_throughput.png             codons/ms vs n_codons (efficiency curve)
  05_elapsed_distribution.png  histogram of all elapsed_ms values
  06_scaling_loglog.png         log-log scatter to identify complexity class

Run:
    python visualize_profile.py
    python visualize_profile.py --csv path/to/other.csv
"""

import sys, os, io, argparse, math
from collections import defaultdict
from typing import Dict, List

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from gecsg.profiler.profiler import load_profile_data

# ── Config ─────────────────────────────────────────────────────────────────

DEFAULT_CSV = os.path.join(
    _ROOT,
    "outputs", "performance_profile", "profile_data.csv",
)
OUT_DIR = os.path.join(
    _ROOT,
    "outputs", "performance_profile",
)

GRAMMAR_COLORS = {
    "earley_deterministic":      "#2E86C1",
    "earley_stochastic_uniform": "#E67E22",
    "earley_stochastic_human":   "#27AE60",
}
DEFAULT_COLOR = "#7F8C8D"

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "axes.grid":        True,
    "grid.color":       "#DEE2E6",
    "grid.linewidth":   0.6,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        11,
}

# ── Helpers ─────────────────────────────────────────────────────────────────

def _color(grammar: str) -> str:
    return GRAMMAR_COLORS.get(grammar, DEFAULT_COLOR)


def _group_by(rows: List[Dict], key: str) -> Dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return dict(groups)


def _mean_by_codons(rows: List[Dict]) -> Dict[int, float]:
    """Aggregate: n_codons -> mean elapsed_ms."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["n_codons"]].append(r["elapsed_ms"])
    return {k: sum(v)/len(v) for k, v in sorted(buckets.items())}


def save(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  {name}")


# ── Plot functions ───────────────────────────────────────────────────────────

def plot_scatter_with_trend(rows: List[Dict]) -> None:
    """01: scatter of every parse time, with per-grammar mean trend line."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        by_grammar = _group_by(rows, "grammar")

        for grammar, grp in sorted(by_grammar.items()):
            c = _color(grammar)
            xs = [r["n_codons"]   for r in grp]
            ys = [r["elapsed_ms"] for r in grp]
            ax.scatter(xs, ys, alpha=0.25, s=14, color=c)

            means = _mean_by_codons(grp)
            mx, my = zip(*sorted(means.items()))
            ax.plot(mx, my, color=c, linewidth=2.0,
                    label=grammar.replace("earley_", ""))

        ax.set_xlabel("Sequence length (n_codons)")
        ax.set_ylabel("Parse time (ms)")
        ax.set_title("Parse time vs sequence length\n"
                     "(dots = individual parses, lines = mean)")
        ax.legend(fontsize=9)
        fig.tight_layout()
    save(fig, "01_elapsed_vs_codons.png")


def plot_boxplot_per_length(rows: List[Dict]) -> None:
    """02: box plot of elapsed_ms at each distinct n_codons (all grammars combined)."""
    with plt.rc_context(STYLE):
        by_len = _group_by(rows, "n_codons")
        lengths = sorted(by_len.keys())
        data    = [[r["elapsed_ms"] for r in by_len[n]] for n in lengths]

        # Thin out labels if many lengths
        step = max(1, len(lengths) // 20)
        labels = [str(n) if i % step == 0 else "" for i, n in enumerate(lengths)]

        fig, ax = plt.subplots(figsize=(max(10, len(lengths) * 0.35), 5))
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="#C0392B", linewidth=1.5))
        for patch in bp["boxes"]:
            patch.set_facecolor("#AED6F1")
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(lengths) + 1))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Sequence length (n_codons)")
        ax.set_ylabel("Parse time (ms)")
        ax.set_title("Parse time distribution per sequence length (all grammars)")
        fig.tight_layout()
    save(fig, "02_boxplot_per_length.png")


def plot_mean_per_grammar(rows: List[Dict]) -> None:
    """03: mean elapsed_ms per n_codons, one line per grammar type."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        by_grammar = _group_by(rows, "grammar")

        for grammar, grp in sorted(by_grammar.items()):
            means = _mean_by_codons(grp)
            xs, ys = zip(*sorted(means.items()))
            c = _color(grammar)
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8, color=c,
                    label=grammar.replace("earley_", ""))

        ax.set_xlabel("Sequence length (n_codons)")
        ax.set_ylabel("Mean parse time (ms)")
        ax.set_title("Mean parse time per grammar type")
        ax.legend(fontsize=9)
        fig.tight_layout()
    save(fig, "03_mean_elapsed_per_grammar.png")


def plot_throughput(rows: List[Dict]) -> None:
    """04: throughput = n_codons / elapsed_ms (efficiency curve)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        by_grammar = _group_by(rows, "grammar")

        for grammar, grp in sorted(by_grammar.items()):
            c = _color(grammar)
            means = _mean_by_codons(grp)
            xs = sorted(means.keys())
            ys = [xs[i] / means[xs[i]] for i in range(len(xs))]   # codons/ms
            ax.plot(xs, ys, marker="o", markersize=4, linewidth=1.8, color=c,
                    label=grammar.replace("earley_", ""))

        ax.set_xlabel("Sequence length (n_codons)")
        ax.set_ylabel("Throughput (codons / ms)")
        ax.set_title("Parser throughput vs sequence length")
        ax.legend(fontsize=9)
        fig.tight_layout()
    save(fig, "04_throughput.png")


def plot_elapsed_distribution(rows: List[Dict]) -> None:
    """05: histogram of all elapsed_ms values, per grammar."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            1, len(set(r["grammar"] for r in rows)),
            figsize=(14, 4), sharey=True,
        )
        if not hasattr(axes, "__iter__"):
            axes = [axes]

        by_grammar = _group_by(rows, "grammar")
        for ax, (grammar, grp) in zip(axes, sorted(by_grammar.items())):
            vals = [r["elapsed_ms"] for r in grp]
            c    = _color(grammar)
            ax.hist(vals, bins=30, color=c, alpha=0.8, edgecolor="white")
            ax.set_title(grammar.replace("earley_", ""), fontsize=9)
            ax.set_xlabel("Parse time (ms)")
            if ax is axes[0]:
                ax.set_ylabel("Count")
            m = sum(vals) / len(vals)
            ax.axvline(m, color="#C0392B", linewidth=1.5, linestyle="--",
                       label=f"mean={m:.2f}")
            ax.legend(fontsize=8)

        fig.suptitle("Parse time distribution per grammar type", fontsize=12)
        fig.tight_layout()
    save(fig, "05_elapsed_distribution.png")


def plot_loglog_scaling(rows: List[Dict]) -> None:
    """06: log-log plot to identify algorithmic complexity class."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        by_grammar = _group_by(rows, "grammar")

        for grammar, grp in sorted(by_grammar.items()):
            c     = _color(grammar)
            means = _mean_by_codons(grp)
            xs    = sorted(means.keys())
            ys    = [means[x] for x in xs]

            lx = [math.log(x) for x in xs if means[x] > 0]
            ly = [math.log(means[x]) for x in xs if means[x] > 0]

            ax.scatter(lx, ly, color=c, alpha=0.7, s=20)
            # Linear fit on log-log: slope = empirical complexity exponent
            if len(lx) > 2:
                coeffs = np.polyfit(lx, ly, 1)
                fit_y  = np.polyval(coeffs, lx)
                ax.plot(lx, fit_y, color=c, linewidth=1.8,
                        label=f"{grammar.replace('earley_','')}  "
                              f"(slope={coeffs[0]:.2f})")

        ax.set_xlabel("log(n_codons)")
        ax.set_ylabel("log(mean elapsed_ms)")
        ax.set_title("Log-log scaling plot\n"
                     "(slope = empirical complexity exponent)")
        ax.legend(fontsize=9)
        # Reference lines
        xs_ref = ax.get_xlim()
        for exp, style, label in [(1, ":", "O(n)"), (2, "--", "O(n²)"),
                                   (3, "-.", "O(n³)")]:
            ref_y = [exp * x for x in xs_ref]
            ax.plot(xs_ref, ref_y, color="#BDC3C7", linestyle=style,
                    linewidth=0.9, label=label, zorder=0)
        ax.legend(fontsize=8)
        fig.tight_layout()
    save(fig, "06_scaling_loglog.png")


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary_table(rows: List[Dict]) -> None:
    """Print a per-grammar summary table to stdout."""
    by_grammar = _group_by(rows, "grammar")
    print("\n" + "=" * 72)
    print(f"{'Grammar':<32} {'N':>5} {'min ms':>8} {'mean ms':>8} "
          f"{'max ms':>8} {'accepted':>9}")
    print("-" * 72)
    for grammar, grp in sorted(by_grammar.items()):
        el  = [r["elapsed_ms"] for r in grp]
        acc = sum(1 for r in grp if r["accepted"])
        print(f"{grammar:<32} {len(grp):>5} {min(el):>8.3f} "
              f"{sum(el)/len(el):>8.3f} {max(el):>8.3f} {acc:>9}")
    print("=" * 72)
    print(f"Total rows: {len(rows)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="GE-CSG profile visualizer")
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help=f"CSV file to read (default: {DEFAULT_CSV})")
    args = ap.parse_args()

    rows = load_profile_data(args.csv)
    if not rows:
        print(f"No data found in {args.csv}")
        print("Run  python run_profiler.py  first to collect data.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loaded {len(rows)} rows from {args.csv}")
    print_summary_table(rows)

    print(f"\nGenerating plots -> {OUT_DIR}")
    plot_scatter_with_trend(rows)
    plot_boxplot_per_length(rows)
    plot_mean_per_grammar(rows)
    plot_throughput(rows)
    plot_elapsed_distribution(rows)
    plot_loglog_scaling(rows)

    print(f"\nDone. 6 plots saved to {OUT_DIR}\n")


if __name__ == "__main__":
    main()
