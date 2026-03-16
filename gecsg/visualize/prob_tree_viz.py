"""
gecsg.visualize.prob_tree_viz
==============================
Parse-tree visualization annotated with Viterbi log-probabilities.

Extends draw_parse_tree from tree_viz with:
  - Per-node probability label (rule weight or breaking prob)
  - Colour intensity proportional to log-prob magnitude
  - Title block showing total log P and P(seq)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse
import matplotlib.colors as mcolors

from gecsg.parser.parse_tree import ParseNode
from gecsg.visualize.tree_viz import (
    _collect_leaves, _assign_positions, _draw_edges,
    EDGE_COLOR, TEXT_COLOR, SEQ_COLOR,
    NT_HW, NT_HH, LEAF_RX, LEAF_RY,
    _draw_sequence_bar,
)


# Probability-aware colour palette
NT_BASE   = "#AED6F1"    # base blue  (internal nodes)
LEAF_BASE = "#A9DFBF"    # base green (leaf nodes)
HIGH_NT   = "#1A5276"    # dark blue  (high prob NT)
HIGH_LEAF = "#1E8449"    # dark green (high prob leaf)


def _prob_color(base_hex: str, high_hex: str, intensity: float) -> str:
    """Blend between base and high colour based on intensity in [0, 1]."""
    r1, g1, b1 = mcolors.to_rgb(base_hex)
    r2, g2, b2 = mcolors.to_rgb(high_hex)
    t = max(0.0, min(1.0, intensity))
    r = r1 + t * (r2 - r1)
    g = g1 + t * (g2 - g1)
    b = b1 + t * (b2 - b1)
    return mcolors.to_hex((r, g, b))


def _fmt_log_prob(lp: float) -> str:
    """Format a log probability for compact display on a node."""
    if lp <= -1e300:
        return "p=0"
    p = math.exp(lp)
    if p >= 0.01:
        return f"p={p:.3f}"
    return f"10^{lp/math.log(10):.1f}"


def _draw_stochastic_nodes(
    ax,
    node:        ParseNode,
    pos:         Dict,
    node_lp:     Dict[int, float],   # id(node) -> log_prob contribution
    min_lp:      float,
    max_lp:      float,
) -> None:
    """Draw all nodes with probability colour coding (children first)."""
    for ch in node.children:
        _draw_stochastic_nodes(ax, ch, pos, node_lp, min_lp, max_lp)

    x, y = pos[id(node)]
    lp   = node_lp.get(id(node), 0.0)

    # Normalise to [0, 1] for colour intensity
    span = max_lp - min_lp
    intensity = (lp - min_lp) / span if span > 1e-9 else 0.5

    if node.is_terminal:
        color = _prob_color(LEAF_BASE, HIGH_LEAF, intensity)
        el = Ellipse(
            (x, y), width=2 * LEAF_RX, height=2 * LEAF_RY,
            facecolor=color, edgecolor=EDGE_COLOR, linewidth=1.2, zorder=2,
        )
        ax.add_patch(el)
        ax.text(x, y + 0.07, node.label,
                ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=TEXT_COLOR, zorder=3)
        ax.text(x, y - 0.06, node.raw_codon or "",
                ha="center", va="center",
                fontsize=8.5, family="monospace", color="#1E8449", zorder=3)
        ax.text(x, y - 0.22, _fmt_log_prob(lp),
                ha="center", va="center",
                fontsize=6.5, color="#555555", zorder=3)
    else:
        color = _prob_color(NT_BASE, HIGH_NT, intensity)
        rect = FancyBboxPatch(
            (x - NT_HW, y - NT_HH), width=2 * NT_HW, height=2 * NT_HH,
            boxstyle="round,pad=0.06",
            facecolor=color, edgecolor=EDGE_COLOR, linewidth=1.2, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.06, node.label,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color=TEXT_COLOR, zorder=3)
        ax.text(x, y - 0.10, _fmt_log_prob(lp),
                ha="center", va="center",
                fontsize=6.5, color="#555555", zorder=3)


def draw_stochastic_parse_tree(
    root:        ParseNode,
    raw_seq:     str,
    node_lp:     Dict[int, float],
    total_log_p: float,
    ax:          Optional["plt.Axes"] = None,
    figsize:     Tuple[float, float] = (0, 0),
    title:       Optional[str] = None,
    show_raw:    bool = True,
    save_path:   Optional[str] = None,
    dpi:         int = 150,
) -> "plt.Figure":
    """
    Draw a GE-CSG parse tree annotated with Viterbi log-probabilities.

    Parameters
    ----------
    root        : root ParseNode
    raw_seq     : original DNA string
    node_lp     : {id(node): log_prob_contribution}  from StochasticParseResult
    total_log_p : total log P(seq | grammar)
    ax          : existing Axes, or None to create new Figure
    figsize     : (w, h) in inches; (0,0) = auto
    title       : plot title (supplemented with probability info)
    show_raw    : draw raw sequence bar below leaves
    save_path   : path to save PNG
    dpi         : resolution

    Returns
    -------
    matplotlib.figure.Figure
    """
    counter = [0]
    pos     = _assign_positions(root, counter)
    leaves  = _collect_leaves(root)
    n_leaves = len(leaves)

    tree_depth = root.depth
    if figsize == (0, 0):
        w = max(7.0, n_leaves * 1.3)
        h = max(4.5, (tree_depth + 3) * 1.1)
        figsize = (w, h)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_aspect("equal")
    ax.axis("off")

    # Build probability range for colour scaling
    lp_vals = list(node_lp.values())
    min_lp  = min(lp_vals) if lp_vals else -5.0
    max_lp  = max(lp_vals) if lp_vals else 0.0
    if min_lp == max_lp:
        min_lp -= 1.0

    # Title with probability info
    p_val = math.exp(total_log_p) if total_log_p > -1e300 else 0.0
    prob_str = (f"log P = {total_log_p:.3f}   |   "
                f"P = {p_val:.3e}")
    full_title = f"{title or 'GE-CSG Stochastic Parse'}  [{prob_str}]"
    ax.set_title(full_title, fontsize=10, pad=10)

    _draw_edges(ax, root, pos)
    _draw_stochastic_nodes(ax, root, pos, node_lp, min_lp, max_lp)

    if show_raw and leaves:
        ys    = [pos[id(lf)][1] for lf in leaves]
        y_bar = min(ys) - 0.85
        _draw_sequence_bar(ax, raw_seq, leaves, pos, y_bar, k=3)

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    pad   = 0.8
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(
        (min(all_y) - 1.4) if show_raw else (min(all_y) - pad),
        max(all_y) + pad,
    )

    # Colour legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor=LEAF_BASE, edgecolor=EDGE_COLOR, label="low log-prob"),
        Patch(facecolor=HIGH_LEAF, edgecolor=EDGE_COLOR, label="high log-prob"),
    ]
    ax.legend(handles=legend_elems, loc="lower right",
              fontsize=7, framealpha=0.7)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
