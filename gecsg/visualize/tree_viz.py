"""
gecsg.visualize.tree_viz
========================
Matplotlib-based visualization of GE-CSG parse trees.

Public API
----------
    fig = draw_parse_tree(root, raw_seq)
    fig.savefig("tree.png", dpi=150, bbox_inches="tight")

Visual design
-------------
  ┌─────────┐
  │  Gene   │  ← internal (NT) node: rounded blue rectangle
  └────┬────┘
       │
  ┌────┴────┐
  │   CDS   │
  └──┬──────┘
     │        └──────────────┐
  ┌──┴──┐                ┌───┴───┐
  │ Cdn │            ... │  CDS  │
  └──┬──┘                └───┬───┘
     │                       │
  ╔══╧══╗                 ╔══╧══╗
  ║ C0  ║                 ║ C1  ║   ← leaf (coset) node: green oval
  ║ ATG ║                 ║ TAA ║
  ╚═════╝                 ╚═════╝

  A  T  G  |  G  C  T  |  T  A  A     ← raw sequence annotation

Layout algorithm
----------------
  1. Assign x to each leaf: leaves get integer positions 0, 1, 2, ...
     (left-to-right order from raw sequence).
  2. Internal nodes get x = mean(x of direct children).
  3. y-coordinate = -(depth from root), so root is at y=0, leaves at y=-depth.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # safe headless default; caller can switch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse

from gecsg.parser.parse_tree import ParseNode


# ── Color palette ──────────────────────────────────────────────────────────
NT_COLOR    = "#AED6F1"   # soft blue   — internal NT nodes
LEAF_COLOR  = "#A9DFBF"   # soft green  — coset leaf nodes
EDGE_COLOR  = "#5D6D7E"   # slate grey  — edges
TEXT_COLOR  = "#1A252F"   # near-black  — labels
SEQ_COLOR   = "#7F8C8D"   # medium grey — sequence annotation


# ─────────────────────────────────────────────────────────────────────────────
# Position assignment
# ─────────────────────────────────────────────────────────────────────────────

def _collect_leaves(node: ParseNode) -> List[ParseNode]:
    """Return leaf nodes in left-to-right order."""
    if node.is_terminal:
        return [node]
    result: List[ParseNode] = []
    for ch in node.children:
        result.extend(_collect_leaves(ch))
    return result


def _assign_positions(
    node:    ParseNode,
    counter: List[int],          # mutable counter for leaf x positions
    depth:   int = 0,
    pos:     Optional[Dict] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Assign (x, y) to every node.  Returns {id(node): (x, y)}.
    y = -depth  (root at 0, leaves most negative).
    """
    if pos is None:
        pos = {}

    if node.is_terminal:
        x = float(counter[0])
        counter[0] += 1
    else:
        for ch in node.children:
            _assign_positions(ch, counter, depth + 1, pos)
        if node.children:
            xs = [pos[id(ch)][0] for ch in node.children]
            x = sum(xs) / len(xs)
        else:
            x = float(counter[0])
            counter[0] += 1

    pos[id(node)] = (x, -depth)
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

# Box half-sizes (in data coordinates)
NT_HW   = 0.42    # half-width
NT_HH   = 0.28    # half-height
LEAF_RX = 0.40    # ellipse x-radius
LEAF_RY = 0.30    # ellipse y-radius


def _draw_edges(ax, node: ParseNode, pos: Dict) -> None:
    """Draw edges from this node to its children (recursively)."""
    px, py = pos[id(node)]
    for ch in node.children:
        cx, cy = pos[id(ch)]
        ax.plot([px, cx], [py, cy],
                color=EDGE_COLOR, linewidth=1.2, zorder=1)
        _draw_edges(ax, ch, pos)


def _draw_nodes(ax, node: ParseNode, pos: Dict) -> None:
    """Draw all nodes (recursively, children first so parents overlay edges)."""
    # Draw children first
    for ch in node.children:
        _draw_nodes(ax, ch, pos)

    x, y = pos[id(node)]

    if node.is_terminal:
        # Leaf: green ellipse with 2-line label
        el = Ellipse(
            (x, y), width=2 * LEAF_RX, height=2 * LEAF_RY,
            facecolor=LEAF_COLOR, edgecolor=EDGE_COLOR,
            linewidth=1.2, zorder=2,
        )
        ax.add_patch(el)
        coset_lbl = node.label
        raw_lbl   = node.raw_codon or ""
        ax.text(x, y + 0.06, coset_lbl,
                ha="center", va="center",
                fontsize=7.5, fontweight="bold",
                color=TEXT_COLOR, zorder=3)
        ax.text(x, y - 0.09, raw_lbl,
                ha="center", va="center",
                fontsize=8.5, family="monospace",
                color="#1E8449", zorder=3)
    else:
        # Internal: blue rounded rectangle
        rect = FancyBboxPatch(
            (x - NT_HW, y - NT_HH),
            width=2 * NT_HW, height=2 * NT_HH,
            boxstyle="round,pad=0.06",
            facecolor=NT_COLOR, edgecolor=EDGE_COLOR,
            linewidth=1.2, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x, y, node.label,
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color=TEXT_COLOR, zorder=3)


def _draw_sequence_bar(
    ax:     "plt.Axes",
    raw_seq: str,
    leaves:  List[ParseNode],
    pos:     Dict,
    y_bar:   float,
    k:       int = 3,
) -> None:
    """
    Draw the raw DNA sequence below the leaves, aligned to leaf positions.
    Each codon is printed in monospace, with '│' separators between codons.
    """
    # Draw separator line
    if not leaves:
        return
    xs = [pos[id(lf)][0] for lf in leaves]
    x_min, x_max = min(xs) - 0.5, max(xs) + 0.5
    ax.axhline(y=y_bar + 0.15, xmin=0, xmax=1,
               color=SEQ_COLOR, linewidth=0.5, alpha=0.4)

    for i, lf in enumerate(leaves):
        x   = pos[id(lf)][0]
        codon_str = raw_seq[k * i: k * (i + 1)] if i * k < len(raw_seq) else ""
        # Individual bases spaced out
        for bi, base in enumerate(codon_str):
            ax.text(x - 0.25 + bi * 0.25, y_bar, base,
                    ha="center", va="center",
                    fontsize=9, family="monospace",
                    color=TEXT_COLOR, fontweight="bold")
        # Separator after each codon (except last)
        if i < len(leaves) - 1:
            ax.text(x + 0.38, y_bar, "│",
                    ha="center", va="center",
                    fontsize=9, color=SEQ_COLOR)

    # Label
    ax.text(x_min - 0.1, y_bar, "5'",
            ha="right", va="center", fontsize=8, color=SEQ_COLOR)
    ax.text(x_max + 0.1, y_bar, "3'",
            ha="left", va="center", fontsize=8, color=SEQ_COLOR)


# ─────────────────────────────────────────────────────────────────────────────
# Public function
# ─────────────────────────────────────────────────────────────────────────────

def draw_parse_tree(
    root:            ParseNode,
    raw_seq:         str,
    ax:              Optional["plt.Axes"] = None,
    figsize:         Tuple[float, float] = (0, 0),   # 0 = auto-size
    title:           Optional[str] = None,
    show_raw:        bool = True,
    save_path:       Optional[str] = None,
    dpi:             int = 150,
) -> "plt.Figure":
    """
    Draw a GE-CSG parse tree.

    Parameters
    ----------
    root       : root ParseNode (from ParseResult.trees()[0])
    raw_seq    : the original DNA string (used for bottom annotation)
    ax         : existing Axes to draw into, or None to create new Figure
    figsize    : (width, height) in inches; (0,0) = auto-compute from tree size
    title      : optional plot title
    show_raw   : if True, draw raw sequence annotation below leaves
    save_path  : if given, save figure to this path
    dpi        : resolution when saving

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── 1. Compute positions ─────────────────────────────────────────────
    counter = [0]
    pos     = _assign_positions(root, counter)
    leaves  = _collect_leaves(root)
    n_leaves = len(leaves)

    # ── 2. Figure setup ──────────────────────────────────────────────────
    tree_depth = root.depth
    if figsize == (0, 0):
        w = max(6.0, n_leaves * 1.2)
        h = max(4.0, (tree_depth + 2) * 1.1)
        figsize = (w, h)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=11, pad=10)

    # ── 3. Draw tree ─────────────────────────────────────────────────────
    _draw_edges(ax, root, pos)
    _draw_nodes(ax, root, pos)

    # ── 4. Sequence bar ──────────────────────────────────────────────────
    if show_raw and leaves:
        ys   = [pos[id(lf)][1] for lf in leaves]
        y_bar = min(ys) - 0.75
        _draw_sequence_bar(ax, raw_seq, leaves, pos, y_bar, k=3)

    # ── 5. Axis limits with padding ──────────────────────────────────────
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    pad   = 0.7
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(
        (min(all_y) - 1.2) if show_raw else (min(all_y) - pad),
        max(all_y) + pad,
    )

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig
