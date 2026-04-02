"""
llm/neural_symbolic/visualize.py
Render a ParseTree to a PNG file using matplotlib.

Two rendering styles:
  - 'tree'    : classic top-down constituency tree diagram
  - 'arc'     : linear sequence with arc spans (useful for long sequences)

Usage:
    from neural_symbolic.visualize import save_tree_png, save_arc_png
    save_tree_png(tree, sequence="ATGGAGG...", path="tree.png")
    save_arc_png(tree,  sequence="ATGGAGG...", path="arcs.png")
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

from .grammar import NT_NAMES, NUM_NT
from .tree import ParseTree

# ---------------------------------------------------------------------------
# Colour palette — one hex per NT, friendly on white backgrounds
# ---------------------------------------------------------------------------

NT_HEX: Dict[int, str] = {
    0:  "#7C3AED",   # GENE        — violet
    1:  "#2563EB",   # TRANSCRIPT  — blue
    2:  "#0891B2",   # LOCUS       — cyan
    3:  "#16A34A",   # EXON        — green
    4:  "#CA8A04",   # INTRON      — amber
    5:  "#15803D",   # CODON       — dark green
    6:  "#DC2626",   # PROMOTER    — red
    7:  "#0E7490",   # UTR         — teal
    8:  "#7E22CE",   # REGULATORY  — purple
    9:  "#6B7280",   # REPEAT      — gray
    10: "#9333EA",   # SPLICE      — fuchsia
    11: "#B45309",   # MOTIF       — brown
}

TERMINAL_HEX: Dict[str, str] = {
    "A": "#22C55E",  # green
    "C": "#3B82F6",  # blue
    "G": "#F59E0B",  # amber
    "T": "#EF4444",  # red
    "N": "#9CA3AF",  # gray
}

EDGE_COLOR   = "#374151"
BG_COLOR     = "#FFFFFF"
FONT_FAMILY  = "DejaVu Sans"


def _node_color(node: ParseTree) -> str:
    if node.is_terminal or node.is_leaf:
        nuc = node.label if isinstance(node.label, str) else "N"
        return TERMINAL_HEX.get(nuc, "#9CA3AF")
    return NT_HEX.get(node.label, "#6B7280")


# ---------------------------------------------------------------------------
# Layout engine — assigns (x, y) to every node
# ---------------------------------------------------------------------------

def _layout(root: ParseTree) -> Dict[int, Tuple[float, float]]:
    """
    Reingold–Tilford–style layout for binary trees.
    Returns {node_id → (x, y)} where y=0 is the root.
    """
    positions: Dict[int, Tuple[float, float]] = {}
    leaf_counter = [0]

    def _assign_x(node: ParseTree) -> float:
        """Post-order: assign x from leaves upward."""
        nid = id(node)
        if node.is_leaf or (node.left is None and node.right is None):
            x = float(leaf_counter[0])
            leaf_counter[0] += 1
            positions[nid] = (x, 0.0)   # y set in next pass
            return x

        children = [c for c in (node.left, node.right) if c is not None]
        xs = [_assign_x(c) for c in children]
        x  = sum(xs) / len(xs)
        positions[nid] = (x, 0.0)
        return x

    def _assign_y(node: ParseTree, depth: int = 0):
        nid = id(node)
        x, _ = positions[nid]
        positions[nid] = (x, -depth)
        for child in (node.left, node.right):
            if child is not None:
                _assign_y(child, depth + 1)

    _assign_x(root)
    _assign_y(root)
    return positions


# ---------------------------------------------------------------------------
# Tree diagram
# ---------------------------------------------------------------------------

def save_tree_png(
    tree:     ParseTree,
    sequence: Optional[str] = None,
    path:     str | Path = "parse_tree.png",
    dpi:      int = 150,
    title:    str = "",
) -> Path:
    """
    Render a constituency tree diagram and save as PNG.

    Args:
        tree     : ParseTree to render
        sequence : original DNA string (used for leaf labels)
        path     : output file path
        dpi      : image resolution
        title    : optional title shown at the top

    Returns:
        Path of the saved file.
    """
    positions = _layout(tree)

    # --- Figure sizing ---
    n_leaves = tree.num_leaves
    depth    = tree.depth
    fig_w    = max(6.0,  n_leaves * 0.9)
    fig_h    = max(4.0,  depth    * 1.2 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, fontfamily=FONT_FAMILY,
                     color="#111827", y=0.98)

    # Normalize positions to [0,1] for stable rendering
    all_x = [x for x, _ in positions.values()]
    all_y = [y for _, y in positions.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    def norm(x, y):
        nx = 0.05 + 0.90 * (x - min_x) / span_x
        ny = 0.90 - 0.80 * (y - min_y) / span_y   # invert: root at top
        return nx, ny

    # --- Draw edges first (behind nodes) ---
    def draw_edges(node: ParseTree):
        px, py = norm(*positions[id(node)])
        for child in (node.left, node.right):
            if child is not None:
                cx, cy = norm(*positions[id(child)])
                ax.plot([px, cx], [py, cy],
                        color=EDGE_COLOR, linewidth=0.8,
                        zorder=1, alpha=0.6)
                draw_edges(child)

    draw_edges(tree)

    # --- Draw nodes ---
    box_w   = min(0.10, 0.80 / max(n_leaves, 1))
    box_h   = 0.055
    fs_nt   = max(5.5, min(8.0, 80.0 / max(n_leaves, 8)))
    fs_term = max(6.0, min(9.0, 90.0 / max(n_leaves, 8)))

    def draw_node(node: ParseTree):
        x, y  = norm(*positions[id(node)])
        color = _node_color(node)

        if node.is_leaf:
            nuc = sequence[node.start] if (sequence and node.start < len(sequence)) \
                  else (node.label if isinstance(node.label, str) else "?")
            # Circle for terminals
            circ = plt.Circle((x, y), radius=box_h * 0.55,
                               color=color, zorder=3, linewidth=0)
            ax.add_patch(circ)
            ax.text(x, y, nuc,
                    ha="center", va="center",
                    fontsize=fs_term, fontweight="bold",
                    fontfamily=FONT_FAMILY, color="white", zorder=4)
        else:
            label = NT_NAMES.get(node.label, str(node.label)) \
                    if isinstance(node.label, int) else str(node.label)
            # Rounded rectangle for non-terminals
            bw = max(box_w, len(label) * 0.009 + 0.025)
            rect = FancyBboxPatch(
                (x - bw / 2, y - box_h / 2), bw, box_h,
                boxstyle="round,pad=0.005",
                facecolor=color, edgecolor="white",
                linewidth=0.6, zorder=3,
            )
            ax.add_patch(rect)
            ax.text(x, y, label,
                    ha="center", va="center",
                    fontsize=fs_nt, fontweight="500",
                    fontfamily=FONT_FAMILY, color="white", zorder=4)

        for child in (node.left, node.right):
            if child is not None:
                draw_node(child)

    draw_node(tree)

    # --- Legend ---
    seen_nts = {n.label for n in _all_nodes(tree) if isinstance(n.label, int)}
    legend_handles = [
        mpatches.Patch(color=NT_HEX.get(nt, "#6B7280"),
                       label=NT_NAMES.get(nt, f"NT{nt}"))
        for nt in sorted(seen_nts)
    ]
    if legend_handles:
        ax.legend(handles=legend_handles,
                  loc="lower right", fontsize=6.5,
                  framealpha=0.85, ncol=min(4, len(legend_handles)),
                  handlelength=0.8, handleheight=0.8,
                  borderpad=0.5, labelspacing=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96] if title else [0, 0, 1, 1])
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Arc span diagram
# ---------------------------------------------------------------------------

def save_arc_png(
    tree:     ParseTree,
    sequence: Optional[str] = None,
    path:     str | Path = "parse_arcs.png",
    dpi:      int = 150,
    title:    str = "",
) -> Path:
    """
    Render a linear arc diagram: sequence on x-axis, spans as coloured arcs.
    Each non-terminal span is shown as a semicircular arc above the sequence.
    Arcs are layered by depth so nested spans don't obscure each other.

    Args:
        tree     : ParseTree to render
        sequence : original DNA string
        path     : output file path
        dpi      : image resolution
        title    : optional title

    Returns:
        Path of the saved file.
    """
    # Collect all NT spans with depth
    spans: List[Tuple[int, int, int, int]] = []   # (start, end, nt, depth)

    def collect(node: ParseTree, depth: int = 0):
        if isinstance(node.label, int):
            spans.append((node.start, node.end, node.label, depth))
        for child in (node.left, node.right):
            if child is not None:
                collect(child, depth + 1)

    collect(tree)
    if not spans:
        spans = [(tree.start, tree.end, 0, 0)]

    seq_len  = tree.end
    seq      = sequence[:seq_len] if sequence else "N" * seq_len
    n_depths = max(d for *_, d in spans) + 1

    fig_w  = max(8.0, seq_len * 0.45)
    fig_h  = max(3.5, n_depths * 0.7 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, fontfamily=FONT_FAMILY,
                     color="#111827", y=0.98)

    # --- Sequence axis ---
    y_seq  = 0.15
    x_step = 1.0 / max(seq_len, 1)

    for i, nuc in enumerate(seq):
        x    = (i + 0.5) * x_step
        col  = TERMINAL_HEX.get(nuc, "#9CA3AF")
        circ = plt.Circle((x, y_seq), radius=x_step * 0.35,
                           color=col, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y_seq, nuc,
                ha="center", va="center",
                fontsize=max(5, min(9, 80 / max(seq_len, 8))),
                fontweight="bold", fontfamily=FONT_FAMILY,
                color="white", zorder=4)
        ax.text(x, y_seq - 0.06, str(i),
                ha="center", va="top",
                fontsize=max(4, min(6, 60 / max(seq_len, 8))),
                color="#9CA3AF", fontfamily=FONT_FAMILY, zorder=4)

    # --- Arcs (sorted shallowest first so deep arcs draw on top) ---
    y_top  = 0.92
    y_base = y_seq + 0.05

    # layer height: distribute depths between y_base and y_top
    layer_h = (y_top - y_base) / max(n_depths, 1)

    for start, end, nt, depth in sorted(spans, key=lambda s: s[3]):
        color    = NT_HEX.get(nt, "#6B7280")
        x_left   = (start + 0.5) * x_step
        x_right  = (end   - 0.5) * x_step
        x_mid    = (x_left + x_right) / 2
        arc_h    = (n_depths - depth) * layer_h * 0.85
        y_arc    = y_base + arc_h

        # Draw arc as a bezier via annotate
        lw = max(1.0, 2.5 - depth * 0.25)
        ax.annotate(
            "", xy=(x_right, y_base), xytext=(x_left, y_base),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                lw=lw,
                connectionstyle=f"arc3,rad=-{0.4 + (n_depths - depth) * 0.08:.2f}",
            ),
            zorder=2 + depth,
        )

        # Label at arc midpoint
        label    = NT_NAMES.get(nt, f"NT{nt}")
        fs       = max(5, min(7.5, 60 / max(seq_len, 8)))
        ax.text(
            x_mid, y_base + arc_h * 0.55, label,
            ha="center", va="bottom",
            fontsize=fs, fontfamily=FONT_FAMILY,
            color=color, fontweight="500",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor=color, linewidth=0.5, alpha=0.85),
            zorder=5 + depth,
        )

    plt.tight_layout(rect=[0, 0.08, 1, 0.96] if title else [0, 0.08, 1, 1])
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Combined: save both views side-by-side in one PNG
# ---------------------------------------------------------------------------

def save_combined_png(
    tree:     ParseTree,
    sequence: Optional[str] = None,
    path:     str | Path = "parse_combined.png",
    dpi:      int = 150,
    title:    str = "",
) -> Path:
    """
    Save tree diagram (left) and arc diagram (right) in a single figure.
    """
    import tempfile
    tmp_tree = Path(tempfile.mktemp(suffix=".png"))
    tmp_arc  = Path(tempfile.mktemp(suffix=".png"))

    save_tree_png(tree, sequence, path=tmp_tree, dpi=dpi)
    save_arc_png( tree, sequence, path=tmp_arc,  dpi=dpi)

    img_tree = plt.imread(str(tmp_tree))
    img_arc  = plt.imread(str(tmp_arc))

    h = max(img_tree.shape[0], img_arc.shape[0])
    w = img_tree.shape[1] + img_arc.shape[1]
    fig_w = w  / dpi
    fig_h = h  / dpi

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                      facecolor=BG_COLOR,
                                      gridspec_kw={"wspace": 0.02})
    for ax, img, sub in ((ax_l, img_tree, "Tree"), (ax_r, img_arc, "Arc spans")):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(sub, fontsize=9, color="#374151",
                     fontfamily=FONT_FAMILY, pad=4)

    if title:
        fig.suptitle(title, fontsize=11, fontfamily=FONT_FAMILY,
                     color="#111827", y=1.01)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight",
                facecolor=BG_COLOR, edgecolor="none")
    plt.close(fig)
    tmp_tree.unlink(missing_ok=True)
    tmp_arc.unlink(missing_ok=True)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_nodes(root: ParseTree) -> List[ParseTree]:
    result = [root]
    if root.left:  result.extend(_all_nodes(root.left))
    if root.right: result.extend(_all_nodes(root.right))
    return result