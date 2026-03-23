"""
visualization/visualize_gtf_cds.py
====================================
Parse synthetic CDS sequences derived from real GENCODE gene sizes and
produce both PNG (fishbone layout) and TXT parse trees.

Fishbone layout
---------------
The paired_chain_grammar produces:
    ORF → Pair ORF → Pair Pair ORF → ...

A conventional top-down tree layout gives a diagonal staircase.
Instead we use a "fishbone" layout:
  - ORF spine runs VERTICALLY down the centre
  - Each Pair branches HORIZONTALLY (two BodyCodon leaves, one each side)
  - Start/Stop codons at top/bottom

This makes the Pair branching structure clearly visible even for long CDS.
"""
import sys, os, re, random
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from gecsg.grammar.paired_chain_grammar import paired_chain_grammar
from gecsg.parser.earley import EquivariantEarleyParser
from gecsg.parser.parse_tree import ParseNode

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT = Path(__file__).parent.parent / "outputs_gtf"
OUT.mkdir(exist_ok=True)

GTF = Path(__file__).parent.parent / "dataset/GENECODE/gencode.v49.annotation.gtf"

# ── Human codon usage ─────────────────────────────────────────────────────────
CODON_FREQ = {
    "ATG": 1.00,
    "TAA": 0.28, "TAG": 0.20, "TGA": 0.52,
    "AAA": 0.42, "AAG": 0.58, "ACA": 0.28, "ACC": 0.36, "ACG": 0.11, "ACT": 0.25,
    "AGA": 0.21, "AGG": 0.21, "AGC": 0.24, "AGT": 0.15,
    "ATA": 0.16, "ATC": 0.48, "ATT": 0.36,
    "TAC": 0.57, "TAT": 0.43, "TCA": 0.15, "TCC": 0.22, "TCG": 0.06, "TCT": 0.15,
    "TGC": 0.55, "TGT": 0.45, "TGG": 1.00,
    "TTA": 0.07, "TTC": 0.55, "TTG": 0.13, "TTT": 0.45,
    "GAA": 0.42, "GAG": 0.58, "GAC": 0.54, "GAT": 0.46,
    "GCA": 0.23, "GCC": 0.40, "GCG": 0.11, "GCT": 0.26,
    "GGA": 0.25, "GGC": 0.34, "GGG": 0.25, "GGT": 0.16,
    "GTA": 0.11, "GTC": 0.24, "GTG": 0.47, "GTT": 0.18,
    "CAA": 0.27, "CAG": 0.73, "CAC": 0.59, "CAT": 0.41,
    "CCA": 0.28, "CCC": 0.32, "CCG": 0.11, "CCT": 0.29,
    "CGA": 0.11, "CGC": 0.19, "CGG": 0.20, "CGT": 0.08,
    "CTA": 0.07, "CTC": 0.20, "CTG": 0.41, "CTT": 0.13,
}
BODY_CODONS  = [c for c in CODON_FREQ if c not in {"ATG","TAA","TAG","TGA"}]
BODY_WEIGHTS = [CODON_FREQ[c] for c in BODY_CODONS]
STOP_CODONS  = ["TAA","TAG","TGA"]
STOP_WEIGHTS = [CODON_FREQ[c] for c in STOP_CODONS]


def synthetic_cds(n_codons: int, rng: random.Random) -> str:
    """Generate synthetic CDS; n_body = n_codons-2 must be even >= 4."""
    n_body = n_codons - 2
    if n_body % 2 != 0:
        n_body -= 1
    body = rng.choices(BODY_CODONS, weights=BODY_WEIGHTS, k=n_body)
    stop = rng.choices(STOP_CODONS,  weights=STOP_WEIGHTS,  k=1)[0]
    return "ATG" + "".join(body) + stop


# ── GTF scan ───────────────────────────────────────────────────────────────────

def find_real_transcripts(gtf_path: Path, targets: list) -> list:
    ATTR_RE = re.compile(r'transcript_id "([^"]+)"')
    NAME_RE  = re.compile(r'gene_name "([^"]+)"')
    GENE_RE  = re.compile(r'gene_id "([^"]+)"')

    cds_len = defaultdict(int)
    tx_info = {}

    print(f"Scanning {gtf_path.name} …", flush=True)
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#") or "\tCDS\t" not in line:
                continue
            parts = line.split("\t", 8)
            if len(parts) < 9:
                continue
            start, end = int(parts[3]), int(parts[4])
            attrs = parts[8]
            m = ATTR_RE.search(attrs)
            if not m:
                continue
            tid = m.group(1)
            cds_len[tid] += end - start + 1
            if tid not in tx_info:
                gn = NAME_RE.search(attrs)
                gi = GENE_RE.search(attrs)
                tx_info[tid] = {
                    "gene_name": gn.group(1) if gn else "",
                    "gene_id":   gi.group(1) if gi else "",
                }

    # paired_chain_grammar: n_body even >= 4
    valid = {tid: l for tid, l in cds_len.items()
             if l % 3 == 0
             and (l // 3) >= 6
             and (l // 3 - 2) % 2 == 0}

    results = []
    for target in targets:
        nt_target = target * 3
        tid, length = min(valid.items(), key=lambda x: abs(x[1] - nt_target))
        info = tx_info[tid]
        results.append({
            "target":    target,
            "transcript": tid,
            "gene_name": info["gene_name"],
            "gene_id":   info["gene_id"],
            "n_codons":  length // 3,
        })
    return results


# ── Parse tree → flat Pair list ────────────────────────────────────────────────

def extract_pairs(root: ParseNode, k: int = 3) -> tuple:
    """
    Walk the paired_chain_grammar parse tree and extract:
      start_codon : str
      stop_codon  : str
      pairs       : list of (codon1, codon2, coset1_label, coset2_label)
                    or (codon1, codon2, codon3, codon4, ...) for the CSG Quad

    Returns (start_codon, pairs, stop_codon).
    """
    start_codon = ""
    stop_codon  = ""
    pairs       = []

    def leaves(node: ParseNode):
        if node.is_terminal:
            yield node.raw_codon or "???"
        else:
            for c in node.children:
                yield from leaves(c)

    def walk(node: ParseNode):
        nonlocal start_codon, stop_codon
        if node.label == "StartCodon":
            start_codon = next(leaves(node), "???")
        elif node.label == "StopCodon":
            stop_codon = next(leaves(node), "???")
        elif node.label == "Pair":
            codons = list(leaves(node))
            pairs.append(codons)
        else:
            for c in node.children:
                walk(c)

    walk(root)
    return start_codon, pairs, stop_codon


# ── Fishbone PNG renderer ─────────────────────────────────────────────────────

def draw_fishbone(start_codon, pairs, stop_codon,
                  title="", save_path=None, max_pairs_shown=80):
    """
    Draw the paired_chain parse tree as a fishbone diagram.

    Vertical spine: Start → ORF_1 → ORF_2 → … → Stop
    Horizontal bones: each Pair branches left (codon1) and right (codon2).
    """
    n = len(pairs)
    shown = min(n, max_pairs_shown)
    truncated = n > max_pairs_shown

    # Layout constants
    SPINE_X   = 0.0
    ROW_H     = 0.7        # vertical spacing between rows
    BONE_W    = 2.8        # horizontal reach of each bone arm
    BOX_W     = 1.0        # codon box width
    BOX_H     = 0.35       # codon box height
    SPINE_TOP = 0.5        # y of start codon
    SPINE_BOT = -(shown + 1) * ROW_H

    fig_h = max(6, (shown + 3) * ROW_H * 0.6)
    fig_w = 10
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal")
    ax.axis("off")

    def codon_color(codon):
        if not codon or len(codon) < 1:
            return "#AED6F1"
        c = codon[0].upper()
        return {"A": "#A9DFBF", "T": "#F9E79F", "G": "#AED6F1", "C": "#F5CBA7"}.get(c, "#AED6F1")

    def draw_box(ax, x, y, text, color, fontsize=7):
        rect = mpatches.FancyBboxPatch(
            (x - BOX_W/2, y - BOX_H/2), BOX_W, BOX_H,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="#5D6D7E", linewidth=0.6, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color="#1A252F",
                fontfamily="monospace", zorder=4)

    def draw_spine_node(ax, x, y, label, color="#D5D8DC"):
        circ = plt.Circle((x, y), 0.18, color=color, zorder=3, linewidth=0.5,
                           ec="#5D6D7E")
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=5.5, color="#1A252F", zorder=4)

    # ── Spine ──────────────────────────────────────────────────────────────────
    spine_ys = [SPINE_TOP - (i+1) * ROW_H for i in range(shown)]

    # Start codon
    draw_box(ax, SPINE_X, SPINE_TOP, f"START\n{start_codon}",
             "#A9DFBF", fontsize=7.5)
    ax.annotate("", xy=(SPINE_X, spine_ys[0] + 0.25),
                xytext=(SPINE_X, SPINE_TOP - BOX_H/2),
                arrowprops=dict(arrowstyle="-", color="#5D6D7E", lw=1.0))

    # ORF spine nodes + bones
    for i, row_y in enumerate(spine_ys):
        # ORF spine node (small circle)
        draw_spine_node(ax, SPINE_X, row_y, "ORF")

        pair = pairs[i]
        # Draw left bone (codon 0, or codon 0+2 for Quad)
        if len(pair) == 4:   # CSG Quad: first pair
            # Left side: codons 0, 2  stacked
            for j, ci in enumerate([0, 2]):
                lx = -BONE_W + (0.3 - j*0.0)
                ly = row_y + 0.15 - j * 0.38
                ax.plot([SPINE_X, -BONE_W*0.55, lx + BOX_W/2],
                        [row_y, row_y + 0.08 - j*0.19, ly], "-",
                        color="#5D6D7E", lw=0.7, zorder=1)
                draw_box(ax, lx, ly, pair[ci], codon_color(pair[ci]), fontsize=6.5)
            # Right side: codons 1, 3  stacked
            for j, ci in enumerate([1, 3]):
                rx = BONE_W - (0.3 - j*0.0)
                ry = row_y + 0.15 - j * 0.38
                ax.plot([SPINE_X, BONE_W*0.55, rx - BOX_W/2],
                        [row_y, row_y + 0.08 - j*0.19, ry], "-",
                        color="#5D6D7E", lw=0.7, zorder=1)
                draw_box(ax, rx, ry, pair[ci], codon_color(pair[ci]), fontsize=6.5)
            # Label the Quad
            ax.text(SPINE_X, row_y + 0.42, "Quad (CSG)",
                    ha="center", va="bottom", fontsize=5.5,
                    color="#884EA0", style="italic")
        else:
            # Regular pair: left / right
            lx, rx = -BONE_W, BONE_W
            ax.plot([SPINE_X, lx + BOX_W/2], [row_y, row_y],
                    "-", color="#5D6D7E", lw=0.8, zorder=1)
            ax.plot([SPINE_X, rx - BOX_W/2], [row_y, row_y],
                    "-", color="#5D6D7E", lw=0.8, zorder=1)
            draw_box(ax, lx, row_y, pair[0], codon_color(pair[0]))
            draw_box(ax, rx, row_y, pair[1], codon_color(pair[1]))

        # Spine arrow down (except last)
        if i < shown - 1:
            ax.annotate("", xy=(SPINE_X, spine_ys[i+1] + 0.20),
                        xytext=(SPINE_X, row_y - 0.22),
                        arrowprops=dict(arrowstyle="-", color="#5D6D7E", lw=0.8))

    # Stop codon
    stop_y = spine_ys[-1] - ROW_H
    ax.annotate("", xy=(SPINE_X, stop_y + BOX_H/2),
                xytext=(SPINE_X, spine_ys[-1] - 0.22),
                arrowprops=dict(arrowstyle="-", color="#5D6D7E", lw=1.0))
    draw_box(ax, SPINE_X, stop_y, f"STOP\n{stop_codon}",
             "#F9E79F", fontsize=7.5)

    if truncated:
        ax.text(SPINE_X, (spine_ys[-1] + stop_y)/2 - ROW_H*0.3,
                f"… {n - max_pairs_shown} more pairs …",
                ha="center", va="center", fontsize=7,
                color="#884EA0", style="italic")

    # Legend
    legend_y = SPINE_TOP + 0.6
    for codon_start, color, label in [
        ("A", "#A9DFBF", "A-start (C0)"),
        ("T", "#F9E79F", "T-start (C1)"),
        ("G", "#AED6F1", "G-start (C2)"),
        ("C", "#F5CBA7", "C-start (C3)"),
    ]:
        r = mpatches.Patch(color=color, label=label)
    ax.legend(
        handles=[
            mpatches.Patch(color="#A9DFBF", label="A-start (C0)"),
            mpatches.Patch(color="#F9E79F", label="T-start (C1)"),
            mpatches.Patch(color="#AED6F1", label="G-start (C2)"),
            mpatches.Patch(color="#F5CBA7", label="C-start (C3)"),
        ],
        loc="upper right", fontsize=6.5, framealpha=0.8,
    )

    # Pair index labels on the right spine
    for i, row_y in enumerate(spine_ys):
        ax.text(BONE_W + BOX_W/2 + 0.15, row_y,
                f"#{i+1}", ha="left", va="center",
                fontsize=5, color="#AAB7B8")

    ax.set_xlim(-BONE_W - BOX_W - 0.3, BONE_W + BOX_W + 0.8)
    ax.set_ylim(stop_y - 0.5, SPINE_TOP + 1.0)
    ax.set_title(title, fontsize=9, pad=6)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig


# ── TXT tree renderer ─────────────────────────────────────────────────────────

def tree_to_text(node: ParseNode, raw_seq: str, k: int = 3) -> str:
    lines = []

    def _label(n: ParseNode) -> str:
        if n.is_terminal:
            return f"{n.label}  [{n.raw_codon or '???'}]"
        return n.label

    def _render(n: ParseNode, prefix: str, is_last: bool) -> None:
        connector = "└── " if is_last else "├── "
        lines.append(prefix + connector + _label(n))
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(n.children):
            _render(child, child_prefix, i == len(n.children) - 1)

    lines.append(node.label)
    for i, child in enumerate(node.children):
        _render(child, "", i == len(node.children) - 1)
    return "\n".join(lines)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    targets = [50, 100, 200, 500]
    rng = random.Random(42)

    print("=" * 68)
    print("GE-CSG Fishbone Parse Trees — GENCODE v49 Real Gene Sizes")
    print("Grammar: paired_chain_grammar (genuine CSG)")
    print("=" * 68)

    real_txs = find_real_transcripts(GTF, targets)

    print()
    print("Building paired_chain_grammar (cf_relaxation=True for Earley) …")
    grammar = paired_chain_grammar(cf_relaxation=True)
    parser  = EquivariantEarleyParser(grammar)

    print()
    print(f"{'Target':>7}  {'Actual':>7}  {'Gene':<12}  {'Transcript':<25}")
    print("-" * 68)

    for info in real_txs:
        target   = info["target"]
        n_codons = info["n_codons"]
        gene     = info["gene_name"]
        tid      = info["transcript"]
        gid      = info["gene_id"]

        seq    = synthetic_cds(n_codons, rng)
        result = parser.parse(seq)
        status = "OK" if result.accepted else "REJECTED"
        print(f"  {target:5d}  {n_codons:7d}  {gene:<12}  {tid:<25}  {status}")

        if not result.accepted:
            continue
        trees = result.trees()
        if not trees:
            print(f"    -> no parse tree"); continue
        tree = trees[0]

        stem = f"GTF_{target:04d}codon_{gene}"

        # ── TXT ───────────────────────────────────────────────────────────────
        txt_path = OUT / (stem + ".txt")
        header = (
            f"Gene: {gene}  ({gid})\n"
            f"Transcript: {tid}\n"
            f"Codons: {n_codons}  (target={target}, synthetic sequence)\n"
            f"Sequence: {seq[:60]}{'…' if len(seq)>60 else ''}\n"
            f"Grammar: paired_chain_grammar (CSG: Pair->4BC after StartCodon)\n"
            f"Accepted: {result.accepted}\n"
            f"\nParse tree (box-drawing):\n"
            f"{'-'*60}\n"
        )
        txt_path.write_text(header + tree_to_text(tree, seq, k=grammar.k) + "\n",
                            encoding="utf-8")
        print(f"    TXT -> {txt_path.name}  ({txt_path.stat().st_size//1024} KB)")

        # ── PNG (fishbone) ────────────────────────────────────────────────────
        png_path = OUT / (stem + "_fishbone.png")
        start_c, pairs, stop_c = extract_pairs(tree, k=grammar.k)
        n_pairs = len(pairs)
        title = (
            f"{gene}  ({gid})\n"
            f"Transcript: {tid}  |  {n_codons} codons  |  {n_pairs} pairs\n"
            f"paired_chain_grammar (CSG) — synthetic CDS, real gene length from GENCODE v49"
        )
        max_show = 60 if target <= 100 else 80
        fig = draw_fishbone(start_c, pairs, stop_c, title=title,
                            save_path=str(png_path), max_pairs_shown=max_show)
        plt.close(fig)
        print(f"    PNG -> {png_path.name}  ({png_path.stat().st_size//1024} KB)")

    print()
    print(f"Done.  Outputs in: {OUT}/")


if __name__ == "__main__":
    main()
