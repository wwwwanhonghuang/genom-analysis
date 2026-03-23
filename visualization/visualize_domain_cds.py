"""
visualization/visualize_domain_cds.py
======================================
GE-CSG Domain Grammar parse trees from real GENCODE v49 gene sizes.

Grammar: domain_grammar  (genuine CSG, balanced binary Core)
  Gene -> CDS -> StartCodon ORF StopCodon
  [CSG] ORF -> NTerm Core CTerm   left_ctx=StartCodon
  NTerm/CTerm: exactly 3 BodyCodons each  (flat nodes)
  Core -> Core Core  (balanced binary split)
  Core -> BodyCodon BodyCodon  (leaf: 2-codon pair)

Tree shape
----------
  Previous grammars:   depth = O(n)   chain / fishbone spine
  This grammar:        depth = O(log n)  genuine balanced binary tree

  For 100-codon CDS (92 Core codons, 46 pairs):
    Core depth = ceil(log2(46)) = 6 levels
    Total depth = Gene + CDS + ORF + Core = ~9 levels
    Width       = 100 leaf positions  (triangle shape, NOT a spine)

Usage
-----
  python visualization/visualize_domain_cds.py
  Outputs -> outputs_domain/  (PNG + TXT per gene)
"""

from __future__ import annotations

import os
import sys
import math
import random
from pathlib import Path
from typing import List, Tuple, Optional

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gecsg.parser.parse_tree import ParseNode
from gecsg.visualize.tree_viz import draw_parse_tree

# ── constants ─────────────────────────────────────────────────────────────────
GTF_PATH   = ROOT / "dataset" / "GENECODE" / "gencode.v49.annotation.gtf"
OUT_DIR    = ROOT / "outputs_domain"
OUT_DIR.mkdir(exist_ok=True)

NTERM_SIZE = 3   # fixed N-terminal body-codon count
CTERM_SIZE = 3   # fixed C-terminal body-codon count
TARGETS    = [50, 100, 200, 500]   # desired total codon counts

# Human codon usage frequencies (from Kazusa database)
CODON_FREQ = {
    "TTT":0.017,"TTC":0.020,"TTA":0.007,"TTG":0.013,
    "CTT":0.013,"CTC":0.019,"CTA":0.007,"CTG":0.040,
    "ATT":0.016,"ATC":0.021,"ATA":0.007,"ATG":0.022,
    "GTT":0.011,"GTC":0.015,"GTA":0.007,"GTG":0.028,
    "TCT":0.015,"TCC":0.018,"TCA":0.012,"TCG":0.004,
    "CCT":0.018,"CCC":0.020,"CCA":0.017,"CCG":0.007,
    "ACT":0.013,"ACC":0.019,"ACA":0.015,"ACG":0.006,
    "GCT":0.018,"GCC":0.028,"GCA":0.016,"GCG":0.007,
    "TAT":0.012,"TAC":0.016,"TAA":0.001,"TAG":0.001,
    "CAT":0.011,"CAC":0.015,"CAA":0.012,"CAG":0.034,
    "AAT":0.017,"AAC":0.020,"AAA":0.024,"AAG":0.033,
    "GAT":0.022,"GAC":0.026,"GAA":0.029,"GAG":0.040,
    "TGT":0.011,"TGC":0.013,"TGA":0.001,"TGG":0.013,
    "CGT":0.005,"CGC":0.011,"CGA":0.006,"CGG":0.012,
    "AGT":0.015,"AGC":0.020,"AGA":0.012,"AGG":0.012,
    "GGT":0.011,"GGC":0.022,"GGA":0.017,"GGG":0.016,
}
START_CODONS = ["ATG"]
STOP_CODONS  = ["TAA", "TAG", "TGA"]
BODY_CODONS  = [c for c in CODON_FREQ
                if c not in START_CODONS + STOP_CODONS]
BODY_WEIGHTS = [CODON_FREQ[c] for c in BODY_CODONS]


# ─────────────────────────────────────────────────────────────────────────────
# GTF scanning
# ─────────────────────────────────────────────────────────────────────────────

def find_real_transcripts(gtf_path: Path, targets: List[int]):
    """
    Scan GTF CDS features to find transcripts whose total CDS length matches
    each target codon count.

    Filters applied
    ---------------
    - CDS length divisible by 3
    - n_codons >= 10  (minimum for domain_grammar: 3+2+3 body + start/stop)
    - n_body = n_codons - 2 is EVEN  (required by domain_grammar)
    - n_body >= 8  (NTerm 3 + Core 2 + CTerm 3)

    Returns
    -------
    dict  target -> (gene_name, transcript_id, actual_n_codons)
    """
    lengths: dict = {}   # transcript_id -> (gene_name, total_length)

    print(f"Scanning {gtf_path.name} ...")
    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            try:
                seg_len = int(fields[4]) - int(fields[3]) + 1
            except ValueError:
                continue

            attr = fields[8]
            tid  = _attr(attr, "transcript_id")
            gname = _attr(attr, "gene_name") or _attr(attr, "gene_id") or "?"
            if not tid:
                continue

            if tid not in lengths:
                lengths[tid] = [gname, 0]
            lengths[tid][1] += seg_len

    # Build candidate list: (n_codons, gene_name, transcript_id)
    candidates = []
    for tid, (gname, total_len) in lengths.items():
        if total_len % 3 != 0:
            continue
        n_codons = total_len // 3
        n_body   = n_codons - 2
        if n_body < 8 or n_body % 2 != 0:
            continue
        candidates.append((n_codons, gname, tid))

    candidates.sort()

    # Pick closest transcript per target
    result = {}
    for target in targets:
        best = min(candidates, key=lambda x: abs(x[0] - target))
        result[target] = (best[1], best[2], best[0])

    return result


def _attr(attr_str: str, key: str) -> Optional[str]:
    """Extract a value from a GTF attribute string."""
    for part in attr_str.split(";"):
        part = part.strip()
        if part.startswith(key + " "):
            val = part[len(key):].strip().strip('"')
            return val
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic CDS generator
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_cds(n_codons: int, rng: random.Random) -> str:
    """
    Generate a synthetic CDS of exactly n_codons codons using human codon
    usage frequencies.  Sequence: ATG + (n_codons-2) body codons + TAA.
    n_body = n_codons - 2 must be even and >= 8.
    """
    n_body = n_codons - 2
    assert n_body % 2 == 0 and n_body >= 8, f"Invalid n_body={n_body}"
    body = rng.choices(BODY_CODONS, weights=BODY_WEIGHTS, k=n_body)
    return "ATG" + "".join(body) + "TAA"


# ─────────────────────────────────────────────────────────────────────────────
# Balanced binary parse tree builder
# ─────────────────────────────────────────────────────────────────────────────

_COSET_LABEL = {"A": "C0", "T": "C1", "G": "C2", "C": "C3"}


def _terminal(codon: str, pos: int) -> ParseNode:
    """Coset leaf node (is_terminal=True)."""
    return ParseNode(
        label=_COSET_LABEL[codon[0].upper()],
        span=(pos, pos + 1),
        is_terminal=True,
        raw_codon=codon,
    )


def _body_codon(codon: str, pos: int) -> ParseNode:
    """BodyCodon internal node wrapping a coset terminal."""
    return ParseNode(
        label="BodyCodon",
        span=(pos, pos + 1),
        is_terminal=False,
        children=(_terminal(codon, pos),),
    )


def _build_core(codons: List[str], start_pos: int) -> ParseNode:
    """
    Recursively build a balanced binary Core subtree.

    Rules used:
      Core -> Core Core        (binary split, both children non-trivial)
      Core -> BodyCodon BodyCodon   (leaf: 2-codon pair)

    The split is ALWAYS at the midpoint (rounded to the nearest even index),
    so the tree depth = ceil(log2(n/2)).  This is O(log n), NOT O(n).
    """
    n = len(codons)
    assert n >= 2 and n % 2 == 0, f"Core must have even length >= 2, got {n}"

    if n == 2:
        # Base case: leaf Core -> BodyCodon BodyCodon
        return ParseNode(
            label="Core",
            span=(start_pos, start_pos + 2),
            is_terminal=False,
            children=(
                _body_codon(codons[0], start_pos),
                _body_codon(codons[1], start_pos + 1),
            ),
        )

    # Recursive case: Core -> Core Core
    # Balanced split: choose midpoint rounded to nearest even number
    mid = n // 2
    if mid % 2 == 1:
        mid += 1   # ensure both halves are even (required by grammar)

    left  = _build_core(codons[:mid],  start_pos)
    right = _build_core(codons[mid:],  start_pos + mid)

    return ParseNode(
        label="Core",
        span=(start_pos, start_pos + n),
        is_terminal=False,
        children=(left, right),
    )


def build_domain_tree(seq: str) -> ParseNode:
    """
    Construct the full domain parse tree for a CDS sequence.

    Tree structure
    --------------
    Gene
    └── CDS
        ├── StartCodon  (1 codon, leaf)
        ├── ORF  [CSG: left_ctx=StartCodon]
        │   ├── NTerm  (NTERM_SIZE body codons, flat)
        │   ├── Core   (balanced binary tree, n_body - 6 codons)
        │   └── CTerm  (CTERM_SIZE body codons, flat)
        └── StopCodon   (1 codon, leaf)
    """
    k = 3
    codons = [seq[i:i+k] for i in range(0, len(seq), k)]
    n = len(codons)
    n_body = n - 2

    assert n_body >= NTERM_SIZE + 2 + CTERM_SIZE, (
        f"Sequence too short: need at least {NTERM_SIZE+CTERM_SIZE+4} codons, "
        f"got {n}"
    )
    n_core = n_body - NTERM_SIZE - CTERM_SIZE
    assert n_core >= 2 and n_core % 2 == 0, (
        f"Core length must be even >= 2, got {n_core}"
    )

    body   = codons[1:-1]   # codons 1 .. n-2  (all body codons)
    nterm_codons = body[:NTERM_SIZE]
    core_codons  = body[NTERM_SIZE : NTERM_SIZE + n_core]
    cterm_codons = body[NTERM_SIZE + n_core :]

    pos = 0   # running position counter

    # ── StartCodon ─────────────────────────────────────────────────────────────
    start_term = _terminal(codons[0], pos)
    start_node = ParseNode(label="StartCodon", span=(pos, pos+1),
                           is_terminal=False, children=(start_term,))
    pos += 1

    # ── NTerm (flat: NTERM_SIZE children) ──────────────────────────────────────
    nterm_ch = tuple(_body_codon(c, pos+i) for i, c in enumerate(nterm_codons))
    nterm_node = ParseNode(label="NTerm", span=(pos, pos+NTERM_SIZE),
                           is_terminal=False, children=nterm_ch)
    pos += NTERM_SIZE

    # ── Core (balanced binary tree) ────────────────────────────────────────────
    core_node = _build_core(core_codons, pos)
    pos += n_core

    # ── CTerm (flat: CTERM_SIZE children) ──────────────────────────────────────
    cterm_ch = tuple(_body_codon(c, pos+i) for i, c in enumerate(cterm_codons))
    cterm_node = ParseNode(label="CTerm", span=(pos, pos+CTERM_SIZE),
                           is_terminal=False, children=cterm_ch)
    pos += CTERM_SIZE

    # ── StopCodon ──────────────────────────────────────────────────────────────
    stop_term = _terminal(codons[-1], pos)
    stop_node = ParseNode(label="StopCodon", span=(pos, pos+1),
                          is_terminal=False, children=(stop_term,))

    # ── Assemble ORF (CSG rule) ────────────────────────────────────────────────
    orf_node = ParseNode(label="ORF", span=(1, n-1),
                         is_terminal=False,
                         children=(nterm_node, core_node, cterm_node))

    # ── Assemble CDS and Gene ──────────────────────────────────────────────────
    cds_node  = ParseNode(label="CDS",  span=(0, n), is_terminal=False,
                          children=(start_node, orf_node, stop_node))
    gene_node = ParseNode(label="Gene", span=(0, n), is_terminal=False,
                          children=(cds_node,))

    return gene_node


# ─────────────────────────────────────────────────────────────────────────────
# Text tree renderer
# ─────────────────────────────────────────────────────────────────────────────

def tree_to_text(node: ParseNode, prefix: str = "", is_last: bool = True) -> str:
    """
    Render a ParseNode tree as Unicode box-drawing text.

    Example output:
        Gene
        └── CDS
            ├── StartCodon
            │   └── [C0] ATG
            ├── ORF  [CSG]
            │   ├── NTerm
            │   │   ├── BodyCodon
            │   │   │   └── [C0] AAA
            ...
    """
    connector = "└── " if is_last else "├── "
    extension = "    " if is_last else "│   "

    if node.is_terminal:
        line = f"{prefix}{connector}[{node.label}] {node.raw_codon or ''}\n"
        return line

    # Label with annotation for special nodes
    label = node.label
    if label == "ORF":
        label += "  [CSG: ORF->NTerm Core CTerm, left_ctx=StartCodon]"
    elif label == "Core" and len(node.children) == 2:
        c0, c1 = node.children
        # Show span sizes for Core binary splits
        s0 = c0.span[1] - c0.span[0]
        s1 = c1.span[1] - c1.span[0]
        if not c0.is_terminal and not c1.is_terminal:
            if c0.label == "Core" and c1.label == "Core":
                label += f"  [{s0}+{s1} codons]"

    line = f"{prefix}{connector}{label}\n"

    children = node.children
    for i, child in enumerate(children):
        last_child = (i == len(children) - 1)
        line += tree_to_text(child, prefix + extension, last_child)

    return line


# ─────────────────────────────────────────────────────────────────────────────
# PNG rendering
# ─────────────────────────────────────────────────────────────────────────────

def render_png(root: ParseNode, seq: str, title: str, save_path: Path,
               n_codons: int) -> None:
    """
    Render the parse tree as PNG using draw_parse_tree.
    Automatically scales leaf spacing to keep figure manageable.
    """
    # Leaves = all is_terminal=True nodes = individual codons
    depth = root.depth

    # Leaf spacing: compress for large sequences so figure stays printable
    if n_codons <= 30:
        leaf_w = 1.2
    elif n_codons <= 60:
        leaf_w = 0.8
    elif n_codons <= 120:
        leaf_w = 0.45
    elif n_codons <= 250:
        leaf_w = 0.25
    else:
        leaf_w = 0.12

    w = max(10.0, n_codons * leaf_w)
    h = max(8.0,  (depth + 3) * 1.1)

    show_seq = n_codons <= 40   # sequence bar only readable for short seqs

    fig = draw_parse_tree(
        root,
        raw_seq=seq,
        figsize=(w, h),
        title=title,
        show_raw=show_seq,
        save_path=str(save_path),
        dpi=100,
    )
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    SEP = "=" * 68
    print(SEP)
    print("GE-CSG Domain Grammar Parse Trees -- GENCODE v49 Real Gene Sizes")
    print("Grammar: domain_grammar  (genuine CSG, balanced binary Core)")
    print(SEP)

    # ── Find real gene transcripts ────────────────────────────────────────────
    transcripts = find_real_transcripts(GTF_PATH, TARGETS)

    rng = random.Random(42)

    print(f"\n {'Target':>7}  {'Actual':>7}  {'Gene':<14}  {'Transcript':<24}")
    print("-" * 68)

    for target in TARGETS:
        gene_name, tid, actual = transcripts[target]

        # Ensure n_body is even (domain_grammar requirement)
        n_codons = actual
        n_body   = n_codons - 2
        if n_body % 2 != 0:
            n_codons -= 1   # trim by 1 codon (still very close)
            n_body    = n_codons - 2

        # Sanity: n_core must be even
        n_core = n_body - NTERM_SIZE - CTERM_SIZE
        if n_core < 2 or n_core % 2 != 0:
            print(f"  {target:>7}  SKIP (n_core={n_core} not usable)")
            continue

        # Generate synthetic CDS at real gene length
        seq   = synthetic_cds(n_codons, rng)
        root  = build_domain_tree(seq)
        depth = root.depth

        # File stems
        stem_base = f"DOM_{n_codons:04d}codon_{gene_name}"
        png_path  = OUT_DIR / f"{stem_base}.png"
        txt_path  = OUT_DIR / f"{stem_base}.txt"

        # Header info
        n_core_pairs = n_core // 2
        core_depth   = math.ceil(math.log2(max(n_core_pairs, 1)))

        title = (
            f"Domain Grammar  |  {gene_name} ({tid})  |  {n_codons} codons\n"
            f"NTerm={NTERM_SIZE}  Core={n_core} ({n_core_pairs} pairs, "
            f"depth={core_depth})  CTerm={CTERM_SIZE}  |  "
            f"Total tree depth={depth}"
        )

        # ── TXT output ────────────────────────────────────────────────────────
        txt_lines  = f"{'='*68}\n"
        txt_lines += f"Gene: {gene_name}  Transcript: {tid}\n"
        txt_lines += f"Total codons: {n_codons}  |  "
        txt_lines += f"NTerm={NTERM_SIZE}  Core={n_core}  CTerm={CTERM_SIZE}\n"
        txt_lines += f"Core pairs: {n_core_pairs}  |  Core depth: {core_depth}\n"
        txt_lines += f"Total tree depth: {depth}\n"
        txt_lines += f"Sequence (first 60 nt): {seq[:60]}...\n"
        txt_lines += "=" * 68 + "\n\n"
        txt_lines += tree_to_text(root, prefix="", is_last=True)

        txt_path.write_text(txt_lines, encoding="utf-8")
        txt_kb = txt_path.stat().st_size // 1024

        # ── PNG output ────────────────────────────────────────────────────────
        render_png(root, seq, title, png_path, n_codons)
        png_kb = png_path.stat().st_size // 1024

        print(f"  {target:>7}  {n_codons:>7}  {gene_name:<14}  {tid:<24}")
        print(f"    TXT -> {txt_path.name}  ({txt_kb} KB)")
        print(f"    PNG -> {png_path.name}  ({png_kb} KB)")
        print(f"    Tree depth={depth}  Core depth={core_depth}  "
              f"(prev. chain depth ~{n_body//2})")

    print(f"\nDone.  Outputs in: {OUT_DIR}/")
    print()
    print("Key: 'Core depth' = O(log n) balanced tree depth.")
    print("     'prev. chain depth' = O(n) that all previous grammars had.")


if __name__ == "__main__":
    main()
