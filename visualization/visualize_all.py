"""
visualize_all.py
================
Generate parse-tree PNG images using the COMPLETE DNA grammar.

Complete grammar structure
--------------------------
  Gene        -> CDS
  CDS         -> StartCodon  ORF  StopCodon
  ORF         -> BodyCodon | BodyCodon ORF

  StartCodon  ->_1 C0   (A-starting codons, canonical: ATG)
  StopCodon   ->_1 C1   (T-starting codons, canonical: TAA / TAG / TGA)
  BodyCodon   ->_1 C0   (orbit: C0 C1 C2 C3 -- all codon classes)

Coset assignment (first nucleotide)
  C0  A__  -- Start class   (includes ATG)
  C1  T__  -- Stop class    (includes TAA, TAG, TGA)
  C2  G__  -- Body class G
  C3  C__  -- Body class C

All sequences below conform to the biological CDS constraint:
  ATG (or other A-starting codon)  +  >= 1 body codon  +  TAA/TAG/TGA

Run:
    python visualize_all.py
"""

import sys, os, io
_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gecsg.grammar.complete_dna_grammar import complete_dna_grammar, describe_complete_grammar_stats
from gecsg.parser.earley import EquivariantEarleyParser
from gecsg.visualize.tree_viz import draw_parse_tree

# ── Setup ──────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

print("Building complete grammar...")
GRAMMAR = complete_dna_grammar()
PARSER  = EquivariantEarleyParser(GRAMMAR)

print()
describe_complete_grammar_stats(GRAMMAR)
print()

# Coset-class representatives
START   = "ATG"    # C0 canonical start codon
STOP_A  = "TAA"    # C1 canonical stop (amber)
STOP_R  = "TAG"    # C1 canonical stop (ochre)
STOP_O  = "TGA"    # C1 canonical stop (opal / umber)

# One body-codon representative per coset class
BODY_A  = "AAA"    # C0  Lys (A-starting body)
BODY_T  = "TTT"    # C1  Phe (T-starting body)
BODY_G  = "GCT"    # C2  Ala (G-starting body)
BODY_C  = "CAG"    # C3  Gln (C-starting body)

POOL    = [BODY_A, BODY_G, BODY_T, BODY_C]   # cycles through all 4 body cosets


def cds(start, *body_codons, stop):
    """Build a CDS string: start + body + stop."""
    return start + "".join(body_codons) + stop


def body_seq(n):
    """Cycling body sequence of n codons (A G T C ...)."""
    return "".join(POOL[i % 4] for i in range(n))


def render(seq: str, filename: str, title: str | None = None) -> None:
    """Parse seq with complete grammar and save parse tree to outputs/<filename>."""
    result = PARSER.parse(seq)
    if not result.accepted:
        print(f"  [SKIP]  {seq!r} not accepted by complete grammar")
        return
    tree = result.trees()[0]
    path = os.path.join(OUT_DIR, filename)
    t = title or f"GE-CSG complete parse: {seq}"
    fig = draw_parse_tree(tree, seq, title=t, save_path=path)
    plt.close(fig)
    kb = os.path.getsize(path) // 1024
    print(f"  [OK]    {filename:<52}  ({kb} KB)  {seq}")


# =============================================================================
# Section 1 -- Grammar overview
# =============================================================================
print("=" * 60)
print("Section 1  Grammar overview (single body codon per class)")
print("=" * 60)

# One body codon from each coset class; TAA stop
for label, bc in [("C0_Lys", BODY_A), ("C1_Phe", BODY_T),
                  ("C2_Ala", BODY_G), ("C3_Gln", BODY_C)]:
    seq = cds(START, bc, stop=STOP_A)
    render(seq, f"S1_body_{label}_{seq}.png",
           f"S1 | body codon {label}: {bc}  (CDS={seq})")

# Three different stop codons with same body
print()
for stop_label, stop in [("TAA", STOP_A), ("TAG", STOP_R), ("TGA", STOP_O)]:
    seq = cds(START, BODY_G, stop=stop)
    render(seq, f"S1_stop_{stop_label}_{seq}.png",
           f"S1 | stop codon {stop_label}: CDS={seq}")


# =============================================================================
# Section 2 -- Short CDS (1-4 body codons)
# =============================================================================
print()
print("=" * 60)
print("Section 2  Short CDS  (1-4 body codons, 3-6 codons total)")
print("=" * 60)

for n_body in range(1, 5):
    seq = START + body_seq(n_body) + STOP_A
    render(seq, f"S2_{n_body}body_{seq[:24]}.png",
           f"S2 | {n_body} body codon(s): {seq}")

# Mixed coset bodies
render(cds(START, BODY_A, BODY_G, stop=STOP_A),
       f"S2_2body_C0C2_{START+BODY_A+BODY_G+STOP_A}.png",
       f"S2 | body C0+C2: ATG|AAA|GCT|TAA")
render(cds(START, BODY_T, BODY_C, stop=STOP_R),
       f"S2_2body_C1C3_{START+BODY_T+BODY_C+STOP_R}.png",
       f"S2 | body C1+C3: ATG|TTT|CAG|TAG")
render(cds(START, BODY_A, BODY_T, BODY_G, stop=STOP_O),
       f"S2_3body_C0C1C2_{START+BODY_A+BODY_T+BODY_G+STOP_O}.png",
       f"S2 | body C0+C1+C2: ATG|AAA|TTT|GCT|TGA")


# =============================================================================
# Section 3 -- Real gene fragments (biologically meaningful CDSs)
# =============================================================================
print()
print("=" * 60)
print("Section 3  Real gene fragments")
print("=" * 60)

# Classic short CDS: Met + Ala + Stop
render("ATGGCTTAA",
       "S3_ATG_GCT_TAA.png",
       "S3 | ATG(Met) GCT(Ala) TAA(Stop) -- 3-codon CDS")

# Met + 4 body codons + Stop
render("ATGAAAGCTTTTGCCTAA",
       "S3_6codon_ATGAAAGCTTTTGCCTAA.png",
       "S3 | 6-codon: ATG|AAA|GCT|TTT|GCC|TAA")

# ATG + 8 body + TAA  (real E.coli-style fragment)
render("ATGGCTAGCAAAGTTCGTCATGCATAA",
       "S3_9codon_real_fragment.png",
       "S3 | 9-codon real fragment: ATG-GCT-AGC-AAA-GTT-CGT-CAT-GCA-TAA")

# Longer realistic fragment
render("ATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTAA",
       "S3_27codon_GFP_fragment.png",
       "S3 | 27-codon GFP-like fragment (ATG...TAA)")


# =============================================================================
# Section 4 -- Medium CDS (5-10 body codons, 7-12 total)
# =============================================================================
print()
print("=" * 60)
print("Section 4  Medium CDS  (5-10 body codons)")
print("=" * 60)

for n_body in range(5, 11):
    seq = START + body_seq(n_body) + STOP_A
    render(seq, f"S4_{n_body}body_{seq[:18]}_etc.png",
           f"S4 | {n_body} body codons (total {n_body+2} codons)")


# =============================================================================
# Section 5 -- Long CDS (15-30 body codons)
# =============================================================================
print()
print("=" * 60)
print("Section 5  Long CDS  (15-30 body codons, stress + performance)")
print("=" * 60)

for n_body in [15, 20, 28]:
    seq = START + body_seq(n_body) + STOP_A
    render(seq, f"S5_{n_body}body.png",
           f"S5 | {n_body} body codons  (total {n_body+2} codons)")


# =============================================================================
# Section 6 -- Homo-coset body (all body codons from same class)
# =============================================================================
print()
print("=" * 60)
print("Section 6  Homo-coset ORF  (all body codons in same coset)")
print("=" * 60)

for coset_label, bc, stop in [
    ("C0_A", BODY_A, STOP_A),
    ("C1_T", BODY_T, STOP_R),
    ("C2_G", BODY_G, STOP_O),
    ("C3_C", BODY_C, STOP_A),
]:
    for n_body in [3, 5]:
        seq = START + bc * n_body + stop
        render(seq, f"S6_homo_{coset_label}_x{n_body}_{seq[:18]}.png",
               f"S6 | homo-{coset_label} body x{n_body}: {seq}")


# =============================================================================
# Section 7 -- All four coset classes in body
# =============================================================================
print()
print("=" * 60)
print("Section 7  All four coset classes represented in ORF body")
print("=" * 60)

# Exactly one codon per body class
seq_4class = cds(START, BODY_A, BODY_T, BODY_G, BODY_C, stop=STOP_A)
render(seq_4class, f"S7_all4cosets_{seq_4class}.png",
       f"S7 | all 4 coset classes in body: ATG|AAA|TTT|GCT|CAG|TAA")

# Two codons from each class (interleaved)
seq_8body = cds(START,
                BODY_A, BODY_G, BODY_T, BODY_C,
                BODY_A, BODY_G, BODY_T, BODY_C,
                stop=STOP_A)
render(seq_8body, f"S7_8body_2x4classes_{seq_8body[:18]}_etc.png",
       f"S7 | 8 body codons, 2x each class")

# Representative codons for each coset
print()
for codon, ci, aa in [
    ("ATG", 0, "Met/Start"),  ("AAA", 0, "Lys"),
    ("TTT", 1, "Phe"),        ("TAC", 1, "Tyr"),
    ("GCT", 2, "Ala"),        ("GGG", 2, "Gly"),
    ("CAG", 3, "Gln"),        ("CTG", 3, "Leu"),
]:
    seq = cds(START, codon, stop=STOP_A)
    render(seq, f"S7_coset{ci}_{codon}_{aa.replace('/','_')}.png",
           f"S7 | body coset C{ci}: {codon} ({aa})")


# =============================================================================
# Section 8 -- Tree structure: spans, coset assignment
# =============================================================================
print()
print("=" * 60)
print("Section 8  Tree structure and coset assignment")
print("=" * 60)

# The canonical 3-codon CDS -- shows Gene/CDS/StartCodon/ORF/BodyCodon/StopCodon
render("ATGGCTTAA",
       "S8_canonical_ATGGCTTAA.png",
       "S8 | canonical 3-codon CDS: ATG(C0)|GCT(C2)|TAA(C1)")

# Mixed coset body -- shows 4 different coset leaf labels
seq_mixed = cds(START, BODY_A, BODY_T, BODY_G, BODY_C, stop=STOP_A)
render(seq_mixed, "S8_mixed_cosets_6codon.png",
       "S8 | mixed body cosets: ATG(C0)|AAA(C0)|TTT(C1)|GCT(C2)|CAG(C3)|TAA(C1)")

# Multiple body codons showing right-recursive ORF spine
render(START + body_seq(5) + STOP_A,
       "S8_orf_spine_7codon.png",
       "S8 | 7-codon CDS showing ORF right-recursive spine")


# =============================================================================
# Section 9 -- Alternate stop codons
# =============================================================================
print()
print("=" * 60)
print("Section 9  All three stop codons  (TAA / TAG / TGA)")
print("=" * 60)

body3 = body_seq(3)   # 3 body codons
for stop_name, stop_codon in [("TAA", STOP_A), ("TAG", STOP_R), ("TGA", STOP_O)]:
    seq = START + body3 + stop_codon
    render(seq, f"S9_stop_{stop_name}_{seq}.png",
           f"S9 | stop={stop_name}: {seq}")


# =============================================================================
# Section 10 -- Alternate start codons (other A-starting, same C0 coset)
# =============================================================================
print()
print("=" * 60)
print("Section 10  Alternate start codons (A-starting, coset C0)")
print("=" * 60)

body1 = body_seq(1)
for start_codon, note in [
    ("ATG", "canonical Met"),
    ("AAA", "Lys  (same C0)"),
    ("AGC", "Ser  (same C0)"),
    ("ACA", "Thr  (same C0)"),
]:
    seq = cds(start_codon, body_seq(2), stop=STOP_A)
    render(seq, f"S10_start_{start_codon}_{note.split()[0]}.png",
           f"S10 | start={start_codon} ({note}): {seq}")


# =============================================================================
# Section 11 -- Case insensitive input
# =============================================================================
print()
print("=" * 60)
print("Section 11  Case-insensitive input")
print("=" * 60)

render("atggcttaa",   "S11_lowercase.png",   "S11 | lowercase: atggcttaa -> ATGGCTTAA")
render("AtGgCtTaA",   "S11_mixedcase.png",   "S11 | mixed-case: AtGgCtTaA -> ATGGCTTAA")
render("ATGGCTTAA".lower() * 1,
       "S11_lower_3codon.png",   "S11 | lowercase 3-codon")


# =============================================================================
# Section 12 -- Multi-coset ORF demonstration (coset-level parse)
# =============================================================================
print()
print("=" * 60)
print("Section 12  Coset-level orbit demonstration")
print("=" * 60)

# Codon pairs to show that each codon pair belongs to a coset orbit
for label, cod1, cod2 in [
    ("C0_pair", "ATG", "AAA"),    # both C0
    ("C1_pair", "TTT", "TAC"),    # both C1
    ("C2_pair", "GCT", "GGG"),    # both C2
    ("C3_pair", "CAG", "CTG"),    # both C3
]:
    seq = cds(START, cod1, cod2, stop=STOP_A)
    render(seq, f"S12_orbit_{label}_{seq}.png",
           f"S12 | orbit demo {label}: {cod1},{cod2} same coset")


# =============================================================================
# Section 13 -- Long synthetic CDS (50 / 100 / 200 codons)
# Uses Earley parser (O(n^3)) so scale is not a problem.
# Body codons are drawn uniformly from all 64 codons for leaf variety.
#
# NOTE: The right-spine tree shape is a grammar property — the CDS grammar
# is right-recursive (ORF → BodyCodon | BodyCodon ORF), so every parse tree
# is a right-leaning chain regardless of sequence.  The variety here is in
# the leaf labels (codon strings and coset classes), not the tree topology.
# =============================================================================
import random as _random

_BASES    = "ATGC"
_ALL64    = [a + b + c for a in _BASES for b in _BASES for c in _BASES]
_STOPS    = ["TAA", "TAG", "TGA"]
_STARTS   = [c for c in _ALL64 if c[0] == "A"]   # all C0 (A-starting) codons

def random_body(n: int, seed: int) -> str:
    """n random body codons drawn uniformly from all 64 codons."""
    rng = _random.Random(seed)
    return "".join(rng.choice(_ALL64) for _ in range(n))

def random_start(seed: int) -> str:
    return _random.Random(seed).choice(_STARTS)

def random_stop(seed: int) -> str:
    return _random.Random(seed).choice(_STOPS)

print()
print("=" * 60)
print("Section 13  Long synthetic CDS  (50 / 100 / 200 codons)")
print("  Body codons: uniform random from all 64  (seed fixed per size)")
print("=" * 60)

for n_body, seed in [(50, 101), (100, 202), (200, 303)]:
    start = random_start(seed)
    stop  = random_stop(seed + 1)
    body  = random_body(n_body, seed + 2)
    seq   = start + body + stop
    render(seq, f"S13_{n_body}body_{n_body+2}codon_random.png",
           f"S13 | {n_body} random body codons (total {n_body+2}, start={start} stop={stop})")


# =============================================================================
print(f"\nDone.  All images saved to:  {OUT_DIR}\n")
