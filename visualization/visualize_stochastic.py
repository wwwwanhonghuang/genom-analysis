"""
visualize_stochastic.py
========================
Generate probability-annotated parse-tree images using the stochastic
complete DNA grammar.

Run:
    python visualize_stochastic.py
"""
import sys, os, io
_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gecsg.grammar.stochastic_dna_grammar import (
    stochastic_complete_dna_grammar,
    describe_stochastic_grammar_stats,
    HUMAN_CODON_USAGE,
)
from gecsg.parser.stochastic_earley import StochasticEarleyParser
from gecsg.visualize.prob_tree_viz import draw_stochastic_parse_tree

OUT_DIR = os.path.join(_ROOT, "outputs_stochastic")
os.makedirs(OUT_DIR, exist_ok=True)

P_TERMINAL = 0.2  # default: mean 5 body codons

print("Building stochastic grammar (uniform breaking probs)...")
GRAMMAR_UNIFORM = stochastic_complete_dna_grammar(p_terminal=P_TERMINAL)
PARSER_UNIFORM  = StochasticEarleyParser(GRAMMAR_UNIFORM)

print("Building stochastic grammar (human codon usage)...")
GRAMMAR_HUMAN   = stochastic_complete_dna_grammar(p_terminal=P_TERMINAL,
                                                   codon_usage=HUMAN_CODON_USAGE)
PARSER_HUMAN    = StochasticEarleyParser(GRAMMAR_HUMAN)

print()
describe_stochastic_grammar_stats(GRAMMAR_UNIFORM, P_TERMINAL)


def render(parser, seq, filename, title=None, *, codon_usage_label="uniform"):
    result = parser.parse(seq)
    if not result.accepted:
        print(f"  [SKIP]  {seq!r} not accepted")
        return
    tree   = result.trees()[0]
    nlp    = result.node_log_probs()
    path   = os.path.join(OUT_DIR, filename)
    t = title or f"Stochastic GE-CSG [{codon_usage_label}]: {seq}"
    fig = draw_stochastic_parse_tree(
        tree, seq, nlp, result.log_prob,
        title=t, save_path=path,
    )
    plt.close(fig)
    kb = os.path.getsize(path) // 1024
    print(f"  [OK]  {filename:<55} log_P={result.log_prob:8.3f}  ({kb} KB)")


# Section 1: canonical sequences with uniform probs
print("\n=== Section 1: Canonical CDS sequences (uniform breaking) ===")
seqs_canonical = [
    ("ATGGCTTAA",                      "S1_3codon_uniform.png",  "3-codon: ATG|GCT|TAA"),
    ("ATGAAAGCTTAA",                   "S1_4codon_uniform.png",  "4-codon: ATG|AAA|GCT|TAA"),
    ("ATGAAAGCTTTTCAGTAA",             "S1_6codon_uniform.png",  "6-codon"),
    ("ATGGCTAGCAAAGTTCGTCATGCATAA",   "S1_9codon_uniform.png",  "9-codon real fragment"),
]
for seq, fname, title in seqs_canonical:
    render(PARSER_UNIFORM, seq, fname, title=f"Uniform [{title}]")

# Section 2: same sequences with human codon usage
print("\n=== Section 2: Same sequences with human codon usage ===")
for seq, fname, title in seqs_canonical:
    hfname = fname.replace("uniform", "human")
    render(PARSER_HUMAN, seq, hfname, title=f"Human codon usage [{title}]",
           codon_usage_label="human")

# Section 3: vary p_terminal (short vs long ORFs)
print("\n=== Section 3: Effect of p_terminal on ORF probability ===")
seq_short = "ATG" + "AAA" + "TAA"           # 1 body codon
seq_mid   = "ATG" + "AAAGCT" + "TAA"        # 2 body codons
seq_long  = "ATG" + "AAAGCTTTTCAGAAAGCT" + "TAA"  # 6 body codons

for p in [0.1, 0.3, 0.5, 0.8]:
    g = stochastic_complete_dna_grammar(p_terminal=p)
    parser = StochasticEarleyParser(g)
    for seq, label in [(seq_short, "1body"), (seq_mid, "2body"), (seq_long, "6body")]:
        fname = f"S3_p{int(p*10):02d}_{label}.png"
        render(parser, seq, fname, title=f"p_terminal={p}: {seq}",
               codon_usage_label=f"p={p}")

# Section 4: homo-coset ORF (probability comparison across codons)
print("\n=== Section 4: Homo-coset ORFs (common vs rare codons) ===")
homo_seqs = [
    ("ATG" + "ATG"*3 + "TAA",   "S4_homo_ATG_x3.png",  "ATG x3 body (high usage)"),
    ("ATG" + "ATA"*3 + "TAA",   "S4_homo_ATA_x3.png",  "ATA x3 body (low usage)"),
    ("ATG" + "CTG"*3 + "TAA",   "S4_homo_CTG_x3.png",  "CTG x3 body (high usage)"),
    ("ATG" + "CGA"*3 + "TAA",   "S4_homo_CGA_x3.png",  "CGA x3 body (low usage)"),
]
for seq, fname, title in homo_seqs:
    render(PARSER_HUMAN, seq, fname, title=f"Human usage [{title}]",
           codon_usage_label="human")

# Section 5: rank sequences by probability
print("\n=== Section 5: Ranking sequences by log-probability ===")
candidates = [
    "ATG" + "CTG"*3 + "TAA",    # CTG: Leu, high usage
    "ATG" + "CGA"*3 + "TAA",    # CGA: Arg, low usage
    "ATG" + "GGC"*3 + "TAA",    # GGC: Gly, medium
    "ATG" + "TTT"*3 + "TAG",    # TTT: Phe + TAG stop
    "ATG" + "GCT"*3 + "TGA",    # GCT: Ala + TGA stop
]
ranked = PARSER_HUMAN.rank(candidates)
print("\n  Ranking (human codon usage):")
for i, (seq, lp) in enumerate(ranked, 1):
    print(f"  #{i}: log_P={lp:8.3f}  {seq}")

# Section 6: long CDS visualization
print("\n=== Section 6: Long CDS (12-codon) ===")
long_seq = "ATG" + "AAAGCTTTTCAGAAAGCTTTTCAG" + "TAA"  # 10 body codons
render(PARSER_UNIFORM, long_seq, "S6_12codon_uniform.png",
       title="12-codon CDS (uniform)")
render(PARSER_HUMAN, long_seq, "S6_12codon_human.png",
       title="12-codon CDS (human codon usage)")

print(f"\nDone. Images saved to: {OUT_DIR}\n")
