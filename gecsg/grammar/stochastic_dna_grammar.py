"""
gecsg.grammar.stochastic_dna_grammar
=====================================
Stochastic (probabilistic) GE-CSG grammar for DNA CDS sequences.

Same structure as complete_dna_grammar but with probability weights on all
generation rules and breaking rules.

Probability model
-----------------
Generation rules (R1):
  Gene -> CDS                        weight=1.0   (only rule for Gene)
  CDS  -> StartCodon ORF StopCodon   weight=1.0   (only rule for CDS)
  ORF  -> BodyCodon                  weight=p_terminal   (terminate)
  ORF  -> BodyCodon ORF              weight=1-p_terminal (extend)

  StartCodon ->_1 C0  weight=0.25  (G-equivariant orbit: each coset 1/4)
  StopCodon  ->_1 C1  weight=0.25
  BodyCodon  ->_1 C0  weight=0.25

Breaking rules (R2):
  Default (uniform): each codon within its coset has prob 1/16.
  Custom: pass a dict {codon_str: float} (must sum to 1.0 per coset class).

ORF length distribution:
  P(n body codons) = p_terminal * (1 - p_terminal)^(n-1)   [Geometric(p)]
  Mean body codons = 1/p_terminal.  Default p=0.2 -> mean 5 body codons.
"""

from __future__ import annotations
from itertools import product as iproduct
from typing import Dict, Optional

from gecsg.core.dna_groups import dna_default_group
from gecsg.grammar.grammar import GECSGGrammar
from gecsg.builder.simple_builder import SimpleGECSGBuilder

NUC_TO_COSET: Dict[str, int] = {"A": 0, "T": 1, "G": 2, "C": 3}

# Human codon usage (approximate, normalized per coset class A/T/G/C)
# Source: adapted from Kazusa human codon usage table
HUMAN_CODON_USAGE: Dict[str, float] = {
    # C0  A__ (16 codons, sum=1)
    "AAA": 0.075, "AAT": 0.048, "AAG": 0.084, "AAC": 0.058,
    "ATA": 0.018, "ATT": 0.043, "ATG": 0.065, "ATC": 0.056,
    "AGA": 0.029, "AGT": 0.025, "AGG": 0.030, "AGC": 0.043,
    "ACA": 0.035, "ACT": 0.027, "ACG": 0.014, "ACC": 0.038,
    # C1  T__ (16 codons, sum=1)
    "TAA": 0.029, "TAT": 0.047, "TAG": 0.020, "TAC": 0.060,
    "TTA": 0.016, "TTT": 0.051, "TTG": 0.033, "TTC": 0.065,
    "TGA": 0.051, "TGT": 0.046, "TGG": 0.068, "TGC": 0.060,
    "TCA": 0.031, "TCT": 0.034, "TCG": 0.012, "TCC": 0.043,
    # C2  G__ (16 codons, sum=1)
    "GAA": 0.067, "GAT": 0.052, "GAG": 0.095, "GAC": 0.071,
    "GTA": 0.018, "GTT": 0.031, "GTG": 0.076, "GTC": 0.040,
    "GGA": 0.041, "GGT": 0.029, "GGG": 0.042, "GGC": 0.053,
    "GCA": 0.039, "GCT": 0.041, "GCG": 0.012, "GCC": 0.067,
    # C3  C__ (16 codons, sum=1)
    "CAA": 0.033, "CAT": 0.035, "CAG": 0.093, "CAC": 0.046,
    "CTA": 0.014, "CTT": 0.033, "CTG": 0.115, "CTC": 0.052,
    "CGA": 0.012, "CGT": 0.012, "CGG": 0.029, "CGC": 0.030,
    "CCA": 0.048, "CCT": 0.047, "CCG": 0.015, "CCC": 0.049,
}

# Normalize each coset to sum exactly to 1.0 (floating-point safety)
def _normalize_codon_usage(usage: Dict[str, float]) -> Dict[str, float]:
    result = {}
    for nuc in "ATGC":
        codons = [nuc+n2+n3 for n2 in "ATGC" for n3 in "ATGC"]
        s = sum(usage[c] for c in codons if c in usage)
        if s > 0:
            for c in codons:
                result[c] = usage.get(c, 0.0) / s
    return result

HUMAN_CODON_USAGE = _normalize_codon_usage(HUMAN_CODON_USAGE)


def stochastic_complete_dna_grammar(
    p_terminal:   float = 0.2,
    codon_usage:  Optional[Dict[str, float]] = None,
    group=None,
) -> GECSGGrammar:
    """
    Build and return the stochastic complete DNA grammar.

    Parameters
    ----------
    p_terminal  : probability that ORF terminates after each body codon.
                  P(ORF length = n body codons) = p * (1-p)^(n-1).
                  Default 0.2 -> mean 5 body codons.
    codon_usage : {codon: float} breaking-rule probabilities, normalized per
                  coset class.  If None, uses uniform (1/16 per codon).
    group       : ambient group G. Default: dna_default_group().

    Returns
    -------
    GECSGGrammar (frozen) with weighted rules.
    """
    if not 0.0 < p_terminal < 1.0:
        raise ValueError(f"p_terminal must be in (0, 1), got {p_terminal}")

    G   = group or dna_default_group()
    H   = SimpleGECSGBuilder._z3_subgroup(G)
    n2c = NUC_TO_COSET
    g   = GECSGGrammar(group=G, subgroup_indices=H, start="Gene", k=3)

    # ── Generation rules ─────────────────────────────────────────────────
    # Structural (orbit size 1 each; only possibility for their LHS, so weight=1)
    g.add_generation_rule("Gene", ["CDS"],                            weight=1.0)
    g.add_generation_rule("CDS",  ["StartCodon", "ORF", "StopCodon"], weight=1.0)

    # ORF length: geometric with parameter p_terminal
    g.add_generation_rule("ORF", ["BodyCodon"],        weight=p_terminal)
    g.add_generation_rule("ORF", ["BodyCodon", "ORF"], weight=1.0 - p_terminal)

    # Terminal-production rules: orbit size 4 (G transitive on cosets)
    # Each orbit member has weight 0.25 -> sum over 4 rules = 1.0
    g.add_generation_rule("StartCodon", [0], weight=0.25)   # C0 orbit rep
    g.add_generation_rule("StopCodon",  [1], weight=0.25)   # C1 orbit rep
    g.add_generation_rule("BodyCodon",  [0], weight=0.25)   # C0 orbit rep

    # ── Breaking rules ────────────────────────────────────────────────────
    usage = codon_usage or {n1+n2+n3: 1.0/16.0
                            for n1 in "ATGC" for n2 in "ATGC" for n3 in "ATGC"}
    for n1, n2, n3 in iproduct("ATGC", repeat=3):
        codon = n1 + n2 + n3
        prob  = usage.get(codon, 1.0/16.0)
        g.add_breaking_rule(coset=n2c[n1], string=(n1, n2, n3), prob=prob)

    return g.freeze()


def describe_stochastic_grammar_stats(grammar: GECSGGrammar, p_terminal: float) -> None:
    """Print a concise summary of the stochastic grammar."""
    import math
    G  = grammar.group
    CS = grammar.coset_space
    print("-" * 58)
    print(f"Stochastic Complete DNA Grammar")
    print(f"  Group          : {G.name}  (order {G.order})")
    print(f"  |G/G_i|        : {CS.size}")
    print(f"  Orbit reps R1/G: {grammar.n_orbits}")
    print(f"  Full rules |R1|: {grammar.n_full_rules}")
    print(f"  Breaking rules : {grammar.n_breaking}")
    print(f"  p_terminal     : {p_terminal}  (mean ORF body = {1/p_terminal:.1f} codons)")
    print(f"  Mean CDS len   : {1/p_terminal + 2:.1f} codons  (start+body+stop)")
    print("-" * 58)
    # Verify normalization
    from collections import defaultdict
    nt_sums: Dict = defaultdict(float)
    for r in grammar.full_rules:
        nt_sums[r.lhs.name] += r.weight
    print("Generation rule weight sums per LHS (should be 1.0):")
    for nt, s in sorted(nt_sums.items()):
        print(f"  {nt:<14} {s:.6f}  {'OK' if abs(s-1.0)<1e-9 else 'FAIL'}")
    coset_sums: Dict = defaultdict(float)
    for br in grammar.breaking_rules:
        coset_sums[br.coset.index] += br.prob
    print("Breaking rule prob sums per coset (should be 1.0):")
    for ci in sorted(coset_sums):
        s = coset_sums[ci]
        print(f"  C{ci}             {s:.6f}  {'OK' if abs(s-1.0)<1e-9 else 'FAIL'}")
    print("-" * 58)
