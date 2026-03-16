"""
gecsg.grammar.dna_grammar
=========================
Complete DNA codon grammar for GE-CSG parsing.

Grammar structure
-----------------
Generation rules (R1/G, 4 orbit reps):
    Gene  ->₁ CDS
    CDS   ->₁ Codon
    CDS   ->₁ Codon CDS
    Codon ->₁ C0          ← one rep; G-equivariant orbit gives C0,C1,C2,C3

Breaking rules (R2, all 64 codons):
    Coset assignment by first nucleotide:
        C0 ← A__ codons (AAA, AAT, ..., ATT)   16 codons
        C1 ← T__ codons (TAA, TAT, ..., TTT)   16 codons
        C2 ← G__ codons (GAA, GAT, ..., GTT)   16 codons
        C3 ← C__ codons (CAA, CAT, ..., CTT)   16 codons

Why first-nucleotide assignment?
    Simple, unambiguous, and biologically motivated: the first base of a codon
    is least degenerate in the genetic code, making it the natural coset
    discriminator for Phase-1 parsing.

Usage
-----
    from gecsg.grammar.dna_grammar import dna_codon_grammar
    grammar = dna_codon_grammar()
    grammar.describe()
"""

from __future__ import annotations
from itertools import product as iproduct
from typing import Optional, Dict

from gecsg.core.group import Group
from gecsg.grammar.grammar import GECSGGrammar
from gecsg.builder.simple_builder import SimpleGECSGBuilder


# Nucleotide -> coset index (by first base of codon)
NUC_TO_COSET: Dict[str, int] = {"A": 0, "T": 1, "G": 2, "C": 3}


def dna_codon_grammar(
    group:        Optional[Group] = None,
    nuc_to_coset: Optional[Dict[str, int]] = None,
) -> GECSGGrammar:
    """
    Build the complete DNA codon grammar with all 64 codons.

    Parameters
    ----------
    group        : ambient group G (default: dna_default_group(), order 12)
    nuc_to_coset : mapping from first nucleotide to coset index 0-3.
                   Default: {"A":0, "T":1, "G":2, "C":3}

    Returns
    -------
    GECSGGrammar (frozen)
    """
    from gecsg.core.dna_groups import dna_default_group

    G   = group or dna_default_group()
    H   = SimpleGECSGBuilder._z3_subgroup(G)
    n2c = nuc_to_coset or NUC_TO_COSET

    g = GECSGGrammar(group=G, subgroup_indices=H, start="Gene", k=3)

    # -- Generation rules (R1) ---------------------------------------------
    # Structural rules (all NTs in RHS -> orbit size 1 each)
    g.add_generation_rule("Gene",  ["CDS"])
    g.add_generation_rule("CDS",   ["Codon"])
    g.add_generation_rule("CDS",   ["Codon", "CDS"])

    # Terminal-production rule: Codon -> [C0]
    # int 0 resolves to CS[0] inside add_generation_rule.
    # G acts transitively on the 4 cosets, so orbit expansion gives:
    #   Codon ->₁ C0, Codon ->₁ C1, Codon ->₁ C2, Codon ->₁ C3
    g.add_generation_rule("Codon", [0])

    # -- Breaking rules (R2) -----------------------------------------------
    for n1, n2, n3 in iproduct("ATGC", repeat=3):
        coset_idx = n2c[n1]
        g.add_breaking_rule(coset=coset_idx, string=(n1, n2, n3))

    return g.freeze()


def describe_grammar_stats(grammar: GECSGGrammar) -> None:
    """Print a compact summary of the grammar."""
    G  = grammar.group
    CS = grammar.coset_space
    print("-" * 50)
    print(f"Group          : {G.name}  (order {G.order})")
    print(f"|G/G_i|        : {CS.size}  (Phase-1 alphabet)")
    print(f"k              : {grammar.k}  (codon level)")
    print(f"Orbit reps R1/G: {grammar.n_orbits}")
    print(f"Full rules |R1|: {grammar.n_full_rules}")
    print(f"Breaking |R2|  : {grammar.n_breaking}  (all 64 codons)")
    print(f"Compression    : {grammar.compression_ratio():.1%} of rules stored")
    print("-" * 50)
    print("Coset -> first-nucleotide mapping:")
    for nuc, idx in NUC_TO_COSET.items():
        coset = CS[idx]
        print(f"  {nuc}__ codons -> {coset}  (coset index {idx})")
    print("-" * 50)
