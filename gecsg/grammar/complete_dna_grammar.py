"""
gecsg.grammar.complete_dna_grammar
====================================
Complete DNA grammar with full biological CDS structure.

Grammar structure (Phase-1 generation rules)
--------------------------------------------
  Gene         ->  CDS
  CDS          ->  StartCodon ORF StopCodon
  ORF          ->  BodyCodon
  ORF          ->  BodyCodon ORF

  StartCodon   ->_1 C0                     coset: A__ codons (e.g. ATG)
  StopCodon    ->_1 C1                     coset: T__ codons (e.g. TAA/TAG/TGA)
  BodyCodon    ->_1 C0                     orbit: all 4 cosets C0..C3

Coset assignment (Phase-2 breaking rules)
-----------------------------------------
  C0 <- A__ codons  (first base A, 16 codons)  -- includes ATG (Met/Start)
  C1 <- T__ codons  (first base T, 16 codons)  -- includes TAA, TAG, TGA (Stop)
  C2 <- G__ codons  (first base G, 16 codons)
  C3 <- C__ codons  (first base C, 16 codons)

G-equivariance and biological constraints
-----------------------------------------
G = Z2_comp x Z2_rev x Z3 acts transitively on the 4 cosets (G/G_i ≅ Z2xZ2).
Every single-coset generation rule  NT ->_1 Ci  therefore orbit-expands to
all 4 coset rules  NT ->_1 C0/C1/C2/C3.

Consequence: StartCodon and StopCodon accept any codon *class*, not only
ATG or TAA/TAG/TGA.  The biological specificity lives in the *coset
assignment* (ATG maps to C0 by the first-nucleotide rule) together with the
*structural grammar rule* (StartCodon occupies position 0, StopCodon the
last position).  Strict single-codon enforcement (e.g. ATG only) would
require per-NT lifting maps, which extends beyond the current formalism.

Comparison with dna_codon_grammar
----------------------------------
                  dna_codon_grammar         complete_dna_grammar
  Structure:      Gene->CDS->Codon          Gene->CDS->Start ORF Stop
  Start codon:    any codon                 C0 (A-starting) by coset
  Stop codon:     any codon                 C1 (T-starting) by coset
  Min CDS length: 1 codon                  3 codons
  NTs:            Gene, CDS, Codon          Gene, CDS, StartCodon,
                                            ORF, BodyCodon, StopCodon
  Orbit reps:     4                         7
  Full rules:     7                         16
"""

from __future__ import annotations
from itertools import product as iproduct
from typing import Dict, Optional

from gecsg.core.group import Group
from gecsg.grammar.grammar import GECSGGrammar
from gecsg.builder.simple_builder import SimpleGECSGBuilder


# Nucleotide -> coset index (first-base rule, shared with dna_grammar)
NUC_TO_COSET: Dict[str, int] = {"A": 0, "T": 1, "G": 2, "C": 3}

# Canonical stop codons (T__ that end translation)
STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})


def complete_dna_grammar(
    group:        Optional[Group] = None,
    nuc_to_coset: Optional[Dict[str, int]] = None,
) -> GECSGGrammar:
    """
    Build the complete DNA grammar with full CDS biological structure.

    Grammar
    -------
    Gene        -> CDS
    CDS         -> StartCodon ORF StopCodon
    ORF         -> BodyCodon | BodyCodon ORF
    StartCodon  ->_1 C0   (A-starting codon class)
    StopCodon   ->_1 C1   (T-starting codon class)
    BodyCodon   ->_1 C0   (all 4 coset classes via G-orbit)

    Returns
    -------
    GECSGGrammar (frozen)
    """
    from gecsg.core.dna_groups import dna_default_group

    G   = group or dna_default_group()
    H   = SimpleGECSGBuilder._z3_subgroup(G)
    n2c = nuc_to_coset or NUC_TO_COSET

    g = GECSGGrammar(group=G, subgroup_indices=H, start="Gene", k=3)

    # ── Structural rules (R1, orbit size = 1 each) ────────────────────────
    g.add_generation_rule("Gene",       ["CDS"])
    g.add_generation_rule("CDS",        ["StartCodon", "ORF", "StopCodon"])
    g.add_generation_rule("ORF",        ["BodyCodon"])
    g.add_generation_rule("ORF",        ["BodyCodon", "ORF"])

    # ── Terminal-production rules (R1, orbit size = 4 each) ───────────────
    # G acts transitively on {C0,C1,C2,C3}, so each single-coset rep
    # expands to all 4 coset rules.
    #
    # StartCodon ->_1 C0   orbit -> StartCodon->C0/C1/C2/C3
    #   Biological role: C0 = A-starting codons, canonical start is ATG
    g.add_generation_rule("StartCodon", [0])   # coset index 0 = C0

    # StopCodon ->_1 C1    orbit -> StopCodon->C0/C1/C2/C3
    #   Biological role: C1 = T-starting codons, canonical stops TAA/TAG/TGA
    g.add_generation_rule("StopCodon",  [1])   # coset index 1 = C1

    # BodyCodon ->_1 C0    orbit -> BodyCodon->C0/C1/C2/C3
    #   Accepts any codon class in the ORF body.
    g.add_generation_rule("BodyCodon",  [0])   # coset index 0 = C0

    # ── Breaking rules (R2, all 64 codons) ────────────────────────────────
    for n1, n2, n3 in iproduct("ATGC", repeat=3):
        g.add_breaking_rule(coset=n2c[n1], string=(n1, n2, n3))

    return g.freeze()


def describe_complete_grammar_stats(grammar: GECSGGrammar) -> None:
    """Print a structured summary of the complete grammar."""
    G  = grammar.group
    CS = grammar.coset_space
    print("-" * 56)
    print(f"Group            : {G.name}  (order {G.order})")
    print(f"|G/G_i|          : {CS.size}  (Phase-1 alphabet size)")
    print(f"k                : {grammar.k}  (codon level, 3 nucleotides)")
    print(f"Orbit reps R1/G  : {grammar.n_orbits}")
    print(f"Full rules |R1|  : {grammar.n_full_rules}")
    print(f"Breaking rules   : {grammar.n_breaking}  (all 64 codons)")
    print(f"Compression      : {grammar.compression_ratio():.1%} stored")
    print("-" * 56)
    print("Nonterminals     : Gene, CDS, StartCodon, ORF,")
    print("                   BodyCodon, StopCodon")
    print("Min CDS length   : 3 codons  (start + 1 body + stop)")
    print("-" * 56)
    print("Coset -> role mapping:")
    roles = {0: "Start  (e.g. ATG)",
             1: "Stop   (e.g. TAA/TAG/TGA)",
             2: "Body   (G-starting)",
             3: "Body   (C-starting)"}
    for nuc, idx in NUC_TO_COSET.items():
        coset = CS[idx]
        print(f"  {nuc}__ -> {coset}  {roles[idx]}")
    print("-" * 56)
