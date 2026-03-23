"""
gecsg.grammar.paired_chain_grammar
=====================================
Paired-Chain CSG Grammar — genuine CSG with branching trees for any CDS length.

Why this grammar produces branching trees (unlike complete_dna_grammar)
-----------------------------------------------------------------------
complete_dna_grammar:
    ORF → BodyCodon  ORF          pure right-recursive chain
    parse tree: ORF → BC → ORF → BC → ORF → ...  (flat spine)

This grammar:
    ORF  → Pair  ORF              ORF node always has TWO children: Pair + ORF
    ORF  → Pair                   base case: one Pair
    Pair → BodyCodon  BodyCodon   Pair ALWAYS has two children (visible branching)

    Every ORF internal node branches into (Pair, ORF).
    Every Pair branches into (BodyCodon, BodyCodon).
    The tree has genuine branching at every level.

Why this grammar is a genuine CSG
-----------------------------------
The CSG rule:

    Pair → BodyCodon BodyCodon BodyCodon BodyCodon   [left_ctx = StartCodon]

fires ONLY when a Pair nonterminal appears immediately after a StartCodon
nonterminal in the current sentential form.  This makes the FIRST Pair in
the ORF a Quad (4 BodyCodons) while all subsequent Pairs remain 2-codon.

Because G acts trivially on NonTerminals, the G-orbit of this rule has
left_ctx = (StartCodon,) in every orbit member — the context is preserved
and the rule is genuinely context-sensitive.

Grammar structure
-----------------
    Gene        →  CDS
    CDS         →  StartCodon  ORF  StopCodon

    ORF         →  Pair                          (CF: base — single pair)
    ORF         →  Pair  ORF                     (CF: chain of pairs)

    [CSG]  Pair → BodyCodon BodyCodon BodyCodon BodyCodon
                  left_ctx = (StartCodon,)
           First Pair (immediately after StartCodon) becomes a Quad (4 BCs).
           G acts trivially on NTs → orbit preserves context → genuine CSG.

    [CF]   Pair → BodyCodon  BodyCodon
           All other Pairs: exactly 2 body codons.

    [CF]   StartCodon →₁ C0
    [CF]   StopCodon  →₁ C1
    [CF]   BodyCodon  →₁ C0   (orbit: all 4 cosets)

Language accepted (BFS)
-----------------------
    { start · w · stop  |  start ∈ C0,  stop ∈ C1,
      |w| = 4 + 2k  for k ≥ 0  }
    i.e. body lengths: 4, 6, 8, 10, …  (even, ≥ 4)

    The first Pair consumes 4 codons (CSG Quad), all others consume 2.

    For real CDS with n_codons total (including start/stop):
      n_body = n_codons - 2
      Accepted iff  n_body ≥ 4  and  n_body is even.

Parse tree shape (6 body codons, 8 total)
-----------------------------------------
    Gene
    └── CDS
        ├── StartCodon  [ATG]
        ├── ORF
        │   ├── Pair  ← CSG Quad (4 children, left=StartCodon)
        │   │   ├── BodyCodon  [GCT]
        │   │   ├── BodyCodon  [AAG]
        │   │   ├── BodyCodon  [TCA]
        │   │   └── BodyCodon  [TTT]
        │   └── ORF
        │       └── Pair  ← CF Pair (2 children)
        │           ├── BodyCodon  [CAG]
        │           └── BodyCodon  [GGC]
        └── StopCodon   [TAA]

Contrast with complete_dna_grammar for the same sequence:
    ORF → BC → ORF → BC → ORF → BC → ORF → BC → ORF → BC → ORF → BC
    (flat chain, no Pair nodes, no branching)
"""

from __future__ import annotations
from itertools import product as iproduct
from typing import Dict, Optional

from gecsg.core.group import Group
from gecsg.grammar.grammar import GECSGGrammar
from gecsg.builder.simple_builder import SimpleGECSGBuilder

NUC_TO_COSET: Dict[str, int] = {"A": 0, "T": 1, "G": 2, "C": 3}


def paired_chain_grammar(
    group:         Optional[Group] = None,
    nuc_to_coset:  Optional[Dict[str, int]] = None,
    cf_relaxation: bool = False,
) -> GECSGGrammar:
    """
    Build the Paired-Chain CSG Grammar.

    The CSG rule  Pair → BodyCodon BodyCodon BodyCodon BodyCodon  [left_ctx=StartCodon]
    uses a NonTerminal left context preserved under G-orbit expansion.

    Parameters
    ----------
    group         : ambient group G (default: dna_default_group(), order 12)
    nuc_to_coset  : first-nucleotide -> coset-index mapping
    cf_relaxation : if True, drop the CSG left_ctx so Earley can find trees
                    (Earley cannot check NT-based contexts against coset input)

    Returns
    -------
    GECSGGrammar (frozen)
    """
    from gecsg.core.dna_groups import dna_default_group

    G   = group or dna_default_group()
    H   = SimpleGECSGBuilder._z3_subgroup(G)
    n2c = nuc_to_coset or NUC_TO_COSET

    g = GECSGGrammar(group=G, subgroup_indices=H, start="Gene", k=3)

    # ── CF structural rules ───────────────────────────────────────────────────
    g.add_generation_rule("Gene", ["CDS"])
    g.add_generation_rule("CDS",  ["StartCodon", "ORF", "StopCodon"])

    # ORF: chain of Pairs (right-recursive)
    g.add_generation_rule("ORF",  ["Pair"])
    g.add_generation_rule("ORF",  ["Pair", "ORF"])

    # ── CSG rule: first Pair becomes a Quad when after StartCodon NT ──────────
    #
    # left_ctx=["StartCodon"] — NonTerminal context.
    # G acts trivially on NTs → all orbit members have the same left_ctx.
    # The rule fires ONLY when Pair is immediately after StartCodon in the
    # sentential form (checked by BFS; Earley uses cf_relaxation mode).
    ctx = [] if cf_relaxation else ["StartCodon"]
    g.add_generation_rule(
        "Pair",
        ["BodyCodon", "BodyCodon", "BodyCodon", "BodyCodon"],
        left_ctx=ctx,     # TRUE CSG when cf_relaxation=False
    )

    # ── CF base rule: Pair → BodyCodon BodyCodon (no context) ────────────────
    g.add_generation_rule("Pair", ["BodyCodon", "BodyCodon"])

    # ── Terminal productions (orbit representatives) ──────────────────────────
    g.add_generation_rule("StartCodon", [0])   # C0: A-starting
    g.add_generation_rule("StopCodon",  [1])   # C1: T-starting
    g.add_generation_rule("BodyCodon",  [0])   # C0 orbit -> all 4 cosets

    # ── Breaking rules: all 64 codons ─────────────────────────────────────────
    for n1, n2, n3 in iproduct("ATGC", repeat=3):
        g.add_breaking_rule(coset=n2c[n1], string=(n1, n2, n3))

    return g.freeze()
