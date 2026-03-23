"""
gecsg.grammar.center_branch_grammar
=====================================
Center-Branching CSG Grammar — produces genuinely NESTED trees, not chains.

Why this grammar is NOT chain-like
------------------------------------
complete_dna_grammar (chain):
    ORF → BodyCodon  ORF      (pure right-recursive — flat spine)

This grammar (nested):
    Body → BodyCodon  Body  BodyCodon    (center-branching — Russian doll)
    Body → BodyCodon  BodyCodon          (base case)

    Every Body internal node has THREE children: left-BC, inner-Body, right-BC.
    The tree has genuine center-branching at every level.
    Depth of nesting = n_body / 2.

Why this grammar is a genuine CSG
------------------------------------
    [CSG] Body → BodyCodon  Body  BodyCodon   left_ctx = (StartCodon,)
          Fires ONLY when Body appears immediately after the StartCodon
          nonterminal in the sentential form.  G acts trivially on NTs →
          the orbit preserves left_ctx → the rule is genuinely context-sensitive.

    [CF]  Body → BodyCodon  Body  BodyCodon   (no context — inner levels)
          Fires at every inner nesting level where Body follows BodyCodon.

    [CF]  Body → BodyCodon  BodyCodon          (base case — innermost pair)

The CSG rule fires for the outermost Body (immediately after StartCodon NT).
Inner Body nodes use the CF wrap rule (preceded by BodyCodon, not StartCodon).
The base case terminates the nesting.

Grammar structure
-----------------
    Gene        →  CDS
    CDS         →  StartCodon  Body  StopCodon

    [CSG]  Body → BodyCodon  Body  BodyCodon   left_ctx=(StartCodon,)
           Outermost wrap — fires when Body follows StartCodon NT.
           G acts trivially on NTs → orbit preserves context → genuine CSG.

    [CF]   Body → BodyCodon  Body  BodyCodon   (no context)
           Inner wraps — fires at all inner nesting levels.

    [CF]   Body → BodyCodon  BodyCodon
           Base case — innermost 2-codon pair.

    [CF]   StartCodon →₁ C0
    [CF]   StopCodon  →₁ C1
    [CF]   BodyCodon  →₁ C0   (orbit: all 4 cosets)

Language accepted
-----------------
    { start · w · stop  |  start ∈ C0,  stop ∈ C1,
      |w| is even,  |w| ≥ 2 }
    i.e. body lengths: 2, 4, 6, 8, 10, …

Parse tree shape (6 body codons)
----------------------------------
    Gene
    └── CDS
        ├── StartCodon  [ATG]
        ├── Body                          ← CSG rule (left=StartCodon NT)
        │   ├── BodyCodon  [GCT]          ← outer left codon
        │   ├── Body                      ← CF wrap rule (left=BodyCodon)
        │   │   ├── BodyCodon  [AAG]      ← inner left codon
        │   │   ├── Body                  ← CF base rule
        │   │   │   ├── BodyCodon  [TCA]
        │   │   │   └── BodyCodon  [TTT]
        │   │   └── BodyCodon  [CAG]      ← inner right codon
        │   └── BodyCodon  [GGC]          ← outer right codon
        └── StopCodon   [TAA]

Contrast with complete_dna_grammar (chain):
    ORF → BC ORF → BC BC ORF → BC BC BC ORF → ...  (flat right-recursive spine)

Contrast with paired_chain_grammar (fishbone):
    ORF → Pair ORF → ...  (right-recursive spine of Pairs, each Pair has 2 BC)

This grammar: each Body wraps around an inner Body — tree depth = n_body/2.
"""

from __future__ import annotations
from itertools import product as iproduct
from typing import Dict, Optional

from gecsg.core.group import Group
from gecsg.grammar.grammar import GECSGGrammar
from gecsg.builder.simple_builder import SimpleGECSGBuilder

NUC_TO_COSET: Dict[str, int] = {"A": 0, "T": 1, "G": 2, "C": 3}


def center_branch_grammar(
    group:         Optional[Group] = None,
    nuc_to_coset:  Optional[Dict[str, int]] = None,
    cf_relaxation: bool = False,
) -> GECSGGrammar:
    """
    Build the Center-Branching CSG Grammar.

    Key rule:
        [CSG]  Body -> BodyCodon Body BodyCodon  [left_ctx=StartCodon]
               Fires ONLY when Body is immediately after StartCodon NT.
               G acts trivially on NTs -> orbit preserves this context.

        [CF]   Body -> BodyCodon Body BodyCodon  (no context, inner levels)
        [CF]   Body -> BodyCodon BodyCodon        (base case)

    Parameters
    ----------
    group         : ambient group G (default: dna_default_group(), order 12)
    nuc_to_coset  : first-nucleotide -> coset-index mapping
    cf_relaxation : if True, treat the CSG rule as CF (drop left_ctx).
                    Use True for Earley (which cannot check NT-based contexts).
                    Use False for BFS (correct CSG semantics).

    Returns
    -------
    GECSGGrammar (frozen)
    """
    from gecsg.core.dna_groups import dna_default_group

    G   = group or dna_default_group()
    H   = SimpleGECSGBuilder._z3_subgroup(G)
    n2c = nuc_to_coset or NUC_TO_COSET

    g = GECSGGrammar(group=G, subgroup_indices=H, start="Gene", k=3)

    # ── CF structural rules ────────────────────────────────────────────────────
    g.add_generation_rule("Gene", ["CDS"])
    g.add_generation_rule("CDS",  ["StartCodon", "Body", "StopCodon"])

    # ── CSG rule: outermost Body wraps when immediately after StartCodon NT ───
    #
    # left_ctx=["StartCodon"]  -- NonTerminal context.
    # G acts trivially on NTs: generate_orbit leaves NonTerminals unchanged.
    # All orbit members have the same left_ctx=(StartCodon,).
    # The rule fires ONLY when Body follows StartCodon NT in the sentential form.
    #
    # cf_relaxation=True: drop left_ctx so Earley can find trees.
    # cf_relaxation=False (default): genuine CSG semantics for BFS.
    ctx = [] if cf_relaxation else ["StartCodon"]
    g.add_generation_rule(
        "Body",
        ["BodyCodon", "Body", "BodyCodon"],
        left_ctx=ctx,   # TRUE CSG when cf_relaxation=False
    )

    # ── CF inner wrap: Body wraps at inner levels (no context required) ────────
    # Only added in BFS/CSG mode; in cf_relaxation mode the above rule already
    # covers all levels (since its context was dropped).
    if not cf_relaxation:
        g.add_generation_rule("Body", ["BodyCodon", "Body", "BodyCodon"])

    # ── CF base rule: innermost pair (2 body codons) ───────────────────────────
    g.add_generation_rule("Body", ["BodyCodon", "BodyCodon"])

    # ── Terminal productions ───────────────────────────────────────────────────
    g.add_generation_rule("StartCodon", [0])   # C0: A-starting codons
    g.add_generation_rule("StopCodon",  [1])   # C1: T-starting codons
    g.add_generation_rule("BodyCodon",  [0])   # C0 orbit -> all 4 cosets

    # ── Breaking rules: all 64 codons ─────────────────────────────────────────
    for n1, n2, n3 in iproduct("ATGC", repeat=3):
        g.add_breaking_rule(coset=n2c[n1], string=(n1, n2, n3))

    return g.freeze()
