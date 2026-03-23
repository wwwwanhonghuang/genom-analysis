"""
gecsg.grammar.domain_grammar
============================
Domain CSG Grammar — three-region protein structure with balanced binary Core.

Why previous grammars were chain-like
--------------------------------------
All previous grammars had the form  A → a A  or  A → a A a  where only ONE
child is a non-terminal recursion.  The resulting parse tree is a *linear
spine* of depth O(n), visually indistinguishable from a simple list.

How this grammar is genuinely branching
-----------------------------------------
The key rule is:

    Core → Core  Core        (binary split, CF)

At each level, BOTH children are non-trivial Core sub-trees spanning half the
remaining codons.  The resulting tree has depth O(log n), not O(n).

For n = 100 body-codons the Core subtree has ~6 levels; the tree looks like
an inverted triangle, NOT a diagonal staircase.

Grammar structure
-----------------
    Gene        →  CDS
    CDS         →  StartCodon  ORF  StopCodon

    [CSG]  ORF  →  NTerm  Core  CTerm      left_ctx = (StartCodon,)
           fires only when ORF is immediately after StartCodon NT.
           G acts trivially on NTs → orbit preserves context → genuine CSG.

    [CF]   NTerm  →  BodyCodon BodyCodon BodyCodon   (3-codon N-terminal)
    [CF]   CTerm  →  BodyCodon BodyCodon BodyCodon   (3-codon C-terminal)

    [CF]   Core  →  Core  Core      (binary split — the non-chain rule)
    [CF]   Core  →  BodyCodon  BodyCodon   (2-codon leaf pair)

    [CF]   StartCodon  →₁  C0   (A-starting)
    [CF]   StopCodon   →₁  C1   (T-starting)
    [CF]   BodyCodon   →₁  C0   (orbit → all 4 cosets)

Biological interpretation
--------------------------
NTerm  — N-terminal signal/localisation region (first ~3 codons)
Core   — main protein-coding domain, modelled as hierarchical codon-pair tree
CTerm  — C-terminal termination/anchor region (last ~3 codons)

Language accepted
-----------------
    n_body = n_codons - 2  (body = all codons except start and stop)
    Accepted iff  n_body ≥ 8  and  n_body is EVEN.

    (n_body = 3  [NTerm] + n_core  [Core] + 3  [CTerm];
     n_core ≥ 2, n_core even → n_body ≥ 8, even.)

Parse tree shape (10 body codons, 12 total)
--------------------------------------------
    Gene
    └── CDS
        ├── StartCodon  [ATG]
        ├── ORF          ← CSG: left_ctx=StartCodon ← 3 children (NOT a chain!)
        │   ├── NTerm
        │   │   ├── BodyCodon  [c1]
        │   │   ├── BodyCodon  [c2]
        │   │   └── BodyCodon  [c3]
        │   ├── Core            ← binary tree (depth = log₂(n_core/2))
        │   │   ├── Core
        │   │   │   ├── BodyCodon  [c4]
        │   │   │   └── BodyCodon  [c5]
        │   │   ├── Core
        │   │   │   ├── BodyCodon  [c6]
        │   │   │   └── BodyCodon  [c7]
        │   │   ├── Core
        │   │   │   ├── BodyCodon  [c8]
        │   │   │   └── BodyCodon  [c9]
        │   │   └── Core
        │   │       ├── BodyCodon  [c10]
        │   │       └── BodyCodon  [c11]
        │   └── CTerm
        │       ├── BodyCodon  [c12]
        │       ├── BodyCodon  [c13]
        │       └── BodyCodon  [c14]  ← wait, 3+8+3 = 14 needs 14 body codons
        └── StopCodon  [TAA]

Contrast with paired_chain_grammar for 50 body codons:
    ORF → Pair → ORF → Pair → ORF → ... (25 chain levels)   ← chain

Domain grammar for 50 body codons (3+44+3):
    ORF → NTerm(3) + Core(44) + CTerm(3)                   ← 3-way branch
    Core: balanced binary tree of depth ≈ log₂(22) ≈ 5     ← triangle
"""

from __future__ import annotations
from itertools import product as iproduct
from typing import Dict, Optional

from gecsg.core.group import Group
from gecsg.grammar.grammar import GECSGGrammar
from gecsg.builder.simple_builder import SimpleGECSGBuilder

NUC_TO_COSET: Dict[str, int] = {"A": 0, "T": 1, "G": 2, "C": 3}

#: Fixed N/C-terminal sizes in body codons
NTERM_SIZE = 3
CTERM_SIZE = 3


def domain_grammar(
    group:         Optional[Group] = None,
    nuc_to_coset:  Optional[Dict[str, int]] = None,
    cf_relaxation: bool = False,
) -> GECSGGrammar:
    """
    Build the Domain CSG Grammar.

    The CSG rule  ORF → NTerm Core CTerm  [left_ctx=StartCodon]
    uses a NonTerminal left context preserved under G-orbit expansion.

    Parameters
    ----------
    group         : ambient group G (default: dna_default_group(), order 12)
    nuc_to_coset  : first-nucleotide -> coset-index mapping
    cf_relaxation : if True, drop CSG left_ctx (for Earley / visualization)

    Returns
    -------
    GECSGGrammar (frozen)
    """
    from gecsg.core.dna_groups import dna_default_group

    G   = group or dna_default_group()
    H   = SimpleGECSGBuilder._z3_subgroup(G)
    n2c = nuc_to_coset or NUC_TO_COSET

    g = GECSGGrammar(group=G, subgroup_indices=H, start="Gene", k=3)

    # ── CF structural backbone ────────────────────────────────────────────────
    g.add_generation_rule("Gene", ["CDS"])
    g.add_generation_rule("CDS",  ["StartCodon", "ORF", "StopCodon"])

    # ── CSG rule: ORF → NTerm Core CTerm when after StartCodon NT ─────────────
    #
    # left_ctx = ["StartCodon"] — NonTerminal context.
    # G acts trivially on NTs  →  orbit of every rule member keeps same left_ctx.
    # Rule fires ONLY once: when ORF is immediately after StartCodon NT in the
    # sentential form.  After that, no remaining ORF NT has StartCodon to its
    # left, so inner recursion uses CF rules.
    ctx = [] if cf_relaxation else ["StartCodon"]
    g.add_generation_rule(
        "ORF",
        ["NTerm", "Core", "CTerm"],
        left_ctx=ctx,          # TRUE CSG when cf_relaxation=False
    )

    # ── CF: NTerm = exactly 3 BodyCodons (N-terminal flanking region) ─────────
    g.add_generation_rule("NTerm", ["BodyCodon", "BodyCodon", "BodyCodon"])

    # ── CF: CTerm = exactly 3 BodyCodons (C-terminal flanking region) ─────────
    g.add_generation_rule("CTerm", ["BodyCodon", "BodyCodon", "BodyCodon"])

    # ── CF: Core — the key non-chain rules ────────────────────────────────────
    #
    # Core → Core Core   (binary split: BOTH children are non-trivial)
    # Core → BC BC       (base: leaf codon pair)
    #
    # For a sequence of n_core codons, the custom balanced-tree builder
    # always splits at the midpoint, giving depth = ceil(log₂(n_core/2)).
    # This is visually a triangle / inverted-V, NOT a chain.
    g.add_generation_rule("Core", ["Core", "Core"])
    g.add_generation_rule("Core", ["BodyCodon", "BodyCodon"])

    # ── CF terminal productions ───────────────────────────────────────────────
    g.add_generation_rule("StartCodon", [0])   # C0: A-starting
    g.add_generation_rule("StopCodon",  [1])   # C1: T-starting
    g.add_generation_rule("BodyCodon",  [0])   # C0 orbit → all 4 cosets

    # ── Breaking rules: all 64 codons ─────────────────────────────────────────
    for n1, n2, n3 in iproduct("ATGC", repeat=3):
        g.add_breaking_rule(coset=n2c[n1], string=(n1, n2, n3))

    return g.freeze()
