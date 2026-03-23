"""
gecsg.grammar.nested_cds_grammar
==================================
Nested CDS Grammar — the first GENUINE context-sensitive GE-CSG in this project.

Why previous versions were CFG, not CSG
-----------------------------------------
The rules  Body → BodyCodon BodyCodon  and  Body → BodyCodon Body BodyCodon
have empty left_ctx and right_ctx (α = β = ε).  That is the definition of a
context-free rule.  No matter how nested the structure looks, every rule that
expands only its own LHS without reading its neighbours is context-free.

Why THIS grammar is a true CSG
--------------------------------
A rule is context-sensitive iff α or β is non-empty in  αAβ →₁ αγβ.

The key CSG rule here uses a NonTerminal left context:

    Body → BodyCodon  Body  BodyCodon   [left_ctx = StartCodon]

This fires ONLY when the Body nonterminal appears immediately to the right of
the StartCodon nonterminal IN THE CURRENT SENTENTIAL FORM.  Without the wrapping
start context, Body uses only the base CF rule:

    Body → BodyCodon  BodyCodon          (no context — fires everywhere)

Why NonTerminal context survives G-orbit expansion
---------------------------------------------------
In generate_orbit (rule.py), the group G acts on sentential-form elements:

    if isinstance(elem, NonTerminal):
        result.append(elem)   # G acts TRIVIALLY on nonterminals

So g · NonTerminal("StartCodon") = NonTerminal("StartCodon") for every g ∈ G.
The orbit of the rule has left_ctx = (StartCodon,) in ALL orbit members.
The context is preserved — the rule remains genuinely context-sensitive.

Contrast: coset-based left_ctx (e.g. left_ctx=[C0]) would orbit-expand to all
four cosets {C0, C1, C2, C3}, making the rule fire regardless of context and
thus context-free.  NT-based context is the correct approach.

Grammar structure
-----------------
    Gene        →  CDS
    CDS         →  StartCodon  Body  StopCodon

    [CSG]  Body → BodyCodon  Body  BodyCodon   left_ctx=(StartCodon,)
           Fires ONLY when Body is immediately preceded by StartCodon NT.
           Produces 3 children: left-BC, nested Body, right-BC.

    [CF]   Body → BodyCodon  BodyCodon
           Fires anywhere — the base case (2-codon leaf pair).

    [CF]   StartCodon →₁ C0     (orbit: all A-starting codons)
    [CF]   StopCodon  →₁ C1     (orbit: all T-starting codons)
    [CF]   BodyCodon  →₁ C0     (orbit: all 4 cosets)

Language accepted (BFS)
-----------------------
  { start · w · stop  |  start ∈ C0, stop ∈ C1, |w| ∈ {2, 4} }

  For |w| = 2: CDS → Start (Body→BC BC) Stop
  For |w| = 4: CDS → Start (Body[CSG]→BC (Body→BC BC) BC) Stop
  For |w| > 4: The CSG rule fires once (at StartCodon level), producing
               an inner Body that must match the base case (2 codons).
               Longer bodies require additional grammar rules.

Parse tree for 4-body codons  (CDS = 6 codons total)
-----------------------------------------------------
  Gene
  └── CDS
      ├── StartCodon  (ATG, C0)
      ├── Body                          ← CSG rule fired (left = StartCodon NT)
      │   ├── BodyCodon  (AAA, C0)      ← left outer codon
      │   ├── Body                      ← inner Body, CF rule
      │   │   ├── BodyCodon  (GCT, C2)
      │   │   └── BodyCodon  (TTT, C1)
      │   └── BodyCodon  (CAG, C3)      ← right outer codon
      └── StopCodon   (TAA, C1)

  Contrast with complete_dna_grammar (right-recursive spine, all CF):
      ORF → BC ORF → BC BC ORF → BC BC BC ORF → ...  (pure chain)

Parser note
-----------
This grammar is designed for the two-phase BFS parser (TwoPhaseBFSParser),
which correctly checks NT-based left context in the sentential form.

The Earley parser works on the lifted coset sequence and cannot check NT
contexts; it treats CSG rules as CF and may find a different (CF) derivation.
For acceptance testing of this grammar, use BFS.
"""

from __future__ import annotations
from itertools import product as iproduct
from typing import Dict, Optional

from gecsg.core.group import Group
from gecsg.grammar.grammar import GECSGGrammar
from gecsg.builder.simple_builder import SimpleGECSGBuilder

NUC_TO_COSET: Dict[str, int] = {"A": 0, "T": 1, "G": 2, "C": 3}


def nested_cds_grammar(
    group:        Optional[Group] = None,
    nuc_to_coset: Optional[Dict[str, int]] = None,
    cf_relaxation: bool = False,
) -> GECSGGrammar:
    """
    Build the Nested CDS Grammar — a genuine GE-CSG.

    The CSG rule  Body → BodyCodon Body BodyCodon  [left_ctx=StartCodon]
    uses a NonTerminal left context.  Because G acts trivially on nonterminals,
    the G-orbit preserves this context: all orbit members have the same
    left_ctx=(StartCodon,).  The rule is genuinely context-sensitive.

    Parameters
    ----------
    group        : ambient group G (default: dna_default_group(), order 12)
    nuc_to_coset : first-nucleotide -> coset-index mapping

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
    g.add_generation_rule("CDS",  ["StartCodon", "Body", "StopCodon"])

    # ── CSG rule: Body wraps only when immediately after StartCodon NT ────────
    #
    # left_ctx=["StartCodon"]  means: this rule fires only when Body appears
    # immediately to the right of a StartCodon nonterminal in the sentential
    # form.  G acts trivially on NonTerminals, so the G-orbit of this rule
    # has left_ctx=(StartCodon,) for every orbit member — the context is
    # preserved and the rule is genuinely context-sensitive.
    #
    # Effect in parse tree: Body node has THREE children
    #   (BodyCodon_left, inner_Body, BodyCodon_right) — visible branching.
    # cf_relaxation=True drops the left_ctx so Earley (which cannot check NT
    # contexts against the lifted coset sequence) can still find parse trees.
    # Use cf_relaxation=False (default) for correctness; True for visualization.
    ctx = [] if cf_relaxation else ["StartCodon"]
    g.add_generation_rule("Body", ["BodyCodon", "Body", "BodyCodon"], left_ctx=ctx)

    # ── CF base rule: Body → BodyCodon BodyCodon (no context) ────────────────
    g.add_generation_rule("Body", ["BodyCodon", "BodyCodon"])

    # ── Terminal productions (orbit representatives) ──────────────────────────
    g.add_generation_rule("StartCodon", [0])   # C0: A-starting codons
    g.add_generation_rule("StopCodon",  [1])   # C1: T-starting codons
    g.add_generation_rule("BodyCodon",  [0])   # C0 orbit -> all 4 cosets

    # ── Breaking rules: all 64 codons ─────────────────────────────────────────
    for n1, n2, n3 in iproduct("ATGC", repeat=3):
        g.add_breaking_rule(coset=n2c[n1], string=(n1, n2, n3))

    return g.freeze()


def describe_nested_grammar_stats(grammar: GECSGGrammar) -> None:
    """Print a summary of the Nested CDS Grammar."""
    G  = grammar.group
    CS = grammar.coset_space
    print("-" * 62)
    print(f"Grammar          : Nested CDS Grammar  (genuine CSG)")
    print(f"Group            : {G.name}  (order {G.order})")
    print(f"|G/G_i|          : {CS.size}  (Phase-1 alphabet size)")
    print(f"k                : {grammar.k}  (codon level, 3 nucleotides)")
    print(f"Orbit reps R1/G  : {grammar.n_orbits}")
    print(f"Full rules |R1|  : {grammar.n_full_rules}")
    print(f"Breaking rules   : {grammar.n_breaking}  (all 64 codons)")
    print(f"Compression      : {grammar.compression_ratio():.1%} stored")
    print("-" * 62)
    print("Nonterminals     : Gene, CDS, StartCodon, Body,")
    print("                   BodyCodon, StopCodon")
    print()
    print("Rules:")
    print("  [CF]  Gene       -> CDS")
    print("  [CF]  CDS        -> StartCodon Body StopCodon")
    print("  [CSG] Body       -> BodyCodon Body BodyCodon")
    print("          left_ctx = (StartCodon,)  <-- NonTerminal context!")
    print("          G acts trivially on NTs: orbit preserves this context.")
    print("          Fires ONLY when Body follows StartCodon in sentential form.")
    print("  [CF]  Body       -> BodyCodon BodyCodon   (base, no context)")
    print("  [CF]  BodyCodon  ->_1 C0  (orbit: C0/C1/C2/C3)")
    print("  [CF]  StartCodon ->_1 C0")
    print("  [CF]  StopCodon  ->_1 C1")
    print()
    print("CSG trigger      : Body immediately after StartCodon NT")
    print("Tree effect      : Body -> BC  Body  BC  (3-child branching node)")
    print("Language (BFS)   : start + {2 or 4} body codons + stop")
    print("Parser           : use TwoPhaseBFSParser (BFS checks NT context)")
    print("-" * 62)
