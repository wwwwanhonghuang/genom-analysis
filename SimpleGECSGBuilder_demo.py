"""
SimpleGECSGBuilder_demo.py
==========================
Demonstrates the grammar module step by step.

Run:
    python SimpleGECSGBuilder_demo.py
    python SimpleGECSGBuilder_demo.py --demo 2
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

from gecsg.core.dna_groups import (
    dna_default_group, Z2ComplementGroup,
    Z2ReversalGroup, Z3CyclicGroup, DirectProductGroup
)
from gecsg.grammar.grammar import GECSGGrammar
from gecsg.builder.simple_builder import SimpleGECSGBuilder

DIV = "═" * 62


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1 — Inspect the group and coset space
# ─────────────────────────────────────────────────────────────────────────────

def demo1_coset_space():
    print(f"\n{DIV}")
    print("Demo 1 — Group and Coset Space")
    print(DIV)

    G = dna_default_group()
    print(f"\nGroup: {G}")
    print(f"Order: {G.order}")
    print(f"Elements (first 6): {G.elements[:6]}")

    # Default subgroup: Z3 subgroup (order 3 => |G/H| = 4)
    H_idx = SimpleGECSGBuilder._z3_subgroup(G)
    print(f"\nSubgroup G_i indices: {H_idx}  (order {len(H_idx)})")
    print(f"|G/G_i| = {G.order // len(H_idx)}  (Phase-1 alphabet size)")

    CS = G.coset_space(H_idx)
    print()
    CS.describe()

    print("\nOrbit sizes for each coset:")
    for c in CS.cosets:
        orb = CS.orbit_of_coset(c)
        print(f"  {c}: orbit_size={len(orb)}, "
              f"members={[str(x) for x in orb]}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2 — Build a grammar manually with SimpleGECSGBuilder
# ─────────────────────────────────────────────────────────────────────────────

def demo2_manual_build():
    print(f"\n{DIV}")
    print("Demo 2 — Manual Grammar Build (SimpleGECSGBuilder)")
    print(DIV)

    G     = dna_default_group()
    H_idx = SimpleGECSGBuilder._z3_subgroup(G)

    print(f"\nBuilding grammar with G={G.name}, |G/G_i|={G.order//len(H_idx)}, k=3")
    print("─" * 40)

    g = (SimpleGECSGBuilder(start="Gene", group=G,
                            subgroup_indices=H_idx, k=3)
         # Structural generation rules (non-terminals only in RHS)
         .rule("Gene",  ["CDS"])
         .rule("CDS",   ["Codon"])
         .rule("CDS",   ["Codon", "CDS"])
         # Breaking rules: one representative per coset orbit
         .break_coset(0, "ATG")   # start codon orbit
         .break_coset(1, "TAA")   # stop codon orbit
         .break_coset(2, "GCT")   # Ala-family orbit
         .break_coset(3, "GGT")   # Gly-family orbit
         .build()
    )

    g.describe(show_full_orbits=True)

    # Show key statistics
    print(f"\nKey statistics:")
    print(f"  Orbit reps (R1/G)    : {g.n_orbits}")
    print(f"  Full rules |R1|      : {g.n_full_rules}")
    print(f"  Breaking rules |R2|  : {g.n_breaking}")
    print(f"  Compression          : {g.compression_ratio():.1%} stored")
    print(f"  Torch NT vocab       : {g.nonterminal_vocab}")
    print(f"  Torch coset vocab    : {g.coset_vocab}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3 — Pre-defined grammars
# ─────────────────────────────────────────────────────────────────────────────

def demo3_predefined():
    print(f"\n{DIV}")
    print("Demo 3 — Pre-defined Grammars")
    print(DIV)

    grammars = {
        "DNA Codon Grammar": SimpleGECSGBuilder.dna_codon_grammar(),
        "Abstract Z2 Grammar": SimpleGECSGBuilder.abstract_z2_grammar(),
    }

    for name, g in grammars.items():
        print(f"\n{'─'*10} {name} {'─'*10}")
        g.describe()


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4 — Lifting map Λ and Λ*
# ─────────────────────────────────────────────────────────────────────────────

def demo4_lifting():
    print(f"\n{DIV}")
    print("Demo 4 — Lifting Map  Λ(w)  and  Λ*(w)")
    print(DIV)

    g = SimpleGECSGBuilder.dna_codon_grammar()
    CS = g.coset_space

    test_blocks = [("A","T","G"), ("T","A","C"), ("T","A","A"),
                   ("A","A","A"), ("G","G","G")]

    print(f"\n{'Block':<10} {'Λ(w)':<20} {'Λ*(w)':<20} {'orbit_size'}")
    print("─" * 62)
    for block in test_blocks:
        candidates = g.lift(block)
        opt        = g.lift_star(block)
        opt_sizes  = [CS.orbit_size(c) for c in opt]
        print(f"{''.join(block):<10} "
              f"{str([str(c) for c in candidates]):<20} "
              f"{str([str(c) for c in opt]):<20} "
              f"{opt_sizes}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5 — Orbit analysis on codons
# ─────────────────────────────────────────────────────────────────────────────

def demo5_orbit_analysis():
    print(f"\n{DIV}")
    print("Demo 5 — Codon Orbit Analysis (G acting on Σ_raw^3)")
    print(DIV)

    from itertools import product
    G     = dna_default_group()
    bases = ["A","T","G","C"]
    codons = [tuple(c) for c in product(bases, repeat=3)]

    visited = set()
    orbits  = []
    for codon in codons:
        if codon in visited:
            continue
        orbit = set(tuple(G.act_on_sequence(g.index, list(codon)))
                    for g in G.elements)
        orbits.append(sorted(orbit))
        visited.update(orbit)

    print(f"\nGroup: {G}  (order={G.order})")
    print(f"Total codons  : {len(codons)}")
    print(f"Orbit count   : {len(orbits)}")
    from collections import Counter
    size_dist = Counter(len(o) for o in orbits)
    print(f"Orbit sizes   : {dict(sorted(size_dist.items()))}")
    mean_size = sum(len(o) for o in orbits) / len(orbits)
    print(f"Mean orbit k̄  : {mean_size:.2f}")

    print(f"\nLargest orbits:")
    for o in sorted(orbits, key=len, reverse=True)[:3]:
        print(f"  size={len(o)}: {[''.join(c) for c in o]}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 6 — Custom group: Z2_comp only (order 2)
# ─────────────────────────────────────────────────────────────────────────────

def demo6_custom_group():
    print(f"\n{DIV}")
    print("Demo 6 — Custom Group: Z2_comp (order 2)")
    print(DIV)

    G     = Z2ComplementGroup()
    H_idx = [0]   # trivial subgroup -> |G/G_i| = 2

    g = (SimpleGECSGBuilder(start="S", group=G,
                            subgroup_indices=H_idx, k=2)
         .rule("S", ["A", "B"])
         .rule("A", ["A"])
         .rule("B", ["B"])
         .break_coset(0, "AT")   # coset {e} -> AT  (orbit: AT, TA, GC, CG)
         .break_coset(1, "GC")   # coset {c} -> GC
         .build()
    )

    print(f"\nGroup   : {G}")
    print(f"Subgroup: trivial {{e}}")
    print(f"|G/G_i|  : {G.order // 1} = 2")
    g.describe(show_full_orbits=True)


# ─────────────────────────────────────────────────────────────────────────────
# Demo 7 — Torch vocabulary hooks
# ─────────────────────────────────────────────────────────────────────────────

def demo7_torch_vocab():
    print(f"\n{DIV}")
    print("Demo 7 — Torch Vocabulary Hooks")
    print(DIV)

    g = SimpleGECSGBuilder.dna_codon_grammar()

    print(f"\nNonTerminal vocab (for embedding table):")
    for name, idx in g.nonterminal_vocab.items():
        print(f"  {name:10s} -> {idx}")

    print(f"\nCoset vocab (Phase-1 token IDs):")
    for coset_idx, token_id in g.coset_vocab.items():
        c = g.coset_space[coset_idx]
        print(f"  coset {coset_idx} ({c}) -> token_id {token_id}")

    print(f"\nRule orbit vocab (for rule embedding table):")
    for rule_id, embed_idx in g.rule_vocab.items():
        rep = g.orbits[rule_id].representative
        print(f"  orbit {rule_id}: '{rep}'  -> embed_idx {embed_idx}")

    print(f"\nTorch-compatible feature dimensions:")
    print(f"  NT embedding dim  : {len(g.nonterminal_vocab)}")
    print(f"  Coset embedding   : {len(g.coset_vocab)}")
    print(f"  Rule embedding    : {len(g.rule_vocab)}")
    print(f"  Breaking rules    : {g.n_breaking}  (|R2| = |Σ|)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SimpleGECSGBuilder Demo")
    parser.add_argument("--demo", type=int, default=0,
                        help="Run specific demo 1-7 (0 = all)")
    args = parser.parse_args()

    demos = {
        1: demo1_coset_space,
        2: demo2_manual_build,
        3: demo3_predefined,
        4: demo4_lifting,
        5: demo5_orbit_analysis,
        6: demo6_custom_group,
        7: demo7_torch_vocab,
    }

    if args.demo == 0:
        for fn in demos.values():
            fn()
    elif args.demo in demos:
        demos[args.demo]()
    else:
        print(f"Unknown demo {args.demo}. Choose 1-7.")

    print(f"\n{DIV}")
    print("All demos complete.")
    print(DIV)


if __name__ == "__main__":
    main()
