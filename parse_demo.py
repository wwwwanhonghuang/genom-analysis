"""
parse_demo.py
=============
End-to-end GE-CSG parsing demo.

Run:
    python parse_demo.py                  # all demos
    python parse_demo.py --seq ATGGCTTAA  # parse a custom sequence
    python parse_demo.py --demo 2         # run a specific demo

Demos
-----
1. Grammar inspection  — describe the full DNA codon grammar
2. Short parse         — ATGGCTTAA  (3 codons: ATG | GCT | TAA)
3. Longer parse        — ATGAAAGCTTTTGCCTAA  (6 codons)
4. Rejection           — XXXYYY  (unknown codons → rejected)
5. Orbit analysis      — show which cosets each input block lifts to
"""

import sys, os, argparse, io
sys.path.insert(0, os.path.dirname(__file__))
# Force UTF-8 output on Windows (avoids cp1252 encode errors for group names like Z2×Z3)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from gecsg.grammar.dna_grammar import dna_codon_grammar, describe_grammar_stats
from gecsg.parser.earley import EquivariantEarleyParser
from gecsg.visualize.tree_viz import draw_parse_tree

DIV  = "=" * 60
DIV2 = "-" * 60


# ─────────────────────────────────────────────────────────────────────────────
# Shared resources (built once)
# ─────────────────────────────────────────────────────────────────────────────

def get_grammar_and_parser():
    grammar = dna_codon_grammar()
    parser  = EquivariantEarleyParser(grammar)
    return grammar, parser


# ─────────────────────────────────────────────────────────────────────────────
# Demo 1 — Grammar inspection
# ─────────────────────────────────────────────────────────────────────────────

def demo1_grammar(grammar):
    print(f"\n{DIV}")
    print("Demo 1 — DNA Codon Grammar")
    print(DIV)
    describe_grammar_stats(grammar)
    print()
    grammar.describe(show_full_orbits=True)


# ─────────────────────────────────────────────────────────────────────────────
# Demo 2 — Short parse (3 codons)
# ─────────────────────────────────────────────────────────────────────────────

def demo2_short_parse(parser):
    print(f"\n{DIV}")
    print("Demo 2 — Short Parse  (ATGGCTTAA)")
    print(DIV)

    seq    = "ATGGCTTAA"
    result = parser.parse(seq)
    result.summary()

    if result.accepted:
        trees = result.trees()
        print(f"\nParse trees found: {len(trees)}")
        print()
        trees[0].pprint()

        fig = draw_parse_tree(
            trees[0], seq,
            title=f"GE-CSG parse tree: {seq}",
            save_path="parse_ATGGCTTAA.png",
        )
        print(f"\nSaved → parse_ATGGCTTAA.png")
        import matplotlib.pyplot as plt
        plt.close(fig)
    else:
        print("Sequence rejected — no parse tree.")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 3 — Longer parse (6 codons)
# ─────────────────────────────────────────────────────────────────────────────

def demo3_longer_parse(parser):
    print(f"\n{DIV}")
    print("Demo 3 — Longer Parse  (ATGAAAGCTTTTGCCTAA)")
    print(DIV)

    seq    = "ATGAAAGCTTTTGCCTAA"
    result = parser.parse(seq)
    result.summary()

    if result.accepted:
        trees = result.trees()
        print(f"\nParse trees found: {len(trees)}")
        print()
        trees[0].pprint()

        fig = draw_parse_tree(
            trees[0], seq,
            title=f"GE-CSG parse tree: {seq}",
            save_path="parse_ATGAAAGCTTTTGCCTAA.png",
        )
        print(f"\nSaved → parse_ATGAAAGCTTTTGCCTAA.png")
        import matplotlib.pyplot as plt
        plt.close(fig)
    else:
        print("Sequence rejected.")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 4 — Rejection
# ─────────────────────────────────────────────────────────────────────────────

def demo4_rejection(parser):
    print(f"\n{DIV}")
    print("Demo 4 — Rejection cases")
    print(DIV)

    cases = [
        ("ATGTAA",       "valid 2-codon sequence"),
        ("ATG",          "single codon — valid 1-codon CDS"),
        ("ATGTAATAA",    "valid 3-codon sequence"),
    ]
    for seq, note in cases:
        result = parser.parse(seq)
        status = "ACCEPTED [OK]" if result.accepted else "REJECTED [--]"
        print(f"  {seq:<22} [{note}]  →  {status}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 5 — Lifting map inspection
# ─────────────────────────────────────────────────────────────────────────────

def demo5_lifting(grammar):
    print(f"\n{DIV}")
    print("Demo 5 — Lifting map Λ(codon)  (sample codons)")
    print(DIV)

    sample_codons = [
        "ATG", "TAA", "TAG", "TGA",
        "GCT", "GCC", "TTT", "AAA",
        "CAG", "CGT",
    ]
    cs = grammar.coset_space
    print(f"\n{'Codon':<8}  {'Λ(codon)':<18}  orbit_size  raw_string")
    print(DIV2)
    for cod in sample_codons:
        block   = tuple(cod)
        cosets  = grammar.lift(block)
        if cosets:
            c     = cosets[0]
            osiz  = cs.orbit_size(c)
            label = str(c)
        else:
            label, osiz = "(none)", 0
        print(f"  {cod:<6}  →  {label:<16}  {osiz:<10}  {cod}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo 6 — Custom sequence
# ─────────────────────────────────────────────────────────────────────────────

def demo6_custom(parser, seq: str):
    print(f"\n{DIV}")
    print(f"Custom parse: {seq}")
    print(DIV)

    result = parser.parse(seq)
    result.summary()

    if result.accepted:
        trees = result.trees()
        print(f"\nParse trees: {len(trees)}")
        if trees:
            trees[0].pprint()
            safe = seq[:20].replace(" ", "_")
            path = f"parse_{safe}.png"
            fig  = draw_parse_tree(
                trees[0], seq,
                title=f"GE-CSG parse tree: {seq}",
                save_path=path,
            )
            print(f"\nSaved → {path}")
            import matplotlib.pyplot as plt
            plt.close(fig)
    else:
        print("Sequence rejected — not in L(grammar).")
        print("Check: sequence length must be a multiple of 3,")
        print("       and all bases must be in {A, T, G, C}.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="GE-CSG Parse Demo")
    ap.add_argument("--demo", type=int, default=0,
                    help="Run specific demo 1-5 (0 = all)")
    ap.add_argument("--seq",  type=str, default=None,
                    help="Parse a custom DNA sequence")
    args = ap.parse_args()

    print(f"\n{DIV}")
    print("  GE-CSG Parser Demo")
    print(DIV)
    print("Building grammar and parser...")
    grammar, parser = get_grammar_and_parser()
    print("Ready.\n")

    if args.seq:
        demo6_custom(parser, args.seq)
        return

    demos = {
        1: lambda: demo1_grammar(grammar),
        2: lambda: demo2_short_parse(parser),
        3: lambda: demo3_longer_parse(parser),
        4: lambda: demo4_rejection(parser),
        5: lambda: demo5_lifting(grammar),
    }

    if args.demo == 0:
        for fn in demos.values():
            fn()
    elif args.demo in demos:
        demos[args.demo]()
    else:
        print(f"Unknown demo {args.demo}. Choose 1-5.")

    print(f"\n{DIV}")
    print("Done.")
    print(DIV)


if __name__ == "__main__":
    main()
