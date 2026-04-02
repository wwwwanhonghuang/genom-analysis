from itertools import product as iproduct

from gecsg.grammar.grammar import GECSGGrammar
from gecsg.core.dna_groups import dna_default_group


def z3_subgroup(G):
    """
    Find a subgroup of order |G|/4.
    For dna_default_group() (order 12), this gives a subgroup of order 3,
    so the coset space has size 4.
    """
    target = G.order // 4
    if target <= 0:
        return [0]

    for start_idx in range(1, G.order):
        subgrp = [0]
        curr = start_idx
        while curr not in subgrp:
            subgrp.append(curr)
            curr = G.multiply(curr, start_idx)
        if len(subgrp) == target and G.is_subgroup(subgrp):
            return subgrp

    return [0]


def build_simple_gecsg_example():
    G = dna_default_group()
    H = z3_subgroup(G)

    # k=3 means each breaking rule maps a coset to a length-3 raw string
    g = GECSGGrammar(
        group=G,
        subgroup_indices=H,
        start="Gene",
        k=3,
    )

    # -------------------------
    # Generation rules (R1/G)
    # -------------------------

    # Gene -> CDS
    g.add_generation_rule("Gene", ["CDS"])

    # CDS -> Start Body Stop
    g.add_generation_rule("CDS", ["StartCodon", "Body", "StopCodon"])

    # Body -> BodyCodon BodyCodon   (base rule)
    g.add_generation_rule("Body", ["BodyCodon", "BodyCodon"])

    # Context-sensitive rule:
    #   StartCodon Body StopCodon  =>  StartCodon BodyCodon Body StopCodon
    #
    # In this API, the rewritten symbol is lhs="Body",
    # and the contexts are given separately.
    g.add_generation_rule(
        lhs="Body",
        rhs=["BodyCodon", "Body"],
        left_ctx=["StartCodon"],
        right_ctx=["StopCodon"],
    )

    # Terminal-producing generation rules:
    # use coset indices directly in RHS
    g.add_generation_rule("StartCodon", [0])
    g.add_generation_rule("StopCodon", [1])
    g.add_generation_rule("BodyCodon", [2])

    # -------------------------
    # Breaking rules (R2)
    # -------------------------
    #
    # Minimal simple version:
    # one codon string per coset
    #
    # Since k=3, each breaking string must have length 3.
    g.add_breaking_rule(0, "ATG")   # start-like
    g.add_breaking_rule(1, "TAA")   # stop-like
    g.add_breaking_rule(2, "GCT")   # body-like
    g.add_breaking_rule(3, "GGT")   # unused extra coset, but valid to include

    return g.freeze()


if __name__ == "__main__":
    g = build_simple_gecsg_example()
    print(g)
    g.describe(show_full_orbits=False)