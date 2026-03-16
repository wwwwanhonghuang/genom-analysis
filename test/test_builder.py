"""
test_builder.py
===============
Tests for SimpleGECSGBuilder and the underlying gecsg layer.

Run:
    conda activate genom
    python test_builder.py
    python test_builder.py -v        # verbose
    python test_builder.py -k group  # run tests matching "group"

Coverage
--------
T01  Group properties       -- order, identity, closure, inverses
T02  Group action           -- Z2_comp, Z2_rev, Z3 cyclic, DirectProduct
T03  Coset space            -- size, partition, subgroup validation
T04  Coset G-action         -- act(), orbit_of_coset(), orbit_size()
T05  Orbit-maximising lift  -- CosetSpace.orbit_maximising_coset()
T06  Builder fluent API     -- rule(), break_coset(), build()
T07  Grammar properties     -- orbits, full_rules, breaking, compression
T08  Orbit generation       -- generate_orbit() expands correctly
T09  Lifting map            -- grammar.lift() and grammar.lift_star()
T10  Pre-defined grammars   -- dna_codon_grammar, abstract_z2_grammar
T11  Freeze guard           -- adding rules to frozen grammar raises
T12  Auto-break             -- auto_break_from_alphabet() assigns cosets
T13  Default subgroup       -- |G/G_i| == 4 for dna_default_group
T14  Torch vocab hooks      -- nonterminal_vocab, coset_vocab, rule_vocab
T15  Z2 palindrome grammar  -- palindrome_grammar() smoke test
"""

import sys, os, unittest
_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)

from gecsg.core.group import GroupElement
from gecsg.core.dna_groups import (
    dna_default_group, Z2ComplementGroup,
    Z2ReversalGroup, Z3CyclicGroup, DirectProductGroup
)
from gecsg.core.coset import Coset, CosetSpace
from gecsg.grammar.grammar import GECSGGrammar
from gecsg.grammar.rule import GenerationRule, BreakingRule, generate_orbit
from gecsg.grammar.symbols import NonTerminal
from gecsg.builder.simple_builder import SimpleGECSGBuilder


# ─────────────────────────────────────────────────────────────────────────────
# T01  Group properties
# ─────────────────────────────────────────────────────────────────────────────

class T01_GroupProperties(unittest.TestCase):

    def setUp(self):
        self.G = dna_default_group()

    def test_order(self):
        self.assertEqual(self.G.order, 12)

    def test_identity_index(self):
        self.assertEqual(self.G.identity.index, 0)

    def test_closure(self):
        G = self.G
        for i in range(G.order):
            for j in range(G.order):
                k = G.multiply(i, j)
                self.assertGreaterEqual(k, 0)
                self.assertLess(k, G.order)

    def test_identity_is_neutral(self):
        G = self.G
        for i in range(G.order):
            self.assertEqual(G.multiply(0, i), i)
            self.assertEqual(G.multiply(i, 0), i)

    def test_inverses(self):
        G = self.G
        for i in range(G.order):
            inv = G.inverse(i)
            self.assertEqual(G.multiply(i, inv), 0)
            self.assertEqual(G.multiply(inv, i), 0)

    def test_z2_comp_order(self):
        G = Z2ComplementGroup()
        self.assertEqual(G.order, 2)

    def test_z3_order(self):
        G = Z3CyclicGroup(k=3, n=3)
        self.assertEqual(G.order, 3)


# ─────────────────────────────────────────────────────────────────────────────
# T02  Group action on sequences
# ─────────────────────────────────────────────────────────────────────────────

class T02_GroupAction(unittest.TestCase):

    def test_z2_comp_identity(self):
        G = Z2ComplementGroup()
        seq = ["A", "T", "G", "C"]
        self.assertEqual(G.act_on_sequence(0, seq), seq)

    def test_z2_comp_complement(self):
        G = Z2ComplementGroup()
        self.assertEqual(G.act_on_sequence(1, ["A", "T", "G", "C"]),
                         ["T", "A", "C", "G"])

    def test_z2_comp_involution(self):
        G = Z2ComplementGroup()
        seq = ["A", "G", "C"]
        result = G.act_on_sequence(1, G.act_on_sequence(1, seq))
        self.assertEqual(result, seq)

    def test_z2_rev_reversal(self):
        G = Z2ReversalGroup()
        self.assertEqual(G.act_on_sequence(1, ["A", "T", "G"]),
                         ["G", "T", "A"])

    def test_z3_cyclic_shift(self):
        G = Z3CyclicGroup(k=3, n=3)
        seq = ["A", "T", "G"]
        self.assertEqual(G.act_on_sequence(1, seq), ["T", "G", "A"])
        self.assertEqual(G.act_on_sequence(2, seq), ["G", "A", "T"])
        # 3-shift = identity
        self.assertEqual(G.act_on_sequence(0, seq), seq)

    def test_direct_product_action(self):
        G = DirectProductGroup(Z2ComplementGroup(), Z2ReversalGroup())
        # element (c, r): complement then reverse
        # index of (c=1, r=1) = 1*2 + 1 = 3
        seq = ["A", "T"]
        result = G.act_on_sequence(3, seq)
        # Z2_comp: A->T, T->A => [T,A]; Z2_rev reverses => [A,T]
        self.assertEqual(result, ["A", "T"])

    def test_orbit_representative_is_minimal(self):
        G = dna_default_group()
        codon = ("A", "T", "G")
        rep = G.orbit_representative(codon)
        orbit = G.orbit_of_sequence(codon)
        self.assertEqual(rep, min(orbit))


# ─────────────────────────────────────────────────────────────────────────────
# T03  Coset space construction
# ─────────────────────────────────────────────────────────────────────────────

class T03_CosetSpace(unittest.TestCase):

    def setUp(self):
        self.G = dna_default_group()
        self.H = SimpleGECSGBuilder._z3_subgroup(self.G)
        self.CS = self.G.coset_space(self.H)

    def test_size(self):
        self.assertEqual(self.CS.size, 4)

    def test_partition(self):
        # All 12 group elements covered exactly once
        covered = set()
        for c in self.CS.cosets:
            for m in c.members:
                self.assertNotIn(m, covered, "Element in two cosets")
                covered.add(m)
        self.assertEqual(covered, set(range(self.G.order)))

    def test_invalid_subgroup_raises(self):
        with self.assertRaises(ValueError):
            self.G.coset_space([0, 1])   # {e, c} is not always a valid subgroup

    def test_coset_indexing(self):
        for i, c in enumerate(self.CS.cosets):
            self.assertEqual(c.index, i)
            self.assertEqual(self.CS[i], c)

    def test_coset_of(self):
        # Every group element maps to some coset
        for g_idx in range(self.G.order):
            c = self.CS.coset_of(g_idx)
            self.assertIn(g_idx, c.members)


# ─────────────────────────────────────────────────────────────────────────────
# T04  G-action on cosets
# ─────────────────────────────────────────────────────────────────────────────

class T04_CosetAction(unittest.TestCase):

    def setUp(self):
        self.G  = dna_default_group()
        self.H  = SimpleGECSGBuilder._z3_subgroup(self.G)
        self.CS = self.G.coset_space(self.H)

    def test_identity_fixes_all_cosets(self):
        for c in self.CS.cosets:
            self.assertEqual(self.CS.act(0, c), c)

    def test_orbit_of_coset_is_subset(self):
        for c in self.CS.cosets:
            orbit = self.CS.orbit_of_coset(c)
            for x in orbit:
                self.assertIn(x, self.CS.cosets)

    def test_orbit_sizes_sum_to_4(self):
        visited = set()
        total = 0
        for c in self.CS.cosets:
            if c.index in visited:
                continue
            orb = self.CS.orbit_of_coset(c)
            total += len(orb)
            for x in orb:
                visited.add(x.index)
        self.assertEqual(total, 4)

    def test_witness_for(self):
        CS = self.CS
        for ref in CS.cosets:
            for target in CS.cosets:
                g = CS.witness_for(target, ref)
                if g is not None:
                    self.assertEqual(CS.act(g, ref), target)


# ─────────────────────────────────────────────────────────────────────────────
# T05  Orbit-maximising lifting
# ─────────────────────────────────────────────────────────────────────────────

class T05_OrbitMaxLift(unittest.TestCase):

    def test_empty_returns_empty(self):
        G  = dna_default_group()
        H  = SimpleGECSGBuilder._z3_subgroup(G)
        CS = G.coset_space(H)
        self.assertEqual(CS.orbit_maximising_coset([]), [])

    def test_single_candidate_returned(self):
        G  = dna_default_group()
        H  = SimpleGECSGBuilder._z3_subgroup(G)
        CS = G.coset_space(H)
        c  = CS[0]
        result = CS.orbit_maximising_coset([c])
        self.assertEqual(result, [c])

    def test_max_orbit_selected(self):
        G  = dna_default_group()
        H  = SimpleGECSGBuilder._z3_subgroup(G)
        CS = G.coset_space(H)
        cosets = list(CS.cosets)
        result = CS.orbit_maximising_coset(cosets)
        max_size = max(CS.orbit_size(c) for c in cosets)
        for c in result:
            self.assertEqual(CS.orbit_size(c), max_size)


# ─────────────────────────────────────────────────────────────────────────────
# T06  Builder fluent API
# ─────────────────────────────────────────────────────────────────────────────

class T06_BuilderAPI(unittest.TestCase):

    def _minimal_grammar(self):
        G  = dna_default_group()
        H  = SimpleGECSGBuilder._z3_subgroup(G)
        return (SimpleGECSGBuilder(start="Gene", group=G,
                                   subgroup_indices=H, k=3)
                .rule("Gene", ["CDS"])
                .rule("CDS",  ["Codon"])
                .rule("CDS",  ["Codon", "CDS"])
                .break_coset(0, "ATG")
                .break_coset(1, "TAA")
                .build())

    def test_build_returns_grammar(self):
        g = self._minimal_grammar()
        self.assertIsInstance(g, GECSGGrammar)

    def test_start_symbol(self):
        g = self._minimal_grammar()
        self.assertEqual(g.start.name, "Gene")

    def test_nonterminals_registered(self):
        g = self._minimal_grammar()
        names = {nt.name for nt in g.nonterminals}
        self.assertIn("Gene", names)
        self.assertIn("CDS", names)

    def test_breaking_rules_count(self):
        g = self._minimal_grammar()
        self.assertEqual(g.n_breaking, 2)

    def test_break_all_cosets(self):
        G  = dna_default_group()
        H  = SimpleGECSGBuilder._z3_subgroup(G)
        g  = (SimpleGECSGBuilder(start="S", group=G,
                                  subgroup_indices=H, k=3)
              .rule("S", ["S"])
              .break_all_cosets({0: "ATG", 1: "TAA", 2: "GCT", 3: "GGT"})
              .build())
        self.assertEqual(g.n_breaking, 4)

    def test_repr(self):
        b = SimpleGECSGBuilder(start="S")
        r = repr(b)
        self.assertIn("SimpleGECSGBuilder", r)
        self.assertIn("rules=0", r)


# ─────────────────────────────────────────────────────────────────────────────
# T07  Grammar properties
# ─────────────────────────────────────────────────────────────────────────────

class T07_GrammarProperties(unittest.TestCase):

    def setUp(self):
        self.g = SimpleGECSGBuilder.dna_codon_grammar()

    def test_n_orbits(self):
        # 3 rules: Gene->CDS, CDS->Codon, CDS->Codon CDS
        self.assertEqual(self.g.n_orbits, 3)

    def test_n_full_rules_ge_n_orbits(self):
        self.assertGreaterEqual(self.g.n_full_rules, self.g.n_orbits)

    def test_n_breaking(self):
        self.assertEqual(self.g.n_breaking, 4)

    def test_compression_ratio_le_1(self):
        ratio = self.g.compression_ratio()
        self.assertLessEqual(ratio, 1.0)
        self.assertGreater(ratio, 0.0)

    def test_coset_space_size(self):
        self.assertEqual(self.g.coset_space.size, 4)

    def test_k(self):
        self.assertEqual(self.g.k, 3)

    def test_is_frozen(self):
        self.assertTrue(self.g.is_frozen)

    def test_full_rules_lhs_coverage(self):
        # Every orbit representative LHS should appear in full_rules
        orbit_lhs = {o.representative.lhs for o in self.g.orbits}
        full_lhs  = {r.lhs for r in self.g.full_rules}
        self.assertEqual(orbit_lhs, full_lhs)


# ─────────────────────────────────────────────────────────────────────────────
# T08  Orbit generation
# ─────────────────────────────────────────────────────────────────────────────

class T08_OrbitGeneration(unittest.TestCase):

    def test_orbit_is_nonempty(self):
        G   = dna_default_group()
        H   = SimpleGECSGBuilder._z3_subgroup(G)
        CS  = G.coset_space(H)
        nt  = NonTerminal("A")
        rep = GenerationRule(lhs=nt, rhs=(nt,))
        orbit = generate_orbit(rep, G, CS)
        self.assertGreater(orbit.size(), 0)

    def test_orbit_contains_representative(self):
        G   = dna_default_group()
        H   = SimpleGECSGBuilder._z3_subgroup(G)
        CS  = G.coset_space(H)
        nt  = NonTerminal("A")
        rep = GenerationRule(lhs=nt, rhs=(nt,))
        orbit = generate_orbit(rep, G, CS)
        self.assertIn(rep, orbit.members)

    def test_orbit_g_indices_length(self):
        G   = dna_default_group()
        H   = SimpleGECSGBuilder._z3_subgroup(G)
        CS  = G.coset_space(H)
        nt  = NonTerminal("X")
        rep = GenerationRule(lhs=nt, rhs=(nt,))
        orbit = generate_orbit(rep, G, CS)
        self.assertEqual(len(orbit.members), len(orbit.g_indices))

    def test_orbit_size_divides_group_order(self):
        G   = dna_default_group()
        H   = SimpleGECSGBuilder._z3_subgroup(G)
        CS  = G.coset_space(H)
        nt  = NonTerminal("B")
        rep = GenerationRule(lhs=nt, rhs=(nt,))
        orbit = generate_orbit(rep, G, CS)
        self.assertEqual(G.order % orbit.size(), 0)


# ─────────────────────────────────────────────────────────────────────────────
# T09  Lifting map
# ─────────────────────────────────────────────────────────────────────────────

class T09_LiftingMap(unittest.TestCase):

    def setUp(self):
        self.g = SimpleGECSGBuilder.dna_codon_grammar()

    def test_lift_atg(self):
        # ATG was registered as break_coset(0, "ATG")
        result = self.g.lift(("A", "T", "G"))
        self.assertGreater(len(result), 0)
        self.assertIsInstance(result[0], Coset)

    def test_lift_unknown_returns_empty(self):
        result = self.g.lift(("X", "X", "X"))
        self.assertEqual(result, [])

    def test_lift_star_subset_of_lift(self):
        block = ("A", "T", "G")
        full  = self.g.lift(block)
        star  = self.g.lift_star(block)
        for c in star:
            self.assertIn(c, full)

    def test_lift_star_max_orbit_size(self):
        block = ("A", "T", "G")
        star  = self.g.lift_star(block)
        if star:
            CS   = self.g.coset_space
            sizes = [CS.orbit_size(c) for c in star]
            self.assertEqual(len(set(sizes)), 1)   # all same size


# ─────────────────────────────────────────────────────────────────────────────
# T10  Pre-defined grammars
# ─────────────────────────────────────────────────────────────────────────────

class T10_PredefinedGrammars(unittest.TestCase):

    def test_dna_codon_grammar_builds(self):
        g = SimpleGECSGBuilder.dna_codon_grammar()
        self.assertIsInstance(g, GECSGGrammar)

    def test_dna_codon_start(self):
        g = SimpleGECSGBuilder.dna_codon_grammar()
        self.assertEqual(g.start.name, "Gene")

    def test_abstract_z2_grammar_builds(self):
        g = SimpleGECSGBuilder.abstract_z2_grammar()
        self.assertIsInstance(g, GECSGGrammar)

    def test_abstract_z2_start(self):
        g = SimpleGECSGBuilder.abstract_z2_grammar()
        self.assertEqual(g.start.name, "S")

    def test_palindrome_grammar_builds(self):
        g = SimpleGECSGBuilder.palindrome_grammar()
        self.assertIsInstance(g, GECSGGrammar)

    def test_palindrome_grammar_k1(self):
        g = SimpleGECSGBuilder.palindrome_grammar()
        self.assertEqual(g.k, 1)


# ─────────────────────────────────────────────────────────────────────────────
# T11  Freeze guard
# ─────────────────────────────────────────────────────────────────────────────

class T11_FreezeGuard(unittest.TestCase):

    def test_add_rule_after_freeze_raises(self):
        g = SimpleGECSGBuilder.dna_codon_grammar()
        self.assertTrue(g.is_frozen)
        with self.assertRaises(RuntimeError):
            g.add_generation_rule("Gene", ["CDS"])

    def test_add_breaking_after_freeze_raises(self):
        g = SimpleGECSGBuilder.dna_codon_grammar()
        with self.assertRaises(RuntimeError):
            g.add_breaking_rule(0, "ATG")

    def test_build_without_freeze(self):
        G  = dna_default_group()
        H  = SimpleGECSGBuilder._z3_subgroup(G)
        g  = (SimpleGECSGBuilder(start="S", group=G,
                                  subgroup_indices=H, k=3)
              .rule("S", ["S"])
              .build(freeze=False))
        self.assertFalse(g.is_frozen)
        # Should not raise
        g.add_generation_rule("S", ["S"])


# ─────────────────────────────────────────────────────────────────────────────
# T12  Auto-break from alphabet
# ─────────────────────────────────────────────────────────────────────────────

class T12_AutoBreak(unittest.TestCase):

    def test_auto_break_assigns_cosets(self):
        G  = dna_default_group()
        H  = SimpleGECSGBuilder._z3_subgroup(G)
        b  = (SimpleGECSGBuilder(start="S", group=G,
                                  subgroup_indices=H, k=3)
              .rule("S", ["S"])
              .auto_break_from_alphabet(["A", "T", "G", "C"], k=3))
        g  = b.build()
        # Number of breaking rules == number of orbits on Σ^3 (at most 10)
        self.assertGreater(g.n_breaking, 0)
        self.assertLessEqual(g.n_breaking, 10)

    def test_auto_break_strings_have_correct_length(self):
        G  = dna_default_group()
        H  = SimpleGECSGBuilder._z3_subgroup(G)
        b  = (SimpleGECSGBuilder(start="S", group=G,
                                  subgroup_indices=H, k=3)
              .rule("S", ["S"])
              .auto_break_from_alphabet(["A", "T", "G", "C"], k=3))
        g  = b.build()
        for br in g.breaking_rules:
            self.assertEqual(len(br.string), 3)


# ─────────────────────────────────────────────────────────────────────────────
# T13  Default subgroup gives 4-letter alphabet
# ─────────────────────────────────────────────────────────────────────────────

class T13_DefaultSubgroup(unittest.TestCase):

    def test_dna_default_gives_4_cosets(self):
        G  = dna_default_group()
        H  = SimpleGECSGBuilder._z3_subgroup(G)
        CS = G.coset_space(H)
        self.assertEqual(CS.size, 4)

    def test_builder_default_subgroup_gives_4_cosets(self):
        b  = SimpleGECSGBuilder(start="S")
        G  = b._G
        H  = b._H_idx
        CS = G.coset_space(H)
        self.assertEqual(CS.size, 4)


# ─────────────────────────────────────────────────────────────────────────────
# T14  Torch vocabulary hooks
# ─────────────────────────────────────────────────────────────────────────────

class T14_TorchVocab(unittest.TestCase):

    def setUp(self):
        self.g = SimpleGECSGBuilder.dna_codon_grammar()

    def test_nonterminal_vocab_keys(self):
        vocab = self.g.nonterminal_vocab
        self.assertIn("Gene", vocab)
        self.assertIn("CDS", vocab)

    def test_nonterminal_vocab_unique_indices(self):
        vocab = self.g.nonterminal_vocab
        vals  = list(vocab.values())
        self.assertEqual(len(vals), len(set(vals)))

    def test_coset_vocab_size(self):
        vocab = self.g.coset_vocab
        self.assertEqual(len(vocab), self.g.coset_space.size)

    def test_rule_vocab_size(self):
        vocab = self.g.rule_vocab
        self.assertEqual(len(vocab), self.g.n_orbits)

    def test_vocab_indices_are_nonneg_ints(self):
        for v in [self.g.nonterminal_vocab, self.g.coset_vocab,
                  self.g.rule_vocab]:
            for idx in v.values():
                self.assertIsInstance(idx, int)
                self.assertGreaterEqual(idx, 0)


# ─────────────────────────────────────────────────────────────────────────────
# T15  Palindrome grammar smoke test
# ─────────────────────────────────────────────────────────────────────────────

class T15_PalindromeGrammar(unittest.TestCase):

    def test_breaking_rules_cover_both_cosets(self):
        g = SimpleGECSGBuilder.palindrome_grammar()
        coset_indices = {br.coset.index for br in g.breaking_rules}
        # coset 0 -> A, G  (two rules)  coset 1 -> T, C  (two rules)
        self.assertIn(0, coset_indices)
        self.assertIn(1, coset_indices)

    def test_breaking_rules_have_k1_strings(self):
        g = SimpleGECSGBuilder.palindrome_grammar()
        for br in g.breaking_rules:
            self.assertEqual(len(br.string), 1)

    def test_group_order_2(self):
        g = SimpleGECSGBuilder.palindrome_grammar()
        self.assertEqual(g.group.order, 2)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pretty header
    print("=" * 65)
    print("  GE-CSG Builder Test Suite")
    print("=" * 65)
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None   # preserve class order
    suite  = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
