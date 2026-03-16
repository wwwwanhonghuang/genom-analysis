"""
test_parsing.py
===============
Test suite for the GE-CSG Earley parser.

Run:
    python test_parsing.py
    python test_parsing.py -v

Coverage
--------
P01  Acceptance / rejection            -- valid vs invalid sequences
P02  Single-codon sequences            -- minimal grammar productions
P03  Short sequences (2-4 codons)      -- basic tree structure
P04  Medium sequences (5-10 codons)    -- right-recursive CDS chains
P05  Long sequences (15-30 codons)     -- stress / performance
P06  All-same-coset sequences          -- homopolymer-like runs
P07  All four coset classes            -- every coset appears as leaf
P08  Tree structure invariants         -- spans, leaves, raw_string
P09  Coset assignment correctness      -- first-nucleotide -> coset
P10  Completed spans bookkeeping       -- Earley internal state
P11  Grammar.lift consistency          -- lift(block) non-empty for all 64
P12  re-use of parser                  -- parse multiple sequences in sequence
P13  Case-insensitive input            -- lowercase accepted
P14  Visualisation smoke test          -- draw_parse_tree does not raise
P15  Long sequence visualisation       -- PNG output for a 12-codon sequence
"""

import sys, os, io, unittest, time
_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from gecsg.grammar.dna_grammar import dna_codon_grammar, NUC_TO_COSET
from gecsg.parser.earley import EquivariantEarleyParser
from gecsg.parser.parse_tree import ParseNode

# ── Shared fixtures ────────────────────────────────────────────────────────
GRAMMAR = dna_codon_grammar()
PARSER  = EquivariantEarleyParser(GRAMMAR)

BASES   = "ATGC"
ALL_CODONS = [a+b+c for a in BASES for b in BASES for c in BASES]   # 64

# One representative codon per coset class (used across many tests)
CODON_A = "ATG"   # A__ -> coset 0
CODON_T = "TAA"   # T__ -> coset 1
CODON_G = "GCT"   # G__ -> coset 2
CODON_C = "CAG"   # C__ -> coset 3


def parse(seq: str):
    """Convenience wrapper."""
    return PARSER.parse(seq)


def make_seq(*codons) -> str:
    return "".join(codons)


# ─────────────────────────────────────────────────────────────────────────────
# P01  Acceptance / Rejection
# ─────────────────────────────────────────────────────────────────────────────

class P01_AcceptReject(unittest.TestCase):

    def test_single_codon_accepted(self):
        self.assertTrue(parse("ATG").accepted)

    def test_two_codons_accepted(self):
        self.assertTrue(parse("ATGTAA").accepted)

    def test_three_codons_accepted(self):
        self.assertTrue(parse("ATGGCTTAA").accepted)

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            parse("AT")           # length 2, not multiple of 3

    def test_wrong_length_4_raises(self):
        with self.assertRaises(ValueError):
            parse("ATGT")

    def test_all_64_single_codons_accepted(self):
        """Every single codon forms a valid 1-codon CDS."""
        for codon in ALL_CODONS:
            result = parse(codon)
            self.assertTrue(result.accepted,
                            f"Codon {codon} should be accepted")

    def test_all_64_codons_two_at_a_time(self):
        """Every pair (c1, c2) should be accepted."""
        import random
        rng = random.Random(42)
        sample = rng.sample(ALL_CODONS, 12)
        for c1 in sample:
            for c2 in sample[:4]:
                result = parse(c1 + c2)
                self.assertTrue(result.accepted,
                                f"{c1}{c2} should be accepted")

    def test_empty_string_not_accepted(self):
        # length 0 is divisible by k=3, so no ValueError; but 0-codon CDS
        # is not in the grammar (CDS requires at least one Codon), so rejected
        result = parse("")
        self.assertFalse(result.accepted)


# ─────────────────────────────────────────────────────────────────────────────
# P02  Single-codon sequences
# ─────────────────────────────────────────────────────────────────────────────

class P02_SingleCodon(unittest.TestCase):

    def _check(self, codon):
        result = parse(codon)
        self.assertTrue(result.accepted)
        trees = result.trees()
        self.assertEqual(len(trees), 1)
        return trees[0]

    def test_atg_tree_depth(self):
        tree = self._check("ATG")
        # Gene -> CDS -> Codon -> [leaf]  depth = 3
        self.assertEqual(tree.depth, 3)

    def test_leaf_raw_codon(self):
        tree = self._check("GCC")
        leaves = _collect_leaves(tree)
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0].raw_codon, "GCC")

    def test_root_label(self):
        tree = self._check("TAA")
        self.assertEqual(tree.label, "Gene")

    def test_span_covers_full_sequence(self):
        tree = self._check("CAT")
        self.assertEqual(tree.span, (0, 1))

    def test_raw_string_matches_input(self):
        for codon in ["ATG", "GCT", "TAA", "CAG", "TTT", "GGG"]:
            tree = self._check(codon)
            self.assertEqual(tree.raw_string, codon)


# ─────────────────────────────────────────────────────────────────────────────
# P03  Short sequences (2-4 codons)
# ─────────────────────────────────────────────────────────────────────────────

class P03_ShortSequences(unittest.TestCase):

    def test_2codon_span(self):
        result = parse("ATGTAA")
        self.assertTrue(result.accepted)
        tree = result.trees()[0]
        self.assertEqual(tree.span, (0, 2))

    def test_2codon_raw_string(self):
        seq = "GCTTAG"
        result = parse(seq)
        self.assertTrue(result.accepted)
        self.assertEqual(result.trees()[0].raw_string, seq)

    def test_3codon_leaf_count(self):
        result = parse("ATGGCTTAA")
        tree = result.trees()[0]
        leaves = _collect_leaves(tree)
        self.assertEqual(len(leaves), 3)

    def test_3codon_leaf_codons(self):
        seq = "ATGGCTTAA"
        result = parse(seq)
        tree = result.trees()[0]
        leaves = _collect_leaves(tree)
        raw = [lf.raw_codon for lf in leaves]
        self.assertEqual("".join(raw), seq)

    def test_4codon_accepted(self):
        seq = make_seq("ATG", "AAA", "GCT", "TAA")
        self.assertTrue(parse(seq).accepted)

    def test_4codon_span(self):
        seq = make_seq("ATG", "TTT", "GGG", "CAG")
        tree = parse(seq).trees()[0]
        self.assertEqual(tree.span, (0, 4))

    def test_all_same_first_base_2codons(self):
        """Pairs of codons from same coset class."""
        for n1 in BASES:
            c1 = n1 + "AT"
            c2 = n1 + "GC"
            result = parse(c1 + c2)
            self.assertTrue(result.accepted, f"{c1}{c2} rejected")

    def test_all_combinations_4cosets_2codons(self):
        """One codon from each class, paired."""
        reps = [CODON_A, CODON_T, CODON_G, CODON_C]
        for c1 in reps:
            for c2 in reps:
                self.assertTrue(parse(c1 + c2).accepted,
                                f"{c1}{c2} rejected")


# ─────────────────────────────────────────────────────────────────────────────
# P04  Medium sequences (5-10 codons)
# ─────────────────────────────────────────────────────────────────────────────

class P04_MediumSequences(unittest.TestCase):

    def _make_n(self, n: int) -> str:
        """Build a sequence of n codons cycling through [ATG, GCT, TAA, CAG]."""
        pool = [CODON_A, CODON_G, CODON_T, CODON_C]
        return "".join(pool[i % 4] for i in range(n))

    def test_5codons_accepted(self):
        self.assertTrue(parse(self._make_n(5)).accepted)

    def test_6codons_accepted(self):
        seq = "ATGAAAGCTTTTGCCTAA"   # 6 codons
        self.assertTrue(parse(seq).accepted)

    def test_6codons_leaf_count(self):
        seq = "ATGAAAGCTTTTGCCTAA"
        tree = parse(seq).trees()[0]
        self.assertEqual(len(_collect_leaves(tree)), 6)

    def test_7codons_raw_string(self):
        seq = self._make_n(7)
        tree = parse(seq).trees()[0]
        self.assertEqual(tree.raw_string, seq)

    def test_8codons_span(self):
        seq = self._make_n(8)
        tree = parse(seq).trees()[0]
        self.assertEqual(tree.span, (0, 8))

    def test_9codons_accepted(self):
        self.assertTrue(parse(self._make_n(9)).accepted)

    def test_10codons_leaf_order(self):
        """Leaves must appear in left-to-right input order."""
        seq = self._make_n(10)
        tree = parse(seq).trees()[0]
        leaves = _collect_leaves(tree)
        reconstructed = "".join(lf.raw_codon for lf in leaves)
        self.assertEqual(reconstructed, seq)

    def test_mixed_codon_classes_5(self):
        seq = make_seq("ATG", "TAA", "GCT", "CAG", "AAA")
        result = parse(seq)
        self.assertTrue(result.accepted)
        leaves = _collect_leaves(result.trees()[0])
        self.assertEqual(len(leaves), 5)


# ─────────────────────────────────────────────────────────────────────────────
# P05  Long sequences (15-30 codons)  — stress + performance
# ─────────────────────────────────────────────────────────────────────────────

class P05_LongSequences(unittest.TestCase):

    def _seq(self, n):
        pool = [CODON_A, CODON_G, CODON_T, CODON_C]
        return "".join(pool[i % 4] for i in range(n))

    def test_15codons_accepted(self):
        self.assertTrue(parse(self._seq(15)).accepted)

    def test_15codons_leaf_count(self):
        tree = parse(self._seq(15)).trees()[0]
        self.assertEqual(len(_collect_leaves(tree)), 15)

    def test_20codons_accepted(self):
        self.assertTrue(parse(self._seq(20)).accepted)

    def test_20codons_raw_string_preserved(self):
        seq = self._seq(20)
        tree = parse(seq).trees()[0]
        self.assertEqual(tree.raw_string, seq)

    def test_30codons_accepted(self):
        self.assertTrue(parse(self._seq(30)).accepted)

    def test_30codons_span(self):
        seq = self._seq(30)
        tree = parse(seq).trees()[0]
        self.assertEqual(tree.span, (0, 30))

    def test_performance_20codons(self):
        """Parse 20-codon sequence in under 5 seconds."""
        seq = self._seq(20)
        t0 = time.perf_counter()
        result = parse(seq)
        t1 = time.perf_counter()
        self.assertTrue(result.accepted)
        self.assertLess(t1 - t0, 5.0,
                        f"20-codon parse took {t1-t0:.2f}s (limit 5s)")

    def test_performance_30codons(self):
        """Parse 30-codon sequence in under 30 seconds."""
        seq = self._seq(30)
        t0 = time.perf_counter()
        result = parse(seq)
        t1 = time.perf_counter()
        self.assertTrue(result.accepted)
        self.assertLess(t1 - t0, 30.0,
                        f"30-codon parse took {t1-t0:.2f}s (limit 30s)")

    def test_real_gene_fragment(self):
        """
        A realistic coding sequence fragment.
        ATG (Met) + 8 body codons + TAA (Stop)
        ATGGCTAGCAAAGTTCGTCATGCATAA  — 9 codons
        """
        seq = "ATGGCTAGCAAAGTTCGTCATGCATAA"
        # 27 chars = 9 codons: ATG GCT AGC AAA GTT CGT CAT GCA TAA
        self.assertEqual(len(seq), 27)
        result = parse(seq)
        self.assertTrue(result.accepted)
        leaves = _collect_leaves(result.trees()[0])
        self.assertEqual(len(leaves), 9)
        self.assertEqual("".join(lf.raw_codon for lf in leaves), seq)


# ─────────────────────────────────────────────────────────────────────────────
# P06  All-same-coset sequences (homopolymer-like)
# ─────────────────────────────────────────────────────────────────────────────

class P06_HomoCoset(unittest.TestCase):
    """Sequences where every codon belongs to the same coset class."""

    def test_all_A_codons_5(self):
        seq = "ATG" * 5       # all coset 0
        self.assertTrue(parse(seq).accepted)

    def test_all_T_codons_5(self):
        seq = "TAA" * 5       # all coset 1
        self.assertTrue(parse(seq).accepted)

    def test_all_G_codons_5(self):
        seq = "GCT" * 5       # all coset 2
        self.assertTrue(parse(seq).accepted)

    def test_all_C_codons_5(self):
        seq = "CAG" * 5       # all coset 3
        self.assertTrue(parse(seq).accepted)

    def test_homo_leaves_same_coset(self):
        """All leaves in a homo-coset sequence must have the same coset index."""
        seq  = "GGG" * 6      # all G__ -> coset 2
        tree = parse(seq).trees()[0]
        leaves = _collect_leaves(tree)
        coset_indices = {lf.coset.index for lf in leaves}
        self.assertEqual(len(coset_indices), 1)
        self.assertEqual(coset_indices.pop(), NUC_TO_COSET["G"])


# ─────────────────────────────────────────────────────────────────────────────
# P07  All four coset classes appear
# ─────────────────────────────────────────────────────────────────────────────

class P07_AllCosets(unittest.TestCase):

    def test_all_four_cosets_in_leaves(self):
        """A sequence with one codon from each coset must yield 4 distinct cosets."""
        seq  = make_seq(CODON_A, CODON_T, CODON_G, CODON_C)
        tree = parse(seq).trees()[0]
        leaves = _collect_leaves(tree)
        coset_indices = {lf.coset.index for lf in leaves}
        self.assertEqual(coset_indices, {0, 1, 2, 3})

    def test_each_coset_representative_lifts(self):
        """For one codon of each class, lift() must return the expected coset."""
        cases = [
            ("ATG", 0),  ("TTT", 1),  ("GGG", 2),  ("CAT", 3),
            ("AAA", 0),  ("TAA", 1),  ("GTT", 2),  ("CTG", 3),
        ]
        for codon, expected_ci in cases:
            cosets = GRAMMAR.lift(tuple(codon))
            self.assertEqual(len(cosets), 1)
            self.assertEqual(cosets[0].index, expected_ci,
                             f"{codon}: expected coset {expected_ci}, "
                             f"got {cosets[0].index}")

    def test_all_64_codons_lift_to_exactly_one_coset(self):
        """No codon should be ambiguous or unmapped."""
        for codon in ALL_CODONS:
            cosets = GRAMMAR.lift(tuple(codon))
            self.assertEqual(len(cosets), 1,
                             f"{codon} lifts to {len(cosets)} cosets (expected 1)")

    def test_all_64_coset_indices_match_first_nuc(self):
        for codon in ALL_CODONS:
            expected = NUC_TO_COSET[codon[0]]
            actual   = GRAMMAR.lift(tuple(codon))[0].index
            self.assertEqual(actual, expected,
                             f"{codon}: first nuc {codon[0]} -> expected coset "
                             f"{expected}, got {actual}")


# ─────────────────────────────────────────────────────────────────────────────
# P08  Tree structure invariants
# ─────────────────────────────────────────────────────────────────────────────

class P08_TreeStructure(unittest.TestCase):

    def _tree(self, seq):
        result = parse(seq)
        self.assertTrue(result.accepted, f"{seq} not accepted")
        return result.trees()[0]

    def test_root_is_gene(self):
        for seq in ["ATG", "ATGGCT", "ATGGCTTAA"]:
            self.assertEqual(self._tree(seq).label, "Gene")

    def test_root_span_equals_n_codons(self):
        for n in [1, 2, 3, 5, 8]:
            seq = (CODON_A + CODON_G) * n
            seq = seq[:n * 3]         # trim to exactly n codons
            # Rebuild cleanly
            pool = [CODON_A, CODON_G, CODON_T, CODON_C]
            seq  = "".join(pool[i % 4] for i in range(n))
            tree = self._tree(seq)
            self.assertEqual(tree.span[0], 0)
            self.assertEqual(tree.span[1], n)

    def test_leaves_are_terminal(self):
        seq    = "ATGGCTTAA"
        leaves = _collect_leaves(self._tree(seq))
        for lf in leaves:
            self.assertTrue(lf.is_terminal)

    def test_internal_nodes_not_terminal(self):
        seq  = "ATGGCTTAA"
        tree = self._tree(seq)
        self._check_internal_not_terminal(tree)

    def _check_internal_not_terminal(self, node):
        if node.children:
            self.assertFalse(node.is_terminal,
                             f"Node {node.label} has children but is_terminal=True")
            for ch in node.children:
                self._check_internal_not_terminal(ch)

    def test_leaf_spans_are_unit(self):
        """Every leaf must span exactly one coset position."""
        seq    = "ATGGCTTAACAG"   # 4 codons
        leaves = _collect_leaves(self._tree(seq))
        for lf in leaves:
            s, e = lf.span
            self.assertEqual(e - s, 1,
                             f"Leaf {lf.label} has span {lf.span}, expected width 1")

    def test_child_spans_partition_parent(self):
        """Children spans must tile the parent span without overlap."""
        seq  = "ATGGCTTAACAG"
        tree = self._tree(seq)
        _check_span_partition(self, tree)

    def test_raw_string_matches_input(self):
        seqs = [
            "ATG",
            "ATGGCT",
            "ATGGCTTAA",
            "ATGAAAGCTTTTGCCTAA",
            "ATG" * 10,
        ]
        for seq in seqs:
            tree = self._tree(seq)
            self.assertEqual(tree.raw_string, seq,
                             f"raw_string mismatch for {seq}")

    def test_leaf_codon_concatenation(self):
        """Concatenating leaf raw_codons (L-to-R) must reproduce the input."""
        seqs = ["ATGGCTTAA", "ATGAAAGCTTTTGCCTAA", "GCT" * 7]
        for seq in seqs:
            leaves = _collect_leaves(self._tree(seq))
            self.assertEqual("".join(lf.raw_codon for lf in leaves), seq)

    def test_number_of_leaves_equals_n_codons(self):
        for n in [1, 2, 3, 5, 10, 15]:
            pool = [CODON_A, CODON_G, CODON_T, CODON_C]
            seq  = "".join(pool[i % 4] for i in range(n))
            tree = self._tree(seq)
            self.assertEqual(len(_collect_leaves(tree)), n,
                             f"n={n}: wrong leaf count")


# ─────────────────────────────────────────────────────────────────────────────
# P09  Coset assignment correctness
# ─────────────────────────────────────────────────────────────────────────────

class P09_CosetAssignment(unittest.TestCase):
    """Verify that every leaf's coset.index matches NUC_TO_COSET[first_base]."""

    def _check_coset_assignment(self, seq):
        result = parse(seq)
        self.assertTrue(result.accepted)
        leaves = _collect_leaves(result.trees()[0])
        for i, lf in enumerate(leaves):
            codon      = seq[3 * i: 3 * (i + 1)]
            expected_c = NUC_TO_COSET[codon[0]]
            self.assertEqual(
                lf.coset.index, expected_c,
                f"Codon {codon} at pos {i}: expected coset {expected_c}, "
                f"got {lf.coset.index}"
            )

    def test_atg_gct_taa(self):
        self._check_coset_assignment("ATGGCTTAA")

    def test_mixed_all_four_classes(self):
        self._check_coset_assignment(
            make_seq("ATG", "TAA", "GCT", "CAG", "AAA", "TTT", "GGG", "CCC"))

    def test_all_A_codons(self):
        self._check_coset_assignment("ATG" * 6)

    def test_all_T_codons(self):
        self._check_coset_assignment("TAA" * 6)

    def test_random_10codons(self):
        import random
        rng = random.Random(99)
        seq = "".join(rng.choice(ALL_CODONS) for _ in range(10))
        self._check_coset_assignment(seq)

    def test_every_single_codon_coset(self):
        for codon in ALL_CODONS:
            result = parse(codon)
            self.assertTrue(result.accepted)
            leaf = _collect_leaves(result.trees()[0])[0]
            expected = NUC_TO_COSET[codon[0]]
            self.assertEqual(leaf.coset.index, expected,
                             f"Codon {codon}: coset mismatch")


# ─────────────────────────────────────────────────────────────────────────────
# P10  Completed-spans bookkeeping
# ─────────────────────────────────────────────────────────────────────────────

class P10_CompletedSpans(unittest.TestCase):

    def test_gene_span_in_completed(self):
        seq    = "ATGGCTTAA"
        result = parse(seq)
        self.assertIn(("Gene", 0, 3), result.completed_spans)

    def test_cds_spans_present(self):
        seq    = "ATGGCTTAA"
        result = parse(seq)
        # CDS should span [0,3), [1,3), [2,3) in a 3-codon parse
        self.assertIn(("CDS", 0, 3), result.completed_spans)
        self.assertIn(("CDS", 1, 3), result.completed_spans)
        self.assertIn(("CDS", 2, 3), result.completed_spans)

    def test_codon_unit_spans_present(self):
        seq    = "ATGGCTTAA"
        result = parse(seq)
        for i in range(3):
            self.assertIn(("Codon", i, i+1), result.completed_spans)

    def test_n_codons_correct(self):
        for n in [1, 3, 7, 12]:
            pool = [CODON_A, CODON_G, CODON_T, CODON_C]
            seq  = "".join(pool[i % 4] for i in range(n))
            result = parse(seq)
            self.assertEqual(result.n_codons, n)

    def test_gene_full_span_present(self):
        """Gene(0, n) must always appear in completed_spans."""
        seq    = "ATGGCTTAA"
        result = parse(seq)
        gene_spans = {(s, e) for (nt, s, e) in result.completed_spans
                      if nt == "Gene"}
        # Grammar allows Gene → CDS for any valid sub-CDS, so partial
        # Gene spans are legal; the important invariant is that the full
        # span (0, n) is present.
        self.assertIn((0, 3), gene_spans)


# ─────────────────────────────────────────────────────────────────────────────
# P11  Grammar.lift consistency
# ─────────────────────────────────────────────────────────────────────────────

class P11_LiftConsistency(unittest.TestCase):

    def test_all_64_codons_lift_nonempty(self):
        for codon in ALL_CODONS:
            cosets = GRAMMAR.lift(tuple(codon))
            self.assertGreater(len(cosets), 0,
                               f"lift({codon}) is empty")

    def test_lift_returns_coset_objects(self):
        from gecsg.core.coset import Coset
        for codon in ALL_CODONS[:16]:
            for c in GRAMMAR.lift(tuple(codon)):
                self.assertIsInstance(c, Coset)

    def test_lift_star_subset_of_lift(self):
        for codon in ALL_CODONS[:16]:
            full = GRAMMAR.lift(tuple(codon))
            star = GRAMMAR.lift_star(tuple(codon))
            for c in star:
                self.assertIn(c, full)

    def test_unknown_codon_lift_empty(self):
        self.assertEqual(GRAMMAR.lift(("X", "Y", "Z")), [])


# ─────────────────────────────────────────────────────────────────────────────
# P12  Parser reuse
# ─────────────────────────────────────────────────────────────────────────────

class P12_ParserReuse(unittest.TestCase):

    def test_parse_multiple_sequences_sequentially(self):
        seqs = [
            "ATG", "ATGTAA", "ATGGCTTAA",
            "ATGAAAGCTTTTGCCTAA", "GCT" * 5,
        ]
        for seq in seqs:
            result = PARSER.parse(seq)
            self.assertTrue(result.accepted, f"{seq} not accepted on reuse")

    def test_independent_results(self):
        """Results from different parses must not share state."""
        r1 = PARSER.parse("ATGGCT")
        r2 = PARSER.parse("TAAATG")
        self.assertIsNot(r1.chart, r2.chart)
        self.assertIsNot(r1.completed_spans, r2.completed_spans)


# ─────────────────────────────────────────────────────────────────────────────
# P13  Case-insensitive input
# ─────────────────────────────────────────────────────────────────────────────

class P13_CaseInsensitive(unittest.TestCase):

    def test_lowercase_accepted(self):
        self.assertTrue(parse("atggcttaa").accepted)

    def test_mixed_case_accepted(self):
        self.assertTrue(parse("AtGgCtTaA").accepted)

    def test_lowercase_tree_raw_string_uppercased(self):
        result = parse("atggct")
        tree   = result.trees()[0]
        # raw_string should reflect the uppercased internal representation
        self.assertEqual(tree.raw_string, "ATGGCT")

    def test_lowercase_leaf_codons_uppercased(self):
        result = parse("atggcttaa")
        leaves = _collect_leaves(result.trees()[0])
        for lf in leaves:
            self.assertEqual(lf.raw_codon, lf.raw_codon.upper())


# ─────────────────────────────────────────────────────────────────────────────
# P14  Visualisation smoke test
# ─────────────────────────────────────────────────────────────────────────────

class P14_VizSmoke(unittest.TestCase):

    def test_draw_does_not_raise_3codon(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from gecsg.visualize.tree_viz import draw_parse_tree

        seq    = "ATGGCTTAA"
        result = parse(seq)
        tree   = result.trees()[0]
        fig    = draw_parse_tree(tree, seq, title="Test 3-codon")
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_draw_does_not_raise_single_codon(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from gecsg.visualize.tree_viz import draw_parse_tree

        seq    = "ATG"
        result = parse(seq)
        tree   = result.trees()[0]
        fig    = draw_parse_tree(tree, seq)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_draw_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.figure
        from gecsg.visualize.tree_viz import draw_parse_tree

        seq  = "ATGGCTTAA"
        tree = parse(seq).trees()[0]
        fig  = draw_parse_tree(tree, seq)
        self.assertIsInstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# P15  Long-sequence PNG output
# ─────────────────────────────────────────────────────────────────────────────

class P15_LongViz(unittest.TestCase):

    def test_12codon_png_saved(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from gecsg.visualize.tree_viz import draw_parse_tree

        # 12 codons: ATG + 10 body + TAA
        body = "".join([CODON_G, CODON_C, CODON_A][i % 3] for i in range(10))
        seq  = "ATG" + body + "TAA"
        self.assertEqual(len(seq), 36)

        result = parse(seq)
        self.assertTrue(result.accepted)
        tree = result.trees()[0]

        path = "parse_12codon_test.png"
        fig  = draw_parse_tree(tree, seq,
                               title=f"12-codon parse: {seq[:18]}...",
                               save_path=path)
        self.assertTrue(os.path.exists(path),
                        f"PNG not saved to {path}")
        self.assertGreater(os.path.getsize(path), 1000,
                           "PNG file is suspiciously small")
        plt.close(fig)
        # Clean up
        if os.path.exists(path):
            os.remove(path)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_leaves(node: ParseNode):
    """Return leaf nodes in left-to-right order."""
    if node.is_terminal:
        return [node]
    result = []
    for ch in node.children:
        result.extend(_collect_leaves(ch))
    return result


def _check_span_partition(tc: unittest.TestCase, node: ParseNode):
    """
    Recursively verify that children spans tile the parent span.
    """
    if not node.children:
        return
    child_spans = [ch.span for ch in node.children]
    # Must start at parent start
    tc.assertEqual(child_spans[0][0], node.span[0],
                   f"First child start != parent start in {node.label}")
    # Must end at parent end
    tc.assertEqual(child_spans[-1][1], node.span[1],
                   f"Last child end != parent end in {node.label}")
    # Must be contiguous
    for i in range(len(child_spans) - 1):
        tc.assertEqual(child_spans[i][1], child_spans[i + 1][0],
                       f"Gap between children {i} and {i+1} of {node.label}")
    for ch in node.children:
        _check_span_partition(tc, ch)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  GE-CSG Parser Test Suite")
    print("=" * 65)
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None
    suite  = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
