"""
test_stochastic.py
==================
Comprehensive tests for the stochastic GE-CSG grammar and Viterbi Earley parser.

Tests T01-T15 cover:
  T01  Grammar construction
  T02  Weight normalization
  T03  p_terminal parameter
  T04  Basic parsing acceptance/rejection
  T05  Manual log-prob calculation
  T06  Longer sequences have lower prob
  T07  p_terminal effect on ORF probability
  T08  Human codon usage
  T09  Probability properties
  T10  Node log-probs
  T11  Rank ordering
  T12  Parser reuse
  T13  Case-insensitive input
  T14  Visualization smoke test
  T15  Uniform vs human codon usage
"""

import unittest
import math
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)

from gecsg.grammar.stochastic_dna_grammar import (
    stochastic_complete_dna_grammar,
    HUMAN_CODON_USAGE,
    NUC_TO_COSET,
)
from gecsg.parser.stochastic_earley import (
    StochasticEarleyParser,
    StochasticParseResult,
    _make_breaking_index,
    _compute_tree_log_prob,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _build_grammar(p=0.2, usage=None):
    return stochastic_complete_dna_grammar(p_terminal=p, codon_usage=usage)

def _build_parser(p=0.2, usage=None):
    return StochasticEarleyParser(_build_grammar(p=p, usage=usage))


# ─────────────────────────────────────────────────────────────────────────────
# T01 — Grammar construction
# ─────────────────────────────────────────────────────────────────────────────

class T01_GrammarConstruction(unittest.TestCase):

    def setUp(self):
        self.g = _build_grammar()

    def test_orbit_reps(self):
        self.assertEqual(self.g.n_orbits, 7,
                         f"Expected 7 orbit reps, got {self.g.n_orbits}")

    def test_full_rules(self):
        self.assertEqual(self.g.n_full_rules, 16,
                         f"Expected 16 full rules, got {self.g.n_full_rules}")

    def test_breaking_rules(self):
        self.assertEqual(self.g.n_breaking, 64,
                         f"Expected 64 breaking rules, got {self.g.n_breaking}")

    def test_weights_in_range(self):
        for r in self.g.full_rules:
            self.assertGreater(r.weight, 0.0,
                               f"Rule weight must be > 0: {r}")
            self.assertLessEqual(r.weight, 1.0,
                                 f"Rule weight must be <= 1: {r}")

    def test_breaking_probs_in_range(self):
        for br in self.g.breaking_rules:
            self.assertGreater(br.prob, 0.0,
                               f"Breaking prob must be > 0: {br}")
            self.assertLessEqual(br.prob, 1.0 + 1e-9,
                                 f"Breaking prob must be <= 1: {br}")

    def test_grammar_frozen(self):
        self.assertTrue(self.g.is_frozen)

    def test_start_symbol(self):
        self.assertEqual(self.g.start.name, "Gene")

    def test_k(self):
        self.assertEqual(self.g.k, 3)


# ─────────────────────────────────────────────────────────────────────────────
# T02 — Weight normalization
# ─────────────────────────────────────────────────────────────────────────────

class T02_WeightNormalization(unittest.TestCase):

    def setUp(self):
        self.g = _build_grammar()

    def test_generation_rule_weights_sum_to_one(self):
        from collections import defaultdict
        sums = defaultdict(float)
        for r in self.g.full_rules:
            sums[r.lhs.name] += r.weight
        for nt_name, s in sums.items():
            self.assertAlmostEqual(
                s, 1.0, places=9,
                msg=f"LHS {nt_name}: weight sum = {s}, expected 1.0"
            )

    def test_breaking_rule_probs_sum_to_one(self):
        from collections import defaultdict
        sums = defaultdict(float)
        for br in self.g.breaking_rules:
            sums[br.coset.index] += br.prob
        for ci, s in sums.items():
            self.assertAlmostEqual(
                s, 1.0, places=9,
                msg=f"Coset C{ci}: prob sum = {s}, expected 1.0"
            )

    def test_four_cosets_covered(self):
        coset_indices = {br.coset.index for br in self.g.breaking_rules}
        self.assertEqual(coset_indices, {0, 1, 2, 3})

    def test_human_codon_usage_normalization(self):
        g = _build_grammar(usage=HUMAN_CODON_USAGE)
        from collections import defaultdict
        sums = defaultdict(float)
        for br in g.breaking_rules:
            sums[br.coset.index] += br.prob
        for ci, s in sums.items():
            self.assertAlmostEqual(
                s, 1.0, places=6,
                msg=f"Human usage coset C{ci}: sum = {s}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# T03 — p_terminal parameter
# ─────────────────────────────────────────────────────────────────────────────

class T03_PTerminalParameter(unittest.TestCase):

    def _orf_weights(self, g):
        """Return (w_terminate, w_extend) for ORF rules."""
        w_term = w_ext = None
        for r in g.full_rules:
            if r.lhs.name == "ORF":
                rhs_names = [s.name if hasattr(s, 'name') else str(s) for s in r.rhs]
                if len(r.rhs) == 1:
                    w_term = r.weight
                elif len(r.rhs) == 2:
                    w_ext = r.weight
        return w_term, w_ext

    def test_p01(self):
        p = 0.1
        g = _build_grammar(p=p)
        wt, we = self._orf_weights(g)
        self.assertAlmostEqual(wt, p, places=12)
        self.assertAlmostEqual(we, 1.0 - p, places=12)

    def test_p03(self):
        p = 0.3
        g = _build_grammar(p=p)
        wt, we = self._orf_weights(g)
        self.assertAlmostEqual(wt, p, places=12)
        self.assertAlmostEqual(we, 1.0 - p, places=12)

    def test_p05(self):
        p = 0.5
        g = _build_grammar(p=p)
        wt, we = self._orf_weights(g)
        self.assertAlmostEqual(wt, p, places=12)
        self.assertAlmostEqual(we, 1.0 - p, places=12)

    def test_p09(self):
        p = 0.9
        g = _build_grammar(p=p)
        wt, we = self._orf_weights(g)
        self.assertAlmostEqual(wt, p, places=12)
        self.assertAlmostEqual(we, 1.0 - p, places=12)

    def test_invalid_p_zero(self):
        with self.assertRaises((ValueError, Exception)):
            stochastic_complete_dna_grammar(p_terminal=0.0)

    def test_invalid_p_one(self):
        with self.assertRaises((ValueError, Exception)):
            stochastic_complete_dna_grammar(p_terminal=1.0)

    def test_various_sequences_accepted(self):
        for p in [0.1, 0.3, 0.5, 0.9]:
            parser = _build_parser(p=p)
            for seq in ["ATGGCTTAA", "ATGAAATAA", "ATGCTGTAA"]:
                r = parser.parse(seq)
                self.assertTrue(r.accepted,
                                f"p={p}: {seq} should be accepted")


# ─────────────────────────────────────────────────────────────────────────────
# T04 — Basic parsing
# ─────────────────────────────────────────────────────────────────────────────

class T04_BasicParsing(unittest.TestCase):

    def setUp(self):
        self.parser = _build_parser()

    def test_valid_3codon_accepted(self):
        r = self.parser.parse("ATGGCTTAA")
        self.assertTrue(r.accepted)
        self.assertGreater(r.log_prob, -math.inf)
        self.assertLess(r.log_prob, 0.0)

    def test_valid_4codon_accepted(self):
        r = self.parser.parse("ATGAAAGCTTAA")
        self.assertTrue(r.accepted)
        self.assertGreater(r.log_prob, -math.inf)

    def test_valid_6codon_accepted(self):
        r = self.parser.parse("ATGAAAGCTTTTCAGTAA")
        self.assertTrue(r.accepted)

    def test_valid_tag_stop(self):
        r = self.parser.parse("ATGGCTTAG")
        self.assertTrue(r.accepted)

    def test_valid_tga_stop(self):
        r = self.parser.parse("ATGGCTTGA")
        self.assertTrue(r.accepted)

    def test_too_short_rejected(self):
        r = self.parser.parse("ATGTAA")  # only 2 codons, no body codon
        self.assertFalse(r.accepted)
        self.assertEqual(r.log_prob, -math.inf)

    def test_wrong_length_raises(self):
        with self.assertRaises(ValueError):
            self.parser.parse("ATGG")  # length 4, not multiple of 3

    def test_rejected_log_prob_is_neg_inf(self):
        r = self.parser.parse("ATGTAA")  # no body codon
        self.assertEqual(r.log_prob, -math.inf)

    def test_rejected_prob_is_zero(self):
        r = self.parser.parse("ATGTAA")
        self.assertEqual(r.prob, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# T05 — Manual log-prob calculation
# ─────────────────────────────────────────────────────────────────────────────

class T05_LogProbManual(unittest.TestCase):
    """
    For "ATGGCTTAA" (ATG|GCT|TAA) with p_terminal=0.2 and uniform breaking:

    Derivation:
      Gene -> CDS                       weight=1.0
      CDS  -> StartCodon ORF StopCodon  weight=1.0
      StartCodon -> C0                  weight=0.25  (ATG is C0)
      ORF  -> BodyCodon                 weight=0.2   (p_terminal)
      BodyCodon -> C2                   weight=0.25  (GCT is C2)
      StopCodon -> C1                   weight=0.25  (TAA is C1)
      C0 -> ATG                         prob=1/16
      C2 -> GCT                         prob=1/16
      C1 -> TAA                         prob=1/16

    Expected log P = log(1) + log(1) + log(0.25) + log(0.2) + log(0.25)
                   + log(0.25) + log(1/16) + log(1/16) + log(1/16)
                   = 3*log(0.25) + 3*log(1/16) + log(0.2)
    """

    def setUp(self):
        self.g = _build_grammar(p=0.2)
        self.parser = StochasticEarleyParser(self.g)
        self.seq = "ATGGCTTAA"

    def test_log_prob_matches_manual(self):
        expected = (
            3 * math.log(0.25)
            + 3 * math.log(1.0 / 16.0)
            + math.log(0.2)
        )
        r = self.parser.parse(self.seq)
        self.assertTrue(r.accepted)
        self.assertAlmostEqual(r.log_prob, expected, places=10,
                               msg=f"log_prob={r.log_prob}, expected={expected}")

    def test_prob_positive(self):
        r = self.parser.parse(self.seq)
        self.assertGreater(r.prob, 0.0)

    def test_prob_less_than_one(self):
        r = self.parser.parse(self.seq)
        self.assertLess(r.prob, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# T06 — Longer sequence has lower log-prob
# ─────────────────────────────────────────────────────────────────────────────

class T06_LongerSequenceLowerProb(unittest.TestCase):

    def setUp(self):
        self.parser = _build_parser(p=0.2)

    def test_more_body_codons_lower_prob(self):
        seq3 = "ATGGCTTAA"                   # 3 codons (1 body)
        seq4 = "ATGAAAGCTTAA"               # 4 codons (2 body)
        seq6 = "ATGAAAGCTTTTCAGTAA"         # 6 codons (4 body)

        r3 = self.parser.parse(seq3)
        r4 = self.parser.parse(seq4)
        r6 = self.parser.parse(seq6)

        self.assertTrue(r3.accepted)
        self.assertTrue(r4.accepted)
        self.assertTrue(r6.accepted)

        self.assertGreater(r3.log_prob, r4.log_prob,
                           "3-codon should have higher log_prob than 4-codon")
        self.assertGreater(r4.log_prob, r6.log_prob,
                           "4-codon should have higher log_prob than 6-codon")

    def test_adding_body_codon_reduces_prob(self):
        """Each additional body codon must reduce log_prob."""
        seq_base = "ATGGCTTAA"   # 1 body
        seq_ext  = "ATGGCTAAATAA"  # 2 body
        r1 = self.parser.parse(seq_base)
        r2 = self.parser.parse(seq_ext)
        self.assertTrue(r1.accepted)
        self.assertTrue(r2.accepted)
        self.assertGreater(r1.log_prob, r2.log_prob)


# ─────────────────────────────────────────────────────────────────────────────
# T07 — p_terminal effect
# ─────────────────────────────────────────────────────────────────────────────

class T07_PTerminalEffect(unittest.TestCase):
    """Higher p_terminal -> ORF->BodyCodon has higher weight -> short ORF more probable."""

    def test_short_orf_more_probable_with_high_p(self):
        # 1 body codon CDS
        seq = "ATGGCTTAA"
        # With high p_terminal, P(ORF->BodyCodon) = p_terminal is larger
        lp_low  = _build_parser(p=0.1).parse(seq).log_prob
        lp_mid  = _build_parser(p=0.4).parse(seq).log_prob
        lp_high = _build_parser(p=0.8).parse(seq).log_prob

        self.assertGreater(lp_mid, lp_low,
                           "p=0.4 should give higher log_prob than p=0.1 for 1-body seq")
        self.assertGreater(lp_high, lp_mid,
                           "p=0.8 should give higher log_prob than p=0.4 for 1-body seq")

    def test_long_orf_more_probable_with_low_p(self):
        # 4 body codons CDS
        seq = "ATGAAAGCTTTTCAGAAATAA"  # 4 body codons
        lp_low  = _build_parser(p=0.1).parse(seq).log_prob
        lp_high = _build_parser(p=0.8).parse(seq).log_prob

        self.assertGreater(lp_low, lp_high,
                           "p=0.1 (low) should give higher log_prob for 4-body seq than p=0.8")


# ─────────────────────────────────────────────────────────────────────────────
# T08 — Human codon usage
# ─────────────────────────────────────────────────────────────────────────────

class T08_HumanCodonUsage(unittest.TestCase):

    def setUp(self):
        self.parser_human   = _build_parser(usage=HUMAN_CODON_USAGE)
        self.parser_uniform = _build_parser()

    def test_high_usage_codon_more_probable(self):
        # CTG (Leu, high usage C3) vs CGA (Arg, low usage C3) as body codon
        seq_ctg = "ATG" + "CTG" + "TAA"  # body=CTG (C3, prob~0.115 in human)
        seq_cga = "ATG" + "CGA" + "TAA"  # body=CGA (C3, prob~0.012 in human)

        lp_ctg = self.parser_human.parse(seq_ctg).log_prob
        lp_cga = self.parser_human.parse(seq_cga).log_prob

        self.assertGreater(lp_ctg, lp_cga,
                           "CTG (high usage) should have higher log_prob than CGA (low usage)")

    def test_atg_higher_than_ata_body(self):
        # ATG (C0, high usage) vs ATA (C0, low usage) as body codon
        seq_atg = "ATG" + "ATG" + "TAA"  # body=ATG  (C0, ~0.065)
        seq_ata = "ATG" + "ATA" + "TAA"  # body=ATA  (C0, ~0.018)

        lp_atg = self.parser_human.parse(seq_atg).log_prob
        lp_ata = self.parser_human.parse(seq_ata).log_prob

        self.assertGreater(lp_atg, lp_ata,
                           "ATG body should have higher log_prob than ATA body (human usage)")

    def test_human_differs_from_uniform(self):
        seq = "ATGCTGTAA"
        lp_human   = self.parser_human.parse(seq).log_prob
        lp_uniform = self.parser_uniform.parse(seq).log_prob
        # They should differ since CTG has non-uniform usage
        self.assertNotAlmostEqual(lp_human, lp_uniform, places=6)


# ─────────────────────────────────────────────────────────────────────────────
# T09 — Probability properties
# ─────────────────────────────────────────────────────────────────────────────

class T09_ProbProperties(unittest.TestCase):

    def setUp(self):
        self.parser = _build_parser()

    def _check_seq(self, seq):
        r = self.parser.parse(seq)
        if not r.accepted:
            return
        self.assertLess(r.log_prob, 0.0,
                        f"log_prob should be < 0 for accepted {seq}")
        self.assertGreater(r.prob, 0.0,
                           f"prob should be > 0 for accepted {seq}")
        self.assertLessEqual(r.prob, 1.0,
                             f"prob should be <= 1 for accepted {seq}")
        self.assertAlmostEqual(r.prob, math.exp(r.log_prob), places=10)

    def test_3codon(self):
        self._check_seq("ATGGCTTAA")

    def test_4codon(self):
        self._check_seq("ATGAAAGCTTAA")

    def test_6codon(self):
        self._check_seq("ATGAAAGCTTTTCAGTAA")

    def test_prob_exp_log_consistency(self):
        r = self.parser.parse("ATGGCTTAA")
        self.assertAlmostEqual(r.prob, math.exp(r.log_prob), places=12)


# ─────────────────────────────────────────────────────────────────────────────
# T10 — Node log-probs
# ─────────────────────────────────────────────────────────────────────────────

class T10_NodeLogProbs(unittest.TestCase):

    def setUp(self):
        self.parser = _build_parser()

    def test_node_log_probs_nonempty(self):
        r = self.parser.parse("ATGGCTTAA")
        self.assertTrue(r.accepted)
        nlp = r.node_log_probs()
        self.assertGreater(len(nlp), 0)

    def test_all_tree_nodes_present(self):
        r = self.parser.parse("ATGGCTTAA")
        tree = r.trees()[0]
        nlp  = r.node_log_probs()

        # Walk all nodes and check presence
        def walk(node):
            self.assertIn(id(node), nlp,
                          f"Node {node.label} missing from node_log_probs")
            for ch in node.children:
                walk(ch)
        walk(tree)

    def test_gene_root_contribution(self):
        """Gene -> CDS has weight=1.0, so log contribution is 0.0."""
        r    = self.parser.parse("ATGGCTTAA")
        tree = r.trees()[0]
        nlp  = r.node_log_probs()
        # Root is Gene node
        self.assertEqual(tree.label, "Gene")
        self.assertAlmostEqual(nlp[id(tree)], 0.0, places=12,
                               msg="Gene root log contribution should be 0.0 (weight=1.0)")

    def test_cds_contribution(self):
        """CDS -> StartCodon ORF StopCodon has weight=1.0."""
        r    = self.parser.parse("ATGGCTTAA")
        tree = r.trees()[0]
        nlp  = r.node_log_probs()
        cds  = tree.children[0]
        self.assertEqual(cds.label, "CDS")
        self.assertAlmostEqual(nlp[id(cds)], 0.0, places=12)

    def test_sum_equals_total_log_prob(self):
        """Sum of all per-node contributions must equal total log_prob."""
        r   = self.parser.parse("ATGGCTTAA")
        nlp = r.node_log_probs()
        total = sum(nlp.values())
        self.assertAlmostEqual(total, r.log_prob, places=10)


# ─────────────────────────────────────────────────────────────────────────────
# T11 — Rank ordering
# ─────────────────────────────────────────────────────────────────────────────

class T11_RankOrdering(unittest.TestCase):

    def setUp(self):
        self.parser = _build_parser(usage=HUMAN_CODON_USAGE)

    def test_rank_returns_sorted_descending(self):
        seqs = [
            "ATG" + "CTG" * 2 + "TAA",   # CTG: high usage
            "ATG" + "CGA" * 2 + "TAA",   # CGA: low usage
            "ATG" + "GCT" * 2 + "TAA",   # GCT: medium usage
        ]
        ranked = self.parser.rank(seqs)
        self.assertEqual(len(ranked), 3)
        # Verify descending order
        for i in range(len(ranked) - 1):
            self.assertGreaterEqual(ranked[i][1], ranked[i+1][1],
                                    f"Rank not descending at position {i}")

    def test_rank_excludes_rejected(self):
        seqs = [
            "ATGGCTTAA",   # valid
            "ATGTAA",      # rejected (no body codon)
            "ATGAAATAA",   # valid
        ]
        ranked = self.parser.rank(seqs)
        seqs_ranked = [s for s, _ in ranked]
        self.assertNotIn("ATGTAA", seqs_ranked)
        self.assertIn("ATGGCTTAA", seqs_ranked)
        self.assertIn("ATGAAATAA", seqs_ranked)

    def test_rank_returns_tuples(self):
        ranked = self.parser.rank(["ATGGCTTAA"])
        self.assertEqual(len(ranked), 1)
        seq, lp = ranked[0]
        self.assertIsInstance(seq, str)
        self.assertIsInstance(lp, float)

    def test_rank_empty_input(self):
        ranked = self.parser.rank([])
        self.assertEqual(ranked, [])


# ─────────────────────────────────────────────────────────────────────────────
# T12 — Parser reuse
# ─────────────────────────────────────────────────────────────────────────────

class T12_ParserReuse(unittest.TestCase):

    def setUp(self):
        self.parser = _build_parser()

    def test_same_seq_same_log_prob(self):
        seq = "ATGGCTTAA"
        lp1 = self.parser.parse(seq).log_prob
        lp2 = self.parser.parse(seq).log_prob
        self.assertEqual(lp1, lp2)

    def test_different_seqs_independent(self):
        # Use sequences with different lengths so log_probs must differ
        seq1 = "ATGGCTTAA"           # 3 codons (1 body)
        seq2 = "ATGAAAGCTTAA"        # 4 codons (2 body)
        r1 = self.parser.parse(seq1)
        r2 = self.parser.parse(seq2)
        # Both accepted
        self.assertTrue(r1.accepted)
        self.assertTrue(r2.accepted)
        # log_probs differ: different number of codons -> different prob
        self.assertNotEqual(r1.log_prob, r2.log_prob,
                            "Sequences with different lengths must have different log_probs")

    def test_parser_handles_multiple_sequences(self):
        seqs = ["ATGGCTTAA", "ATGAAAGCTTAA", "ATGCTGTAA"]
        for seq in seqs:
            r = self.parser.parse(seq)
            self.assertTrue(r.accepted, f"{seq} should be accepted")
            self.assertGreater(r.log_prob, -math.inf)


# ─────────────────────────────────────────────────────────────────────────────
# T13 — Case-insensitive input
# ─────────────────────────────────────────────────────────────────────────────

class T13_CaseInsensitive(unittest.TestCase):

    def setUp(self):
        self.parser = _build_parser()

    def test_lowercase_accepted(self):
        r = self.parser.parse("atggcttaa")
        self.assertTrue(r.accepted)

    def test_lowercase_same_log_prob(self):
        r_upper = self.parser.parse("ATGGCTTAA")
        r_lower = self.parser.parse("atggcttaa")
        self.assertAlmostEqual(r_upper.log_prob, r_lower.log_prob, places=12)

    def test_mixed_case_accepted(self):
        r = self.parser.parse("AtGgCtTaA")
        self.assertTrue(r.accepted)

    def test_mixed_case_same_log_prob(self):
        r_upper = self.parser.parse("ATGGCTTAA")
        r_mixed = self.parser.parse("AtGgCtTaA")
        self.assertAlmostEqual(r_upper.log_prob, r_mixed.log_prob, places=12)


# ─────────────────────────────────────────────────────────────────────────────
# T14 — Visualization smoke test
# ─────────────────────────────────────────────────────────────────────────────

class T14_VisualizationSmoke(unittest.TestCase):

    def setUp(self):
        import matplotlib
        matplotlib.use("Agg")

    def test_draw_stochastic_parse_tree_returns_figure(self):
        import matplotlib.pyplot as plt
        from gecsg.visualize.prob_tree_viz import draw_stochastic_parse_tree

        parser = _build_parser()
        r = parser.parse("ATGGCTTAA")
        self.assertTrue(r.accepted)

        tree = r.trees()[0]
        nlp  = r.node_log_probs()

        fig = draw_stochastic_parse_tree(
            tree, "ATGGCTTAA", nlp, r.log_prob,
            title="Smoke test"
        )
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, plt.Figure)
        plt.close(fig)

    def test_draw_with_save_path(self):
        import matplotlib.pyplot as plt
        import tempfile
        from gecsg.visualize.prob_tree_viz import draw_stochastic_parse_tree

        parser = _build_parser()
        r = parser.parse("ATGGCTTAA")
        tree = r.trees()[0]
        nlp  = r.node_log_probs()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name

        try:
            fig = draw_stochastic_parse_tree(
                tree, "ATGGCTTAA", nlp, r.log_prob,
                save_path=path, title="Save test"
            )
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)
            plt.close(fig)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_draw_4codon(self):
        import matplotlib.pyplot as plt
        from gecsg.visualize.prob_tree_viz import draw_stochastic_parse_tree

        parser = _build_parser()
        r = parser.parse("ATGAAAGCTTAA")
        self.assertTrue(r.accepted)

        tree = r.trees()[0]
        nlp  = r.node_log_probs()

        fig = draw_stochastic_parse_tree(
            tree, "ATGAAAGCTTAA", nlp, r.log_prob
        )
        self.assertIsNotNone(fig)
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# T15 — Uniform vs human codon usage
# ─────────────────────────────────────────────────────────────────────────────

class T15_UniformVsHumanUsage(unittest.TestCase):

    def test_atg_higher_prob_in_human_vs_uniform(self):
        """ATG has usage 0.065 in human, which is > 1/16 = 0.0625 (uniform)."""
        g_uniform = _build_grammar()
        g_human   = _build_grammar(usage=HUMAN_CODON_USAGE)

        # Find breaking prob for ATG (C0) in each grammar
        def get_break_prob(grammar, codon):
            target = tuple(codon)
            for br in grammar.breaking_rules:
                if br.string == target:
                    return br.prob
            return None

        p_uniform = get_break_prob(g_uniform, "ATG")
        p_human   = get_break_prob(g_human,   "ATG")

        self.assertAlmostEqual(p_uniform, 1.0 / 16.0, places=12)
        # ATG human usage > uniform (0.065 > 0.0625)
        self.assertGreater(p_human, p_uniform,
                           f"ATG human prob ({p_human}) should be > uniform ({p_uniform})")

    def test_log_prob_difference_matches_breaking_prob_ratio(self):
        """
        For a 3-codon CDS ATG|X|TAA, the log_prob difference between human
        and uniform grammars equals log(p_human_X / p_uniform_X)
        + log(p_human_ATG / p_uniform_ATG) + log(p_human_TAA / p_uniform_TAA).
        """
        seq = "ATGCTGTAA"  # ATG (C0) | CTG (C3) | TAA (C1)

        p_unif = _build_parser()
        p_human = _build_parser(usage=HUMAN_CODON_USAGE)

        lp_unif  = p_unif.parse(seq).log_prob
        lp_human = p_human.parse(seq).log_prob

        # Compute expected difference
        g_uniform = _build_grammar()
        g_human   = _build_grammar(usage=HUMAN_CODON_USAGE)

        def get_prob(grammar, codon):
            target = tuple(codon)
            for br in grammar.breaking_rules:
                if br.string == target:
                    return br.prob
            return 1.0 / 16.0

        diff_expected = (
            math.log(get_prob(g_human,   "ATG")) - math.log(get_prob(g_uniform, "ATG"))
            + math.log(get_prob(g_human, "CTG")) - math.log(get_prob(g_uniform, "CTG"))
            + math.log(get_prob(g_human, "TAA")) - math.log(get_prob(g_uniform, "TAA"))
        )

        diff_actual = lp_human - lp_unif

        self.assertAlmostEqual(diff_actual, diff_expected, places=9,
                               msg="log_prob difference must equal sum of breaking prob log ratios")

    def test_cga_lower_than_uniform_in_human(self):
        """CGA has usage 0.012 in human, which is < 1/16 = 0.0625 (uniform)."""
        g_uniform = _build_grammar()
        g_human   = _build_grammar(usage=HUMAN_CODON_USAGE)

        def get_break_prob(grammar, codon):
            target = tuple(codon)
            for br in grammar.breaking_rules:
                if br.string == target:
                    return br.prob
            return None

        p_uniform = get_break_prob(g_uniform, "CGA")
        p_human   = get_break_prob(g_human,   "CGA")

        self.assertAlmostEqual(p_uniform, 1.0 / 16.0, places=12)
        self.assertLess(p_human, p_uniform,
                        f"CGA human prob ({p_human}) should be < uniform ({p_uniform})")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        T01_GrammarConstruction,
        T02_WeightNormalization,
        T03_PTerminalParameter,
        T04_BasicParsing,
        T05_LogProbManual,
        T06_LongerSequenceLowerProb,
        T07_PTerminalEffect,
        T08_HumanCodonUsage,
        T09_ProbProperties,
        T10_NodeLogProbs,
        T11_RankOrdering,
        T12_ParserReuse,
        T13_CaseInsensitive,
        T14_VisualizationSmoke,
        T15_UniformVsHumanUsage,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed
    print(f"\n{'='*60}")
    print(f"Total: {total}  |  Passed: {passed}  |  Failed: {failed}")
    print(f"{'='*60}")
    sys.exit(0 if result.wasSuccessful() else 1)
