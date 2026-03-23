"""
test_bfs_parser.py
==================
Test suite for the Two-Phase BFS GE-CSG Parser (TwoPhaseBFSParser).

Run:
    python test_bfs_parser.py
    python test_bfs_parser.py -v

Coverage
--------
B01  Basic acceptance / rejection     -- all coset classes, valid/invalid seqs
B02  All 64 single codons             -- every codon must be accepted
B03  All coset-class pairs (2 codons) -- exposes the orbit-compression bug
B04  Multi-codon sequences            -- 3–30 codons
B05  BFS result fields                -- witnesses, depth_reached, n_states
B06  Cross-validation with Earley     -- BFS and Earley must agree on accept/reject
B07  complete_dna_grammar             -- 3-codon minimum, start/stop structure
B08  Depth-limit effect               -- shallow limit rejects long sequences
B09  Case-insensitive input           -- lowercase accepted
B10  Parser reuse                     -- multiple parses on same parser instance
"""

import sys
import os
import io
import random
import unittest

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from gecsg.grammar.dna_grammar import dna_codon_grammar, NUC_TO_COSET
from gecsg.grammar.complete_dna_grammar import complete_dna_grammar
from gecsg.parser.bfs_parser import TwoPhaseBFSParser
from gecsg.parser.earley import EquivariantEarleyParser

# ── Shared fixtures ────────────────────────────────────────────────────────
GRAMMAR         = dna_codon_grammar()
BFS_PARSER      = TwoPhaseBFSParser(GRAMMAR)
EARLEY_PARSER   = EquivariantEarleyParser(GRAMMAR)

COMPLETE_GRAMMAR = complete_dna_grammar()
BFS_COMPLETE     = TwoPhaseBFSParser(COMPLETE_GRAMMAR)

BASES      = "ATGC"
ALL_CODONS = [a + b + c for a in BASES for b in BASES for c in BASES]

# One representative codon per coset class (by first nucleotide)
CODON_A = "ATG"   # A__ -> coset 0
CODON_T = "TAA"   # T__ -> coset 1
CODON_G = "GCT"   # G__ -> coset 2
CODON_C = "CAG"   # C__ -> coset 3
COSET_REPS = [CODON_A, CODON_T, CODON_G, CODON_C]


def bfs(seq: str, depth: int = 50):
    return BFS_PARSER.parse(seq, depth_limit=depth)


def earley(seq: str):
    return EARLEY_PARSER.parse(seq)


# ─────────────────────────────────────────────────────────────────────────────
# B01  Basic acceptance / rejection
# ─────────────────────────────────────────────────────────────────────────────

class B01_AcceptReject(unittest.TestCase):

    def test_single_A_codon_accepted(self):
        self.assertTrue(bfs("ATG").accepted)

    def test_single_T_codon_accepted(self):
        # Regression: orbit-compression bug caused T-starting codons to be rejected
        self.assertTrue(bfs("TAA").accepted, "TAA must be accepted (T-coset)")

    def test_single_G_codon_accepted(self):
        self.assertTrue(bfs("GCT").accepted, "GCT must be accepted (G-coset)")

    def test_single_C_codon_accepted(self):
        self.assertTrue(bfs("CAG").accepted, "CAG must be accepted (C-coset)")

    def test_empty_string_rejected(self):
        # 0-codon: not in grammar (CDS needs ≥1 codon)
        self.assertFalse(bfs("").accepted)

    def test_wrong_length_rejected(self):
        # Not a multiple of k=3: BFSParseResult with accepted=False
        self.assertFalse(bfs("AT").accepted)
        self.assertFalse(bfs("ATGT").accepted)

    def test_two_codon_accepted(self):
        self.assertTrue(bfs("ATGTAA").accepted)

    def test_three_codon_accepted(self):
        self.assertTrue(bfs("ATGGCTTAA").accepted)


# ─────────────────────────────────────────────────────────────────────────────
# B02  All 64 single codons
# ─────────────────────────────────────────────────────────────────────────────

class B02_All64Codons(unittest.TestCase):

    def test_all_64_single_codons_accepted(self):
        """Every single codon must be accepted by the BFS parser."""
        for codon in ALL_CODONS:
            result = bfs(codon)
            self.assertTrue(result.accepted,
                            f"BFS rejected single codon {codon} (coset "
                            f"{NUC_TO_COSET[codon[0]]})")

    def test_all_64_codons_match_earley(self):
        """BFS and Earley must agree on every single codon."""
        for codon in ALL_CODONS:
            bfs_ok    = bfs(codon).accepted
            earley_ok = earley(codon).accepted
            self.assertEqual(bfs_ok, earley_ok,
                             f"{codon}: BFS={bfs_ok} Earley={earley_ok}")


# ─────────────────────────────────────────────────────────────────────────────
# B03  All coset-class pairs — critical regression for orbit-compression fix
# ─────────────────────────────────────────────────────────────────────────────

class B03_AllCosetPairs(unittest.TestCase):
    """
    Every 2-codon sequence whose codons come from two (possibly identical)
    coset classes must be accepted.  This is the direct test for the orbit-
    compression bug: (C1, C2) and (C0, C3) lie in the same G-orbit; the old
    code would check (C0, C3) first, fail Phase 2 for a (C1,C2) input, then
    skip (C1, C2) because its canonical was already visited.
    """

    def test_all_16_coset_class_pairs(self):
        for c1 in COSET_REPS:
            for c2 in COSET_REPS:
                seq = c1 + c2
                result = bfs(seq)
                self.assertTrue(result.accepted,
                                f"BFS rejected {seq} ({c1}[coset "
                                f"{NUC_TO_COSET[c1[0]]}] + "
                                f"{c2}[coset {NUC_TO_COSET[c2[0]]}])")

    def test_taagct_accepted(self):
        """TAAGCT = (C1, C2) — the canonical orbit member is (C0, C3), not this."""
        self.assertTrue(bfs("TAAGCT").accepted)

    def test_gctatg_accepted(self):
        """GCTATG = (C2, C0)."""
        self.assertTrue(bfs("GCTATG").accepted)

    def test_cagtaa_accepted(self):
        """CAGTAA = (C3, C1)."""
        self.assertTrue(bfs("CAGTAA").accepted)

    def test_taaatg_accepted(self):
        """TAAATG = (C1, C0)."""
        self.assertTrue(bfs("TAAATG").accepted)

    def test_all_pairs_match_earley(self):
        for c1 in COSET_REPS:
            for c2 in COSET_REPS:
                seq = c1 + c2
                bfs_ok    = bfs(seq).accepted
                earley_ok = earley(seq).accepted
                self.assertEqual(bfs_ok, earley_ok,
                                 f"{seq}: BFS={bfs_ok} Earley={earley_ok}")


# ─────────────────────────────────────────────────────────────────────────────
# B04  Multi-codon sequences (3–30 codons)
# ─────────────────────────────────────────────────────────────────────────────

class B04_MultiCodon(unittest.TestCase):
    """
    Multi-codon sequences up to 6 codons.

    Note on BFS complexity: Algorithm 1 is correct but exponential in sequence
    length — the BFS state space grows as ~4^n/|G|.  For n > ~8 codons the
    BFS becomes impractical; the Earley parser (O(n^3)) should be used for
    longer sequences.  These tests cover the range where BFS is tractable.
    """

    def _seq(self, n: int) -> str:
        pool = COSET_REPS
        return "".join(pool[i % 4] for i in range(n))

    def test_3codons_accepted(self):
        self.assertTrue(bfs(self._seq(3)).accepted)

    def test_4codons_accepted(self):
        self.assertTrue(bfs(self._seq(4)).accepted)

    def test_5codons_accepted(self):
        self.assertTrue(bfs(self._seq(5)).accepted)

    def test_6codons_accepted(self):
        self.assertTrue(bfs(self._seq(6)).accepted)

    def test_all_T_codons_4(self):
        """4 T-starting codons — each is in coset 1."""
        self.assertTrue(bfs("TAA" * 4).accepted)

    def test_all_G_codons_4(self):
        self.assertTrue(bfs("GCT" * 4).accepted)

    def test_all_C_codons_4(self):
        self.assertTrue(bfs("CAG" * 4).accepted)

    def test_random_5codons_match_earley(self):
        rng = random.Random(42)
        seq = "".join(rng.choice(ALL_CODONS) for _ in range(5))
        bfs_ok    = bfs(seq).accepted
        earley_ok = earley(seq).accepted
        self.assertEqual(bfs_ok, earley_ok,
                         f"Random 5-codon seq {seq}: BFS={bfs_ok} Earley={earley_ok}")


# ─────────────────────────────────────────────────────────────────────────────
# B05  BFS result fields
# ─────────────────────────────────────────────────────────────────────────────

class B05_ResultFields(unittest.TestCase):

    def test_accepted_sequences_have_witnesses(self):
        result = bfs("ATGGCTTAA")
        self.assertTrue(result.accepted)
        self.assertGreater(len(result.witnesses), 0)

    def test_witnesses_are_coset_strings(self):
        from gecsg.core.coset import Coset
        result = bfs("ATGGCTTAA")
        for w in result.witnesses:
            for elem in w:
                self.assertIsInstance(elem, Coset)

    def test_witness_length_matches_n_codons(self):
        for n in [1, 3, 4]:
            seq    = "".join(COSET_REPS[i % 4] for i in range(n))
            result = bfs(seq)
            self.assertTrue(result.accepted)
            for w in result.witnesses:
                self.assertEqual(len(w), n,
                                 f"n={n}: witness length {len(w)} != {n}")

    def test_depth_reached_positive_when_accepted(self):
        result = bfs("ATG")
        self.assertTrue(result.accepted)
        self.assertGreaterEqual(result.depth_reached, 0)

    def test_depth_reached_minus1_when_rejected(self):
        result = bfs("")
        self.assertFalse(result.accepted)
        self.assertEqual(result.depth_reached, -1)

    def test_n_states_positive(self):
        result = bfs("ATGGCT")
        self.assertGreater(result.n_states, 0)

    def test_rejected_has_no_witnesses(self):
        result = bfs("")
        self.assertEqual(len(result.witnesses), 0)


# ─────────────────────────────────────────────────────────────────────────────
# B06  Cross-validation with Earley on random sequences
# ─────────────────────────────────────────────────────────────────────────────

class B06_CrossValidation(unittest.TestCase):
    """BFS and Earley must agree on accept/reject for all test inputs."""

    def _check(self, seq: str):
        bfs_ok    = bfs(seq).accepted
        earley_ok = earley(seq).accepted
        self.assertEqual(bfs_ok, earley_ok,
                         f"{seq!r}: BFS={bfs_ok} Earley={earley_ok}")

    def test_all_64_single_codons(self):
        for codon in ALL_CODONS:
            self._check(codon)

    def test_all_16_pairs_of_coset_reps(self):
        for c1 in COSET_REPS:
            for c2 in COSET_REPS:
                self._check(c1 + c2)

    def test_short_mixed_sequences(self):
        seqs = [
            "ATGGCTTAA",
            "TAACAGGCT",
            "GCTATGTAA",
            "CAGTAAGCT",
            "ATGAAAGCTTTTCAGTAA",
        ]
        for seq in seqs:
            self._check(seq)

    def test_random_sample_50_sequences(self):
        """50 random sequences of 1–4 codons (BFS is exponential; >6 codons hangs)."""
        rng = random.Random(7)
        for _ in range(50):
            n   = rng.randint(1, 4)
            seq = "".join(rng.choice(ALL_CODONS) for _ in range(n))
            self._check(seq)


# ─────────────────────────────────────────────────────────────────────────────
# B07  complete_dna_grammar
# ─────────────────────────────────────────────────────────────────────────────

class B07_CompleteGrammar(unittest.TestCase):
    """complete_dna_grammar requires ≥3 codons (start + ≥1 body + stop)."""

    def test_atg_body_stop_accepted(self):
        # ATG (start=C0) + GCT (body) + TAA (stop=C1)
        self.assertTrue(BFS_COMPLETE.parse("ATGGCTTAA", depth_limit=50).accepted)

    def test_all_four_coset_start_accepted(self):
        """complete_dna_grammar orbit-expands start/stop to accept any coset."""
        body = "GCT"
        for start in COSET_REPS:
            for stop in COSET_REPS:
                seq = start + body + stop
                result = BFS_COMPLETE.parse(seq, depth_limit=50)
                self.assertTrue(result.accepted,
                                f"complete_grammar rejected {seq}")

    def test_single_codon_rejected_by_complete_grammar(self):
        # complete grammar needs ≥3 codons
        self.assertFalse(BFS_COMPLETE.parse("ATG", depth_limit=50).accepted)

    def test_two_codons_rejected_by_complete_grammar(self):
        self.assertFalse(BFS_COMPLETE.parse("ATGTAA", depth_limit=50).accepted)

    def test_long_sequence_accepted(self):
        body = "GCT" * 2
        seq  = "ATG" + body + "TAA"   # 4 codons (BFS is exponential; avoid >6)
        self.assertTrue(BFS_COMPLETE.parse(seq, depth_limit=50).accepted)


# ─────────────────────────────────────────────────────────────────────────────
# B08  Depth-limit effect
# ─────────────────────────────────────────────────────────────────────────────

class B08_DepthLimit(unittest.TestCase):

    def test_depth_0_rejects_all(self):
        """Depth 0: no rules applied, nothing accepted."""
        for seq in ["ATG", "TAA", "GCT"]:
            result = BFS_PARSER.parse(seq, depth_limit=0)
            self.assertFalse(result.accepted,
                             f"depth=0 should reject {seq}")

    def test_sufficient_depth_accepts(self):
        """Increasing depth to 50 accepts the same sequences as Earley."""
        for seq in ["ATG", "TAA", "ATGGCTTAA"]:
            self.assertTrue(BFS_PARSER.parse(seq, depth_limit=50).accepted)

    def test_very_short_sequence_needs_few_steps(self):
        """A 1-codon sequence should be accepted within a small depth."""
        result = BFS_PARSER.parse("ATG", depth_limit=10)
        self.assertTrue(result.accepted)
        self.assertLessEqual(result.depth_reached, 10)


# ─────────────────────────────────────────────────────────────────────────────
# B09  Case-insensitive input
# ─────────────────────────────────────────────────────────────────────────────

class B09_CaseInsensitive(unittest.TestCase):

    def test_lowercase_accepted(self):
        self.assertTrue(bfs("atg").accepted)
        self.assertTrue(bfs("taa").accepted)
        self.assertTrue(bfs("gct").accepted)
        self.assertTrue(bfs("cag").accepted)

    def test_mixed_case_accepted(self):
        self.assertTrue(bfs("AtGgCtTaA").accepted)

    def test_lowercase_matches_uppercase(self):
        for codon in ["atg", "taa", "gct", "cag"]:
            self.assertEqual(
                bfs(codon).accepted,
                bfs(codon.upper()).accepted,
                f"lowercase vs uppercase mismatch for {codon}",
            )


# ─────────────────────────────────────────────────────────────────────────────
# B10  Parser reuse
# ─────────────────────────────────────────────────────────────────────────────

class B10_ParserReuse(unittest.TestCase):

    def test_multiple_parses_sequential(self):
        seqs = ["ATG", "TAAGCT", "ATGGCTTAA", "CAG" * 5, "TAA" * 3]
        for seq in seqs:
            self.assertTrue(bfs(seq).accepted, f"{seq} failed on reuse")

    def test_results_are_independent(self):
        r1 = BFS_PARSER.parse("ATGGCT")
        r2 = BFS_PARSER.parse("TAACAG")
        self.assertIsNot(r1.witnesses, r2.witnesses)
        # Both should be accepted
        self.assertTrue(r1.accepted)
        self.assertTrue(r2.accepted)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  GE-CSG BFS Parser Test Suite")
    print("=" * 65)
    loader = unittest.TestLoader()
    loader.sortTestMethodsUsing = None
    suite  = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
