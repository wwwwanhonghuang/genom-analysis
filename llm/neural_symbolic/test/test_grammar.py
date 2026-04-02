"""
llm/neural_symbolic/test/test_grammar.py
Unit tests for the genomic PCFG grammar — no model loading required.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from neural_symbolic.grammar import (
    BINARY_RULES, NUM_NT, NUM_T, NT_NAMES, TERMINAL_VOCAB,
    NT_GENE, NT_EXON, NT_INTRON, NT_CODON, NT_PROMOTER,
    NT_SPLICE, NT_MOTIF, NT_REGULATORY, NT_UTR,
    ROOT_PRIORS, rule_mask, root_prior_tensor,
)


class TestNTAlphabet:

    def test_all_nt_have_names(self):
        assert len(NT_NAMES) == NUM_NT
        for i in range(NUM_NT):
            assert i in NT_NAMES, f"NT index {i} missing from NT_NAMES"

    def test_nt_names_are_strings(self):
        for nt, name in NT_NAMES.items():
            assert isinstance(name, str) and len(name) > 0

    def test_no_duplicate_names(self):
        names = list(NT_NAMES.values())
        assert len(names) == len(set(names)), "Duplicate NT names"

    def test_terminal_vocab_size(self):
        assert NUM_T == 5, f"Expected 5 terminals (ACGTN), got {NUM_T}"
        assert set(TERMINAL_VOCAB) == {"A", "C", "G", "T", "N"}

    def test_terminal_indices_contiguous(self):
        indices = sorted(TERMINAL_VOCAB.values())
        assert indices == list(range(NUM_T))


class TestBinaryRules:

    def test_rules_not_empty(self):
        assert len(BINARY_RULES) > 0

    def test_all_rule_indices_in_range(self):
        for rule in BINARY_RULES:
            for attr in ("parent", "left", "right"):
                val = getattr(rule, attr)
                assert 0 <= val < NUM_NT, (
                    f"Rule {rule}: {attr}={val} out of range [0, {NUM_NT})"
                )

    def test_gene_rule_exists(self):
        """GENE must appear as parent in at least one rule."""
        parents = {r.parent for r in BINARY_RULES}
        assert NT_GENE in parents, "GENE has no production rules"

    def test_all_nt_are_reachable(self):
        """Every NT should appear as either parent or child in some rule."""
        mentioned = set()
        for r in BINARY_RULES:
            mentioned |= {r.parent, r.left, r.right}
        for nt in range(NUM_NT):
            assert nt in mentioned, f"{NT_NAMES[nt]} is unreachable in grammar"

    def test_prior_values_are_finite(self):
        for r in BINARY_RULES:
            assert isinstance(r.prior, (int, float))
            assert -1e6 < r.prior < 1e6, f"Prior out of range: {r}"

    def test_biologically_motivated_rules(self):
        """Check specific rules that should exist for cancer genomics."""
        rule_set = {(r.parent, r.left, r.right) for r in BINARY_RULES}
        # Promoter should precede transcript in gene structure
        assert (NT_GENE, NT_PROMOTER, NT_EXON) in rule_set or \
               any(r.parent == NT_GENE and r.left == NT_PROMOTER for r in BINARY_RULES), \
               "Missing GENE → PROMOTER ... rule"
        # Exon should be decomposable to codons
        assert any(r.parent == NT_EXON and (r.left == NT_CODON or r.right == NT_CODON)
                   for r in BINARY_RULES), "Missing EXON → CODON rule"
        # Splice signal should be part of intron
        assert any(r.parent == NT_INTRON and (r.left == NT_SPLICE or r.right == NT_SPLICE)
                   for r in BINARY_RULES), "Missing INTRON → SPLICE rule"


class TestRuleMask:

    def test_mask_shape(self):
        mask = rule_mask()
        assert mask.shape == (NUM_NT, NUM_NT, NUM_NT)

    def test_mask_is_binary(self):
        mask = rule_mask()
        vals = mask.unique()
        assert set(vals.tolist()).issubset({0.0, 1.0}), "Mask is not binary"

    def test_mask_has_correct_entries(self):
        mask = rule_mask()
        for r in BINARY_RULES:
            assert mask[r.parent, r.left, r.right] == 1.0, (
                f"Rule ({NT_NAMES[r.parent]} → {NT_NAMES[r.left]} {NT_NAMES[r.right]}) "
                f"not set in mask"
            )

    def test_mask_has_no_extra_entries(self):
        mask = rule_mask()
        allowed = {(r.parent, r.left, r.right) for r in BINARY_RULES}
        nonzero = mask.nonzero(as_tuple=False)
        for idx in nonzero:
            triple = tuple(idx.tolist())
            assert triple in allowed, f"Mask has unexpected entry {triple}"

    def test_mask_dtype(self):
        mask = rule_mask()
        assert mask.dtype == torch.float32


class TestRootPrior:

    def test_prior_shape(self):
        prior = root_prior_tensor()
        assert prior.shape == (NUM_NT,)

    def test_prior_is_log_normalized(self):
        prior = root_prior_tensor()
        log_sum = torch.logsumexp(prior, dim=0)
        assert abs(log_sum.item()) < 1e-4, f"Root prior not log-normalized: {log_sum}"

    def test_gene_has_highest_prior(self):
        prior = root_prior_tensor()
        best = prior.argmax().item()
        assert best == NT_GENE, (
            f"Expected GENE to have highest root prior, got {NT_NAMES[best]}"
        )

    def test_all_priors_are_negative(self):
        """Log probabilities must all be ≤ 0."""
        prior = root_prior_tensor()
        assert (prior <= 0).all(), "Root prior contains positive log-prob"

    def test_root_priors_cover_expected_nt(self):
        prior = root_prior_tensor()
        for nt in (NT_GENE, NT_EXON):
            assert prior[nt] > -10.0, f"{NT_NAMES[nt]} has near-zero root prior"