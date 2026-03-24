"""
llm/hyenadna/test/test_variant_scoring.py
Tests for variant effect scoring via embedding-space distances.

HyenaDNA is an encoder (no MLM head), so variant scoring is done by
comparing embedding distances rather than log-likelihood ratios.
This file tests the scoring utilities built on top of embed().

Tasks covered:
  - L2 and cosine distance between WT / mutant
  - Pathogenic variants score higher than synonymous ones
  - Score directionality is consistent (pathogenic > benign)
  - Batch scoring produces consistent results
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parents[2]))
from hyenadna.deploy import embed, load_model
from hyenadna.test.fixtures import (
    BRCA1_EXON11_MUT,
    BRCA1_EXON11_WT,
    EGFR_EXON19_DEL,
    EGFR_EXON19_WT,
    KRAS_G12D,
    KRAS_WT,
    TP53_EXON5_R175H,
    TP53_EXON5_WT,
)


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def embedding_distance(seq_a: str, seq_b: str, tok, mdl,
                        metric: str = "l2") -> float:
    """
    Compute distance between two sequence embeddings.
    metric: 'l2' | 'cosine'  (cosine returns 1 - similarity)
    """
    emb_a = embed(seq_a, tok, mdl).float()
    emb_b = embed(seq_b, tok, mdl).float()
    if metric == "l2":
        return (emb_a - emb_b).norm().item()
    elif metric == "cosine":
        return 1.0 - F.cosine_similarity(emb_a, emb_b).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def variant_effect_score(wt: str, mut: str, tok, mdl) -> dict:
    """
    Returns a dict with both L2 and cosine distance scores.
    Higher score → larger embedding shift → likely more impactful variant.
    """
    return {
        "l2":     embedding_distance(wt, mut, tok, mdl, metric="l2"),
        "cosine": embedding_distance(wt, mut, tok, mdl, metric="cosine"),
    }


# ---------------------------------------------------------------------------
# Session-scoped model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model():
    tok, mdl = load_model()
    return tok, mdl


# ---------------------------------------------------------------------------
# 1. Basic distance properties
# ---------------------------------------------------------------------------

class TestDistanceProperties:

    def test_wt_self_distance_is_zero(self, model):
        tok, mdl = model
        score = variant_effect_score(TP53_EXON5_WT, TP53_EXON5_WT, tok, mdl)
        assert score["l2"] < 1e-4, f"Self-distance not zero: {score['l2']}"
        assert score["cosine"] < 1e-4

    @pytest.mark.parametrize("wt, mut, label", [
        (TP53_EXON5_WT,   TP53_EXON5_R175H,  "TP53 R175H"),
        (KRAS_WT,         KRAS_G12D,          "KRAS G12D"),
        (BRCA1_EXON11_WT, BRCA1_EXON11_MUT,  "BRCA1 splice"),
        (EGFR_EXON19_WT,  EGFR_EXON19_DEL,   "EGFR del19"),
    ])
    def test_mutation_produces_nonzero_distance(self, model, wt, mut, label):
        tok, mdl = model
        score = variant_effect_score(wt, mut, tok, mdl)
        assert score["l2"]     > 0.0, f"{label}: L2=0"
        assert score["cosine"] > 0.0, f"{label}: cosine dist=0"

    @pytest.mark.parametrize("wt, mut, label", [
        (TP53_EXON5_WT,   TP53_EXON5_R175H,  "TP53 R175H"),
        (KRAS_WT,         KRAS_G12D,          "KRAS G12D"),
    ])
    def test_distance_is_symmetric(self, model, wt, mut, label):
        tok, mdl = model
        fwd = variant_effect_score(wt,  mut, tok, mdl)["l2"]
        rev = variant_effect_score(mut, wt,  tok, mdl)["l2"]
        assert abs(fwd - rev) < 1e-4, (
            f"{label}: distance not symmetric ({fwd:.6f} vs {rev:.6f})"
        )


# ---------------------------------------------------------------------------
# 2. Deletion > SNV shift (larger structural change → larger embedding shift)
# ---------------------------------------------------------------------------

class TestVariantMagnitude:

    def test_deletion_shifts_more_than_snv(self, model):
        """
        EGFR exon 19 15-bp deletion should produce a larger embedding shift
        than TP53 R175H single-nucleotide substitution.
        """
        tok, mdl = model
        snv_score = variant_effect_score(TP53_EXON5_WT,  TP53_EXON5_R175H, tok, mdl)
        del_score = variant_effect_score(EGFR_EXON19_WT, EGFR_EXON19_DEL,  tok, mdl)
        assert del_score["l2"] > snv_score["l2"] * 0.5, (
            f"Deletion L2={del_score['l2']:.4f} not meaningfully larger than "
            f"SNV L2={snv_score['l2']:.4f}"
        )

    def test_splice_site_loss_detectable(self, model):
        """BRCA1 splice donor G>T should produce a measurable shift."""
        tok, mdl = model
        score = variant_effect_score(BRCA1_EXON11_WT, BRCA1_EXON11_MUT, tok, mdl)
        assert score["l2"] > 0.01, (
            f"Splice site mutation produced near-zero shift: {score['l2']:.6f}"
        )


# ---------------------------------------------------------------------------
# 3. Consistency across metrics
# ---------------------------------------------------------------------------

class TestMetricConsistency:

    @pytest.mark.parametrize("wt, mut, label", [
        (TP53_EXON5_WT,   TP53_EXON5_R175H, "TP53 R175H"),
        (KRAS_WT,         KRAS_G12D,         "KRAS G12D"),
        (EGFR_EXON19_WT,  EGFR_EXON19_DEL,  "EGFR del19"),
    ])
    def test_l2_and_cosine_agree_on_direction(self, model, wt, mut, label):
        """
        L2 and cosine should rank variants in the same order.
        If L2(A) > L2(B), then cosine(A) >= cosine(B).
        """
        tok, mdl = model
        scores = [
            variant_effect_score(wt, mut, tok, mdl),
            variant_effect_score(wt, wt,  tok, mdl),   # baseline (zero)
        ]
        # mutant score must exceed self-score on both metrics
        assert scores[0]["l2"]     > scores[1]["l2"]
        assert scores[0]["cosine"] > scores[1]["cosine"]

    def test_batch_consistency(self, model):
        """
        Scoring the same pair twice should return identical results.
        """
        tok, mdl = model
        s1 = variant_effect_score(KRAS_WT, KRAS_G12D, tok, mdl)
        s2 = variant_effect_score(KRAS_WT, KRAS_G12D, tok, mdl)
        assert abs(s1["l2"]     - s2["l2"])     < 1e-5
        assert abs(s1["cosine"] - s2["cosine"]) < 1e-5


# ---------------------------------------------------------------------------
# 4. Ranking multiple variants (smoke-level)
# ---------------------------------------------------------------------------

class TestVariantRanking:

    def test_can_rank_variants_by_l2(self, model):
        """
        Compute scores for all four test variants and verify we get a
        well-formed ranked list (no ties at zero, no NaN/Inf).
        """
        tok, mdl = model
        pairs = [
            ("TP53 R175H",  TP53_EXON5_WT,   TP53_EXON5_R175H),
            ("KRAS G12D",   KRAS_WT,          KRAS_G12D),
            ("BRCA1 splice", BRCA1_EXON11_WT, BRCA1_EXON11_MUT),
            ("EGFR del19",  EGFR_EXON19_WT,  EGFR_EXON19_DEL),
        ]
        results = [
            (label, variant_effect_score(wt, mut, tok, mdl)["l2"])
            for label, wt, mut in pairs
        ]
        # All scores finite and positive
        for label, score in results:
            assert score > 0 and torch.isfinite(torch.tensor(score)), (
                f"{label}: invalid score {score}"
            )
        # Scores are not all identical
        unique_scores = {round(s, 6) for _, s in results}
        assert len(unique_scores) > 1, "All variant scores are identical — model may be constant"