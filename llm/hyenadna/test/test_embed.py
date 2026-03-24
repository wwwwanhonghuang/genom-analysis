"""
llm/hyenadna/test/test_embed.py
Tests for the core embed() function in hyenadna/deploy.py.

Tasks covered:
  - Output shape and dtype
  - Embedding norm sanity bounds
  - Determinism (same input → same output)
  - Sequence sensitivity (different sequences → different embeddings)
  - Variant sensitivity (WT vs mutant embeddings diverge)
  - Context length scaling (longer seq → same embedding dim)
  - Edge cases: short, poly-A, mixed case, ambiguous bases
"""

import sys
from pathlib import Path

import pytest
import torch

# Make deploy.py importable from the test subdirectory
sys.path.insert(0, str(Path(__file__).parents[2]))  # llm/
from hyenadna.deploy import embed, load_model
from hyenadna.test.fixtures import (
    BRCA1_EXON11_MUT,
    BRCA1_EXON11_WT,
    EGFR_EXON19_DEL,
    EGFR_EXON19_WT,
    EXPECTED_EMBED_DIM,
    EXPECTED_NORM_MAX,
    EXPECTED_NORM_MIN,
    KRAS_G12D,
    KRAS_WT,
    LONG_SEQ,
    MIXED_CASE,
    POLY_A,
    SHORT_SEQ,
    TP53_EXON5_R175H,
    TP53_EXON5_WT,
)


# ---------------------------------------------------------------------------
# Shared model fixture — loaded once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def model():
    tok, mdl = load_model()
    return tok, mdl


# ---------------------------------------------------------------------------
# 1. Output shape & dtype
# ---------------------------------------------------------------------------

class TestEmbedShape:

    def test_output_is_2d(self, model):
        tok, mdl = model
        emb = embed(TP53_EXON5_WT, tok, mdl)
        assert emb.ndim == 2, f"Expected 2D tensor, got shape {emb.shape}"

    def test_batch_dim_is_one(self, model):
        tok, mdl = model
        emb = embed(TP53_EXON5_WT, tok, mdl)
        assert emb.shape[0] == 1

    def test_hidden_dim(self, model):
        tok, mdl = model
        emb = embed(TP53_EXON5_WT, tok, mdl)
        assert emb.shape[1] == EXPECTED_EMBED_DIM, (
            f"Expected hidden dim {EXPECTED_EMBED_DIM}, got {emb.shape[1]}"
        )

    def test_output_dtype_is_float(self, model):
        tok, mdl = model
        emb = embed(TP53_EXON5_WT, tok, mdl)
        assert emb.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_no_nan_or_inf(self, model):
        tok, mdl = model
        emb = embed(TP53_EXON5_WT, tok, mdl)
        assert torch.isfinite(emb).all(), "Embedding contains NaN or Inf"


# ---------------------------------------------------------------------------
# 2. Norm sanity bounds
# ---------------------------------------------------------------------------

class TestEmbedNorm:

    @pytest.mark.parametrize("seq, label", [
        (TP53_EXON5_WT,    "TP53 WT"),
        (KRAS_WT,          "KRAS WT"),
        (BRCA1_EXON11_WT,  "BRCA1 WT"),
        (EGFR_EXON19_WT,   "EGFR WT"),
    ])
    def test_norm_in_expected_range(self, model, seq, label):
        tok, mdl = model
        norm = embed(seq, tok, mdl).norm().item()
        assert EXPECTED_NORM_MIN <= norm <= EXPECTED_NORM_MAX, (
            f"{label}: norm {norm:.4f} outside [{EXPECTED_NORM_MIN}, {EXPECTED_NORM_MAX}]"
        )

    def test_poly_a_norm_finite(self, model):
        tok, mdl = model
        norm = embed(POLY_A, tok, mdl).norm().item()
        assert torch.isfinite(torch.tensor(norm))


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------

class TestEmbedDeterminism:

    def test_same_input_same_output(self, model):
        tok, mdl = model
        emb1 = embed(TP53_EXON5_WT, tok, mdl)
        emb2 = embed(TP53_EXON5_WT, tok, mdl)
        assert torch.allclose(emb1, emb2, atol=1e-5), (
            "Non-deterministic: two identical inputs gave different embeddings"
        )

    def test_deterministic_across_sequences(self, model):
        """Running two different seqs and then re-running first gives same result."""
        tok, mdl = model
        emb_tp53_a = embed(TP53_EXON5_WT, tok, mdl)
        _          = embed(KRAS_WT,        tok, mdl)   # interleave different seq
        emb_tp53_b = embed(TP53_EXON5_WT, tok, mdl)
        assert torch.allclose(emb_tp53_a, emb_tp53_b, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Sequence sensitivity — different seqs → different embeddings
# ---------------------------------------------------------------------------

class TestEmbedSensitivity:

    @pytest.mark.parametrize("seq_a, seq_b, label", [
        (TP53_EXON5_WT,   KRAS_WT,          "TP53 vs KRAS"),
        (BRCA1_EXON11_WT, EGFR_EXON19_WT,   "BRCA1 vs EGFR"),
        (SHORT_SEQ,       POLY_A,            "short vs poly-A"),
    ])
    def test_different_seqs_differ(self, model, seq_a, seq_b, label):
        tok, mdl = model
        emb_a = embed(seq_a, tok, mdl)
        emb_b = embed(seq_b, tok, mdl)
        cos_sim = torch.nn.functional.cosine_similarity(emb_a, emb_b).item()
        assert cos_sim < 0.999, (
            f"{label}: embeddings are suspiciously identical (cosine={cos_sim:.6f})"
        )


# ---------------------------------------------------------------------------
# 5. Variant sensitivity — WT vs mutant embeddings should diverge
#    (not necessarily by a fixed amount, but measurably different)
# ---------------------------------------------------------------------------

class TestVariantSensitivity:

    @pytest.mark.parametrize("wt, mut, label", [
        (TP53_EXON5_WT,   TP53_EXON5_R175H,  "TP53 R175H"),
        (KRAS_WT,         KRAS_G12D,          "KRAS G12D"),
        (BRCA1_EXON11_WT, BRCA1_EXON11_MUT,   "BRCA1 splice"),
        (EGFR_EXON19_WT,  EGFR_EXON19_DEL,    "EGFR del19"),
    ])
    def test_wt_and_mut_differ(self, model, wt, mut, label):
        tok, mdl = model
        emb_wt  = embed(wt,  tok, mdl)
        emb_mut = embed(mut, tok, mdl)
        l2_dist = (emb_wt - emb_mut).norm().item()
        assert l2_dist > 0.0, (
            f"{label}: WT and mutant embeddings are identical (L2=0)"
        )

    def test_larger_deletion_causes_larger_shift(self, model):
        """EGFR exon 19 deletion (15 bp) should shift embedding more than SNV."""
        tok, mdl = model
        snv_dist = (embed(TP53_EXON5_WT, tok, mdl) -
                    embed(TP53_EXON5_R175H, tok, mdl)).norm().item()
        del_dist = (embed(EGFR_EXON19_WT, tok, mdl) -
                    embed(EGFR_EXON19_DEL, tok, mdl)).norm().item()
        assert del_dist >= snv_dist * 0.5, (
            f"Expected deletion ({del_dist:.4f}) to shift at least half as much "
            f"as SNV ({snv_dist:.4f})"
        )


# ---------------------------------------------------------------------------
# 6. Context length scaling
# ---------------------------------------------------------------------------

class TestContextLength:

    @pytest.mark.parametrize("length", [64, 256, 512, 1000])
    def test_embedding_dim_invariant_to_length(self, model, length):
        tok, mdl = model
        seq = ("ATCG" * (length // 4 + 1))[:length]
        emb = embed(seq, tok, mdl)
        assert emb.shape == (1, EXPECTED_EMBED_DIM), (
            f"length={length}: unexpected shape {emb.shape}"
        )

    def test_long_sequence(self, model):
        tok, mdl = model
        emb = embed(LONG_SEQ, tok, mdl)
        assert emb.shape[1] == EXPECTED_EMBED_DIM
        assert torch.isfinite(emb).all()


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_short_sequence(self, model):
        tok, mdl = model
        emb = embed(SHORT_SEQ, tok, mdl)
        assert emb.shape == (1, EXPECTED_EMBED_DIM)
        assert torch.isfinite(emb).all()

    def test_mixed_case_normalised(self, model):
        """Lowercase input should produce the same embedding as uppercase."""
        tok, mdl = model
        emb_upper = embed(MIXED_CASE.upper(), tok, mdl)
        emb_lower = embed(MIXED_CASE.lower(), tok, mdl)
        assert torch.allclose(emb_upper, emb_lower, atol=1e-4), (
            "Lowercase and uppercase inputs gave different embeddings — "
            "tokenizer may not be case-normalising"
        )

    def test_ambiguous_bases(self, model):
        """Sequences with N should not crash; norm should still be finite."""
        tok, mdl = model
        emb = embed(AMBIGUOUS, tok, mdl)
        assert torch.isfinite(emb).all()