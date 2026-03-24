"""
llm/hyenadna/test/test_context_length.py
Tests that exercise HyenaDNA's context-length handling.

Fast tests (always run):
  - Valid output across lengths the *loaded* checkpoint supports
  - Chunked embedding for sequences longer than model_max_length
  - Mean-pool invariance: output dim is constant regardless of length

Slow tests (-m slow):
  - Long-context checkpoints (32k / 160k / 1M) loaded per-test
  - Skipped automatically if the checkpoint isn't available or
    HYENA_SKIP_LONG_CONTEXT=1 is set
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
import hyenadna.deploy as deploy_mod
from hyenadna.deploy import DEVICE, embed, load_model
from hyenadna.test.fixtures import EXPECTED_EMBED_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_seq(length: int, pattern: str = "ATCG") -> str:
    return (pattern * (length // len(pattern) + 1))[:length]


@pytest.fixture(scope="session")
def model():
    tok, mdl = load_model()
    return tok, mdl


def _model_max(tokenizer) -> int:
    return tokenizer.model_max_length - 2


# ---------------------------------------------------------------------------
# 1. Valid output within the loaded checkpoint's native context window
# ---------------------------------------------------------------------------

class TestNativeContextRange:

    @pytest.mark.parametrize("length", [8, 64, 256, 512])
    def test_valid_output_short(self, model, length):
        tok, mdl = model
        emb = embed(make_seq(length), tok, mdl)
        assert emb.shape == (1, EXPECTED_EMBED_DIM)
        assert torch.isfinite(emb).all()
        assert emb.norm().item() > 0

    def test_at_model_max_len(self, model):
        """Sequence exactly at the checkpoint's context limit."""
        tok, mdl = model
        max_len = _model_max(tok)
        emb = embed(make_seq(max_len), tok, mdl)
        assert emb.shape == (1, EXPECTED_EMBED_DIM)
        assert torch.isfinite(emb).all()


# ---------------------------------------------------------------------------
# 2. Chunked embedding — sequences beyond model_max_length
#    deploy.embed() splits into chunks and averages; test that it doesn't crash
# ---------------------------------------------------------------------------

class TestChunkedEmbedding:

    def test_2x_max_len(self, model):
        tok, mdl = model
        length = (_model_max(tok) * 2) + 50
        emb = embed(make_seq(length), tok, mdl)
        assert emb.shape == (1, EXPECTED_EMBED_DIM)
        assert torch.isfinite(emb).all()

    def test_chunked_differs_from_truncated(self, model):
        """
        Chunked (full sequence) embedding should differ from embedding of
        only the first chunk — the extra context must affect the result.
        """
        tok, mdl = model
        max_len = _model_max(tok)
        long_seq  = make_seq(max_len * 2)
        short_seq = long_seq[:max_len]
        emb_long  = embed(long_seq,  tok, mdl)
        emb_short = embed(short_seq, tok, mdl)
        dist = (emb_long - emb_short).norm().item()
        assert dist > 0.0, "Chunked and truncated embeddings are identical"

    def test_chunked_is_deterministic(self, model):
        tok, mdl = model
        length = _model_max(tok) + 200
        seq  = make_seq(length)
        emb1 = embed(seq, tok, mdl)
        emb2 = embed(seq, tok, mdl)
        assert torch.allclose(emb1, emb2, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. Embedding dim is invariant to length
# ---------------------------------------------------------------------------

class TestMeanPoolInvariance:

    @pytest.mark.parametrize("len_a, len_b", [(64, 128), (256, 512)])
    def test_output_shape_same_regardless_of_length(self, model, len_a, len_b):
        tok, mdl = model
        emb_a = embed(make_seq(len_a), tok, mdl)
        emb_b = embed(make_seq(len_b), tok, mdl)
        assert emb_a.shape == emb_b.shape == (1, EXPECTED_EMBED_DIM)

    def test_longer_sequence_not_identical_to_shorter(self, model):
        tok, mdl = model
        seq_long  = make_seq(512)
        seq_short = seq_long[:256]
        dist = (embed(seq_long, tok, mdl) - embed(seq_short, tok, mdl)).norm().item()
        assert dist > 0.0, "512-bp and 256-bp embeddings are identical"


# ---------------------------------------------------------------------------
# 4. Long-context checkpoints (slow — need separate model download)
#    Each test loads the appropriate variant and skips if unavailable.
# ---------------------------------------------------------------------------

def _load_variant(variant_key: str):
    """Load a specific HyenaDNA variant; return (tok, mdl) or skip."""
    if os.environ.get("HYENA_SKIP_LONG_CONTEXT"):
        pytest.skip("HYENA_SKIP_LONG_CONTEXT is set")
    model_id = deploy_mod.VARIANTS.get(variant_key)
    if not model_id:
        pytest.skip(f"Unknown variant {variant_key!r}")
    try:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        mdl = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        mdl.to(DEVICE).eval()
        return tok, mdl
    except Exception as e:
        pytest.skip(f"Could not load {model_id}: {e}")


class TestLongContextCheckpoints:
    """
    Each variant has its own hidden_size (tiny=128, small/medium/large=256).
    Assert shape is (1, D) for any D > 0 — don't hardcode EXPECTED_EMBED_DIM
    which is only valid for the tiny checkpoint.
    """

    @pytest.mark.slow
    def test_32k_checkpoint(self):
        tok, mdl = _load_variant("32k")
        emb = embed(make_seq(32_000), tok, mdl)
        assert emb.ndim == 2 and emb.shape[0] == 1 and emb.shape[1] > 0
        assert torch.isfinite(emb).all()

    @pytest.mark.slow
    def test_160k_checkpoint(self):
        tok, mdl = _load_variant("160k")
        emb = embed(make_seq(160_000), tok, mdl)
        assert emb.ndim == 2 and emb.shape[0] == 1 and emb.shape[1] > 0
        assert torch.isfinite(emb).all()

    @pytest.mark.slow
    def test_1m_checkpoint(self):
        tok, mdl = _load_variant("1m")
        emb = embed(make_seq(1_000_000), tok, mdl)
        assert emb.ndim == 2 and emb.shape[0] == 1 and emb.shape[1] > 0
        assert torch.isfinite(emb).all()