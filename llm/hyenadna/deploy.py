"""
HyenaDNA — Tier 1 deploy script

Model variants (select via HYENA_MODEL_ID env var or --variant flag):
  1k    LongSafari/hyenadna-tiny-1k-seqlen-hf      1,026 bp   1.6M params  (default)
  32k   LongSafari/hyenadna-small-32k-seqlen-hf   32,768 bp   7.4M params
  160k  LongSafari/hyenadna-medium-160k-seqlen-hf 160,000 bp  28M  params
  450k  LongSafari/hyenadna-medium-450k-seqlen-hf 450,000 bp  28M  params
  1m    LongSafari/hyenadna-large-1m-seqlen-hf  1,000,000 bp  49M  params

Context limits are hard — each checkpoint's positional state is fixed at
training time.  Sequences longer than model_max_length are chunked and
their embeddings averaged.
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModel

VARIANTS = {
    "1k":   "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    "32k":  "LongSafari/hyenadna-small-32k-seqlen-hf",
    "160k": "LongSafari/hyenadna-medium-160k-seqlen-hf",
    "450k": "LongSafari/hyenadna-medium-450k-seqlen-hf",
    "1m":   "LongSafari/hyenadna-large-1m-seqlen-hf",
}

MODEL_ID = os.environ.get("HYENA_MODEL_ID", VARIANTS["1k"])
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# Set after load_model() — reflects the actual checkpoint's hard limit
MODEL_MAX_LEN: int = 1000

# TP53 exon 5 — frequent mutation hotspot in cancer
TEST_SEQ = (
    "ATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAATGAGCCCCATGAGCCGCCTGAGGGCCCAGAGGGCC"
    "CATGGAGGATCCCCAGCCCTGGGCGTCAAGAGCCACTTGTACTGGCCCTTCTTGCAGACTGTGTCCAGG"
)


def load_model():
    global MODEL_MAX_LEN
    print(f"[hyenadna] Loading {MODEL_ID} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model     = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()
    # model_max_length includes special tokens; subtract 2 for [CLS]/[SEP]
    MODEL_MAX_LEN = tokenizer.model_max_length - 2
    n = sum(p.numel() for p in model.parameters())
    print(f"[hyenadna] Parameters   : {n:,}")
    print(f"[hyenadna] Max seq len  : {MODEL_MAX_LEN} bp")
    return tokenizer, model


def embed(seq: str, tokenizer, model) -> torch.Tensor:
    """
    Embed a DNA sequence.  If len(seq) > MODEL_MAX_LEN the sequence is split
    into non-overlapping chunks; their embeddings are averaged.
    """
    max_len = tokenizer.model_max_length - 2  # exclude special tokens

    if len(seq) <= max_len:
        inputs = tokenizer(seq, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        return out.last_hidden_state.mean(dim=1)

    # Chunked embedding for sequences that exceed the checkpoint's context
    chunks = [seq[i:i + max_len] for i in range(0, len(seq), max_len)]
    chunk_embs = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
        chunk_embs.append(out.last_hidden_state.mean(dim=1))
    return torch.stack(chunk_embs).mean(dim=0)


def smoke_test(tokenizer, model):
    print("[hyenadna] Smoke test on TP53 exon 5 ...")
    emb = embed(TEST_SEQ, tokenizer, model)
    print(f"[hyenadna] Embedding shape : {emb.shape}")
    print(f"[hyenadna] Embedding norm  : {emb.norm().item():.4f}")
    print("[hyenadna] ✓ Smoke test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",    action="store_true")
    parser.add_argument("--variant", choices=list(VARIANTS), default=None,
                        help="Model variant (overrides HYENA_MODEL_ID env var)")
    parser.add_argument("--seq",     type=str, default=None)
    args = parser.parse_args()

    if args.variant:
        MODEL_ID = VARIANTS[args.variant]  # noqa: F811

    tokenizer, model = load_model()

    if args.test or args.seq is None:
        smoke_test(tokenizer, model)
    else:
        emb = embed(args.seq, tokenizer, model)
        print(f"Embedding shape          : {emb.shape}")
        print(f"Embedding (first 8 dims) : {emb[0, :8].tolist()}")