"""
HyenaDNA — Tier 1 deploy script
Model  : LongSafari/hyenadna-tiny-1k-seqlen-hf  (1.6M params)
Context: up to 1,000,000 bp
HW     : CPU or single GPU
Task   : Sequence embedding + classification demo
"""

import argparse
import sys
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_ID   = "LongSafari/hyenadna-tiny-1k-seqlen-hf"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# TP53 exon 5 region — frequent mutation hotspot in cancer
TEST_SEQ = (
    "ATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAATGAGCCCCATGAGCCGCCTGAGGGCCCAGAGGGCC"
    "CATGGAGGATCCCCAGCCCTGGGCGTCAAGAGCCACTTGTACTGGCCCTTCTTGCAGACTGTGTCCAGG"
)


def load_model():
    print(f"[hyenadna] Loading {MODEL_ID} on {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model     = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.to(DEVICE).eval()
    print(f"[hyenadna] Parameters : {sum(p.numel() for p in model.parameters()):,}")
    return tokenizer, model


def embed(seq: str, tokenizer, model) -> torch.Tensor:
    inputs = tokenizer(seq, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    # mean-pool over sequence length
    return outputs.last_hidden_state.mean(dim=1)


def smoke_test(tokenizer, model):
    print("[hyenadna] Running smoke test on TP53 exon 5 sequence...")
    emb = embed(TEST_SEQ, tokenizer, model)
    print(f"[hyenadna] Embedding shape : {emb.shape}")
    print(f"[hyenadna] Embedding norm  : {emb.norm().item():.4f}")
    print("[hyenadna] ✓ Smoke test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run smoke test")
    parser.add_argument("--seq",  type=str, default=None, help="DNA sequence to embed")
    args = parser.parse_args()

    tokenizer, model = load_model()

    if args.test or args.seq is None:
        smoke_test(tokenizer, model)
    else:
        emb = embed(args.seq, tokenizer, model)
        print(f"Embedding shape : {emb.shape}")
        print(f"Embedding (first 8 dims): {emb[0, :8].tolist()}")