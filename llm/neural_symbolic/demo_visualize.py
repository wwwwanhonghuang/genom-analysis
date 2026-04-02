"""
llm/neural_symbolic/demo_visualize.py
Parse a set of cancer-relevant sequences with NeuralPCFGParser and save
tree + arc + combined PNGs to llm/neural_symbolic/output/.

Usage:
    python llm/neural_symbolic/demo_visualize.py
    python llm/neural_symbolic/demo_visualize.py --model-id LongSafari/hyenadna-small-32k-seqlen-hf
    python llm/neural_symbolic/demo_visualize.py --no-encoder   # random hidden states (fast demo)
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))

from neural_symbolic.parser   import NeuralPCFGParser
from neural_symbolic.visualize import save_tree_png, save_arc_png, save_combined_png

OUTPUT_DIR = Path(__file__).parent / "output"

# ---------------------------------------------------------------------------
# Demo sequences
# ---------------------------------------------------------------------------

SEQUENCES = {
    "TP53_exon5_WT": (
        "ATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAATGAGCCCC"
        "ATGAGCCGCCTGAGGGCCCAGAGGGCCC"
    ),
    "TP53_exon5_R175H": (
        "ATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAATGAGCCCC"
        "ATGAGCCGCCTGAGGGCCCAGAGGGCCA"   # G→A at codon 175
    ),
    "KRAS_WT": (
        "ATGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGT"
        "AGGCAAGAGTGCCTTGACGATACAGCTA"
    ),
    "KRAS_G12D": (
        "ATGACTGAATATAAACTTGTGGTAGTTGGAGCTGATGGCGT"  # G→A pos 34
        "AGGCAAGAGTGCCTTGACGATACAGCTA"
    ),
    "BRCA1_splice": (
        "CAGCTACAATTTGCTTTTACACACTTTAGTTTGTTTATTTT"
        "TCTAAAGCATCTGATAGTTGGAGGTTTG"
    ),
    "EGFR_exon19_WT": (
        "CATGGTGGAGGGCATGAACCTGGCCCTCAAGAAAGTAGCCAT"
        "CATCACAGAGGGCATGAGCTGGGTCATC"
    ),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Neural PCFG parse tree visualizer")
    parser.add_argument("--model-id",   default=None,
                        help="HyenaDNA model ID (default: hyenadna-tiny-1k)")
    parser.add_argument("--no-encoder", action="store_true",
                        help="Use random hidden states instead of loading HyenaDNA")
    parser.add_argument("--seq-len",    type=int, default=16,
                        help="Max tokens per sequence for demo (default: 16)")
    parser.add_argument("--dpi",        type=int, default=150)
    parser.add_argument("--out-dir",    default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load encoder ---
    if args.no_encoder:
        print("[demo] Using random hidden states (--no-encoder)")
        tokenizer = None
        encoder   = None
        embed_dim = 128
    else:
        try:
            from transformers import AutoTokenizer, AutoModel
            model_id  = args.model_id or "LongSafari/hyenadna-tiny-1k-seqlen-hf"
            print(f"[demo] Loading encoder: {model_id}")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            encoder   = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
            embed_dim = encoder.config.d_model
        except Exception as e:
            print(f"[demo] Encoder load failed ({e}), falling back to random states")
            tokenizer = None
            encoder   = None
            embed_dim = 128

    # --- Parser ---
    pcfg = NeuralPCFGParser(embed_dim=embed_dim, hidden_dim=64)
    pcfg.eval()

    print(f"\n[demo] Output directory: {out_dir}\n")
    print(f"  {'Sequence':<25}  {'logZ':>10}  {'depth':>6}  {'leaves':>6}  {'Files'}")
    print("  " + "-" * 75)

    for name, seq in SEQUENCES.items():
        n   = min(len(seq), args.seq_len)
        sub = seq[:n]

        # Encode
        if encoder is not None and tokenizer is not None:
            max_len = tokenizer.model_max_length - 2
            inputs  = tokenizer(sub[:max_len], return_tensors="pt")
            with torch.no_grad():
                hidden = encoder(**inputs).last_hidden_state
            n = hidden.shape[1]
        else:
            torch.manual_seed(sum(ord(c) for c in name))
            hidden = torch.randn(1, n, embed_dim)

        lengths = torch.tensor([n])

        with torch.no_grad():
            tree, log_Z = pcfg.parse(hidden, sub[:n], lengths)

        # Save PNG files
        title_base = name.replace("_", " ")
        tree_path  = out_dir / f"{name}_tree.png"
        arc_path   = out_dir / f"{name}_arcs.png"
        comb_path  = out_dir / f"{name}_combined.png"

        save_tree_png(
            tree, sequence=sub[:n],
            title=f"{title_base} — constituency tree",
            path=tree_path, dpi=args.dpi,
        )
        save_arc_png(
            tree, sequence=sub[:n],
            title=f"{title_base} — span arcs",
            path=arc_path, dpi=args.dpi,
        )
        save_combined_png(
            tree, sequence=sub[:n],
            title=title_base,
            path=comb_path, dpi=args.dpi,
        )

        files = f"{tree_path.name}, {arc_path.name}, {comb_path.name}"
        print(f"  {name:<25}  {log_Z.item():>10.1f}  {tree.depth:>6}  {tree.num_leaves:>6}  {files}")

    print(f"\n[demo] ✓ {len(SEQUENCES) * 3} PNG files saved to {out_dir}/\n")


if __name__ == "__main__":
    main()