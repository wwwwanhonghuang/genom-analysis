"""
run_profiler.py
===============
Systematic performance benchmark for GE-CSG parsers.

Sweeps sequence lengths from 1 to MAX_CODONS body codons (3 to MAX+2 total),
parses REPS_PER_LENGTH sequences of each length, and saves every timing row
to profile_data.csv.

Run:
    python run_profiler.py                  # full sweep
    python run_profiler.py --max 20         # only up to 20 body codons
    python run_profiler.py --reps 5         # 5 repetitions per length

CSV output:  profile_data.csv
"""

import sys, os, io, argparse, random
sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from gecsg.grammar.complete_dna_grammar   import complete_dna_grammar
from gecsg.grammar.stochastic_dna_grammar import (
    stochastic_complete_dna_grammar, HUMAN_CODON_USAGE,
)
from gecsg.parser.earley            import EquivariantEarleyParser
from gecsg.parser.stochastic_earley import StochasticEarleyParser
from gecsg.profiler                 import ParseProfiler

CSV_PATH = os.path.join(os.path.dirname(__file__),
                        "outputs", "performance_profile", "profile_data.csv")

# Codon pools (cycling body codons covering all 4 cosets)
START      = "ATG"
STOP_CODONS = ["TAA", "TAG", "TGA"]
BODY_POOL   = [
    "AAA","AAG","ATG","ACA",     # C0  A-starting
    "TTT","TCA","TGG","TAC",     # C1  T-starting
    "GCT","GGG","GTG","GAA",     # C2  G-starting
    "CAG","CTG","CGC","CCC",     # C3  C-starting
]

RNG = random.Random(42)


def make_seq(n_body: int) -> str:
    """Build a valid CDS: ATG + n_body random body codons + random stop."""
    body = "".join(RNG.choice(BODY_POOL) for _ in range(n_body))
    stop = RNG.choice(STOP_CODONS)
    return START + body + stop


def run_sweep(profiler, max_body, reps, label):
    """Sweep lengths 1..max_body, REPS repetitions each."""
    print(f"\n  Sweep: label={label!r}  max_body={max_body}  reps={reps}")
    for n_body in range(1, max_body + 1):
        for _ in range(reps):
            seq = make_seq(n_body)
            profiler.parse(seq, label=label)
        if n_body % 10 == 0 or n_body == max_body:
            print(f"    ... body={n_body}/{max_body} done")
    profiler.summary()


def main():
    ap = argparse.ArgumentParser(description="GE-CSG parser performance benchmark")
    ap.add_argument("--max",  type=int, default=50,
                    help="Max number of body codons (default 50)")
    ap.add_argument("--reps", type=int, default=10,
                    help="Repetitions per length (default 10)")
    ap.add_argument("--csv",  type=str, default=CSV_PATH,
                    help=f"Output CSV path (default: {CSV_PATH})")
    args = ap.parse_args()

    print("=" * 60)
    print("  GE-CSG Parser Performance Benchmark")
    print("=" * 60)
    print(f"  Max body codons : {args.max}  (total CDS = {args.max+2} max)")
    print(f"  Reps per length : {args.reps}")
    print(f"  CSV output      : {args.csv}")
    print(f"  Total parses    : {args.max * args.reps * 2} (2 parser types)")

    # ── Build parsers ─────────────────────────────────────────────────────
    print("\nBuilding parsers...")
    g_det   = complete_dna_grammar()
    g_sto_u = stochastic_complete_dna_grammar(p_terminal=0.2)
    g_sto_h = stochastic_complete_dna_grammar(p_terminal=0.2,
                                               codon_usage=HUMAN_CODON_USAGE)

    p_det   = EquivariantEarleyParser(g_det)
    p_sto_u = StochasticEarleyParser(g_sto_u)
    p_sto_h = StochasticEarleyParser(g_sto_h)
    print("Ready.")

    # ── Profilers ─────────────────────────────────────────────────────────
    prof_det   = ParseProfiler(p_det,   csv_path=args.csv,
                               grammar_tag="earley_deterministic")
    prof_sto_u = ParseProfiler(p_sto_u, csv_path=args.csv,
                               grammar_tag="earley_stochastic_uniform")
    prof_sto_h = ParseProfiler(p_sto_h, csv_path=args.csv,
                               grammar_tag="earley_stochastic_human")

    # ── Sweeps ────────────────────────────────────────────────────────────
    run_sweep(prof_det,   args.max, args.reps, label="deterministic")
    run_sweep(prof_sto_u, args.max, args.reps, label="stochastic_uniform")
    run_sweep(prof_sto_h, args.max, args.reps, label="stochastic_human")

    # ── Final report ──────────────────────────────────────────────────────
    from gecsg.profiler.profiler import load_profile_data
    rows = load_profile_data(args.csv)
    total = len(rows)
    labels = sorted({r["label"] for r in rows})
    print("\n" + "=" * 60)
    print(f"  CSV file: {args.csv}")
    print(f"  Total rows: {total}")
    print(f"  Labels present: {labels}")
    if rows:
        elapsed = [r["elapsed_ms"] for r in rows]
        codons  = [r["n_codons"]   for r in rows]
        print(f"  n_codons range  : {min(codons)} .. {max(codons)}")
        print(f"  elapsed_ms range: {min(elapsed):.3f} .. {max(elapsed):.3f}")
    print("=" * 60)
    print(f"\nDone.  Load results with:\n"
          f"  from gecsg.profiler.profiler import load_profile_data\n"
          f"  rows = load_profile_data('{args.csv}')\n")


if __name__ == "__main__":
    main()
