"""
visualize_nested.py
====================
Parse and visualize sequences using the Center-Branch CSG Grammar.

Grammar structure (produces GENUINELY NESTED trees, NOT chains):
  Gene        -> CDS
  CDS         -> StartCodon  Body  StopCodon
  [CSG] Body  -> BodyCodon  Body  BodyCodon   [left_ctx=StartCodon]
  [CF]  Body  -> BodyCodon  Body  BodyCodon   (inner levels, no context)
  [CF]  Body  -> BodyCodon  BodyCodon         (base case)

  StartCodon  ->_1 C0   (A-starting)
  StopCodon   ->_1 C1   (T-starting)
  BodyCodon   ->_1 C0   (all 4 cosets via G-orbit)

Language: start + even body codons (>= 2) + stop.
Tree depth: n_body / 2  (e.g. 100 body codons -> 50 levels deep).

BFS is used for correctness checking on small sequences.
Earley (cf_relaxation=True) is used for all parse tree construction.

Run:
    python visualization/visualize_nested.py
"""

import sys, os, io, random, time
_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, _ROOT)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gecsg.grammar.center_branch_grammar import center_branch_grammar
from gecsg.parser.bfs_parser import TwoPhaseBFSParser
from gecsg.parser.earley import EquivariantEarleyParser
from gecsg.visualize.tree_viz import draw_parse_tree

# ── Setup ────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(_ROOT, "outputs_nested")
os.makedirs(OUT_DIR, exist_ok=True)

print("Building Center-Branch CSG Grammar ...")
GRAMMAR_BFS    = center_branch_grammar(cf_relaxation=False)  # BFS: genuine CSG
GRAMMAR_EARLEY = center_branch_grammar(cf_relaxation=True)   # Earley: CF relaxed
BFS    = TwoPhaseBFSParser(GRAMMAR_BFS)
EARLEY = EquivariantEarleyParser(GRAMMAR_EARLEY)

print("Grammar rules (cf_relaxation=False):")
print("  [CSG] Body -> BodyCodon Body BodyCodon  [left_ctx=StartCodon]")
print("  [CF]  Body -> BodyCodon Body BodyCodon  (inner levels)")
print("  [CF]  Body -> BodyCodon BodyCodon        (base case)")
print("  Tree depth = n_body / 2  (genuinely nested, not a chain)")
print()

# Codon pools
BASES = "ATGC"
ALL64 = [a + b + c for a in BASES for b in BASES for c in BASES]
STARTS = [c for c in ALL64 if c[0] == "A"]
STOPS  = ["TAA", "TAG", "TGA"]


def make_seq(n_body: int, seed: int = 42) -> str:
    """Generate start + n_body random codons + stop. n_body must be even >= 2."""
    rng = random.Random(seed)
    start = rng.choice(STARTS)
    stop  = rng.choice(STOPS)
    body  = "".join(rng.choice(ALL64) for _ in range(n_body))
    return start + body + stop


def render(seq: str, filename: str, title: str,
           use_bfs: bool = True, bfs_depth_limit: int = 80) -> None:
    """
    Verify acceptance (BFS), build parse tree (Earley), save PNG.
    """
    n_body = len(seq) // 3 - 2

    # BFS acceptance check (skip for large sequences — too slow)
    bfs_status = "BFS:skipped"
    if use_bfs:
        t0 = time.time()
        bfs_result = BFS.parse(seq, depth_limit=bfs_depth_limit)
        bfs_t = time.time() - t0
        bfs_status = f"BFS:{'ACCEPTED' if bfs_result.accepted else 'REJECTED'}({bfs_t:.2f}s)"
        if not bfs_result.accepted:
            print(f"  [SKIP]  {filename}  -- {bfs_status}")
            return

    # Earley parse tree
    t1 = time.time()
    earley_result = EARLEY.parse(seq)
    ear_t = time.time() - t1
    if not earley_result.accepted:
        print(f"  [SKIP]  {filename}  -- Earley rejected ({ear_t:.2f}s)")
        return

    tree = earley_result.trees()[0]
    path = os.path.join(OUT_DIR, filename)
    fig  = draw_parse_tree(tree, seq, title=title, save_path=path)
    plt.close(fig)
    kb = os.path.getsize(path) // 1024
    print(f"  [OK]    {filename:<55}  {kb:>4} KB  n_body={n_body:<4}  "
          f"{bfs_status}  Earley:{ear_t:.2f}s")


# =============================================================================
# Section 1 -- BFS acceptance verification
# =============================================================================
print("=" * 68)
print("Section 1  BFS acceptance / rejection")
print("=" * 68)

for n_body in [2, 4, 6, 8, 10, 12, 20]:
    seq = make_seq(n_body, seed=n_body)
    r   = BFS.parse(seq, depth_limit=120)
    n   = len(seq) // 3
    print(f"  n_body={n_body:<4}  total={n:<4}  BFS={'ACCEPTED' if r.accepted else 'REJECTED'}"
          f"  depth={r.depth_reached}  states={r.n_states}")

print()
for n_body in [1, 3, 5]:
    seq = "ATG" + "AAA" * n_body + "TAA"
    r   = BFS.parse(seq, depth_limit=60)
    print(f"  n_body={n_body} (odd)  BFS={'ACCEPTED' if r.accepted else 'REJECTED'}"
          f"  (expected: REJECTED)")


# =============================================================================
# Section 2 -- Small sequences: genuine nested tree structure
# =============================================================================
print()
print("=" * 68)
print("Section 2  Small sequences — nested tree (NOT a chain)")
print("           Each Body node has 3 children: BC / Body / BC")
print("=" * 68)

render("ATG" + "AAA" + "GCT" + "TAA",
       "N01_2body_base.png",
       "Center-Branch | 2 body codons\nBody -> BC BC  (base: 2 children)")

render("ATG" + "AAA" + "GCT" + "TTT" + "CAG" + "TAA",
       "N02_4body_one_wrap.png",
       "Center-Branch | 4 body codons\nBody -> BC [Body->BC BC] BC  (1 wrap, depth=2)")

render("ATG" + "AAA" + "GCT" + "TTT" + "CAG" + "AAA" + "GCT" + "TAA",
       "N03_6body_two_wraps.png",
       "Center-Branch | 6 body codons\nBody -> BC [Body -> BC [Body->BC BC] BC] BC  (2 wraps, depth=3)")

render("ATG" + "AAA" + "GCT" + "TTT" + "CAG" + "AAA" + "GCT" + "TTT" + "CAG" + "TAA",
       "N04_8body_three_wraps.png",
       "Center-Branch | 8 body codons\n3 wraps, depth=4")

render(make_seq(10, seed=10),
       "N05_10body_four_wraps.png",
       "Center-Branch | 10 body codons\n4 wraps, depth=5")

render(make_seq(16, seed=16),
       "N06_16body_seven_wraps.png",
       "Center-Branch | 16 body codons\n7 wraps, depth=8")


# =============================================================================
# Section 3 -- Random codons (varied leaf labels, same nested structure)
# =============================================================================
print()
print("=" * 68)
print("Section 3  Random body codons")
print("=" * 68)

rng = random.Random(999)
for n_body, seed in [(2,10), (4,20), (6,30), (8,40), (12,60), (20,70)]:
    seq = make_seq(n_body, seed=seed)
    render(seq, f"N1{n_body}_random_{n_body}body.png",
           f"Center-Branch | {n_body} random body codons",
           use_bfs=True, bfs_depth_limit=100)


# =============================================================================
# Section 4 -- Scaling: 50 / 100 / 200 body codons
#   BFS disabled for n_body > 20 (exponential state space).
#   Earley (CF-relaxed) verifies acceptance and builds tree.
# =============================================================================
print()
print("=" * 68)
print("Section 4  Scaling: 50 / 100 / 200 body codons")
print("           BFS skipped (too slow); Earley (CF-relaxed) used.")
print("=" * 68)

for n_body, seed in [(50, 1), (100, 2), (200, 3)]:
    seq = make_seq(n_body, seed=seed)
    title = (
        f"Center-Branch CSG | {n_body} body codons  (total {n_body+2})\n"
        f"Tree depth = {n_body//2}  (genuine nesting, not a chain)"
    )
    render(seq, f"N_scale_{n_body}body.png", title,
           use_bfs=False)

print(f"\nDone.  All images saved to: {OUT_DIR}\n")
