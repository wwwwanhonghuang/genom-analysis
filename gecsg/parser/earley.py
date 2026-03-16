"""
gecsg.parser.earley
===================
Equivariant Earley parser for GE-CSG grammars.

Algorithm overview
------------------
Input  : raw DNA string w ∈ Σ_raw^n  (n must be a multiple of k=3)
Output : ParseResult containing parse trees and acceptance flag

Step 1 — Codon segmentation
    Split w into blocks of length k:  b_0, b_1, ..., b_{n/k - 1}

Step 2 — Lifting
    For each block b_i, compute lifted_cosets[i] = grammar.lift(b_i)
    (all cosets that have a breaking rule mapping to b_i)

Step 3 — Earley chart parsing on the coset string
    Chart[j] = set of EarleyItems active at coset-string position j
    Three operations (run to fixed-point per position):
      PREDICT  : for (rule, dot, orig) where rhs[dot] is NonTerminal B,
                 add (r, 0, j) for every rule r with lhs==B
      SCAN     : for (rule, dot, orig) where rhs[dot] is Coset C,
                 if C ∈ lifted_cosets[j], add (rule, dot+1, orig) to Chart[j+1]
      COMPLETE : for (rule, len(rhs), orig) completed at j,
                 advance all (r2, d2, k) in Chart[orig] where r2.rhs[d2]==lhs

Step 4 — Tree reconstruction
    After parsing, collect completed_spans = {(nt, start, end)} from chart.
    Build trees via recursive descent: try each full rule for (nt, start, end),
    find consistent splits, recurse on children.

Complexity
----------
    O(|R1| · (n/k)³)  for context-free rules  (standard Earley)
    Equivariant compression: only |R1/G| orbit reps stored in grammar,
    but full_rules expansion is used here for correctness.
    Orbit-sharing in completed_spans gives constant-factor savings
    for orbit-equivalent subsequences.
"""

from __future__ import annotations

import itertools
import functools
from dataclasses import dataclass, field
from typing import (
    Dict, FrozenSet, List, Optional, Set, Tuple
)

from gecsg.grammar.grammar import GECSGGrammar
from gecsg.grammar.rule import GenerationRule
from gecsg.grammar.symbols import NonTerminal
from gecsg.core.coset import Coset
from gecsg.parser.parse_tree import ParseNode


# ─────────────────────────────────────────────────────────────────────────────
# EarleyItem
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EarleyItem:
    """
    One Earley chart item.

    Attributes
    ----------
    rule   : GenerationRule  the production being tracked
    dot    : int             position of the dot within rule.rhs (0..len(rhs))
    origin : int             chart position (coset index) where this item started
    """
    rule:   GenerationRule
    dot:    int
    origin: int

    @property
    def complete(self) -> bool:
        return self.dot >= len(self.rule.rhs)

    @property
    def next_sym(self):
        """Symbol immediately after the dot, or None if complete."""
        if self.complete:
            return None
        return self.rule.rhs[self.dot]

    def __repr__(self) -> str:
        rhs = self.rule.rhs
        before = " ".join(str(s) for s in rhs[:self.dot])
        after  = " ".join(str(s) for s in rhs[self.dot:])
        dot_s  = f"• {after}" if after else "•"
        return (f"[{self.rule.lhs} → {before} {dot_s}, "
                f"orig={self.origin}]")


# ─────────────────────────────────────────────────────────────────────────────
# ParseResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParseResult:
    """
    Result of parsing a DNA sequence with EquivariantEarleyParser.

    Attributes
    ----------
    accepted        : bool   True iff the sequence is in L(grammar)
    chart           : completed Earley chart (List[Set[EarleyItem]])
    completed_spans : set of (nt_name, start, end) triples known to be derivable
    raw_seq         : the uppercased input sequence
    grammar         : the grammar used for parsing
    n_codons        : number of codons in the input
    """
    accepted:        bool
    chart:           List[Set[EarleyItem]]
    completed_spans: Set[Tuple[str, int, int]]
    raw_seq:         str
    grammar:         GECSGGrammar
    n_codons:        int

    def trees(self) -> List[ParseNode]:
        """
        Reconstruct all parse trees from the completed chart.

        Uses recursive descent guided by completed_spans.
        Memoised to avoid redundant work.

        Returns
        -------
        List[ParseNode]  — possibly multiple trees if grammar is ambiguous.
        """
        # Convert to frozenset for hashability (lru_cache requirement)
        frozen_spans = frozenset(self.completed_spans)
        return _build_trees(
            self.grammar.start.name, 0, self.n_codons,
            frozen_spans, self.raw_seq, self.grammar,
        )

    def summary(self) -> None:
        """Print a one-line parse summary."""
        status = "ACCEPTED" if self.accepted else "REJECTED"
        print(f"[{status}]  {self.raw_seq}  "
              f"({self.n_codons} codons, "
              f"{len(self.completed_spans)} completed spans)")


# ─────────────────────────────────────────────────────────────────────────────
# EquivariantEarleyParser
# ─────────────────────────────────────────────────────────────────────────────

class EquivariantEarleyParser:
    """
    Equivariant Earley parser for GE-CSG grammars.

    Parameters
    ----------
    grammar : GECSGGrammar
        Must be frozen and contain Codon →₁ coset rules plus all
        desired breaking rules.
    """

    def __init__(self, grammar: GECSGGrammar):
        self._grammar = grammar
        # Pre-index full rules by LHS for O(1) Predict lookup
        self._by_lhs: Dict[NonTerminal, List[GenerationRule]] = {}
        for rule in grammar.full_rules:
            self._by_lhs.setdefault(rule.lhs, []).append(rule)

    # ── Public API ────────────────────────────────────────────────────────

    def parse(self, raw_seq: str) -> ParseResult:
        """
        Parse a raw DNA sequence.

        Parameters
        ----------
        raw_seq : str  DNA string (case-insensitive), length must be
                       a multiple of grammar.k (default 3).

        Returns
        -------
        ParseResult
        """
        k = self._grammar.k
        raw = raw_seq.upper().strip()

        if len(raw) % k != 0:
            raise ValueError(
                f"Sequence length {len(raw)} is not a multiple of k={k}."
            )

        n = len(raw) // k   # number of coset positions

        # ── Step 1: Lift each block ──────────────────────────────────────
        # lifted[i] = list of Coset objects that match block i
        lifted: List[List[Coset]] = []
        for i in range(n):
            block = tuple(raw[k * i: k * (i + 1)])
            lifted.append(self._grammar.lift(block))

        # ── Step 2: Initialise chart ─────────────────────────────────────
        chart: List[Set[EarleyItem]] = [set() for _ in range(n + 1)]
        start_nt = self._grammar.start

        for rule in self._by_lhs.get(start_nt, []):
            chart[0].add(EarleyItem(rule=rule, dot=0, origin=0))

        completed_spans: Set[Tuple[str, int, int]] = set()

        # ── Step 3: Process each position ───────────────────────────────
        for j in range(n + 1):
            agenda   = list(chart[j])
            seen     = set(chart[j])

            while agenda:
                item = agenda.pop()

                if item.complete:
                    # COMPLETE
                    B   = item.rule.lhs
                    key = (B.name, item.origin, j)
                    completed_spans.add(key)

                    for parent in list(chart[item.origin]):
                        if (not parent.complete
                                and isinstance(parent.next_sym, NonTerminal)
                                and parent.next_sym == B):
                            advanced = EarleyItem(
                                rule=parent.rule,
                                dot=parent.dot + 1,
                                origin=parent.origin,
                            )
                            if advanced not in seen:
                                seen.add(advanced)
                                chart[j].add(advanced)
                                agenda.append(advanced)

                else:
                    sym = item.next_sym

                    if isinstance(sym, NonTerminal):
                        # PREDICT
                        for r in self._by_lhs.get(sym, []):
                            new = EarleyItem(rule=r, dot=0, origin=j)
                            if new not in seen:
                                seen.add(new)
                                chart[j].add(new)
                                agenda.append(new)

                    elif isinstance(sym, Coset) and j < n:
                        # SCAN  (add directly to chart[j+1])
                        if sym in lifted[j]:
                            advanced = EarleyItem(
                                rule=item.rule,
                                dot=item.dot + 1,
                                origin=item.origin,
                            )
                            if advanced not in chart[j + 1]:
                                chart[j + 1].add(advanced)
                                # also add to next position's agenda when it runs

        # ── Step 4: Acceptance check ─────────────────────────────────────
        accepted = any(
            item.complete and item.origin == 0 and item.rule.lhs == start_nt
            for item in chart[n]
        )

        return ParseResult(
            accepted=accepted,
            chart=chart,
            completed_spans=completed_spans,
            raw_seq=raw,
            grammar=self._grammar,
            n_codons=n,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tree reconstruction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_trees(
    nt_name:         str,
    start:           int,
    end:             int,
    completed_spans: FrozenSet[Tuple[str, int, int]],
    raw_seq:         str,
    grammar:         GECSGGrammar,
) -> List[ParseNode]:
    """
    Recursively build all parse trees for (nt_name, start, end).

    Uses completed_spans as a pruning guide.  Memoised via a local cache
    (cache is per ParseResult.trees() call, so different parses don't share).
    """
    return _build_trees_inner(nt_name, start, end,
                              completed_spans, raw_seq, grammar)


@functools.lru_cache(maxsize=None)
def _build_trees_inner(
    nt_name:         str,
    start:           int,
    end:             int,
    completed_spans: FrozenSet[Tuple[str, int, int]],
    raw_seq:         str,
    grammar:         GECSGGrammar,
) -> List[ParseNode]:
    """Memoised inner implementation."""
    results: List[ParseNode] = []
    k = grammar.k

    for rule in grammar.full_rules:
        if rule.lhs.name != nt_name:
            continue
        rhs = rule.rhs
        m   = len(rhs)

        if m == 0:
            if start == end:
                results.append(ParseNode(
                    label=nt_name, span=(start, end), is_terminal=False,
                    children=(), rule=rule,
                ))
            continue

        # Generate all valid split-point tuples
        for splits in _splits(start, end, m):
            children: List[ParseNode] = []
            valid = True

            for idx, sym in enumerate(rhs):
                s, e = splits[idx], splits[idx + 1]

                if isinstance(sym, NonTerminal):
                    if (sym.name, s, e) not in completed_spans:
                        valid = False
                        break
                    sub = _build_trees_inner(
                        sym.name, s, e, completed_spans, raw_seq, grammar
                    )
                    if not sub:
                        valid = False
                        break
                    children.append(sub[0])   # take first (leftmost) tree

                elif isinstance(sym, Coset):
                    if e - s != 1:
                        valid = False
                        break
                    block = tuple(raw_seq[k * s: k * e])
                    if sym not in grammar.lift(block):
                        valid = False
                        break
                    children.append(ParseNode(
                        label=sym.label,
                        span=(s, e),
                        is_terminal=True,
                        coset=sym,
                        raw_codon="".join(block),
                    ))

            if valid:
                results.append(ParseNode(
                    label=nt_name,
                    span=(start, end),
                    is_terminal=False,
                    children=tuple(children),
                    rule=rule,
                ))
                break   # return first valid tree (deterministic grammar)

    return results


def _splits(start: int, end: int, n: int):
    """
    Generate all monotone split-point tuples
    (start = p0, p1, ..., p_{n-1}, end = pn)
    such that each segment is non-empty.

    Equivalent to choosing n-1 interior split points from
    {start+1, ..., end-1} in sorted order.
    """
    if n == 1:
        yield (start, end)
        return
    length = end - start
    if length < n:
        return
    # Choose n-1 split points from {start+1, ..., end-1}
    interior = range(start + 1, end)
    for pts in itertools.combinations(interior, n - 1):
        yield (start,) + pts + (end,)
