"""
gecsg.parser.earley
===================
Equivariant Earley parser for GE-CSG grammars (Algorithm 1 from the paper).

Two-phase structure
-------------------
Phase 1 — Coset parsing (this file)
    S ⟹₁* x   where x ∈ (G/G_i)^{n/k}

    The raw input w is lifted block-by-block to a coset sequence x via the
    orbit-maximising lifting map Λ*(w_i).  Earley chart parsing then runs
    on the coset sequence x using Phase-1 generation rules.

Phase 2 — Symmetry breaking (handled in grammar.lift_star)
    x ⟹₂* w   (each coset x_i maps to the concrete block w_i via R₂)

    This phase is inverted during preprocessing: given w, we compute
    x = Λ*(w_1) Λ*(w_2) ... Λ*(w_{n/k}).  The breaking rules are thus
    used only for lifting (Phase-2 inversion), not during Earley itself.

Generation rules: αAβ →₁ αγβ
------------------------------
Rules carry left context α and right context β (both may be empty, making
the rule context-free; or non-empty, making it context-sensitive).

Context is checked during parsing:
  PREDICT  — left context α of a candidate rule is verified against the
             already-lifted cosets immediately before position j.
  COMPLETE — right context β of a completing rule is verified against the
             already-lifted cosets immediately after position j.

Because the full lifted sequence is computed upfront, both left and right
contexts can be checked against known input at any point.

Complexity
----------
    Serial: O(|R₁|/|G| · (n/k)^5) for ternary rules (e.g. CDS→SC ORF ST),
            O(|R₁|/|G| · (n/k)^3) for binary rules.
    The O(n^5) arises from 3-child splits in the CKY-style tree search,
    NOT from CSG context-sensitivity — context checking is O(|ctx|) per item.

    The current implementation uses standard Earley (not CKY), so actual
    runtime is O(n^3) for the chart fill regardless of rule arity; the O(n^5)
    bound appears in the tree-reconstruction step (_splits over ternary rules).
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
from gecsg.grammar.symbols import NonTerminal, TerminalSymbol, Symbol
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
    rule   : GenerationRule  the production being tracked (may carry left_ctx
                             and right_ctx for CSG rules)
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
        ctx_l  = "".join(str(s) for s in self.rule.left_ctx)
        ctx_r  = "".join(str(s) for s in self.rule.right_ctx)
        ctx    = f" [{ctx_l}___{ctx_r}]" if (ctx_l or ctx_r) else ""
        return (f"[{self.rule.lhs} → {before} {dot_s}{ctx}, "
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
                      (only spans whose completing rule's right context passed)
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
    Equivariant Earley parser for GE-CSG grammars (Algorithm 1).

    Implements the two-phase GE-CSG parsing:
      Phase 1: Earley chart parsing over the coset sequence.
      Phase 2: Inverted via orbit-maximising lifting Λ* in preprocessing.

    Context-sensitive rules αAβ →₁ αγβ are handled correctly:
      - Left context α is checked at PREDICT time.
      - Right context β is checked at COMPLETE time.
    Rules with empty context (α=β=ε) are standard context-free rules and
    behave identically to a plain CFG Earley parser.

    Parameters
    ----------
    grammar : GECSGGrammar
        Must be frozen and contain generation rules plus all desired
        breaking rules.
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

        # ── Phase 2 inversion: lift each block via Λ* ───────────────────
        # lifted[i] = list of Cosets that block i maps to (orbit-maximising).
        # This inverts Phase 2 (breaking): given raw block w_i, recover the
        # coset(s) c_i such that c_i →₂ w_i is a valid breaking rule.
        # Λ*(w_i) prefers cosets with the largest orbit, maximising chart
        # entry sharing across orbit-equivalent positions.
        lifted: List[List[Coset]] = []
        for i in range(n):
            block = tuple(raw[k * i: k * (i + 1)])
            lifted.append(self._grammar.lift_star(block))

        # ── Phase 1: Earley chart parsing on the coset sequence ──────────
        chart: List[Set[EarleyItem]] = [set() for _ in range(n + 1)]
        start_nt = self._grammar.start

        for rule in self._by_lhs.get(start_nt, []):
            # Start rules: left context must be empty (nothing precedes S)
            if self._check_left_ctx(rule.left_ctx, lifted, 0):
                chart[0].add(EarleyItem(rule=rule, dot=0, origin=0))

        completed_spans: Set[Tuple[str, int, int]] = set()

        for j in range(n + 1):
            agenda   = list(chart[j])
            seen     = set(chart[j])

            while agenda:
                item = agenda.pop()

                if item.complete:
                    # ── COMPLETE ────────────────────────────────────────
                    # Rule αAβ →₁ αγβ has been fully matched (γ consumed).
                    # Verify right context β against the input just after j.
                    # Only register the completion if the context passes.
                    if not self._check_right_ctx(item.rule.right_ctx,
                                                 lifted, j, n):
                        continue   # right context failed — invalid completion

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
                        # ── PREDICT ─────────────────────────────────────
                        # For rule αBβ →₁ αγβ predicted at position j:
                        # verify that the left context α matches the input
                        # cosets immediately before position j.
                        for r in self._by_lhs.get(sym, []):
                            if not self._check_left_ctx(r.left_ctx,
                                                        lifted, j):
                                continue   # left context failed — skip rule
                            new = EarleyItem(rule=r, dot=0, origin=j)
                            if new not in seen:
                                seen.add(new)
                                chart[j].add(new)
                                agenda.append(new)

                    elif isinstance(sym, Coset) and j < n:
                        # ── SCAN ────────────────────────────────────────
                        # Advance dot if the expected coset c is in Λ*(w_{j+1}).
                        # Uses orbit-maximising lifting: c ∈ Λ*(block_j).
                        if sym in lifted[j]:
                            advanced = EarleyItem(
                                rule=item.rule,
                                dot=item.dot + 1,
                                origin=item.origin,
                            )
                            if advanced not in chart[j + 1]:
                                chart[j + 1].add(advanced)

        # ── Acceptance: S derives the full coset sequence ────────────────
        # A completing start-rule item at chart[n] with origin=0 and
        # right context passing confirms w ∈ L(G).
        accepted = any(
            item.complete
            and item.origin == 0
            and item.rule.lhs == start_nt
            and self._check_right_ctx(item.rule.right_ctx, lifted, n, n)
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

    # ── Context-checking helpers ──────────────────────────────────────────

    def _check_left_ctx(
        self,
        ctx:    tuple,
        lifted: List[List[Coset]],
        j:      int,
    ) -> bool:
        """
        Return True iff the input cosets at positions [j-|ctx|, j) match ctx.

        For an empty context (ctx = ()) this is trivially True.
        For a non-empty context, the |ctx| positions before j must each
        contain the expected coset in their Λ* set.
        """
        if not ctx:
            return True
        if j < len(ctx):
            return False   # not enough input before j
        return all(
            ctx[ki] in lifted[j - len(ctx) + ki]
            for ki in range(len(ctx))
        )

    def _check_right_ctx(
        self,
        ctx:    tuple,
        lifted: List[List[Coset]],
        j:      int,
        n:      int,
    ) -> bool:
        """
        Return True iff the input cosets at positions [j, j+|ctx|) match ctx.

        For an empty context (ctx = ()) this is trivially True.
        For a non-empty context, the |ctx| positions after j must each
        contain the expected coset in their Λ* set.
        """
        if not ctx:
            return True
        if j + len(ctx) > n:
            return False   # not enough input after j
        return all(
            ctx[ki] in lifted[j + ki]
            for ki in range(len(ctx))
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
        # For ternary rules (m=3) this is O((end-start)^2) splits — the
        # source of the O(n^5) term in the overall complexity.
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
    interior = range(start + 1, end)
    for pts in itertools.combinations(interior, n - 1):
        yield (start,) + pts + (end,)
