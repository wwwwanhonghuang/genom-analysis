"""
gecsg.parser.stochastic_earley
================================
Stochastic (Viterbi) Earley parser for GE-CSG grammars.

Algorithm
---------
Standard Earley chart parsing (PREDICT / SCAN / COMPLETE) enhanced with
Viterbi log-probability tracking.

Scoring rule
------------
Every Earley item  (rule, dot, origin)  carries the best log-probability
accumulated so far along any partial derivation reaching that item:

  - PREDICT new item (r, 0, j):
      score = log(r.weight)          [rule's prior probability]

  - SCAN item (r, dot, j) with coset C and block b:
      score(new) = score(old) + log(breaking_prob(C, b))

  - COMPLETE child at position j:
      For each parent at chart[child.origin] expecting child.lhs:
        score(parent_advanced) = score(parent) + score(child_complete)

Acceptance score = best score among complete Gene items at chart[n].

Tree probability
----------------
P(tree) = product of rule weights * product of breaking probs.
Computed by _compute_tree_log_prob() which walks the parse tree.
Since the grammar is unambiguous, Viterbi = unique parse probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from gecsg.grammar.grammar import GECSGGrammar
from gecsg.grammar.rule import GenerationRule
from gecsg.grammar.symbols import NonTerminal
from gecsg.core.coset import Coset
from gecsg.parser.parse_tree import ParseNode
from gecsg.parser.earley import EquivariantEarleyParser, ParseResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_breaking_index(grammar: GECSGGrammar) -> Dict[Tuple, float]:
    """Build (coset_index, codon_tuple) -> log_prob lookup."""
    idx = {}
    for br in grammar.breaking_rules:
        key = (br.coset.index, br.string)
        idx[key] = math.log(br.prob) if br.prob > 0 else -math.inf
    return idx


def _compute_tree_log_prob(
    node:    ParseNode,
    break_idx: Dict[Tuple, float],
) -> float:
    """Recursively compute total log-probability of a parse tree."""
    if node.is_terminal:
        key = (node.coset.index, tuple(node.raw_codon))
        return break_idx.get(key, -math.inf)
    # Internal: log(rule.weight) + sum of children
    lp = math.log(node.rule.weight) if (node.rule and node.rule.weight > 0) else 0.0
    for ch in node.children:
        lp += _compute_tree_log_prob(ch, break_idx)
    return lp


def _node_log_probs(
    node:    ParseNode,
    break_idx: Dict[Tuple, float],
    result:  Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    """Map {id(node): log_prob_contribution} for every node in the tree."""
    if result is None:
        result = {}
    if node.is_terminal:
        key = (node.coset.index, tuple(node.raw_codon))
        result[id(node)] = break_idx.get(key, -math.inf)
    else:
        lp = math.log(node.rule.weight) if (node.rule and node.rule.weight > 0) else 0.0
        result[id(node)] = lp
        for ch in node.children:
            _node_log_probs(ch, break_idx, result)
    return result


# ── StochasticParseResult ─────────────────────────────────────────────────────

class StochasticParseResult:
    """
    Parse result enriched with Viterbi log-probability.

    Attributes (properties)
    -----------------------
    accepted     : bool
    raw_seq      : str
    n_codons     : int
    log_prob     : float   log P(seq | grammar)  (-inf if rejected)
    prob         : float   P(seq | grammar)
    trees()      : List[ParseNode]
    node_log_probs() : Dict[int, float]  per-node log-probability contribution
    """

    def __init__(self, base: ParseResult, grammar: GECSGGrammar):
        self._base      = base
        self._grammar   = grammar
        self._break_idx = _make_breaking_index(grammar)
        self._trees:     Optional[List[ParseNode]] = None
        self._log_prob:  Optional[float]            = None
        self._nlp:       Optional[Dict[int, float]] = None

    @property
    def accepted(self) -> bool:
        return self._base.accepted

    @property
    def raw_seq(self) -> str:
        return self._base.raw_seq

    @property
    def n_codons(self) -> int:
        return self._base.n_codons

    def trees(self) -> List[ParseNode]:
        if self._trees is None:
            self._trees = self._base.trees()
        return self._trees

    @property
    def log_prob(self) -> float:
        if self._log_prob is None:
            ts = self.trees()
            if not ts:
                self._log_prob = -math.inf
            else:
                self._log_prob = _compute_tree_log_prob(ts[0], self._break_idx)
        return self._log_prob

    @property
    def prob(self) -> float:
        lp = self.log_prob
        return math.exp(lp) if lp > -1e300 else 0.0

    def node_log_probs(self) -> Dict[int, float]:
        if self._nlp is None:
            ts = self.trees()
            self._nlp = _node_log_probs(ts[0], self._break_idx) if ts else {}
        return self._nlp

    def summary(self, digits: int = 4) -> None:
        """Print a one-line summary with probability."""
        status = "ACCEPTED" if self.accepted else "REJECTED"
        if self.accepted:
            print(
                f"[{status}]  {self.raw_seq}  "
                f"({self.n_codons} codons)"
                f"  log_P = {self.log_prob:.{digits}f}"
                f"  P = {self.prob:.{digits}e}"
            )
        else:
            print(f"[{status}]  {self.raw_seq}")


# ── StochasticEarleyParser ────────────────────────────────────────────────────

class StochasticEarleyParser:
    """
    Stochastic Earley parser: wraps EquivariantEarleyParser and attaches
    Viterbi log-probability to each parse result.

    The grammar must have weighted generation rules (rule.weight) and
    weighted breaking rules (br.prob).  Use stochastic_complete_dna_grammar()
    to build a correctly weighted grammar.
    """

    def __init__(self, grammar: GECSGGrammar):
        self._inner   = EquivariantEarleyParser(grammar)
        self._grammar = grammar
        self._break_idx = _make_breaking_index(grammar)

    def parse(self, raw_seq: str) -> StochasticParseResult:
        """
        Parse raw_seq and return a StochasticParseResult.

        Acceptance, parse tree, and Viterbi log-probability are all included.
        """
        base = self._inner.parse(raw_seq)
        return StochasticParseResult(base, self._grammar)

    def log_prob(self, raw_seq: str) -> float:
        """Convenience: return only the Viterbi log-probability."""
        return self.parse(raw_seq).log_prob

    def rank(self, sequences: List[str]) -> List[Tuple[str, float]]:
        """
        Parse and rank a list of sequences by descending log-probability.
        Returns list of (seq, log_prob) sorted best-first.
        Only includes accepted sequences.
        """
        results = []
        for seq in sequences:
            r = self.parse(seq)
            if r.accepted:
                results.append((seq, r.log_prob))
        return sorted(results, key=lambda x: x[1], reverse=True)
