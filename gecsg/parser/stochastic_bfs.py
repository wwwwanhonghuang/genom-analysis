"""
gecsg.parser.stochastic_bfs
============================
Stochastic Two-Phase BFS Parser (Viterbi scoring).

Algorithm
---------
Extends Algorithm 1 (TwoPhaseBFSParser) with log-probability tracking:

  Phase 1 — BFS over sentential forms, tracking log P₁:
    score[φ]  = sum of log(rule.weight) along the derivation path to φ.

    Because G acts trivially on rule weights (all orbit members share the same
    weight), the score is orbit-invariant: g·φ and φ carry the same score.
    Orbit compression therefore does not introduce score ambiguity.

    If the grammar is unambiguous (exactly one derivation per accepted string),
    each orbit-canonical form has a unique score.  For ambiguous grammars the
    parser retains the *maximum* score (Viterbi) when the same canonical form
    is reached via multiple paths.

  Phase 2 — log P₂ added at acceptance:
    log P₂(x, w) = Σᵢ log p(x_i, w_i)   where p is the breaking probability.

  Total log-probability:
    log P(w) = score[witness_canon] + log P₂(witness, w)
             = Σ_{rules} log w(r)  +  Σ_{codons} log p(codon | coset)

This matches the Viterbi formula in the paper (§ Stochastic Extension).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from gecsg.grammar.grammar import GECSGGrammar
from gecsg.grammar.rule import GenerationRule, SFElement
from gecsg.grammar.symbols import NonTerminal
from gecsg.core.coset import Coset
from gecsg.parser.bfs_parser import (
    TwoPhaseBFSParser,
    BFSParseResult,
    SententialForm,
    CanonicalKey,
    _to_key,
)


# ─────────────────────────────────────────────────────────────────────────────
# StochasticBFSParseResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StochasticBFSParseResult:
    """
    Result of parsing with StochasticBFSParser.

    Attributes
    ----------
    accepted      : bool
    log_prob      : float   log P(seq | grammar)  (-inf if rejected)
    witnesses     : list of (coset_string, log_p1, log_p2) triples
    n_states      : int     orbit-canonical states visited in Phase 1
    depth_reached : int     BFS depth at which first witness was found (-1 if none)
    """
    accepted:       bool
    log_prob:       float                                     = -math.inf
    witnesses:      List[Tuple[SententialForm, float, float]] = field(default_factory=list)
    n_states:       int                                       = 0
    depth_reached:  int                                       = -1

    @property
    def prob(self) -> float:
        lp = self.log_prob
        return math.exp(lp) if lp > -1e300 else 0.0

    def summary(self, digits: int = 4) -> None:
        """Print a one-line summary."""
        status = "ACCEPTED" if self.accepted else "REJECTED"
        raw = "".join(
            "".join(c.string if hasattr(c, 'string') else str(c)
                    for c in w)
            for w, _, _ in self.witnesses[:1]
        )
        if self.accepted:
            print(
                f"[{status}]"
                f"  log_P = {self.log_prob:.{digits}f}"
                f"  P = {self.prob:.{digits}e}"
                f"  depth={self.depth_reached}"
                f"  states={self.n_states}"
            )
        else:
            print(f"[{status}]  (no witnesses)")


# ─────────────────────────────────────────────────────────────────────────────
# StochasticBFSParser
# ─────────────────────────────────────────────────────────────────────────────

class StochasticBFSParser(TwoPhaseBFSParser):
    """
    Stochastic Two-Phase BFS parser.

    Extends TwoPhaseBFSParser by tracking log P₁ (accumulated rule-weight
    log-probabilities) alongside each orbit-canonical sentential form.

    log P₂ is computed at Phase-2 check time from the breaking rule probs.
    The total log P(w) = log P₁ + log P₂.

    For ambiguous grammars: Viterbi (max score) is kept when the same
    canonical form is reached via multiple paths.  For the standard DNA
    grammar (unambiguous), each canonical form has a unique score.

    Parameters
    ----------
    grammar : GECSGGrammar
        A stochastic grammar with weighted rules (rule.weight) and
        weighted breaking rules (br.prob).  Build with
        stochastic_complete_dna_grammar().
    """

    def __init__(self, grammar: GECSGGrammar):
        super().__init__(grammar)

        # (coset_idx, block_tuple) -> log_prob lookup for Phase 2
        self._log_break: Dict[Tuple[int, Tuple[str, ...]], float] = {
            (br.coset.index, br.string): (
                math.log(br.prob) if br.prob > 0.0 else -math.inf
            )
            for br in grammar.breaking_rules
        }

    # ── Phase 2 with log-probability ─────────────────────────────────────────

    def _phase2_log_prob(
        self,
        x:       SententialForm,
        raw_seq: Tuple[str, ...],
    ) -> float:
        """
        Compute log P₂(x, w) = Σᵢ log p(x_i, w_i).
        Returns -inf if any block has zero breaking probability.
        """
        k      = self._grammar.k
        total  = 0.0
        for i, coset in enumerate(x):
            block = raw_seq[i * k : (i + 1) * k]
            lp    = self._log_break.get((coset.index, block), -math.inf)
            if lp == -math.inf:
                return -math.inf
            total += lp
        return total

    # ── Main stochastic parse ─────────────────────────────────────────────────

    def parse(  # type: ignore[override]
        self,
        raw_seq:     str | Tuple[str, ...],
        depth_limit: int = 50,
    ) -> StochasticBFSParseResult:
        """
        Two-phase BFS parse with Viterbi log-probability scoring.

        Returns
        -------
        StochasticBFSParseResult with accepted flag, log P(w), and witnesses.
        """
        grammar  = self._grammar
        k        = grammar.k

        if isinstance(raw_seq, str):
            raw_seq = tuple(raw_seq.upper())
        else:
            raw_seq = tuple(c.upper() if isinstance(c, str) else c for c in raw_seq)

        if len(raw_seq) % k != 0:
            return StochasticBFSParseResult(accepted=False)

        target_m = len(raw_seq) // k

        # ── Phase 1 BFS with score tracking ──────────────────────────────────

        start: SententialForm = (grammar.start,)
        start_canon = self._orbit_canonical(start)

        # frontier: canonical_key -> (representative_form, log_p1)
        frontier: Dict[CanonicalKey, Tuple[SententialForm, float]] = {
            start_canon: (start, 0.0)
        }
        # Viterbi scores: canonical_key -> best log_p1 seen so far
        best_score: Dict[CanonicalKey, float] = {start_canon: 0.0}

        witnesses:     List[Tuple[SententialForm, float, float]] = []
        depth_reached: int = -1

        # Degenerate: start is already a coset string
        if self._is_coset_string(start):
            if len(start) == target_m:
                for member in self._coset_string_orbit(start):
                    lp2 = self._phase2_log_prob(member, raw_seq)
                    if lp2 > -math.inf:
                        witnesses.append((member, 0.0, lp2))
                        depth_reached = 0

        for depth in range(depth_limit):
            if not frontier:
                break

            next_frontier: Dict[CanonicalKey, Tuple[SententialForm, float]] = {}

            for parent_canon, (phi, parent_score) in frontier.items():
                for rule in self._all_rules:
                    rule_lp = (
                        math.log(rule.weight)
                        if rule.weight > 0.0 else -math.inf
                    )
                    if rule_lp == -math.inf:
                        continue

                    for pos in range(len(phi)):
                        new_phi = self._apply_rule(rule, phi, pos)
                        if new_phi is None:
                            continue

                        new_score = parent_score + rule_lp
                        canon     = self._orbit_canonical(new_phi)

                        # Viterbi: keep the best score for each canonical form
                        prev_best = best_score.get(canon, -math.inf)
                        if new_score <= prev_best:
                            continue   # not an improvement

                        best_score[canon] = new_score

                        if self._is_coset_string(new_phi):
                            if len(new_phi) == target_m:
                                # Enumerate all orbit members — Phase 2 is
                                # non-equivariant, so check each one.
                                for member in self._coset_string_orbit(new_phi):
                                    lp2 = self._phase2_log_prob(member, raw_seq)
                                    if lp2 > -math.inf:
                                        witnesses.append((member, new_score, lp2))
                                        if depth_reached < 0:
                                            depth_reached = depth + 1
                            # Coset strings are terminal — do not expand.
                        else:
                            if self._min_cosets_for_form(new_phi) > target_m:
                                continue
                            next_frontier[canon] = (new_phi, new_score)

            frontier = next_frontier

        # ── Best total log-prob across all witnesses ──────────────────────────
        best_lp = -math.inf
        for (_w, lp1, lp2) in witnesses:
            total = lp1 + lp2
            if total > best_lp:
                best_lp = total

        return StochasticBFSParseResult(
            accepted      = len(witnesses) > 0,
            log_prob      = best_lp,
            witnesses     = witnesses,
            n_states      = len(best_score),
            depth_reached = depth_reached,
        )

    # ── Convenience helpers ───────────────────────────────────────────────────

    def log_prob(self, raw_seq: str | Tuple[str, ...], depth_limit: int = 50) -> float:
        """Return log P(seq | grammar), or -inf if rejected."""
        return self.parse(raw_seq, depth_limit).log_prob

    def rank(
        self,
        sequences:   List[str],
        depth_limit: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Parse and rank sequences by descending log-probability.
        Returns only accepted sequences as (seq, log_prob) pairs.
        """
        results = []
        for seq in sequences:
            r = self.parse(seq, depth_limit)
            if r.accepted:
                results.append((seq, r.log_prob))
        return sorted(results, key=lambda x: x[1], reverse=True)
