"""
gecsg.parser.bfs_parser
=======================
Two-Phase GE-CSG Parser (Algorithm 1, revised paper).

Architecture
------------
This replaces the Earley-based approach with the BFS-over-sentential-forms
algorithm described in the revised paper.  The key insight is that GE-CSG
production rules are genuinely context-sensitive (αAβ →₁ αγβ), so Earley
chart parsing is not the right fit; instead we do explicit BFS over
sentential forms with orbit compression.

Algorithm 1 (Two-Phase GE-CSG Parser)
--------------------------------------

  Phase 2 check (inline subroutine):
    for i = 0 .. |x|-1:
      if (x_i, w[i·k : (i+1)·k]) ∉ R₂: return False
    return True

  Phase 1 BFS:
    start ← (S,)
    frontier ← {start}
    visited  ← {[start]_G}         # orbit-canonical visited set
    witnesses ← []

    for depth = 0, 1, ..., D:
      next_frontier ← {}
      for each φ ∈ frontier:
        if φ is a coset string of length n/k:
          if Phase2(φ, w): witnesses.append(φ)
        else:
          for each rule r ∈ R₁, position i where r matches φ:
            φ' ← apply(r, φ, i)
            c  ← [φ']_G                  # orbit canonical form
            if c ∉ visited:
              visited.add(c)
              next_frontier.add(c)
      frontier ← next_frontier

    return (len(witnesses) > 0, witnesses)

Orbit canonical form
--------------------
[φ]_G = argmin_{g∈G} g·φ   under lexicographic order on (type_tag, value)

Since G acts trivially on nonterminals, two g-images of φ agree on every NT
position and differ only at Coset positions.  The canonical form is
determined by the group element g that gives the lex-minimal Coset sequence.

Orbit compression
-----------------
Every sentential form is stored and compared in its canonical orbit
representative.  This collapses equivalent derivation branches and
dramatically reduces the BFS state space, as proved in the revised paper's
Proposition 5.3 (Orbit Equivalence Theorem).

Complexity
----------
- General: PSPACE-complete (no better bound is claimed for arbitrary CSGs).
- With orbit compression: the BFS visits at most |G| times fewer states
  than without compression, but the asymptotic class is unchanged.
- Depth limit D bounds parseable CDS length: a CDS of n codons needs
  depth at most O(n) (one rule application per position in the derivation).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from gecsg.grammar.grammar import GECSGGrammar
from gecsg.grammar.rule import GenerationRule, SFElement
from gecsg.grammar.symbols import NonTerminal
from gecsg.core.coset import Coset


# ─────────────────────────────────────────────────────────────────────────────
# Internal representation helpers
# ─────────────────────────────────────────────────────────────────────────────

# A sentential form is a tuple of SFElements (NonTerminals and/or Cosets).
SententialForm = Tuple[SFElement, ...]

# Canonical key for a sentential form: tuple of (0, NT_name) | (1, coset_idx).
# This is hashable, sortable, and G-invariant at NT positions.
CanonicalKey = Tuple[Tuple[int, object], ...]


def _to_key(phi: SententialForm) -> CanonicalKey:
    """Convert a sentential form to its canonical-key representation."""
    result = []
    for e in phi:
        if isinstance(e, NonTerminal):
            result.append((0, e.name))
        else:
            result.append((1, e.index))
    return tuple(result)


# ─────────────────────────────────────────────────────────────────────────────
# BFSParseResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BFSParseResult:
    """
    Result of parsing with TwoPhaseBFSParser.

    Attributes
    ----------
    accepted   : bool
        True iff the sequence is in L(grammar).
    witnesses  : list of SententialForms
        All Phase-1 terminal coset strings φ ∈ (G/G_i)^m that passed Phase 2.
        Each is a tuple of Coset objects aligned to the input blocks.
    n_states   : int
        Number of distinct orbit-canonical sentential forms visited in Phase 1.
    depth_reached : int
        BFS depth at which the first witness was found (-1 if none).
    """
    accepted:       bool
    witnesses:      List[SententialForm] = field(default_factory=list)
    n_states:       int = 0
    depth_reached:  int = -1


# ─────────────────────────────────────────────────────────────────────────────
# TwoPhaseBFSParser
# ─────────────────────────────────────────────────────────────────────────────

class TwoPhaseBFSParser:
    """
    Two-Phase GE-CSG parser implementing Algorithm 1 (revised paper).

    Parameters
    ----------
    grammar : GECSGGrammar
        A frozen GECSGGrammar instance (six-tuple G, G_i, V, S, R₁, R₂).

    Usage
    -----
    >>> from gecsg.grammar.complete_dna_grammar import complete_dna_grammar
    >>> from gecsg.parser.bfs_parser import TwoPhaseBFSParser
    >>> g  = complete_dna_grammar()
    >>> p  = TwoPhaseBFSParser(g)
    >>> r  = p.parse("ATGAAATAA", depth_limit=30)
    >>> r.accepted
    True
    """

    def __init__(self, grammar: GECSGGrammar):
        self._grammar = grammar

        # Pre-expand all R₁ orbits once at construction time.
        self._all_rules: List[GenerationRule] = grammar.full_rules

        # Pre-build R₂ as a frozenset of (coset_idx, block_tuple) pairs for O(1) lookup.
        self._r2: FrozenSet[Tuple[int, Tuple[str, ...]]] = frozenset(
            (br.coset.index, br.string)
            for br in grammar.breaking_rules
        )

        # Cache group elements list for orbit canonical computation.
        self._group_elements = grammar.group.elements
        self._coset_space    = grammar.coset_space

        # Precompute min derivation length (in cosets) for each nonterminal.
        # min_len[NT] = minimum number of Cosets any complete derivation from NT produces.
        # Used for length-based pruning: prune φ if sum(min_len(e) for e in φ) > target_m.
        self._min_coset_len: Dict[str, int] = self._compute_min_coset_lengths()

    # ── Minimum-derivation-length table ──────────────────────────────────────

    def _compute_min_coset_lengths(self) -> Dict[str, int]:
        """
        Fixpoint computation of min_len[NT] = minimum number of Cosets that
        any complete derivation from NT yields.

        Cosets contribute 1; NTs contribute min_len[NT].  The fixpoint is
        reached when no entry changes (standard shortest-derivation algorithm).
        """
        INF = 10 ** 9
        min_len: Dict[str, int] = {
            nt.name: INF for nt in self._grammar.nonterminals
        }

        changed = True
        while changed:
            changed = False
            for rule in self._all_rules:
                total = sum(
                    1 if isinstance(e, Coset) else min_len.get(e.name, INF)
                    for e in rule.rhs
                )
                if total < min_len.get(rule.lhs.name, INF):
                    min_len[rule.lhs.name] = total
                    changed = True

        return min_len

    def _min_cosets_for_form(self, phi: SententialForm) -> int:
        """Lower-bound on the number of Cosets that φ derives to."""
        total = 0
        for e in phi:
            if isinstance(e, Coset):
                total += 1
            else:
                total += self._min_coset_len.get(e.name, 10 ** 9)
        return total

    # ── Orbit canonical form ──────────────────────────────────────────────────

    def _orbit_canonical(self, phi: SententialForm) -> CanonicalKey:
        """
        Compute [φ]_G = argmin_{g∈G} g·φ under lexicographic order.

        Only Coset positions are affected by G; NonTerminal positions are
        invariant.  The returned value is the lex-minimal CanonicalKey over
        all g ∈ G.
        """
        CS = self._coset_space
        best: Optional[CanonicalKey] = None

        for g in self._group_elements:
            # Apply g to each element: trivial on NT, coset action on Coset.
            candidate: List[Tuple[int, object]] = []
            for e in phi:
                if isinstance(e, NonTerminal):
                    candidate.append((0, e.name))
                else:
                    acted = CS.act(g.index, e)
                    candidate.append((1, acted.index))
            key = tuple(candidate)
            if best is None or key < best:
                best = key
        return best  # type: ignore[return-value]

    # ── Sentential form predicates ────────────────────────────────────────────

    def _is_coset_string(self, phi: SententialForm) -> bool:
        """True iff φ ∈ (G/G_i)^m (every element is a Coset)."""
        return all(isinstance(e, Coset) for e in phi)

    def _coset_string_orbit(self, phi: SententialForm) -> List[SententialForm]:
        """
        Return all distinct G-orbit members of a pure coset string φ.

        The orbit is { g·φ | g ∈ G } where g·(c₁,…,cₘ) = (g·c₁,…,g·cₘ).
        Returns one tuple per distinct orbit member (typically |G|/|Stab(φ)|
        members, at most |G|).
        """
        CS   = self._coset_space
        seen: Set[SententialForm] = set()
        result: List[SententialForm] = []
        for g in self._group_elements:
            member: SententialForm = tuple(CS.act(g.index, e) for e in phi)  # type: ignore[assignment]
            if member not in seen:
                seen.add(member)
                result.append(member)
        return result

    # ── Phase 2 check ─────────────────────────────────────────────────────────

    def _phase2_check(
        self,
        x: SententialForm,          # a coset string (all Cosets)
        raw_seq: Tuple[str, ...],   # the raw input
    ) -> bool:
        """
        Phase 2 check: for all i, (x_i, raw_seq[i·k:(i+1)·k]) ∈ R₂.
        """
        k = self._grammar.k
        if len(raw_seq) != len(x) * k:
            return False
        for i, coset in enumerate(x):
            block = raw_seq[i * k : (i + 1) * k]
            if (coset.index, block) not in self._r2:
                return False
        return True

    # ── Rule application ──────────────────────────────────────────────────────

    def _apply_rule(
        self,
        rule: GenerationRule,
        phi:  SententialForm,
        pos:  int,
    ) -> Optional[SententialForm]:
        """
        Try to apply rule  αAβ →₁ αγβ  at position `pos` in φ.

        Returns the new sentential form, or None if the rule does not match.

        Matching requires:
          1. φ[pos] is a NonTerminal equal to rule.lhs.
          2. φ[pos - |α| : pos] == α  (left context check).
          3. φ[pos + 1 : pos + 1 + |β|] == β  (right context check).
        """
        # 1. LHS check
        e = phi[pos]
        if not isinstance(e, NonTerminal) or e != rule.lhs:
            return None

        n     = len(phi)
        lctx  = rule.left_ctx
        rctx  = rule.right_ctx
        llen  = len(lctx)
        rlen  = len(rctx)

        # 2. Left context
        if llen > 0:
            if pos < llen:
                return None
            for i in range(llen):
                if phi[pos - llen + i] != lctx[i]:
                    return None

        # 3. Right context
        if rlen > 0:
            if pos + 1 + rlen > n:
                return None
            for i in range(rlen):
                if phi[pos + 1 + i] != rctx[i]:
                    return None

        # Apply the rule: replace φ[pos] with rule.rhs
        return phi[:pos] + rule.rhs + phi[pos + 1:]

    # ── Main parse ────────────────────────────────────────────────────────────

    def parse(
        self,
        raw_seq:     str | Tuple[str, ...],
        depth_limit: int = 50,
    ) -> BFSParseResult:
        """
        Two-phase BFS parse.

        Parameters
        ----------
        raw_seq     : the raw DNA sequence to parse.
                      Either a string (e.g. "ATGAAATAA") or a tuple of
                      characters/codons.  The sequence must be divisible
                      by k (the breaking granularity of the grammar).
        depth_limit : maximum BFS depth (= derivation length) to explore.
                      Controls the search space; sequences requiring deeper
                      derivations are not found.

        Returns
        -------
        BFSParseResult

        Design note — orbit enumeration for coset strings
        --------------------------------------------------
        Orbit compression (canonical-form deduplication) is used for ALL
        sentential forms in the BFS frontier, including pure coset strings.
        This prevents re-exploring G-equivalent derivation states.

        However, orbit-equivalent coset strings may differ in which member
        matches the input w under Phase 2, because Phase 2 (breaking) is
        deliberately NON-equivariant.  To preserve completeness, when the
        BFS first encounters a coset-string orbit class (canonical not yet
        visited), it enumerates ALL orbit members of that coset string and
        Phase-2 checks each one.

        Example: for input "TAAGCT" the target coset string is (C1, C2).
        The BFS generates (C0, C3) as the canonical orbit representative.
        Checking only (C0, C3) would miss the match.  By enumerating the
        full orbit {(C0,C3),(C1,C2),(C2,C1),(C3,C0)}, we find (C1,C2) and
        correctly accept the input.
        """
        grammar = self._grammar
        k       = grammar.k

        # Normalise input: always a tuple of upper-case single characters.
        if isinstance(raw_seq, str):
            raw_seq = tuple(raw_seq.upper())
        else:
            raw_seq = tuple(c.upper() if isinstance(c, str) else c for c in raw_seq)

        # Fast reject: length must be a multiple of k.
        if len(raw_seq) % k != 0:
            return BFSParseResult(accepted=False)

        target_m = len(raw_seq) // k   # expected coset-string length

        # ── Phase 1 BFS ──────────────────────────────────────────────────────

        start: SententialForm = (grammar.start,)
        start_canon = self._orbit_canonical(start)

        # frontier: dict canonical_key -> one representative sentential form
        frontier: Dict[CanonicalKey, SententialForm] = {start_canon: start}
        # visited: canonical-form deduplication for ALL sentential forms
        visited:  Set[CanonicalKey]                  = {start_canon}

        witnesses:     List[SententialForm] = []
        depth_reached: int                  = -1

        # Special case: start is already a coset string (degenerate grammar)
        if self._is_coset_string(start):
            if len(start) == target_m:
                for member in self._coset_string_orbit(start):
                    if self._phase2_check(member, raw_seq):
                        witnesses.append(member)
                        depth_reached = 0

        for depth in range(depth_limit):
            if not frontier:
                break

            next_frontier: Dict[CanonicalKey, SententialForm] = {}

            for phi in frontier.values():
                # Try every rule at every position in φ.
                for rule in self._all_rules:
                    for pos in range(len(phi)):
                        new_phi = self._apply_rule(rule, phi, pos)
                        if new_phi is None:
                            continue

                        canon = self._orbit_canonical(new_phi)
                        if canon in visited:
                            continue

                        if self._is_coset_string(new_phi):
                            # ── Coset string: orbit-canonical deduplication,
                            # but Phase-2 check ALL orbit members. ──────────
                            if len(new_phi) != target_m:
                                # Wrong length — no orbit member can match.
                                # Still mark visited to avoid re-processing.
                                visited.add(canon)
                                continue
                            visited.add(canon)
                            # Enumerate the full G-orbit of new_phi and check
                            # each member against w.  Necessary because Phase 2
                            # is non-equivariant: only one orbit member of the
                            # coset string typically matches a given input.
                            for member in self._coset_string_orbit(new_phi):
                                if self._phase2_check(member, raw_seq):
                                    witnesses.append(member)
                                    if depth_reached < 0:
                                        depth_reached = depth + 1
                            # Coset strings are terminal — do not expand.
                        else:
                            # ── Non-terminal form: canonical deduplication ───
                            # Length pruning: minimum possible coset yield
                            # already exceeds target — prune this branch.
                            if self._min_cosets_for_form(new_phi) > target_m:
                                continue
                            visited.add(canon)
                            next_frontier[canon] = new_phi

            frontier = next_frontier

        return BFSParseResult(
            accepted      = len(witnesses) > 0,
            witnesses     = witnesses,
            n_states      = len(visited),
            depth_reached = depth_reached,
        )

    # ── Convenience ───────────────────────────────────────────────────────────

    def accepts(self, raw_seq: str | Tuple[str, ...], depth_limit: int = 50) -> bool:
        """Return True iff raw_seq ∈ L(grammar)."""
        return self.parse(raw_seq, depth_limit).accepted
