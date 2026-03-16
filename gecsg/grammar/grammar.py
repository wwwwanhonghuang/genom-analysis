"""
gecsg.grammar.grammar
=====================
The central GECSGGrammar class implementing the v4 six-tuple:

    G = (G, G_i, V, S, R1, R2)

Design
------
- Grammar stores only R1/G (orbit representatives) and the full R2.
- Full rule set is generated on demand via RuleOrbit.
- CosetSpace G/G_i is built automatically from G and G_i subgroup.
- Torch-compatibility hooks are provided (embedding indices, vocab).
- The grammar is mutable during construction, immutable after .freeze().

Torch integration points
------------------------
  grammar.nonterminal_vocab   -> {NT_name: int}   for embedding lookup
  grammar.coset_vocab         -> {coset_idx: int} for embedding lookup
  grammar.rule_vocab          -> {rule_id: int}   for rule orbit embeddings
"""

from __future__ import annotations
from typing import (
    List, Dict, Optional, Tuple, Set, FrozenSet, Iterator
)
from gecsg.core.group import Group
from gecsg.core.coset import Coset, CosetSpace
from gecsg.grammar.symbols import NonTerminal, TerminalSymbol, Symbol
from gecsg.grammar.rule import (
    GenerationRule, BreakingRule, RuleOrbit, generate_orbit, SFElement
)


class GECSGGrammar:
    """
    Group-Equivariant Context-Sensitive Grammar (v4).

    Parameters
    ----------
    group             : G  (ambient finite group)
    subgroup_indices  : indices of G_i ⊆ G elements
    start             : name of the start non-terminal
    k                 : breaking granularity (length of each raw block)
    """

    def __init__(
        self,
        group:            Group,
        subgroup_indices: List[int],
        start:            str,
        k:                int = 3,
    ):
        self._G   = group
        self._k   = k

        # Build coset space G/G_i
        self._CS  = group.coset_space(subgroup_indices)

        # Symbol registries
        self._nonterminals: Dict[str, NonTerminal] = {}
        self._start_name: str = start

        # Rule stores
        self._orbits:   List[RuleOrbit]    = []   # R1/G
        self._breaking: List[BreakingRule] = []   # R2

        # Frozen flag
        self._frozen: bool = False

        # Register start symbol
        self._register_nt(start)

        # Orbit cache: full rule set (generated lazily)
        self._full_rules_cache: Optional[List[GenerationRule]] = None

    # -- Symbol registration ---------------------------------------------------

    def _register_nt(self, name: str) -> NonTerminal:
        if name not in self._nonterminals:
            self._nonterminals[name] = NonTerminal(name)
        return self._nonterminals[name]

    def NT(self, name: str) -> NonTerminal:
        """Get or create a NonTerminal by name."""
        return self._register_nt(name)

    def coset(self, idx: int) -> Coset:
        """Get coset by index."""
        return self._CS[idx]

    # -- Properties ------------------------------------------------------------

    @property
    def group(self) -> Group:
        return self._G

    @property
    def coset_space(self) -> CosetSpace:
        return self._CS

    @property
    def k(self) -> int:
        """Breaking granularity."""
        return self._k

    @property
    def start(self) -> NonTerminal:
        return self._nonterminals[self._start_name]

    @property
    def nonterminals(self) -> List[NonTerminal]:
        return list(self._nonterminals.values())

    @property
    def orbits(self) -> List[RuleOrbit]:
        """R1/G: stored orbit representatives."""
        return self._orbits

    @property
    def breaking_rules(self) -> List[BreakingRule]:
        """R2: all breaking rules."""
        return self._breaking

    @property
    def n_orbits(self) -> int:
        return len(self._orbits)

    @property
    def n_full_rules(self) -> int:
        return sum(o.size() for o in self._orbits)

    @property
    def n_breaking(self) -> int:
        return len(self._breaking)

    def compression_ratio(self) -> float:
        """Fraction of rules stored: |R1/G| / |R1|."""
        full = self.n_full_rules
        return len(self._orbits) / full if full > 0 else 1.0

    # -- Adding rules ----------------------------------------------------------

    def add_generation_rule(
        self,
        lhs:       str | NonTerminal,
        rhs:       List[str | NonTerminal | Coset | int],
        left_ctx:  Optional[List[str | NonTerminal | Coset | int]] = None,
        right_ctx: Optional[List[str | NonTerminal | Coset | int]] = None,
        weight:    float = 1.0,
    ) -> "GECSGGrammar":
        """
        Add one orbit-representative generation rule.

        rhs elements can be:
          str        -> interpreted as NonTerminal name
          NonTerminal -> used directly
          Coset       -> used directly
          int         -> coset index shorthand

        Returns self for chaining.
        """
        if self._frozen:
            raise RuntimeError("Grammar is frozen; cannot add rules.")

        def _resolve(x) -> SFElement:
            if isinstance(x, NonTerminal):
                return self._register_nt(x.name)
            if isinstance(x, Coset):
                return x
            if isinstance(x, int):
                return self._CS[x]
            if isinstance(x, str):
                return self._register_nt(x)
            raise TypeError(f"Cannot resolve SFElement from {x!r}")

        lhs_nt = self._register_nt(lhs.name if isinstance(lhs, NonTerminal) else lhs)
        rhs_t  = tuple(_resolve(x) for x in rhs)
        lctx_t = tuple(_resolve(x) for x in (left_ctx or []))
        rctx_t = tuple(_resolve(x) for x in (right_ctx or []))

        rep = GenerationRule(
            lhs=lhs_nt, rhs=rhs_t,
            left_ctx=lctx_t, right_ctx=rctx_t,
            weight=weight,
        )
        orbit = generate_orbit(rep, self._G, self._CS)
        self._orbits.append(orbit)
        self._full_rules_cache = None   # invalidate cache
        return self

    def add_breaking_rule(
        self,
        coset:  Coset | int,
        string: str | Tuple[str, ...],
        prob:   float = 1.0,
    ) -> "GECSGGrammar":
        """
        Add one breaking rule: coset ->₂ string.

        coset  : Coset object or integer coset index
        string : raw string of length k (e.g. "ATG") or tuple
        """
        if self._frozen:
            raise RuntimeError("Grammar is frozen; cannot add rules.")

        c = self._CS[coset] if isinstance(coset, int) else coset
        s = tuple(string) if isinstance(string, str) else string
        if len(s) != self._k:
            raise ValueError(
                f"Breaking rule string length {len(s)} != k={self._k}. "
                f"Got '{string}'."
            )
        self._breaking.append(BreakingRule(coset=c, string=s, prob=prob))
        return self

    # -- Full rule set (generated lazily) --------------------------------------

    @property
    def full_rules(self) -> List[GenerationRule]:
        """R1: all rules (expand all orbits)."""
        if self._full_rules_cache is None:
            result = []
            for orbit in self._orbits:
                result.extend(orbit.members)
            self._full_rules_cache = result
        return self._full_rules_cache

    def rules_for_lhs(self, lhs: NonTerminal) -> Iterator[GenerationRule]:
        """Iterate over all full rules with the given LHS."""
        for r in self.full_rules:
            if r.lhs == lhs:
                yield r

    def orbit_reps_for_lhs(self, lhs: NonTerminal) -> List[GenerationRule]:
        """Return orbit representatives whose LHS is `lhs`."""
        return [o.representative for o in self._orbits
                if o.representative.lhs == lhs]

    # -- Lifting map Λ ---------------------------------------------------------

    def lift(self, block: Tuple[str, ...]) -> List[Coset]:
        """
        Λ(block): return all cosets c such that (c, block) ∈ R2.
        """
        return [br.coset for br in self._breaking if br.string == block]

    def lift_star(self, block: Tuple[str, ...]) -> List[Coset]:
        """
        Λ*(block): orbit-maximising lifting.
        Returns cosets with maximum orbit size among Λ(block).
        """
        candidates = self.lift(block)
        return self._CS.orbit_maximising_coset(candidates)

    # -- Terminal alphabet Σ = R2 ----------------------------------------------

    @property
    def terminal_alphabet(self) -> List[BreakingRule]:
        """Σ = R2: the terminal alphabet equals the breaking rules."""
        return self._breaking

    # -- Torch vocabulary helpers ----------------------------------------------

    @property
    def nonterminal_vocab(self) -> Dict[str, int]:
        """Map NT name -> integer index (for embedding lookup)."""
        return {name: i for i, name in enumerate(self._nonterminals)}

    @property
    def coset_vocab(self) -> Dict[int, int]:
        """Map coset.index -> integer index (for embedding lookup)."""
        return {c.index: i for i, c in enumerate(self._CS.cosets)}

    @property
    def rule_vocab(self) -> Dict[int, int]:
        """Map orbit index -> integer index (for rule orbit embeddings)."""
        return {i: i for i in range(len(self._orbits))}

    # -- Freeze ----------------------------------------------------------------

    def freeze(self) -> "GECSGGrammar":
        """Prevent further modification. Returns self for chaining."""
        self._frozen = True
        return self

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    # -- Describe --------------------------------------------------------------

    def describe(self, show_full_orbits: bool = False) -> None:
        """Print a structured summary of the grammar."""
        G  = self._G
        CS = self._CS
        print("+- GE-CSG Grammar " + "-" * 44)
        print(f"|  Group        : {G.name}  (order {G.order})")
        print(f"|  Coset space  : |G/G_i| = {CS.size}  "
              f"(subgroup order {G.order // CS.size})")
        print(f"|  k (breaking) : {self._k}")
        print(f"|  Start        : {self.start}")
        print(f"|  NonTerminals : {[str(nt) for nt in self.nonterminals]}")
        print(f"|  Orbit reps   : {self.n_orbits}")
        print(f"|  Full rules   : {self.n_full_rules}")
        print(f"|  Breaking rules: {self.n_breaking}")
        print(f"|  Compression  : {self.compression_ratio():.1%} stored")
        print("+- Generation rules (grouped by orbit) " + "-" * 23)
        for i, orbit in enumerate(self._orbits):
            rep = orbit.representative
            print(f"|  > {rep}   [orbit size={orbit.size()}]")
            if show_full_orbits:
                for j, member in enumerate(orbit.members[1:], 1):
                    print(f"|    {member}   [g={orbit.g_indices[j]}]")
        print("+- Breaking rules (R2 = Σ) " + "-" * 35)
        for br in self._breaking:
            print(f"|  {br}")
        print("+" + "-" * 61)

    def __repr__(self) -> str:
        return (f"GECSGGrammar(G={self._G.name}, "
                f"|G/Gi|={self._CS.size}, "
                f"start={self._start_name}, "
                f"orbits={self.n_orbits}, "
                f"breaking={self.n_breaking})")
