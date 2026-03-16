"""
gecsg.grammar.rule
==================
Rule types for GE-CSG (v4 formalism).

Rule types
----------
GenerationRule   -- Arrow 1:  α A β → α γ β   (G-equivariant)
BreakingRule     -- Arrow 2:  [g_x]_{G_i} → w ∈ Σ_raw^k
RuleOrbit        -- the full G-orbit of a GenerationRule, stored
                    as (representative, [all orbit members])
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, TYPE_CHECKING

from gecsg.grammar.symbols import NonTerminal, TerminalSymbol, Symbol
from gecsg.core.coset import Coset

if TYPE_CHECKING:
    from gecsg.core.group import Group
    from gecsg.core.coset import CosetSpace


# ─────────────────────────────────────────────────────────────────────────────
# Sentential-form element type
# ─────────────────────────────────────────────────────────────────────────────
# In Phase 1, sentential forms contain NonTerminals and Cosets.
# We use the union type SFElement = NonTerminal | Coset.
SFElement = NonTerminal | Coset


# ─────────────────────────────────────────────────────────────────────────────
# GenerationRule  (Arrow 1)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GenerationRule:
    """
    One Arrow-1 rule: α A β →₁ α γ β.

    Attributes
    ----------
    lhs      : NonTerminal          the rewritten non-terminal A
    rhs      : Tuple[SFElement,...] the replacement γ
    left_ctx : Tuple[SFElement,...] left context α  (empty = context-free)
    right_ctx: Tuple[SFElement,...] right context β (empty = context-free)
    weight   : float                prior weight (for PGE-CSG); default 1.0
    """
    lhs:       NonTerminal
    rhs:       Tuple[SFElement, ...]
    left_ctx:  Tuple[SFElement, ...] = ()
    right_ctx: Tuple[SFElement, ...] = ()
    weight:    float = 1.0

    def __repr__(self) -> str:
        ctx_l = "".join(str(s) for s in self.left_ctx)
        ctx_r = "".join(str(s) for s in self.right_ctx)
        rhs_s = " ".join(str(s) for s in self.rhs)
        if self.left_ctx or self.right_ctx:
            return f"{ctx_l} {self.lhs} {ctx_r} →₁ {ctx_l} {rhs_s} {ctx_r}"
        return f"{self.lhs} →₁ {rhs_s}"

    @property
    def is_context_free(self) -> bool:
        return len(self.left_ctx) == 0 and len(self.right_ctx) == 0

    @property
    def rhs_nonterminals(self) -> List[NonTerminal]:
        return [s for s in self.rhs if isinstance(s, NonTerminal)]

    @property
    def rhs_cosets(self) -> List[Coset]:
        return [s for s in self.rhs if isinstance(s, Coset)]


# ─────────────────────────────────────────────────────────────────────────────
# BreakingRule  (Arrow 2)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BreakingRule:
    """
    One Arrow-2 rule: [g_x]_{G_i} →₂ w ∈ Σ_raw^k.

    This is simultaneously:
      - a breaking rule (maps coset to concrete string)
      - a terminal symbol (coset, string) ∈ Σ = R₂

    Attributes
    ----------
    coset  : Coset             the Phase-1 leaf [g_x]_{G_i}
    string : Tuple[str,...]    the concrete realisation w ∈ Σ_raw^k
    prob   : float             P₂(coset, string); default 1.0
    """
    coset:  Coset
    string: Tuple[str, ...]
    prob:   float = 1.0

    def __repr__(self) -> str:
        return f"{self.coset} →₂ {''.join(self.string)}"

    @property
    def k(self) -> int:
        """Breaking granularity."""
        return len(self.string)

    @property
    def as_terminal(self) -> TerminalSymbol:
        """Convert to TerminalSymbol for use in parse trees."""
        from gecsg.grammar.symbols import TerminalSymbol
        return TerminalSymbol(coset=self.coset, string=self.string)


# ─────────────────────────────────────────────────────────────────────────────
# RuleOrbit  (G-orbit of a GenerationRule)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleOrbit:
    """
    The G-orbit of a GenerationRule.

    Stores:
      representative : the canonical orbit representative (stored in grammar)
      members        : all rules in the orbit (generated on demand)
      g_indices      : the group element that maps rep → each member

    The grammar stores only `representative`; members are generated
    lazily when needed by the parser.
    """
    representative: GenerationRule
    members:        List[GenerationRule] = field(default_factory=list)
    g_indices:      List[int]           = field(default_factory=list)

    def __post_init__(self):
        if not self.members:
            self.members  = [self.representative]
            self.g_indices = [0]

    def size(self) -> int:
        return len(self.members)

    def __repr__(self) -> str:
        return (f"RuleOrbit({self.representative!r}, "
                f"size={self.size()})")


# ─────────────────────────────────────────────────────────────────────────────
# Orbit generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_orbit(
    rep:    GenerationRule,
    group,
    coset_space,
) -> RuleOrbit:
    """
    Expand one GenerationRule representative into its full G-orbit.

    Parameters
    ----------
    rep          : the orbit representative rule
    group        : the ambient group G
    coset_space  : G/G_i (to map g-actions on cosets)
    """
    seen:    dict = {}   # (lhs, rhs, lctx, rctx) -> bool
    members: List[GenerationRule] = []
    g_idxs:  List[int]            = []

    def act_on_sf(g_idx: int, sf: Tuple[SFElement, ...]) -> Tuple[SFElement, ...]:
        """Apply group element to a sentential-form tuple."""
        result = []
        for elem in sf:
            if isinstance(elem, NonTerminal):
                result.append(elem)   # G acts trivially on nonterminals
            elif isinstance(elem, Coset):
                result.append(coset_space.act(g_idx, elem))
            else:
                raise TypeError(f"Unexpected SFElement type: {type(elem)}")
        return tuple(result)

    for g in group.elements:
        new_lhs   = rep.lhs   # G acts trivially on nonterminals
        new_rhs   = act_on_sf(g.index, rep.rhs)
        new_lctx  = act_on_sf(g.index, rep.left_ctx)
        new_rctx  = act_on_sf(g.index, rep.right_ctx)

        key = (new_lhs, new_rhs, new_lctx, new_rctx)
        if key not in seen:
            seen[key] = True
            new_rule = GenerationRule(
                lhs=new_lhs, rhs=new_rhs,
                left_ctx=new_lctx, right_ctx=new_rctx,
                weight=rep.weight,
            )
            members.append(new_rule)
            g_idxs.append(g.index)

    return RuleOrbit(
        representative=rep,
        members=members,
        g_indices=g_idxs,
    )
