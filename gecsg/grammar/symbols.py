"""
gecsg.grammar.symbols
=====================
Symbol types for GE-CSG (v4 formalism).

Symbol hierarchy
----------------
  Symbol (abstract)
    ├── NonTerminal      -- grammatical category (e.g. Gene, CDS)
    └── TerminalSymbol   -- pair (coset, raw_string)
                            coset  ∈ G/G_i  (the symmetry class)
                            string ∈ Σ_raw^k (the concrete realisation)

Design notes
------------
- Symbols are frozen (hashable, usable as dict keys / set members).
- TerminalSymbol carries BOTH the coset and the raw string,
  making R2 = Σ (the terminal alphabet equals the breaking rules).
- For display, TerminalSymbol shows just the raw string; the coset
  information is accessed via .coset.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
from gecsg.core.coset import Coset


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class Symbol:
    """Abstract base for all GE-CSG symbols."""

    @property
    def is_terminal(self) -> bool:
        return isinstance(self, TerminalSymbol)

    @property
    def is_nonterminal(self) -> bool:
        return isinstance(self, NonTerminal)


# ─────────────────────────────────────────────────────────────────────────────
# NonTerminal
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NonTerminal(Symbol):
    """
    A grammatical category / non-terminal symbol.

    Attributes
    ----------
    name : str      e.g. "Gene", "CDS", "Codon", "S"
    """
    name: str

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name


# ─────────────────────────────────────────────────────────────────────────────
# TerminalSymbol
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TerminalSymbol(Symbol):
    """
    A terminal symbol = breaking rule instance.

    In GE-CSG v4:  R2 = Σ ⊆ (G/G_i) × Σ_raw^k
    A TerminalSymbol is one element of R2, i.e., one pair (coset, string).

    Attributes
    ----------
    coset  : Coset          the coset [g_x]_{G_i} (symmetry origin)
    string : Tuple[str,...] the concrete raw string of length k
    """
    coset:  Coset
    string: Tuple[str, ...]

    def __repr__(self) -> str:
        return "".join(self.string)

    def __str__(self) -> str:
        return "".join(self.string)

    @property
    def raw(self) -> str:
        """The concrete string as a plain Python string."""
        return "".join(self.string)

    @property
    def k(self) -> int:
        """Breaking granularity: length of the raw string."""
        return len(self.string)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience constructors
# ─────────────────────────────────────────────────────────────────────────────

def NT(name: str) -> NonTerminal:
    """Shorthand for NonTerminal(name)."""
    return NonTerminal(name)


def TS(coset: Coset, raw: str | Tuple[str, ...]) -> TerminalSymbol:
    """Shorthand for TerminalSymbol(coset, string)."""
    if isinstance(raw, str):
        raw = tuple(raw)
    return TerminalSymbol(coset=coset, string=raw)
