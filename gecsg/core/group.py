"""
gecsg.core.group
================
Abstract base class for finite groups used in GE-CSG.

Design principles
-----------------
- A Group knows how to act on raw symbol strings (for R2 breaking).
- It exposes its elements, subgroups, and coset space G/G_i.
- All group operations are index-based (integers 0..order-1) for
  performance; human-readable names are optional.
- Torch-friendly: permutation matrices available as tensors.

Extending
---------
To add a new group, subclass Group and implement:
  - elements          (list of GroupElement)
  - multiply(i, j)    (index multiplication table)
  - act_on_symbol(g, s)   (single-symbol action)
  - act_on_sequence(g, seq)  (sequence action: handles reversal etc.)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Sequence
import functools


# ─────────────────────────────────────────────────────────────────────────────
# GroupElement
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GroupElement:
    """
    A lightweight handle for a group element.

    Attributes
    ----------
    index : int     canonical index 0..order-1
    name  : str     human-readable label (optional, for display)
    """
    index: int
    name:  str = ""

    def __repr__(self) -> str:
        return self.name if self.name else f"g{self.index}"

    def __hash__(self) -> int:
        return hash(self.index)

    def __eq__(self, other) -> bool:
        if isinstance(other, GroupElement):
            return self.index == other.index
        return NotImplemented


# ─────────────────────────────────────────────────────────────────────────────
# Group (abstract base)
# ─────────────────────────────────────────────────────────────────────────────

class Group(ABC):
    """
    Abstract base class for finite groups in GE-CSG.

    A concrete subclass must implement:
      elements, multiply, act_on_symbol, act_on_sequence.

    Everything else (inverse, orbit, coset_space, permutation_matrix)
    is derived automatically.
    """

    def __init__(self, name: str = "G"):
        self._name = name
        self._inv_table: Optional[List[int]] = None
        self._mul_table: Optional[List[List[int]]] = None

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def elements(self) -> List[GroupElement]:
        """All group elements, index 0 = identity."""
        ...

    @property
    def order(self) -> int:
        return len(self.elements)

    @property
    def identity(self) -> GroupElement:
        return self.elements[0]

    # ── Multiplication ────────────────────────────────────────────────────────

    @abstractmethod
    def multiply(self, i: int, j: int) -> int:
        """Return index of elements[i] * elements[j]."""
        ...

    def inverse(self, i: int) -> int:
        """Return index of elements[i]^{-1}."""
        if self._inv_table is None:
            self._inv_table = [
                next(j for j in range(self.order)
                     if self.multiply(i, j) == 0)
                for i in range(self.order)
            ]
        return self._inv_table[i]

    # ── Action on symbols / sequences ─────────────────────────────────────────

    @abstractmethod
    def act_on_symbol(self, g_idx: int, symbol: str) -> str:
        """
        Apply group element g to a single raw symbol.
        Used for complement-type operations.
        """
        ...

    @abstractmethod
    def act_on_sequence(self, g_idx: int, seq: List[str]) -> List[str]:
        """
        Apply group element g to a sequence of raw symbols.
        Handles both symbol-level (complement) and positional
        (reversal, cyclic shift) operations.
        """
        ...

    # ── Orbit computation ─────────────────────────────────────────────────────

    def orbit_of_sequence(self, seq: Tuple[str, ...]) -> List[Tuple[str, ...]]:
        """Return the G-orbit of a sequence (as a sorted list of tuples)."""
        seen = set()
        result = []
        for g in self.elements:
            acted = tuple(self.act_on_sequence(g.index, list(seq)))
            if acted not in seen:
                seen.add(acted)
                result.append(acted)
        return sorted(result)

    def orbit_representative(self, seq: Tuple[str, ...]) -> Tuple[str, ...]:
        """Return the lexicographically minimal orbit member."""
        return min(self.orbit_of_sequence(seq))

    def witness_element(self, seq: Tuple[str, ...]) -> int:
        """
        Return index of g such that g * orbit_rep = seq.
        Used in Scan step for witness recording.
        """
        rep = self.orbit_representative(seq)
        for g in self.elements:
            if tuple(self.act_on_sequence(g.index, list(rep))) == seq:
                return g.index
        raise ValueError(f"No witness found for {seq}")

    # ── Permutation matrix (for Torch) ────────────────────────────────────────

    def permutation_matrix(self, g_idx: int, alphabet: List[str]):
        """
        Return the permutation matrix P_g such that
        P_g @ one_hot(s) = one_hot(g·s).

        Returns a plain list-of-lists (converts to tensor outside).
        """
        n = len(alphabet)
        idx = {s: i for i, s in enumerate(alphabet)}
        P = [[0.0] * n for _ in range(n)]
        for i, s in enumerate(alphabet):
            j = idx[self.act_on_symbol(g_idx, s)]
            P[j][i] = 1.0
        return P

    # ── Subgroup utilities ────────────────────────────────────────────────────

    def is_subgroup(self, indices: List[int]) -> bool:
        """Check whether a subset (given by indices) forms a subgroup."""
        s = set(indices)
        if 0 not in s:
            return False
        for i in s:
            if self.inverse(i) not in s:
                return False
            for j in s:
                if self.multiply(i, j) not in s:
                    return False
        return True

    def coset_space(self, subgroup_indices: List[int]) -> "CosetSpace":
        """
        Build the left coset space G / H where H = subgroup_indices.
        """
        from gecsg.core.coset import CosetSpace
        return CosetSpace(self, subgroup_indices)

    def __repr__(self) -> str:
        return f"{self._name}(order={self.order})"
