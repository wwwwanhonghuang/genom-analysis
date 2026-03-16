"""
gecsg.core.dna_groups
=====================
Concrete Group subclasses for DNA symmetries.

Groups provided
---------------
TrivialGroup              {e}
Z2ComplementGroup         A↔T, G↔C
Z2ReversalGroup           reverse sequence
Z2RCGroup                 reverse complement
Z3CyclicGroup             cyclic shift of k positions
S3PermGroup               all permutations of 3 positions
DirectProductGroup        G1 × G2 (composite)

Usage
-----
    from gecsg.core.dna_groups import Z2ComplementGroup, DirectProductGroup, Z3CyclicGroup

    G = DirectProductGroup(
            DirectProductGroup(Z2ComplementGroup(), Z2ReversalGroup()),
            Z3CyclicGroup(k=3)
        )
    print(G)           # ((Z2_comp×Z2_rev)×Z3)  order=12
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from gecsg.core.group import Group, GroupElement


# ─────────────────────────────────────────────────────────────────────────────
# TrivialGroup
# ─────────────────────────────────────────────────────────────────────────────

class TrivialGroup(Group):
    """The trivial group {e} of order 1."""

    def __init__(self):
        super().__init__("Z1")
        self._elements = [GroupElement(0, "e")]

    @property
    def elements(self) -> List[GroupElement]:
        return self._elements

    def multiply(self, i: int, j: int) -> int:
        return 0

    def act_on_symbol(self, g_idx: int, symbol: str) -> str:
        return symbol

    def act_on_sequence(self, g_idx: int, seq: List[str]) -> List[str]:
        return list(seq)


# ─────────────────────────────────────────────────────────────────────────────
# Z2ComplementGroup
# ─────────────────────────────────────────────────────────────────────────────

class Z2ComplementGroup(Group):
    """
    Z2 complement: A↔T, G↔C.
    Elements: {e=0, c=1}.
    """
    _COMP = {"A": "T", "T": "A", "G": "C", "C": "G",
             "a": "t", "t": "a", "g": "c", "c": "g"}

    def __init__(self):
        super().__init__("Z2_comp")
        self._elements = [GroupElement(0, "e"), GroupElement(1, "c")]
        self._mul = [[0, 1], [1, 0]]

    @property
    def elements(self) -> List[GroupElement]:
        return self._elements

    def multiply(self, i: int, j: int) -> int:
        return self._mul[i][j]

    def act_on_symbol(self, g_idx: int, symbol: str) -> str:
        if g_idx == 0:
            return symbol
        return self._COMP.get(symbol, symbol)

    def act_on_sequence(self, g_idx: int, seq: List[str]) -> List[str]:
        if g_idx == 0:
            return list(seq)
        return [self._COMP.get(s, s) for s in seq]


# ─────────────────────────────────────────────────────────────────────────────
# Z2ReversalGroup
# ─────────────────────────────────────────────────────────────────────────────

class Z2ReversalGroup(Group):
    """
    Z2 reversal: reverse the sequence order.
    Elements: {e=0, r=1}.
    Symbol-level: identity (no per-symbol change).
    """

    def __init__(self):
        super().__init__("Z2_rev")
        self._elements = [GroupElement(0, "e"), GroupElement(1, "r")]
        self._mul = [[0, 1], [1, 0]]

    @property
    def elements(self) -> List[GroupElement]:
        return self._elements

    def multiply(self, i: int, j: int) -> int:
        return self._mul[i][j]

    def act_on_symbol(self, g_idx: int, symbol: str) -> str:
        return symbol   # reversal does not change individual symbols

    def act_on_sequence(self, g_idx: int, seq: List[str]) -> List[str]:
        if g_idx == 0:
            return list(seq)
        return list(reversed(seq))


# ─────────────────────────────────────────────────────────────────────────────
# Z2RCGroup (Reverse Complement)
# ─────────────────────────────────────────────────────────────────────────────

class Z2RCGroup(Group):
    """
    Z2 reverse complement: rc(w) = complement(reverse(w)).
    Elements: {e=0, rc=1}.
    """
    _COMP = Z2ComplementGroup._COMP

    def __init__(self):
        super().__init__("Z2_RC")
        self._elements = [GroupElement(0, "e"), GroupElement(1, "rc")]
        self._mul = [[0, 1], [1, 0]]

    @property
    def elements(self) -> List[GroupElement]:
        return self._elements

    def multiply(self, i: int, j: int) -> int:
        return self._mul[i][j]

    def act_on_symbol(self, g_idx: int, symbol: str) -> str:
        if g_idx == 0:
            return symbol
        return self._COMP.get(symbol, symbol)

    def act_on_sequence(self, g_idx: int, seq: List[str]) -> List[str]:
        if g_idx == 0:
            return list(seq)
        return [self._COMP.get(s, s) for s in reversed(seq)]


# ─────────────────────────────────────────────────────────────────────────────
# Z3CyclicGroup (codon position cyclic shift)
# ─────────────────────────────────────────────────────────────────────────────

class Z3CyclicGroup(Group):
    """
    Z_n cyclic group acting as left-rotation on sequences of length k.

    Parameters
    ----------
    k : int   sequence length this group acts on (default 3 for codons)
    n : int   group order (default 3 = Z3)
    """

    def __init__(self, k: int = 3, n: int = 3):
        super().__init__(f"Z{n}")
        self._k = k
        self._n = n
        self._elements = [GroupElement(i, f"r{i}") for i in range(n)]

    @property
    def elements(self) -> List[GroupElement]:
        return self._elements

    def multiply(self, i: int, j: int) -> int:
        return (i + j) % self._n

    def act_on_symbol(self, g_idx: int, symbol: str) -> str:
        return symbol   # cyclic shift is positional, not per-symbol

    def act_on_sequence(self, g_idx: int, seq: List[str]) -> List[str]:
        if g_idx == 0:
            return list(seq)
        # Left-rotate by g_idx positions
        n = len(seq)
        shift = g_idx % n if n > 0 else 0
        return seq[shift:] + seq[:shift]


# ─────────────────────────────────────────────────────────────────────────────
# S3PermGroup (all permutations of 3 positions)
# ─────────────────────────────────────────────────────────────────────────────

class S3PermGroup(Group):
    """
    S3: all 6 permutations of 3 codon positions.
    Elements indexed 0..5 corresponding to:
      (0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)
    """
    _PERMS = [
        (0, 1, 2),   # e
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
    _NAMES = ["e", "(12)", "(01)", "(012)", "(02)", "(021)"]

    def __init__(self):
        super().__init__("S3_pos")
        self._elements = [
            GroupElement(i, self._NAMES[i]) for i in range(6)
        ]
        # Precompute multiplication table
        def compose(p, q):
            return tuple(p[q[i]] for i in range(3))
        perms = self._PERMS
        self._mul_table = [
            [perms.index(compose(perms[i], perms[j])) for j in range(6)]
            for i in range(6)
        ]

    @property
    def elements(self) -> List[GroupElement]:
        return self._elements

    def multiply(self, i: int, j: int) -> int:
        return self._mul_table[i][j]

    def act_on_symbol(self, g_idx: int, symbol: str) -> str:
        return symbol   # permutation is positional

    def act_on_sequence(self, g_idx: int, seq: List[str]) -> List[str]:
        perm = self._PERMS[g_idx]
        n = len(seq)
        if n != 3:
            # fallback: only permute first 3, leave rest
            result = list(seq)
            result[:3] = [seq[perm[i]] for i in range(min(3, n))]
            return result
        return [seq[perm[i]] for i in range(3)]


# ─────────────────────────────────────────────────────────────────────────────
# DirectProductGroup
# ─────────────────────────────────────────────────────────────────────────────

class DirectProductGroup(Group):
    """
    Direct product G1 × G2.

    Elements are pairs (i, j) encoded as a single index i * |G2| + j.
    Action: apply G1 first, then G2.

    Parameters
    ----------
    g1 : Group   left factor
    g2 : Group   right factor
    """

    def __init__(self, g1: Group, g2: Group):
        name = f"({g1.name}×{g2.name})"
        super().__init__(name)
        self._g1 = g1
        self._g2 = g2
        n1, n2 = g1.order, g2.order
        self._elements = [
            GroupElement(i * n2 + j, f"({g1.elements[i]},{g2.elements[j]})")
            for i in range(n1) for j in range(n2)
        ]

    @property
    def elements(self) -> List[GroupElement]:
        return self._elements

    def _decode(self, idx: int) -> Tuple[int, int]:
        n2 = self._g2.order
        return idx // n2, idx % n2

    def multiply(self, i: int, j: int) -> int:
        i1, i2 = self._decode(i)
        j1, j2 = self._decode(j)
        k1 = self._g1.multiply(i1, j1)
        k2 = self._g2.multiply(i2, j2)
        return k1 * self._g2.order + k2

    def act_on_symbol(self, g_idx: int, symbol: str) -> str:
        i1, i2 = self._decode(g_idx)
        s = self._g1.act_on_symbol(i1, symbol)
        s = self._g2.act_on_symbol(i2, s)
        return s

    def act_on_sequence(self, g_idx: int, seq: List[str]) -> List[str]:
        i1, i2 = self._decode(g_idx)
        # Apply G1 (may involve reversal/cyclic) then G2 (symbol-level)
        s = self._g1.act_on_sequence(i1, seq)
        s = self._g2.act_on_sequence(i2, s)
        return s


# ─────────────────────────────────────────────────────────────────────────────
# Factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def dna_default_group() -> DirectProductGroup:
    """
    G_DNA = Z2_comp × Z2_rev × Z3  (order 12, default for GE-CSG v4).
    Gives 10 codon orbits, mean orbit size 6.4.
    """
    return DirectProductGroup(
        DirectProductGroup(Z2ComplementGroup(), Z2ReversalGroup()),
        Z3CyclicGroup(k=3, n=3)
    )
