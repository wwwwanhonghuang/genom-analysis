"""
gecsg.core.coset
================
Coset space G / G_i: the Phase-1 alphabet of GE-CSG (v4).

Key objects
-----------
Coset          -- one equivalence class [g_x]_{G_i}
CosetSpace     -- the full quotient G / G_i; enumerates cosets,
                  computes orbits, orbit representatives, orbit sizes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, FrozenSet, Dict, Optional, Tuple
import functools


# -----------------------------------------------------------------------------
# Coset
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Coset:
    """
    One coset [g_x]_{G_i} = { g_x * h | h ∈ G_i }.

    Attributes
    ----------
    index          : canonical index within CosetSpace (0..n_cosets-1)
    representative : index of the canonical representative g_x ∈ G
    members        : frozenset of all group-element indices in this coset
    label          : human-readable label (optional)
    """
    index:          int
    representative: int                  # index in G.elements
    members:        FrozenSet[int]       # all g-indices in this coset
    label:          str = ""

    def __repr__(self) -> str:
        return self.label if self.label else f"C{self.index}"

    def __hash__(self) -> int:
        return hash(self.index)

    def __eq__(self, other) -> bool:
        if isinstance(other, Coset):
            return self.index == other.index
        return NotImplemented

    def __lt__(self, other) -> bool:
        """Enable sorting of cosets."""
        if isinstance(other, Coset):
            return self.index < other.index
        return NotImplemented


# -----------------------------------------------------------------------------
# CosetSpace
# -----------------------------------------------------------------------------

class CosetSpace:
    """
    The left coset space G / G_i.

    Parameters
    ----------
    group            : the ambient group G
    subgroup_indices : list of element indices forming G_i ⊆ G
    """

    def __init__(self, group, subgroup_indices: List[int]):
        from gecsg.core.group import Group
        self._G     = group
        self._H_idx = frozenset(subgroup_indices)

        if not group.is_subgroup(subgroup_indices):
            raise ValueError(
                f"Indices {subgroup_indices} do not form a subgroup of {group}."
            )

        # Build cosets by partitioning G
        self._cosets: List[Coset] = []
        self._elem_to_coset: Dict[int, int] = {}  # g_idx -> coset_index
        self._build_cosets()

    # -- Construction ----------------------------------------------------------

    def _build_cosets(self) -> None:
        G   = self._G
        assigned: Dict[int, int] = {}   # g_idx -> coset_idx
        coset_idx = 0

        for g_idx in range(G.order):
            if g_idx in assigned:
                continue
            # Build the left coset g * H
            members = frozenset(
                G.multiply(g_idx, h_idx) for h_idx in self._H_idx
            )
            # Canonical representative = smallest member index
            rep = min(members)
            label = f"[{G.elements[rep].name or str(rep)}]"
            coset = Coset(
                index=coset_idx,
                representative=rep,
                members=members,
                label=label,
            )
            self._cosets.append(coset)
            for m in members:
                assigned[m] = coset_idx
            coset_idx += 1

        self._elem_to_coset = assigned

    # -- Properties ------------------------------------------------------------

    @property
    def group(self):
        return self._G

    @property
    def subgroup_indices(self) -> FrozenSet[int]:
        return self._H_idx

    @property
    def cosets(self) -> List[Coset]:
        return self._cosets

    @property
    def size(self) -> int:
        """Number of cosets = |G| / |G_i|."""
        return len(self._cosets)

    def coset_of(self, g_idx: int) -> Coset:
        """Return the coset containing group element g_idx."""
        return self._cosets[self._elem_to_coset[g_idx]]

    def __getitem__(self, idx: int) -> Coset:
        return self._cosets[idx]

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        return iter(self._cosets)

    def __repr__(self) -> str:
        return (f"CosetSpace({self._G.name} / H, "
                f"|G|={self._G.order}, |H|={len(self._H_idx)}, "
                f"|G/H|={self.size})")

    # -- G-action on cosets ----------------------------------------------------

    def act(self, g_idx: int, coset: Coset) -> Coset:
        """
        Left action of g on coset [g_x]_{G_i}:
        g · [g_x]_{G_i} = [g * g_x]_{G_i}.
        """
        new_rep = self._G.multiply(g_idx, coset.representative)
        return self._cosets[self._elem_to_coset[new_rep]]

    # -- Orbit of a coset under G ----------------------------------------------

    def orbit_of_coset(self, coset: Coset) -> List[Coset]:
        """Return the G-orbit of a coset: { g · coset | g ∈ G }."""
        seen = set()
        result = []
        for g in self._G.elements:
            c = self.act(g.index, coset)
            if c.index not in seen:
                seen.add(c.index)
                result.append(c)
        return sorted(result)

    def orbit_size(self, coset: Coset) -> int:
        """|{g · coset | g ∈ G}|."""
        return len(self.orbit_of_coset(coset))

    def orbit_representative_coset(self, coset: Coset) -> Coset:
        """Return the coset with the smallest index in the orbit."""
        return min(self.orbit_of_coset(coset), key=lambda c: c.index)

    def witness_for(self, target: Coset, reference: Coset) -> Optional[int]:
        """
        Find g_idx such that g · reference = target.
        Used for witness recording in parsing.
        """
        for g in self._G.elements:
            if self.act(g.index, reference).index == target.index:
                return g.index
        return None

    # -- Orbit-maximising lifting ----------------------------------------------

    def orbit_maximising_coset(self, candidates: List[Coset]) -> List[Coset]:
        """
        Given a list of candidate cosets (from lifting a block w),
        return those with the maximum orbit size.
        Used in Λ*(w) = argmax_{c ∈ Λ(w)} N_orb(c).
        """
        if not candidates:
            return []
        sizes = {c.index: self.orbit_size(c) for c in candidates}
        max_size = max(sizes.values())
        return [c for c in candidates if sizes[c.index] == max_size]

    # -- Describe --------------------------------------------------------------

    def describe(self) -> None:
        """Print a summary of the coset space and orbit structure."""
        print(f"+= {self!r}")
        orbits: Dict[int, List[Coset]] = {}
        visited = set()
        for c in self._cosets:
            if c.index in visited:
                continue
            orb = self.orbit_of_coset(c)
            orb_rep = orb[0].index
            orbits[orb_rep] = orb
            for x in orb:
                visited.add(x.index)

        print(f"|  Cosets ({self.size}):")
        for c in self._cosets:
            rep_g = self._G.elements[c.representative]
            print(f"|    {c}  (rep=g{c.representative}:{rep_g}, "
                  f"members={sorted(c.members)})")

        print(f"|  G-orbits on coset space ({len(orbits)}):")
        for orb_rep_idx, orb in sorted(orbits.items()):
            labels = [str(c) for c in orb]
            print(f"|    orbit size={len(orb)}: {labels}")
        print("+=")
