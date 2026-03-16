"""
gecsg.parser.parse_tree
=======================
ParseNode: one node in a GE-CSG two-phase derivation tree.

Tree anatomy
------------
Internal node  (is_terminal=False)
    label    = NT name, e.g. "Gene", "CDS", "Codon"
    span     = (start, end) in coset-string coordinates (one unit = one codon)
    children = tuple of child ParseNodes
    rule     = the GenerationRule (Arrow 1) that was applied

Leaf node  (is_terminal=True)
    label     = coset label, e.g. "[e]"
    span      = (i, i+1)
    coset     = Coset object from the Phase-1 string
    raw_codon = concrete codon string, e.g. "ATG"
    children  = ()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

from gecsg.core.coset import Coset
from gecsg.grammar.rule import GenerationRule


@dataclass
class ParseNode:
    """
    One node in a GE-CSG parse tree.

    Attributes
    ----------
    label       : NT name (internal) or coset label (leaf)
    span        : (start_incl, end_excl) in coset-string positions
    is_terminal : True for leaf nodes (coset positions), False for NT nodes
    children    : child ParseNodes; empty for leaves
    rule        : GenerationRule applied at this node (internal only)
    coset       : Coset object (leaf only)
    raw_codon   : concrete 3-char string (leaf only), e.g. "ATG"
    """
    label:       str
    span:        Tuple[int, int]
    is_terminal: bool                        = False
    children:    Tuple["ParseNode", ...]     = field(default_factory=tuple)
    rule:        Optional[GenerationRule]    = None
    coset:       Optional[Coset]             = None
    raw_codon:   Optional[str]               = None

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def raw_string(self) -> str:
        """Reconstruct the raw DNA string covered by this subtree."""
        if self.is_terminal:
            return self.raw_codon or ""
        return "".join(ch.raw_string for ch in self.children)

    @property
    def depth(self) -> int:
        """Depth of the subtree rooted here."""
        if not self.children:
            return 0
        return 1 + max(ch.depth for ch in self.children)

    def pprint(self, indent: int = 0) -> None:
        """Pretty-print the tree to stdout."""
        prefix = "  " * indent
        if self.is_terminal:
            print(f"{prefix}[LEAF] {self.label} → {self.raw_codon}  "
                  f"span={self.span}")
        else:
            rule_s = f"  via {self.rule}" if self.rule else ""
            print(f"{prefix}[{self.label}] span={self.span}{rule_s}")
            for ch in self.children:
                ch.pprint(indent + 1)

    def __repr__(self) -> str:
        kind = "LEAF" if self.is_terminal else "NT"
        return f"ParseNode({kind}:{self.label}{list(self.span)})"
