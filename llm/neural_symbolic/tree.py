"""
llm/neural_symbolic/tree.py
ParseTree — the symbolic output of the neural PCFG parser.

Responsibilities:
  - Decoded from a (n, n, NT) span-indicator tensor produced by torch-struct
  - Pretty-print with ANSI colours aligned to non-terminal type
  - Convert to/from nltk.Tree for downstream analysis
  - Yield spans as (start, end, nt_label) triples
  - Verify structural properties (binary, covers full sequence, no overlaps)
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

from .grammar import NT_NAMES, NT_COLORS, RESET, NUM_NT


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------

@dataclass
class ParseTree:
    """
    A node in a binary parse tree over a DNA sequence.

    Attributes:
        label   : non-terminal index (int) or terminal nucleotide (str)
        start   : inclusive start position in the original sequence
        end     : exclusive end position
        left    : left child (ParseTree or None for terminals)
        right   : right child (ParseTree or None for terminals)
        score   : log-probability of this sub-tree under the grammar
    """
    label:  int | str
    start:  int
    end:    int
    left:   Optional[ParseTree] = field(default=None, repr=False)
    right:  Optional[ParseTree] = field(default=None, repr=False)
    score:  float = 0.0

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        return isinstance(self.label, str)

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    @property
    def span_length(self) -> int:
        return self.end - self.start

    @property
    def label_name(self) -> str:
        if self.is_terminal:
            return self.label
        return NT_NAMES.get(self.label, f"NT{self.label}")

    @property
    def depth(self) -> int:
        if self.is_leaf:
            return 0
        children = [c for c in (self.left, self.right) if c is not None]
        return 1 + max(c.depth for c in children)

    @property
    def num_nodes(self) -> int:
        n = 1
        if self.left:  n += self.left.num_nodes
        if self.right: n += self.right.num_nodes
        return n

    @property
    def num_leaves(self) -> int:
        if self.is_leaf:
            return 1
        return sum(c.num_leaves for c in (self.left, self.right) if c)

    # ------------------------------------------------------------------
    # Span enumeration
    # ------------------------------------------------------------------

    def spans(self) -> Iterator[Tuple[int, int, int | str]]:
        """Yield all (start, end, label) triples in top-down left-to-right order."""
        yield (self.start, self.end, self.label)
        if self.left:
            yield from self.left.spans()
        if self.right:
            yield from self.right.spans()

    def leaves(self) -> Iterator["ParseTree"]:
        if self.is_leaf:
            yield self
        else:
            if self.left:  yield from self.left.leaves()
            if self.right: yield from self.right.leaves()

    def nt_spans(self) -> List[Tuple[int, int, int]]:
        """Return only non-terminal spans (excludes terminal leaves)."""
        return [(s, e, l) for s, e, l in self.spans() if isinstance(l, int)]

    # ------------------------------------------------------------------
    # Structural validation
    # ------------------------------------------------------------------

    def is_valid(self) -> Tuple[bool, str]:
        """
        Check the tree is structurally valid:
          - binary (each internal node has exactly 2 children)
          - children spans are contiguous and non-overlapping
          - leaves cover [start, end) exactly
        Returns (True, "") or (False, reason).
        """
        if self.is_leaf:
            if self.span_length != 1:
                return False, f"Leaf at [{self.start},{self.end}) has span != 1"
            return True, ""

        # A single-child node is valid only when it is a length-1 span
        # wrapping one terminal leaf (NT → nucleotide) — produced by Viterbi
        if self.right is None:
            if self.left is None:
                return False, f"Internal node {self.label_name} has no children"
            if self.span_length != 1:
                return False, f"Single-child node {self.label_name} has span > 1"
            return self.left.is_valid()
        if self.left is None:
            return False, f"Internal node {self.label_name} missing left child"

        # Child spans must be contiguous
        if self.left.end != self.right.start:
            return False, (
                f"{self.label_name}: left ends at {self.left.end}, "
                f"right starts at {self.right.start} (gap/overlap)"
            )

        # Children must cover parent exactly
        if self.left.start != self.start or self.right.end != self.end:
            return False, (
                f"{self.label_name}[{self.start},{self.end}): "
                f"children [{self.left.start},{self.right.end}) don't match"
            )

        ok_l, msg_l = self.left.is_valid()
        if not ok_l:
            return False, msg_l
        ok_r, msg_r = self.right.is_valid()
        if not ok_r:
            return False, msg_r

        return True, ""

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def pprint(self, sequence: Optional[str] = None,
               use_color: bool = True, indent: int = 0) -> str:
        """
        Return a human-readable string representation.
        If sequence is provided, terminal nodes show their nucleotide.
        """
        color = ""
        reset = ""
        if use_color and not self.is_terminal:
            color = NT_COLORS.get(self.label, "")
            reset = RESET

        label_str = self.label_name
        span_str  = f"[{self.start}:{self.end}]"

        if self.is_leaf:
            nuc = ""
            if sequence:
                nuc = " " + sequence[self.start:self.end]
            return " " * indent + f"{color}{label_str}{span_str}{nuc}{reset}"

        lines = [" " * indent + f"{color}({label_str}{span_str}{reset}"]
        if self.left:
            lines.append(self.left.pprint(sequence, use_color, indent + 2))
        if self.right:
            lines.append(self.right.pprint(sequence, use_color, indent + 2))
        lines.append(" " * indent + f"{color}){reset}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.pprint(use_color=False)

    def to_bracket(self, sequence: Optional[str] = None) -> str:
        """Penn Treebank bracket notation: (GENE (EXON ...) (INTRON ...))"""
        if self.is_leaf:
            nuc = sequence[self.start] if sequence else f"pos{self.start}"
            return f"({self.label_name} {nuc})"
        left  = self.left.to_bracket(sequence)  if self.left  else ""
        right = self.right.to_bracket(sequence) if self.right else ""
        return f"({self.label_name} {left} {right})"

    # ------------------------------------------------------------------
    # nltk bridge
    # ------------------------------------------------------------------

    def to_nltk(self, sequence: Optional[str] = None) -> "nltk.Tree":
        """
        Convert to an nltk.Tree object.
        Requires nltk to be installed (not a hard dependency).
        """
        try:
            from nltk import Tree as NLTKTree
        except ImportError:
            raise ImportError("nltk is required for to_nltk(). pip install nltk")

        if self.is_leaf:
            nuc = sequence[self.start] if sequence else f"pos{self.start}"
            return NLTKTree(self.label_name, [nuc])

        children = []
        if self.left:
            children.append(self.left.to_nltk(sequence))
        if self.right:
            children.append(self.right.to_nltk(sequence))
        return NLTKTree(self.label_name, children)


# ---------------------------------------------------------------------------
# Decode from torch-struct SentCFG argmax
# ---------------------------------------------------------------------------

def decode_tree(
    argmax: "torch.Tensor",
    sequence: str,
    lengths: Optional[List[int]] = None,
    batch_idx: int = 0,
) -> ParseTree:
    """
    Decode a ParseTree from the argmax tensor returned by torch-struct's SentCFG.

    The SentCFG argmax is a (batch, n, n, NT) sparse binary tensor where
    argmax[b, i, j, nt] = 1 means non-terminal `nt` spans positions [i, j+1).
    (torch-struct uses inclusive-end indexing for spans)

    Algorithm:
      - Collect all active (i, j, nt) triples from the argmax chart
      - Build a chart dict: chart[i][j] → ParseTree node
      - Recursively reconstruct the binary tree top-down via the CYK split points
      - Attach terminal leaves from the sequence

    Args:
        argmax   : (batch, n, n, NT) or (n, n, NT) binary tensor
        sequence : original DNA string (length n)
        lengths  : optional list of per-batch actual lengths
        batch_idx: which batch element to decode

    Returns:
        ParseTree rooted at the root span
    """
    import torch

    # Strip batch dim
    if argmax.ndim == 4:
        chart_tensor = argmax[batch_idx]   # (n, n, NT)
    else:
        chart_tensor = argmax              # (n, n, NT)

    n = chart_tensor.shape[0]
    seq_len = lengths[batch_idx] if lengths else n

    # Collect active spans: dict[(i,j)] = nt_index
    active: dict[tuple[int, int], int] = {}
    chart_np = chart_tensor[:seq_len, :seq_len].detach()

    # torch-struct SentCFG uses (i, j) where j is inclusive end
    nonzero = chart_np.nonzero(as_tuple=False)
    for idx in nonzero:
        i, j, nt = idx[0].item(), idx[1].item(), idx[2].item()
        active[(i, j)] = nt

    if not active:
        # Fallback: trivial flat tree if no spans found (e.g. length-1 input)
        return _make_flat_tree(sequence[:seq_len], 0)

    # Find root span: largest span covering [0, seq_len-1]
    root_span = (0, seq_len - 1)
    if root_span not in active:
        # Find the widest span starting at 0
        root_span = max(
            ((i, j) for (i, j) in active if i == 0),
            key=lambda x: x[1],
            default=(0, 0),
        )

    def build(i: int, j: int) -> ParseTree:
        """Recursively build tree from span chart."""
        nt = active.get((i, j))

        if i == j:
            # Terminal leaf
            nuc = sequence[i] if i < len(sequence) else "N"
            leaf = ParseTree(label=nuc, start=i, end=i + 1)
            if nt is not None:
                # Wrap in an NT node
                return ParseTree(label=nt, start=i, end=i + 1, left=leaf)
            return leaf

        if nt is None:
            # No label for this span — infer from children
            nt = 0  # default to GENE

        # Find the split point: look for child spans that partition [i,j]
        split = None
        for mid in range(i, j):
            if (i, mid) in active and (mid + 1, j) in active:
                split = mid
                break

        if split is None:
            # No clean split found — binary-split at midpoint (fallback)
            split = (i + j) // 2

        left  = build(i, split)
        right = build(split + 1, j)
        return ParseTree(
            label=nt,
            start=i,
            end=j + 1,
            left=left,
            right=right,
        )

    root = build(*root_span)
    return root


def _make_flat_tree(sequence: str, nt: int = 0) -> ParseTree:
    """Trivial right-branching tree for sequences too short for CYK."""
    from .grammar import NT_GENE
    n = len(sequence)
    if n == 0:
        return ParseTree(label=sequence or "N", start=0, end=1)
    if n == 1:
        leaf = ParseTree(label=sequence[0], start=0, end=1)
        return ParseTree(label=NT_GENE, start=0, end=1, left=leaf)

    # Right-branching
    nodes = [
        ParseTree(label=sequence[i], start=i, end=i + 1)
        for i in range(n)
    ]
    while len(nodes) > 1:
        new_nodes = []
        i = 0
        while i < len(nodes) - 1:
            l, r = nodes[i], nodes[i + 1]
            new_nodes.append(ParseTree(
                label=NT_GENE,
                start=l.start,
                end=r.end,
                left=l,
                right=r,
            ))
            i += 2
        if i < len(nodes):
            new_nodes.append(nodes[i])
        nodes = new_nodes
    return nodes[0]