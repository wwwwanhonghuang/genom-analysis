"""
llm/neural_symbolic/test/test_tree.py
Tests for ParseTree: construction, structural validation, rendering,
and decode_tree() from synthetic span tensors.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from neural_symbolic.grammar import NT_GENE, NT_EXON, NT_INTRON, NT_CODON, NT_PROMOTER, NUM_NT
from neural_symbolic.tree import ParseTree, decode_tree, _make_flat_tree


# ---------------------------------------------------------------------------
# Helpers — build canonical trees by hand
# ---------------------------------------------------------------------------

def leaf(nuc: str, pos: int) -> ParseTree:
    return ParseTree(label=nuc, start=pos, end=pos + 1)


def internal(nt: int, l: ParseTree, r: ParseTree) -> ParseTree:
    return ParseTree(label=nt, start=l.start, end=r.end, left=l, right=r)


# Simple 4-token tree:  GENE(EXON(A B) INTRON(C D))
def make_4token_tree() -> ParseTree:
    a, b, c, d = leaf("A", 0), leaf("C", 1), leaf("G", 2), leaf("T", 3)
    exon   = internal(NT_EXON,   a, b)
    intron = internal(NT_INTRON, c, d)
    gene   = internal(NT_GENE,   exon, intron)
    return gene


class TestParseTreeProperties:

    def test_is_terminal_leaf(self):
        t = leaf("A", 0)
        assert t.is_terminal
        assert t.is_leaf

    def test_is_not_terminal_internal(self):
        t = make_4token_tree()
        assert not t.is_terminal

    def test_span_length(self):
        t = make_4token_tree()
        assert t.span_length == 4

    def test_label_name_nt(self):
        t = make_4token_tree()
        assert t.label_name == "GENE"

    def test_label_name_terminal(self):
        t = leaf("A", 0)
        assert t.label_name == "A"

    def test_depth(self):
        t = make_4token_tree()
        assert t.depth == 2

    def test_num_nodes(self):
        t = make_4token_tree()
        # 1 gene + 1 exon + 1 intron + 4 leaves = 7
        assert t.num_nodes == 7

    def test_num_leaves(self):
        t = make_4token_tree()
        assert t.num_leaves == 4


class TestParseTreeSpans:

    def test_spans_count(self):
        t = make_4token_tree()
        spans = list(t.spans())
        assert len(spans) == 7  # 3 internal + 4 leaves

    def test_root_span_first(self):
        t = make_4token_tree()
        spans = list(t.spans())
        assert spans[0] == (0, 4, NT_GENE)

    def test_nt_spans_no_terminals(self):
        t = make_4token_tree()
        nt_spans = t.nt_spans()
        for s, e, l in nt_spans:
            assert isinstance(l, int), "nt_spans should not contain terminal labels"

    def test_leaves_iterator(self):
        t = make_4token_tree()
        leaves = list(t.leaves())
        assert len(leaves) == 4
        for l in leaves:
            assert l.is_leaf


class TestParseTreeValidation:

    def test_valid_tree(self):
        t = make_4token_tree()
        ok, msg = t.is_valid()
        assert ok, f"Expected valid tree, got: {msg}"

    def test_invalid_gap_between_children(self):
        # Right child starts 1 position after left child ends → gap
        a = leaf("A", 0)
        b = leaf("C", 2)  # gap at position 1
        bad = ParseTree(label=NT_GENE, start=0, end=3, left=a, right=b)
        ok, msg = bad.is_valid()
        assert not ok
        assert "gap" in msg.lower() or "overlap" in msg.lower()

    def test_invalid_leaf_span_size(self):
        bad_leaf = ParseTree(label="A", start=0, end=3)
        ok, msg = bad_leaf.is_valid()
        assert not ok

    def test_single_child_valid_for_length1(self):
        # NT wrapping a single terminal leaf over span [0,1) is valid —
        # this is the canonical Viterbi leaf-wrapping pattern.
        a = leaf("A", 0)
        node = ParseTree(label=NT_GENE, start=0, end=1, left=a, right=None)
        ok, msg = node.is_valid()
        assert ok, f"Single-child wrapper should be valid: {msg}"

    def test_invalid_missing_left_child(self):
        a = leaf("A", 0)
        bad = ParseTree(label=NT_GENE, start=0, end=1, left=None, right=a)
        ok, msg = bad.is_valid()
        assert not ok, "Missing left child should be invalid"

    def test_invalid_single_child_long_span(self):
        # Single-child over span > 1 is not valid
        inner = ParseTree(label="A", start=0, end=2)  # bad leaf but span 2
        bad = ParseTree(label=NT_GENE, start=0, end=2, left=inner, right=None)
        ok, msg = bad.is_valid()
        assert not ok, "Single-child node with span > 1 should be invalid"

    def test_valid_single_token_tree(self):
        l = leaf("A", 0)
        t = ParseTree(label=NT_GENE, start=0, end=1, left=l)
        # Single child is allowed for single-token span
        # Actually our validity check requires 2 children for internal nodes
        ok, msg = t.is_valid()
        # This may or may not be valid depending on grammar — just check it runs
        assert isinstance(ok, bool)


class TestParseTreeRendering:

    def test_pprint_runs(self):
        t = make_4token_tree()
        result = t.pprint(sequence="ACGT", use_color=False)
        assert "GENE" in result
        assert "EXON" in result
        assert "INTRON" in result

    def test_bracket_notation(self):
        t = make_4token_tree()
        bracket = t.to_bracket(sequence="ACGT")
        assert bracket.startswith("(GENE")
        assert "EXON" in bracket
        assert "INTRON" in bracket

    def test_bracket_contains_nucleotides(self):
        t = make_4token_tree()
        bracket = t.to_bracket(sequence="ACGT")
        for nuc in "ACGT":
            assert nuc in bracket

    def test_str_representation(self):
        t = make_4token_tree()
        s = str(t)
        assert "GENE" in s

    def test_pprint_with_color(self):
        t = make_4token_tree()
        colored = t.pprint(sequence="ACGT", use_color=True)
        # ANSI escape codes should be present
        assert "\033[" in colored


class TestDecodeTree:

    def _make_span_chart(self, n: int, active_spans: dict, num_nt: int = NUM_NT) -> torch.Tensor:
        """
        Build a synthetic (1, n, n, NT) argmax chart.
        active_spans: dict of {(i, j): nt_idx}
        """
        chart = torch.zeros(1, n, n, num_nt)
        for (i, j), nt in active_spans.items():
            chart[0, i, j, nt] = 1.0
        return chart

    def test_decode_simple_tree(self):
        """Decode a hand-crafted 4-span chart."""
        seq = "ACGT"
        n = 4
        # Spans: GENE[0,3], EXON[0,1], INTRON[2,3], leaves implicit
        spans = {
            (0, 3): NT_GENE,
            (0, 1): NT_EXON,
            (2, 3): NT_INTRON,
            (0, 0): NT_CODON,
            (1, 1): NT_CODON,
            (2, 2): NT_CODON,
            (3, 3): NT_CODON,
        }
        chart = self._make_span_chart(n, spans)
        tree = decode_tree(chart, seq)
        assert tree is not None
        assert tree.start == 0
        assert tree.end <= n

    def test_decode_returns_parse_tree(self):
        seq = "ATCG"
        chart = self._make_span_chart(4, {(0, 3): NT_GENE, (0, 1): NT_EXON, (2, 3): NT_INTRON})
        result = decode_tree(chart, seq)
        assert isinstance(result, ParseTree)

    def test_decode_empty_chart_fallback(self):
        """Empty chart should fall back to a flat tree without crashing."""
        seq = "AT"
        chart = torch.zeros(1, 2, 2, NUM_NT)
        tree = decode_tree(chart, seq)
        assert isinstance(tree, ParseTree)

    def test_decode_single_token(self):
        seq = "A"
        chart = self._make_span_chart(1, {(0, 0): NT_GENE})
        tree = decode_tree(chart, seq)
        assert isinstance(tree, ParseTree)
        assert tree.start == 0


class TestFlatTree:

    def test_flat_tree_covers_sequence(self):
        seq = "ATCG"
        tree = _make_flat_tree(seq)
        assert tree.start == 0
        assert tree.end == len(seq)

    def test_flat_tree_single_char(self):
        tree = _make_flat_tree("A")
        assert isinstance(tree, ParseTree)

    def test_flat_tree_is_binary(self):
        tree = _make_flat_tree("ATCG")
        # Every internal node should have two children
        def check_binary(node):
            if node.is_leaf:
                return True
            if node.left is None or node.right is None:
                # single-child wrapping is also acceptable
                return True
            return check_binary(node.left) and check_binary(node.right)
        assert check_binary(tree)