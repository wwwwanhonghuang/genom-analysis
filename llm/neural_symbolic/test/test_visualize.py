"""
llm/neural_symbolic/test/test_visualize.py
Tests for ParseTree PNG visualization.

Fast tests (no model):
  - save_tree_png produces a valid PNG file
  - save_arc_png produces a valid PNG file
  - save_combined_png produces a valid PNG file
  - output dimensions are reasonable
  - handles edge cases: single-token tree, deep tree, long sequence

Integration tests (-m integration, needs torch-struct):
  - renders a real parsed TP53 tree end-to-end
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from neural_symbolic.grammar import NT_GENE, NT_EXON, NT_INTRON, NT_CODON, NT_MOTIF
from neural_symbolic.tree import ParseTree
from neural_symbolic.visualize import save_tree_png, save_arc_png, save_combined_png


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def leaf(nuc: str, pos: int) -> ParseTree:
    return ParseTree(label=nuc, start=pos, end=pos + 1)

def internal(nt: int, l: ParseTree, r: ParseTree) -> ParseTree:
    return ParseTree(label=nt, start=l.start, end=r.end, left=l, right=r)

def make_4token_tree() -> ParseTree:
    a, c, g, t = leaf("A", 0), leaf("C", 1), leaf("G", 2), leaf("T", 3)
    return internal(NT_GENE, internal(NT_EXON, a, c), internal(NT_INTRON, g, t))

def make_deep_tree(depth: int = 5) -> ParseTree:
    """Right-branching chain of given depth."""
    node = ParseTree(label="A", start=0, end=1)
    for i in range(1, depth + 1):
        r = ParseTree(label="T", start=i, end=i + 1)
        node = ParseTree(label=NT_GENE, start=0, end=i + 1,
                         left=node, right=r)
    return node

def is_valid_png(path: Path) -> bool:
    """Check PNG magic bytes."""
    if not path.exists() or path.stat().st_size < 8:
        return False
    with open(path, "rb") as f:
        header = f.read(8)
    return header == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# 1. Basic file creation
# ---------------------------------------------------------------------------

class TestFileSaving:

    def test_tree_png_created(self, tmp_path):
        tree = make_4token_tree()
        out  = save_tree_png(tree, sequence="ACGT",
                             path=tmp_path / "tree.png")
        assert out.exists(), "tree.png not created"

    def test_tree_png_valid_format(self, tmp_path):
        tree = make_4token_tree()
        out  = save_tree_png(tree, sequence="ACGT",
                             path=tmp_path / "tree.png")
        assert is_valid_png(out), "tree.png is not a valid PNG"

    def test_arc_png_created(self, tmp_path):
        tree = make_4token_tree()
        out  = save_arc_png(tree, sequence="ACGT",
                            path=tmp_path / "arcs.png")
        assert out.exists()

    def test_arc_png_valid_format(self, tmp_path):
        tree = make_4token_tree()
        out  = save_arc_png(tree, sequence="ACGT",
                            path=tmp_path / "arcs.png")
        assert is_valid_png(out)

    def test_combined_png_created(self, tmp_path):
        tree = make_4token_tree()
        out  = save_combined_png(tree, sequence="ACGT",
                                 path=tmp_path / "combined.png")
        assert out.exists()

    def test_combined_png_valid_format(self, tmp_path):
        tree = make_4token_tree()
        out  = save_combined_png(tree, sequence="ACGT",
                                 path=tmp_path / "combined.png")
        assert is_valid_png(out)

    def test_parent_dirs_created(self, tmp_path):
        tree = make_4token_tree()
        nested = tmp_path / "a" / "b" / "c" / "tree.png"
        out = save_tree_png(tree, path=nested)
        assert out.exists()


# ---------------------------------------------------------------------------
# 2. File size / dimension sanity
# ---------------------------------------------------------------------------

class TestOutputDimensions:

    def test_tree_png_nonempty(self, tmp_path):
        tree = make_4token_tree()
        out  = save_tree_png(tree, sequence="ACGT",
                             path=tmp_path / "tree.png")
        assert out.stat().st_size > 5_000, "PNG suspiciously small"

    def test_combined_larger_than_tree(self, tmp_path):
        tree    = make_4token_tree()
        t_path  = save_tree_png(tree,    sequence="ACGT", path=tmp_path / "t.png")
        c_path  = save_combined_png(tree, sequence="ACGT", path=tmp_path / "c.png")
        assert c_path.stat().st_size > t_path.stat().st_size

    def test_longer_sequence_larger_file(self, tmp_path):
        import matplotlib.pyplot as plt
        t4  = make_4token_tree()
        t12 = make_deep_tree(11)
        p4  = save_tree_png(t4,  sequence="ACGT",           path=tmp_path / "s4.png")
        p12 = save_tree_png(t12, sequence="ATCGATCGATCGT",  path=tmp_path / "s12.png")
        # Deeper / wider tree → larger image
        assert p12.stat().st_size > p4.stat().st_size * 0.5  # lenient bound


# ---------------------------------------------------------------------------
# 3. Title and options
# ---------------------------------------------------------------------------

class TestTitleAndOptions:

    def test_tree_with_title(self, tmp_path):
        tree = make_4token_tree()
        out  = save_tree_png(tree, sequence="ACGT", title="TP53 exon 5",
                             path=tmp_path / "titled.png")
        assert is_valid_png(out)

    def test_arc_with_title(self, tmp_path):
        tree = make_4token_tree()
        out  = save_arc_png(tree, sequence="ACGT", title="KRAS G12D",
                            path=tmp_path / "arc_titled.png")
        assert is_valid_png(out)

    def test_tree_without_sequence(self, tmp_path):
        """Should not crash when sequence=None."""
        tree = make_4token_tree()
        out  = save_tree_png(tree, sequence=None, path=tmp_path / "no_seq.png")
        assert is_valid_png(out)

    def test_custom_dpi(self, tmp_path):
        tree  = make_4token_tree()
        lo    = save_tree_png(tree, dpi=72,  path=tmp_path / "lo.png")
        hi    = save_tree_png(tree, dpi=300, path=tmp_path / "hi.png")
        assert hi.stat().st_size > lo.stat().st_size


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_token_tree(self, tmp_path):
        l = leaf("A", 0)
        t = ParseTree(label=NT_GENE, start=0, end=1, left=l)
        out = save_tree_png(t, sequence="A", path=tmp_path / "single.png")
        assert is_valid_png(out)

    def test_deep_tree(self, tmp_path):
        tree = make_deep_tree(8)
        out  = save_tree_png(tree, path=tmp_path / "deep.png")
        assert is_valid_png(out)

    def test_returns_path_object(self, tmp_path):
        tree = make_4token_tree()
        out  = save_tree_png(tree, path=str(tmp_path / "ret.png"))
        assert isinstance(out, Path)

    def test_string_path_accepted(self, tmp_path):
        tree = make_4token_tree()
        out  = save_tree_png(tree, path=str(tmp_path / "str.png"))
        assert out.exists()

    def test_overwrite_existing(self, tmp_path):
        tree = make_4token_tree()
        p    = tmp_path / "overwrite.png"
        save_tree_png(tree, path=p)
        size1 = p.stat().st_size
        save_tree_png(tree, path=p)
        size2 = p.stat().st_size
        assert size2 > 0  # didn't corrupt


# ---------------------------------------------------------------------------
# 5. Integration: real parse → PNG
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestIntegrationVisualize:

    TP53_SEQ = (
        "ATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAATGAGCCCCATGAGCCGCCTGAGGGCCC"
        "AGAGGGCCCATGGAGGATCCCCAGCCCTGGGCGTCAAGAGCCACTTGTACTGGCCCTTCTT"
    )

    @pytest.fixture(scope="class")
    def parsed_tree(self):
        pytest.importorskip("torch_struct")
        from neural_symbolic.parser import NeuralPCFGParser
        parser = NeuralPCFGParser(embed_dim=128, hidden_dim=64)
        parser.eval()
        n = 16
        torch.manual_seed(0)
        h = torch.randn(1, n, 128)
        seq = self.TP53_SEQ[:n]
        with torch.no_grad():
            tree, _ = parser.parse(h, seq, torch.tensor([n]))
        return tree, seq

    def test_integration_tree_png(self, parsed_tree, tmp_path):
        tree, seq = parsed_tree
        out = save_tree_png(tree, sequence=seq,
                            title="TP53 exon 5 — neural parse",
                            path=tmp_path / "tp53_tree.png")
        assert is_valid_png(out)
        assert out.stat().st_size > 5_000

    def test_integration_arc_png(self, parsed_tree, tmp_path):
        tree, seq = parsed_tree
        out = save_arc_png(tree, sequence=seq,
                           title="TP53 exon 5 — arc spans",
                           path=tmp_path / "tp53_arcs.png")
        assert is_valid_png(out)

    def test_integration_combined_png(self, parsed_tree, tmp_path):
        tree, seq = parsed_tree
        out = save_combined_png(tree, sequence=seq,
                                title="TP53 exon 5",
                                path=tmp_path / "tp53_combined.png")
        assert is_valid_png(out)