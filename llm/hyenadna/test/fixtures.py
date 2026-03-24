"""
llm/hyenadna/test/fixtures.py
Shared DNA sequences and ground-truth expectations used across all test cases.

Sequences are drawn from clinically significant cancer loci so the tests
double as sanity-checks on biologically meaningful inputs.
"""

# ---------------------------------------------------------------------------
# TP53 — exon 5 (codons 126–169), frequent somatic mutation hotspot
# Wildtype vs R175H (c.524G>A) — most common p53 gain-of-function mutation
# ---------------------------------------------------------------------------
TP53_EXON5_WT = (
    "ATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAATGAGCCCCATGAGCCGCCTGAGGGCCC"
    "AGAGGGCCCATGGAGGATCCCCAGCCCTGGGCGTCAAGAGCCACTTGTACTGGCCCTTCTT"
    "GCAGACTGTGTCCAGGG"
)
TP53_EXON5_R175H = (
    "ATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAATGAGCCCCATGAGCCGCCTGAGGGCCC"
    "AGAGGGCCCATGGAGGATCCCCAGCCCTGGGCGTCAAGAGCCACTTGTACTGGCCCTTCTT"
    "GCAGACTGTGTCCAAGG"   # G>A at codon 175
)

# ---------------------------------------------------------------------------
# BRCA1 — exon 11 region, splice site neighbourhood
# Wildtype vs c.4185+1G>T — pathogenic splice donor loss
# ---------------------------------------------------------------------------
BRCA1_EXON11_WT  = (
    "CAGCTACAATTTGCTTTTACACACTTTAGTTTGTTTATTTTTCTAAAGCATCTGATAGTTG"
    "GAGGTTTGTTCATCTTTATGAAAACTAAAACCTGTGTTTACAAAAACTTGCAAGGAAGAATC"
)
BRCA1_EXON11_MUT = (
    "CAGCTACAATTTGCTTTTACACACTTTAGTTTGTTTATTTTTCTAAAGCATCTGATAGTTG"
    "GAGGTTTGTTCATCTTTATGAAAACTAAAACCTGTGTTTACAAAAACTTGCAAGGAAGAATC"[:60]
    + "T"  # splice donor G>T
    + "GAGGTTTGTTCATCTTTATGAAAACTAAAACCTGTGTTTACAAAAACTTGCAAGGAAGAATC"[61:]
)

# ---------------------------------------------------------------------------
# KRAS — codon 12 (G12D: c.35G>A), most common oncogenic point mutation
# ---------------------------------------------------------------------------
KRAS_WT  = (
    "ATGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGA"
    "TACAGCTAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGG"
)
KRAS_G12D = (
    "ATGACTGAATATAAACTTGTGGTAGTTGGAGCTGATGGCGTAGGCAAGAGTGCCTTGACGA"  # G>A pos 34
    "TACAGCTAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGG"
)

# ---------------------------------------------------------------------------
# EGFR — exon 19 deletion hotspot (del E746-A750), lung cancer driver
# ---------------------------------------------------------------------------
EGFR_EXON19_WT  = (
    "CATGGTGGAGGGCATGAACCTGGCCCTCAAGAAAGTAGCCATCATCACAGAGGGCATGAGCT"
    "GGGTCATCGAGGCCATCAAGAAGCTGGAGAAGGAGATGGCAGAGGGCCTGAACAACATCCTG"
)
EGFR_EXON19_DEL = (
    "CATGGTGGAGGGCATGAACCTGGCCCTCAAGAAAGTAGCCATCATCACAGAGGGCATGAGCT"
    "GGGTCATCGAGGCCATCAAGAAGCTGGAG"  # del GAATTAAGAGAAGCA (E746-A750)
    "AAGGAGATGGCAGAGGGCCTGAACAACATCCTG"
)

# ---------------------------------------------------------------------------
# Random / edge cases
# ---------------------------------------------------------------------------
SHORT_SEQ   = "ATCGATCG"                          # 8 bp — below any meaningful window
POLY_A      = "A" * 256                            # homopolymer run
LONG_SEQ    = ("ATCG" * 250)[:1000]               # 1000 bp — near DNABERT context limit
AMBIGUOUS   = "ATCGNNNATCG"                        # contains N (unknown bases)
MIXED_CASE  = "atcgATCGatcg"                       # lowercase input

# ---------------------------------------------------------------------------
# Expected embedding properties (computed empirically on hyenadna-tiny)
# These are soft bounds, not exact values — allow ±20% tolerance
# ---------------------------------------------------------------------------
# Hidden sizes differ by checkpoint — do not use EXPECTED_EMBED_DIM for cross-variant tests
# tiny-1k   : 128
# small-32k : 256
# medium-160k/450k : 256
# large-1m  : 256
EXPECTED_EMBED_DIM   = 128    # valid for hyenadna-tiny-1k only
EXPECTED_NORM_MIN    = 1.0    # embedding norm lower bound
EXPECTED_NORM_MAX    = 50.0   # embedding norm upper bound