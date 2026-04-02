"""
llm/neural_symbolic/grammar.py
Genomic PCFG grammar — non-terminal symbols, terminal alphabet,
and biologically-motivated structural constraints.

Non-terminal hierarchy (loosely following SO/eukaryotic gene structure):
  GENE        → root; covers the full locus
  TRANSCRIPT  → spliced product (exons only)
  LOCUS       → genomic region including UTRs and introns
  EXON        → coding or non-coding exon
  INTRON      → intronic region flanked by splice signals
  CODON       → triplet coding unit
  PROMOTER    → upstream regulatory element
  UTR         → untranslated region (5' or 3')
  REGULATORY  → enhancer / silencer / TFBS
  REPEAT      → repetitive element (SINE/LINE/tandem)
  SPLICE      → splice site signal (donor or acceptor)
  MOTIF       → short conserved sequence motif (e.g. TATA box)

Terminal symbols = single nucleotides: A C G T N
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Non-terminal index (NT_*)
# ---------------------------------------------------------------------------

NT_GENE       = 0
NT_TRANSCRIPT = 1
NT_LOCUS      = 2
NT_EXON       = 3
NT_INTRON     = 4
NT_CODON      = 5
NT_PROMOTER   = 6
NT_UTR        = 7
NT_REGULATORY = 8
NT_REPEAT     = 9
NT_SPLICE     = 10
NT_MOTIF      = 11

NUM_NT = 12

NT_NAMES: Dict[int, str] = {
    NT_GENE:       "GENE",
    NT_TRANSCRIPT: "TRANSCRIPT",
    NT_LOCUS:      "LOCUS",
    NT_EXON:       "EXON",
    NT_INTRON:     "INTRON",
    NT_CODON:      "CODON",
    NT_PROMOTER:   "PROMOTER",
    NT_UTR:        "UTR",
    NT_REGULATORY: "REGULATORY",
    NT_REPEAT:     "REPEAT",
    NT_SPLICE:     "SPLICE",
    NT_MOTIF:      "MOTIF",
}

NT_COLORS: Dict[int, str] = {
    NT_GENE:       "\033[1;35m",   # bold magenta
    NT_TRANSCRIPT: "\033[1;34m",   # bold blue
    NT_LOCUS:      "\033[0;34m",   # blue
    NT_EXON:       "\033[1;32m",   # bold green
    NT_INTRON:     "\033[0;33m",   # yellow
    NT_CODON:      "\033[0;32m",   # green
    NT_PROMOTER:   "\033[1;31m",   # bold red
    NT_UTR:        "\033[0;36m",   # cyan
    NT_REGULATORY: "\033[1;36m",   # bold cyan
    NT_REPEAT:     "\033[0;37m",   # gray
    NT_SPLICE:     "\033[0;35m",   # magenta
    NT_MOTIF:      "\033[0;31m",   # red
}

RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Terminal index
# ---------------------------------------------------------------------------

TERMINAL_VOCAB = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
NUM_T = len(TERMINAL_VOCAB)
T_NAMES = {v: k for k, v in TERMINAL_VOCAB.items()}

# ---------------------------------------------------------------------------
# Rule table
# Biologically-motivated binary rules: (parent, left-child, right-child)
# Each rule is a soft prior — the neural scorer learns deviations.
# ---------------------------------------------------------------------------

@dataclass
class BinaryRule:
    parent: int
    left:   int
    right:  int
    prior:  float = 0.0     # log-scale prior weight (0 = uniform)
    note:   str   = ""

BINARY_RULES: List[BinaryRule] = [
    # Gene-level composition
    BinaryRule(NT_GENE,       NT_PROMOTER,   NT_TRANSCRIPT,  prior= 1.0, note="promoter → transcript"),
    BinaryRule(NT_GENE,       NT_GENE,       NT_GENE,        prior= 0.5, note="gene concatenation"),
    BinaryRule(NT_GENE,       NT_LOCUS,      NT_UTR,         prior= 0.5, note="locus + 3'UTR"),
    BinaryRule(NT_GENE,       NT_UTR,        NT_LOCUS,       prior= 0.5, note="5'UTR + locus"),

    # Transcript = ordered exons
    BinaryRule(NT_TRANSCRIPT, NT_EXON,       NT_TRANSCRIPT,  prior= 1.0, note="exon + rest of transcript"),
    BinaryRule(NT_TRANSCRIPT, NT_TRANSCRIPT, NT_EXON,        prior= 1.0, note="transcript + terminal exon"),
    BinaryRule(NT_TRANSCRIPT, NT_EXON,       NT_EXON,        prior= 0.8, note="two exons"),

    # Locus = exon-intron structure
    BinaryRule(NT_LOCUS,      NT_EXON,       NT_INTRON,      prior= 1.0, note="exon + intron"),
    BinaryRule(NT_LOCUS,      NT_INTRON,     NT_EXON,        prior= 1.0, note="intron + exon"),
    BinaryRule(NT_LOCUS,      NT_LOCUS,      NT_LOCUS,       prior= 0.5, note="locus concatenation"),

    # Exon = codons
    BinaryRule(NT_EXON,       NT_CODON,      NT_EXON,        prior= 1.0, note="codon + rest of exon"),
    BinaryRule(NT_EXON,       NT_EXON,       NT_CODON,       prior= 1.0, note="exon + codon"),
    BinaryRule(NT_EXON,       NT_CODON,      NT_CODON,       prior= 0.8, note="two codons"),
    BinaryRule(NT_EXON,       NT_MOTIF,      NT_EXON,        prior= 0.5, note="motif in exon"),

    # Intron = splice signals flanking intronic sequence
    BinaryRule(NT_INTRON,     NT_SPLICE,     NT_INTRON,      prior= 1.0, note="splice donor + intron body"),
    BinaryRule(NT_INTRON,     NT_INTRON,     NT_SPLICE,      prior= 1.0, note="intron body + splice acceptor"),
    BinaryRule(NT_INTRON,     NT_REPEAT,     NT_INTRON,      prior= 0.5, note="repeat in intron"),
    BinaryRule(NT_INTRON,     NT_INTRON,     NT_REPEAT,      prior= 0.5, note="intron + repeat"),

    # Promoter = regulatory motifs
    BinaryRule(NT_PROMOTER,   NT_MOTIF,      NT_REGULATORY,  prior= 0.8, note="motif + regulatory"),
    BinaryRule(NT_PROMOTER,   NT_REGULATORY, NT_MOTIF,       prior= 0.8, note="regulatory + motif"),
    BinaryRule(NT_PROMOTER,   NT_PROMOTER,   NT_MOTIF,       prior= 0.5, note="promoter + motif"),

    # Codon = triplets (any binary split of 3 nucleotides)
    BinaryRule(NT_CODON,      NT_CODON,      NT_CODON,       prior= 0.5, note="codon recursion"),
    BinaryRule(NT_CODON,      NT_MOTIF,      NT_MOTIF,       prior= 0.3, note="codon as two motifs"),

    # UTR
    BinaryRule(NT_UTR,        NT_UTR,        NT_UTR,         prior= 0.5, note="UTR extension"),
    BinaryRule(NT_UTR,        NT_MOTIF,      NT_UTR,         prior= 0.5, note="motif in UTR"),
    BinaryRule(NT_UTR,        NT_REGULATORY, NT_UTR,         prior= 0.5, note="regulatory in UTR"),

    # Regulatory
    BinaryRule(NT_REGULATORY, NT_MOTIF,      NT_MOTIF,       prior= 0.8, note="two motifs"),
    BinaryRule(NT_REGULATORY, NT_REGULATORY, NT_REGULATORY,  prior= 0.5, note="regulatory extension"),

    # Repeat
    BinaryRule(NT_REPEAT,     NT_REPEAT,     NT_REPEAT,      prior= 0.8, note="tandem repeat"),

    # Splice site (donor GT / acceptor AG)
    BinaryRule(NT_SPLICE,     NT_MOTIF,      NT_MOTIF,       prior= 1.0, note="splice dinucleotide"),

    # Motif (leaf-level short conserved element)
    BinaryRule(NT_MOTIF,      NT_MOTIF,      NT_MOTIF,       prior= 0.5, note="motif extension"),
]

NUM_RULES = len(BINARY_RULES)

# Root distribution prior — gene structure starts at GENE
ROOT_PRIORS: Dict[int, float] = {
    NT_GENE:      2.0,
    NT_TRANSCRIPT:1.0,
    NT_LOCUS:     0.5,
    NT_EXON:      0.3,
    NT_PROMOTER:  0.3,
}


def rule_mask() -> "torch.Tensor":
    """
    Returns a (NT, NT, NT) binary tensor where 1 means the rule
    (parent → left, right) is in the grammar.  Used to zero out
    impossible rule probabilities during neural scoring.
    """
    import torch
    mask = torch.zeros(NUM_NT, NUM_NT, NUM_NT)
    for r in BINARY_RULES:
        mask[r.parent, r.left, r.right] = 1.0
    return mask


def root_prior_tensor() -> "torch.Tensor":
    import torch
    t = torch.full((NUM_NT,), -10.0)
    for nt, p in ROOT_PRIORS.items():
        t[nt] = p
    return t.log_softmax(0)