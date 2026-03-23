"""
gecsg.io.gtf
============
GTF (Gene Transfer Format) file reader and object model.

GTF format (tab-separated, 9 fields per line)
----------------------------------------------
  1  seqname    chromosome / scaffold name
  2  source     annotation source (e.g. "ensembl", "havana")
  3  feature    record type: gene | transcript | exon | CDS |
                             start_codon | stop_codon | UTR | ...
  4  start      1-based genomic start (inclusive)
  5  end        1-based genomic end   (inclusive)
  6  score      numeric or '.'
  7  strand     '+' | '-' | '.'
  8  frame      0 | 1 | 2 | '.'   (reading frame of CDS)
  9  attributes key "value" pairs separated by ';'

Object hierarchy
----------------
  GTFRecord          -- one raw parsed line (all 9 fields)
  Exon               -- an exon interval
  CDSSegment         -- a CDS interval (with reading frame)
  StartCodonRecord   -- start codon annotation
  StopCodonRecord    -- stop codon annotation
  Transcript         -- groups Exons + CDSSegments for one transcript
  Gene               -- groups Transcripts for one gene_id

Usage
-----
  from gecsg.io.gtf import GTFReader

  reader = GTFReader()
  genes  = reader.read("Homo_sapiens.GRCh38.gtf")

  for gene in genes.values():
      for tx in gene.transcripts.values():
          cds_seq = tx.cds_sequence(fasta)   # requires a FASTA dict
          result  = bfs_parser.parse(cds_seq)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Raw record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GTFRecord:
    """
    One parsed GTF line (all 9 fields).

    All coordinate fields use GTF convention (1-based, inclusive on both ends).
    """
    seqname:    str
    source:     str
    feature:    str               # 'gene', 'transcript', 'exon', 'CDS', ...
    start:      int               # 1-based
    end:        int               # 1-based inclusive
    score:      Optional[float]   # None if '.'
    strand:     str               # '+', '-', or '.'
    frame:      Optional[int]     # 0/1/2 for CDS; None if '.'
    attributes: Dict[str, str]    # parsed key→value pairs

    @property
    def length(self) -> int:
        """Genomic length of this record (end - start + 1)."""
        return self.end - self.start + 1

    def attr(self, key: str, default: str = "") -> str:
        """Return attribute value, or default if key is absent."""
        return self.attributes.get(key, default)

    def __str__(self) -> str:
        score_s = f"{self.score:.3f}" if self.score is not None else "."
        frame_s = str(self.frame) if self.frame is not None else "."
        attrs   = "; ".join(f'{k} "{v}"' for k, v in self.attributes.items())
        return (f"{self.seqname}\t{self.source}\t{self.feature}\t"
                f"{self.start}\t{self.end}\t{score_s}\t{self.strand}\t"
                f"{frame_s}\t{attrs}")


# ─────────────────────────────────────────────────────────────────────────────
# Typed interval classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Exon:
    """One exon interval."""
    seqname:       str
    source:        str
    start:         int            # 1-based
    end:           int            # 1-based inclusive
    strand:        str
    transcript_id: str
    gene_id:       str
    exon_number:   Optional[int]  # None if not provided
    attributes:    Dict[str, str] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @classmethod
    def from_record(cls, r: GTFRecord) -> "Exon":
        return cls(
            seqname=r.seqname,
            source=r.source,
            start=r.start,
            end=r.end,
            strand=r.strand,
            transcript_id=r.attr("transcript_id"),
            gene_id=r.attr("gene_id"),
            exon_number=_parse_exon_number(r.attributes),
            attributes=dict(r.attributes),
        )

    def __repr__(self) -> str:
        return (f"Exon({self.seqname}:{self.start}-{self.end}"
                f"[{self.strand}] tx={self.transcript_id})")


@dataclass
class CDSSegment:
    """One CDS interval (a contiguous coding region fragment)."""
    seqname:       str
    source:        str
    start:         int            # 1-based
    end:           int            # 1-based inclusive
    strand:        str
    frame:         int            # reading frame: 0, 1, or 2
    transcript_id: str
    gene_id:       str
    attributes:    Dict[str, str] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @classmethod
    def from_record(cls, r: GTFRecord) -> "CDSSegment":
        return cls(
            seqname=r.seqname,
            source=r.source,
            start=r.start,
            end=r.end,
            strand=r.strand,
            frame=r.frame if r.frame is not None else 0,
            transcript_id=r.attr("transcript_id"),
            gene_id=r.attr("gene_id"),
            attributes=dict(r.attributes),
        )

    def __repr__(self) -> str:
        return (f"CDS({self.seqname}:{self.start}-{self.end}"
                f"[{self.strand}] frame={self.frame} tx={self.transcript_id})")


@dataclass
class StartCodonRecord:
    """Start codon annotation."""
    seqname:       str
    start:         int
    end:           int
    strand:        str
    frame:         int
    transcript_id: str
    gene_id:       str
    attributes:    Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_record(cls, r: GTFRecord) -> "StartCodonRecord":
        return cls(
            seqname=r.seqname,
            start=r.start,
            end=r.end,
            strand=r.strand,
            frame=r.frame if r.frame is not None else 0,
            transcript_id=r.attr("transcript_id"),
            gene_id=r.attr("gene_id"),
            attributes=dict(r.attributes),
        )


@dataclass
class StopCodonRecord:
    """Stop codon annotation."""
    seqname:       str
    start:         int
    end:           int
    strand:        str
    frame:         int
    transcript_id: str
    gene_id:       str
    attributes:    Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_record(cls, r: GTFRecord) -> "StopCodonRecord":
        return cls(
            seqname=r.seqname,
            start=r.start,
            end=r.end,
            strand=r.strand,
            frame=r.frame if r.frame is not None else 0,
            transcript_id=r.attr("transcript_id"),
            gene_id=r.attr("gene_id"),
            attributes=dict(r.attributes),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Transcript
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Transcript:
    """
    One annotated transcript.

    Groups all Exon, CDSSegment, StartCodonRecord, and StopCodonRecord
    lines that share the same transcript_id.
    """
    transcript_id:  str
    gene_id:        str
    seqname:        str
    source:         str
    start:          int
    end:            int
    strand:         str
    biotype:        str                       # transcript_biotype or empty
    attributes:     Dict[str, str]            = field(default_factory=dict)
    exons:          List[Exon]                = field(default_factory=list)
    cds_segments:   List[CDSSegment]          = field(default_factory=list)
    start_codons:   List[StartCodonRecord]    = field(default_factory=list)
    stop_codons:    List[StopCodonRecord]     = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_coding(self) -> bool:
        """True iff the transcript has at least one CDS segment."""
        return len(self.cds_segments) > 0

    @property
    def cds_length(self) -> int:
        """Total length of all CDS segments (nucleotides)."""
        return sum(s.length for s in self.cds_segments)

    @property
    def sorted_cds(self) -> List[CDSSegment]:
        """CDS segments sorted by genomic position (ascending for +, descending for -)."""
        segs = sorted(self.cds_segments, key=lambda s: s.start)
        if self.strand == "-":
            segs = list(reversed(segs))
        return segs

    @property
    def sorted_exons(self) -> List[Exon]:
        exons = sorted(self.exons, key=lambda e: e.start)
        if self.strand == "-":
            exons = list(reversed(exons))
        return exons

    def cds_sequence(self, fasta: Dict[str, str]) -> str:
        """
        Reconstruct the CDS nucleotide sequence from a FASTA dictionary.

        Parameters
        ----------
        fasta : dict mapping seqname -> full chromosome/scaffold sequence

        Returns
        -------
        str : concatenated CDS sequence in 5'→3' direction.
              Raises KeyError if seqname is absent from fasta.
              Raises ValueError if any CDS segment is out of range.
        """
        chrom = fasta.get(self.seqname)
        if chrom is None:
            raise KeyError(
                f"Sequence '{self.seqname}' not found in FASTA."
            )
        parts: List[str] = []
        for seg in self.sorted_cds:
            # GTF is 1-based; Python strings are 0-based
            s = seg.start - 1
            e = seg.end        # inclusive → exclusive in Python
            if e > len(chrom):
                raise ValueError(
                    f"CDS segment {seg} extends beyond chromosome length {len(chrom)}."
                )
            seq_piece = chrom[s:e]
            if self.strand == "-":
                seq_piece = _reverse_complement(seq_piece)
            parts.append(seq_piece)
        return "".join(parts)

    def __repr__(self) -> str:
        return (f"Transcript({self.transcript_id} "
                f"{self.seqname}:{self.start}-{self.end}[{self.strand}] "
                f"exons={len(self.exons)} cds_segs={len(self.cds_segments)})")


# ─────────────────────────────────────────────────────────────────────────────
# Gene
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Gene:
    """
    One annotated gene.

    Groups all Transcript objects that share the same gene_id.
    """
    gene_id:     str
    seqname:     str
    source:      str
    start:       int
    end:         int
    strand:      str
    gene_name:   str                      # gene_name attribute or empty
    biotype:     str                      # gene_biotype or empty
    attributes:  Dict[str, str]           = field(default_factory=dict)
    transcripts: Dict[str, Transcript]    = field(default_factory=dict)

    @property
    def n_transcripts(self) -> int:
        return len(self.transcripts)

    @property
    def coding_transcripts(self) -> List[Transcript]:
        return [tx for tx in self.transcripts.values() if tx.is_coding]

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def __repr__(self) -> str:
        return (f"Gene({self.gene_id} {self.gene_name!r} "
                f"{self.seqname}:{self.start}-{self.end}[{self.strand}] "
                f"tx={self.n_transcripts})")


# ─────────────────────────────────────────────────────────────────────────────
# GTFReader
# ─────────────────────────────────────────────────────────────────────────────

class GTFReader:
    """
    Read a GTF file and return a dict of Gene objects keyed by gene_id.

    Usage
    -----
    >>> reader = GTFReader()
    >>> genes  = reader.read("Homo_sapiens.GRCh38.109.gtf")
    >>> gene   = genes["ENSG00000139618"]   # BRCA2
    >>> tx     = list(gene.transcripts.values())[0]
    >>> seq    = tx.cds_sequence(fasta_dict)

    Parameters
    ----------
    strict : bool
        If True (default), raise ValueError for malformed lines.
        If False, skip malformed lines with a warning.
    """

    def __init__(self, strict: bool = True):
        self.strict = strict

    # ── Public API ─────────────────────────────────────────────────────────────

    def read(self, path: str | Path) -> Dict[str, Gene]:
        """
        Parse a GTF file and return {gene_id: Gene}.

        Lines beginning with '#' are treated as comments and skipped.
        Only lines with feature ∈ {gene, transcript, exon, CDS,
        start_codon, stop_codon} are processed; others are stored as
        GTFRecord inside the matching transcript (if one exists).
        """
        path = Path(path)
        genes: Dict[str, Gene] = {}
        tx_map: Dict[str, Transcript] = {}   # transcript_id -> Transcript

        for rec in self._iter_records(path):
            gid = rec.attr("gene_id")
            tid = rec.attr("transcript_id")

            feat = rec.feature.lower()

            if feat == "gene":
                if gid not in genes:
                    genes[gid] = _make_gene(rec)
                continue

            if feat == "transcript":
                if gid not in genes:
                    genes[gid] = _make_gene_stub(rec)
                tx = _make_transcript(rec)
                genes[gid].transcripts[tid] = tx
                tx_map[tid] = tx
                continue

            # For sub-transcript features, ensure parent objects exist
            if gid and gid not in genes:
                genes[gid] = _make_gene_stub(rec)
            if tid and tid not in tx_map:
                tx = _make_transcript_stub(rec)
                genes[gid].transcripts[tid] = tx
                tx_map[tid] = tx

            tx = tx_map.get(tid)
            if tx is None:
                continue

            if feat == "exon":
                tx.exons.append(Exon.from_record(rec))
            elif feat == "cds":
                tx.cds_segments.append(CDSSegment.from_record(rec))
            elif feat == "start_codon":
                tx.start_codons.append(StartCodonRecord.from_record(rec))
            elif feat == "stop_codon":
                tx.stop_codons.append(StopCodonRecord.from_record(rec))
            # Other features (UTR, Selenocysteine, etc.) are silently ignored.

        return genes

    def iter_records(self, path: str | Path) -> Iterator[GTFRecord]:
        """Iterate over all non-comment GTF records without building the hierarchy."""
        yield from self._iter_records(Path(path))

    # ── Parsing helpers ────────────────────────────────────────────────────────

    def _iter_records(self, path: Path) -> Iterator[GTFRecord]:
        with open(path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.rstrip("\n\r")
                if not line or line.startswith("#"):
                    continue
                try:
                    yield _parse_line(line)
                except (ValueError, IndexError) as exc:
                    if self.strict:
                        raise ValueError(
                            f"{path}:{lineno}: {exc}\n  Line: {line!r}"
                        ) from exc
                    import warnings
                    warnings.warn(f"{path}:{lineno}: skipping malformed line: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

_ATTR_RE = re.compile(r'(\w+)\s+"([^"]*)"')


def _parse_attributes(attr_str: str) -> Dict[str, str]:
    """
    Parse GTF attribute string  key "value"; key "value"; ...

    Returns an ordered dict of key -> value.
    Duplicate keys: last value wins.
    """
    return {m.group(1): m.group(2) for m in _ATTR_RE.finditer(attr_str)}


def _parse_line(line: str) -> GTFRecord:
    """Parse one tab-separated GTF data line into a GTFRecord."""
    parts = line.split("\t", 8)
    if len(parts) < 8:
        raise ValueError(f"Expected ≥8 tab-separated fields, got {len(parts)}")

    seqname = parts[0]
    source  = parts[1]
    feature = parts[2]
    start   = int(parts[3])
    end     = int(parts[4])
    score   = None if parts[5] == "." else float(parts[5])
    strand  = parts[6]
    frame   = None if parts[7] == "." else int(parts[7])
    attrs   = _parse_attributes(parts[8]) if len(parts) == 9 else {}

    return GTFRecord(
        seqname=seqname,
        source=source,
        feature=feature,
        start=start,
        end=end,
        score=score,
        strand=strand,
        frame=frame,
        attributes=attrs,
    )


def _parse_exon_number(attrs: Dict[str, str]) -> Optional[int]:
    v = attrs.get("exon_number") or attrs.get("exon_id")
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Object constructors
# ─────────────────────────────────────────────────────────────────────────────

def _make_gene(r: GTFRecord) -> Gene:
    return Gene(
        gene_id=r.attr("gene_id"),
        seqname=r.seqname,
        source=r.source,
        start=r.start,
        end=r.end,
        strand=r.strand,
        gene_name=r.attr("gene_name"),
        biotype=r.attr("gene_biotype"),
        attributes=dict(r.attributes),
    )


def _make_gene_stub(r: GTFRecord) -> Gene:
    """Create a minimal Gene from a non-gene record (best-effort coordinates)."""
    return Gene(
        gene_id=r.attr("gene_id"),
        seqname=r.seqname,
        source=r.source,
        start=r.start,
        end=r.end,
        strand=r.strand,
        gene_name=r.attr("gene_name"),
        biotype=r.attr("gene_biotype"),
        attributes={"gene_id": r.attr("gene_id")},
    )


def _make_transcript(r: GTFRecord) -> Transcript:
    return Transcript(
        transcript_id=r.attr("transcript_id"),
        gene_id=r.attr("gene_id"),
        seqname=r.seqname,
        source=r.source,
        start=r.start,
        end=r.end,
        strand=r.strand,
        biotype=r.attr("transcript_biotype"),
        attributes=dict(r.attributes),
    )


def _make_transcript_stub(r: GTFRecord) -> Transcript:
    return Transcript(
        transcript_id=r.attr("transcript_id"),
        gene_id=r.attr("gene_id"),
        seqname=r.seqname,
        source=r.source,
        start=r.start,
        end=r.end,
        strand=r.strand,
        biotype="",
        attributes={"transcript_id": r.attr("transcript_id")},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sequence utilities
# ─────────────────────────────────────────────────────────────────────────────

_COMPLEMENT: Dict[str, str] = str.maketrans("ACGTacgt", "TGCAtgca")


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]
