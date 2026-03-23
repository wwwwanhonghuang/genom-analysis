"""gecsg.io — file I/O utilities for genomic annotation formats."""
from gecsg.io.gtf import GTFRecord, Exon, CDSSegment, Transcript, Gene, GTFReader

__all__ = [
    "GTFRecord",
    "Exon",
    "CDSSegment",
    "Transcript",
    "Gene",
    "GTFReader",
]
