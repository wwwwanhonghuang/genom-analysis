"""
gecsg.profiler.profiler
=======================
Performance profiler for GE-CSG parsers.

Records (sequence_length, n_codons, elapsed_ms) for every parse call and
appends rows to a CSV file so that data survives across multiple runs.

CSV schema
----------
  run_id      : UUID for this profiler session
  label       : user-supplied tag (e.g. "benchmark_uniform")
  grammar     : grammar class / type name
  seq_len     : raw sequence length in characters
  n_codons    : number of codons (seq_len / 3)
  accepted    : bool
  elapsed_ms  : wall-clock parse time in milliseconds (perf_counter)
  timestamp   : ISO-8601 UTC timestamp of the call

Usage
-----
    from gecsg.profiler import ParseProfiler

    profiler = ParseProfiler(parser, csv_path="profile_data.csv")

    # drop-in replacement for parser.parse()
    result = profiler.parse("ATGGCTTAA")

    # batch benchmark
    profiler.benchmark(list_of_seqs, label="my_run")

    # print summary
    profiler.summary()
"""

from __future__ import annotations

import csv
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# CSV column order (fixed so files are always append-compatible)
_FIELDNAMES = [
    "run_id", "label", "grammar",
    "seq_len", "n_codons", "accepted",
    "elapsed_ms", "timestamp",
]


class ParseProfiler:
    """
    Thin wrapper around any parser that exposes a `.parse(seq)` method.

    Every call to `profiler.parse()` is timed and a row is written to the
    CSV file.  Multiple profiler instances (or runs) append to the same file,
    so the dataset grows automatically over time.

    Parameters
    ----------
    parser      : object with a `.parse(raw_seq: str)` method
    csv_path    : path to the output CSV file (created / appended to)
    label       : default label tag for rows (can be overridden per call)
    grammar_tag : short description of the grammar/parser type.
                  If None, uses type(parser).__name__.
    """

    def __init__(
        self,
        parser,
        csv_path:    str | Path = "outputs/performance_profile/profile_data.csv",
        label:       str = "default",
        grammar_tag: Optional[str] = None,
    ):
        self._parser      = parser
        self._csv_path    = Path(csv_path)
        self._label       = label
        self._grammar_tag = grammar_tag or type(parser).__name__
        self._run_id      = str(uuid.uuid4())[:8]   # short 8-char session ID

        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: List[Dict] = []   # in-memory buffer for this session
        self._ensure_csv_header()

    # ── Public API ────────────────────────────────────────────────────────

    def parse(self, raw_seq: str, label: Optional[str] = None):
        """
        Parse raw_seq, record timing, return the original ParseResult.

        Parameters
        ----------
        raw_seq : DNA string
        label   : override the session label for this single row
        """
        t0     = time.perf_counter()
        result = self._parser.parse(raw_seq)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        n_codons = getattr(result, "n_codons", len(raw_seq.strip()) // 3)
        accepted  = bool(getattr(result, "accepted", False))

        row = {
            "run_id":     self._run_id,
            "label":      label or self._label,
            "grammar":    self._grammar_tag,
            "seq_len":    len(raw_seq.strip()),
            "n_codons":   n_codons,
            "accepted":   accepted,
            "elapsed_ms": round(elapsed_ms, 4),
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }
        self._rows.append(row)
        self._append_row(row)
        return result

    def benchmark(
        self,
        sequences:    List[str],
        label:        Optional[str] = None,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Parse every sequence in `sequences`, record all timings.

        Returns the list of newly added rows.
        """
        tag    = label or self._label
        added  = []
        total  = len(sequences)
        for i, seq in enumerate(sequences):
            if show_progress and (i % max(1, total // 10) == 0):
                print(f"  [{i+1}/{total}] seq_len={len(seq)}")
            result = self.parse(seq, label=tag)
            added.append(self._rows[-1])
        return added

    def summary(self) -> None:
        """Print a statistical summary of all rows recorded this session."""
        if not self._rows:
            print("No data recorded yet.")
            return

        elapsed = [r["elapsed_ms"] for r in self._rows]
        codons  = [r["n_codons"]   for r in self._rows]
        n       = len(self._rows)
        accepted = sum(1 for r in self._rows if r["accepted"])

        print("-" * 52)
        print(f"ParseProfiler session  run_id={self._run_id}")
        print(f"  Grammar    : {self._grammar_tag}")
        print(f"  Rows       : {n}  ({accepted} accepted)")
        print(f"  n_codons   : min={min(codons)}  max={max(codons)}  "
              f"mean={sum(codons)/n:.1f}")
        print(f"  elapsed_ms : min={min(elapsed):.3f}  "
              f"max={max(elapsed):.3f}  "
              f"mean={sum(elapsed)/n:.3f}")
        print(f"  CSV file   : {self._csv_path}")
        print("-" * 52)

    @property
    def rows(self) -> List[Dict]:
        """All rows recorded during this session (in-memory copy)."""
        return list(self._rows)

    @property
    def csv_path(self) -> Path:
        return self._csv_path

    # ── CSV helpers ───────────────────────────────────────────────────────

    def _ensure_csv_header(self) -> None:
        """Write CSV header if the file is new/empty."""
        if not self._csv_path.exists() or self._csv_path.stat().st_size == 0:
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                writer.writeheader()

    def _append_row(self, row: Dict) -> None:
        """Append a single row to the CSV file."""
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writerow(row)


# ── Convenience: load CSV for analysis ────────────────────────────────────────

def load_profile_data(csv_path: str | Path) -> List[Dict]:
    """
    Load all rows from a profile CSV file.

    Returns a list of dicts (same schema as _FIELDNAMES).
    Numeric columns (seq_len, n_codons, elapsed_ms) are cast automatically.
    """
    path = Path(csv_path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["seq_len"]    = int(row["seq_len"])
            row["n_codons"]   = int(row["n_codons"])
            row["elapsed_ms"] = float(row["elapsed_ms"])
            row["accepted"]   = row["accepted"] == "True"
            rows.append(row)
    return rows
