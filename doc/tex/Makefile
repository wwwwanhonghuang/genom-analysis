# ============================================================
#  Makefile — Genomic Syntactic Analysis
#  Usage:
#    make          → build PDF (two pdflatex passes)
#    make once     → single pass (fast draft)
#    make clean    → remove auxiliary files
#    make distclean→ remove aux files AND the PDF
#    make view     → build then open PDF
# ============================================================

# ── Configuration ───────────────────────────────────────────
MAIN     := genome_syntax
TEX      := $(MAIN).tex
PDF      := $(MAIN).pdf
LATEX    := pdflatex
FLAGS    := -interaction=nonstopmode -halt-on-error

# Auxiliary file extensions produced by pdflatex / hyperref / etc.
AUX_EXTS := aux log toc lof lot out bbl blg fls fdb_latexmk synctex.gz \
            bcf run.xml idx ilg ind

AUX_FILES := $(foreach ext,$(AUX_EXTS),$(MAIN).$(ext))

# PDF viewer (tries common options; override with  make view VIEWER=okular)
VIEWER   ?= $(shell \
	command -v evince  2>/dev/null || \
	command -v okular  2>/dev/null || \
	command -v zathura 2>/dev/null || \
	command -v open    2>/dev/null || \
	echo xdg-open)

# ── Default target ──────────────────────────────────────────
.PHONY: all
all: $(PDF)

# Two-pass build so TOC, cross-references and hyperlinks are correct
$(PDF): $(TEX)
	@echo "==> Pass 1/2: $(LATEX) $(MAIN)"
	$(LATEX) $(FLAGS) $(TEX)
	@echo "==> Pass 2/2: $(LATEX) $(MAIN)"
	$(LATEX) $(FLAGS) $(TEX)
	@echo "==> Built:    $(PDF)"

# ── Single-pass draft ───────────────────────────────────────
.PHONY: once
once: $(TEX)
	@echo "==> Draft pass: $(LATEX) $(MAIN)"
	$(LATEX) $(FLAGS) $(TEX)

# ── View ────────────────────────────────────────────────────
.PHONY: view
view: $(PDF)
	@echo "==> Opening $(PDF) with $(VIEWER)"
	$(VIEWER) $(PDF) &

# ── Clean auxiliary files ───────────────────────────────────
.PHONY: clean
clean:
	@echo "==> Removing auxiliary files"
	@rm -f $(AUX_FILES)

# ── Full clean (aux + PDF) ──────────────────────────────────
.PHONY: distclean
distclean: clean
	@echo "==> Removing $(PDF)"
	@rm -f $(PDF)