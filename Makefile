# =============================================================================
# Genome Analysis — top-level Makefile
# =============================================================================

PYTHON   ?= python3
LLM_DIR  := ./llm
VENV     := .venv
VENV_PY  := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

CYAN   := \033[0;36m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m

include Makefile.llm

.DEFAULT_GOAL := help

# =============================================================================
# Virtualenv + deps
# =============================================================================

.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	@echo "$(GREEN)✓ Virtualenv ready$(NC)"

.PHONY: deps-cpu
deps-cpu: venv
	$(VENV_PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(VENV_PIP) install transformers accelerate einops

.PHONY: deps-gpu
deps-gpu: venv
	$(VENV_PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	$(VENV_PIP) install transformers accelerate einops bitsandbytes

.PHONY: deps-serve
deps-serve: venv
	$(VENV_PIP) install fastapi uvicorn[standard] pydantic --quiet

.PHONY: deps-onnx
deps-onnx: venv
	$(VENV_PIP) install optimum[onnxruntime] onnx onnxruntime --quiet

# =============================================================================
# Cleanup
# =============================================================================

.PHONY: clean
clean:
	rm -rf ~/.cache/huggingface/hub/models--LongSafari*
	rm -rf ~/.cache/huggingface/hub/models--PoetschLab*
	rm -rf ~/.cache/huggingface/hub/models--zhihan1996*
	rm -rf ~/.cache/huggingface/hub/models--songlab-pl*
	rm -rf ~/.cache/huggingface/hub/models--InstaDeepAI*
	rm -rf ~/.cache/huggingface/hub/models--togethercomputer*
	@echo "$(GREEN)✓ Cache cleared$(NC)"

.PHONY: clean-venv
clean-venv:
	rm -rf $(VENV)