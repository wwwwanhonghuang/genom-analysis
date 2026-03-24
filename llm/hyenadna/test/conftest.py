"""
llm/hyenadna/test/conftest.py
pytest configuration for HyenaDNA tests.

Markers:
  slow    — tests that run sequences >32k bp; skip unless -m slow is passed.

Usage:
  pytest llm/hyenadna/test/           # all fast tests
  pytest llm/hyenadna/test/ -m slow   # include long-context tests
  pytest llm/hyenadna/test/ -v        # verbose output
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: long-context tests (>32k bp) — run with -m slow"
    )