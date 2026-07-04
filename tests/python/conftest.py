"""
Shared pytest fixtures for the trizaval test suite.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def fake_api_keys(monkeypatch):
    """Ensures every test has fake credentials available, so provider
    construction never fails on missing environment variables during
    tests. Applied automatically to every test in this suite.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-testing")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-for-testing")


@pytest.fixture
def example_suite_path():
    """Path to the real example suite YAML, relative to repo root."""
    return "suites/example_suite.yaml"