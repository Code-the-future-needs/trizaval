"""
Abstract interface every trizaval provider adapter implements.

A "provider" is anything that can take a prompt and return a text
response - an OpenAI model, an Anthropic model, or a locally-hosted
model via Ollama/vLLM. The harness only depends on this interface, so
adding a new provider kind never requires touching the runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResponse:
    """Result of a single call to a provider."""

    text: str
    latency_seconds: float
    raw_response: dict
    """Full raw API response, kept for debugging/auditing; not
    assumed to have any particular shape by the rest of the harness."""


class ProviderError(Exception):
    """Raised when a provider call fails (network error, API error,
    rate limit, etc.) after any built-in retry policy is exhausted."""


class Provider(ABC):
    """Abstract base class for all model provider adapters."""

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float, max_tokens: int | None) -> ProviderResponse:
        """Sends `prompt` to the underlying model and returns its
        response. Raises `ProviderError` on failure."""
        raise NotImplementedError