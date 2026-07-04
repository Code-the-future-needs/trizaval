"""
Anthropic provider adapter.
"""

from __future__ import annotations

import time

from trizaval.harness.providers.base import Provider, ProviderError, ProviderResponse

try:
    import anthropic
except ImportError as e:
    raise ImportError(
        "the 'anthropic' package is required to use AnthropicProvider; install it with "
        "`pip install anthropic`"
    ) from e


class AnthropicProvider(Provider):
    """Calls the Anthropic Messages API for a single model."""

    # Anthropic's API requires max_tokens explicitly on every request
    # (unlike OpenAI, where it's optional) -- this default only
    # applies when the suite config didn't specify one.
    _DEFAULT_MAX_TOKENS = 1024

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        self.model = model
        # `api_key=None` lets the underlying SDK fall back to the
        # ANTHROPIC_API_KEY environment variable by default.
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, *, temperature: float, max_tokens: int | None) -> ProviderResponse:
        start = time.monotonic()
        try:
            response = self._client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens or self._DEFAULT_MAX_TOKENS,
            )
        except anthropic.APIError as e:
            raise ProviderError(f"Anthropic API error for model '{self.model}': {e}") from e
        except Exception as e:
            raise ProviderError(f"unexpected error calling Anthropic model '{self.model}': {e}") from e

        latency = time.monotonic() - start

        # Anthropic responses can contain multiple content blocks;
        # concatenate any text blocks to form the full response text
        # rather than assuming exactly one block.
        text = "".join(block.text for block in response.content if hasattr(block, "text"))

        return ProviderResponse(
            text=text,
            latency_seconds=latency,
            raw_response=response.model_dump(),
        )