"""
OpenAI provider adapter.
"""

from __future__ import annotations

import time

from trizaval.harness.providers.base import Provider, ProviderError, ProviderResponse

try:
    import openai
except ImportError as e:
    raise ImportError(
        "the 'openai' package is required to use OpenAIProvider; install it with "
        "`pip install openai`"
    ) from e


class OpenAIProvider(Provider):
    """Calls the OpenAI Chat Completions API for a single model."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        self.model = model
        # `api_key=None` lets the underlying SDK fall back to the
        # OPENAI_API_KEY environment variable, which is the standard
        # convention and avoids forcing callers to plumb secrets
        # through suite config files. Subclasses (e.g.
        # OpenAICompatibleProvider) resolve their own env var name
        # before calling this constructor.
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, *, temperature: float, max_tokens: int | None) -> ProviderResponse:
        start = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except openai.APIError as e:
            raise ProviderError(f"OpenAI API error for model '{self.model}': {e}") from e
        except Exception as e:
            # Network errors, timeouts, etc. from the underlying HTTP
            # client are not always `openai.APIError` subclasses, so
            # this catch-all ensures the harness never sees a raw,
            # unrelated exception type it can't handle uniformly.
            raise ProviderError(f"unexpected error calling OpenAI model '{self.model}': {e}") from e

        latency = time.monotonic() - start

        choice = response.choices[0]
        text = choice.message.content or ""

        return ProviderResponse(
            text=text,
            latency_seconds=latency,
            raw_response=response.model_dump(),
        )