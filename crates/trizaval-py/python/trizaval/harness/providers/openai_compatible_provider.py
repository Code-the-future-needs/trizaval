"""
Generic provider adapter for any model served via an OpenAI-compatible
chat completions API. This single class is what actually gives
trizaval broad provider coverage: DeepSeek, xAI/Grok, Moonshot/Kimi,
Mistral, Groq, Together, Google Gemini's OpenAI-compat endpoint, and
locally-hosted models via Ollama/vLLM all speak this same protocol, so
none of them need a bespoke SDK integration -- just a base_url, a
model name, and (usually) an API key.

Adding support for a new OpenAI-compatible provider is a suite config
change, not a code change:

    kind: openai_compatible
    model: deepseek-chat
    base_url: https://api.deepseek.com/v1
    api_key_env_var: DEEPSEEK_API_KEY
"""

from __future__ import annotations

import os

from trizaval.harness.providers.openai_provider import OpenAIProvider


class OpenAICompatibleProvider(OpenAIProvider):
    """Calls any OpenAI-compatible endpoint. `base_url` is required --
    there is no sensible default across arbitrary providers."""

    def __init__(self, model: str, base_url: str, api_key_env_var: str | None = None):
        if not base_url:
            raise ValueError(
                "OpenAICompatibleProvider requires base_url "
                "(e.g. 'https://api.deepseek.com/v1', 'http://localhost:11434/v1')"
            )

        api_key = os.environ.get(api_key_env_var) if api_key_env_var else None
        # Local servers generally don't check the API key, but the
        # OpenAI SDK requires a non-empty string to be passed
        super().__init__(model=model, api_key=api_key or "not-needed", base_url=base_url)