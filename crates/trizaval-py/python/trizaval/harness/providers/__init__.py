"""
Provider adapters: turn a `ProviderConfig` (from suite schema) into a
live `Provider` instance.
"""

from __future__ import annotations

import os

from trizaval.harness.providers.base import Provider, ProviderError, ProviderResponse
from trizaval.suite.schema import ProviderConfig, ProviderKind


def build_provider(config: ProviderConfig) -> Provider:
    """Constructs the correct `Provider` implementation for `config`.

    Only OpenAI and Anthropic get bespoke SDK integrations, since
    their APIs have genuinely different shapes. Every other provider
    -- DeepSeek, xAI/Grok, Moonshot/Kimi, Mistral, Groq, Together,
    Google Gemini's OpenAI-compat endpoint, and locally-hosted models
    -- is handled by the single generic OpenAICompatibleProvider via
    kind='openai_compatible', since they all speak the same protocol.
    """
    if config.kind == ProviderKind.OPENAI:
        from trizaval.harness.providers.openai_provider import OpenAIProvider

        api_key = os.environ.get(config.api_key_env_var) if config.api_key_env_var else None
        return OpenAIProvider(model=config.model, api_key=api_key, base_url=config.base_url)

    if config.kind == ProviderKind.ANTHROPIC:
        from trizaval.harness.providers.anthropic_provider import AnthropicProvider

        api_key = os.environ.get(config.api_key_env_var) if config.api_key_env_var else None
        return AnthropicProvider(model=config.model, api_key=api_key, base_url=config.base_url)

    if config.kind == ProviderKind.OPENAI_COMPATIBLE:
        from trizaval.harness.providers.openai_compatible_provider import OpenAICompatibleProvider

        assert config.base_url is not None
        return OpenAICompatibleProvider(
            model=config.model,
            base_url=config.base_url,
            api_key_env_var=config.api_key_env_var,
        )

    raise ValueError(f"unknown provider kind: {config.kind}")


__all__ = ["Provider", "ProviderError", "ProviderResponse", "build_provider"]