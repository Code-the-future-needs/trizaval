"""
Tests for trizaval.harness.providers -- provider adapters and the
build_provider factory.
"""

from unittest.mock import MagicMock, patch

import pytest

from trizaval.harness.providers import build_provider
from trizaval.harness.providers.anthropic_provider import AnthropicProvider
from trizaval.harness.providers.base import ProviderError, ProviderResponse
from trizaval.harness.providers.openai_compatible_provider import OpenAICompatibleProvider
from trizaval.harness.providers.openai_provider import OpenAIProvider
from trizaval.suite.schema import ProviderConfig, ProviderKind


class TestBuildProviderFactory:
    def test_openai_kind_builds_openai_provider(self):
        cfg = ProviderConfig(name="p", kind=ProviderKind.OPENAI, model="gpt-4o-mini")
        provider = build_provider(cfg)
        assert isinstance(provider, OpenAIProvider)

    def test_anthropic_kind_builds_anthropic_provider(self):
        cfg = ProviderConfig(name="p", kind=ProviderKind.ANTHROPIC, model="claude-sonnet-4-6")
        provider = build_provider(cfg)
        assert isinstance(provider, AnthropicProvider)

    @pytest.mark.parametrize(
        "model,base_url",
        [
            ("deepseek-chat", "https://api.deepseek.com/v1"),
            ("grok-4", "https://api.x.ai/v1"),
            ("kimi-k2", "https://api.moonshot.cn/v1"),
            ("llama3", "http://localhost:11434/v1"),
        ],
    )
    def test_openai_compatible_kind_builds_generic_provider_for_any_company(self, model, base_url):
        """This is the test that matters most for broad provider
        coverage: DeepSeek, Grok, Kimi, and a local model all build
        successfully through the SAME class, with zero per-company
        code -- only config differs.
        """
        cfg = ProviderConfig(
            name="p", kind=ProviderKind.OPENAI_COMPATIBLE, model=model, base_url=base_url
        )
        provider = build_provider(cfg)
        assert isinstance(provider, OpenAICompatibleProvider)
        assert isinstance(provider, OpenAIProvider)  # reuses the OpenAI SDK client

    def test_unknown_provider_kind_raises(self):
        cfg = ProviderConfig(name="p", kind=ProviderKind.OPENAI, model="gpt-4o-mini")
        cfg.kind = "not_a_real_kind"  # bypass enum validation to test the factory's fallback branch
        with pytest.raises(ValueError, match="unknown provider kind"):
            build_provider(cfg)


class TestOpenAIProvider:
    def test_generate_returns_provider_response(self):
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-fake")
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=MagicMock(content="4"))]
        fake_response.model_dump.return_value = {"id": "fake"}

        with patch.object(provider._client.chat.completions, "create", return_value=fake_response):
            result = provider.generate("What is 2+2?", temperature=0.0, max_tokens=10)

        assert isinstance(result, ProviderResponse)
        assert result.text == "4"
        assert result.latency_seconds >= 0

    def test_generate_wraps_unexpected_exceptions_as_provider_error(self):
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-fake")
        with patch.object(
            provider._client.chat.completions, "create", side_effect=RuntimeError("connection refused")
        ):
            with pytest.raises(ProviderError, match="connection refused"):
                provider.generate("test", temperature=0.0, max_tokens=10)

    def test_none_content_becomes_empty_string(self):
        """Some API responses can have a None content field (e.g. a
        pure tool-call response); this should not crash."""
        provider = OpenAIProvider(model="gpt-4o-mini", api_key="sk-fake")
        fake_response = MagicMock()
        fake_response.choices = [MagicMock(message=MagicMock(content=None))]
        fake_response.model_dump.return_value = {"id": "fake"}

        with patch.object(provider._client.chat.completions, "create", return_value=fake_response):
            result = provider.generate("test", temperature=0.0, max_tokens=10)

        assert result.text == ""


class TestAnthropicProvider:
    def test_generate_returns_provider_response(self):
        provider = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-ant-fake")
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="4")]
        fake_response.model_dump.return_value = {"id": "fake"}

        with patch.object(provider._client.messages, "create", return_value=fake_response):
            result = provider.generate("What is 2+2?", temperature=0.0, max_tokens=10)

        assert result.text == "4"

    def test_concatenates_multiple_content_blocks(self):
        provider = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-ant-fake")
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="Hello, "), MagicMock(text="world!")]
        fake_response.model_dump.return_value = {"id": "fake"}

        with patch.object(provider._client.messages, "create", return_value=fake_response):
            result = provider.generate("hi", temperature=0.0, max_tokens=None)

        assert result.text == "Hello, world!"

    def test_missing_max_tokens_uses_default(self):
        """Anthropic's API requires max_tokens on every request, unlike
        OpenAI where it's optional -- confirm our default kicks in."""
        provider = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-ant-fake")
        fake_response = MagicMock()
        fake_response.content = [MagicMock(text="ok")]
        fake_response.model_dump.return_value = {"id": "fake"}

        with patch.object(provider._client.messages, "create", return_value=fake_response) as mock_create:
            provider.generate("test", temperature=0.0, max_tokens=None)

        assert mock_create.call_args.kwargs["max_tokens"] == AnthropicProvider._DEFAULT_MAX_TOKENS

    def test_generate_wraps_unexpected_exceptions_as_provider_error(self):
        provider = AnthropicProvider(model="claude-sonnet-4-6", api_key="sk-ant-fake")
        with patch.object(
            provider._client.messages, "create", side_effect=RuntimeError("timeout")
        ):
            with pytest.raises(ProviderError, match="timeout"):
                provider.generate("test", temperature=0.0, max_tokens=10)


class TestOpenAICompatibleProvider:
    def test_requires_base_url(self):
        with pytest.raises(ValueError, match="base_url"):
            OpenAICompatibleProvider(model="deepseek-chat", base_url="")

    def test_works_without_api_key_env_var_for_local_servers(self):
        # Should not raise -- local servers typically don't need a key.
        provider = OpenAICompatibleProvider(model="llama3", base_url="http://localhost:11434/v1")
        assert provider.model == "llama3"

    def test_resolves_api_key_from_named_env_var(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-real")
        provider = OpenAICompatibleProvider(
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key_env_var="DEEPSEEK_API_KEY",
        )
        # The underlying SDK stores the resolved key; confirm it picked
        # up our env var rather than falling back to the placeholder.
        assert provider._client.api_key == "sk-deepseek-real"