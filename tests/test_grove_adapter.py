"""
Tests for the Grove API gateway LLM adapter.

The adapter imports ``langchain_openai.ChatOpenAI`` lazily inside ``get_llm()``.
These tests inject a fake ``langchain_openai`` module into ``sys.modules`` so
they run without the real (heavy) dependency installed and let us assert
exactly what kwargs the adapter passes to ``ChatOpenAI``.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from agent_builder.llms.adapters import GroveLLMAdapter, LLMAdapterFactory
from agent_builder.llms.loader import LLMConfig


@pytest.fixture
def fake_chat_openai(monkeypatch):
    """Install a fake ``langchain_openai`` module exposing a ChatOpenAI mock.

    Returns the ChatOpenAI mock so tests can inspect the call kwargs.
    """
    fake_module = types.ModuleType("langchain_openai")
    chat_openai = MagicMock(name="ChatOpenAI")
    chat_openai.return_value = MagicMock(name="chat_openai_instance")
    fake_module.ChatOpenAI = chat_openai
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)
    # Ensure Grove env vars never leak in from the host environment.
    for var in ("GROVE_API_BASE", "GROVE_API_GATEWAY_URL", "GROVE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return chat_openai


# ---------------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------------

class TestGroveRegistration:
    def test_grove_is_registered(self):
        assert "grove" in LLMAdapterFactory.available_providers()

    def test_factory_returns_grove_adapter(self):
        config = LLMConfig(name="g", provider="grove", model_name="claude-3-5-sonnet")
        adapter = LLMAdapterFactory.create(config)
        assert isinstance(adapter, GroveLLMAdapter)

    def test_provider_lookup_is_case_insensitive(self):
        config = LLMConfig(name="g", provider="GROVE", model_name="claude-3-5-sonnet")
        adapter = LLMAdapterFactory.create(config)
        assert isinstance(adapter, GroveLLMAdapter)


# ---------------------------------------------------------------------------
# base_url / api_key resolution
# ---------------------------------------------------------------------------

class TestGroveResolution:
    def test_base_url_from_additional_kwargs(self, fake_chat_openai):
        config = LLMConfig(
            name="g",
            provider="grove",
            model_name="claude-3-5-sonnet",
            additional_kwargs={
                "base_url": "https://grove.internal/v1",
                "api_key": "sk-grove-123",
            },
        )
        GroveLLMAdapter(config).get_llm()

        _, kwargs = fake_chat_openai.call_args
        assert kwargs["base_url"] == "https://grove.internal/v1"
        assert kwargs["api_key"] == "sk-grove-123"
        assert kwargs["model"] == "claude-3-5-sonnet"

    def test_base_url_from_env(self, fake_chat_openai, monkeypatch):
        monkeypatch.setenv("GROVE_API_BASE", "https://grove.env/v1")
        monkeypatch.setenv("GROVE_API_KEY", "sk-env-key")
        config = LLMConfig(name="g", provider="grove", model_name="gpt-4o")
        GroveLLMAdapter(config).get_llm()

        _, kwargs = fake_chat_openai.call_args
        assert kwargs["base_url"] == "https://grove.env/v1"
        assert kwargs["api_key"] == "sk-env-key"

    def test_additional_kwargs_take_precedence_over_env(self, fake_chat_openai, monkeypatch):
        monkeypatch.setenv("GROVE_API_BASE", "https://grove.env/v1")
        config = LLMConfig(
            name="g",
            provider="grove",
            model_name="gpt-4o",
            additional_kwargs={"base_url": "https://grove.explicit/v1"},
        )
        GroveLLMAdapter(config).get_llm()

        _, kwargs = fake_chat_openai.call_args
        assert kwargs["base_url"] == "https://grove.explicit/v1"

    def test_gateway_url_env_fallback(self, fake_chat_openai, monkeypatch):
        # Only the alternate env var is set.
        monkeypatch.setenv("GROVE_API_GATEWAY_URL", "https://grove.alt/v1")
        config = LLMConfig(name="g", provider="grove", model_name="gpt-4o")
        GroveLLMAdapter(config).get_llm()

        _, kwargs = fake_chat_openai.call_args
        assert kwargs["base_url"] == "https://grove.alt/v1"

    def test_missing_base_url_raises(self, fake_chat_openai):
        config = LLMConfig(name="g", provider="grove", model_name="gpt-4o")
        with pytest.raises(ValueError, match="base URL is required"):
            GroveLLMAdapter(config).get_llm()

    def test_placeholder_api_key_when_none_provided(self, fake_chat_openai):
        config = LLMConfig(
            name="g",
            provider="grove",
            model_name="gpt-4o",
            additional_kwargs={"base_url": "https://grove.internal/v1"},
        )
        GroveLLMAdapter(config).get_llm()

        _, kwargs = fake_chat_openai.call_args
        assert kwargs["api_key"] == GroveLLMAdapter._PLACEHOLDER_API_KEY


# ---------------------------------------------------------------------------
# Passthrough behaviour
# ---------------------------------------------------------------------------

class TestGrovePassthrough:
    def test_default_headers_pass_through(self, fake_chat_openai):
        config = LLMConfig(
            name="g",
            provider="grove",
            model_name="gpt-4o",
            additional_kwargs={
                "base_url": "https://grove.internal/v1",
                "api_key": "k",
                "default_headers": {"x-tenant-id": "acme"},
            },
        )
        GroveLLMAdapter(config).get_llm()

        _, kwargs = fake_chat_openai.call_args
        assert kwargs["default_headers"] == {"x-tenant-id": "acme"}
        # base_url/api_key must be consumed, not leaked as a nested key.
        assert "base_url" in kwargs and "api_key" in kwargs

    def test_core_params_forwarded(self, fake_chat_openai):
        config = LLMConfig(
            name="g",
            provider="grove",
            model_name="gpt-4o",
            temperature=0.2,
            max_tokens=1500,
            streaming=True,
            additional_kwargs={"base_url": "https://grove.internal/v1"},
        )
        GroveLLMAdapter(config).get_llm()

        _, kwargs = fake_chat_openai.call_args
        assert kwargs["temperature"] == 0.2
        assert kwargs["max_tokens"] == 1500
        assert kwargs["streaming"] is True

    def test_config_additional_kwargs_not_mutated(self, fake_chat_openai):
        """get_llm() must not pop keys out of the caller's config dict."""
        additional = {
            "base_url": "https://grove.internal/v1",
            "api_key": "k",
        }
        config = LLMConfig(
            name="g", provider="grove", model_name="gpt-4o",
            additional_kwargs=additional,
        )
        GroveLLMAdapter(config).get_llm()

        # Original dict still intact for any later reuse.
        assert additional == {
            "base_url": "https://grove.internal/v1",
            "api_key": "k",
        }
