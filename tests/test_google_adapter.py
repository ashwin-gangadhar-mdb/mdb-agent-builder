"""
Tests for the Google Gemini LLM and embedding adapters.

Both adapters import their ``langchain_google_genai`` classes lazily inside
``get_llm()`` / ``get_embedding_model()``.  These tests inject a fake
``langchain_google_genai`` module into ``sys.modules`` so they run without the
real dependency installed and let us assert exactly which kwargs the adapters
forward to the underlying clients.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from agent_builder.embeddings.adapters import (
    EmbeddingAdapterFactory,
    GoogleGenAIEmbeddingAdapter,
)
from agent_builder.embeddings.loader import EmbeddingConfig
from agent_builder.llms.adapters import GoogleGenAILLMAdapter, LLMAdapterFactory
from agent_builder.llms.loader import LLMConfig


@pytest.fixture
def fake_google_genai(monkeypatch):
    """Install a fake ``langchain_google_genai`` module.

    Returns the (ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings) mocks so
    tests can inspect the call kwargs.
    """
    fake_module = types.ModuleType("langchain_google_genai")
    chat = MagicMock(name="ChatGoogleGenerativeAI")
    chat.return_value = MagicMock(name="chat_instance")
    embed = MagicMock(name="GoogleGenerativeAIEmbeddings")
    embed.return_value = MagicMock(name="embed_instance")
    fake_module.ChatGoogleGenerativeAI = chat
    fake_module.GoogleGenerativeAIEmbeddings = embed
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)
    # Ensure host-environment keys never leak into the assertions.
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return chat, embed


# ---------------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------------

class TestGoogleRegistration:
    @pytest.mark.parametrize("provider", ["google", "gemini", "GOOGLE", "Gemini"])
    def test_llm_provider_resolves_to_gemini_adapter(self, provider):
        config = LLMConfig(name="g", provider=provider, model_name="gemini-1.5-pro")
        adapter = LLMAdapterFactory.create(config)
        assert isinstance(adapter, GoogleGenAILLMAdapter)

    @pytest.mark.parametrize("provider", ["google", "gemini", "GOOGLE", "Gemini"])
    def test_embedding_provider_resolves_to_gemini_adapter(self, provider):
        config = EmbeddingConfig(
            name="g", provider=provider, model_name="models/text-embedding-004"
        )
        adapter = EmbeddingAdapterFactory.create(config)
        assert isinstance(adapter, GoogleGenAIEmbeddingAdapter)

    def test_providers_are_listed(self):
        assert {"google", "gemini"} <= set(LLMAdapterFactory.available_providers())
        assert {"google", "gemini"} <= set(
            EmbeddingAdapterFactory.available_providers()
        )


# ---------------------------------------------------------------------------
# LLM adapter behaviour
# ---------------------------------------------------------------------------

class TestGeminiLLM:
    def test_max_tokens_is_mapped_to_max_output_tokens(self, fake_google_genai):
        chat, _ = fake_google_genai
        config = LLMConfig(
            name="g",
            provider="google",
            model_name="gemini-1.5-pro",
            temperature=0.3,
            max_tokens=2048,
            streaming=True,
            additional_kwargs={"api_key": "key-123"},
        )
        GoogleGenAILLMAdapter(config).get_llm()

        _, kwargs = chat.call_args
        assert kwargs["model"] == "gemini-1.5-pro"
        assert kwargs["temperature"] == 0.3
        assert kwargs["google_api_key"] == "key-123"
        assert kwargs["max_output_tokens"] == 2048
        # ChatGoogleGenerativeAI has no ``max_tokens`` / ``streaming`` ctor args.
        assert "max_tokens" not in kwargs
        assert "streaming" not in kwargs

    def test_api_key_from_env(self, fake_google_genai, monkeypatch):
        chat, _ = fake_google_genai
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        config = LLMConfig(name="g", provider="google", model_name="gemini-1.5-flash")
        GoogleGenAILLMAdapter(config).get_llm()

        _, kwargs = chat.call_args
        assert kwargs["google_api_key"] == "env-key"

    def test_gemini_api_key_env_fallback(self, fake_google_genai, monkeypatch):
        chat, _ = fake_google_genai
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-env-key")
        config = LLMConfig(name="g", provider="gemini", model_name="gemini-1.5-flash")
        GoogleGenAILLMAdapter(config).get_llm()

        _, kwargs = chat.call_args
        assert kwargs["google_api_key"] == "gemini-env-key"

    def test_additional_kwargs_pass_through(self, fake_google_genai):
        chat, _ = fake_google_genai
        config = LLMConfig(
            name="g",
            provider="google",
            model_name="gemini-1.5-pro",
            additional_kwargs={"api_key": "k", "top_p": 0.9},
        )
        GoogleGenAILLMAdapter(config).get_llm()

        _, kwargs = chat.call_args
        assert kwargs["top_p"] == 0.9
        # Key-resolution entries must be consumed, not forwarded verbatim.
        assert "api_key" not in kwargs

    def test_missing_api_key_raises(self, fake_google_genai):
        config = LLMConfig(name="g", provider="google", model_name="gemini-1.5-pro")
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GoogleGenAILLMAdapter(config).get_llm()

    def test_config_additional_kwargs_not_mutated(self, fake_google_genai):
        additional = {"api_key": "k", "top_p": 0.5}
        config = LLMConfig(
            name="g",
            provider="google",
            model_name="gemini-1.5-pro",
            additional_kwargs=additional,
        )
        GoogleGenAILLMAdapter(config).get_llm()
        assert additional == {"api_key": "k", "top_p": 0.5}


# ---------------------------------------------------------------------------
# Embedding adapter behaviour
# ---------------------------------------------------------------------------

class TestGeminiEmbeddings:
    def test_model_and_key_forwarded(self, fake_google_genai):
        _, embed = fake_google_genai
        config = EmbeddingConfig(
            name="e",
            provider="google",
            model_name="models/text-embedding-004",
            api_key="key-123",
        )
        GoogleGenAIEmbeddingAdapter(config).get_embedding_model()

        _, kwargs = embed.call_args
        assert kwargs["model"] == "models/text-embedding-004"
        assert kwargs["google_api_key"] == "key-123"

    def test_api_key_from_env(self, fake_google_genai, monkeypatch):
        _, embed = fake_google_genai
        monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
        config = EmbeddingConfig(
            name="e", provider="gemini", model_name="models/embedding-001"
        )
        GoogleGenAIEmbeddingAdapter(config).get_embedding_model()

        _, kwargs = embed.call_args
        assert kwargs["google_api_key"] == "env-key"

    def test_missing_api_key_raises(self, fake_google_genai):
        config = EmbeddingConfig(
            name="e", provider="google", model_name="models/embedding-001"
        )
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GoogleGenAIEmbeddingAdapter(config).get_embedding_model()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
