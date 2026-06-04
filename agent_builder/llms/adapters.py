"""
Concrete LLM adapters — one per supported provider.

Each adapter wraps a third-party LangChain chat-model class and implements
``BaseLLMAdapter.get_llm()``, making the rest of the framework
provider-agnostic.

Usage::

    from agent_builder.llms.adapters import LLMAdapterFactory
    from agent_builder.llms.loader import LLMConfig

    config = LLMConfig(name="my-llm", provider="anthropic", model_name="claude-3-5-sonnet-20241022")
    adapter = LLMAdapterFactory.create(config)
    llm = adapter.get_llm()
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from langchain_core.language_models import BaseLLM

from agent_builder.core.interfaces import BaseLLMAdapter
from agent_builder.llms.loader import LLMConfig
from agent_builder.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: build common kwargs from an LLMConfig
# ---------------------------------------------------------------------------

def _base_kwargs(config: LLMConfig) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "model": config.model_name,
        "temperature": config.temperature,
        "streaming": config.streaming,
    }
    if config.max_tokens:
        kwargs["max_tokens"] = config.max_tokens
    return kwargs


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------

class BedrockLLMAdapter(BaseLLMAdapter):
    """Adapter for Amazon Bedrock chat models."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def get_llm(self) -> BaseLLM:
        from langchain_aws import ChatBedrock

        kwargs = _base_kwargs(self._config)
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug("Initialising Bedrock LLM with model: %s", self._config.model_name)
        return ChatBedrock(**kwargs)


class FireworksLLMAdapter(BaseLLMAdapter):
    """Adapter for Fireworks AI chat models."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def get_llm(self) -> BaseLLM:
        from langchain_fireworks import ChatFireworks

        kwargs = _base_kwargs(self._config)
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug("Initialising Fireworks LLM with model: %s", self._config.model_name)
        return ChatFireworks(**kwargs)


class TogetherLLMAdapter(BaseLLMAdapter):
    """Adapter for Together AI chat models."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def get_llm(self) -> BaseLLM:
        from langchain_together import ChatTogether

        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable is required for Together AI")

        kwargs = _base_kwargs(self._config)
        kwargs["api_key"] = api_key
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug("Initialising Together LLM with model: %s", self._config.model_name)
        return ChatTogether(**kwargs)


class CohereLLMAdapter(BaseLLMAdapter):
    """Adapter for Cohere chat models."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def get_llm(self) -> BaseLLM:
        from langchain_cohere import ChatCohere

        kwargs = _base_kwargs(self._config)
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug("Initialising Cohere LLM with model: %s", self._config.model_name)
        return ChatCohere(**kwargs)


class AnthropicLLMAdapter(BaseLLMAdapter):
    """Adapter for Anthropic (Claude) chat models."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def get_llm(self) -> BaseLLM:
        from langchain_anthropic import ChatAnthropic

        kwargs = _base_kwargs(self._config)
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug("Initialising Anthropic LLM with model: %s", self._config.model_name)
        return ChatAnthropic(**kwargs)


class AzureLLMAdapter(BaseLLMAdapter):
    """Adapter for Azure OpenAI chat models."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def get_llm(self) -> BaseLLM:
        from langchain_openai import AzureChatOpenAI

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required for Azure OpenAI")

        additional = self._config.additional_kwargs or {}
        azure_endpoint = additional.get("azure_endpoint") or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            raise ValueError("Azure endpoint is required (set AZURE_OPENAI_ENDPOINT or pass azure_endpoint in additional_kwargs)")

        kwargs = _base_kwargs(self._config)
        kwargs["azure_endpoint"] = azure_endpoint
        kwargs.update(additional)
        logger.debug("Initialising Azure OpenAI LLM with model: %s", self._config.model_name)
        return AzureChatOpenAI(**kwargs)


class OllamaLLMAdapter(BaseLLMAdapter):
    """Adapter for locally-hosted Ollama models."""

    _DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def get_llm(self) -> BaseLLM:
        from langchain_ollama.llms import OllamaLLM

        base_url = (self._config.additional_kwargs or {}).get(
            "base_url", self._DEFAULT_BASE_URL
        )
        kwargs = _base_kwargs(self._config)
        kwargs["base_url"] = base_url
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug("Initialising Ollama LLM with model: %s, base_url: %s", self._config.model_name, base_url)
        return OllamaLLM(**kwargs)


class SageMakerLLMAdapter(BaseLLMAdapter):
    """Adapter for AWS SageMaker endpoint-hosted models."""

    _DEFAULT_REGION = "us-east-1"

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    def get_llm(self) -> BaseLLM:
        from langchain_community.llms.sagemaker_endpoint import SagemakerEndpoint

        additional = self._config.additional_kwargs
        if not additional:
            raise ValueError("SageMaker requires additional_kwargs (at minimum 'endpoint_name')")
        if "endpoint_name" not in additional:
            raise ValueError("Missing required configuration for SageMaker: endpoint_name")

        model_kwargs = _base_kwargs(self._config)
        logger.debug(
            "Initialising SageMaker LLM endpoint: %s, region: %s",
            additional["endpoint_name"],
            additional.get("region_name", self._DEFAULT_REGION),
        )
        return SagemakerEndpoint(
            endpoint_name=additional["endpoint_name"],
            region_name=additional.get("region_name", self._DEFAULT_REGION),
            model_kwargs=model_kwargs,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: Dict[str, type] = {
    "bedrock": BedrockLLMAdapter,
    "fireworks": FireworksLLMAdapter,
    "together": TogetherLLMAdapter,
    "cohere": CohereLLMAdapter,
    "anthropic": AnthropicLLMAdapter,
    "azure": AzureLLMAdapter,
    "ollama": OllamaLLMAdapter,
    "sagemaker": SageMakerLLMAdapter,
}


class LLMAdapterFactory:
    """
    Factory that maps a provider name to its concrete ``BaseLLMAdapter``.

    Supports runtime registration of new adapter types via
    ``LLMAdapterFactory.register()``.
    """

    @classmethod
    def create(cls, config: LLMConfig) -> BaseLLMAdapter:
        """
        Create and return the appropriate ``BaseLLMAdapter`` for *config*.

        Args:
            config: An ``LLMConfig`` describing the provider and model.

        Returns:
            A concrete ``BaseLLMAdapter`` instance.

        Raises:
            ValueError: If the provider is not registered.
        """
        provider = config.provider.lower()
        adapter_cls = _ADAPTER_REGISTRY.get(provider)
        if adapter_cls is None:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                f"Available providers: {sorted(_ADAPTER_REGISTRY)}"
            )
        logger.info("Creating LLM adapter for provider: %s, model: %s", provider, config.model_name)
        return adapter_cls(config)

    @classmethod
    def register(cls, provider: str, adapter_cls: type) -> None:
        """
        Register a custom ``BaseLLMAdapter`` subclass for *provider*.

        Args:
            provider:    The lower-case provider key (e.g. ``"mycloud"``).
            adapter_cls: A subclass of ``BaseLLMAdapter``.
        """
        if not issubclass(adapter_cls, BaseLLMAdapter):
            raise TypeError(f"{adapter_cls} must be a subclass of BaseLLMAdapter")
        _ADAPTER_REGISTRY[provider.lower()] = adapter_cls
        logger.info("Registered custom LLM adapter: %s -> %s", provider, adapter_cls.__name__)

    @classmethod
    def available_providers(cls) -> list:
        """Return the list of currently registered provider keys."""
        return sorted(_ADAPTER_REGISTRY.keys())
