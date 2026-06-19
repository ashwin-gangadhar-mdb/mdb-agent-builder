"""
Concrete embedding-model adapters — one per supported provider.

Each adapter wraps a third-party LangChain ``Embeddings`` class and implements
``BaseEmbeddingAdapter.get_embedding_model()``, making the rest of the framework
provider-agnostic.

Usage::

    from agent_builder.embeddings.adapters import EmbeddingAdapterFactory
    from agent_builder.embeddings.loader import EmbeddingConfig

    config = EmbeddingConfig(name="e1", provider="voyageai", model_name="voyage-3")
    adapter = EmbeddingAdapterFactory.create(config)
    model = adapter.get_embedding_model()
"""

from __future__ import annotations

import os
from typing import Any, Dict

from langchain_core.embeddings import Embeddings

from agent_builder.core.interfaces import BaseEmbeddingAdapter
from agent_builder.embeddings.loader import EmbeddingConfig
from agent_builder.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper: resolve API key from config or environment
# ---------------------------------------------------------------------------

def _resolve_api_key(config: EmbeddingConfig, env_var: str) -> str:
    """Return the API key from the config or the environment variable."""
    key = config.api_key or os.environ.get(env_var)
    if not key:
        raise ValueError(
            f"API key required for provider '{config.provider}'. "
            f"Set {env_var} or pass api_key in the configuration."
        )
    return key


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------

class BedrockEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for Amazon Bedrock embedding models."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    def get_embedding_model(self) -> Embeddings:
        from langchain_aws import BedrockEmbeddings

        kwargs: Dict[str, Any] = {"model_id": self._config.model_name}
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug("Initialising Bedrock embedding model: %s", self._config.model_name)
        return BedrockEmbeddings(**kwargs)


class SageMakerEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for AWS SageMaker embedding endpoints."""

    _DEFAULT_REGION = "us-east-1"

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    def get_embedding_model(self) -> Embeddings:
        from langchain_community.embeddings import SagemakerEndpointEmbeddings

        additional = self._config.additional_kwargs
        if not additional or "endpoint_name" not in additional:
            raise ValueError(
                "SageMaker embedding requires 'endpoint_name' in additional_kwargs"
            )

        kwargs: Dict[str, Any] = {}
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)

        logger.debug(
            "Initialising SageMaker embedding endpoint: %s", additional["endpoint_name"]
        )
        return SagemakerEndpointEmbeddings(
            endpoint_name=additional["endpoint_name"],
            region_name=additional.get("region_name", self._DEFAULT_REGION),
            content_handler=additional.get("content_handler"),
        )


class VertexAIEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for Google Cloud Vertex AI embedding models."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    def get_embedding_model(self) -> Embeddings:
        from langchain_community.embeddings import VertexAIEmbeddings

        kwargs: Dict[str, Any] = {"model_name": self._config.model_name}
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug("Initialising Vertex AI embedding model: %s", self._config.model_name)
        return VertexAIEmbeddings(**kwargs)


class AzureEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for Azure OpenAI embedding models."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    def get_embedding_model(self) -> Embeddings:
        from langchain_openai import AzureOpenAIEmbeddings

        api_key = _resolve_api_key(self._config, "AZURE_OPENAI_API_KEY")
        additional = self._config.additional_kwargs or {}
        azure_endpoint = additional.get("azure_endpoint") or os.environ.get(
            "AZURE_OPENAI_ENDPOINT"
        )
        if not azure_endpoint:
            raise ValueError(
                "Azure endpoint is required (set AZURE_OPENAI_ENDPOINT or pass "
                "azure_endpoint in additional_kwargs)"
            )

        kwargs: Dict[str, Any] = {
            "deployment": self._config.model_name,
            "api_key": api_key,
            "azure_endpoint": azure_endpoint,
        }
        kwargs.update(additional)
        logger.debug("Initialising Azure OpenAI embedding model: %s", self._config.model_name)
        return AzureOpenAIEmbeddings(**kwargs)


class TogetherEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for Together AI embedding models."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    def get_embedding_model(self) -> Embeddings:
        from langchain_together import TogetherEmbeddings

        api_key = _resolve_api_key(self._config, "TOGETHER_API_KEY")
        kwargs: Dict[str, Any] = {
            "model_name": self._config.model_name,
            "api_key": api_key,
        }
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug("Initialising Together embedding model: %s", self._config.model_name)
        return TogetherEmbeddings(**kwargs)


class FireworksEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for Fireworks AI embedding models."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    def get_embedding_model(self) -> Embeddings:
        from langchain_fireworks import FireworksEmbeddings

        api_key = _resolve_api_key(self._config, "FIREWORKS_API_KEY")
        kwargs: Dict[str, Any] = {
            "model": self._config.model_name,
            "api_key": api_key,
        }
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug("Initialising Fireworks embedding model: %s", self._config.model_name)
        return FireworksEmbeddings(**kwargs)


class CohereEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for Cohere embedding models."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    def get_embedding_model(self) -> Embeddings:
        from langchain_community.embeddings import CohereEmbeddings

        api_key = _resolve_api_key(self._config, "COHERE_API_KEY")
        kwargs: Dict[str, Any] = {
            "model": self._config.model_name,
            "api_key": api_key,
        }
        if self._config.dimensions:
            kwargs["dimensions"] = self._config.dimensions
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug("Initialising Cohere embedding model: %s", self._config.model_name)
        return CohereEmbeddings(**kwargs)


class VoyageAIEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for VoyageAI embedding models."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    def get_embedding_model(self) -> Embeddings:
        from langchain_voyageai import VoyageAIEmbeddings

        api_key = _resolve_api_key(self._config, "VOYAGE_API_KEY")
        kwargs: Dict[str, Any] = {
            "model": self._config.model_name,
            "api_key": api_key,
        }
        if self._config.dimensions:
            kwargs["output_dimension"] = self._config.dimensions
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug("Initialising VoyageAI embedding model: %s", self._config.model_name)
        return VoyageAIEmbeddings(**kwargs)


class OllamaEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for locally-hosted Ollama embedding models."""

    _DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    def get_embedding_model(self) -> Embeddings:
        from langchain_community.embeddings import OllamaEmbeddings

        base_url = (self._config.additional_kwargs or {}).get(
            "base_url", self._DEFAULT_BASE_URL
        )
        kwargs: Dict[str, Any] = {
            "model": self._config.model_name,
            "base_url": base_url,
        }
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug(
            "Initialising Ollama embedding model: %s, base_url: %s",
            self._config.model_name,
            base_url,
        )
        return OllamaEmbeddings(**kwargs)


class GoogleGenAIEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for Google Gemini (Generative AI) embedding models.

    Uses ``langchain_google_genai.GoogleGenerativeAIEmbeddings`` with a Google
    AI Studio API key.  This is distinct from the ``vertexai`` provider, which
    targets Vertex AI on Google Cloud and authenticates with GCP credentials.

    Configuration (YAML)::

        embeddings:
          - name: gemini-embed
            provider: google              # or the "gemini" alias
            model_name: models/text-embedding-004

    API key resolution: ``config.api_key`` → ``GOOGLE_API_KEY`` env (set
    ``GEMINI_API_KEY`` as an override via ``additional_kwargs`` if preferred).

    Note: output dimensionality for the newer embedding models is selected at
    query time rather than via the constructor, so a configured ``dimensions``
    value is ignored here (with a debug log) unless explicitly passed through
    ``additional_kwargs``.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    def get_embedding_model(self) -> Embeddings:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        additional = dict(self._config.additional_kwargs or {})
        api_key = (
            self._config.api_key
            or additional.pop("google_api_key", None)
            or additional.pop("api_key", None)
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is required for the Google "
                "Gemini embedding provider. Set the environment variable or pass "
                "api_key in the configuration."
            )

        kwargs: Dict[str, Any] = {
            "model": self._config.model_name,
            "google_api_key": api_key,
        }
        if self._config.dimensions:
            logger.debug(
                "Ignoring 'dimensions=%s' for Google Gemini model '%s' "
                "(output dimensionality is selected at query time).",
                self._config.dimensions,
                self._config.model_name,
            )
        kwargs.update(additional)
        logger.debug(
            "Initialising Google Gemini embedding model: %s", self._config.model_name
        )
        return GoogleGenerativeAIEmbeddings(**kwargs)


class HuggingFaceEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for HuggingFace sentence-transformer embedding models."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    def get_embedding_model(self) -> Embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings

        # HuggingFaceEmbeddings does not accept ``dimensions``/``normalize``
        # directly. Normalisation is configured through ``encode_kwargs`` and
        # the output dimension is fixed by the chosen sentence-transformer.
        kwargs: Dict[str, Any] = {"model_name": self._config.model_name}
        if self._config.normalize:
            kwargs["encode_kwargs"] = {"normalize_embeddings": True}
        if self._config.dimensions:
            logger.debug(
                "Ignoring 'dimensions=%s' for HuggingFace model '%s' "
                "(dimension is determined by the model).",
                self._config.dimensions,
                self._config.model_name,
            )
        if self._config.additional_kwargs:
            kwargs.update(self._config.additional_kwargs)
        logger.debug(
            "Initialising HuggingFace embedding model: %s", self._config.model_name
        )
        return HuggingFaceEmbeddings(**kwargs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: Dict[str, type] = {
    "bedrock": BedrockEmbeddingAdapter,
    "sagemaker": SageMakerEmbeddingAdapter,
    "vertexai": VertexAIEmbeddingAdapter,
    "azure": AzureEmbeddingAdapter,
    "together": TogetherEmbeddingAdapter,
    "fireworks": FireworksEmbeddingAdapter,
    "cohere": CohereEmbeddingAdapter,
    "voyageai": VoyageAIEmbeddingAdapter,
    "ollama": OllamaEmbeddingAdapter,
    "huggingface": HuggingFaceEmbeddingAdapter,
    "google": GoogleGenAIEmbeddingAdapter,
    "gemini": GoogleGenAIEmbeddingAdapter,
}


class EmbeddingAdapterFactory:
    """
    Factory that maps a provider name to its concrete ``BaseEmbeddingAdapter``.

    Supports runtime registration of new adapter types via
    ``EmbeddingAdapterFactory.register()``.
    """

    @classmethod
    def create(cls, config: EmbeddingConfig) -> BaseEmbeddingAdapter:
        """
        Create and return the appropriate ``BaseEmbeddingAdapter`` for *config*.

        Args:
            config: An ``EmbeddingConfig`` describing the provider and model.

        Returns:
            A concrete ``BaseEmbeddingAdapter`` instance.

        Raises:
            ValueError: If the provider is not registered.
        """
        provider = config.provider.lower()
        adapter_cls = _ADAPTER_REGISTRY.get(provider)
        if adapter_cls is None:
            raise ValueError(
                f"Unsupported embedding provider: '{provider}'. "
                f"Available providers: {sorted(_ADAPTER_REGISTRY)}"
            )
        logger.info(
            "Creating embedding adapter for provider: %s, model: %s",
            provider,
            config.model_name,
        )
        return adapter_cls(config)

    @classmethod
    def register(cls, provider: str, adapter_cls: type) -> None:
        """
        Register a custom ``BaseEmbeddingAdapter`` subclass for *provider*.

        Args:
            provider:    The lower-case provider key (e.g. ``"mycloud"``).
            adapter_cls: A subclass of ``BaseEmbeddingAdapter``.
        """
        if not issubclass(adapter_cls, BaseEmbeddingAdapter):
            raise TypeError(
                f"{adapter_cls} must be a subclass of BaseEmbeddingAdapter"
            )
        _ADAPTER_REGISTRY[provider.lower()] = adapter_cls
        logger.info(
            "Registered custom embedding adapter: %s -> %s",
            provider,
            adapter_cls.__name__,
        )

    @classmethod
    def available_providers(cls) -> list:
        """Return the list of currently registered provider keys."""
        return sorted(_ADAPTER_REGISTRY.keys())
