"""
Embeddings loader for various providers using LangChain.

This module provides functionality to load embedding models from different
providers including Bedrock, SageMaker, VertexAI, Azure, Together, Fireworks,
Cohere, VoyageAI, and Ollama.

The loader now delegates all provider-specific construction to the adapter
classes defined in ``agent_builder.embeddings.adapters``, following the adapter
design pattern.  The ``load_embedding_model`` / ``load_embedding_models`` public
API is preserved for backward compatibility.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from langchain_core.embeddings import Embeddings

from agent_builder.utils.logging_config import get_logger

# Set up module logger
logger = get_logger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""

    name: str
    provider: str
    model_name: str
    normalize: bool = False
    dimensions: Optional[int] = None
    additional_kwargs: Optional[Dict[str, Any]] = None
    api_key: Optional[str] = None


def load_embedding_model(config: EmbeddingConfig) -> Embeddings:
    """
    Load an embedding model based on the provided configuration.

    Delegates to ``EmbeddingAdapterFactory.create(config).get_embedding_model()``
    so that provider-specific construction logic lives in the corresponding
    adapter class rather than in a monolithic conditional block here.

    Args:
        config: EmbeddingConfig containing provider, model name, and other parameters

    Returns:
        An initialized LangChain Embeddings instance

    Raises:
        ValueError: If the provider is not supported or required configuration is missing
    """
    # Import here to avoid circular dependency at module load time
    from agent_builder.embeddings.adapters import EmbeddingAdapterFactory

    logger.info(
        "Loading embedding model for provider: %s, model: %s",
        config.provider,
        config.model_name,
    )
    adapter = EmbeddingAdapterFactory.create(config)
    return adapter.get_embedding_model()


def load_embedding_models(
    configs: Union[EmbeddingConfig, List[EmbeddingConfig]],
) -> Dict[str, Embeddings]:
    """
    Load multiple embedding models based on the provided configurations.

    Args:
        configs: Either a single EmbeddingConfig or a list of EmbeddingConfigs

    Returns:
        A dictionary mapping embedding model names to their initialized instances
    """
    if isinstance(configs, EmbeddingConfig):
        configs = [configs]

    embeddings = {}
    for config in configs:
        embeddings[config.name] = load_embedding_model(config)

    return embeddings
