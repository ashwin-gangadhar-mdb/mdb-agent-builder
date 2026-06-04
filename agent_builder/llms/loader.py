"""
LLM loader for various providers using LangChain.

This module provides functionality to load LLM models from different providers
including Bedrock, Fireworks, TogetherAI, Cohere, Anthropic, Azure, Google Gemini,
Ollama, and AWS SageMaker.

The loader now delegates all provider-specific construction to the adapter
classes defined in ``agent_builder.llms.adapters``, following the adapter
design pattern.  The ``load_llm`` / ``load_llms`` public API is preserved for
backward compatibility.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from langchain_core.language_models import BaseLLM

from agent_builder.utils.logging_config import get_logger

# Set up module logger
logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM models."""

    name: str
    provider: str
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    streaming: bool = False
    additional_kwargs: Optional[Dict[str, Any]] = None


def load_llm(config: LLMConfig) -> BaseLLM:
    """
    Load an LLM based on the provided configuration.

    Delegates to ``LLMAdapterFactory.create(config).get_llm()`` so that
    provider-specific construction logic lives in the corresponding adapter
    class rather than in a monolithic conditional block here.

    Args:
        config: LLMConfig containing provider, model name, and other parameters

    Returns:
        An initialized LangChain LLM instance

    Raises:
        ValueError: If the provider is not supported or required configuration is missing
    """
    # Import here to avoid circular dependency at module load time
    from agent_builder.llms.adapters import LLMAdapterFactory

    logger.info("Loading LLM for provider: %s, model: %s", config.provider, config.model_name)
    adapter = LLMAdapterFactory.create(config)
    return adapter.get_llm()


def load_llms(configs: Union[LLMConfig, List[LLMConfig]]) -> Dict[str, BaseLLM]:
    """
    Load multiple LLMs based on the provided configurations.

    Args:
        configs: Either a single LLMConfig or a list of LLMConfigs

    Returns:
        A dictionary mapping LLM names to their initialized instances
    """
    if isinstance(configs, LLMConfig):
        configs = [configs]

    llms = {}
    for config in configs:
        llms[config.name] = load_llm(config)

    return llms
