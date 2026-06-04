"""
Memory adapter factory.

``MemoryAdapterFactory`` reads a ``MemoryConfig`` dataclass and returns the
appropriate concrete ``BaseMemoryAdapter`` subclass, decoupling callers from
the concrete MongoDB implementations.

Supported ``memory_type`` values
---------------------------------
* ``"episodic"``       → ``MongoDBEpisodicMemoryAdapter``
* ``"observational"``  → ``MongoDBObservationalMemoryAdapter``
* ``"general"``        → ``MongoDBMemoryProvider`` (type-agnostic)

Usage (programmatic)::

    from agent_builder.memory.adapters import MemoryAdapterFactory, MemoryConfig
    from agent_builder.core.interfaces import BaseMemoryAdapter

    cfg = MemoryConfig(
        name="recall",
        memory_type="episodic",
        connection_str="mongodb+srv://...",
        namespace="mydb.memories",
        embedding_model=my_embedding_instance,
        index_name="episodic_index",
    )
    adapter: BaseMemoryAdapter = MemoryAdapterFactory.create(cfg)

Usage (from YAML via ``load_application``)::

    memory:
      - name: recall
        memory_type: episodic
        connection_str: ${MONGODB_URI}
        namespace: mydb.episodic_memories
        embedding_model: my_embedding       # reference to embeddings section
        index_name: episodic_index

      - name: observations
        memory_type: observational
        connection_str: ${MONGODB_URI}
        namespace: mydb.observational_memories
        embedding_model: my_embedding
        llm: my_llm                         # reference to llms section
        index_name: observational_index
        extraction_prompt: |                # optional — override default
          Extract key user facts: {text}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM

from agent_builder.core.interfaces import BaseMemoryAdapter
from agent_builder.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class MemoryConfig:
    """
    Configuration for a single memory adapter instance.

    Attributes:
        name:              Logical name for this memory store (used as a key
                           in the loaded components dict).
        memory_type:       One of ``"episodic"``, ``"observational"``, or
                           ``"general"`` (default ``"episodic"``).
        connection_str:    MongoDB Atlas connection string.
        namespace:         MongoDB namespace in ``"db.collection"`` format.
        embedding_model:   An initialised LangChain ``Embeddings`` instance
                           (resolved from the ``embeddings`` section of the
                           YAML before this config is constructed).
        llm:               An initialised LangChain LLM / chat-model instance.
                           Required only for ``memory_type = "observational"``.
        index_name:        Atlas Vector Search index name
                           (default ``"agent_memory_index"``).
        embedding_field:   MongoDB field that stores the embedding vector
                           (default ``"embedding"``).
        text_field:        MongoDB field that stores the raw text
                           (default ``"text"``).
        top_k:             Default number of memories to retrieve per search
                           (default ``5``).
        extraction_prompt: Optional custom prompt template for the
                           observational adapter's LLM extraction step.
                           Must contain a ``{text}`` placeholder.
        additional_kwargs: Catch-all for any extra constructor arguments.
    """

    name: str
    memory_type: str = "episodic"
    connection_str: Optional[str] = None
    namespace: Optional[str] = None
    embedding_model: Optional[Embeddings] = None
    llm: Optional[BaseLLM] = None
    index_name: str = "agent_memory_index"
    embedding_field: str = "embedding"
    text_field: str = "text"
    top_k: int = 5
    extraction_prompt: Optional[str] = None
    additional_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class MemoryAdapterFactory:
    """
    Factory that maps a ``memory_type`` string to its concrete adapter class.

    Supports runtime registration of new adapter types via
    ``MemoryAdapterFactory.register()``.
    """

    # registry maps lowercase memory_type → (module_path, class_name)
    # Using strings for lazy import keeps startup cost low.
    _REGISTRY: Dict[str, tuple] = {
        "episodic": (
            "agent_builder.memory.mongodb_memory",
            "MongoDBEpisodicMemoryAdapter",
        ),
        "observational": (
            "agent_builder.memory.mongodb_memory",
            "MongoDBObservationalMemoryAdapter",
        ),
        "general": (
            "agent_builder.memory.mongodb_memory",
            "MongoDBMemoryProvider",
        ),
    }

    @classmethod
    def create(cls, config: MemoryConfig) -> BaseMemoryAdapter:
        """
        Instantiate and return the appropriate memory adapter for *config*.

        The factory builds a uniform ``kwargs`` dict from every field in
        *config* and passes it to the adapter constructor.  Each concrete
        class accepts only what it needs via its explicit parameters; any
        surplus keys are absorbed by the ``**kwargs`` catch-all that every
        adapter in the hierarchy defines.  No type-specific branching is
        required here.

        Args:
            config: A ``MemoryConfig`` describing the adapter and its
                    connection parameters.

        Returns:
            A concrete ``BaseMemoryAdapter`` instance.

        Raises:
            ValueError: If required fields are missing or the memory_type is
                        not registered.
        """
        memory_type = config.memory_type.lower()

        if memory_type not in cls._REGISTRY:
            raise ValueError(
                f"Unsupported memory_type: '{memory_type}'. "
                f"Available types: {sorted(cls._REGISTRY)}"
            )

        cls._validate_config(config, memory_type)

        import importlib
        module_path, class_name = cls._REGISTRY[memory_type]
        adapter_cls = getattr(importlib.import_module(module_path), class_name)

        # Build kwargs from all MemoryConfig fields; each adapter __init__
        # accepts only what it needs — surplus keys absorbed by **kwargs.
        kwargs: Dict[str, Any] = {
            "connection_str": config.connection_str,
            "namespace": config.namespace,
            "embedding_model": config.embedding_model,
            "index_name": config.index_name,
            "embedding_field": config.embedding_field,
            "text_field": config.text_field,
        }
        if config.llm is not None:
            kwargs["llm"] = config.llm
        if config.extraction_prompt is not None:
            kwargs["extraction_prompt"] = config.extraction_prompt
        if config.additional_kwargs:
            kwargs.update(config.additional_kwargs)

        logger.info(
            "Creating memory adapter: type='%s', name='%s', namespace='%s'",
            memory_type, config.name, config.namespace,
        )
        return adapter_cls(**kwargs)

    @classmethod
    def create_many(cls, configs: List[MemoryConfig]) -> Dict[str, BaseMemoryAdapter]:
        """
        Instantiate multiple memory adapters and return them keyed by name.

        Args:
            configs: List of ``MemoryConfig`` objects.

        Returns:
            Dict mapping ``config.name`` → ``BaseMemoryAdapter`` instance.
        """
        return {cfg.name: cls.create(cfg) for cfg in configs}

    @classmethod
    def register(cls, memory_type: str, module_path: str, class_name: str) -> None:
        """
        Register a custom memory adapter class.

        Args:
            memory_type:  Lower-case key (e.g. ``"redis"``).
            module_path:  Dotted module path where the class lives.
            class_name:   Class name within that module.
        """
        cls._REGISTRY[memory_type.lower()] = (module_path, class_name)
        logger.info(
            "Registered custom memory adapter: '%s' → %s.%s",
            memory_type, module_path, class_name,
        )

    @classmethod
    def available_types(cls) -> List[str]:
        """Return the list of currently registered memory type keys."""
        return sorted(cls._REGISTRY.keys())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _validate_config(cls, config: MemoryConfig, memory_type: str) -> None:
        required = ["connection_str", "namespace", "embedding_model"]
        if memory_type == "observational":
            required.append("llm")
        missing = [f for f in required if not getattr(config, f, None)]
        if missing:
            raise ValueError(
                f"Missing required field(s) for '{memory_type}' memory adapter "
                f"'{config.name}': {', '.join(missing)}"
            )
