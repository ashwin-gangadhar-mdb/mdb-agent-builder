"""
Concrete tool adapters — one per supported tool type.

Each adapter wraps an underlying tool implementation and implements
``BaseToolAdapter.get_tools()``, making the rest of the framework agnostic
to how specific tools are constructed.

Usage::

    from agent_builder.tools.adapters import ToolAdapterFactory
    from agent_builder.tools.loader import ToolConfig

    config = ToolConfig(
        tool_type="vector_search",
        name="product_search",
        connection_str="mongodb+srv://...",
        namespace="mydb.products",
        embedding_model=my_embedding,
    )
    adapter = ToolAdapterFactory.create(config)
    tools = adapter.get_tools()
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.tools import BaseTool

from agent_builder.core.interfaces import BaseToolAdapter
from agent_builder.tools.loader import ToolConfig, ToolType, _check_required_fields
from agent_builder.utils.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Concrete adapters
# ---------------------------------------------------------------------------

class VectorSearchToolAdapter(BaseToolAdapter):
    """Adapter that exposes a MongoDB Atlas vector-search tool."""

    def __init__(self, config: ToolConfig) -> None:
        _check_required_fields(config, ["connection_str", "namespace", "embedding_model"], ToolType.VECTOR_SEARCH)
        self._config = config

    def get_tools(self) -> List[BaseTool]:
        from agent_builder.tools.mongodb import MongoDBTools

        tool_name = self._config.name or "vector_search_tool"
        logger.debug("Building VectorSearch tool: %s", tool_name)
        mongodb_tools = MongoDBTools(
            name=tool_name,
            connection_str=self._config.connection_str,
            namespace=self._config.namespace,
            embedding_model=self._config.embedding_model,
            **(self._config.additional_kwargs or {}),
        )
        return [mongodb_tools.get_vector_retriever_tool()]


class FullTextSearchToolAdapter(BaseToolAdapter):
    """Adapter that exposes a MongoDB Atlas full-text-search tool."""

    def __init__(self, config: ToolConfig) -> None:
        _check_required_fields(config, ["connection_str", "namespace"], ToolType.FULL_TEXT_SEARCH)
        self._config = config

    def get_tools(self) -> List[BaseTool]:
        from agent_builder.tools.mongodb import MongoDBTools

        tool_name = self._config.name or "full_text_search_tool"
        logger.debug("Building FullTextSearch tool: %s", tool_name)
        mongodb_tools = MongoDBTools(
            name=tool_name,
            connection_str=self._config.connection_str,
            namespace=self._config.namespace,
            embedding_model=None,
            **(self._config.additional_kwargs or {}),
        )
        return [mongodb_tools.get_full_text_search_tool()]


class MongoDBToolkitAdapter(BaseToolAdapter):
    """Adapter that exposes the full MongoDB toolkit (requires an LLM)."""

    def __init__(self, config: ToolConfig) -> None:
        _check_required_fields(config, ["connection_str", "namespace", "llm"], ToolType.MONGODB_TOOLKIT)
        self._config = config

    def get_tools(self) -> List[BaseTool]:
        from agent_builder.tools.mongodb import MongoDBTools

        tool_name = self._config.name or "mongodb_toolkit"
        logger.debug("Building MongoDB toolkit: %s", tool_name)
        mongodb_tools = MongoDBTools(
            name=tool_name,
            connection_str=self._config.connection_str,
            namespace=self._config.namespace,
            embedding_model=None,
            **(self._config.additional_kwargs or {}),
        )
        return mongodb_tools.get_mdb_toolkit(self._config.llm)


class NLToMQLToolAdapter(BaseToolAdapter):
    """Adapter that exposes a natural-language-to-MQL conversion tool."""

    def __init__(self, config: ToolConfig) -> None:
        _check_required_fields(config, ["connection_str", "namespace", "llm"], ToolType.NL_TO_MQL)
        self._config = config

    def get_tools(self) -> List[BaseTool]:
        from agent_builder.tools.mongodb import MongoDBTools

        tool_name = self._config.name or "nl_to_mql_tool"
        logger.debug("Building NL-to-MQL tool: %s", tool_name)
        mongodb_tools = MongoDBTools(
            name=tool_name,
            connection_str=self._config.connection_str,
            namespace=self._config.namespace,
            embedding_model=None,
            **(self._config.additional_kwargs or {}),
        )
        return [mongodb_tools.get_nl_to_mql_tool(self._config.llm)]


class MCPToolAdapter(BaseToolAdapter):
    """Adapter that exposes tools sourced from MCP (Model Context Protocol) servers."""

    def __init__(self, config: ToolConfig) -> None:
        _check_required_fields(config, ["servers_config"], ToolType.MCP)
        self._config = config

    def get_tools(self) -> List[BaseTool]:
        from agent_builder.tools.mcp import get_mcp_tools

        server_name = self._config.name
        logger.debug("Building MCP tools for server: %s", server_name)
        result = get_mcp_tools(self._config.servers_config, server_name)
        # get_mcp_tools may return a list or a single tool
        return result if isinstance(result, list) else [result]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: Dict[str, type] = {
    ToolType.VECTOR_SEARCH: VectorSearchToolAdapter,
    ToolType.FULL_TEXT_SEARCH: FullTextSearchToolAdapter,
    ToolType.MONGODB_TOOLKIT: MongoDBToolkitAdapter,
    ToolType.NL_TO_MQL: NLToMQLToolAdapter,
    ToolType.MCP: MCPToolAdapter,
}


class ToolAdapterFactory:
    """
    Factory that maps a tool-type key to its concrete ``BaseToolAdapter``.

    Supports runtime registration of new adapter types via
    ``ToolAdapterFactory.register()``.
    """

    @classmethod
    def create(cls, config: ToolConfig) -> BaseToolAdapter:
        """
        Create and return the appropriate ``BaseToolAdapter`` for *config*.

        Args:
            config: A ``ToolConfig`` describing the tool type and its parameters.

        Returns:
            A concrete ``BaseToolAdapter`` instance.

        Raises:
            ValueError: If the tool type is not registered.
        """
        tool_type = config.tool_type.lower()
        adapter_cls = _ADAPTER_REGISTRY.get(tool_type)
        if adapter_cls is None:
            raise ValueError(
                f"Unsupported tool type: '{tool_type}'. "
                f"Available types: {sorted(_ADAPTER_REGISTRY)}"
            )
        logger.info("Creating tool adapter for type: %s, name: %s", tool_type, config.name)
        return adapter_cls(config)

    @classmethod
    def register(cls, tool_type: str, adapter_cls: type) -> None:
        """
        Register a custom ``BaseToolAdapter`` subclass for *tool_type*.

        Args:
            tool_type:   The lower-case tool-type key (e.g. ``"my_tool"``).
            adapter_cls: A subclass of ``BaseToolAdapter``.
        """
        if not issubclass(adapter_cls, BaseToolAdapter):
            raise TypeError(f"{adapter_cls} must be a subclass of BaseToolAdapter")
        _ADAPTER_REGISTRY[tool_type.lower()] = adapter_cls
        logger.info(
            "Registered custom tool adapter: %s -> %s", tool_type, adapter_cls.__name__
        )

    @classmethod
    def available_tool_types(cls) -> list:
        """Return the list of currently registered tool-type keys."""
        return sorted(_ADAPTER_REGISTRY.keys())
