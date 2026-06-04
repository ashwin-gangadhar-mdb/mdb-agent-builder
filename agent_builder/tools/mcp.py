"""
MCP (Model Context Protocol) tools for MDB Agent Builder.

This module provides integration with MCP servers using langchain adapters.
It supports connecting to MCP servers using various transports (stdio, streamable-http, etc.)
and loading MCP tools for use with agents. It handles proper async/await patterns
to ensure coroutines are properly managed.
"""

import asyncio
import traceback
from typing import Any, Callable, Dict, List, Optional

from agent_builder.utils.logging_config import get_logger

# Set up module logger
logger = get_logger(__name__)

_MCP_IMPORT_ERROR = (
    "MCP integration requires additional dependencies. "
    "Please install them with: pip install langchain-mcp-adapters mcp"
)


def _require_mcp() -> Dict[str, Any]:
    """Import the optional MCP dependencies on demand.

    The MCP extras are optional, so we import them lazily here rather than at
    module load time.  Importing them eagerly would make the entire
    ``agent_builder.tools`` package fail to import whenever MCP is not
    installed, even for users who never use MCP tools.

    Raises:
        ImportError: with installation guidance if the extras are missing.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools, to_fastmcp
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:
        raise ImportError(_MCP_IMPORT_ERROR) from exc

    return {
        "MultiServerMCPClient": MultiServerMCPClient,
        "load_mcp_tools": load_mcp_tools,
        "to_fastmcp": to_fastmcp,
        "ClientSession": ClientSession,
        "stdio_client": stdio_client,
        "streamablehttp_client": streamablehttp_client,
    }


class MCPToolManager:
    """
    Tool manager for Model Context Protocol (MCP) servers.

    This class provides functionality to connect to MCP servers and load tools from them.
    It supports both single-server connections and multi-server setups, with proper handling
    of asynchronous operations to prevent coroutine warnings or errors.

    The manager handles:
    - Connection to MCP servers via different transport mechanisms (stdio, HTTP)
    - Proper async/await handling of operations
    - Error handling during server connections
    - Loading and processing of tools from servers
    """

    def __init__(self):
        """Initialize the MCP tool manager with empty client and tool dictionaries."""
        self.clients = {}  # Dictionary of connected clients
        self.tools = {}  # Dictionary of loaded tools

    async def get_tools_from_server(
        self, server_name: str, config: Dict[str, Any]
    ) -> List[Any]:
        """
        Get tools from a specific MCP server asynchronously.

        This coroutine connects to a specified MCP server using the configured transport
        mechanism (stdio or streamable-http), initializes the connection, and loads
        all available tools from that server.

        Args:
            server_name: Name of the server for identification
            config: Server configuration dictionary with transport details

        Returns:
            List of LangChain tools from the MCP server

        Raises:
            ValueError: If the transport is not supported or configuration is invalid
        """
        mcp = _require_mcp()
        stdio_client = mcp["stdio_client"]
        streamablehttp_client = mcp["streamablehttp_client"]
        ClientSession = mcp["ClientSession"]
        load_mcp_tools = mcp["load_mcp_tools"]

        # Get the transport type, defaulting to stdio if not specified
        transport = config.get("transport", "stdio").lower()
        logger.info(
            "Loading MCP tools from server '%s' using %s transport",
            server_name,
            transport,
        )

        # Handle stdio transport type (command-line based MCP server)
        if transport == "stdio":
            # Validate required configuration
            if not config.get("command"):
                raise ValueError(
                    f"Server '{server_name}' config must include 'command'"
                )

            # Get command and arguments
            command = config["command"]
            args = config.get("args", [])

            logger.debug(
                "Connecting to MCP server '%s' with command: %s %s",
                server_name,
                command,
                " ".join(args),
            )

            # Create stdio client connection
            async with stdio_client({"command": command, "args": args}) as (
                read,
                write,
            ):
                async with ClientSession(read, write) as session:
                    # Initialize the connection to the MCP server
                    await session.initialize()

                    # Load tools from the connected session
                    tools = await load_mcp_tools(session)
                    logger.info(
                        "Loaded %d tools from MCP server '%s'", len(tools), server_name
                    )
                    return tools

        # Handle HTTP transport type (web-based MCP server)
        elif transport in ["streamable_http", "streamable-http"]:
            # Validate required configuration
            if not config.get("url"):
                raise ValueError(f"Server '{server_name}' config must include 'url'")

            # Get URL and optional headers
            url = config["url"]
            headers = config.get("headers", {})

            logger.debug("Connecting to MCP server '%s' at URL: %s", server_name, url)

            # Create HTTP client connection
            async with streamablehttp_client(url, headers=headers) as (read, write, _):
                async with ClientSession(read, write) as session:
                    # Initialize the connection to the MCP server
                    await session.initialize()

                    # Load tools from the connected session
                    tools = await load_mcp_tools(session)
                    logger.info(
                        "Loaded %d tools from MCP server '%s'", len(tools), server_name
                    )
                    return tools
        else:
            raise ValueError(
                f"Unsupported transport '{transport}' for server '{server_name}'"
            )

    async def load_tools_from_servers(
        self, servers_config: Dict[str, Dict[str, Any]]
    ) -> List[Any]:
        """
        Load tools from multiple MCP servers using MultiServerMCPClient.

        This coroutine connects to multiple MCP servers simultaneously
        using the MultiServerMCPClient, which efficiently manages
        connections to different servers based on their configurations.

        Args:
            servers_config: Dictionary mapping server names to their configurations

        Returns:
            List of LangChain tools from all MCP servers
        """
        logger.info("Loading MCP tools from %d servers", len(servers_config))

        MultiServerMCPClient = _require_mcp()["MultiServerMCPClient"]
        client = MultiServerMCPClient(servers_config)
        all_tools = await client.get_tools()

        logger.info("Loaded a total of %d tools from all MCP servers", len(all_tools))
        return all_tools

    def run_async(self, coro):
        """
        Run an async coroutine in a synchronous context with proper error handling.

        This method ensures that coroutines are properly awaited, handling the event loop
        management automatically. It also provides proper error handling to avoid
        uncaught exceptions from asyncio operations.

        Args:
            coro: The coroutine object to execute

        Returns:
            The result of the coroutine execution, or an empty list if execution fails
        """
        try:
            # Get or create the event loop
            loop = None
            created_new_loop = False

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop exists in current thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                created_new_loop = True

            # Run the coroutine
            result = loop.run_until_complete(coro)

            # Close the loop if we created a new one
            if created_new_loop:
                loop.close()

            return result
        except Exception as e:  # pylint: disable=broad-except
            # Log any errors that occur during async execution
            logger.error(
                "Error running async coroutine: %s\n%s", str(e), traceback.format_exc()
            )
            return []

    def get_tools(
        self,
        servers_config: Dict[str, Dict[str, Any]],
        server_name: Optional[str] = None,
    ) -> List[Any]:
        """
        Synchronous method to get tools from MCP servers.

        This method handles the loading of tools from either a specific MCP server
        or from multiple servers. It properly manages coroutines and ensures they are
        awaited correctly to prevent RuntimeWarnings.

        Args:
            servers_config: Dictionary mapping server names to their configurations
            server_name: Optional specific server to load tools from

        Returns:
            List of LangChain tools from the MCP servers
        """
        try:
            if server_name:
                if server_name not in servers_config:
                    logger.warning(
                        "Server '%s' not found in configuration", server_name
                    )
                    return []

                logger.info("Loading tools from specific server: %s", server_name)
                try:
                    # Create the coroutine object first, then pass it to run_async
                    # This ensures proper handling of the async code
                    coroutine_object = self.get_tools_from_server(
                        server_name, servers_config[server_name]
                    )
                    tools = self.run_async(coroutine_object)

                    # Apply filtering if specified in the config
                    if "filter" in servers_config[server_name]:
                        filter_list = servers_config[server_name]["filter"]
                        tools = list(
                            filter(lambda tool: tool.name in filter_list, tools)
                        )
                    return tools
                except Exception as e:  # pylint: disable=broad-except
                    logger.error(
                        "Error getting tools from server %s: %s\n%s",
                        server_name,
                        str(e),
                        traceback.format_exc(),
                    )
                    return []

            # For multiple servers
            try:
                # Create the coroutine object first, then pass it to run_async
                # This is the key fix to prevent the "coroutine was never awaited" warning
                coroutine_object = self.load_tools_from_servers(servers_config)
                return self.run_async(coroutine_object)
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    "Error loading tools from servers: %s\n%s",
                    str(e),
                    traceback.format_exc(),
                )
                return []

        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Unexpected error in get_tools: %s\n%s", str(e), traceback.format_exc()
            )
            return []


def get_mcp_tools(
    servers_config: Dict[str, Dict[str, Any]], server_name: Optional[str] = None
) -> List[Any]:
    """
    Get tools from MCP servers.

    This is the main entry point for loading tools from MCP servers.
    It handles the creation and management of the MCPToolManager and
    provides error handling for the tool loading process.

    Args:
        servers_config: Dictionary mapping server names to their configurations
        server_name: Optional specific server to load tools from

    Returns:
        List of LangChain tools from the MCP servers

    Example:
        ```python
        servers = {
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": ["/path/to/math_server.py"]
            },
            "weather": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp/"
            }
        }

        tools = get_mcp_tools(servers)
        ```
    """
    try:
        # Create a tool manager instance
        manager = MCPToolManager()

        try:
            # Get tools using the manager
            return manager.get_tools(servers_config, server_name)
        except Exception as e:  # pylint: disable=broad-except
            # Handle errors in the tool loading process
            logger.error(
                "Error getting tools from server %s: %s\n%s",
                server_name or "multiple",
                str(e),
                traceback.format_exc(),
            )
            return []

    except Exception as e:  # pylint: disable=broad-except
        # Handle errors in the manager creation or other unexpected issues
        logger.error(
            "Fatal error in get_mcp_tools: %s\n%s", str(e), traceback.format_exc()
        )
        return []


def convert_langchain_tool_to_mcp(langchain_tool) -> Any:
    """
    Convert a LangChain tool to an MCP-compatible tool.

    This utility function takes a standard LangChain tool and converts it
    to an MCP-compatible format that can be used with MCP servers.

    Args:
        langchain_tool: A LangChain tool to convert

    Returns:
        MCP-compatible tool object

    Example:
        ```python
        from langchain_core.tools import tool

        @tool
        def add(a: int, b: int) -> int:
            '''Add two numbers'''
            return a + b

        mcp_tool = convert_langchain_tool_to_mcp(add)
        ```
    """
    to_fastmcp = _require_mcp()["to_fastmcp"]
    return to_fastmcp(langchain_tool)
