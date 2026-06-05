"""
Checkpointer utilities for LangGraph agents.
This module provides functionality to create various types of checkpointers,
including MongoDB for persistent state.
"""

from typing import Any

from pymongo import MongoClient

from agent_builder.utils.logging_config import get_logger, sanitize_connection_string

# Set up module logger
logger = get_logger(__name__)

try:
    from langgraph.checkpoint.mongodb import MongoDBSaver
except ImportError:
    logger.warning(
        "MongoDBSaver not available. Install langgraph with extras: pip install langgraph[mongodb]"
    )
    MongoDBSaver = None


def get_mongodb_checkpointer(
    connection_str: str,
    db_name: str = "langgraph",
    collection_name: str = "checkpoints",
    name: str = "mongodb_checkpointer",
) -> Any:
    """
    Create a MongoDB checkpointer for LangGraph.

    Args:
        connection_str: MongoDB connection string
        db_name: Database name
        collection_name: Collection name
        name: Name of the checkpointer

    Returns:
        A MongoDBSaver instance

    Raises:
        ImportError: If langgraph[mongodb] is not installed
        ValueError: If the connection string is invalid
    """
    if MongoDBSaver is None:
        logger.error(
            "MongoDBSaver not available. Install langgraph with extras: pip install langgraph[mongodb]"
        )
        raise ImportError(
            "MongoDBSaver not available. Install langgraph with extras: pip install langgraph[mongodb]"
        )

    logger.info(
        "Creating MongoDB checkpointer %s for %s.%s (conn=%s)",
        name, db_name, collection_name, sanitize_connection_string(connection_str),
    )

    try:
        client = MongoClient(connection_str)
        db = client[db_name]
        return MongoDBSaver(db, collection_name, name=name)
    except Exception as e:
        logger.error("Failed to create MongoDB checkpointer: %s", str(e))
        raise ValueError("Failed to create MongoDB checkpointer") from e
