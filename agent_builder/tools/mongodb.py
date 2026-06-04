"""
MongoDB tools for MAAP Agent Builder.

This module provides integration with MongoDB Atlas for vector search,
full text search, and natural language to MQL query conversion.
It enables the agent to interact with MongoDB databases efficiently.
"""

import json
from typing import Any, Dict, List, Optional, Union

import certifi
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import ToolException, tool
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.agent_toolkit import (
    MONGODB_AGENT_SYSTEM_PROMPT,
    MongoDBDatabase,
    MongoDBDatabaseToolkit,
)
from langchain_mongodb.retrievers import MongoDBAtlasFullTextSearchRetriever
from langgraph.prebuilt import create_react_agent
from pymongo import MongoClient

from agent_builder.utils.logging_config import get_logger

# Create module-level logger
logger = get_logger(__name__)


class MongoDBTools:
    """
    Tool for retrieving relevant products and their information from a vector store,
    and for natural language to MQL conversion and execution.
    """

    def __init__(
        self,
        connection_str: str,
        namespace: str,
        embedding_model: Optional[Embeddings],
        name: str = "mongodb_toolkit",
        index_name: Optional[str] = "vector_index",
        embedding_field: Optional[str] = "embedding",
        text_field: Optional[str] = "text",
        top_k: Optional[int] = 5,
        num_candidates: Optional[int] = 100,
        min_score: Optional[float] = 0.7,
        max_edits: Optional[int] = 2,
        prefix_length: Optional[int] = 3,
    ):
        # Create class logger with appropriate name
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Initializing MongoDB tools for namespace: %s", namespace)
        self.name = name
        self.top_k = top_k
        self.connection_str = connection_str
        self.namespace = namespace
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.embedding_field = embedding_field
        self.text_field = text_field
        self.min_score = min_score
        self.max_edits = max_edits
        self.prefix_length = prefix_length

        try:
            self.client = MongoClient(connection_str, tlsCAFile=certifi.where())
            self.logger.info("Successfully connected to MongoDB")
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise ConnectionError(f"Could not connect to MongoDB: {e}")

        self.database_name, self.collection_name = namespace.split(".", 1)
        self.logger.debug(
            f"Connected to MongoDB database: {self.database_name}, collection: {self.collection_name}"
        )

    def _init_vector_retriever(self):
        self.logger.info(f"Initializing vector retriever with index: {self.index_name}")
        vector_store = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string=self.connection_str,
            namespace=self.namespace,
            embedding=self.embedding_model,
            embedding_field=self.embedding_field,
            index_name=self.index_name,
            text_field=self.text_field,
        )
        self.logger.info(
            f"Vector store initialized, creating retriever with top_k={self.top_k}"
        )
        return vector_store.as_retriever(
            search_kwargs={"k": self.top_k, "score_threshold": self.min_score}
        )

    def _init_full_text_retriever(self):
        self.logger.info(
            f"Initializing full-text search retriever with index: {self.index_name}"
        )
        db = self.client[self.database_name]
        collection = db[self.collection_name]

        def retriever(query: str) -> List[Dict[str, Any]]:
            self.logger.info(f"Performing full-text search with query: {query}")
            if not query:
                self.logger.warning("Empty query provided for full-text search")
                return []
            pipeline = [
                {
                    "$search": {
                        "index": self.index_name,
                        "text": {
                            "query": query,
                            "path": [self.text_field],
                            "matchCriteria": "any",
                            "fuzzy": {
                                "maxEdits": self.max_edits,
                                "prefixLength": self.prefix_length,
                            },
                        },
                    }
                },
                {"$project": {"_id": 0}},
                {"$limit": self.top_k},
            ]
            self.logger.debug(
                f"Full-text search pipeline: {json.dumps(pipeline, indent=2)}"
            )
            results = list(collection.aggregate(pipeline))
            self.logger.debug(f"Full-text search returned {len(results)} results")
            self.logger.debug(f"Results: {json.dumps(results, indent=2)}")
            documents = [
                {"source": self.collection_name, "page_content": json.dumps(doc)}
                for doc in results
            ]
            self.logger.debug(f"Found {len(documents)} documents matching the query")
            return documents

        self.logger.debug("Full-text search retriever initialized")
        return retriever

    def get_vector_retriever_tool(self):
        # Create tool-specific logger with tool name
        tool_logger_name = f"{__name__}.{self.__class__.__name__}.{self.name}"
        tool_logger = get_logger(tool_logger_name)
        tool_logger.info(f"Creating vector retriever tool: {self.name}")

        vector_retriever = self._init_vector_retriever()

        @tool
        def vector_retriever_tool(search_query: str) -> str:
            """
            Retrieve relevant documents and their information from a vector store based on provided search query.
            Args:
                search_query (str): The query to search for relevant documents.
            Returns:
                str: A formatted string containing the retrieved documents and their sources.
            """
            tool_logger.info(f"Tool {self.name}: Retrieving documents for query")
            tool_logger.debug(f"Tool {self.name}: Query: {search_query}")
            try:
                results = vector_retriever.invoke(search_query)
            except Exception as e:
                # A genuine retrieval error: surface it as a ToolException so the
                # agent framework can handle it, instead of masking it as data.
                tool_logger.exception(f"Tool {self.name}: Error during retrieval: {e}")
                raise ToolException(f"Vector retrieval failed: {e}") from e

            if not results:
                tool_logger.warning(f"Tool {self.name}: No results found for the query")
                return "No results found for the query."

            tool_logger.info(
                f"Tool {self.name}: Found {len(results)} relevant documents"
            )
            return "Retrieved Documents:\n\n" + "\n\n".join(
                f"text_{i}: {doc.page_content} \nsource_{i}: {doc.metadata.get('source', 'N/A')}"
                for i, doc in enumerate(results)
            )

        return vector_retriever_tool

    def get_full_text_search_tool(self):
        """
        Returns a tool that performs full-text search on the MongoDB collection.
        """
        # Create tool-specific logger with tool name
        tool_logger_name = f"{__name__}.{self.__class__.__name__}.{self.name}"
        tool_logger = get_logger(tool_logger_name)
        tool_logger.info(f"Creating full-text search tool: {self.name}")
        full_text_search_retriever = self._init_full_text_retriever()

        @tool
        def full_text_search_tool(query: str) -> str:
            """
            Perform a full-text search on the MongoDB collection.
            Args:
                query (str): The search query.
            Returns:
                str: The results of the full-text search.
            """
            tool_logger.info(f"Tool {self.name}: Performing full-text search")
            tool_logger.debug(f"Tool {self.name}: Query: {query}")
            if not query:
                tool_logger.warning(
                    f"Tool {self.name}: Empty query provided for full-text search"
                )
                return "Empty query provided for full-text search."
            try:
                tool_logger.info(
                    f"Tool {self.name}: Executing full-text search \t Query: {query}"
                )
                results = full_text_search_retriever(query)
                tool_logger.info(
                    f"Tool {self.name}: Full-text search returned {len(results)} results"
                )
                if not results:
                    tool_logger.warning(
                        f"Tool {self.name}: No results found for the query"
                    )
                    return "No results found."

                context = "Search Results:\n\n" + "\n\n".join(
                    f"text_{i}: {doc['page_content']} \nsource_{i}: {doc.get('source', 'N/A')}"
                    for i, doc in enumerate(results)
                )
                return context
            except Exception as e:
                tool_logger.error(
                    f"Tool {self.name}: Failed to perform full-text search"
                )
                tool_logger.exception(f"Tool {self.name}: Error during search: {e}")
                return "Full-text search failed."

        return full_text_search_tool

    def get_mdb_toolkit(self, llm):
        name = self.name
        toolkit_logger = get_logger(f"{__name__}.{self.__class__.__name__}.{name}")
        toolkit_logger.info(f"Creating MongoDB toolkit: {name}")
        if llm is None:
            toolkit_logger.error("LLM must be provided to create the toolkit")
            raise ValueError(
                "A language model (llm) must be provided to create the toolkit."
            )
        db = MongoDBDatabase(self.client, self.database_name)
        toolkit = MongoDBDatabaseToolkit(db=db, llm=llm)
        toolkit_logger.debug(
            "MongoDB toolkit created with database: %s", self.database_name
        )
        return toolkit.get_tools()

    def get_nl_to_mql_tool(self, llm):
        """
        Returns a tool that converts natural language to MongoDB queries (MQL) and executes them.
        Requires a language model (llm) as input.
        """
        # Create tool-specific logger with tool name
        name = self.name
        tool_logger_name = f"{__name__}.{self.__class__.__name__}.{name}"
        tool_logger = get_logger(tool_logger_name)
        tool_logger.info(f"Creating NL to MQL tool: {name}")

        if llm is None:
            tool_logger.error("LLM must be provided to create the NL to MQL tool")
            raise ValueError(
                "A language model (llm) must be provided to create the NL to MQL tool."
            )

        tools = self.get_mdb_toolkit(llm)
        system_message = MONGODB_AGENT_SYSTEM_PROMPT.format(top_k=self.top_k)

        tool_logger.debug("Creating React agent for NL to MQL conversion")
        tool_logger.debug(f"System message for agent: {system_message}")
        agent = create_react_agent(model=llm, tools=tools)

        @tool
        def nl_to_mql_tool(nl_query: str) -> str:
            """
            Convert a natural language query to MongoDB MQL and execute it.
            Args:
                nl_query (str): The user's natural language query.
            Returns:
                str: The result of the MongoDB query.
            """
            tool_logger.info(f"Tool {name}: Processing natural language query")
            tool_logger.debug(f"Tool {name}: NL Query: {nl_query}")

            try:
                events = agent.invoke({"messages": [("user", f"Input: {nl_query}")]})
                messages = events.get("messages", [])

                if messages:
                    tool_logger.info(f"Tool {name}: Successfully processed query")
                    return messages[-1].content

                tool_logger.warning(
                    f"Tool {name}: No response from agent after processing"
                )
                raise ToolException(
                    "No response from the agent after processing the query."
                )
            except Exception as e:
                tool_logger.error(
                    f"Tool {name}: Failed to convert NL to MQL or execute query"
                )
                tool_logger.exception(f"Tool {name}: Error: {e}")
                return "Natural language to MQL conversion or execution failed."

        return nl_to_mql_tool
