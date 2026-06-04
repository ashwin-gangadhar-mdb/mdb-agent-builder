"""
Long-term memory agent.

Two invocation modes are supported:

1. **Adapter-powered (preferred)**
   Pass one or both of ``episodic_memory`` and ``observational_memory``
   (instances of ``BaseEpisodicMemoryAdapter`` /
   ``BaseObservationalMemoryAdapter``).  The agent uses the injected adapters
   for all memory I/O — no hard-coded embedding model, no hard-coded MongoDB
   collection wiring.

   * ``episodic_memory`` backs the ``save_recall_memory`` tool: the LLM stores
     verbatim conversation fragments.
   * ``observational_memory`` backs the ``save_observation`` tool: the LLM
     triggers LLM-powered extraction of structured facts from raw text.  Both
     tools are only registered when the corresponding adapter is supplied.

2. **Legacy mode (backward-compatible)**
   Pass ``connection_str`` + ``namespace`` without memory adapters.  The agent
   falls back to the original hard-coded ``HuggingFaceEmbeddings`` path so
   that existing YAML files continue to work without modification.

Both modes share the same graph topology (``load_memories → agent ↔ tools``),
prompt, and node implementations; only the memory I/O differs.
"""

import uuid
from typing import Callable, List, Optional

import certifi
import tiktoken
from langchain_core.documents import Document
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agent_builder.core.interfaces import (
    BaseEpisodicMemoryAdapter,
    BaseObservationalMemoryAdapter,
)
from agent_builder.utils.logging_config import get_logger

logger = get_logger(__name__)

# Max tokens of recent conversation used to query the recall store.
_RECALL_QUERY_TOKEN_LIMIT = 2048

# Generic system prompt shared by both modes.  It intentionally refers to "the
# available memory tools" rather than naming specific tools, because the set of
# registered tools differs between adapter and legacy modes (and between
# episodic-only vs observational configurations).
_MEMORY_SYSTEM_PROMPT = (
    "You are a helpful assistant with advanced long-term memory "
    "capabilities. Powered by a stateless LLM, you must rely on external "
    "memory to store information between conversations. Utilize the available "
    "memory tools to store and retrieve important details that will help you "
    "better attend to the user's needs and understand their context.\n\n"
    "Memory Usage Guidelines:\n"
    "1. Actively use the available memory tools to build a comprehensive "
    "understanding of the user.\n"
    "2. Store verbatim fragments — exact quotes, stated preferences, specific "
    "events — when a tool for that is available.\n"
    "3. Store higher-level patterns and inferences — personality traits, "
    "recurring themes, goals — when a tool for that is available.\n"
    "4. Make informed suppositions and extrapolations based on stored "
    "memories.\n"
    "5. Regularly reflect on past interactions to identify patterns and "
    "preferences.\n"
    "6. Update your mental model of the user with each new piece of "
    "information.\n"
    "7. Cross-reference new information with existing memories for "
    "consistency.\n"
    "8. Prioritize storing emotional context and personal values alongside "
    "facts.\n"
    "9. Use memory to anticipate needs and tailor responses to the user's "
    "style.\n"
    "10. Recognize and acknowledge changes in the user's situation or "
    "perspectives over time.\n\n"
    "## Recall Memories\n"
    "Contextually retrieved memories from previous conversations:\n"
    "{recall_memories}\n\n"
    "## Instructions\n"
    "Engage with the user naturally, as a trusted colleague or friend. "
    "There's no need to explicitly mention your memory capabilities. Instead, "
    "seamlessly incorporate your understanding of the user into your "
    "responses. Be attentive to subtle cues and underlying emotions. Adapt "
    "your communication style to match the user's preferences and current "
    "emotional state. Use tools to persist information you want to retain in "
    "the next conversation. If you do call tools, all text preceding the tool "
    "call is an internal message. Respond AFTER calling the tool, once you "
    "have confirmation that the tool completed successfully.\n\n"
)


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def create_long_term_memory_agent(
    model,
    # ── adapter-powered path ──────────────────────────────────────────────
    episodic_memory: Optional[BaseEpisodicMemoryAdapter] = None,
    observational_memory: Optional[BaseObservationalMemoryAdapter] = None,
    # ── legacy path (kept for backward-compatibility) ─────────────────────
    connection_str: Optional[str] = None,
    namespace: Optional[str] = None,
    # ── common ────────────────────────────────────────────────────────────
    tools: Optional[List] = None,
    checkpointer=None,
    name: str = "long_term_memory_agent",
):
    """
    Create a long-term memory agent with MongoDB Atlas Vector Search.

    Adapter-powered invocation (preferred)::

        agent = create_long_term_memory_agent(
            model=llm,
            episodic_memory=MongoDBEpisodicMemoryAdapter(...),
            observational_memory=MongoDBObservationalMemoryAdapter(...),
        )

    Legacy invocation (backward-compatible)::

        agent = create_long_term_memory_agent(
            model=llm,
            connection_str="mongodb+srv://...",
            namespace="db.collection",
        )

    Returns:
        A compiled LangGraph ``StateGraph``.
    """
    if checkpointer is None:
        checkpointer = InMemorySaver()

    if not model:
        raise ValueError("Language model (model) is required.")

    logger.info("Creating long-term memory agent: '%s'", name)

    # Decide which memory backend(s) to use
    _use_adapters = episodic_memory is not None or observational_memory is not None
    _use_legacy = (not _use_adapters) and connection_str and namespace

    if not _use_adapters and not _use_legacy:
        raise ValueError(
            "Provide either (episodic_memory / observational_memory) adapters "
            "or (connection_str + namespace) for legacy mode."
        )

    if _use_legacy:
        return _create_legacy_agent(
            model=model,
            connection_str=connection_str,
            namespace=namespace,
            tools=tools,
            checkpointer=checkpointer,
            name=name,
        )

    return _create_adapter_agent(
        model=model,
        episodic_memory=episodic_memory,
        observational_memory=observational_memory,
        tools=tools,
        checkpointer=checkpointer,
        name=name,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _MemoryState(MessagesState):
    recall_memories: List[str]


def _truncate_for_recall(text: str) -> str:
    """Trim *text* to the recall-query token budget using the cl100k encoding."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) > _RECALL_QUERY_TOKEN_LIMIT:
        tokens = tokens[:_RECALL_QUERY_TOKEN_LIMIT]
    return encoding.decode(tokens)


def _build_memory_agent_graph(
    model,
    all_tools: List,
    load_memories: Callable,
    checkpointer,
):
    """Assemble the shared ``load_memories → agent ↔ tools`` graph.

    ``all_tools`` are bound to the model **once** here rather than on every
    ``agent`` invocation.  ``load_memories`` is a mode-specific callable with
    signature ``(state, config) -> {"recall_memories": [...]}``.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", _MEMORY_SYSTEM_PROMPT), ("placeholder", "{messages}")]
    )
    chain = prompt | model.bind_tools(all_tools)

    def agent(state: _MemoryState) -> _MemoryState:
        recall_str = (
            "<recall_memory>\n"
            + "\n".join(state.get("recall_memories", []))
            + "\n</recall_memory>"
        )
        prediction = chain.invoke(
            {"messages": state["messages"], "recall_memories": recall_str}
        )
        return {"messages": [prediction]}

    def route_tools(state: _MemoryState):
        msg = state["messages"][-1]
        return "tools" if getattr(msg, "tool_calls", None) else END

    builder = StateGraph(_MemoryState)
    builder.add_node("load_memories", load_memories)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", "agent")
    builder.add_conditional_edges("agent", route_tools, ["tools", END])
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)


def _configurable(config: RunnableConfig) -> dict:
    return config.get("configurable") or {}


# ---------------------------------------------------------------------------
# Adapter-powered implementation
# ---------------------------------------------------------------------------

def _create_adapter_agent(
    model,
    episodic_memory: Optional[BaseEpisodicMemoryAdapter],
    observational_memory: Optional[BaseObservationalMemoryAdapter],
    tools: Optional[List],
    checkpointer,
    name: str,
):
    """Build the agent graph using injected memory adapters."""

    logger.info(
        "Agent '%s': adapter mode — episodic=%s, observational=%s",
        name,
        episodic_memory.__class__.__name__ if episodic_memory else "None",
        observational_memory.__class__.__name__ if observational_memory else "None",
    )

    def _require_user_id(config: RunnableConfig) -> str:
        user_id = _configurable(config).get("user_id")
        if user_id is None:
            raise ValueError(
                "user_id must be provided in the run config's 'configurable' dict."
            )
        return user_id

    def _ids(config: RunnableConfig):
        cfg = _configurable(config)
        return (
            _require_user_id(config),
            cfg.get("agent_id", name),
            cfg.get("tenant_id", "default"),
        )

    memory_tools: List = []

    if episodic_memory is not None:
        _ep_mem = episodic_memory  # capture for closure

        @tool
        def save_recall_memory(memory: str, config: RunnableConfig) -> str:
            """
            Save a verbatim memory fragment to the episodic store for later
            semantic retrieval.  Call this whenever the user shares a
            preference, biographical detail, or anything worth remembering.
            """
            user_id, agent_id, tenant_id = _ids(config)
            logger.info("Agent '%s': saving episodic memory for user '%s'", name, user_id)
            _ep_mem.save_episode(
                text=memory, tenant_id=tenant_id, user_id=user_id, agent_id=agent_id
            )
            return memory

        @tool
        def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
            """Search the episodic store for memories relevant to *query*."""
            user_id, agent_id, tenant_id = _ids(config)
            logger.info(
                "Agent '%s': searching episodic memories for user '%s'", name, user_id
            )
            try:
                docs = _ep_mem.search_episodes(
                    query=query, tenant_id=tenant_id, user_id=user_id, agent_id=agent_id
                )
                return [d.page_content for d in docs]
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Agent '%s': episodic search failed — %s", name, exc)
                return []

        memory_tools += [save_recall_memory, search_recall_memories]

    if observational_memory is not None:
        _ob_mem = observational_memory  # capture for closure

        @tool
        def save_observation(raw_text: str, config: RunnableConfig) -> str:
            """
            Use the configured LLM to extract key observations from *raw_text*
            and persist them to the observational memory store.  Call this when
            you notice patterns, preferences, or facts that are better expressed
            as structured observations rather than verbatim quotes.
            """
            user_id, agent_id, tenant_id = _ids(config)
            logger.info(
                "Agent '%s': extracting and saving observations for user '%s'",
                name, user_id,
            )
            ids = _ob_mem.extract_and_save(
                raw_text=raw_text, tenant_id=tenant_id, user_id=user_id, agent_id=agent_id
            )
            return f"Saved {len(ids)} observation(s)."

        memory_tools.append(save_observation)

    all_tools = memory_tools + (tools or [])

    def load_memories(state: _MemoryState, config: RunnableConfig) -> _MemoryState:
        if episodic_memory is None:
            return {"recall_memories": []}
        cfg = _configurable(config)
        convo_str = _truncate_for_recall(get_buffer_string(state["messages"]))
        try:
            docs = episodic_memory.search_episodes(
                query=convo_str,
                tenant_id=cfg.get("tenant_id", "default"),
                user_id=cfg.get("user_id", "anonymous"),
                agent_id=cfg.get("agent_id", name),
            )
            return {"recall_memories": [d.page_content for d in docs]}
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Agent '%s': load_memories failed — %s", name, exc)
            return {"recall_memories": []}

    return _build_memory_agent_graph(model, all_tools, load_memories, checkpointer)


# ---------------------------------------------------------------------------
# Legacy implementation (HuggingFace + MongoDBAtlasVectorSearch)
# ---------------------------------------------------------------------------

def _ensure_recall_index(vector_store, index_name: str, embedding_dim: int, name: str) -> None:
    """Best-effort, **non-blocking** creation of the Atlas vector-search index.

    Unlike the previous implementation this neither runs on every save nor busy
    -waits (``sleep``) for the index to become queryable — Atlas builds indexes
    asynchronously, so blocking a request thread for minutes was incorrect.
    """
    try:
        existing = list(vector_store.collection.list_search_indexes())
        if any(x["name"] == index_name for x in existing):
            return
        logger.info(
            "Agent '%s': creating recall memory index '%s' "
            "(Atlas builds it asynchronously; first searches may return no "
            "results until it is ready).",
            name, index_name,
        )
        vector_store.create_vector_search_index(
            dimensions=embedding_dim,
            filters=[{"type": "filter", "path": "user_id"}],
            update=True,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "Agent '%s': could not ensure recall index '%s': %s", name, index_name, exc
        )


def _create_legacy_agent(
    model,
    connection_str: str,
    namespace: str,
    tools: Optional[List],
    checkpointer,
    name: str,
):
    """
    Original hard-coded HuggingFace + MongoDBAtlasVectorSearch implementation.
    Preserved so that existing YAML files with ``agent_type: long_term_memory``
    and no ``memory:`` section continue to work without modification.
    """
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    from langchain_mongodb import MongoDBAtlasVectorSearch

    logger.info(
        "Agent '%s': legacy mode (connection_str + namespace). "
        "Consider migrating to the adapter-powered mode by adding a "
        "'memory:' section to your YAML configuration.",
        name,
    )

    index_name = "recall_memory_index"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    recall_vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=connection_str,
        namespace=namespace,
        embedding=embeddings,
        embedding_field="embedding",
        index_name=index_name,
        text_field="text",
    )
    embedding_dim = len(embeddings.embed_query("this is a test"))
    logger.debug("Agent '%s': embedding dimension: %d", name, embedding_dim)

    # Ensure the index exists once, up front — not on every save call.
    _ensure_recall_index(recall_vector_store, index_name, embedding_dim, name)

    def get_user_id(config: RunnableConfig) -> str:
        user_id = _configurable(config).get("user_id")
        if user_id is None:
            raise ValueError("User ID needs to be provided to save a memory.")
        return user_id

    @tool
    def save_recall_memory(memory: str, config: RunnableConfig) -> str:
        """Save memory to vectorstore for later semantic retrieval."""
        user_id = get_user_id(config)
        logger.info("Agent '%s': saving recall memory for user '%s'", name, user_id)
        document = Document(
            page_content=memory,
            id=str(uuid.uuid4()),
            metadata={"user_id": user_id},
        )
        recall_vector_store.add_documents([document])
        return memory

    @tool
    def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
        """Search for relevant memories."""
        user_id = get_user_id(config)
        logger.info(
            "Agent '%s': searching recall memories for user '%s'", name, user_id
        )
        try:
            documents = recall_vector_store.similarity_search(
                query, k=3, pre_filter={"user_id": user_id}
            )
            return [document.page_content for document in documents]
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Agent '%s': error during memory search: %s", name, exc)
            return ["No relevant memories found."]

    all_tools = (tools or []) + [save_recall_memory, search_recall_memories]

    def load_memories(state: _MemoryState, config: RunnableConfig) -> _MemoryState:
        convo_str = _truncate_for_recall(get_buffer_string(state["messages"]))
        recall_memories = search_recall_memories.invoke(convo_str, config)
        return {"recall_memories": recall_memories}

    return _build_memory_agent_graph(model, all_tools, load_memories, checkpointer)
