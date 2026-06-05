"""
MongoDB Atlas Vector Search memory adapters.

Hierarchy
---------

::

    BaseMemoryAdapter          (core/interfaces.py – abstract)
      └── MongoDBMemoryAdapter          (concrete base – owns ALL shared code)
            ├── MongoDBEpisodicMemoryAdapter     (extends base, locks type tag)
            └── MongoDBObservationalMemoryAdapter (extends base, adds LLM extraction)

``MongoDBMemoryAdapter``
    The single concrete implementation that holds the MongoDB client, the
    Atlas Vector Search store, and the complete ``save_memory`` /
    ``search_memories`` logic.  Every subclass reuses this infrastructure
    without duplicating a single line of setup or I/O code.

``MongoDBEpisodicMemoryAdapter``
    Extends ``MongoDBMemoryAdapter``.  Overrides ``save_memory`` to always
    tag documents ``memory_type = "episodic"`` and overrides
    ``search_memories`` to always filter on that tag.  Adds the convenience
    helpers ``save_episode`` / ``search_episodes`` from
    ``BaseEpisodicMemoryAdapter``.

``MongoDBObservationalMemoryAdapter``
    Extends ``MongoDBMemoryAdapter``.  Overrides ``save_memory`` to tag
    ``memory_type = "observational"``, overrides ``search_memories`` to
    filter on that tag, and implements ``extract_and_save`` — the
    LLM-powered extraction pipeline that converts raw text into a set of
    concise, structured observations before persisting them.  The injected
    ``llm`` is the *only* additional dependency compared to the base class.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM

from agent_builder.core.interfaces import (
    BaseEpisodicMemoryAdapter,
    BaseMemoryAdapter,
    BaseObservationalMemoryAdapter,
)
from agent_builder.core.types import utc_now
from agent_builder.utils.logging_config import get_logger, sanitize_connection_string

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default LLM extraction prompt (used by ObservationalMemoryAdapter)
# ---------------------------------------------------------------------------

_DEFAULT_EXTRACTION_PROMPT = (
    "You are a memory distillation assistant. Given the following raw "
    "conversation excerpt, extract a concise list of key observations worth "
    "remembering. Each observation should be a single, self-contained sentence "
    "capturing a user preference, biographical fact, stated goal, or notable "
    "behavioural pattern. Output a JSON array of strings and nothing else.\n\n"
    "Conversation excerpt:\n{text}\n\nObservations (JSON array):"
)


# ---------------------------------------------------------------------------
# Shared document factory
# ---------------------------------------------------------------------------

def _make_document(
    text: str,
    tenant_id: str,
    user_id: str,
    agent_id: str,
    thread_id: Optional[str],
    memory_type: str,
    metadata: Optional[Dict[str, Any]],
):
    """Return a LangChain Document with standard memory metadata fields."""
    from langchain_core.documents import Document

    return Document(
        page_content=text,
        metadata={
            "tenant_id": tenant_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "thread_id": thread_id,
            "memory_type": memory_type,
            "created_at": utc_now().isoformat(),
            **(metadata or {}),
        },
    )


# ---------------------------------------------------------------------------
# Concrete base — owns ALL shared infrastructure
# ---------------------------------------------------------------------------

class MongoDBMemoryAdapter(BaseMemoryAdapter):
    """
    Concrete base memory adapter backed by MongoDB Atlas Vector Search.

    This class owns **all** shared infrastructure:

    * MongoDB client connection
    * ``MongoDBAtlasVectorSearch`` store construction
    * ``save_memory`` — builds the document and calls ``add_documents``
    * ``search_memories`` — builds the pre-filter and calls ``similarity_search``

    Subclasses inherit all of the above for free and only override the pieces
    that differ for their specific memory strategy (type tag, extra LLM step,
    etc.).  There is no code duplication across the hierarchy.

    This class can also be used directly when the caller wants to control
    ``memory_type`` at call time (i.e. the "general" use-case from the
    factory).

    Configuration example (YAML)::

        memory:
          - name: general_store
            memory_type: general
            connection_str: ${MONGODB_URI}
            namespace: mydb.memories
            embedding_model: my_embedding
            index_name: memory_index
    """

    def __init__(
        self,
        connection_str: str,
        namespace: str,
        embedding_model: Embeddings,
        index_name: str = "agent_memory_index",
        embedding_field: str = "embedding",
        text_field: str = "text",
        **kwargs: Any,
    ) -> None:
        """
        Initialise the shared MongoDB + Atlas Vector Search infrastructure.

        Args:
            connection_str:  MongoDB Atlas connection URI.
            namespace:       ``"database.collection"`` pair.
            embedding_model: LangChain ``Embeddings`` instance used to embed
                             documents and query vectors.
            index_name:      Atlas Vector Search index name on the collection.
            embedding_field: MongoDB field name for the stored vector.
            text_field:      MongoDB field name for the stored text content.
            **kwargs:        Absorbed by subclasses; ignored here so the
                             constructor signature stays stable across the
                             hierarchy.
        """
        import certifi
        from langchain_mongodb import MongoDBAtlasVectorSearch
        from pymongo import MongoClient

        self.connection_str = connection_str
        self.namespace = namespace
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.embedding_field = embedding_field
        self.text_field = text_field

        self.client = MongoClient(connection_str, tlsCAFile=certifi.where())
        self.vector_store = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string=connection_str,
            namespace=namespace,
            embedding=embedding_model,
            embedding_field=embedding_field,
            index_name=index_name,
            text_field=text_field,
        )
        logger.info(
            "%s initialised — namespace='%s', index='%s'",
            self.__class__.__name__, namespace, index_name,
        )

    # ------------------------------------------------------------------
    # BaseMemoryAdapter contract  (fully implemented here, once)
    # ------------------------------------------------------------------

    def save_memory(
        self,
        text: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        thread_id: Optional[str] = None,
        memory_type: str = BaseMemoryAdapter.MEMORY_TYPE_EPISODIC,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Persist *text* as a single vector-search document.

        The ``memory_type`` argument controls the ``memory_type`` metadata
        field stored on the document.  Subclasses override this method to
        enforce a fixed type tag; callers of this base class may pass any
        string.

        Returns:
            List of document IDs assigned by MongoDB.
        """
        doc = _make_document(
            text, tenant_id, user_id, agent_id, thread_id, memory_type, metadata
        )
        ids = self.vector_store.add_documents([doc])
        logger.debug(
            "%s.save_memory: type='%s', user='%s', ids=%s",
            self.__class__.__name__, memory_type, user_id, ids,
        )
        return ids

    def search_memories(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        agent_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Any]:
        """
        Retrieve the *top_k* most relevant memory documents for *query*.

        Applies a mandatory ``tenant_id`` + ``user_id`` pre-filter so that
        cross-tenant data leakage is structurally impossible.  Callers may
        pass additional ``filters`` to further narrow the result set
        (e.g. ``{"memory_type": "episodic"}``).

        Returns:
            List of LangChain ``Document`` objects.
        """
        pre_filter: Dict[str, Any] = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            **(filters or {}),
        }
        if agent_id:
            pre_filter["agent_id"] = agent_id

        results = self.vector_store.similarity_search(
            query, k=top_k, pre_filter=pre_filter
        )
        logger.debug(
            "%s.search_memories: returned %d docs for user='%s'",
            self.__class__.__name__, len(results), user_id,
        )
        return results


# ---------------------------------------------------------------------------
# Episodic memory adapter — extends base, locks the type tag
# ---------------------------------------------------------------------------

class MongoDBEpisodicMemoryAdapter(MongoDBMemoryAdapter, BaseEpisodicMemoryAdapter):
    """
    Episodic memory adapter backed by MongoDB Atlas.

    Extends ``MongoDBMemoryAdapter`` — inherits the MongoDB client, the vector
    store, and the base ``save_memory`` / ``search_memories`` implementation.

    **What this class adds / overrides:**

    * ``save_memory`` — always tags stored documents
      ``memory_type = "episodic"``; ignores the caller-supplied type.
    * ``search_memories`` — automatically adds
      ``{"memory_type": "episodic"}`` to the pre-filter so searches are
      scoped to episodic memories only.
    * ``save_episode`` / ``search_episodes`` — convenience methods
      (provided by ``BaseEpisodicMemoryAdapter``; no override needed).

    Episodic memories are verbatim fragments of conversational experience —
    what the user said, what happened, how they felt — stored exactly as
    provided and recalled by semantic similarity.

    Configuration example (YAML)::

        memory:
          - name: recall
            memory_type: episodic
            connection_str: ${MONGODB_URI}
            namespace: mydb.episodic_memories
            embedding_model: my_embedding
            index_name: episodic_index
    """

    # No __init__ override needed — MongoDBMemoryAdapter.__init__ is reused
    # via normal MRO.  The default index_name is changed only so that the
    # Atlas index for this adapter doesn't collide with the general-purpose one.

    def __init__(
        self,
        connection_str: str,
        namespace: str,
        embedding_model: Embeddings,
        index_name: str = "episodic_memory_index",
        embedding_field: str = "embedding",
        text_field: str = "text",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            connection_str=connection_str,
            namespace=namespace,
            embedding_model=embedding_model,
            index_name=index_name,
            embedding_field=embedding_field,
            text_field=text_field,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Overrides — enforce the episodic type tag
    # ------------------------------------------------------------------

    def save_memory(
        self,
        text: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        thread_id: Optional[str] = None,
        memory_type: str = BaseMemoryAdapter.MEMORY_TYPE_EPISODIC,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Persist *text* as an episodic memory fragment.

        The ``memory_type`` argument is accepted for interface compliance but
        is always overridden to ``"episodic"``.
        """
        return super().save_memory(
            text=text,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            thread_id=thread_id,
            memory_type=self.MEMORY_TYPE_EPISODIC,
            metadata=metadata,
        )

    def search_memories(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        agent_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Any]:
        """
        Search episodic memories.

        Merges ``{"memory_type": "episodic"}`` into *filters* before
        delegating to the base implementation.
        """
        episodic_filters = {"memory_type": self.MEMORY_TYPE_EPISODIC, **(filters or {})}
        return super().search_memories(
            query=query,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            filters=episodic_filters,
            top_k=top_k,
        )


# ---------------------------------------------------------------------------
# Observational memory adapter — extends base, adds LLM extraction
# ---------------------------------------------------------------------------

class MongoDBObservationalMemoryAdapter(
    MongoDBMemoryAdapter, BaseObservationalMemoryAdapter
):
    """
    Observational memory adapter backed by MongoDB Atlas.

    Extends ``MongoDBMemoryAdapter`` — inherits the MongoDB client, the vector
    store, and the base ``save_memory`` / ``search_memories`` implementation.

    **What this class adds / overrides:**

    * ``__init__`` — accepts an additional ``llm`` argument and an optional
      ``extraction_prompt`` template; passes remaining args to the base.
    * ``extract_and_save`` — the LLM-powered pipeline: prompts the LLM to
      decompose *raw_text* into a JSON array of concise observations, then
      calls ``save_memory`` (inherited from the base) for each one.
    * ``save_memory`` — always tags stored documents
      ``memory_type = "observational"``.
    * ``search_memories`` — automatically scopes searches to observational
      documents.
    * ``search_observations`` — convenience method (from
      ``BaseObservationalMemoryAdapter``; no override needed).

    Observational memories are LLM-distilled inferences — structured facts,
    user preferences, recurring patterns — rather than verbatim quotes.  The
    LLM decides what is worth retaining and how to express it concisely.

    Configuration example (YAML)::

        memory:
          - name: observations
            memory_type: observational
            connection_str: ${MONGODB_URI}
            namespace: mydb.observational_memories
            embedding_model: my_embedding
            llm: my_llm
            index_name: observational_index
            extraction_prompt: |        # optional — overrides default
              Extract key facts: {text}
    """

    def __init__(
        self,
        connection_str: str,
        namespace: str,
        embedding_model: Embeddings,
        llm: BaseLLM,
        index_name: str = "observational_memory_index",
        embedding_field: str = "embedding",
        text_field: str = "text",
        extraction_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            llm:               LangChain LLM used to extract observations.
                               **Required** — there is no default.
            extraction_prompt: Optional prompt template.  Must contain a
                               ``{text}`` placeholder.  Defaults to a built-in
                               prompt that asks the LLM for a JSON array of
                               concise observation strings.
            (all other args):  Forwarded to ``MongoDBMemoryAdapter.__init__``.
        """
        super().__init__(
            connection_str=connection_str,
            namespace=namespace,
            embedding_model=embedding_model,
            index_name=index_name,
            embedding_field=embedding_field,
            text_field=text_field,
            **kwargs,
        )
        self.llm = llm
        self.extraction_prompt = extraction_prompt or _DEFAULT_EXTRACTION_PROMPT

    # ------------------------------------------------------------------
    # BaseObservationalMemoryAdapter contract
    # ------------------------------------------------------------------

    def extract_and_save(
        self,
        raw_text: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Use the configured LLM to extract key observations from *raw_text*,
        then persist each one as an independent ``"observational"`` document.

        The LLM is prompted to return a JSON array of strings.  Storing each
        observation as a separate document makes retrieval granular — the most
        relevant individual fact is returned rather than a large composite blob.

        Falls back to storing *raw_text* verbatim as a single document if the
        LLM response cannot be parsed as a JSON array.

        Args:
            raw_text:  Raw conversation excerpt or any text to distil.
            tenant_id: Tenant owner of the resulting memories.
            user_id:   User associated with the memories.
            agent_id:  Agent that produced/observed this text.
            thread_id: Optional conversation thread identifier.
            metadata:  Optional extra metadata applied to every document.

        Returns:
            Flat list of document IDs for every persisted observation.
        """
        logger.info(
            "ObservationalAdapter: extracting observations for user='%s'", user_id
        )
        prompt = self.extraction_prompt.format(text=raw_text)

        try:
            raw_response = self.llm.invoke(prompt)
            response_text = (
                raw_response.content
                if hasattr(raw_response, "content")
                else str(raw_response)
            )
            observations: List[str] = json.loads(response_text)
            if not isinstance(observations, list):
                raise ValueError("LLM did not return a JSON array")
        except Exception as exc:
            logger.warning(
                "ObservationalAdapter: LLM extraction failed (%s); "
                "storing raw text as a single observation",
                exc,
            )
            observations = [raw_text]

        all_ids: List[str] = []
        for obs in observations:
            if not isinstance(obs, str) or not obs.strip():
                continue
            # Reuse the inherited save_memory (which enforces "observational" tag)
            ids = self.save_memory(
                text=obs.strip(),
                tenant_id=tenant_id,
                user_id=user_id,
                agent_id=agent_id,
                thread_id=thread_id,
                metadata=metadata,
            )
            all_ids.extend(ids)

        logger.debug(
            "ObservationalAdapter: persisted %d observations for user='%s'",
            len(all_ids), user_id,
        )
        return all_ids

    # ------------------------------------------------------------------
    # Overrides — enforce the observational type tag
    # ------------------------------------------------------------------

    def save_memory(
        self,
        text: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        thread_id: Optional[str] = None,
        memory_type: str = BaseMemoryAdapter.MEMORY_TYPE_OBSERVATIONAL,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Persist *text* as an observational memory fragment.

        The ``memory_type`` argument is accepted for interface compliance but
        is always overridden to ``"observational"``.  For the full LLM
        extraction pipeline use ``extract_and_save``.
        """
        return super().save_memory(
            text=text,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            thread_id=thread_id,
            memory_type=self.MEMORY_TYPE_OBSERVATIONAL,
            metadata=metadata,
        )

    def search_memories(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        agent_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Any]:
        """
        Search observational memories.

        Merges ``{"memory_type": "observational"}`` into *filters* before
        delegating to the base implementation.
        """
        observational_filters = {
            "memory_type": self.MEMORY_TYPE_OBSERVATIONAL,
            **(filters or {}),
        }
        return super().search_memories(
            query=query,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            filters=observational_filters,
            top_k=top_k,
        )
