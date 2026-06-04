"""
Abstract base classes (target interfaces) for the adapter design pattern.

Each interface defines the contract that all concrete adapters must fulfil,
decoupling the rest of the framework from specific third-party libraries or
storage backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from langchain_core.tools import BaseTool

from agent_builder.core.types import AccessPolicy, AuditEvent, IdentityContext


# ---------------------------------------------------------------------------
# LLM provider interface
# ---------------------------------------------------------------------------

class BaseLLMAdapter(ABC):
    """
    Target interface for LLM providers.

    All LLM adapters must be able to return a LangChain-compatible
    ``BaseLLM`` / ``BaseChatModel`` instance so that agent code remains
    provider-agnostic.
    """

    @abstractmethod
    def get_llm(self) -> BaseLLM:
        """Return an initialised LangChain LLM / chat-model instance."""


# ---------------------------------------------------------------------------
# Embedding provider interface
# ---------------------------------------------------------------------------

class BaseEmbeddingAdapter(ABC):
    """
    Target interface for embedding-model providers.

    All embedding adapters must return a LangChain-compatible
    ``Embeddings`` instance.
    """

    @abstractmethod
    def get_embedding_model(self) -> Embeddings:
        """Return an initialised LangChain Embeddings instance."""


# ---------------------------------------------------------------------------
# Tool provider interface
# ---------------------------------------------------------------------------

class BaseToolAdapter(ABC):
    """
    Target interface for tool providers.

    A tool adapter may expose a single tool or a list of tools (e.g. a
    toolkit).  The framework calls ``get_tools()`` and flattens the result.
    """

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Return a list of initialised LangChain BaseTool instances."""


# ---------------------------------------------------------------------------
# Policy provider interface
# ---------------------------------------------------------------------------

class BasePolicyAdapter(ABC):
    """
    Target interface for access-policy providers.

    Policy adapters resolve an ``IdentityContext`` to an ``AccessPolicy``
    that the guardrail engine and other framework components consume.
    """

    @abstractmethod
    def get_policy(self, identity: IdentityContext) -> AccessPolicy:
        """
        Resolve and return the access policy for the given identity.

        Args:
            identity: The tenant / user identity whose policy is requested.

        Returns:
            An ``AccessPolicy`` for the identity.
        """


# ---------------------------------------------------------------------------
# Audit provider interface
# ---------------------------------------------------------------------------

class BaseAuditAdapter(ABC):
    """
    Target interface for audit / event-sink providers.

    Audit adapters persist governance and runtime events to a durable store.
    """

    @abstractmethod
    def record(self, event: AuditEvent) -> None:
        """
        Persist a structured audit event.

        Args:
            event: The ``AuditEvent`` to store.
        """

    @abstractmethod
    def record_raw(
        self,
        event_type: str,
        tenant_id: str,
        user_id: str,
        payload: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """
        Convenience method to record an audit event from raw fields.

        Args:
            event_type:  A short string classifying the event (e.g. ``"agent.chat.completed"``).
            tenant_id:   The tenant that owns this event.
            user_id:     The user that triggered this event.
            payload:     Optional arbitrary metadata dictionary.
            agent_id:    Optional agent identifier.
            thread_id:   Optional conversation thread identifier.
        """


# ---------------------------------------------------------------------------
# State / session provider interface
# ---------------------------------------------------------------------------

class BaseStateAdapter(ABC):
    """
    Target interface for thread-state / session providers.

    State adapters persist and retrieve conversation thread state so that
    it survives process restarts.
    """

    @abstractmethod
    def load_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Load the persisted state for a thread.

        Args:
            thread_id: Unique identifier of the conversation thread.

        Returns:
            A dictionary containing the thread state, or ``None`` if not found.
        """

    @abstractmethod
    def save_thread(
        self,
        thread_id: str,
        state: Dict[str, Any],
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """
        Persist or update the state for a thread.

        Args:
            thread_id: Unique identifier of the conversation thread.
            state:     The state dictionary to persist.
            tenant_id: Optional tenant identifier (default: ``"default"``).
            user_id:   Optional user identifier (default: ``"anonymous"``).

        Returns:
            The persisted document as a dictionary.
        """


# ---------------------------------------------------------------------------
# Memory provider interface (base + specialised variants)
# ---------------------------------------------------------------------------

class BaseMemoryAdapter(ABC):
    """
    Target interface for long-term memory providers.

    Memory adapters store and retrieve memories using a vector-search backend
    (MongoDB Atlas by default).  Two specialised sub-interfaces extend this
    contract for distinct memory strategies:

    * ``BaseEpisodicMemoryAdapter``   — stores verbatim conversational moments.
    * ``BaseObservationalMemoryAdapter`` — uses an LLM to distil raw text into
      structured observations before persisting.

    Use the most specific interface available when type-annotating references
    to memory providers; fall back to ``BaseMemoryAdapter`` when the caller
    is strategy-agnostic.
    """

    #: String constant identifying episodic memories in the data store.
    MEMORY_TYPE_EPISODIC: str = "episodic"
    #: String constant identifying observational memories in the data store.
    MEMORY_TYPE_OBSERVATIONAL: str = "observational"

    @abstractmethod
    def save_memory(
        self,
        text: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        thread_id: Optional[str] = None,
        memory_type: str = "episodic",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Persist a memory fragment.

        Args:
            text:        The text content of the memory.
            tenant_id:   Tenant that owns this memory.
            user_id:     User associated with this memory.
            agent_id:    Agent that produced this memory.
            thread_id:   Optional conversation thread identifier.
            memory_type: Category of the memory (``"episodic"`` or
                         ``"observational"``).
            metadata:    Optional extra metadata dictionary.

        Returns:
            A list of document identifiers for the stored memory fragments.
        """

    @abstractmethod
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
        Retrieve memories relevant to a query.

        Args:
            query:     The search query string.
            tenant_id: Tenant that owns the memories.
            user_id:   User associated with the memories.
            agent_id:  Optional filter by agent identifier.
            filters:   Optional additional metadata filters.
            top_k:     Maximum number of results to return (default: 5).

        Returns:
            A list of matching memory documents.
        """


class BaseEpisodicMemoryAdapter(BaseMemoryAdapter):
    """
    Sub-interface for **episodic** memory adapters.

    Episodic memories are verbatim conversational fragments — what was said,
    what happened, how the user felt.  They are stored exactly as provided and
    retrieved by semantic similarity.

    Concrete implementations inherit ``save_memory`` and ``search_memories``
    from ``BaseMemoryAdapter`` (or any concrete base that satisfies that
    contract) and **must** ensure those methods enforce
    ``memory_type = "episodic"``.

    This interface adds two convenience methods that call through to the
    inherited abstract methods with the type tag already fixed:

    * ``save_episode``   — calls ``save_memory`` with ``memory_type="episodic"``
    * ``search_episodes``— calls ``search_memories`` filtered to episodic docs
    """

    def save_episode(
        self,
        text: str,
        tenant_id: str,
        user_id: str,
        agent_id: str,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Persist an episodic memory fragment (always tagged ``"episodic"``).

        Convenience wrapper around :meth:`save_memory`.
        """
        return self.save_memory(
            text=text,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            thread_id=thread_id,
            memory_type=self.MEMORY_TYPE_EPISODIC,
            metadata=metadata,
        )

    def search_episodes(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        agent_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Any]:
        """
        Search exclusively within episodic memories.

        Convenience wrapper around :meth:`search_memories` that injects
        ``{"memory_type": "episodic"}`` into the filter.
        """
        return self.search_memories(
            query=query,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            filters={"memory_type": self.MEMORY_TYPE_EPISODIC},
            top_k=top_k,
        )


class BaseObservationalMemoryAdapter(BaseMemoryAdapter):
    """
    Sub-interface for **observational** memory adapters.

    Observational memories are LLM-distilled inferences drawn from raw
    conversation text — structured facts, preferences, behavioural patterns —
    rather than verbatim quotes.

    Concrete implementations inherit ``save_memory`` and ``search_memories``
    from ``BaseMemoryAdapter`` (or any concrete base) and **must** ensure
    those methods enforce ``memory_type = "observational"``.

    This interface adds:

    * ``extract_and_save`` (abstract) — runs an LLM extraction step on raw
      text and then persists each resulting observation.
    * ``search_observations`` (concrete) — convenience wrapper around
      :meth:`search_memories` filtered to observational documents.
    """

    @abstractmethod
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
        Use the configured LLM to extract key observations from *raw_text*
        and persist each one as a separate ``"observational"`` document.

        Args:
            raw_text:  Raw conversation excerpt or document to distil.
            tenant_id: Tenant that owns these memories.
            user_id:   User associated with the memories.
            agent_id:  Agent that produced/observed this text.
            thread_id: Optional conversation thread identifier.
            metadata:  Optional extra metadata applied to every document.

        Returns:
            Flat list of document identifiers for every persisted observation.
        """

    def search_observations(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        agent_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Any]:
        """
        Search exclusively within observational memories.

        Convenience wrapper around :meth:`search_memories` that injects
        ``{"memory_type": "observational"}`` into the filter.
        """
        return self.search_memories(
            query=query,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id,
            filters={"memory_type": self.MEMORY_TYPE_OBSERVATIONAL},
            top_k=top_k,
        )
