"""
MongoDB-backed audit / event-sink provider.

``MongoDBAuditProvider`` implements the ``BaseAuditAdapter`` interface so
that the rest of the framework can depend on the abstract contract rather than
on this specific MongoDB implementation.
"""

from typing import Any, Dict, Optional

from agent_builder.core.interfaces import BaseAuditAdapter
from agent_builder.core.types import AuditEvent, utc_now
from agent_builder.utils.logging_config import get_logger, sanitize_connection_string

logger = get_logger(__name__)


class MongoDBAuditProvider(BaseAuditAdapter):
    """
    Persist governance and runtime audit events to MongoDB.

    Implements ``BaseAuditAdapter`` — use the interface type annotation when
    referencing audit providers from other modules to stay backend-agnostic.
    """

    def __init__(
        self,
        connection_str: str,
        db_name: str = "agent_control_plane",
        collection_name: str = "agent_audit_events",
    ):
        import certifi
        from pymongo import MongoClient

        self.client = MongoClient(connection_str, tlsCAFile=certifi.where())
        logger.info(
            "MongoDB audit provider initialised — db='%s', collection='%s', conn='%s'",
            db_name, collection_name, sanitize_connection_string(connection_str),
        )
        self.collection = self.client[db_name][collection_name]

    def record(self, event: AuditEvent) -> None:
        """Persist a structured ``AuditEvent`` document to the audit collection."""
        doc = {
            "event_type": event.event_type,
            "tenant_id": event.tenant_id,
            "user_id": event.user_id,
            "agent_id": event.agent_id,
            "thread_id": event.thread_id,
            "payload": event.payload,
            "created_at": event.created_at,
        }
        self.collection.insert_one(doc)

    def record_raw(
        self,
        event_type: str,
        tenant_id: str,
        user_id: str,
        payload: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> None:
        """Convenience method to record an event from raw fields."""
        self.record(
            AuditEvent(
                event_type=event_type,
                tenant_id=tenant_id,
                user_id=user_id,
                agent_id=agent_id,
                thread_id=thread_id,
                payload=payload or {},
                created_at=utc_now(),
            )
        )
