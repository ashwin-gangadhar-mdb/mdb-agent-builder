"""
MongoDB-backed thread / session state provider.

``MongoDBStateProvider`` implements the ``BaseStateAdapter`` interface so
that the rest of the framework can depend on the abstract contract rather than
on this specific MongoDB implementation.
"""

from typing import Any, Dict, Optional

from agent_builder.core.interfaces import BaseStateAdapter
from agent_builder.core.types import utc_now


class MongoDBStateProvider(BaseStateAdapter):
    """
    Persist generic agent thread state in MongoDB.

    Implements ``BaseStateAdapter`` — use the interface type annotation when
    referencing state providers from other modules to stay backend-agnostic.
    """

    def __init__(
        self,
        connection_str: str,
        db_name: str = "agent_control_plane",
        collection_name: str = "agent_sessions",
    ):
        import certifi
        from pymongo import MongoClient

        self.client = MongoClient(connection_str, tlsCAFile=certifi.where())
        self.collection = self.client[db_name][collection_name]

    def load_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Load the persisted state for a thread from MongoDB."""
        return self.collection.find_one({"thread_id": thread_id}, {"_id": 0})

    def save_thread(
        self,
        thread_id: str,
        state: Dict[str, Any],
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> Dict[str, Any]:
        """Persist or update the state for a thread using an upsert."""
        now = utc_now()
        from pymongo import ReturnDocument

        doc = {
            "thread_id": thread_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "state": state,
            "updated_at": now,
        }
        return self.collection.find_one_and_update(
            {"thread_id": thread_id},
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
            projection={"_id": 0},
        )
