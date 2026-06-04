"""
Policy providers for access control and guardrail configuration.

Both ``StaticPolicyProvider`` and ``MongoDBPolicyProvider`` implement the
``BasePolicyAdapter`` interface so that the rest of the framework can depend on
the abstract contract rather than on concrete provider classes.
"""

from typing import Any, Dict, List, Optional

from agent_builder.core.interfaces import BasePolicyAdapter
from agent_builder.core.types import AccessPolicy, IdentityContext
from agent_builder.utils.logging_config import get_logger

logger = get_logger(__name__)


class StaticPolicyProvider(BasePolicyAdapter):
    """
    Simple policy provider used when MongoDB policy storage is disabled.

    Returns a fixed ``AccessPolicy`` built from a static default document,
    regardless of the requesting identity.
    """

    def __init__(self, default_policy: Optional[Dict[str, Any]] = None):
        self.default_policy = default_policy or {"permissions": ["*"]}

    def get_policy(self, identity: IdentityContext) -> AccessPolicy:
        return _policy_from_document(self.default_policy, identity)


class MongoDBPolicyProvider(BasePolicyAdapter):
    """
    Load tenant, role, or user policies from MongoDB.

    Queries the ``agent_policies`` collection for documents that match the
    requesting identity and merges them into a single ``AccessPolicy``.
    Falls back to the default policy when no documents are found.
    """

    def __init__(
        self,
        connection_str: str,
        db_name: str = "agent_control_plane",
        collection_name: str = "agent_policies",
        default_policy: Optional[Dict[str, Any]] = None,
    ):
        import certifi
        from pymongo import MongoClient

        self.client = MongoClient(connection_str, tlsCAFile=certifi.where())
        self.collection = self.client[db_name][collection_name]
        self.default_policy = default_policy or {"permissions": ["*"]}

    def get_policy(self, identity: IdentityContext) -> AccessPolicy:
        query = {
            "tenant_id": identity.tenant_id,
            "$or": [
                {"user_id": identity.user_id},
                {"role": {"$in": identity.roles}},
                {"scope": "tenant_default"},
            ],
        }
        docs = list(self.collection.find(query, {"_id": 0}))
        if not docs:
            logger.debug("No policy found for %s, using default policy", identity.user_id)
            return _policy_from_document(self.default_policy, identity)
        return _merge_policy_documents(docs, identity)


def _policy_from_document(doc: Dict[str, Any], identity: IdentityContext) -> AccessPolicy:
    return AccessPolicy(
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
        roles=identity.roles,
        permissions=list(doc.get("permissions", [])),
        denied_tools=list(doc.get("denied_tools", [])),
        blocked_topics=list(doc.get("blocked_topics", [])),
        retrieval_filters=dict(doc.get("retrieval_filters", {})),
        pii_redaction=bool(doc.get("pii_redaction", True)),
        prompt_injection_detection=bool(doc.get("prompt_injection_detection", True)),
    )


def _merge_policy_documents(docs: List[Dict[str, Any]], identity: IdentityContext) -> AccessPolicy:
    permissions = set()
    denied_tools = set()
    blocked_topics = set()
    retrieval_filters: Dict[str, Any] = {}
    pii_redaction = True
    prompt_injection_detection = True

    for doc in docs:
        permissions.update(doc.get("permissions", []))
        denied_tools.update(doc.get("denied_tools", []))
        blocked_topics.update(doc.get("blocked_topics", []))
        retrieval_filters.update(doc.get("retrieval_filters", {}))
        pii_redaction = pii_redaction and bool(doc.get("pii_redaction", True))
        prompt_injection_detection = prompt_injection_detection and bool(
            doc.get("prompt_injection_detection", True)
        )

    return AccessPolicy(
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
        roles=identity.roles,
        permissions=sorted(permissions),
        denied_tools=sorted(denied_tools),
        blocked_topics=sorted(blocked_topics),
        retrieval_filters=retrieval_filters,
        pii_redaction=pii_redaction,
        prompt_injection_detection=prompt_injection_detection,
    )
