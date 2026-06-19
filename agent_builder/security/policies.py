"""
Policy providers for access control and guardrail configuration.

Both ``StaticPolicyProvider`` and ``MongoDBPolicyProvider`` implement the
``BasePolicyAdapter`` interface so that the rest of the framework can depend on
the abstract contract rather than on concrete provider classes.
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from agent_builder.core.interfaces import BasePolicyAdapter
from agent_builder.core.types import AccessPolicy, IdentityContext
from agent_builder.utils.logging_config import get_logger, sanitize_connection_string

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


_CacheKey = Tuple[str, str, str]
_CacheEntry = Tuple[float, AccessPolicy]


class MongoDBPolicyProvider(BasePolicyAdapter):
    """
    Load tenant, role, or user policies from MongoDB.

    Queries the ``agent_policies`` collection for documents that match the
    requesting identity and merges them into a single ``AccessPolicy``.
    Falls back to the default policy when no documents are found.

    Caches resolved policies per identity with a short TTL to avoid the
    per-request MongoDB lookup.  The cache key is ``(tenant_id, user_id,
    sorted(roles))``; stale entries are evicted lazily on next read.
    """

    def __init__(
        self,
        connection_str: str,
        db_name: str = "agent_control_plane",
        collection_name: str = "agent_policies",
        default_policy: Optional[Dict[str, Any]] = None,
        cache_ttl: float = 5.0,
    ):
        import certifi
        from pymongo import MongoClient

        self.client = MongoClient(connection_str, tlsCAFile=certifi.where())
        logger.info(
            "MongoDB policy provider initialised — db='%s', collection='%s', conn='%s'",
            db_name, collection_name, sanitize_connection_string(connection_str),
        )
        self.collection = self.client[db_name][collection_name]
        self.default_policy = default_policy or {"permissions": ["*"]}
        self._cache: Dict[_CacheKey, _CacheEntry] = {}
        self._cache_ttl = cache_ttl

    def _cache_key(self, identity: IdentityContext) -> _CacheKey:
        return (
            identity.tenant_id,
            identity.user_id,
            ",".join(sorted(identity.roles)),
        )

    def get_policy(self, identity: IdentityContext) -> AccessPolicy:
        key = self._cache_key(identity)
        now = time.monotonic()
        entry = self._cache.get(key)
        if entry is not None:
            ts, policy = entry
            if now - ts < self._cache_ttl:
                return policy

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
            policy = _policy_from_document(self.default_policy, identity)
        else:
            policy = _merge_policy_documents(docs, identity)

        self._cache[key] = (now, policy)
        return policy


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
