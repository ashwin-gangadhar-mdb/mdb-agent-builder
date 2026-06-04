"""Shared framework-neutral types for identity, policies, and guardrails."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass
class IdentityContext:
    """Identity and tenancy context used for policy enforcement."""

    tenant_id: str = "default"
    user_id: str = "anonymous"
    roles: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "IdentityContext":
        identity = config.get("identity", {}) if config else {}
        roles = identity.get("roles") or config.get("roles") or []
        if isinstance(roles, str):
            roles = [roles]
        return cls(
            tenant_id=identity.get("tenant_id") or config.get("tenant_id") or "default",
            user_id=identity.get("user_id") or config.get("user_id") or "anonymous",
            roles=roles,
            attributes=identity.get("attributes", {}),
        )


@dataclass
class AccessPolicy:
    """Resolved access policy for a user, role, or tenant."""

    tenant_id: str = "default"
    user_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    denied_tools: List[str] = field(default_factory=list)
    blocked_topics: List[str] = field(default_factory=list)
    retrieval_filters: Dict[str, Any] = field(default_factory=dict)
    pii_redaction: bool = True
    prompt_injection_detection: bool = True

    def allows(self, permission: str) -> bool:
        return "*" in self.permissions or permission in self.permissions


@dataclass
class GuardrailDecision:
    """Decision returned by a guardrail stage."""

    allowed: bool
    stage: str
    reason: str = ""
    transformed_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """Audit event stored for governance and traceability."""

    event_type: str
    tenant_id: str
    user_id: str
    agent_id: Optional[str] = None
    thread_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
