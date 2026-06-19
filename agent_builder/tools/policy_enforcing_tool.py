"""
Per-tool policy-enforcing wrapper (Approach A).

Wraps every ``BaseTool`` at tool-assembly time so that policy enforcement
happens at tool-invoke time.  The wrapper reads the per-request
``AccessPolicy`` from ``config["configurable"]["policy"]``, runs
``check_tool`` (fail-closed, audited), and then delegates to the
wrapped tool.

For retrieval-bearing tools (``retrieval_aware = True``), the wrapper
also calls ``apply_retrieval_filters`` and passes the result as
the tool's ``pre_filter`` keyword argument.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from agent_builder.core.types import (
    AccessPolicy,
    GuardrailDecision,
    RetrievalFilterConflict,
)
from agent_builder.security.guardrails import GuardrailEngine

logger = logging.getLogger(__name__)

_REASON_CODES = {
    "MISSING_POLICY": (
        "Policy information is not available. "
        "Ensure governance is configured correctly."
    ),
    "MISSING_TENANT_ID": (
        "Tenant isolation requires a valid tenant ID."
    ),
    "PERMISSION_DENIED": "You do not have permission to use this tool.",
    "DENIED_TOOL": "Access to this tool is denied by your policy.",
    "RETRIEVAL_FILTER_CONFLICT": (
        "The retrieval filters from your policy conflict with "
        "the tool's configuration."
    ),
}

AuditSink = Callable[
    [str, Dict[str, Any]], None
]


class PolicyEnforcingTool(BaseTool):
    """
    Wraps a ``BaseTool`` to enforce ``AccessPolicy`` at invoke time.

    Policy is read from ``config["configurable"]["policy"]`` on every
    invocation.  The wrapper fails closed: if the policy is absent,
    resolution failed, or ``tenant_id`` is missing, the tool is denied
    and an audit event is emitted — the underlying tool never runs.

    For tools with ``retrieval_aware = True``, ``apply_retrieval_filters``
    is called and the merged filter is passed as a ``pre_filter`` kwarg
    so that downstream retrieval stores can apply tenant scoping.
    """

    name: str = ""
    description: str = ""

    _wrapped: BaseTool
    _guardrails: GuardrailEngine
    _audit_sink: Optional[AuditSink]
    _retrieval_aware: bool

    def __init__(
        self,
        wrapped: BaseTool,
        guardrails: GuardrailEngine,
        audit_sink: Optional[AuditSink] = None,
    ) -> None:
        super().__init__(
            name=wrapped.name,
            description=wrapped.description,
            return_direct=wrapped.return_direct,
            args_schema=wrapped.args_schema,
            response_format=wrapped.response_format,
            metadata=wrapped.metadata,
            tags=wrapped.tags,
        )
        self._wrapped = wrapped
        self._guardrails = guardrails
        self._audit_sink = audit_sink
        self._retrieval_aware = bool(getattr(wrapped, "retrieval_aware", False))

    def _resolve_policy(self, config: Optional[RunnableConfig]) -> AccessPolicy:
        raw = config.get("configurable", {}).get("policy") if config else None
        if raw is None:
            return AccessPolicy()
        if isinstance(raw, AccessPolicy):
            return raw
        if isinstance(raw, dict):
            return AccessPolicy(**raw)
        return AccessPolicy()

    def _deny(
        self,
        policy: AccessPolicy,
        reason_code: str,
        config: Optional[RunnableConfig] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        reason_text = _REASON_CODES.get(reason_code, reason_code)
        if self._audit_sink:
            payload: Dict[str, Any] = {
                "allowed": False,
                "tool": self.name,
                "reason_code": reason_code,
                "reason": reason_text,
                "tenant_id": policy.tenant_id,
                "user_id": policy.user_id or "",
            }
            if extra:
                payload.update(extra)
            try:
                self._audit_sink("guardrail.tool", payload)
            except Exception:
                logger.exception("Audit sink failed for tool deny event")
        return f"Tool '{self.name}' is unavailable: {reason_text}"

    def _run(
        self,
        *args: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        policy = self._resolve_policy(config)
        decision = self._guardrails.check_tool(self.name, policy)

        if not decision.allowed:
            reason_code = (
                "DENIED_TOOL"
                if self.name in policy.denied_tools
                else "PERMISSION_DENIED"
            )
            return self._deny(policy, reason_code, config)

        if self._audit_sink:
            try:
                self._audit_sink(
                    "guardrail.tool",
                    {
                        "allowed": True,
                        "tool": self.name,
                        "tenant_id": policy.tenant_id,
                        "user_id": policy.user_id or "",
                    },
                )
            except Exception:
                logger.exception("Audit sink failed for tool allow event")

        if self._retrieval_aware:
            return self._run_retrieval(policy, *args, config=config, **kwargs)

        return self._delegate_run(*args, config=config, **kwargs)

    async def _arun(
        self,
        *args: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        policy = self._resolve_policy(config)
        decision = self._guardrails.check_tool(self.name, policy)

        if not decision.allowed:
            reason_code = (
                "DENIED_TOOL"
                if self.name in policy.denied_tools
                else "PERMISSION_DENIED"
            )
            return self._deny(policy, reason_code, config)

        if self._audit_sink:
            try:
                self._audit_sink(
                    "guardrail.tool",
                    {
                        "allowed": True,
                        "tool": self.name,
                        "tenant_id": policy.tenant_id,
                        "user_id": policy.user_id or "",
                    },
                )
            except Exception:
                logger.exception("Audit sink failed for tool allow event")

        if self._retrieval_aware:
            return self._run_retrieval(policy, *args, config=config, **kwargs)

        return await self._delegate_arun(*args, config=config, **kwargs)

    def _run_retrieval(
        self,
        policy: AccessPolicy,
        *args: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Any:
        static_filter = kwargs.pop("pre_filter", None)
        try:
            merged = self._guardrails.apply_retrieval_filters(
                static_filter or {}, policy
            )
        except RetrievalFilterConflict as exc:
            return self._deny(
                policy,
                "RETRIEVAL_FILTER_CONFLICT",
                config,
                extra={"conflict": str(exc)},
            )

        return self._delegate_run(*args, config=config, pre_filter=merged, **kwargs)

    def _delegate_run(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self._wrapped, "_run"):
            return self._wrapped._run(*args, **kwargs)
        return self._wrapped.run(*args, **kwargs)

    async def _delegate_arun(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self._wrapped, "_arun"):
            return await self._wrapped._arun(*args, **kwargs)
        return await self._wrapped.arun(*args, **kwargs)


def wrap_tools(
    tools: List[BaseTool],
    guardrails: GuardrailEngine,
    audit_sink: Optional[AuditSink] = None,
) -> List[BaseTool]:
    """Wrap every tool in *tools* with ``PolicyEnforcingTool``.

    The audit sink callback receives ``(event_type, payload_dict)`` on
    every allow/deny decision.
    """
    wrapped: List[BaseTool] = []
    for tool in tools:
        wrapped.append(PolicyEnforcingTool(tool, guardrails, audit_sink))
    return wrapped