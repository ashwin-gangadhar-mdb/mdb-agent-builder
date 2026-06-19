"""
Policy-aware tool-execution node for LangGraph agents.

This module provides a single auditable choke-point for enforcing access
policies on every tool call the agent makes, regardless of agent type.

Supports two integration patterns:

1. **Static policy** — the policy is known at agent-construction time
   and passed directly to ``create_policy_aware_tool_node()``.

2. **Dynamic policy** — the policy is resolved per-request and placed in
   LangGraph's ``config["configurable"]["policy"]`` dictionary.  The
   tool node reads it from there at execution time, enabling per-tenant /
   per-user enforcement without rebuilding the agent graph.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from agent_builder.core.types import AccessPolicy
from agent_builder.security.guardrails import GuardrailEngine
from agent_builder.utils.logging_config import get_logger

logger = get_logger(__name__)

AuditSink = Callable[[str, str, Optional[Dict[str, Any]]], None]


def enforce_tool_access(
    tool_name: str,
    policy: AccessPolicy,
    guardrails: GuardrailEngine,
    audit_sink: Optional[AuditSink] = None,
) -> bool:
    """
    Pure enforcement check: is *tool_name* permitted under *policy*?

    Returns True if the tool may be called; False otherwise.
    When *audit_sink* is supplied, a ``guardrail.tool`` audit event is
    emitted on every check (allowed or denied).
    """
    decision = guardrails.check_tool(tool_name, policy)
    if audit_sink:
        audit_sink(
            "guardrail.tool",
            {
                "allowed": decision.allowed,
                "tool": tool_name,
                "reason": decision.reason,
            },
        )
    return decision.allowed


def _policy_from_config(config: Optional[RunnableConfig]) -> Optional[AccessPolicy]:
    """Extract an AccessPolicy from a LangGraph RunnableConfig, if present.

    The policy is serialised as a JSON dict under
    ``config["configurable"]["policy"]``.  Returns ``None`` when no policy
    is available (e.g. governance is disabled), which disables enforcement.
    """
    if config is None:
        return None
    configurable = config.get("configurable") or {}
    raw = configurable.get("policy")
    if raw is None:
        return None
    if isinstance(raw, AccessPolicy):
        return raw
    if isinstance(raw, dict):
        return AccessPolicy(**raw)
    if isinstance(raw, str):
        return AccessPolicy(**json.loads(raw))
    return None


def create_policy_aware_tool_node(
    tools: List[BaseTool],
    policy: Optional[AccessPolicy] = None,
    guardrails: Optional[GuardrailEngine] = None,
    audit_sink: Optional[AuditSink] = None,
    use_dynamic_policy: bool = False,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Build a LangGraph tool-execution node that enforces policy on every call.

    This is the single-choke-point approach: instead of wrapping each tool
    individually, one graph node checks policy before dispatching.

    Two modes:
    - **Static** (default): *policy* is fixed at construction time.
    - **Dynamic** (*use_dynamic_policy=True*): The policy is read from
      ``config["configurable"]["policy"]`` on every execution.  The
      *policy* argument is then used as a fallback when the config doesn't
      carry a policy.

    Args:
        tools: The LangChain tools the agent is allowed to invoke.
        policy: Resolved ``AccessPolicy``.  In dynamic mode this is a
                fallback; in static mode it's the enforced policy.
        guardrails: Guardrail engine (defaults to a new ``GuardrailEngine``).
        audit_sink: Optional callback ``(event_type, payload_dict)`` for
                    audit logging.
        use_dynamic_policy: When True, read the policy from the run config
                           at invocation time rather than from *policy*.

    Returns:
        A callable suitable for use as a LangGraph tool node ``(state, config) → dict``.
    """
    if guardrails is None:
        guardrails = GuardrailEngine()

    tool_map: Dict[str, BaseTool] = {t.name: t for t in tools}
    engine = guardrails

    def tool_node(state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        if use_dynamic_policy:
            effective_policy = _policy_from_config(config) or policy
        else:
            effective_policy = policy

        if effective_policy is None:
            return state

        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]
        tool_calls = _extract_tool_calls(last_message)
        if not tool_calls:
            return state

        results: List[Any] = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("args", {})
            call_id = tc.get("id")

            if tool_name not in tool_map:
                logger.warning("Unknown tool requested: %s", tool_name)
                results.append(
                    _tool_result(call_id, tool_name,
                                 f"Tool '{tool_name}' is not available.")
                )
                continue

            if not enforce_tool_access(tool_name, effective_policy, engine,
                                       audit_sink=audit_sink):
                logger.info("Policy denied tool: %s", tool_name)
                results.append(
                    _tool_result(call_id, tool_name,
                                 "Access to this tool is denied by your current policy.")
                )
                continue

            tool = tool_map[tool_name]
            try:
                output = tool.invoke(tool_args)
                results.append(_tool_result(call_id, tool_name, str(output)))
            except Exception as exc:
                logger.exception("Tool '%s' failed: %s", tool_name, exc)
                results.append(
                    _tool_result(call_id, tool_name, f"Tool error: {exc}")
                )

        return {"messages": results}

    return tool_node


def _extract_tool_calls(message) -> List[Dict[str, Any]]:
    if hasattr(message, "tool_calls"):
        return message.tool_calls
    if isinstance(message, dict):
        return message.get("tool_calls", [])
    return []


def _tool_result(
    call_id: Optional[str],
    tool_name: str,
    content: str,
) -> Any:
    return ToolMessage(content=content, tool_call_id=call_id or "", name=tool_name)