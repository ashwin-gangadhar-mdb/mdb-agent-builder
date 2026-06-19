"""Runtime guardrails for requests, responses, tools, and retrieval."""

import logging
import re
from typing import Any, Dict, List, Set

from agent_builder.core.types import (
    AccessPolicy,
    GuardrailDecision,
    RetrievalFilterConflict,
)

logger = logging.getLogger(__name__)

_OPERATOR_KEYS: Set[str] = {"$in", "$nin", "$gt", "$gte", "$lt", "$lte", "$ne",
                              "$eq", "$exists", "$regex", "$elemMatch", "$all",
                              "$size", "$and", "$or", "$nor", "$not"}


class GuardrailEngine:
    """Policy-driven guardrail checks that are independent of agent runtime."""

    PROMPT_INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
        re.compile(r"reveal\s+(the\s+)?(system|developer)\s+prompt", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    ]
    PII_PATTERNS = [
        (re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
        (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
    ]

    def check_input(self, text: str, policy: AccessPolicy) -> GuardrailDecision:
        lowered = text.lower()
        for topic in policy.blocked_topics:
            if topic.lower() in lowered:
                return GuardrailDecision(
                    allowed=False,
                    stage="input_guardrail",
                    reason=f"Blocked topic detected: {topic}",
                )

        if policy.prompt_injection_detection:
            for pattern in self.PROMPT_INJECTION_PATTERNS:
                if pattern.search(text):
                    return GuardrailDecision(
                        allowed=False,
                        stage="input_guardrail",
                        reason="Potential prompt-injection attempt detected",
                    )

        transformed = self._redact_pii(text) if policy.pii_redaction else text
        return GuardrailDecision(
            allowed=True,
            stage="input_guardrail",
            transformed_text=transformed,
        )

    def check_output(self, text: str, policy: AccessPolicy) -> GuardrailDecision:
        transformed = self._redact_pii(text) if policy.pii_redaction else text
        return GuardrailDecision(
            allowed=True,
            stage="output_guardrail",
            transformed_text=transformed,
        )

    def check_tool(self, tool_name: str, policy: AccessPolicy) -> GuardrailDecision:
        if tool_name in policy.denied_tools:
            return GuardrailDecision(
                allowed=False,
                stage="tool_guardrail",
                reason=f"Tool denied by policy: {tool_name}",
            )
        if policy.allows(f"tools.call.{tool_name}") or policy.allows("tools.call.*"):
            return GuardrailDecision(allowed=True, stage="tool_guardrail")
        return GuardrailDecision(
            allowed=False,
            stage="tool_guardrail",
            reason=f"Missing permission: tools.call.{tool_name}",
        )

    def apply_retrieval_filters(
        self, filters: Dict[str, Any], policy: AccessPolicy
    ) -> Dict[str, Any]:
        """Merge policy retrieval filters with tool-static filters.

        Merge per shared key using the precedence rules:
        - Scalar equality vs scalar: equal → keep; different → raise
        - ``$in`` vs ``$in``: set intersection; empty → raise
        - Scalar vs ``$in``: keep scalar if member of list, else raise
        - Key on only one side: carry through unchanged
        - Any operator outside {scalar, ``$in``}: tool floor wins,
          policy value ignored, raise on conflict, log a warning.

        ``tenant_id`` is forced last and **cannot** be overridden.
        """
        tool_dict = dict(filters or {})
        policy_dict = dict(policy.retrieval_filters or {})
        merged: Dict[str, Any] = {}
        fallback_keys: List[str] = []

        all_keys = set(tool_dict) | set(policy_dict)
        for key in all_keys:
            tv = tool_dict.get(key)
            pv = policy_dict.get(key)

            if tv is None:
                merged[key] = pv
                continue
            if pv is None:
                merged[key] = tv
                continue

            tv_scalar = _is_scalar_value(tv)
            pv_scalar = _is_scalar_value(pv)
            tv_is_in = _is_in_operator(tv)
            pv_is_in = _is_in_operator(pv)

            if (tv_scalar or tv_is_in) and (pv_scalar or pv_is_in):
                _merge_scalar_or_in(key, tv, pv, merged)
            else:
                fallback_keys.append(key)
                if not _values_compatible(key, tv, pv):
                    raise RetrievalFilterConflict(key, tv, pv)
                merged[key] = tv

        if fallback_keys:
            logger.warning(
                "Retrieval filter keys outside {scalar, $in} — tool floor wins: %s",
                fallback_keys,
            )

        merged["tenant_id"] = policy.tenant_id
        if policy.user_id:
            merged.setdefault("user_id", policy.user_id)
        return merged

    def _redact_pii(self, text: str) -> str:
        redacted = text
        for pattern, replacement in self.PII_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted


# ---------------------------------------------------------------------------
# Retrieval filter merge helpers (module-level so they're testable)
# ---------------------------------------------------------------------------

def _is_scalar_value(val: Any) -> bool:
    return not isinstance(val, dict)


def _is_in_operator(val: Any) -> bool:
    return isinstance(val, dict) and len(val) == 1 and "$in" in val


def _in_list(val: Any) -> List[Any]:
    """Extract the list from a ``{'$in': [...]}`` operator dict."""
    return list(val.get("$in", []))


def _values_compatible(key: str, tool_val: Any, policy_val: Any) -> bool:
    if isinstance(tool_val, dict) and isinstance(policy_val, dict):
        return True
    return tool_val == policy_val


def _merge_scalar_or_in(key: str, tool_val: Any, policy_val: Any,
                         merged: Dict[str, Any]) -> None:
    """Merge tool and policy values for a single scalar/``$in`` key.

    Raises ``RetrievalFilterConflict`` if the values are irreconcilable.
    """
    tv_scalar = _is_scalar_value(tool_val)
    pv_scalar = _is_scalar_value(policy_val)

    if tv_scalar and pv_scalar:
        if tool_val == policy_val:
            merged[key] = tool_val
        else:
            raise RetrievalFilterConflict(key, tool_val, policy_val)
        return

    tv_in = _in_list(tool_val) if not tv_scalar else [tool_val]
    pv_in = _in_list(policy_val) if not pv_scalar else [policy_val]

    if tv_scalar:
        if tool_val in pv_in:
            merged[key] = tool_val
        else:
            raise RetrievalFilterConflict(key, tool_val, policy_val)
        return

    if pv_scalar:
        if policy_val in tv_in:
            merged[key] = policy_val
        else:
            raise RetrievalFilterConflict(key, tool_val, policy_val)
        return

    intersection = [v for v in tv_in if v in pv_in]
    if not intersection:
        raise RetrievalFilterConflict(key, tool_val, policy_val)
    merged[key] = {"$in": intersection}
