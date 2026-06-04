"""Runtime guardrails for requests, responses, tools, and retrieval."""

import re
from typing import Any, Dict

from agent_builder.core.types import AccessPolicy, GuardrailDecision


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
        enforced_filters = dict(filters or {})
        enforced_filters.update(policy.retrieval_filters)
        enforced_filters["tenant_id"] = policy.tenant_id
        if policy.user_id:
            enforced_filters.setdefault("user_id", policy.user_id)
        return enforced_filters

    def _redact_pii(self, text: str) -> str:
        redacted = text
        for pattern, replacement in self.PII_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        return redacted
