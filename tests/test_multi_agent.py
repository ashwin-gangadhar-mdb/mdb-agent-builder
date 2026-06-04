"""
Tests for multi-agent graph building and handoff routing.
"""
from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _LLMCallsHandoff(BaseChatModel):
    """Fake LLM whose first response is a handoff tool call."""

    target_agent: str
    call_count: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        if self.call_count == 1:
            tool_call = {
                "name": f"transfer_to_{self.target_agent}",
                "args": {"reason": "routing test"},
                "id": "call_1",
                "type": "tool_call",
            }
            msg = AIMessage(content="", tool_calls=[tool_call])
        else:
            msg = AIMessage(content=f"Response from {self.target_agent}")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @property
    def _llm_type(self) -> str:
        return "fake_handoff_llm"

    def bind_tools(self, tools, **kwargs):
        return self


class _LLMSimpleReply(BaseChatModel):
    """Fake LLM that always returns a plain text reply."""

    reply: str = "I can help with that."

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    @property
    def _llm_type(self) -> str:
        return "fake_simple_llm"

    def bind_tools(self, tools, **kwargs):
        return self


# ---------------------------------------------------------------------------
# create_handoff_tool
# ---------------------------------------------------------------------------

class TestCreateHandoffTool:
    def test_tool_name(self):
        from agent_builder.agents.multi_agent import create_handoff_tool

        tool = create_handoff_tool("billing_agent")
        assert tool.name == "transfer_to_billing_agent"

    def test_tool_name_custom(self):
        from agent_builder.agents.multi_agent import create_handoff_tool

        tool = create_handoff_tool("tech", description="Route to tech")
        assert tool.name == "transfer_to_tech"
        assert "Route to tech" in tool.description

    def test_tool_returns_command(self):
        from agent_builder.agents.multi_agent import create_handoff_tool
        from langgraph.types import Command

        tool = create_handoff_tool("billing_agent")
        result = tool.invoke({"reason": "billing issue"})
        assert isinstance(result, Command)
        assert result.goto == "billing_agent"
        assert result.graph is Command.PARENT

    def test_default_description_contains_agent_name(self):
        from agent_builder.agents.multi_agent import create_handoff_tool

        tool = create_handoff_tool("support_agent")
        assert "support_agent" in tool.description


# ---------------------------------------------------------------------------
# create_multi_agent_graph
# ---------------------------------------------------------------------------

class TestCreateMultiAgentGraph:
    def _make_fake_agent(self, reply: str = "OK"):
        """Return a minimal compiled LangGraph agent using a fake LLM."""
        from langgraph.prebuilt import create_react_agent

        llm = _LLMSimpleReply(reply=reply)
        return create_react_agent(llm, [], name="fake")

    def test_graph_built_with_correct_entry(self):
        from agent_builder.agents.multi_agent import create_multi_agent_graph

        agent_a = self._make_fake_agent("hello from A")
        agent_b = self._make_fake_agent("hello from B")

        graph = create_multi_agent_graph(
            {"agent_a": agent_a, "agent_b": agent_b},
            entry_agent="agent_a",
        )
        assert graph is not None

    def test_raises_on_missing_entry_agent(self):
        from agent_builder.agents.multi_agent import create_multi_agent_graph

        agent_a = self._make_fake_agent()
        with pytest.raises(ValueError, match="entry_agent 'missing'"):
            create_multi_agent_graph({"agent_a": agent_a}, entry_agent="missing")

    def test_single_agent_invocation(self):
        from agent_builder.agents.multi_agent import create_multi_agent_graph

        agent_a = self._make_fake_agent("hello from A")
        graph = create_multi_agent_graph({"agent_a": agent_a}, entry_agent="agent_a")
        result = graph.invoke({"messages": [HumanMessage(content="hi")]})
        contents = [m.content for m in result["messages"] if m.content]
        assert "hello from A" in contents

    def test_handoff_routing(self):
        """Agent A calls a handoff tool and execution continues in agent B."""
        from agent_builder.agents.multi_agent import (
            create_handoff_tool,
            create_multi_agent_graph,
        )
        from langgraph.prebuilt import create_react_agent

        handoff_tool = create_handoff_tool("agent_b", "Route to B")
        llm_a = _LLMCallsHandoff(target_agent="agent_b")
        llm_b = _LLMSimpleReply(reply="Response from B")

        agent_a = create_react_agent(llm_a, [handoff_tool], name="agent_a")
        agent_b = create_react_agent(llm_b, [], name="agent_b")

        graph = create_multi_agent_graph(
            {"agent_a": agent_a, "agent_b": agent_b},
            entry_agent="agent_a",
        )
        result = graph.invoke({"messages": [HumanMessage(content="hello")]})
        contents = [m.content for m in result["messages"] if m.content]
        assert "Response from B" in contents


# ---------------------------------------------------------------------------
# AgentConfig.handoff_targets field
# ---------------------------------------------------------------------------

class TestAgentConfigHandoffTargets:
    def test_field_defaults_empty(self):
        from agent_builder.agents.loader import AgentConfig

        cfg = AgentConfig(agent_type="react", name="test")
        assert cfg.handoff_targets == []

    def test_field_accepts_strings(self):
        from agent_builder.agents.loader import AgentConfig

        cfg = AgentConfig(
            agent_type="react",
            name="test",
            handoff_targets=["agent_b", "agent_c"],
        )
        assert cfg.handoff_targets == ["agent_b", "agent_c"]

    def test_field_accepts_dicts(self):
        from agent_builder.agents.loader import AgentConfig

        cfg = AgentConfig(
            agent_type="react",
            name="test",
            handoff_targets=[{"name": "billing", "description": "for billing"}],
        )
        assert cfg.handoff_targets[0]["name"] == "billing"


# ---------------------------------------------------------------------------
# load_application with agents: list
# ---------------------------------------------------------------------------

class TestLoadApplicationMultiAgent:
    """Integration-level tests that patch the LLM and tool loading."""

    def _make_llm_mock(self):
        llm = _LLMSimpleReply()
        llm.name = "mock_llm"
        return llm

    def test_multi_agent_result_keys(self, tmp_path):
        """load_application populates result['agent'] and result['agents']."""
        from agent_builder.yaml_loader import load_application

        config_file = tmp_path / "multi.yaml"
        config_file.write_text(
            """
llms:
  - name: my_llm
    provider: openai
    model_name: gpt-4o

agents:
  - name: agent_a
    agent_type: react
    llm: my_llm
    system_prompt: "Agent A"

  - name: agent_b
    agent_type: react
    llm: my_llm
    system_prompt: "Agent B"

entry_agent: agent_a
"""
        )

        mock_llm = self._make_llm_mock()

        with (
            patch("agent_builder.llms.adapters.LLMAdapterFactory.create") as mock_factory,
        ):
            adapter_mock = MagicMock()
            adapter_mock.get_llm.return_value = mock_llm
            mock_factory.return_value = adapter_mock

            result = load_application(str(config_file))

        assert "agent" in result, "result['agent'] should be the compiled multi-agent graph"
        assert "agents" in result, "result['agents'] should map agent names to compiled subgraphs"
        assert set(result["agents"].keys()) == {"agent_a", "agent_b"}

    def test_multi_agent_with_handoffs_injects_tools(self, tmp_path):
        """Handoff tools are injected into the agent's tool list."""
        from agent_builder.yaml_loader import load_application

        config_file = tmp_path / "multi_handoffs.yaml"
        config_file.write_text(
            """
llms:
  - name: my_llm
    provider: openai
    model_name: gpt-4o

agents:
  - name: triage
    agent_type: react
    llm: my_llm
    system_prompt: "Triage"
    handoffs:
      - name: billing
        description: "Transfer for billing"
      - name: technical

  - name: billing
    agent_type: react
    llm: my_llm
    system_prompt: "Billing"

  - name: technical
    agent_type: react
    llm: my_llm
    system_prompt: "Technical"

entry_agent: triage
"""
        )

        mock_llm = self._make_llm_mock()
        captured_tools: list = []

        original_create = __import__(
            "agent_builder.agents.agent_gen", fromlist=["create_react_agent"]
        ).create_react_agent

        def capture_tools(*args, **kwargs):
            captured_tools.extend(kwargs.get("tools", []))
            return original_create(*args, **kwargs)

        with (
            patch("agent_builder.llms.adapters.LLMAdapterFactory.create") as mock_factory,
            patch(
                "agent_builder.agents.agent_gen.create_react_agent",
                side_effect=capture_tools,
            ),
        ):
            adapter_mock = MagicMock()
            adapter_mock.get_llm.return_value = mock_llm
            mock_factory.return_value = adapter_mock

            load_application(str(config_file))

        handoff_tool_names = [t.name for t in captured_tools]
        assert "transfer_to_billing" in handoff_tool_names
        assert "transfer_to_technical" in handoff_tool_names

    def test_entry_agent_defaults_to_first(self, tmp_path):
        """entry_agent defaults to the first agent when not specified."""
        from agent_builder.yaml_loader import load_application
        from agent_builder.agents.multi_agent import create_multi_agent_graph

        config_file = tmp_path / "no_entry.yaml"
        config_file.write_text(
            """
llms:
  - name: my_llm
    provider: openai
    model_name: gpt-4o

agents:
  - name: first_agent
    agent_type: react
    llm: my_llm
    system_prompt: "First"

  - name: second_agent
    agent_type: react
    llm: my_llm
    system_prompt: "Second"
"""
        )

        mock_llm = self._make_llm_mock()
        captured: dict = {}

        original = create_multi_agent_graph

        def capture_entry(agents, entry_agent, **kwargs):
            captured["entry_agent"] = entry_agent
            return original(agents, entry_agent, **kwargs)

        with (
            patch("agent_builder.llms.adapters.LLMAdapterFactory.create") as mock_factory,
            patch(
                "agent_builder.agents.multi_agent.create_multi_agent_graph",
                side_effect=capture_entry,
            ),
        ):
            adapter_mock = MagicMock()
            adapter_mock.get_llm.return_value = mock_llm
            mock_factory.return_value = adapter_mock

            load_application(str(config_file))

        assert captured.get("entry_agent") == "first_agent"
