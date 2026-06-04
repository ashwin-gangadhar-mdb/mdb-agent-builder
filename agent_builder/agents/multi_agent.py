"""
Multi-agent graph builder.

Creates a LangGraph StateGraph from a collection of individually compiled
agents.  Agents communicate via *handoff tools* — LangChain tools that return
``Command(goto=<agent_name>, graph=Command.PARENT)`` — which cause the LangGraph
runtime to route execution to the named node in the outer graph.

Usage (programmatic)::

    from agent_builder.agents.multi_agent import (
        create_handoff_tool,
        create_multi_agent_graph,
    )

    # Build individual agents (each is a compiled LangGraph subgraph)
    triage_agent   = create_react_agent(llm, [create_handoff_tool("billing")])
    billing_agent  = create_react_agent(llm, billing_tools)

    graph = create_multi_agent_graph(
        agents={"triage": triage_agent, "billing": billing_agent},
        entry_agent="triage",
    )
    result = graph.invoke({"messages": [("user", "I have a billing question")]})

Usage (via YAML)::

    agents:
      - name: triage_agent
        agent_type: react
        llm: my_llm
        system_prompt: "Route requests to the right specialist."
        handoffs:
          - name: billing_agent
            description: "Transfer for billing or payment questions"
          - name: technical_agent
            description: "Transfer for technical support issues"

      - name: billing_agent
        agent_type: react
        llm: my_llm
        system_prompt: "You handle billing and payment questions."

      - name: technical_agent
        agent_type: react
        llm: my_llm
        system_prompt: "You handle technical support issues."

    entry_agent: triage_agent
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.tools import BaseTool
from langgraph.types import Command  # must be module-level for @lc_tool type introspection

from agent_builder.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_handoff_tool(
    agent_name: str,
    description: Optional[str] = None,
) -> BaseTool:
    """
    Create a LangChain tool that hands off execution to *agent_name* in the
    parent multi-agent graph.

    When the agent calls this tool, LangGraph routes execution to the node
    named *agent_name* in the outer ``StateGraph``.  The ``reason`` argument
    is logged and passed as context so the receiving agent understands why it
    was invoked.

    Args:
        agent_name:  The name of the target agent node in the outer graph.
        description: Human-readable description shown to the LLM. Defaults to
                     a generic "Transfer the conversation to <agent_name>."

    Returns:
        A ``BaseTool`` instance named ``transfer_to_<agent_name>``.
    """
    from langchain_core.tools import tool as lc_tool

    desc = (
        description
        or f"Transfer the conversation to the {agent_name} agent."
    )

    # The closure captures agent_name so each tool routes to a distinct target.
    # Return type is Any (not Command) to avoid pydantic forward-ref resolution
    # errors when @lc_tool introspects type hints at decoration time.
    @lc_tool(description=desc)
    def handoff(reason: str) -> Any:
        """Hand off to another agent. Provide a brief reason for the transfer."""
        logger.debug("Handing off to agent '%s': %s", agent_name, reason)
        return Command(goto=agent_name, graph=Command.PARENT)

    handoff.name = f"transfer_to_{agent_name}"
    return handoff


def create_multi_agent_graph(
    agents: Dict[str, Any],
    entry_agent: str,
    checkpointer: Optional[Any] = None,
) -> Any:
    """
    Build a compiled LangGraph ``StateGraph`` from a mapping of named agents.

    Each agent becomes a node.  Agents that were given handoff tools (via
    ``create_handoff_tool``) can route to other agents dynamically by calling
    those tools — no static edges between agent nodes are needed.

    When an agent finishes without calling a handoff tool, it terminates the
    conversation via the default ``agent → END`` edge.

    Args:
        agents:      Mapping of ``agent_name → compiled agent subgraph``.
                     Typically produced by ``AgentFactory.create_agent()``.
        entry_agent: Name of the agent that receives the first user message.
                     Must be a key in *agents*.
        checkpointer: Optional LangGraph checkpointer for durable, cross-worker
                      conversation state.  Individual agent subgraphs should
                      NOT have their own checkpointers when this is provided.

    Returns:
        A compiled ``StateGraph`` whose ``.invoke()`` and ``.stream()``
        signatures match those of a single compiled agent.

    Raises:
        ValueError: If *entry_agent* is not a key in *agents*.
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import MessagesState

    if entry_agent not in agents:
        raise ValueError(
            f"entry_agent '{entry_agent}' not found in agents. "
            f"Available agents: {sorted(agents)}"
        )

    builder = StateGraph(MessagesState)

    for name, agent in agents.items():
        builder.add_node(name, agent)
        # Static fallback edge: when the agent finishes without a handoff
        # Command, execution terminates here.  A Command(goto=X, graph=PARENT)
        # returned from a handoff tool overrides this edge dynamically.
        builder.add_edge(name, END)

    builder.add_edge(START, entry_agent)

    logger.info(
        "Building multi-agent graph: entry_agent='%s', agents=%s",
        entry_agent,
        sorted(agents),
    )
    return builder.compile(checkpointer=checkpointer)
