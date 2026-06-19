"""
Agent loader for various agent types using the MDB Agent Builder.
This module provides functionality to load different types of agents
including React, Reflection, Plan-Execute-Replan, and Long-Term Memory agents.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.language_models import BaseLLM
from langchain_core.tools import BaseTool

from agent_builder.agents.agent_gen import AgentFactory, AgentType
from agent_builder.core.interfaces import (
    BaseEpisodicMemoryAdapter,
    BaseObservationalMemoryAdapter,
)
from agent_builder.utils.checkpointer import get_mongodb_checkpointer
from agent_builder.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """Configuration for Agent setup."""

    agent_type: str
    name: str = "default_agent"
    system_prompt: Optional[str] = None
    reflection_prompt: Optional[str] = None
    system_prompt_path: Optional[str] = None
    reflection_prompt_path: Optional[str] = None
    llm: Optional[BaseLLM] = None
    tools: List[BaseTool] = field(default_factory=list)
    verbose: bool = False
    checkpointer_config: Optional[Dict[str, Any]] = None
    connection_str: Optional[str] = None
    namespace: Optional[str] = None
    # ------------------------------------------------------------------
    # Memory adapters (adapter-powered path)
    # When supplied, the long_term_memory agent uses these adapters instead
    # of the legacy hard-coded HuggingFace + MongoDBAtlasVectorSearch path.
    episodic_memory: Optional[BaseEpisodicMemoryAdapter] = None
    observational_memory: Optional[BaseObservationalMemoryAdapter] = None
    # ------------------------------------------------------------------
    # Multi-agent handoffs
    # Each entry is a plain agent-name string or a dict with 'name' and an
    # optional 'description' key.  The loader converts these into
    # create_handoff_tool() instances appended to the agent's tool list.
    handoff_targets: List[Any] = field(default_factory=list)
    # ------------------------------------------------------------------
    # Governance — policy-aware tool node (Approach B)
    # When set the factory injects a single policy-enforcing ToolNode
    # instead of wrapping each tool individually.
    policy: Any = None
    guardrails: Any = None
    audit_sink: Any = None
    # ------------------------------------------------------------------
    # Catch-all
    additional_kwargs: Optional[Dict[str, Any]] = None


# Agent type configuration map
AGENT_CONFIG = {
    "react": {
        "agent_type": AgentType.REACT,
        "required_fields": ["llm", "system_prompt"],
        "optional_fields": ["tools", "system_prompt_path", "checkpointer_config"],
        "description": "ReAct agent that thinks step-by-step and uses tools",
    },
    "tool_call": {
        "agent_type": AgentType.TOOL_CALL,
        "required_fields": ["llm"],
        "optional_fields": [
            "tools",
            "system_prompt",
            "system_prompt_path",
            "checkpointer_config",
        ],
        "description": "Agent that uses OpenAI-style tool calling",
    },
    "reflect": {
        "agent_type": AgentType.REFLECT,
        "required_fields": ["llm", "system_prompt", "reflection_prompt"],
        "optional_fields": [
            "tools",
            "checkpointer_config",
            "system_prompt_path",
            "reflection_prompt_path",
        ],
        "description": "Agent that uses a generate-reflect loop for improved reasoning",
    },
    "plan_execute_replan": {
        "agent_type": AgentType.PLAN_EXECUTE_REPLAN,
        "required_fields": ["llm", "system_prompt"],
        "optional_fields": ["tools", "checkpointer_config", "system_prompt_path"],
        "description": "Agent that plans, executes steps, and replans as needed",
    },
    "long_term_memory": {
        "agent_type": AgentType.LONG_TERM_MEMORY,
        "required_fields": ["llm", "connection_str", "namespace"],
        "optional_fields": ["tools", "checkpointer_config"],
        "description": "Agent with vector store-backed long-term memory",
    },
}


# ---------------------------------------------------------------------------
# Path-traversal-safe prompt loading
# ---------------------------------------------------------------------------

_PWD = Path.cwd().resolve()


def _safe_prompt_path(user_path: str) -> str:
    """Resolve *user_path* against CWD and refuse path-traversal attempts.

    Prompts are loaded from paths specified in YAML.  To prevent a
    compromised configuration from reading arbitrary filesystem paths
    (e.g. ``../../etc/shadow``), the resolved realpath must stay within
    the current working directory tree.
    """
    resolved = Path(os.path.realpath(os.path.join(_PWD, user_path)))
    try:
        resolved.relative_to(_PWD)
    except ValueError:
        raise ValueError(
            f"Prompt path '{user_path}' escapes the working directory"
        )
    return str(resolved)


def load_agent(config: AgentConfig) -> Any:
    """
    Load an agent based on the provided configuration.

    Args:
        config: AgentConfig containing agent type, LLM, tools, and other parameters

    Returns:
        An initialized agent instance

    Raises:
        ValueError: If the agent type is not supported or required configuration is missing
    """
    agent_type = config.agent_type.lower()
    logger.info("Loading agent of type: %s, name: %s", agent_type, config.name)

    # Check if agent type is supported
    if agent_type not in AGENT_CONFIG:
        available_types = list(AGENT_CONFIG.keys())
        logger.error(
            "Unsupported agent type: %s. Available types: %s",
            agent_type,
            available_types,
        )
        raise ValueError(
            f"Unsupported agent type: {agent_type}. Available types: {available_types}"
        )

    agent_info = AGENT_CONFIG[agent_type]

    # Load prompts from files if paths are provided
    def load_prompt_from_file(prompt, path, prompt_type="system"):
        """Helper function to load prompt from file with path-traversal protection."""
        if not prompt and path:
            try:
                safe_path = _safe_prompt_path(path)
                with open(safe_path, "r", encoding="utf-8") as f:
                    loaded_prompt = f.read()
                    logger.info("Loaded %s prompt from: %s", prompt_type, safe_path)
                    return loaded_prompt
            except Exception as e:
                logger.error(
                    "Failed to load %s prompt from %s: %s", prompt_type, path, str(e)
                )
                raise ValueError(
                    f"Failed to load {prompt_type} prompt from {path}: {str(e)}"
                )
        return prompt

    # Load prompts if needed
    config.system_prompt = load_prompt_from_file(
        config.system_prompt, config.system_prompt_path, "system"
    )

    config.reflection_prompt = load_prompt_from_file(
        config.reflection_prompt, config.reflection_prompt_path, "reflection"
    )

    # Verify required fields
    missing_fields = [
        field for field in agent_info["required_fields"] if not getattr(config, field)
    ]

    if missing_fields:
        error_msg = f"Missing required field(s) for {agent_type} agent: {', '.join(missing_fields)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Set up checkpointer if provided
    checkpointer = None
    if config.checkpointer_config:
        logger.info("Setting up checkpointer for agent %s", config.name)
        try:
            checkpointer = get_mongodb_checkpointer(**config.checkpointer_config)
        except Exception as e:
            logger.warning(
                "Failed to create checkpointer, using in-memory default: %s", str(e)
            )

    # Get system prompt from file if provided
    system_prompt = config.system_prompt
    reflection_prompt = config.reflection_prompt

    # Prepare kwargs for agent creation
    agent_kwargs: Dict[str, Any] = {
        "name": config.name,
        "model": config.llm,
        "tools": config.tools or [],
    }

# Governance — pass policy-aware tool-node parameters through to the
    # agent factory so that Approach B (single-choke-point enforcement)
    # replaces per-tool wrappers.
    if config.policy is not None:
        agent_kwargs["policy"] = config.policy
        agent_kwargs["guardrails"] = config.guardrails
        agent_kwargs["audit_sink"] = config.audit_sink
        agent_kwargs["use_dynamic_policy"] = bool(
            config.additional_kwargs.get("use_dynamic_policy", False)
            if config.additional_kwargs
            else False
        )

    # Add specific parameters based on agent type
    if agent_type in ["react", "tool_call"]:
        agent_kwargs["prompt"] = system_prompt
    elif agent_type == "reflect":
        # Basic reflection agent needs both generate and reflection prompts.
        agent_kwargs["generate_prompt"] = system_prompt
        agent_kwargs["reflection_prompt"] = reflection_prompt
    elif agent_type == "plan_execute_replan":
        agent_kwargs["execute_prompt"] = system_prompt
    elif agent_type == "long_term_memory":
        # Adapter-powered path takes precedence when adapters are supplied.
        if config.episodic_memory or config.observational_memory:
            if config.episodic_memory:
                agent_kwargs["episodic_memory"] = config.episodic_memory
            if config.observational_memory:
                agent_kwargs["observational_memory"] = config.observational_memory
        else:
            # Legacy path: pass raw connection params.
            agent_kwargs["connection_str"] = config.connection_str
            agent_kwargs["namespace"] = config.namespace

    # Add checkpointer if available
    if checkpointer:
        agent_kwargs["checkpointer"] = checkpointer

    # Add any additional kwargs
    if config.additional_kwargs:
        for key, value in config.additional_kwargs.items():
            if key not in agent_kwargs:
                agent_kwargs[key] = value

    # Inject handoff tools for multi-agent routing.  Must happen after
    # agent_kwargs["tools"] is finalised so the base tools are not lost.
    if config.handoff_targets:
        from agent_builder.agents.multi_agent import create_handoff_tool

        handoff_tools = []
        for target in config.handoff_targets:
            if isinstance(target, str):
                handoff_tools.append(create_handoff_tool(target))
            elif isinstance(target, dict):
                handoff_tools.append(
                    create_handoff_tool(target["name"], target.get("description"))
                )
        agent_kwargs["tools"] = list(agent_kwargs.get("tools") or []) + handoff_tools
        logger.info(
            "Injected %d handoff tool(s) into agent '%s': %s",
            len(handoff_tools),
            config.name,
            [t.name for t in handoff_tools],
        )

    logger.debug("Creating %s agent with parameters: %s", agent_type, agent_kwargs)

    # Create the agent using AgentFactory
    try:
        return AgentFactory.create_agent(agent_info["agent_type"], **agent_kwargs)
    except Exception as e:
        logger.error("Failed to create agent: %s", str(e))
        raise RuntimeError(f"Failed to create agent: {str(e)}")