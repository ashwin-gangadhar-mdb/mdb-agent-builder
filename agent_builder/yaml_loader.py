import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

import yaml
from pydantic import BaseModel, create_model
from agent_builder.utils.logging_config import get_logger

# Set up module logger
logger = get_logger(__name__)


def parse_response_model(response_dict: dict) -> Type[BaseModel]:
    """
    Builds a Pydantic model class from a field specification dictionary.

    Args:
        response_dict (dict): Mapping of field name to a ``(type, default)`` pair.

    Returns:
        Type[BaseModel]: A dynamically created Pydantic model class.
    """
    fields = {key: tuple(value) for key, value in response_dict.items()}
    return create_model("ResponseModel", **fields)


# Whitelist of environment variable name patterns that YAML configs may
# resolve.  Set YAML_ENV_VAR_ALLOWLIST to a comma-separated list of regex
# patterns to override.  Default: typical LLM / MongoDB / Grove / AWS vars.
_YAML_ENV_VAR_ALLOWLIST = [
    re.compile(r)
    for r in os.environ.get(
        "YAML_ENV_VAR_ALLOWLIST",
        "MONGODB_.*,OPENAI_.*,ANTHROPIC_.*,FIREWORKS_.*,COHERE_.*,"
        "TOGETHER_.*,VOYAGEAI_.*,AZURE_.*,GROVE_.*,AWS_.*,"
        "GOOGLE_.*,GEMINI_.*,"
        "OLLAMA_.*,LOG_LEVEL,FLASK_.*,AGENT_CONFIG_PATH,"
        "CHECKPOINT_.*,PORT,GUNICORN_.*,PYTHONPATH,LANGCHAIN_.*,"
        "MDB_.*,PYTHON.*",
    ).split(",")
]


def _env_var_allowed(name: str) -> bool:
    """Return True if *name* matches at least one allowlist pattern."""
    return any(pattern.fullmatch(name) for pattern in _YAML_ENV_VAR_ALLOWLIST)


def resolve_env_variables(data, allowlist_check=True):
    """
    Recursively resolves environment variables in a dictionary or string.

    Handles both ``${VAR_NAME}`` and ``${VAR_NAME:-default_value}`` syntax.

    When *allowlist_check* is True (the default), only environment variables
    whose names match the ``YAML_ENV_VAR_ALLOWLIST`` patterns can be resolved.
    Unknown variables cause a ``ValueError`` so that a compromised YAML file
    cannot exfiltrate arbitrary secrets from the server's environment.
    """
    if isinstance(data, dict):
        return {k: resolve_env_variables(v, allowlist_check) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_env_variables(elem, allowlist_check) for elem in data]
    elif isinstance(data, str):
        pattern = re.compile(r"\$\{(\w+)(:-([^}]*))?\}")

        def replace_match(match):
            var_name = match.group(1)
            default_value = match.group(3)

            if allowlist_check and not _env_var_allowed(var_name):
                raise ValueError(
                    f"Environment variable '{var_name}' is not in the YAML env-var "
                    f"allowlist.  Add it to YAML_ENV_VAR_ALLOWLIST to permit "
                    f"resolution from configuration files."
                )

            value = os.environ.get(var_name, default_value)
            if value is None:
                logger.warning(
                    "Environment variable '%s' not set and no default provided.",
                    var_name,
                )
                raise ValueError(
                    f"Environment variable '{var_name}' not set and no default provided."
                )
            return value

        return pattern.sub(replace_match, data)
    return data


def _resolve_ref(value, registry: dict, kind: str, owner: str):
    """Resolve a string *value* against a ``name -> object`` *registry*.

    Args:
        value:    The reference (a name) to resolve. Non-strings pass through.
        registry: Mapping of names to already-built component instances.
        kind:     Human-readable component kind for error messages.
        owner:    Human-readable description of what holds the reference.

    Raises:
        ValueError: if *value* is a name that is not present in *registry*.
    """
    if not isinstance(value, str):
        return value
    if value not in registry:
        msg = f"{owner} references {kind} '{value}' which was not found."
        logger.error(msg)
        raise ValueError(msg)
    return registry[value]


def _resolve_component_refs(item: dict, result: dict, owner: str) -> None:
    """Resolve the ``embedding_model`` and ``llm`` references of *item* in-place."""
    for field, registry_key, kind in (
        ("embedding_model", "embeddings", "embedding model"),
        ("llm", "llms", "LLM"),
    ):
        if field in item:
            item[field] = _resolve_ref(
                item[field], result.get(registry_key, {}), kind, owner
            )


def load_yaml(file_path) -> dict:
    """
    Load and parse a YAML configuration file with environment variable resolution.

    Args:
        file_path: Path to the YAML file

    Returns:
        The parsed and resolved configuration dictionary

    Raises:
        FileNotFoundError: If the file doesn't exist
        YAMLError: If there's an error parsing the YAML
    """
    try:
        logger.info(f"Loading configuration from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config:
            logger.warning(f"Empty or invalid YAML file: {file_path}")
            return {}

        logger.debug(f"Resolving environment variables in configuration")
        resolved_config = resolve_env_variables(config)
        return resolved_config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading YAML file {file_path}: {e}")
        raise


def load_application(config_path: str, governance_bundle=None):
    """
    Load application components from a YAML configuration file.

    ...

    Args:
        config_path: Path to the YAML configuration file
        governance_bundle: Optional dict with ``guardrails`` (GuardrailEngine),
                           ``audit_sink`` (callback), and ``governance`` (dict)
                           for wrapping tools in PolicyEnforcingTool.  Passed
                           in by app.py when governance is enabled.

    Returns:
        A dictionary containing the loaded application components
    """
    config = load_yaml(config_path)
    from agent_builder.agents.loader import AgentConfig, load_agent
    from agent_builder.embeddings.loader import EmbeddingConfig, load_embedding_models
    from agent_builder.llms.loader import LLMConfig, load_llms
    from agent_builder.memory.adapters import MemoryAdapterFactory, MemoryConfig
    from agent_builder.tools.loader import ToolConfig, load_tools

    result = {}

    # Load embeddings
    if "embeddings" in config:
        logger.info("Loading embedding models")
        emb_configs = [EmbeddingConfig(**emb) for emb in config["embeddings"]]
        result["embeddings"] = load_embedding_models(emb_configs)

    # Load LLMs
    if "llms" in config:
        logger.info("Loading language models")
        llm_configs = [LLMConfig(**llm) for llm in config["llms"]]
        result["llms"] = load_llms(llm_configs)

    # Load memory adapters (must come after embeddings + LLMs so refs resolve)
    if "memory" in config:
        logger.info("Loading memory adapters")
        memory_configs_raw = deepcopy(config["memory"])
        # Normalise: allow a single dict or a list of dicts
        if isinstance(memory_configs_raw, dict):
            memory_configs_raw = [memory_configs_raw]

        memory_adapters = {}
        for mem_raw in memory_configs_raw:
            _resolve_component_refs(
                mem_raw, result, f"Memory adapter '{mem_raw.get('name')}'"
            )
            mem_cfg = MemoryConfig(**mem_raw)
            adapter = MemoryAdapterFactory.create(mem_cfg)
            memory_adapters[mem_cfg.name] = adapter
            logger.info(
                "Loaded memory adapter '%s' (type=%s)", mem_cfg.name, mem_cfg.memory_type
            )

        result["memory_adapters"] = memory_adapters

    # Load tools with resolved references
    if "tools" in config:
        logger.info("Loading tools")
        tools_config = deepcopy(config["tools"])

        # Resolve embedding model / LLM references on each tool
        for tool in tools_config:
            _resolve_component_refs(tool, result, f"Tool '{tool.get('name')}'")

        tool_configs = [ToolConfig(**tool) for tool in tools_config]
        result["tools"] = load_tools(tool_configs)

    # A1 — Wrap tools in PolicyEnforcingTool when governance is enabled.
    # Must happen AFTER tools are loaded but BEFORE the agent graph is
    # compiled, because tools are baked into the graph at compile time.
    _maybe_wrap_tools_in_governance(result, config, governance_bundle)

    # Load agent with resolved references
    if "agent" in config:
        logger.info("Loading agent")
        agent_config = deepcopy(config["agent"])

        # Resolve LLM reference
        if "llm" in agent_config:
            agent_config["llm"] = _resolve_ref(
                agent_config["llm"], result.get("llms", {}), "LLM", "Agent"
            )

        # Resolve tool references
        if "tools" in agent_config and isinstance(agent_config["tools"], list):
            agent_config["tools"] = [
                _resolve_ref(tool_name, result.get("tools", {}), "tool", "Agent")
                for tool_name in agent_config["tools"]
            ]

        if "checkpointer" in config:
            agent_config["checkpointer_config"] = config["checkpointer"]

        # Inject memory adapters into the agent config.
        # Convention:
        #   • The first adapter whose memory_type == "episodic" becomes
        #     agent_config["episodic_memory"].
        #   • The first adapter whose memory_type == "observational" becomes
        #     agent_config["observational_memory"].
        # Explicit references can also be specified as:
        #   agent:
        #     episodic_memory: recall       # name of a memory adapter
        #     observational_memory: observations
        memory_adapters = result.get("memory_adapters", {})
        if memory_adapters:
            _wire_memory_adapters(agent_config, memory_adapters, config)

        # Load the agent
        agent_config_obj = AgentConfig(**agent_config)
        result["agent"] = load_agent(agent_config_obj)

    # Multi-agent mode: load a list of agents and wire them into a
    # single StateGraph with handoff routing.  Mutually exclusive with
    # the singular ``agent:`` key — use one or the other, not both.
    if "agents" in config:
        logger.info("Loading multi-agent configuration")
        from agent_builder.agents.multi_agent import create_multi_agent_graph
        from agent_builder.utils.checkpointer import get_mongodb_checkpointer

        agents_config_list = deepcopy(config["agents"])
        entry_agent_name = config.get("entry_agent") or agents_config_list[0]["name"]
        built_agents: Dict[str, Any] = {}

        for agent_raw in agents_config_list:
            agent_name = agent_raw.get("name", "agent")

            # Resolve LLM reference
            if "llm" in agent_raw:
                agent_raw["llm"] = _resolve_ref(
                    agent_raw["llm"], result.get("llms", {}), "LLM",
                    f"Agent '{agent_name}'"
                )

            # Resolve tool references
            if "tools" in agent_raw and isinstance(agent_raw["tools"], list):
                agent_raw["tools"] = [
                    _resolve_ref(t, result.get("tools", {}), "tool",
                                 f"Agent '{agent_name}'")
                    for t in agent_raw["tools"]
                ]

            # Wire memory adapters if configured
            memory_adapters = result.get("memory_adapters", {})
            if memory_adapters:
                _wire_memory_adapters(agent_raw, memory_adapters, config)

            # Individual agents must NOT have their own checkpointer — the
            # outer multi-agent graph owns checkpointing so state is shared
            # across all agents in the same conversation thread.
            agent_raw.pop("checkpointer_config", None)

            # Map YAML key 'handoffs' → AgentConfig field 'handoff_targets'
            if "handoffs" in agent_raw:
                agent_raw["handoff_targets"] = agent_raw.pop("handoffs")

            agent_cfg = AgentConfig(**agent_raw)
            built_agents[agent_cfg.name] = load_agent(agent_cfg)
            logger.info("Loaded agent '%s' for multi-agent graph", agent_cfg.name)

        # Build the outer graph with an optional shared checkpointer
        outer_checkpointer = None
        if "checkpointer" in config:
            try:
                outer_checkpointer = get_mongodb_checkpointer(**config["checkpointer"])
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "Failed to create checkpointer for multi-agent graph: %s", e
                )

        result["agent"] = create_multi_agent_graph(
            built_agents, entry_agent_name, checkpointer=outer_checkpointer
        )
        result["agents"] = built_agents
        logger.info(
            "Multi-agent graph built: entry_agent='%s', agents=%s",
            entry_agent_name, sorted(built_agents),
        )

    # Keep control-plane configuration available to the serving layer.
    for config_key in ["governance", "state"]:
        if config_key in config:
            result[f"{config_key}_config"] = config[config_key]
    # Keep raw memory config for introspection (adapters already instantiated above)
    if "memory" in config:
        result["memory_config"] = config["memory"]

    logger.info("Application components loaded successfully")
    return result


def _wire_memory_adapters(
    agent_config: dict,
    memory_adapters: dict,
    full_config: dict,
) -> None:
    """
    Resolve memory adapter references in *agent_config* in-place.

    If the agent YAML explicitly names adapters::

        agent:
          episodic_memory: recall
          observational_memory: observations

    those names are looked up in *memory_adapters*.  Otherwise, the first
    adapter of each type is auto-assigned.
    """
    from agent_builder.core.interfaces import (
        BaseEpisodicMemoryAdapter,
        BaseObservationalMemoryAdapter,
    )

    # Resolve explicit name references
    for field_name, adapter_class in [
        ("episodic_memory", BaseEpisodicMemoryAdapter),
        ("observational_memory", BaseObservationalMemoryAdapter),
    ]:
        if field_name in agent_config and isinstance(agent_config[field_name], str):
            ref = agent_config[field_name]
            if ref not in memory_adapters:
                raise ValueError(
                    f"Agent references memory adapter '{ref}' under '{field_name}' "
                    f"but no adapter with that name was found. "
                    f"Available adapters: {list(memory_adapters)}"
                )
            agent_config[field_name] = memory_adapters[ref]
            logger.info("Wired %s → adapter '%s'", field_name, ref)

    # Auto-assign by type if not already set
    if "episodic_memory" not in agent_config:
        for adapter in memory_adapters.values():
            if isinstance(adapter, BaseEpisodicMemoryAdapter):
                agent_config["episodic_memory"] = adapter
                logger.info(
                    "Auto-wired episodic_memory → adapter '%s'",
                    adapter.__class__.__name__,
                )
                break

    if "observational_memory" not in agent_config:
        for adapter in memory_adapters.values():
            if isinstance(adapter, BaseObservationalMemoryAdapter):
                agent_config["observational_memory"] = adapter
                logger.info(
                    "Auto-wired observational_memory → adapter '%s'",
                    adapter.__class__.__name__,
                )
                break


def _maybe_wrap_tools_in_governance(
    result: dict, config: dict, governance_bundle: Any
) -> None:
    """Wrap every loaded tool in ``PolicyEnforcingTool`` if governance is on.

    Called from ``load_application`` immediately after tools are loaded and
    before the agent graph is compiled.  Only active when
    ``config['governance']['enabled']`` is true and a governance bundle was
    supplied that carries a ``GuardrailEngine``.

    Per the plan, ``nl_to_mql`` and ``mongodb_toolkit`` tools are always
    denied under governance (T8) — their ``AccessPolicy.allows()`` returns
    ``False`` because no permission grant matches their tool names (the
    wrapper's ``check_tool`` handles the deny).
    """
    if not config.get("governance", {}).get("enabled"):
        return
    if not governance_bundle or "guardrails" not in governance_bundle:
        logger.warning(
            "Governance enabled but no guardrails in bundle — tools will NOT be wrapped"
        )
        return

    from agent_builder.tools.policy_enforcing_tool import wrap_tools

    guardrails = governance_bundle["guardrails"]
    audit_sink = governance_bundle.get("audit_sink")
    tools_dict = result.get("tools")
    if not tools_dict:
        logger.info("No tools to wrap under governance")
        return

    tools_list = list(tools_dict.values())
    wrapped = wrap_tools(tools_list, guardrails, audit_sink)
    result["tools"] = {tool.name: tool for tool in wrapped}
    logger.info("Wrapped %d tools with PolicyEnforcingTool under governance", len(wrapped))

    # T12 — warn if governance is on but default policy is empty (fail-open risk)
    governance_cfg = config.get("governance", {})
    default_policy = governance_cfg.get("default_policy", {})
    permissions = default_policy.get("permissions", [])
    if not permissions:
        logger.warning(
            "Governance is enabled but default_policy.permissions is empty. "
            "Unless MongoDB-backed per-tenant policies exist, ALL tool calls "
            "will be denied."
        )

    # T12 — warn on governance + handoffs without explicit grants
    agents_cfg = config.get("agents", [])
    if agents_cfg:
        handoff_agents = [
            a.get("name") for a in agents_cfg
            if a.get("handoffs") or a.get("handoff_targets")
        ]
        if handoff_agents:
            transfer_grants = [
                p for p in permissions
                if p.startswith("tools.call.transfer_to_")
            ]
            if not transfer_grants and "*" not in permissions:
                logger.warning(
                    "Governance enabled with handoff agents (%s) but no "
                    "transfer_to_* permission grants found in default_policy. "
                    "Handoffs will be denied at runtime unless per-tenant "
                    "policies grant them.",
                    ", ".join(handoff_agents),
                )
