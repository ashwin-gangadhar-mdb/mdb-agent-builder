from pathlib import Path
import unittest

from agent_builder.yaml_loader import load_yaml


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

SUPPORTED_AGENT_TYPES = {
    "react",
    "tool_call",
    "reflect",
    "plan_execute_replan",
    "long_term_memory",
}
SUPPORTED_LLM_PROVIDERS = {
    "bedrock",
    "fireworks",
    "together",
    "cohere",
    "anthropic",
    "azure",
    "ollama",
    "sagemaker",
}
SUPPORTED_EMBEDDING_PROVIDERS = {
    "bedrock",
    "sagemaker",
    "vertexai",
    "azure",
    "together",
    "fireworks",
    "cohere",
    "voyageai",
    "ollama",
    "huggingface",
}
SUPPORTED_TOOL_TYPES = {
    "vector_search",
    "mongodb_toolkit",
    "nl_to_mql",
    "mcp",
    "full_text_search",
}


class ExampleConfigTests(unittest.TestCase):
    def test_expected_example_patterns_exist(self):
        expected = {
            "react_rag_mongodb.yaml",
            "tool_call_mcp_agent.yaml",
            "reflection_quality_reviewer.yaml",
            "plan_execute_replan_research.yaml",
            "long_term_memory_assistant.yaml",
            "governed_enterprise_support.yaml",
        }

        self.assertEqual({path.name for path in EXAMPLES.glob("*.yaml")}, expected)

    def test_all_examples_are_structurally_valid(self):
        for path in sorted(EXAMPLES.glob("*.yaml")):
            with self.subTest(example=path.name):
                config = load_yaml(str(path))
                self.assertIn("agent", config)
                self._assert_agent_is_valid(config)
                self._assert_llms_are_valid(config)
                self._assert_embeddings_are_valid(config)
                self._assert_tools_are_valid(config)
                self._assert_agent_references_are_valid(config)
                self._assert_prompt_paths_exist(config)

    def test_governed_example_enables_control_plane(self):
        config = load_yaml(str(EXAMPLES / "governed_enterprise_support.yaml"))

        self.assertTrue(config["governance"]["enabled"])
        self.assertEqual(
            config["governance"]["default_policy"]["permissions"],
            ["tools.call.support_docs"],
        )
        self.assertIn("audit", config["governance"])
        self.assertIn("state", config["governance"])

    def test_reflection_example_has_both_prompts(self):
        config = load_yaml(str(EXAMPLES / "reflection_quality_reviewer.yaml"))
        agent = config["agent"]

        self.assertEqual(agent["agent_type"], "reflect")
        self.assertIn("system_prompt_path", agent)
        self.assertIn("reflection_prompt_path", agent)

    def test_long_term_memory_example_has_memory_config(self):
        config = load_yaml(str(EXAMPLES / "long_term_memory_assistant.yaml"))
        agent = config["agent"]

        self.assertEqual(agent["agent_type"], "long_term_memory")

        # Adapter-powered path: memory adapters are declared in the top-level
        # memory: section and referenced by name from the agent.
        if "memory" in config:
            memory_names = {m["name"] for m in config["memory"]}
            # Each referenced adapter name must exist in the memory section.
            for field in ("episodic_memory", "observational_memory"):
                if field in agent:
                    self.assertIn(
                        agent[field],
                        memory_names,
                        f"agent.{field} references unknown memory adapter '{agent[field]}'",
                    )
            # Each memory entry must have the required fields.
            for mem in config["memory"]:
                self.assertIn("name", mem)
                self.assertIn("memory_type", mem)
                self.assertIn(mem["memory_type"], {"episodic", "observational", "general"})
                self.assertIn("connection_str", mem)
                self.assertIn("namespace", mem)
        else:
            # Legacy path: connection_str + namespace on the agent itself.
            self.assertIn("connection_str", agent)
            self.assertIn("namespace", agent)

    def _assert_agent_is_valid(self, config):
        agent = config["agent"]
        self.assertIn(agent["agent_type"], SUPPORTED_AGENT_TYPES)
        self.assertIn("name", agent)
        self.assertIn("llm", agent)

    def _assert_llms_are_valid(self, config):
        llms = config.get("llms", [])
        self.assertGreater(len(llms), 0)
        for llm in llms:
            self.assertIn(llm["provider"], SUPPORTED_LLM_PROVIDERS)
            self.assertIn("name", llm)
            self.assertIn("model_name", llm)

    def _assert_embeddings_are_valid(self, config):
        for embedding in config.get("embeddings", []):
            self.assertIn(embedding["provider"], SUPPORTED_EMBEDDING_PROVIDERS)
            self.assertIn("name", embedding)
            self.assertIn("model_name", embedding)

    def _assert_tools_are_valid(self, config):
        embeddings = {embedding["name"] for embedding in config.get("embeddings", [])}
        for tool in config.get("tools", []):
            self.assertIn(tool["tool_type"], SUPPORTED_TOOL_TYPES)
            self.assertIn("name", tool)
            if tool["tool_type"] == "vector_search":
                self.assertIn(tool["embedding_model"], embeddings)
            if tool["tool_type"] == "mcp":
                self.assertIn("servers_config", tool)

    def _assert_agent_references_are_valid(self, config):
        llms = {llm["name"] for llm in config.get("llms", [])}
        tools = {tool["name"] for tool in config.get("tools", [])}
        agent = config["agent"]

        self.assertIn(agent["llm"], llms)
        for tool_name in agent.get("tools", []):
            self.assertIn(tool_name, tools)

    def _assert_prompt_paths_exist(self, config):
        agent = config["agent"]
        for key in ["system_prompt_path", "reflection_prompt_path"]:
            if key in agent:
                prompt_path = ROOT / agent[key]
                self.assertTrue(prompt_path.exists(), f"Missing prompt: {prompt_path}")


if __name__ == "__main__":
    unittest.main()
