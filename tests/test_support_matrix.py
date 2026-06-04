import importlib
import re
from pathlib import Path
import unittest

from agent_builder.agents.agent_gen import AgentFactory


ROOT = Path(__file__).resolve().parents[1]


def read_source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def extract_quoted_keys(source):
    return set(re.findall(r'^\s+"([a-zA-Z0-9_]+)"\s*:', source, flags=re.MULTILINE))


def module_path_to_file(module_path):
    relative_path = Path(*module_path.split(".")).with_suffix(".py")
    return ROOT / relative_path


class SupportMatrixTests(unittest.TestCase):
    def test_all_configured_agent_creator_modules_are_importable(self):
        for agent_type, creator in AgentFactory._AGENT_CREATORS.items():
            with self.subTest(agent_type=agent_type.value):
                module_path, function_name = creator
                if module_path == "agent_builder.agents.agent_gen":
                    module = importlib.import_module(module_path)
                    self.assertTrue(hasattr(module, function_name))
                    continue
                source_path = module_path_to_file(module_path)
                self.assertTrue(source_path.exists(), f"Missing {source_path}")
                self.assertIn(f"def {function_name}", source_path.read_text())

    def test_documented_agent_types_are_registered(self):
        source = read_source("agent_builder/agents/loader.py")
        agent_config = {
            key
            for key in [
                "react",
                "tool_call",
                "reflect",
                "plan_execute_replan",
                "long_term_memory",
            ]
            if f'"{key}":' in source
        }
        self.assertEqual(
            agent_config,
            {"react", "tool_call", "reflect", "plan_execute_replan", "long_term_memory"},
        )

    def test_tool_types_are_registered(self):
        source = read_source("agent_builder/tools/loader.py")
        tool_types = set(re.findall(r'= "([a-z_]+)"', source))
        self.assertEqual(
            tool_types,
            {
                "vector_search",
                "mongodb_toolkit",
                "nl_to_mql",
                "mcp",
                "full_text_search",
            },
        )

    def test_llm_provider_matrix(self):
        # Provider registries now live in the adapters module (adapter pattern).
        source = read_source("agent_builder/llms/adapters.py")
        provider_config = {
            key
            for key in [
                "bedrock",
                "fireworks",
                "together",
                "cohere",
                "anthropic",
                "azure",
                "ollama",
                "sagemaker",
            ]
            if f'"{key}":' in source
        }
        self.assertEqual(
            provider_config,
            {
                "bedrock",
                "fireworks",
                "together",
                "cohere",
                "anthropic",
                "azure",
                "ollama",
                "sagemaker",
            },
        )

    def test_embedding_provider_matrix(self):
        # Provider registries now live in the adapters module (adapter pattern).
        source = read_source("agent_builder/embeddings/adapters.py")
        provider_config = {
            key
            for key in [
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
            ]
            if f'"{key}":' in source
        }
        self.assertEqual(
            provider_config,
            {
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
            },
        )


if __name__ == "__main__":
    unittest.main()
