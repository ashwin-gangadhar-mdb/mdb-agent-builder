import importlib
import inspect
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PlanExecuteReplanTests(unittest.TestCase):
    def test_correctly_spelled_module_holds_the_implementation(self):
        # The implementation used to live in the misspelled
        # ``plan_excute_replan`` module behind a shim. It must now live in the
        # correctly-spelled module, and the typo module must be gone.
        module = importlib.import_module(
            "agent_builder.agents.plan_execute_replan"
        )
        self.assertTrue(hasattr(module, "create_plan_execute_replan_agent"))
        self.assertFalse(
            (ROOT / "agent_builder/agents/plan_excute_replan.py").exists(),
            "misspelled plan_excute_replan.py should have been removed",
        )

    def test_checkpointer_default_is_not_a_shared_instance(self):
        # Mutable-default regression: the default must be None (a fresh saver is
        # created per call) rather than a single shared InMemorySaver().
        from agent_builder.agents.plan_execute_replan import (
            create_plan_execute_replan_agent,
        )

        default = inspect.signature(
            create_plan_execute_replan_agent
        ).parameters["checkpointer"].default
        self.assertIsNone(default)

    def test_factory_resolves_to_correctly_spelled_module(self):
        from agent_builder.agents.agent_gen import AgentFactory, AgentType

        module_path, function_name = AgentFactory._AGENT_CREATORS[
            AgentType.PLAN_EXECUTE_REPLAN
        ]
        self.assertEqual(module_path, "agent_builder.agents.plan_execute_replan")
        self.assertEqual(function_name, "create_plan_execute_replan_agent")


class ReflectionAgentTests(unittest.TestCase):
    def test_unused_response_schema_param_removed(self):
        from agent_builder.agents.reflection import create_basic_reflection_agent

        params = inspect.signature(create_basic_reflection_agent).parameters
        self.assertNotIn("response_schema", params)

    def test_no_bogus_temperature_invoke_kwarg(self):
        source = (ROOT / "agent_builder/agents/reflection.py").read_text()
        self.assertNotIn("temperature=0.0", source)
        # The ReAct sub-agent should be constructed once, not per iteration.
        self.assertEqual(source.count("create_react_agent("), 1)


class AgentGenTypingTests(unittest.TestCase):
    def test_no_builtin_generic_annotations(self):
        # Builtin generics (list[str]/tuple[...]) in evaluated annotations break
        # on older interpreters and were inconsistent with the rest of the file.
        source = (ROOT / "agent_builder/agents/agent_gen.py").read_text()
        self.assertNotIn("-> list[", source)
        self.assertNotIn("tuple[str, str]", source)


if __name__ == "__main__":
    unittest.main()
