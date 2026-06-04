import os
import tempfile
import textwrap
import unittest

from agent_builder.yaml_loader import load_yaml, resolve_env_variables


class YamlConfigTests(unittest.TestCase):
    def test_resolves_env_variable_default(self):
        self.assertEqual(resolve_env_variables("${MAAP_MISSING:-fallback}"), "fallback")

    def test_resolves_env_variable_value(self):
        os.environ["MAAP_TEST_VALUE"] = "actual"
        try:
            self.assertEqual(resolve_env_variables("${MAAP_TEST_VALUE:-fallback}"), "actual")
        finally:
            del os.environ["MAAP_TEST_VALUE"]

    def test_load_yaml_preserves_governance_config(self):
        config_text = textwrap.dedent(
            """
            governance:
              enabled: false
              connection_str: ${MONGODB_URI:-mongodb://localhost:27017}
              db_name: agent_control_plane
              default_policy:
                permissions:
                  - "*"
            """
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as config_file:
            config_file.write(config_text)
            config_path = config_file.name

        try:
            loaded = load_yaml(config_path)
        finally:
            os.unlink(config_path)

        self.assertFalse(loaded["governance"]["enabled"])
        self.assertEqual(loaded["governance"]["db_name"], "agent_control_plane")
        self.assertEqual(loaded["governance"]["default_policy"]["permissions"], ["*"])


if __name__ == "__main__":
    unittest.main()
