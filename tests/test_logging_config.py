import logging
import unittest

from agent_builder.utils.logger import AgentLogger
from agent_builder.utils.logger import logger as shim_logger
from agent_builder.utils.logging_config import configure_logging, get_logger


class ConfigureLoggingTests(unittest.TestCase):
    def test_accepts_string_level(self):
        # Regression: configure_logging used to require a string while its
        # default was an int — calling with a string name must work.
        configure_logging(level="DEBUG")
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_accepts_numeric_level(self):
        # Regression: wsgi.py passed a numeric level; that must not raise.
        configure_logging(level=logging.WARNING)
        self.assertEqual(logging.getLogger().level, logging.WARNING)
        # Restore a sane default for other tests.
        configure_logging(level="INFO")

    def test_invalid_level_raises(self):
        with self.assertRaises(ValueError):
            configure_logging(level="NOT_A_LEVEL")


class LoggingShimTests(unittest.TestCase):
    def test_agentlogger_delegates_without_duplicate_handlers(self):
        # The legacy AgentLogger must reuse the centrally-configured logger and
        # never attach its own (previously it added a fresh file handler each
        # time, duplicating log lines and leaking handles).
        name = "agent_builder.tests.shim"
        first = AgentLogger(name).get_logger()
        before = len(first.handlers)
        second = AgentLogger(name).get_logger()
        self.assertIs(first, second)
        self.assertIs(first, get_logger(name))
        self.assertEqual(len(second.handlers), before)

    def test_module_level_shim_logger_is_a_real_logger(self):
        self.assertIsInstance(shim_logger, logging.Logger)


if __name__ == "__main__":
    unittest.main()
