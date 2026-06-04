"""
Backward-compatibility shim for the legacy ``AgentLogger`` API.

The project now has a single logging entry point in
:mod:`agent_builder.utils.logging_config`.  This module is kept only so that
older imports (``from agent_builder.utils.logger import logger`` /
``AgentLogger``) keep working; everything delegates to ``get_logger`` so there
is exactly one set of handlers per logger name and no duplicate file handlers.

Prefer importing :func:`agent_builder.utils.logging_config.get_logger`
directly in new code.
"""

from agent_builder.utils.logging_config import get_logger


class AgentLogger:
    """Thin wrapper kept for backward compatibility.

    Historically this class attached its own console + file handlers (and even
    printed on construction).  That caused duplicate log lines and leaked file
    handles when instantiated repeatedly.  It now simply delegates to the
    centralised :func:`get_logger`.
    """

    def __init__(self, name: str = "maap-agent-builder", level: str = "INFO"):
        self._logger = get_logger(name)

    def get_logger(self):
        """Return the underlying centrally-configured logger."""
        return self._logger


# Module-level logger preserved for ``from ... import logger`` call sites.
logger = get_logger("maap-agent-builder")
