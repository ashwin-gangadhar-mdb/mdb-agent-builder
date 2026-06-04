import os

from agent_builder.app import create_app
from agent_builder.utils.logging_config import configure_logging, get_logger

# Configure logging (configure_logging accepts a string level name directly)
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
configure_logging(level=log_level)

logger = get_logger(__name__)

# Get configuration path from environment
config_path = os.environ.get("AGENT_CONFIG_PATH")
if not config_path:
    raise ValueError("AGENT_CONFIG_PATH environment variable must be set")

# Create Flask application
application = create_app(config_path)

# For direct execution
if __name__ == "__main__":
    application.run()
