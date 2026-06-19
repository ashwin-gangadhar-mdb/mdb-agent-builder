#!/usr/bin/env python3
"""
Command Line Interface for MDB Agent Builder.

This module provides a command-line interface to the MDB Agent Builder,
allowing users to start the server with configuration options.
"""

import argparse
import sys

from agent_builder.app import AgentApp
from agent_builder.utils.logging_config import configure_logging, get_logger

# Initialize logger
logger = get_logger(__name__)


def main():
    """Main entry point for the MDB Agent Builder CLI."""
    parser = argparse.ArgumentParser(
        description="MDB Agent Builder Command Line Interface"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Server command
    server_parser = subparsers.add_parser("serve", help="Run the agent as a web server")
    server_parser.add_argument(
        "--config", "-c", required=True, help="Path to the YAML configuration file"
    )
    server_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to run the server on (default: 127.0.0.1; "
        "use 0.0.0.0 to expose on the network)",
    )
    server_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)",
    )
    server_parser.add_argument(
        "--debug", "-d", action="store_true", help="Run in debug mode"
    )
    server_parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    server_parser.add_argument(
        "--env-file", "-e", help="Path to a .env file to load environment variables"
    )

    # Parse arguments
    args = parser.parse_args()

    # Configure logging based on command-line arguments
    # Pass the string log level directly, not the numeric value
    log_level = args.log_level if hasattr(args, "log_level") else "INFO"
    configure_logging(level=log_level)

    # Execute the command
    if not args.command:
        parser.print_help()
        return 1

    if args.command == "serve":
        try:
            logger.info("Starting agent server with configuration: %s", args.config)
            if args.env_file:
                from dotenv import load_dotenv

                load_dotenv(args.env_file)
                logger.info("Loaded environment variables from %s", args.env_file)
            agent_app = AgentApp(args.config)
            agent_app.run(host=args.host, port=args.port, debug=args.debug)
        except Exception as e:
            logger.exception("Failed to start server: %s", str(e))
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
