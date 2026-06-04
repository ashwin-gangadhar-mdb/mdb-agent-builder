"""Illustrative MCP server placeholder for the YAML examples.

Replace this file with a real MCP server implementation when running
`examples/tool_call_mcp_agent.yaml`.
"""


def get_weather(location: str) -> str:
    """Example tool signature for weather lookup."""
    return f"Weather lookup is not implemented for {location}."


def calculate(expression: str) -> str:
    """Example tool signature for calculation."""
    return f"Calculation is not implemented for {expression}."
