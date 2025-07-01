""" 
MCP Server Wrapping Flask API (`mcp_server.py`)
"""

from mcp.server.fastmcp import FastMCP
import requests

# Initialize MCP server
mcp = FastMCP("To-Do API MCP Server")


@mcp.resource("todo://list")
def list_tasks() -> list:
    """Fetch all tasks from the Flask API."""
    response = requests.get("http://localhost:5001/tasks")
    response.raise_for_status()
    return response.json()


@mcp.tool()
def add_task(title: str) -> dict:
    """Add a new task via the Flask API."""
    payload = {"title": title}
    response = requests.post("http://localhost:5001/tasks", json=payload)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    # Run MCP server with stdio transport for local testing
    mcp.run(transport="stdio")
