from mcp.server.fastmcp import FastMCP

# Create FastMCP server instance with a friendly name
mcp = FastMCP("Interactive Chat Server")

# Simple in-memory conversation history (per server instance)
conversation_history = []

@mcp.tool()
def chat(message: str) -> str:
    """
    A simple chat tool that echoes user messages and keeps conversation history.
    """
    conversation_history.append(f"User: {message}")
    response = f"Echo ({len(conversation_history)}): {message}"
    conversation_history.append(f"Assistant: {response}")
    return response

if __name__ == "__main__":
    # Run the server with stdio transport (default)
    mcp.run(transport="stdio")
