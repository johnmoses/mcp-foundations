# from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP
import mcp.types as types
import re

mcp = FastMCP("Combined Server")

@mcp.tool()
def add(a: int, b: int) -> tuple[list[types.Content], dict]:
    result = a + b
    content = [types.TextContent(type="text", text=f"{a} + {b} = {result}")]
    structured = {"a": a, "b": b, "result": result}
    return content, structured

@mcp.tool()
def multiply(a: int, b: int) -> tuple[list[types.Content], dict]:
    result = a * b
    content = [types.TextContent(type="text", text=f"{a} * {b} = {result}")]
    structured = {"a": a, "b": b, "result": result}
    return content, structured

@mcp.tool()
def chat(message: str) -> tuple[list[types.Content], dict]:
    # Handle addition command if client didn't parse it
    match_add = re.match(r"add (\d+) and (\d+)", message.lower())
    if match_add:
        a, b = int(match_add.group(1)), int(match_add.group(2))
        result = a + b
        content = [types.TextContent(type="text", text=f"(Chat) {a} + {b} = {result}")]
        structured = {"a": a, "b": b, "result": result}
        return content, structured

    # Handle multiplication command if client didn't parse it
    match_mul = re.match(r"multiply (\d+) and (\d+)", message.lower())
    if match_mul:
        a, b = int(match_mul.group(1)), int(match_mul.group(2))
        result = a * b
        content = [types.TextContent(type="text", text=f"(Chat) {a} * {b} = {result}")]
        structured = {"a": a, "b": b, "result": result}
        return content, structured

    # Default echo
    content = [types.TextContent(type="text", text=f"Echo: {message}")]
    structured = {"message": message}
    return content, structured

if __name__ == "__main__":
    mcp.run(transport="stdio")
