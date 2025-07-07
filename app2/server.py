from fastmcp import FastMCP
import mcp.types as types
import re
import asyncio
from typing import List

mcp = FastMCP("Enhanced MCP Server")

# In-memory session storage for chat history keyed by session id
session_histories = {}

# Helper: get or create chat history list for session
def get_history(session_id: str) -> List[str]:
    return session_histories.setdefault(session_id, [])

# Tools for arithmetic operations
@mcp.tool()
def add(a: int, b: int) -> tuple[list[types.Content], dict]:
    result = a + b
    content = [types.TextContent(type="text", text=f"{a} + {b} = {result}")]
    structured = {"a": a, "b": b, "result": result}
    return content, structured

@mcp.tool()
def subtract(a: int, b: int) -> tuple[list[types.Content], dict]:
    result = a - b
    content = [types.TextContent(type="text", text=f"{a} - {b} = {result}")]
    structured = {"a": a, "b": b, "result": result}
    return content, structured

@mcp.tool()
def multiply(a: int, b: int) -> tuple[list[types.Content], dict]:
    result = a * b
    content = [types.TextContent(type="text", text=f"{a} * {b} = {result}")]
    structured = {"a": a, "b": b, "result": result}
    return content, structured

@mcp.tool()
def divide(a: int, b: int) -> tuple[list[types.Content], dict]:
    if b == 0:
        content = [types.TextContent(type="text", text="Error: Division by zero.")]
        structured = {"error": "division_by_zero"}
        return content, structured
    result = a / b
    content = [types.TextContent(type="text", text=f"{a} / {b} = {result:.4f}")]
    structured = {"a": a, "b": b, "result": result}
    return content, structured

# Simulated external API call (async)
async def fetch_weather(location: str) -> str:
    # Simulate delay and fake weather data
    await asyncio.sleep(1)
    return f"The weather in {location} is sunny, 25°C."

@mcp.tool()
async def weather(location: str) -> tuple[list[types.Content], dict]:
    report = await fetch_weather(location)
    content = [types.TextContent(type="text", text=report)]
    structured = {"location": location, "report": report}
    return content, structured

# Async streaming example (simulate streaming chunks)
@mcp.tool()
async def stream_count(to: int) -> tuple[list[types.Content], dict]:
    content = []
    for i in range(1, to + 1):
        content.append(types.TextContent(type="text", text=f"Count: {i}"))
        await asyncio.sleep(0.2)  # simulate streaming delay
    structured = {"counted_to": to}
    return content, structured

# Chat tool with session context and fallback command parsing
@mcp.tool()
async def chat(message: str, session_id: str = "") -> tuple[list[types.Content], dict]:
    history = get_history(session_id)
    history.append(f"User: {message}")

    # Parse commands client might miss
    # Arithmetic commands
    for cmd, func in [("add", add), ("subtract", subtract), ("multiply", multiply), ("divide", divide)]:
        match = re.match(rf"{cmd} (\d+) and (\d+)", message.lower())
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            # Call the corresponding tool function directly
            content, structured = func(a, b)
            history.append(f"Assistant: {content[0].text}")
            return content, structured

    # Weather command
    match_weather = re.match(r"weather in ([a-zA-Z\s]+)", message.lower())
    if match_weather:
        location = match_weather.group(1).strip()
        report = await fetch_weather(location)
        content = [types.TextContent(type="text", text=report)]
        structured = {"location": location, "report": report}
        history.append(f"Assistant: {report}")
        return content, structured

    # Default echo
    response = f"Echo: {message}"
    content = [types.TextContent(type="text", text=response)]
    structured = {"message": message}
    history.append(f"Assistant: {response}")
    return content, structured

if __name__ == "__main__":
    mcp.run(transport="stdio")
