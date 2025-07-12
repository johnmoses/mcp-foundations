# mcp_servers/combined_mcp_server.py

from fastmcp import FastMCP
import mcp.types as types
import re
import asyncio
from typing import List

# Declare globals
rag = None
call_llm = None
generate_quiz_text = None

def initialize_services():
    global rag, call_llm, generate_quiz_text
    from app.services.service import rag as rag_service, call_llm as llm_callable, generate_quiz_text as quiz_func
    rag = rag_service
    call_llm = llm_callable
    generate_quiz_text = quiz_func

mcp = FastMCP("Enhanced MCP Server")

# In-memory session storage for chat history keyed by session id
session_histories = {}

def get_history(session_id: str) -> List[str]:
    return session_histories.setdefault(session_id, [])

# Arithmetic tools
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

# Simulated async external API call for weather
async def fetch_weather(location: str) -> str:
    await asyncio.sleep(1)  # Simulate delay
    return f"The weather in {location} is sunny, 25°C."

@mcp.tool()
async def weather(location: str) -> tuple[list[types.Content], dict]:
    report = await fetch_weather(location)
    content = [types.TextContent(type="text", text=report)]
    structured = {"location": location, "report": report}
    return content, structured

# Async streaming example
@mcp.tool()
async def stream_count(to: int) -> tuple[list[types.Content], dict]:
    content = []
    for i in range(1, to + 1):
        content.append(types.TextContent(type="text", text=f"Count: {i}"))
        await asyncio.sleep(0.2)
    structured = {"counted_to": to}
    return content, structured

# Chat tool with session context and command parsing
@mcp.tool()
async def chat(message: str, session_id: str = "") -> tuple[list[types.Content], dict]:
    history = get_history(session_id)
    history.append(f"User: {message}")

    # Parse arithmetic commands
    for cmd, func in [("add", add), ("subtract", subtract), ("multiply", multiply), ("divide", divide)]:
        match = re.match(rf"{cmd} (\d+) and (\d+)", message.lower())
        if match:
            a, b = int(match.group(1)), int(match.group(2))
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

# --- New extended tools ---

@mcp.tool()
async def rag_search(query: str, session_id: str = "") -> tuple[list[types.Content], dict]:
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, lambda: rag.chat(query, [], call_llm))
    content = [types.TextContent(type="text", text=answer)]
    structured = {"answer": answer}
    return content, structured

@mcp.tool()
async def generate_quiz(topic: str, num_questions: int = 5) -> tuple[list[types.Content], dict]:
    loop = asyncio.get_event_loop()
    questions = await loop.run_in_executor(None, lambda: generate_quiz_text(topic, num_questions))
    formatted = "\n\n".join(
        f"Q{i+1}: {q['question']}\nOptions: {', '.join(q['options'])}" for i, q in enumerate(questions)
    )
    content = [types.TextContent(type="text", text=formatted)]
    structured = {"topic": topic, "num_questions": num_questions}
    return content, structured
    
if __name__ == "__main__":
    # Run MCP server with SSE transport on port 8000 (change as needed)
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
