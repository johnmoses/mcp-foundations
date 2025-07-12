import multiprocessing
import time
import asyncio
import re
from app import create_app  # Your Flask app factory
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

def run_flask():
    app = create_app()
    # Disable reloader to prevent double spawning
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)

async def display_tool_result(result):
    if isinstance(result, tuple) and len(result) == 2:
        content_blocks, structured_data = result
        print("\n--- Human-readable output ---")
        for block in content_blocks:
            if hasattr(block, "text"):
                print(block.text)
        print("\n--- Structured data ---")
        print(structured_data)
    else:
        print("Result:", result)

async def chat_loop(session: ClientSession):
    print("Chat session started. Type 'exit' or 'quit' to end.")
    session_id = "user-session-1"

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Ending chat session...")
            break
        if not user_input:
            continue

        for cmd in ["add", "subtract", "multiply", "divide"]:
            pattern = rf"{cmd} (\d+) and (\d+)"
            match = re.match(pattern, user_input.lower())
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                result = await session.call_tool(cmd, {"a": a, "b": b})
                await display_tool_result(result)
                break
        else:
            match_weather = re.match(r"weather in ([a-zA-Z\s]+)", user_input.lower())
            if match_weather:
                location = match_weather.group(1).strip()
                result = await session.call_tool("weather", {"location": location})
                await display_tool_result(result)
            else:
                response = await session.call_tool("chat", {"message": user_input, "session_id": session_id})
                await display_tool_result(response)

async def connect_with_retry(server_params, max_retries=10, delay=2):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempt {attempt} to connect to MCP server...")
            async with stdio_client(server_params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    print("Connected to MCP server!")
                    return session
        except Exception as e:
            print(f"Connection attempt {attempt} failed: {e}")
            await asyncio.sleep(delay * (2 ** (attempt - 1)))  # exponential backoff
    raise RuntimeError("Failed to connect to MCP server after multiple retries")

async def run_mcp_client():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],  # Adjust path to your MCP server script
    )
    session = await connect_with_retry(server_params)
    await chat_loop(session)

def start_mcp_client_process():
    asyncio.run(run_mcp_client())

if __name__ == "__main__":
    flask_process = multiprocessing.Process(target=run_flask)
    mcp_process = multiprocessing.Process(target=start_mcp_client_process)

    flask_process.start()
    mcp_process.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down processes...")
        flask_process.terminate()
        mcp_process.terminate()
        flask_process.join()
        mcp_process.join()
