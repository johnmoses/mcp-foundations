import multiprocessing
import time
import asyncio
import re
from app import create_app  # Your Flask app factory
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# --- Flask server process ---
def run_flask():
    app = create_app()
    # Disable reloader to avoid double spawning
    print("[Flask] Starting Flask server...")
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
    print("[Flask] Flask server stopped.")

# --- MCP client helper functions ---
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
    print("[MCP Client] Chat session started. Type 'exit' or 'quit' to end.")
    session_id = "user-session-1"  # Manage session IDs as needed

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("[MCP Client] Ending chat session...")
            break
        if not user_input:
            continue

        # Command routing for arithmetic tools
        for cmd in ["add", "subtract", "multiply", "divide"]:
            pattern = rf"{cmd} (\d+) and (\d+)"
            match = re.match(pattern, user_input.lower())
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                print(f"[MCP Client] Calling tool '{cmd}' with args: a={a}, b={b}")
                result = await session.call_tool(cmd, {"a": a, "b": b})
                await display_tool_result(result)
                break
        else:
            # Weather command
            match_weather = re.match(r"weather in ([a-zA-Z\s]+)", user_input.lower())
            if match_weather:
                location = match_weather.group(1).strip()
                print(f"[MCP Client] Calling tool 'weather' with location: {location}")
                result = await session.call_tool("weather", {"location": location})
                await display_tool_result(result)
            else:
                # Default to chat tool
                print(f"[MCP Client] Calling tool 'chat' with message: {user_input}")
                response = await session.call_tool("chat", {"message": user_input, "session_id": session_id})
                await display_tool_result(response)

async def connect_with_retry(server_params, max_retries=10, delay=2):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[MCP Client] Attempt {attempt} to connect to MCP server...")
            async with stdio_client(server_params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    print("[MCP Client] Connected to MCP server!")
                    return session
        except Exception as e:
            print(f"[MCP Client] Connection attempt {attempt} failed: {e}")
            await asyncio.sleep(delay * (2 ** (attempt - 1)))  # exponential backoff
    raise RuntimeError("[MCP Client] Failed to connect to MCP server after multiple retries")

async def run_mcp_client():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],  # Adjust path to your MCP server script
    )
    session = await connect_with_retry(server_params)
    await chat_loop(session)

def start_mcp_client_process():
    print("[MCP Client] Starting MCP client process...")
    asyncio.run(run_mcp_client())
    print("[MCP Client] MCP client process finished.")

# --- Main entrypoint ---
if __name__ == "__main__":
    flask_process = multiprocessing.Process(target=run_flask)
    mcp_process = multiprocessing.Process(target=start_mcp_client_process)

    print("[Main] Starting Flask server process...")
    flask_process.start()

    # Optional: wait a few seconds to ensure Flask and MCP server are ready
    time.sleep(3)

    print("[Main] Starting MCP client process...")
    mcp_process.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received. Shutting down processes...")
        flask_process.terminate()
        mcp_process.terminate()
        flask_process.join()
        mcp_process.join()
        print("[Main] Processes terminated. Exiting.")
