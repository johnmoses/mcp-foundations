import asyncio
import re
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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
    session_id = "user-session-1"  # simple static session id for demo

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Ending chat session...")
            break
        if not user_input:
            continue

        # Client-side parsing for some commands to call dedicated tools
        # Arithmetic commands
        for cmd in ["add", "subtract", "multiply", "divide"]:
            pattern = rf"{cmd} (\d+) and (\d+)"
            match = re.match(pattern, user_input.lower())
            if match:
                a, b = int(match.group(1)), int(match.group(2))
                result = await session.call_tool(cmd, {"a": a, "b": b})
                await display_tool_result(result)
                break
        else:
            # Weather command
            match_weather = re.match(r"weather in ([a-zA-Z\s]+)", user_input.lower())
            if match_weather:
                location = match_weather.group(1).strip()
                result = await session.call_tool("weather", {"location": location})
                await display_tool_result(result)
            else:
                # Fallback: send to chat tool with session_id for context
                response = await session.call_tool("chat", {"message": user_input, "session_id": session_id})
                await display_tool_result(response)

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
    )

    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            await chat_loop(session)

if __name__ == "__main__":
    asyncio.run(main())
