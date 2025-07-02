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
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Ending chat session...")
            break
        if not user_input:
            continue

        # Client parses addition command
        match_add = re.match(r"add (\d+) and (\d+)", user_input.lower())
        if match_add:
            a, b = int(match_add.group(1)), int(match_add.group(2))
            result = await session.call_tool("add", {"a": a, "b": b})
            await display_tool_result(result)
            continue

        # Client parses multiplication command
        match_mul = re.match(r"multiply (\d+) and (\d+)", user_input.lower())
        if match_mul:
            a, b = int(match_mul.group(1)), int(match_mul.group(2))
            result = await session.call_tool("multiply", {"a": a, "b": b})
            await display_tool_result(result)
            continue

        # Fallback: send to chat tool
        response = await session.call_tool("chat", {"message": user_input})
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
