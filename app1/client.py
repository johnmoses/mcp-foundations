import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(command="python", args=["server.py"])

    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            print("Type your messages (type 'exit' to quit):")
            while True:
                user_input = input("You: ")
                if user_input.lower() in ("exit", "quit"):
                    break
                response = await session.call_tool("chat", {"message": user_input})
                print(f"Assistant: {response}")

if __name__ == "__main__":
    asyncio.run(main())
