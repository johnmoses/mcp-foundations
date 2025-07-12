# services/mcp_client.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class MCPClientWrapper:
    def __init__(self, server_command="python", server_args=None):
        if server_args is None:
            server_args = ["mcp_server.py"]
        self.server_params = StdioServerParameters(command=server_command, args=server_args)
        self.session = None

    async def __aenter__(self):
        self._stdio_client = await stdio_client(self.server_params).__aenter__()
        reader, writer = self._stdio_client
        self.session = await ClientSession(reader, writer).__aenter__()
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.__aexit__(exc_type, exc, tb)
        await self._stdio_client.__aexit__(exc_type, exc, tb)

    async def rag_search(self, query: str):
        return await self.session.call_tool("rag_search", {"query": query})

    async def calculate(self, expression: str):
        return await self.session.call_tool("calculate", {"expression": expression})

    async def generate_quiz(self, topic: str, num_questions: int = 5):
        return await self.session.call_tool("generate_quiz", {"topic": topic, "num_questions": num_questions})

# Usage example
async def main():
    async with MCPClientWrapper() as client:
        result = await client.rag_search("Hello MCP")
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
