# app/services/mcp_client.py

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClientWrapper:
    def __init__(self, server_command="python", server_args=None):
        if server_args is None:
            server_args = ["mcp_server.py"]
        self.server_params = StdioServerParameters(
            command=server_command, args=server_args
        )
        self.session = None
        self._stdio_client = None

    async def __aenter__(self):
        self._stdio_client = await stdio_client(self.server_params).__aenter__()
        reader, writer = self._stdio_client
        self.session = await ClientSession(reader, writer).__aenter__()
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.__aexit__(exc_type, exc, tb)
        if self._stdio_client:
            await self._stdio_client.__aexit__(exc_type, exc, tb)

    async def call_tool(self, tool_name: str, params: dict = None):
        if self.session is None:
            raise RuntimeError("MCPClientWrapper session is not initialized. Use 'async with MCPClientWrapper()'.")
        if params is None:
            params = {}
        return await self.session.call_tool(tool_name, params)

    async def list_todos(self, query: str = None):
        params = {"query": query} if query else {}
        return await self.call_tool("list_todos", params)

    async def add_todo(self, task: str):
        return await self.call_tool("add_todo", {"task": task})

    async def calculate(self, expression: str):
        return await self.call_tool("calculate", {"expression": expression})
