import asyncio
import os
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPHealthcareClient:
    def __init__(self, command="python", args=None, timeout=30):
        if args is None:
            args = [os.path.abspath("mcp_server.py")]
        self.server_params = StdioServerParameters(command=command, args=args)
        self.session = None
        self._client_context = None
        self._timeout = timeout

    async def __aenter__(self):
        logger.info("Starting MCP client connection...")
        self._client_context = stdio_client(self.server_params)
        try:
            read, write = await asyncio.wait_for(self._client_context.__aenter__(), timeout=self._timeout)
            self.session = ClientSession(read, write)
            await asyncio.wait_for(self.session.initialize(), timeout=self._timeout)
            logger.info("MCP client connected and initialized.")
            return self
        except asyncio.TimeoutError:
            logger.error("Timeout initializing MCP client")
            await self.__aexit__(None, None, None)
            raise RuntimeError("Timeout initializing MCP client")

    async def __aexit__(self, exc_type, exc, tb):
        logger.info("Closing MCP client connection...")
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                logger.warning(f"Error closing MCP session: {e}")
            self.session = None
        if self._client_context:
            try:
                await self._client_context.__aexit__(exc_type, exc, tb)
            except Exception as e:
                logger.warning(f"Error closing stdio client: {e}")
            self._client_context = None
        logger.info("MCP client closed.")

    async def call_tool(self, tool_name: str, input_str: str):
        if not self.session:
            raise RuntimeError("Client session not initialized. Use 'async with MCPHealthcareClient()'.")
        try:
            response = await self.session.call(tool_name, input_str)
            return response
        except Exception as e:
            logger.error(f"Error calling MCP tool '{tool_name}': {e}")
            raise

# Helper to run async from sync code if needed
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        new_loop = asyncio.new_event_loop()
        result = new_loop.run_until_complete(coro)
        new_loop.close()
        return result
    else:
        return loop.run_until_complete(coro)
