# from mcp.client.stdio import stdio_client, StdioServerParameters
# from mcp import ClientSession

from app import create_app

app = create_app()

# server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

# mcp_session = None

# async def init_mcp_session():
#     global mcp_session
#     if mcp_session is None:
#         reader, writer = await stdio_client(server_params)
#         mcp_session = await ClientSession(reader, writer).__aenter__()
#         await mcp_session.initialize()

# async def process_user_query(user_input):
#     await init_mcp_session()


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5002)
