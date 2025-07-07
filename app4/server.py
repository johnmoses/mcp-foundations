from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="Interactive Chat Server")

@mcp.tool()
def chat(message: str) -> str:
    # Your chat logic here
    return f"You said: {message}"

@mcp.tool()
def weather(city: str) -> str:
    return f"The weather in {city} is sunny with 25°C."

@mcp.tool()
def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": None})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    mcp.run()
