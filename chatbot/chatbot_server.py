""" 
Chatbot server for chatbot tools
"""

from flask import Flask, request, jsonify
from datetime import datetime
from typing import Dict
class MCPChatbotServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.tools = {}
        self.conversation_memory = []
        self.setup_routes()
        self.setup_chatbot_tools()

    def setup_routes(self):
        """Setup Flask routes for MCP protocol"""

        @self.app.route("/mcp/initialize", methods=["POST"])
        def initialize():
            data = request.get_json()
            response = {
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "MCP Chatbot Server", "version": "1.0.0"},
                },
            }
            return jsonify(response)

        @self.app.route("/mcp/tools/list", methods=["POST"])
        def list_tools():
            data = request.get_json()
            tools_list = [
                {
                    "name": name,
                    "description": tool["description"],
                    "inputSchema": tool["inputSchema"],
                }
                for name, tool in self.tools.items()
            ]

            response = {
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "result": {"tools": tools_list},
            }
            return jsonify(response)

        @self.app.route("/mcp/tools/call", methods=["POST"])
        def call_tool():
            data = request.get_json()
            tool_name = data["params"]["name"]
            arguments = data["params"].get("arguments", {})

            if tool_name not in self.tools:
                return jsonify(
                    {
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "error": {
                            "code": -32601,
                            "message": f"Tool '{tool_name}' not found",
                        },
                    }
                )

            try:
                result = self.tools[tool_name]["handler"](arguments)
                response = {
                    "jsonrpc": "2.0",
                    "id": data.get("id"),
                    "result": {"content": [{"type": "text", "text": str(result)}]},
                }
                return jsonify(response)
            except Exception as e:
                return jsonify(
                    {
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "error": {
                            "code": -32603,
                            "message": f"Tool execution failed: {str(e)}",
                        },
                    }
                )

    def setup_chatbot_tools(self):
        """Setup tools specific for chatbot functionality"""

        # Weather tool (simulated)
        def weather_tool(args):
            city = args.get("city", "Unknown")
            weather_data = {
                "london": "Cloudy, 15°C",
                "new york": "Sunny, 22°C",
                "tokyo": "Rainy, 18°C",
                "paris": "Partly cloudy, 17°C",
                "sydney": "Sunny, 25°C",
            }
            return weather_data.get(
                city.lower(), f"Weather data not available for {city}"
            )

        self.add_tool(
            name="get_weather",
            description="Get current weather information for a city",
            input_schema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Name of the city"}
                },
                "required": ["city"],
            },
            handler=weather_tool,
        )

        # Calculator tool
        def calculator_tool(args):
            expression = args.get("expression", "")
            try:
                # Simple eval for demo - in production, use a proper math parser
                result = eval(expression.replace("^", "**"))
                return f"{expression} = {result}"
            except:  # noqa: E722
                return f"Invalid mathematical expression: {expression}"

        self.add_tool(
            name="calculate",
            description="Perform mathematical calculations",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
            handler=calculator_tool,
        )

        # Time tool
        def time_tool(args):
            timezone = args.get("timezone", "UTC")
            current_time = datetime.now()
            return f"Current time ({timezone}): {current_time.strftime('%Y-%m-%d %H:%M:%S')}"

        self.add_tool(
            name="get_time",
            description="Get current time",
            input_schema={
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone (optional)"}
                },
            },
            handler=time_tool,
        )

        # Memory tool
        def remember_tool(args):
            info = args.get("information", "")
            self.conversation_memory.append(
                {"timestamp": datetime.now().isoformat(), "information": info}
            )
            return f"I'll remember: {info}"

        self.add_tool(
            name="remember",
            description="Remember important information from the conversation",
            input_schema={
                "type": "object",
                "properties": {
                    "information": {
                        "type": "string",
                        "description": "Information to remember",
                    }
                },
                "required": ["information"],
            },
            handler=remember_tool,
        )

        # Recall tool
        def recall_tool(args):
            if not self.conversation_memory:
                return "I don't have any stored memories yet."

            memories = "\n".join(
                [
                    f"- {mem['information']} (remembered at {mem['timestamp'][:19]})"
                    for mem in self.conversation_memory[-5:]  # Last 5 memories
                ]
            )
            return f"Here's what I remember:\n{memories}"

        self.add_tool(
            name="recall",
            description="Recall previously stored information",
            input_schema={"type": "object", "properties": {}},
            handler=recall_tool,
        )

    def add_tool(self, name: str, description: str, input_schema: Dict, handler):
        """Add a tool to the server"""
        self.tools[name] = {
            "description": description,
            "inputSchema": input_schema,
            "handler": handler,
        }

    def run(self, host="localhost", port=5001, debug=False):
        """Run the MCP server"""
        print(f"Starting MCP Chatbot Server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

