# MCP Chatbot Implementation with Flask
# A simple chatbot that uses Model Context Protocol for tool integration

from flask import Flask, request, jsonify, render_template_string
import time
import re
from datetime import datetime
from typing import Dict
import threading
import requests

# =============================================================================
# MCP SERVER FOR CHATBOT TOOLS
# =============================================================================


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


# =============================================================================
# CHATBOT WITH MCP CLIENT
# =============================================================================


class MCPChatbot:
    def __init__(self, mcp_server_url: str = "http://localhost:5001"):
        self.mcp_server_url = mcp_server_url
        self.request_id = 0
        self.available_tools = []
        self.conversation_history = []
        self.initialize_mcp()

    def initialize_mcp(self):
        """Initialize MCP connection and get available tools"""
        try:
            # Get available tools
            tools_response = self._make_mcp_request("tools/list")
            self.available_tools = tools_response.get("result", {}).get("tools", [])
            print(
                f"Connected to MCP server with {len(self.available_tools)} tools available"
            )
        except Exception as e:
            print(f"Failed to connect to MCP server: {e}")
            self.available_tools = []

    def _make_mcp_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a request to the MCP server"""
        self.request_id += 1

        payload = {"jsonrpc": "2.0", "id": self.request_id}

        if params:
            payload["params"] = params

        response = requests.post(
            f"{self.mcp_server_url}/mcp/{endpoint}",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        return response.json()

    def _call_tool(self, tool_name: str, arguments: Dict) -> str:
        """Call an MCP tool"""
        try:
            params = {"name": tool_name, "arguments": arguments}
            response = self._make_mcp_request("tools/call", params)

            if "error" in response:
                return f"Error calling {tool_name}: {response['error']['message']}"

            content = response.get("result", {}).get("content", [])
            if content and content[0].get("type") == "text":
                return content[0]["text"]
            return "No response from tool"
        except Exception as e:
            return f"Failed to call tool {tool_name}: {str(e)}"

    def _detect_intent(self, message: str) -> tuple:
        """Simple intent detection to determine which tool to use"""
        message_lower = message.lower()

        # Weather intent
        weather_keywords = ["weather", "temperature", "rain", "sunny", "cloudy"]
        if any(keyword in message_lower for keyword in weather_keywords):
            # Extract city name (simple regex)
            city_match = re.search(r"(?:in|for|at)\s+([a-zA-Z\s]+)", message_lower)
            if city_match:
                city = city_match.group(1).strip()
                return ("get_weather", {"city": city})

        # Math intent
        math_patterns = [r"\d+[\+\-\*/]\d+", r"calculate", r"math", r"compute"]
        if any(re.search(pattern, message_lower) for pattern in math_patterns):
            # Extract mathematical expression
            math_match = re.search(r"[\d\+\-\*/\(\)\.\s]+", message)
            if math_match:
                expression = math_match.group().strip()
                return ("calculate", {"expression": expression})

        # Time intent
        if any(word in message_lower for word in ["time", "clock", "hour", "minute"]):
            return ("get_time", {})

        # Remember intent
        if any(
            phrase in message_lower
            for phrase in ["remember", "don't forget", "keep in mind"]
        ):
            # Extract what to remember
            remember_match = re.search(
                r"(?:remember|don\'t forget|keep in mind)(?:\s+that)?\s+(.+)",
                message_lower,
            )
            if remember_match:
                info = remember_match.group(1).strip()
                return ("remember", {"information": info})

        # Recall intent
        if any(
            phrase in message_lower
            for phrase in ["what do you remember", "recall", "what did i tell you"]
        ):
            return ("recall", {})

        return (None, {})

    def process_message(self, message: str) -> str:
        """Process a user message and generate a response"""
        self.conversation_history.append({"role": "user", "content": message})

        # Detect if we need to use a tool
        tool_name, tool_args = self._detect_intent(message)

        if tool_name and any(
            tool["name"] == tool_name for tool in self.available_tools
        ):
            # Use the detected tool
            tool_response = self._call_tool(tool_name, tool_args)
            response = f"Let me help you with that! {tool_response}"
        else:
            # Generate a conversational response
            response = self._generate_conversational_response(message)

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def _generate_conversational_response(self, message: str) -> str:
        """Generate a simple conversational response"""
        message_lower = message.lower()

        greetings = [
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
        ]
        if any(greeting in message_lower for greeting in greetings):
            return "Hello! I'm your MCP-powered assistant. I can help you with weather, calculations, time, and remembering information. How can I assist you today?"

        if "help" in message_lower:
            tools_list = ", ".join([tool["name"] for tool in self.available_tools])
            return f"I can help you with various tasks using these tools: {tools_list}. Just ask me about weather, math calculations, current time, or tell me to remember something!"

        if any(word in message_lower for word in ["bye", "goodbye", "see you"]):
            return "Goodbye! It was nice chatting with you. Feel free to come back anytime!"

        if "thank" in message_lower:
            return "You're welcome! I'm happy to help. Is there anything else you'd like to know?"

        # Default response
        responses = [
            "I'm not sure I understand. Can you try asking about weather, math, time, or tell me something to remember?",
            "That's interesting! Is there something specific I can help you with?",
            "I'm here to help! You can ask me about weather in different cities, mathematical calculations, current time, or ask me to remember information.",
            "Could you be more specific? I can assist with weather forecasts, calculations, time queries, and memory tasks.",
        ]

        import random

        return random.choice(responses)


# =============================================================================
# WEB INTERFACE
# =============================================================================


class ChatbotWebApp:
    def __init__(self, chatbot: MCPChatbot):
        self.app = Flask(__name__)
        self.chatbot = chatbot
        self.setup_routes()

    def setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template_string(self.get_html_template())

        @self.app.route("/chat", methods=["POST"])
        def chat():
            data = request.get_json()
            message = data.get("message", "")

            if not message.strip():
                return jsonify({"error": "Empty message"})

            response = self.chatbot.process_message(message)

            return jsonify(
                {
                    "response": response,
                    "available_tools": [
                        tool["name"] for tool in self.chatbot.available_tools
                    ],
                }
            )

        @self.app.route("/history")
        def history():
            return jsonify({"history": self.chatbot.conversation_history})

    def get_html_template(self):
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .chat-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .chat-header {
            text-align: center;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .chat-messages {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 15px;
            margin-bottom: 20px;
            background-color: #fafafa;
            border-radius: 5px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
            max-width: 80%;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background-color: #e9ecef;
            color: #333;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        .message-input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        .send-button {
            padding: 12px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .send-button:hover {
            background-color: #0056b3;
        }
        .tools-info {
            margin-top: 15px;
            padding: 10px;
            background-color: #d4edda;
            border-radius: 5px;
            font-size: 14px;
        }
        .loading {
            display: none;
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 MCP Chatbot</h1>
            <p>Powered by Model Context Protocol</p>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot-message">
                Hello! I'm your MCP-powered assistant. I can help you with weather information, calculations, current time, and remembering things. Try asking me something!
            </div>
        </div>
        
        <div class="loading" id="loading">Bot is thinking...</div>
        
        <div class="input-container">
            <input type="text" id="messageInput" class="message-input" 
                   placeholder="Type your message here..." 
                   onkeypress="handleKeyPress(event)">
            <button class="send-button" onclick="sendMessage()">Send</button>
        </div>
        
        <div class="tools-info">
            <strong>Available tools:</strong> <span id="toolsList">Loading...</span>
        </div>
    </div>

    <script>
        let availableTools = [];
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        function addMessage(content, isUser = false) {
            const messagesContainer = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            messageDiv.textContent = content;
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            addMessage(message, true);
            input.value = '';
            showLoading(true);
            
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                showLoading(false);
                if (data.error) {
                    addMessage('Error: ' + data.error);
                } else {
                    addMessage(data.response);
                    if (data.available_tools) {
                        availableTools = data.available_tools;
                        document.getElementById('toolsList').textContent = availableTools.join(', ');
                    }
                }
            })
            .catch(error => {
                showLoading(false);
                addMessage('Error: Failed to send message');
                console.error('Error:', error);
            });
        }
        
        // Load available tools on page load
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: 'help' })
        })
        .then(response => response.json())
        .then(data => {
            if (data.available_tools) {
                availableTools = data.available_tools;
                document.getElementById('toolsList').textContent = availableTools.join(', ');
            }
        });
    </script>
</body>
</html>
        """

    def run(self, host="localhost", port=3000, debug=False):
        print(f"Starting Chatbot Web Interface on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    # Start MCP server in a separate thread
    mcp_server = MCPChatbotServer()
    server_thread = threading.Thread(target=mcp_server.run, kwargs={"debug": False})
    server_thread.daemon = True
    server_thread.start()

    # Wait for server to start
    time.sleep(2)

    # Create chatbot with MCP client
    chatbot = MCPChatbot()

    # Create and run web interface
    web_app = ChatbotWebApp(chatbot)

    print("=" * 50)
    print("MCP Chatbot is starting!")
    print("- MCP Server running on http://localhost:5001")
    print("- Chatbot Web Interface on http://localhost:3000")
    print("=" * 50)
    print("Try asking:")
    print("- 'What's the weather in London?'")
    print("- 'Calculate 15 * 8 + 3'")
    print("- 'What time is it?'")
    print("- 'Remember that I like pizza'")
    print("- 'What do you remember?'")
    print("=" * 50)

    web_app.run(debug=False)


if __name__ == "__main__":
    main()
