""" 
Chatbot with MCP client
"""

import re
from typing import Dict
import requests


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
