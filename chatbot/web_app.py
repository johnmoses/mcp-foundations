""" 
Web 
"""

from flask import Flask, request, jsonify, render_template_string
from chatbot.chatbot import MCPChatbot


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
