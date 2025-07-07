from flask import Flask, request, render_template_string
import asyncio
import json
import re
from os.path import expanduser

from langchain_community.llms import LlamaCpp
from sentence_transformers import SentenceTransformer

from milvus_client import MilvusClient  # Your Milvus wrapper
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

app = Flask(__name__)

# Your existing prompts
TOOL_SELECTION_PROMPT = """
You are an assistant that can call these tools:
- chat(message)
- weather(city)
- calculate(expression)

Given the user input, decide which tool to call and with what arguments.
Respond with a JSON object ONLY, with this exact format (no extra text):

{{"tool": "tool_name", "args": {{"param": "value"}}}}

User input:
{input}
"""

RAG_PROMPT = """
You are a financial AI assistant. Use the following financial documents to answer the question.

<context>
{context}
</context>

Question:
{question}

Answer:
"""

def extract_json_from_text(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return None

def extract_result_content(call_tool_result):
    content_blocks = getattr(call_tool_result, "content", [])
    if not content_blocks:
        return str(call_tool_result)
    texts = []
    for block in content_blocks:
        text = getattr(block, "text", None)
        if text:
            texts.append(text)
        else:
            texts.append(str(block))
    return "\n".join(texts)

# Initialize LLM, embedder, Milvus client, MCP session (async context)
model_path = expanduser("~/Models/llama-2-7b-chat.Q4_K_M.gguf")
llm = LlamaCpp(model_path=model_path, n_ctx=2048, temperature=0, streaming=False)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
milvus_client = MilvusClient("finance_rag_db.db")
server_params = StdioServerParameters(command="python", args=["finance_server.py"])

# We'll keep MCP session global and initialize on first request
mcp_session = None

async def init_mcp_session():
    global mcp_session
    if mcp_session is None:
        reader, writer = await stdio_client(server_params)
        mcp_session = await ClientSession(reader, writer).__aenter__()
        await mcp_session.initialize()

async def process_query(user_input):
    await init_mcp_session()

    tool_prompt = TOOL_SELECTION_PROMPT.format(input=user_input)
    tool_response = llm(tool_prompt)
    tool_call = extract_json_from_text(tool_response)

    if tool_call is None:
        tool_name = "chat"
        args = {"message": user_input}
    else:
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})

    if tool_name == "chat":
        query_embedding = embedder.encode(user_input).tolist()
        try:
            retrieved_docs = milvus_client.search(query_embedding, limit=5)
            context = "\n---\n".join(retrieved_docs) if retrieved_docs else "No financial documents found."
        except Exception:
            context = ""

        rag_prompt = RAG_PROMPT.format(context=context, question=user_input)
        answer = llm(rag_prompt)
        return answer
    else:
        try:
            response = await mcp_session.call_tool(tool_name, args)
            return extract_result_content(response)
        except Exception as e:
            return f"Error calling tool '{tool_name}': {e}"

# Flask route for GET and POST
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        if user_input.strip() == "":
            return render_template_string(TEMPLATE, response="Please enter a query.", query=user_input)

        # Run the async MCP processing in event loop
        response = asyncio.run(process_query(user_input))
        return render_template_string(TEMPLATE, response=response, query=user_input)

    return render_template_string(TEMPLATE, response=None, query="")

# Simple HTML template embedded here; you can move to separate file if preferred
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MCP Finance Assistant</title>
</head>
<body>
    <h1>MCP Finance Assistant</h1>
    <form method="post">
        <textarea name="user_input" rows="4" cols="60" placeholder="Enter your financial query here...">{{ query }}</textarea><br>
        <button type="submit">Ask</button>
    </form>
    {% if response %}
    <h2>Response:</h2>
    <pre>{{ response }}</pre>
    {% endif %}
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5001)
