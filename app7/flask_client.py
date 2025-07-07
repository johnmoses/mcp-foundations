import asyncio
import json
import re
from flask import Flask, request, render_template_string
from os.path import expanduser

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

from langchain_community.llms import LlamaCpp
from sentence_transformers import SentenceTransformer

from milvus import MilvusClient  # Milvus Lite client

app = Flask(__name__)

# Fixed prompt string (no dynamic templating)
TOOL_SELECTION_PROMPT = """
You are a healthcare AI assistant that can call these tools:
- chat(message): general healthcare questions or fallback
- medical_calc(type, weight, height): perform medical calculations like BMI
- symptom_checker(symptoms): analyze symptoms
- get_fda_drug_info(drug_name): retrieve drug information from FDA
- search_pubmed(query, max_results): search medical literature on PubMed
- lookup_icd10_code(code=None, term=None): look up ICD-10 codes
- clinical_trials_search(condition, location=None, status="Recruiting"): search clinical trials

Given a user input, decide which tool to call and with what arguments.
Respond ONLY with JSON in this exact format:

{{"tool": "tool_name", "args": {{"param1": "value1", "param2": "value2"}}}}

Example:

Input: "Calculate BMI for weight 70kg and height 175cm."
Output: {{"tool": "medical_calc", "args": {{"type": "bmi", "weight": 70, "height": 175}}}}

Input: "{input}"
Output:
"""

RAG_PROMPT = """
You are a healthcare AI assistant. Use the following documents to answer the question.

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

# Initialize LLM, embedder, Milvus client, MCP session globally
model_path = expanduser("~/Models/llama-2-7b-chat.Q4_K_M.gguf")
llm = LlamaCpp(model_path=model_path, n_ctx=2048, temperature=0, streaming=False)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
milvus_client = MilvusClient(uri="./milvus_demo.db")
collection_name = "my_rag_collection"
server_params = StdioServerParameters(command="python", args=["healthcare_mcp_server.py"])

mcp_session = None

async def init_mcp_session():
    global mcp_session
    if mcp_session is None:
        reader, writer = await stdio_client(server_params)
        mcp_session = await ClientSession(reader, writer).__aenter__()
        await mcp_session.initialize()

async def process_user_query(user_input):
    await init_mcp_session()

    # Build the fixed prompt by inserting user input
    tool_prompt = TOOL_SELECTION_PROMPT.replace("{input}", user_input)
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
            search_results = milvus_client.search(
                collection_name=collection_name,
                data=[query_embedding],
                limit=5,
                search_params={"metric_type": "IP", "params": {}},
                output_fields=["text"],
            )
            docs = [res["entity"]["text"] for res in search_results[0]] if search_results else []
            context = "\n---\n".join(docs) if docs else "No relevant documents found."
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

# Flask routes

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    user_query = ""
    if request.method == "POST":
        user_query = request.form.get("user_input", "")
        if user_query.strip():
            response = asyncio.run(process_user_query(user_query.strip()))
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Healthcare MCP Assistant</title>
    </head>
    <body>
        <h1>Healthcare MCP Assistant</h1>
        <form method="post">
            <textarea name="user_input" rows="5" cols="60" placeholder="Enter your healthcare question here...">{{ user_query }}</textarea><br>
            <button type="submit">Ask</button>
        </form>
        {% if response %}
        <h2>Response:</h2>
        <pre>{{ response }}</pre>
        {% endif %}
    </body>
    </html>
    """, response=response, user_query=user_query)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5001)
