from flask import Flask, Response, request, jsonify, render_template, send_from_directory, stream_with_context
from llama_cpp import Llama
from milvus_rag import MilvusRAG
from mcp_client import MCPHealthcareClient, run_async
from agents import DiagnosisAgent, PrescriptionAgent, EducationAgent
from rag_agent import RAGAgent
import os
import asyncio
import logging

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize RAG system
milvus_rag = MilvusRAG("milvus_rag_db.db")

# Create collection
milvus_rag.create_collection()

# Seed db
milvus_rag.seed_db()

# --- Load Llama 3B model ---
model_path = os.path.expanduser(
    "/Users/johnmoses/.cache/lm-studio/models/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
)  # Update path as needed
if not os.path.exists(model_path):
    logger.error(f"Model file not found at {model_path}")
    raise FileNotFoundError(f"Model file not found at {model_path}")

llm = Llama(model_path=model_path, n_ctx=2048, temperature=0.7)

# SYSTEM_CHAT_PROMPT = "You are a helpful healthcare assistant."

prompt = "Hello, how are you?"

for output in llm(prompt=prompt, max_tokens=50, stream=True):
    print(output['choices'][0]['text'].strip(), end="", flush=True)
print()

# Global MCP client instance
mcp_client = None

async def get_mcp_client():
    global mcp_client
    if mcp_client is None:
        mcp_client = MCPHealthcareClient()
        await mcp_client.__aenter__()
    return mcp_client

# diagnosis_agent = DiagnosisAgent(mcp_client)
# prescription_agent = PrescriptionAgent(mcp_client)
# education_agent = EducationAgent(llm)
# rag_agent = RAGAgent(llm, milvus_rag)


class MultiAgentOrchestrator:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client

    async def diagnose_stream(self, symptoms: str):
        async for token in self.mcp_client.stream_tool("diagnose_symptoms", symptoms):
            yield token

    async def prescribe_stream(self, diagnosis: str):
        async for token in self.mcp_client.stream_tool("recommend_treatment", diagnosis):
            yield token

    async def suggest_specialist_stream(self, diagnosis: str):
        async for token in self.mcp_client.stream_tool("suggest_specialist", diagnosis):
            yield token


orchestrator = None

async def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        client = await get_mcp_client()
        orchestrator = MultiAgentOrchestrator(client)
    return orchestrator


def generate_stream(prompt):
    for output in llm(prompt=prompt, max_tokens=512, stream=True):
        text_chunk = output['choices'][0].get('text', '')
        if text_chunk:
            yield text_chunk

@app.route('/stream', methods=['POST'])
def stream():
    data = request.get_json(force=True)
    prompt = data.get('prompt', '')
    if not prompt:
        return "Prompt is required", 400

    return Response(stream_with_context(generate_stream(prompt)), mimetype='text/plain')

def stream_agent_response(prompt):
    for output in llm(prompt=prompt, max_tokens=512, stream=True):
        text_chunk = output['choices'][0].get('text', '')
        if text_chunk:
            yield text_chunk

@app.route("/")
def index():
    return render_template("chat.html")

@app.route('/diagnose', methods=['POST'])
async def diagnose():
    data = await request.get_json()
    symptoms = data.get('symptoms')
    if not symptoms:
        return jsonify({"error": "Missing 'symptoms'"}), 400

    try:
        client = await get_mcp_client()
        response = await client.call_tool("diagnose_symptoms", symptoms)
        return jsonify({"diagnosis": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/shutdown', methods=['POST'])
async def shutdown():
    global mcp_client, orchestrator
    if orchestrator:
        orchestrator = None
    if mcp_client:
        await mcp_client.__aexit__(None, None, None)
        mcp_client = None
    return jsonify({"status": "MCP client shutdown successfully"})


def stream_response(async_gen):
    """
    Convert async generator to sync generator for Flask streaming.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agen = async_gen.__aiter__()

    def gen():
        try:
            while True:
                token = loop.run_until_complete(agen.__anext__())
                yield token
        except StopAsyncIteration:
            pass
        finally:
            loop.close()

    return gen()

@app.route('/multiagent', methods=['POST'])
def multiagent():
    data = request.get_json()
    symptoms = data.get('symptoms')
    if not symptoms:
        return jsonify({"error": "Missing 'symptoms' field"}), 400

    async def generate():
        agent = await get_orchestrator()
        # First stream diagnosis tokens
        diagnosis_tokens = []
        async for token in agent.diagnose_stream(symptoms):
            diagnosis_tokens.append(token)
            yield token

        diagnosis = ''.join(diagnosis_tokens)

        # Stream treatment tokens
        async for token in agent.prescribe_stream(diagnosis):
            yield token

        # Stream specialist tokens
        async for token in agent.suggest_specialist_stream(diagnosis):
            yield token

    return Response(stream_with_context(stream_response(generate())), mimetype='text/plain')


async def test_async():
    async with MCPHealthcareClient() as client:
        result = await client.call_tool("diagnose_symptoms", "I have fever and cough")
        print("Async result:", result)

def test_sync():
    async def inner():
        async with MCPHealthcareClient() as client:
            return await client.call_tool("diagnose_symptoms", "I have fever and cough")
    result = run_async(inner())
    print("Sync result:", result)

if __name__ == "__main__":
    # Run async test
    # asyncio.run(test_async())
    # Or run sync test
    # test_sync()

    app.run(debug=True, use_reloader=False, port=5001)
