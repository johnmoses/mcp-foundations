import asyncio
import json
import re
from os.path import expanduser

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

from langchain_community.llms import LlamaCpp
from sentence_transformers import SentenceTransformer

from pymilvus import (
    MilvusClient,
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)

MILVUS_DB_URI = "milvus_rag_db.db"
COLLECTION_NAME = "rag_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

milvus_client = MilvusClient(MILVUS_DB_URI)
connections.connect(alias="default", uri=MILVUS_DB_URI)


# --- Step 2: Create collection with primary key if not exists ---
def create_collection():
    if COLLECTION_NAME in milvus_client.list_collections():
        return Collection(COLLECTION_NAME, using="default")

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    ]
    schema = CollectionSchema(fields, description="RAG collection")
    return Collection(name=COLLECTION_NAME, schema=schema, using="default")


collection = create_collection()

# --- Step 3: Prepare in-memory documents ---
documents = [
    "Milvus is an open-source vector database built for scalable similarity search.",
    "It supports embedding-based search for images, video, and text.",
    "You can use SentenceTransformers to generate embeddings for your documents.",
    "GPT-2 is an open-source language model suitable for text generation tasks.",
]

# --- Step 4: Embed documents ---
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)


# --- Step 5: Insert data programmatically ---
def insert_data(collection, embeddings, texts):
    entities = [
        embeddings.tolist(),  # embeddings
        texts,  # texts
    ]
    collection.insert(entities)
    collection.flush()


if collection.num_entities == 0:
    insert_data(collection, doc_embeddings, documents)
else:
    print(
        f"Collection already has {collection.num_entities} entities, skipping insert."
    )

# --- Step 5.1: Create index and load collection ---
index_params = {
    "metric_type": "IP",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}

try:
    print("Creating index on embedding field...")
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index created.")
except Exception as e:
    print(f"Index creation skipped or failed: {e}")

print("Loading collection into memory...")
collection.load()
print("Collection loaded.")


# Tool selection prompt for LLM
TOOL_SELECTION_PROMPT = """
You are a healthcare AI assistant that can call these tools:
- chat(message): general healthcare questions or fallback
- medical_calc(type, weight, height): perform medical calculations like BMI (e.g., 'bmi', 70, 175)
- symptom_checker(symptoms): analyze symptoms (e.g., 'fever, cough')
- get_fda_drug_info(drug_name): retrieve drug information from FDA (e.g., 'Lipitor')
- search_pubmed(query, max_results): search medical literature on PubMed (e.g., 'COVID-19 treatment', 5)
- lookup_icd10_code(code=None, term=None): look up ICD-10 codes by code or term (e.g., 'I10' or 'hypertension')
- clinical_trials_search(condition, location=None, status="Recruiting"): search for clinical trials (e.g., 'diabetes', 'USA', 'Recruiting')

Given the user input, decide which tool to call and with what arguments.
Respond ONLY with JSON in this format:

{{"tool": "tool_name", "args": {{"param1": "value1", "param2": "value2"}}}}

Examples:

Input: "Calculate BMI for weight 70kg and height 175cm."
Output: {{"tool": "medical_calc", "args": {{"type": "bmi", "weight": 70, "height": 175}}}}

Input: "I have headache and fever, what should I do?"
Output: {{"tool": "symptom_checker", "args": {{"symptoms": "headache, fever"}}}}

Input: "What are the uses of Amoxicillin?"
Output: {{"tool": "get_fda_drug_info", "args": {{"drug_name": "Amoxicillin"}}}}

Input: "Find articles on gene therapy for cancer on PubMed."
Output: {{"tool": "search_pubmed", "args": {{"query": "gene therapy cancer", "max_results": 3}}}}

Input: "What is ICD-10 code I10?"
Output: {{"tool": "lookup_icd10_code", "args": {{"code": "I10"}}}}

Input: "Find recruiting clinical trials for breast cancer in New York."
Output: {{"tool": "clinical_trials_search", "args": {{"condition": "breast cancer", "location": "New York", "status": "Recruiting"}}}}

Input: "{input}"
Output:
"""

# Prompt template for RAG generation
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


async def main():
    model_path = expanduser(
        "/Users/johnmoses/.cache/lm-studio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
    )
    llm = LlamaCpp(model_path=model_path, n_ctx=2048, temperature=0, streaming=False)

    server_params = StdioServerParameters(command="python", args=["server.py"])

    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            print("Connected to Healthcare MCP server. Type 'quit' to exit.")

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break
                if not user_input:
                    continue

                # Step 1: Tool selection via LLM
                tool_prompt = TOOL_SELECTION_PROMPT.format(input=user_input)
                tool_response = llm(tool_prompt)
                tool_call = extract_json_from_text(tool_response)

                if tool_call is None:
                    # Default to chat tool if parsing fails
                    tool_name = "chat"
                    args = {"message": user_input}
                else:
                    tool_name = tool_call.get("tool")
                    args = tool_call.get("args", {})

                # Step 2: If chat tool, perform RAG locally using Milvus Lite
                if tool_name == "chat":
                    query_embedding = embedder.encode(user_input).tolist()
                    try:
                        search_results = milvus_client.search(
                            collection_name=COLLECTION_NAME,
                            data=[query_embedding],
                            limit=5,
                            search_params={"metric_type": "IP", "params": {}},
                            output_fields=["text"],
                        )
                        docs = (
                            [res["entity"]["text"] for res in search_results[0]]
                            if search_results
                            else []
                        )
                        context = (
                            "\n---\n".join(docs)
                            if docs
                            else "No relevant documents found."
                        )
                    except Exception as e:
                        print(f"Error during retrieval: {e}")
                        context = ""

                    rag_prompt = RAG_PROMPT.format(context=context, question=user_input)
                    answer = llm(rag_prompt)
                    print(f"Agent (RAG): {answer}")

                else:
                    # Step 3: Call the selected MCP tool on server
                    try:
                        response = await session.call_tool(tool_name, args)
                        output_text = extract_result_content(response)
                        print(f"Agent ({tool_name}): {output_text}")
                    except Exception as e:
                        print(f"Error calling tool '{tool_name}': {e}")


if __name__ == "__main__":
    asyncio.run(main())
