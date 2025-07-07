import asyncio
import json
import re
from os.path import expanduser

from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession

from langchain_community.llms import LlamaCpp

# Strict prompt asking for JSON-only output
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

def extract_json_from_text(text):
    """
    Extract the first JSON object found in the text using regex.
    Returns parsed JSON dict or None if not found/parseable.
    """
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except json.JSONDecodeError:
        pass
    return None

def extract_result_content(call_tool_result):
    """
    Extract human-readable text from MCP CallToolResult object.
    """
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
    model_path = expanduser("/Users/johnmoses/.cache/lm-studio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf")  # Adjust path as needed

    # Initialize LlamaCpp with deterministic output
    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        temperature=0,  # deterministic output
        streaming=False,
    )

    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],  # Your MCP server script
    )

    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            print("Connected to MCP server. Type 'quit' to exit.")

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == "quit":
                    print("Goodbye!")
                    break
                if not user_input:
                    continue

                prompt = TOOL_SELECTION_PROMPT.format(input=user_input)
                llm_response = llm(prompt)

                tool_call = extract_json_from_text(llm_response)
                if tool_call is None:
                    print("Failed to parse LLM response. Defaulting to chat tool.")
                    tool_name = "chat"
                    args = {"message": user_input}
                else:
                    tool_name = tool_call.get("tool")
                    args = tool_call.get("args", {})

                try:
                    response = await session.call_tool(tool_name, args)
                    print(f"Agent ({tool_name}): {extract_result_content(response)}")
                except Exception as e:
                    print(f"Error calling tool '{tool_name}': {e}")

if __name__ == "__main__":
    asyncio.run(main())
