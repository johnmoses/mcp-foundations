import asyncio
import json
from llama_cpp import Llama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL_PATH = "/Users/johnmoses/.cache/lm-studio/models/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"


class LocalLlamaCompletionProvider:
    def __init__(self, model_path):
        self.llm = Llama(model_path=model_path)

    def build_prompt(self, conversation_history: str, user_message: str) -> str:
        system_prompt = (
            "You are a helpful assistant managing a to-do list. "
            "You can respond naturally or call tools by outputting JSON like:\n"
            '{"tool_call": {"name": "tool_name", "arguments": { ... }}}\n'
            "Only output JSON if you want to call a tool. Otherwise, answer naturally.\n\n"
        )
        few_shot_examples = """
        User: Add a task to buy groceries.
        Assistant: {"tool_call": {"name": "add_todo", "arguments": {"task": "buy groceries"}}}

        User: What tasks do I have?
        Assistant: {"tool_call": {"name": "list_todos", "arguments": {}}}

        User: Hello, how are you?
        Assistant: I'm doing great! How can I assist you with your to-do list today?
        """
        return f"{system_prompt}{few_shot_examples}{conversation_history}\nUser: {user_message}\nAssistant:"

    def generate(self, prompt: str, max_tokens=256, temperature=0.3):
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["\nUser:", "\nAssistant:"],
        )
        return response["choices"][0]["text"].strip()


def parse_tool_call(assistant_output: str):
    """Try to extract tool_call JSON from assistant output."""
    try:
        json_start = assistant_output.find("{")
        json_end = assistant_output.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            return None
        json_str = assistant_output[json_start:json_end]
        parsed = json.loads(json_str)
        if "tool_call" in parsed:
            return parsed["tool_call"]
    except Exception:
        return None
    return None


async def main():
    server_params = StdioServerParameters(command="python", args=["server.py"])
    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()

            completion_provider = LocalLlamaCompletionProvider(MODEL_PATH)
            conversation_history = ""

            print("Interactive MCP chat client with local LLM and tool calling")
            print("Type 'exit' to quit.\n")

            while True:
                user_input = input("You: ").strip()
                if user_input.lower() in ("exit", "quit"):
                    break

                prompt = completion_provider.build_prompt(
                    conversation_history, user_input
                )
                assistant_output = completion_provider.generate(prompt)

                tool_call = parse_tool_call(assistant_output)

                if tool_call:
                    tool_name = tool_call.get("name")
                    arguments = tool_call.get("arguments", {})
                    print(
                        f"LLM requested tool call: {tool_name} with arguments {arguments}"
                    )
                    try:
                        result = await session.call_tool(tool_name, arguments)
                        print(
                            f"Tool '{tool_name}' result:\n{json.dumps(result.result, indent=2)}"
                        )
                        conversation_history += f"\nUser: {user_input}\nAssistant: {assistant_output}\nTool {tool_name} output: {json.dumps(result.result)}"
                    except Exception as e:
                        print(f"Error calling tool '{tool_name}': {e}")
                        conversation_history += f"\nUser: {user_input}\nAssistant: Sorry, I failed to call the tool '{tool_name}'."
                else:
                    print(f"Assistant: {assistant_output}")
                    conversation_history += (
                        f"\nUser: {user_input}\nAssistant: {assistant_output}"
                    )


if __name__ == "__main__":
    asyncio.run(main())
