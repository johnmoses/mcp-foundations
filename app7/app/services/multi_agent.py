from typing import Dict, List, Callable
import asyncio
from app.services.service import rag, call_llm, llm  # Your MilvusRAG instance


class TodoAgent:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client

    async def run(self, query: str, history: List[Dict]) -> str:
        # Pass query to MCP client list_todos (adjust as per your client API)
        return await self.mcp_client.list_todos(query)


class CalculatorAgent:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client

    async def run(self, query: str, history: List[Dict]) -> str:
        return await self.mcp_client.calculate(query)


class RAGAgent:
    def __init__(self, llm_callable: Callable):
        """
        llm_callable: function(prompt:str, max_tokens:int, temperature:float) -> dict
        """
        self.llm_callable = call_llm

    async def run(self, user_message: str, conversation_history: List[Dict]) -> str:
        """
        Generate AI response using RAG + LLM.

        Args:
            user_message: latest user input
            conversation_history: list of dicts with 'role' and 'content'

        Returns:
            AI response string
        """
        loop = asyncio.get_event_loop()

        def sync_call():
            return rag.chat(user_message, conversation_history, self.llm_callable)

        response = await loop.run_in_executor(None, sync_call)
        return response


class AgentOrchestrator:
    def __init__(self, agents: Dict[str, object]):
        self.agents = agents

    def select_agent(self, query: str) -> str:
        q = query.lower()
        if any(word in q for word in ['calculate', 'sum', 'plus', 'minus']):
            return 'calculator'
        if any(word in q for word in ['todo', 'task', 'list', 'show']):
            return 'todo'
        # Default to RAG agent for general queries
        return 'rag'

    async def handle_query(self, query: str, history: List[Dict]) -> str:
        agent_key = self.select_agent(query)
        agent = self.agents.get(agent_key)
        if not agent:
            return "Sorry, I don't know how to handle that."
        try:
            return await agent.run(query, history)
        except Exception as e:
            # Log error if you have logging configured
            return f"Error processing your request: {str(e)}"
