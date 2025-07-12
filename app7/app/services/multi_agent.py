# app/services/multi_agent.py
class RagAgent:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client

    async def run(self, query, history):
        return await self.mcp_client.rag_search(query)

class CalculatorAgent:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client

    async def run(self, query, history):
        return await self.mcp_client.calculate(query)

class QuizAgent:
    def __init__(self, mcp_client):
        self.mcp_client = mcp_client

    async def run(self, query, history):
        return await self.mcp_client.generate_quiz(query)

class AgentOrchestrator:
    def __init__(self, agents):
        self.agents = agents

    def select_agent(self, query):
        q = query.lower()
        if any(word in q for word in ['calculate', 'sum', 'plus', 'minus']):
            return 'calculator'
        if 'quiz' in q:
            return 'quiz'
        return 'rag'

    async def handle_query(self, query, history):
        agent_key = self.select_agent(query)
        agent = self.agents.get(agent_key)
        if not agent:
            return "Sorry, I don't know how to handle that."
        return await agent.run(query, history)