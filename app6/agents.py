# agents.py
from mcp_client import MCPHealthcareClient, run_async

class DiagnosisAgent:
    def __init__(self, mcp_client: MCPHealthcareClient):
        self.mcp_client = mcp_client

    async def run_async(self, symptoms: str) -> str:
        return await self.mcp_client.call_tool("diagnose_symptoms", symptoms)

    def run(self, symptoms: str) -> str:
        # Synchronous wrapper calling async method
        return run_async(self.run_async(symptoms))


class PrescriptionAgent:
    def __init__(self, mcp_client: MCPHealthcareClient):
        self.mcp_client = mcp_client

    async def run_async(self, diagnosis: str) -> str:
        return await self.mcp_client.call_tool("recommend_treatment", diagnosis)

    def run(self, diagnosis: str) -> str:
        return run_async(self.run_async(diagnosis))


class EducationAgent:
    def __init__(self, llm):
        self.llm = llm

    def run(self, term: str) -> str:
        prompt = f"Explain the medical term or diagnosis: {term}"
        response = self.llm(prompt=prompt, max_tokens=256)
        return response['choices'][0]['text'].strip()
