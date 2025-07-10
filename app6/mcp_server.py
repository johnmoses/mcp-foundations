import sys
import logging
from fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("HealthcareAssistant")

@mcp.tool()
def diagnose_symptoms(symptoms: str) -> str:
    # Simulated diagnosis logic
    mapping = {
        "fever": "flu, COVID-19, malaria",
        "cough": "bronchitis, asthma, COVID-19",
        "headache": "migraine, tension headache, flu",
    }
    found = []
    for symptom in mapping:
        if symptom in symptoms.lower():
            found.append(mapping[symptom])
    if found:
        return f"Possible diagnoses based on symptoms: {', '.join(set(', '.join(found).split(', ')))}"
    return "No diagnosis found for given symptoms."

@mcp.tool()
def recommend_treatment(diagnosis: str) -> str:
    treatments = {
        "flu": "rest, fluids, paracetamol",
        "covid-19": "isolation, hydration, medical supervision",
        "asthma": "inhalers, avoid triggers",
    }
    for key in treatments:
        if key in diagnosis.lower():
            return f"Recommended treatment for {key}: {treatments[key]}"
    return "No treatment recommendations available."

@mcp.tool()
def suggest_specialist(diagnosis: str) -> str:
    specialists = {
        "flu": "General Physician",
        "covid-19": "Infectious Disease Specialist",
        "asthma": "Pulmonologist",
    }
    for key in specialists:
        if key in diagnosis.lower():
            return f"Suggested specialist: {specialists[key]}"
    return "No specialist suggestion available."

if __name__ == "__main__":
    logger.info("Starting MCP server...")
    mcp.run()
