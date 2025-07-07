from fastmcp import FastMCP
import httpx  # For making HTTP requests to external APIs

mcp = FastMCP(name="Robust Healthcare MCP Server")

# --- Configuration for external APIs ---
# In a real application, these would come from environment variables or a secure config system
# For demonstration, use placeholders.
# You would need to sign up for API keys where necessary (e.g., NCBI E-utilities for PubMed)
NCBI_API_KEY = "YOUR_NCBI_API_KEY"  # For PubMed
# FDA API key might not be strictly needed for public data, but check documentation
# Clinical Trials API details would go here if using a specific external service


# --- General Helper for External API Calls ---
async def fetch_data_from_api(url: str, params: dict = None) -> dict:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": f"API request failed with status {e.response.status_code}: {e.response.text}"
            }
        except httpx.RequestError as e:
            return {"error": f"An error occurred while requesting {e.request.url}: {e}"}
        except Exception as e:
            return {"error": f"An unexpected error occurred: {e}"}


# --- Medical Calculation Tool (Retained) ---
@mcp.tool()
def medical_calc(type: str, weight: float = None, height: float = None) -> str:
    """
    Calculate medical metrics, e.g., BMI.
    """
    if type.lower() == "bmi":
        if weight is None or height is None:
            return "Weight (kg) and height (cm) must be provided for BMI calculation."
        try:
            height_m = height / 100
            bmi = weight / (height_m**2)
            return (
                f"The BMI for weight {weight} kg and height {height} cm is {bmi:.2f}."
            )
        except Exception as e:
            return f"Error calculating BMI: {e}"
    else:
        return f"Medical calculation type '{type}' not supported or implemented."


# --- Symptom Checker Tool (Retained, expanded placeholder) ---
@mcp.tool()
def symptom_checker(symptoms: str) -> str:
    """
    Analyzes symptoms and provides basic health guidance or suggests consulting a professional.
    Note: This is a simplified placeholder. Real symptom checkers are complex.
    """
    symptoms_lower = symptoms.lower()
    if (
        "fever" in symptoms_lower
        and "cough" in symptoms_lower
        and "fatigue" in symptoms_lower
    ):
        return "You may have a common viral infection like the flu or cold. Rest, stay hydrated, and consider consulting a doctor if symptoms worsen or persist."
    elif (
        "severe chest pain" in symptoms_lower
        or "sudden difficulty breathing" in symptoms_lower
    ):
        return "These are serious symptoms. Please seek immediate medical attention or call emergency services."
    elif "headache" in symptoms_lower and "blurred vision" in symptoms_lower:
        return "This combination of symptoms warrants medical evaluation. Please consult a healthcare professional."
    else:
        return "Based on the symptoms provided, general advice is to monitor and consult a doctor if concerned. This tool is not a substitute for professional medical advice."


# --- New: FDA Drug Information Tool ---
@mcp.tool()
async def get_fda_drug_info(drug_name: str) -> str:
    """
    Searches and retrieves comprehensive drug information from a simulated FDA database endpoint.
    In a real scenario, this would query FDA's Open API or a similar authoritative source.
    """
    # Example FDA Open API endpoint (replace with actual if you implement it)
    # FDA_API_BASE = "https://api.fda.gov/drug/label.json"
    # params = {"search": f"openfda.brand_name:\"{drug_name}\"", "limit": 1}
    # data = await fetch_data_from_api(FDA_API_BASE, params)

    # Simulated response for demonstration
    simulated_data = {
        "Lipitor": {
            "generic_name": "Atorvastatin",
            "class": "Statin (HMG-CoA reductase inhibitor)",
            "uses": "Lowers cholesterol (LDL) and triglycerides; reduces risk of heart attack and stroke.",
            "side_effects": "Muscle pain, headache, nausea, abnormal liver function tests.",
            "warnings": "Do not use during pregnancy. Avoid grapefruit juice.",
        },
        "Amoxicillin": {
            "generic_name": "Amoxicillin",
            "class": "Penicillin antibiotic",
            "uses": "Treats bacterial infections (e.g., ear infections, strep throat, pneumonia).",
            "side_effects": "Diarrhea, nausea, rash. Allergic reactions (rare but severe).",
            "warnings": "Inform doctor of penicillin allergy. May cause C. difficile-associated diarrhea.",
        },
    }

    info = simulated_data.get(
        drug_name.capitalize()
    )  # Simple case-insensitive match for demo
    if info:
        return (
            f"**Drug Name:** {drug_name.capitalize()}\n"
            f"**Generic Name:** {info['generic_name']}\n"
            f"**Class:** {info['class']}\n"
            f"**Uses:** {info['uses']}\n"
            f"**Side Effects:** {info['side_effects']}\n"
            f"**Warnings:** {info['warnings']}"
        )
    else:
        return f"Could not find information for drug: {drug_name}. Please try a different name or provide more details."


# --- New: PubMed Research Search Tool ---
@mcp.tool()
async def search_pubmed(query: str, max_results: int = 3) -> str:
    """
    Searches medical literature from PubMed's database of scientific articles.
    Uses NCBI E-utilities API.
    """
    # Example NCBI E-utilities ESearch endpoint
    PUBMED_ESEARCH_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    PUBMED_EFETCH_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "api_key": NCBI_API_KEY,  # If you have one
    }

    esearch_data = await fetch_data_from_api(PUBMED_ESEARCH_BASE, esearch_params)

    if esearch_data and "esearchresult" in esearch_data:
        ids = esearch_data["esearchresult"]["idlist"]
        if not ids:
            return f"No PubMed articles found for query: '{query}'."

        efetch_params = {
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "xml",  # Request XML to parse titles/abstracts if needed
            "api_key": NCBI_API_KEY,
        }
        # For simplicity, we just return IDs; a full implementation would parse XML for titles/abstracts
        return f"Found PubMed articles with IDs: {', '.join(ids)}. You can search these IDs on PubMed for full details."
    else:
        return f"Error searching PubMed for '{query}': {esearch_data.get('error', 'Unknown error.')}"


# --- New: ICD-10 Code Lookup Tool ---
@mcp.tool()
async def lookup_icd10_code(code: str = None, term: str = None) -> str:
    """
    Looks up ICD-10 codes or definitions.
    This would typically query a specialized ICD-10 API or local database.
    """
    # Simulated ICD-10 lookup
    icd10_data = {
        "I10": "Essential (primary) hypertension",
        "J45.909": "Unspecified asthma, uncomplicated",
        "E11.9": "Type 2 diabetes mellitus without complications",
        "F43.10": "Post-traumatic stress disorder, unspecified",
    }

    if code:
        definition = icd10_data.get(code.upper())
        if definition:
            return f"ICD-10 Code: {code.upper()} - Definition: {definition}"
        else:
            return f"No definition found for ICD-10 code: {code.upper()}."
    elif term:
        results = [
            f"Code: {c}, Definition: {d}"
            for c, d in icd10_data.items()
            if term.lower() in d.lower()
        ]
        if results:
            return "Found matching ICD-10 codes:\n" + "\n".join(results)
        else:
            return f"No ICD-10 codes found for term: '{term}'."
    else:
        return "Please provide an ICD-10 code or a search term."


# --- New: Clinical Trials Search Tool ---
@mcp.tool()
async def clinical_trials_search(
    condition: str, location: str = None, status: str = "Recruiting"
) -> str:
    """
    Searches for clinical trials. Typically queries ClinicalTrials.gov or similar.
    """
    # Simulated ClinicalTrials.gov search
    # In a real scenario, you'd use ClinicalTrials.gov API (if available) or a web scraper.
    # Base URL: https://clinicaltrials.gov/api/query/full_studies?expr={query_term}

    simulated_trials = {
        "diabetes": [
            "Trial A (NCT01234567): New Insulin Therapy for Type 2 Diabetes (Status: Recruiting, Location: USA)",
            "Trial B (NCT07654321): Lifestyle Intervention for Prediabetes (Status: Recruiting, Location: Europe)",
        ],
        "cancer": [
            "Trial C (NCT09876543): Immunotherapy for Lung Cancer (Status: Recruiting, Location: USA, Canada)",
            "Trial D (NCT01239876): Targeted Therapy for Breast Cancer (Status: Active, not recruiting, Location: Europe)",
        ],
    }

    found_trials = simulated_trials.get(condition.lower(), [])

    filtered_trials = []
    for trial_info in found_trials:
        if status.lower() in trial_info.lower():
            if location and location.lower() not in trial_info.lower():
                continue
            filtered_trials.append(trial_info)

    if filtered_trials:
        return (
            f"Found clinical trials for '{condition}' (Status: {status}):\n"
            + "\n".join(filtered_trials)
        )
    else:
        return (
            f"No '{status}' clinical trials found for '{condition}' matching criteria."
        )


# --- Chat Fallback Tool (Retained) ---
@mcp.tool()
def chat(message: str) -> str:
    """
    A general chat tool for when no specific tool is matched.
    """
    return f"I received your message: '{message}'. How else can I assist you with healthcare information?"


if __name__ == "__main__":
    mcp.run()
