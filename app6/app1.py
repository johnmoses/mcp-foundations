# Multi-Agent RAG Healthcare App
# Requirements: pip install flask sentence-transformers pymilvus llama-cpp-python mcp

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

# Flask and web components
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# RAG components
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
import numpy as np

# Local LLM
from llama_cpp import Llama

# MCP (Model Context Protocol) components
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HealthcareDocument:
    """Healthcare document structure"""
    id: str
    title: str
    content: str
    category: str
    specialty: str
    timestamp: datetime
    metadata: Dict[str, Any]

class MilvusRAG:
    """Milvus RAG implementation for healthcare documents"""
    
    def __init__(self, db_path: str = "milvus_rag_db.db"):
        self.db_path = db_path
        self.client = MilvusClient(uri=db_path)
        self.collection_name = "healthcare_documents"
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384
        
        # Initialize collection
        self._create_collection()
        
    def _create_collection(self):
        """Create Milvus collection for healthcare documents"""
        try:
            # Drop existing collection if it exists
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
            
            # Create collection schema
            schema = [
                {"field_name": "id", "datatype": DataType.VARCHAR, "is_primary": True, "max_length": 100},
                {"field_name": "title", "datatype": DataType.VARCHAR, "max_length": 500},
                {"field_name": "content", "datatype": DataType.VARCHAR, "max_length": 10000},
                {"field_name": "category", "datatype": DataType.VARCHAR, "max_length": 100},
                {"field_name": "specialty", "datatype": DataType.VARCHAR, "max_length": 100},
                {"field_name": "timestamp", "datatype": DataType.VARCHAR, "max_length": 50},
                {"field_name": "embedding", "datatype": DataType.FLOAT_VECTOR, "dim": self.embedding_dim}
            ]
            
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type="IP",
                consistency_level="Strong"
            )
            
            logger.info(f"Created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def add_document(self, doc: HealthcareDocument) -> bool:
        """Add a healthcare document to the RAG system"""
        try:
            # Generate embedding
            embedding = self.embedding_model.encode(doc.content).tolist()
            
            # Prepare data
            data = [{
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
                "category": doc.category,
                "specialty": doc.specialty,
                "timestamp": doc.timestamp.isoformat(),
                "embedding": embedding
            }]
            
            # Insert into Milvus
            self.client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            logger.info(f"Added document: {doc.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return False
    
    def search(self, query: str, limit: int = 5, category_filter: Optional[str] = None) -> List[Dict]:
        """Search for relevant documents"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Prepare search parameters
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10}
            }
            
            # Add category filter if specified
            filter_expr = f"category == '{category_filter}'" if category_filter else None
            
            # Search
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding],
                limit=limit,
                search_params=search_params,
                filter=filter_expr,
                output_fields=["id", "title", "content", "category", "specialty", "timestamp"]
            )
            
            # Process results
            documents = []
            for result in results[0]:
                doc = {
                    "id": result["entity"]["id"],
                    "title": result["entity"]["title"],
                    "content": result["entity"]["content"],
                    "category": result["entity"]["category"],
                    "specialty": result["entity"]["specialty"],
                    "timestamp": result["entity"]["timestamp"],
                    "score": result["distance"]
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

class LocalLlamaModel:
    """Local Llama 3 model wrapper"""
    
    def __init__(self, model_path: str = "/Users/johnmoses/.cache/lm-studio/models/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"):
        self.model_path = model_path
        self.llm = None
        self._load_model()
    
    def _load_model(self):
        """Load the local Llama model"""
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_threads=8,
                n_gpu_layers=0,  # Adjust based on your GPU
                verbose=False
            )
            logger.info("Loaded Llama 3 model successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate text using the local model"""
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "Human:", "Assistant:"],
                echo=False
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return "Error generating response"

class HealthcareAgent:
    """Base healthcare agent class"""
    
    def __init__(self, name: str, specialty: str, rag_system: MilvusRAG, llm: LocalLlamaModel):
        self.name = name
        self.specialty = specialty
        self.rag_system = rag_system
        self.llm = llm
        self.conversation_history = []
    
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Process a healthcare query"""
        # Search for relevant documents
        relevant_docs = self.rag_system.search(
            query, 
            limit=3, 
            category_filter=self.specialty if self.specialty != "general" else None
        )
        
        # Build context from retrieved documents
        context_text = "\n\n".join([
            f"Document: {doc['title']}\n{doc['content'][:500]}..."
            for doc in relevant_docs
        ])
        
        # Create prompt
        prompt = self._build_prompt(query, context_text, context)
        
        # Generate response
        response = self.llm.generate(prompt, max_tokens=512)
        
        # Store conversation
        self.conversation_history.append({
            "query": query,
            "response": response,
            "relevant_docs": relevant_docs,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "agent": self.name,
            "specialty": self.specialty,
            "response": response,
            "relevant_documents": relevant_docs,
            "confidence": self._calculate_confidence(relevant_docs)
        }
    
    def _build_prompt(self, query: str, context: str, additional_context: Optional[Dict] = None) -> str:
        """Build prompt for the LLM"""
        base_prompt = f"""You are {self.name}, a healthcare AI assistant specializing in {self.specialty}.

Context from medical documents:
{context}

Patient Query: {query}

Please provide a helpful, accurate response based on the context provided. If you're unsure about something, clearly state that you recommend consulting with a healthcare professional.

Response:"""
        
        return base_prompt
    
    def _calculate_confidence(self, docs: List[Dict]) -> float:
        """Calculate confidence score based on retrieved documents"""
        if not docs:
            return 0.0
        
        # Simple confidence calculation based on similarity scores
        scores = [doc['score'] for doc in docs]
        return min(sum(scores) / len(scores), 1.0)

class MultiAgentCoordinator:
    """MCP-based coordinator for multiple healthcare agents"""
    
    def __init__(self, rag_system: MilvusRAG, llm: LocalLlamaModel):
        self.rag_system = rag_system
        self.llm = llm
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize specialized healthcare agents"""
        specialties = [
            ("General Practitioner", "general"),
            ("Cardiologist", "cardiology"),
            ("Neurologist", "neurology"),
            ("Pediatrician", "pediatrics"),
            ("Psychiatrist", "psychiatry"),
            ("Pharmacist", "pharmacy")
        ]
        
        for name, specialty in specialties:
            self.agents[specialty] = HealthcareAgent(name, specialty, self.rag_system, self.llm)
    
    def route_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Route query to appropriate agent(s)"""
        # Simple routing logic - can be enhanced with ML classification
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["heart", "cardiac", "chest pain", "blood pressure"]):
            primary_agent = "cardiology"
        elif any(word in query_lower for word in ["brain", "headache", "seizure", "memory"]):
            primary_agent = "neurology"
        elif any(word in query_lower for word in ["child", "baby", "infant", "pediatric"]):
            primary_agent = "pediatrics"
        elif any(word in query_lower for word in ["medication", "drug", "prescription", "dosage"]):
            primary_agent = "pharmacy"
        elif any(word in query_lower for word in ["depression", "anxiety", "mental", "stress"]):
            primary_agent = "psychiatry"
        else:
            primary_agent = "general"
        
        # Get response from primary agent
        primary_response = self.agents[primary_agent].process_query(query, context)
        
        # For complex cases, might consult multiple agents
        if primary_response["confidence"] < 0.7:
            # Consult general practitioner for second opinion
            secondary_response = self.agents["general"].process_query(query, context)
            
            return {
                "primary_response": primary_response,
                "secondary_response": secondary_response,
                "recommendation": "Multiple perspectives provided due to complexity"
            }
        
        return {"primary_response": primary_response}

class HealthcareRAGApp:
    """Main healthcare RAG application"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize components
        self.rag_system = MilvusRAG("milvus_rag_db.db")
        self.llm = LocalLlamaModel()
        self.coordinator = MultiAgentCoordinator(self.rag_system, self.llm)
        
        # Initialize with sample data
        self._load_sample_data()
        
        # Setup routes
        self._setup_routes()
    
    def _load_sample_data(self):
        """Load sample healthcare documents"""
        sample_docs = [
            HealthcareDocument(
                id="doc_001",
                title="Hypertension Management Guidelines",
                content="High blood pressure (hypertension) is a common condition where blood pressure is consistently elevated. Management includes lifestyle changes like diet modification, regular exercise, and medication when necessary. ACE inhibitors, beta blockers, and diuretics are commonly prescribed.",
                category="cardiology",
                specialty="cardiology",
                timestamp=datetime.now(),
                metadata={"source": "clinical_guidelines"}
            ),
            HealthcareDocument(
                id="doc_002",
                title="Pediatric Fever Management",
                content="Fever in children is a common symptom that can indicate various conditions. For children over 3 months, fever under 38.5°C can be managed with rest and fluids. Acetaminophen or ibuprofen may be given for comfort. Seek medical attention if fever exceeds 39°C or persists.",
                category="pediatrics",
                specialty="pediatrics",
                timestamp=datetime.now(),
                metadata={"source": "pediatric_guidelines"}
            ),
            HealthcareDocument(
                id="doc_003",
                title="Common Drug Interactions",
                content="Drug interactions can occur when medications affect each other's effectiveness or increase side effects. Common interactions include warfarin with NSAIDs, statins with certain antibiotics, and ACE inhibitors with potassium supplements. Always check for interactions when prescribing multiple medications.",
                category="pharmacy",
                specialty="pharmacy",
                timestamp=datetime.now(),
                metadata={"source": "pharmacy_reference"}
            )
        ]
        
        for doc in sample_docs:
            self.rag_system.add_document(doc)
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Healthcare RAG Assistant</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .chat-container { max-width: 800px; margin: 0 auto; }
                    .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
                    .user-message { background-color: #e3f2fd; }
                    .bot-message { background-color: #f5f5f5; }
                    .input-area { margin-top: 20px; }
                    input[type="text"] { width: 70%; padding: 10px; }
                    button { padding: 10px 20px; margin-left: 10px; }
                </style>
            </head>
            <body>
                <div class="chat-container">
                    <h1>Healthcare RAG Assistant</h1>
                    <div id="chat-history"></div>
                    <div class="input-area">
                        <input type="text" id="user-input" placeholder="Ask a healthcare question...">
                        <button onclick="sendMessage()">Send</button>
                    </div>
                </div>
                
                <script>
                    function sendMessage() {
                        const input = document.getElementById('user-input');
                        const message = input.value.trim();
                        if (!message) return;
                        
                        // Add user message to chat
                        addMessage(message, 'user');
                        input.value = '';
                        
                        // Send to backend
                        fetch('/query', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ query: message })
                        })
                        .then(response => response.json())
                        .then(data => {
                            addMessage(data.response, 'bot');
                        })
                        .catch(error => {
                            addMessage('Error: ' + error, 'bot');
                        });
                    }
                    
                    function addMessage(message, sender) {
                        const chatHistory = document.getElementById('chat-history');
                        const messageDiv = document.createElement('div');
                        messageDiv.className = 'message ' + sender + '-message';
                        messageDiv.textContent = message;
                        chatHistory.appendChild(messageDiv);
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                    }
                    
                    // Allow Enter key to send message
                    document.getElementById('user-input').addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') {
                            sendMessage();
                        }
                    });
                </script>
            </body>
            </html>
            """)
        
        @self.app.route('/query', methods=['POST'])
        def query():
            try:
                data = request.json
                query_text = data.get('query', '')
                
                if not query_text:
                    return jsonify({"error": "No query provided"}), 400
                
                # Process query through multi-agent system
                result = self.coordinator.route_query(query_text)
                
                # Extract primary response
                primary_response = result.get('primary_response', {})
                response_text = primary_response.get('response', 'No response generated')
                
                return jsonify({
                    "response": response_text,
                    "agent": primary_response.get('agent', 'Unknown'),
                    "specialty": primary_response.get('specialty', 'general'),
                    "confidence": primary_response.get('confidence', 0.0),
                    "relevant_documents": primary_response.get('relevant_documents', [])
                })
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                return jsonify({"error": "Internal server error"}), 500
        
        @self.app.route('/add_document', methods=['POST'])
        def add_document():
            try:
                data = request.json
                doc = HealthcareDocument(
                    id=data['id'],
                    title=data['title'],
                    content=data['content'],
                    category=data['category'],
                    specialty=data['specialty'],
                    timestamp=datetime.now(),
                    metadata=data.get('metadata', {})
                )
                
                success = self.rag_system.add_document(doc)
                
                if success:
                    return jsonify({"message": "Document added successfully"})
                else:
                    return jsonify({"error": "Failed to add document"}), 500
                    
            except Exception as e:
                logger.error(f"Error adding document: {e}")
                return jsonify({"error": "Internal server error"}), 500
        
        @self.app.route('/agents', methods=['GET'])
        def get_agents():
            """Get list of available agents"""
            agents_info = []
            for specialty, agent in self.coordinator.agents.items():
                agents_info.append({
                    "name": agent.name,
                    "specialty": agent.specialty,
                    "conversations": len(agent.conversation_history)
                })
            return jsonify({"agents": agents_info})
    
    def run(self, host='0.0.0.0', port=5001, debug=True):
        """Run the Flask application"""
        self.app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    # Create and run the healthcare RAG app
    app = HealthcareRAGApp()
    print("Starting Healthcare RAG Assistant...")
    print("Make sure you have the Meta-Llama-3-8B-Instruct.Q4_K_M.gguf model file in the current directory")
    app.run()