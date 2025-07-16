from flask import Blueprint, request, jsonify, render_template
from flask_login import login_required, current_user
from ..models.chat import ChatMessage
from ..extensions import db
from ..services.multi_agent import TodoAgent, CalculatorAgent, RAGAgent, AgentOrchestrator
from ..services.mcp_client import MCPClientWrapper
from ..services.service import llm
import asyncio
import logging

chat_bp = Blueprint("chat", __name__, template_folder="../templates")


def llm_callable(prompt, max_tokens=512, temperature=0.3):
    return llm(prompt, max_tokens=max_tokens, temperature=temperature)


@chat_bp.route("/", methods=["GET"])
@login_required
def chat_page():
    # Load last 20 chat messages for current user, ordered oldest first
    messages = (
        ChatMessage.query.filter_by(user_id=current_user.id)
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )
    return render_template("chat.html", messages=messages)


@chat_bp.route("/message", methods=["POST"])
@login_required
async def chat_message():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please enter a message."})

    # Retrieve recent conversation history (last 20 messages)
    recent_msgs = (
        ChatMessage.query.filter_by(user_id=current_user.id)
        .order_by(ChatMessage.timestamp.desc())
        .limit(20)
        .all()
    )
    recent_msgs.reverse()  # oldest first

    conversation_history = []
    for msg in recent_msgs:
        role = "assistant" if msg.is_ai else "user"
        conversation_history.append({"role": role, "content": msg.message})

    try:
        # Initialize MCP client with async context manager
        async with MCPClientWrapper() as mcp_client:
            # Instantiate agents with the MCP client and LLM callable
            todo_agent = TodoAgent(mcp_client)
            calculator_agent = CalculatorAgent(mcp_client)
            rag_agent = RAGAgent(llm_callable)

            # Create orchestrator with all agents
            orchestrator = AgentOrchestrator({
                'todo': todo_agent,
                'calculator': calculator_agent,
                'rag': rag_agent,
            })

            # Delegate query to appropriate agent
            ai_response = await orchestrator.handle_query(user_message, conversation_history)

    except Exception as e:
        logging.error(f"Error during chat message processing: {e}", exc_info=True)
        return jsonify({"error": "Failed to process your request."}), 500
    
    # Save user message and AI response to DB
    try:
        user_msg_db = ChatMessage(user_id=current_user.id, message=user_message, is_ai=False)
        ai_msg_db = ChatMessage(user_id=current_user.id, message=ai_response, is_ai=True)
        db.session.add_all([user_msg_db, ai_msg_db])
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logging.error(f"DB error saving chat messages: {e}", exc_info=True)
        return jsonify({"error": "Failed to save chat messages."}), 500

    return jsonify({"response": ai_response})