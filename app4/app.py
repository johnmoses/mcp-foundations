# app.py

import asyncio
import json
import jwt
import sqlite3
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from queue import Queue

SECRET_KEY = "your-secret-key"  # Change this to a secure secret in production
ALGORITHM = "HS256"

app = FastAPI()

# CORS setup (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo only; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# SQLite session storage
conn = sqlite3.connect("sessions.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT,
    data TEXT
)
""")
conn.commit()

def read_session(session_id):
    cursor.execute("SELECT data, user_id FROM sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if row:
        return json.loads(row[0]), row[1]
    return None, None

def write_session(session_id, data, user_id):
    json_data = json.dumps(data)
    cursor.execute("""
    INSERT INTO sessions (session_id, user_id, data) VALUES (?, ?, ?)
    ON CONFLICT(session_id) DO UPDATE SET data=excluded.data, user_id=excluded.user_id
    """, (session_id, user_id, json_data))
    conn.commit()

def get_or_create_session(session_id, user_id):
    if session_id:
        data, uid = read_session(session_id)
        if data and uid == user_id:
            return session_id
    new_id = str(uuid.uuid4())
    write_session(new_id, {"history": []}, user_id)
    return new_id

def verify_jwt(token):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception:
        return None

# Load Hugging Face model and tokenizer
model_name = "gpt2"  # Replace with your preferred model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_text(prompt, queue):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    thread = Thread(target=model.generate, kwargs={
        "input_ids": inputs["input_ids"],
        "max_new_tokens": 100,
        "do_sample": True,
        "top_p": 0.95,
        "temperature": 0.7,
        "streamer": streamer
    })
    thread.start()

    try:
        for new_text in streamer:
            queue.put(new_text)
    except Exception:
        pass
    queue.put(None)  # Signal end

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/login")
async def login(username: str = Form(...)):
    # Demo: accept any username, no password
    user_id = username
    token = jwt.encode({"user_id": user_id, "username": username}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token}

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket, token: str):
    await websocket.accept()
    payload = verify_jwt(token)
    if not payload:
        await websocket.close(code=1008)
        return
    user_id = payload["user_id"]
    session_id = None

    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)

            if "params" in request and "arguments" in request["params"]:
                if session_id is None:
                    session_id = request["params"]["arguments"].get("session_id")
                session_id = get_or_create_session(session_id, user_id)
                request["params"]["arguments"]["session_id"] = session_id
                request["params"]["arguments"]["user_id"] = user_id

            prompt = request["params"]["arguments"]["prompt"]

            queue = Queue()
            loop = asyncio.get_event_loop()

            # Send initial empty response to initiate streaming UI
            await websocket.send_json({
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"content": [{"type": "text", "text": ""}]}
            })

            def run_generation():
                generate_text(prompt, queue)

            thread = Thread(target=run_generation)
            thread.start()

            full_response = ""
            while True:
                token_text = await loop.run_in_executor(None, queue.get)
                if token_text is None:
                    break
                full_response += token_text
                await websocket.send_json({
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"content": [{"type": "text", "text": full_response}]}
                })

            # Save session history (optional)
            session_data, _ = read_session(session_id)
            if not session_data:
                session_data = {"history": []}
            session_data["history"].append({"role": "user", "content": prompt})
            session_data["history"].append({"role": "assistant", "content": full_response})
            write_session(session_id, session_data, user_id)

    except WebSocketDisconnect:
        print(f"User {user_id} disconnected")
