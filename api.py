from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict, Iterator
import uuid
import json

# MLX Imports
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

# LangChain Imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, AIMessageChunk
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Model State ---
model_ref = None
tokenizer_ref = None

@app.on_event("startup")
def load_model():
    global model_ref, tokenizer_ref
    print("Loading model...")
    model_ref, tokenizer_ref = load("mlx-community/LFM2-1.2B-8bit")
    print("Model loaded.")

# --- Custom LangChain Wrapper for MLX ---
class MLXChatModel(BaseChatModel):
    model_id: str = "mlx-community/LFM2-1.2B-8bit"
    temperature: float = 0.7
    max_tokens: int = 512

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Not implementing non-streaming for this demo validation, 
        # but required by abstract base class.
        # We will primarily use _stream.
        content = ""
        for chunk in self._stream(messages, stop, run_manager, **kwargs):
            content += chunk.message.content
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        
        if not model_ref or not tokenizer_ref:
            raise ValueError("Model not loaded yet")

        # Convert LangChain messages to MLX/HF format
        formatted_messages = []
        for m in messages:
            role = "user"
            if isinstance(m, AIMessage): role = "assistant"
            elif isinstance(m, SystemMessage): role = "system"
            formatted_messages.append({"role": role, "content": m.content})

        prompt = tokenizer_ref.apply_chat_template(
            formatted_messages, 
            tokenizer=False, 
            add_generation_prompt=True
        )
        
        sampler = make_sampler(temp=self.temperature)

        for response in stream_generate(
            model_ref, 
            tokenizer_ref, 
            prompt=prompt, 
            sampler=sampler, 
            max_tokens=self.max_tokens
        ):
            chunk_text = response.text
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
            
            if run_manager:
                run_manager.on_llm_new_token(chunk_text)

    @property
    def _llm_type(self) -> str:
        return "mlx-lm-custom"

# --- History Management ---
# --- History Management ---
# Replacing LangChain's SQLChatMessageHistory due to async driver compatibility issues.
import sqlite3
import aiosqlite
from langchain_core.messages import messages_from_dict, messages_to_dict

class SQLiteHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str, db_file: str = "localgpt.db"):
        self.session_id = session_id
        self.db_file = db_file
        self._init_table()

    def _init_table(self):
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message TEXT NOT NULL
                )
            """)
            conn.commit()

    @property
    def messages(self) -> List[BaseMessage]:
        # Sync retrieval
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute(
                "SELECT message FROM message_store WHERE session_id = ? ORDER BY id ASC", 
                (self.session_id,)
            )
            rows = cursor.fetchall()
        
        loaded_messages = []
        for row in rows:
            try:
               data = json.loads(row[0])
               loaded_messages.extend(messages_from_dict([data]))
            except:
                pass
        return loaded_messages

    async def aget_messages(self) -> List[BaseMessage]:
        # Async retrieval
        async with aiosqlite.connect(self.db_file) as db:
            async with db.execute(
                "SELECT message FROM message_store WHERE session_id = ? ORDER BY id ASC", 
                (self.session_id,)
            ) as cursor:
                rows = await cursor.fetchall()
        
        loaded_messages = []
        for row in rows:
            try:
               data = json.loads(row[0])
               loaded_messages.extend(messages_from_dict([data]))
            except:
                pass
        return loaded_messages

    def add_message(self, message: BaseMessage) -> None:
        # Sync add
        msg_json = json.dumps(messages_to_dict([message])[0])
        with sqlite3.connect(self.db_file) as conn:
            conn.execute(
                "INSERT INTO message_store (session_id, message) VALUES (?, ?)", 
                (self.session_id, msg_json)
            )
            conn.commit()
    
    async def aadd_messages(self, messages: List[BaseMessage]) -> None:
        # Async add
        async with aiosqlite.connect(self.db_file) as db:
            for message in messages:
                msg_json = json.dumps(messages_to_dict([message])[0])
                await db.execute(
                    "INSERT INTO message_store (session_id, message) VALUES (?, ?)", 
                    (self.session_id, msg_json)
                )
            await db.commit()
    
    def clear(self) -> None:
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("DELETE FROM message_store WHERE session_id = ?", (self.session_id,))
            conn.commit()

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLiteHistory(session_id=session_id)

# --- API Models ---
class Message(BaseModel):
    role: str
    content: str
    
class SessionInfo(BaseModel):
    session_id: str
    last_message: str
    timestamp: str = "" # Placeholder for now

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    session_id: Optional[str] = None # Added for history

# --- Endpoints ---

@app.get("/v1/sessions", response_model=List[SessionInfo])
def get_sessions():
    # Helper to get unique sessions from SQLite
    # This is a bit manual because SQLChatMessageHistory doesn't expose a "list sessions" method easily
    # We will connect directly to read unique session_ids
    import sqlite3
    try:
        conn = sqlite3.connect("localgpt.db")
        cursor = conn.cursor()
        # Get unique sessions and their last message (simplification)
        # Table name is typically 'message_store' created by LangChain
        cursor.execute("SELECT DISTINCT session_id FROM message_store")
        rows = cursor.fetchall()
        
        sessions = []
        for row in rows:
            sid = row[0]
            # Get last message for preview
            cursor.execute("SELECT message FROM message_store WHERE session_id = ? ORDER BY id DESC LIMIT 1", (sid,))
            last_msg_row = cursor.fetchone()
            last_msg_content = "New Conversation"
            if last_msg_row:
                try:
                    # LangChain stores messages as JSON string
                    msg_json = json.loads(last_msg_row[0])
                    last_msg_content = msg_json.get("data", {}).get("content", "Conversation")
                    if len(last_msg_content) > 50:
                        last_msg_content = last_msg_content[:50] + "..."
                except:
                    pass
            
            sessions.append(SessionInfo(session_id=sid, last_message=last_msg_content))
            
        conn.close()
        return sessions
    except Exception as e:
        print(f"Error reading sessions: {e}")
        return []

@app.delete("/v1/history/{session_id}")
def delete_history(session_id: str):
    history = get_session_history(session_id)
    if isinstance(history, SQLiteHistory):
        history.clear()
        return {"status": "deleted", "session_id": session_id}
    else:
         raise HTTPException(status_code=500, detail="History store not compatible")

@app.get("/v1/history/{session_id}")
def get_history(session_id: str):
    history = get_session_history(session_id)
    # Convert LangChain messages to our Message dict format
    messages = []
    for m in history.messages:
        role = "user"
        if isinstance(m, AIMessage): role = "assistant"
        elif isinstance(m, SystemMessage): role = "system"
        messages.append({"role": role, "content": m.content})
    return messages

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if not model_ref:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # If no session_id provided, generate one (stateless mode effectively, but we'll create a chain)
    session_id = request.session_id or str(uuid.uuid4())
    
    # Initialize implementation
    llm = MLXChatModel(
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens or 512
    )
    
    # Create the prompt template that includes history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | llm
    
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # We only need the LAST message from the user, because history handles the rest!
    # The frontend currently sends ALL messages. We can either:
    # 1. Ignore frontend history and trust server history (Better for "LangChain with Memory" goal)
    # 2. Use frontend history and ignore server memory (Stateless)
    
    # ADOPTING STRATEGY 1: Trust Server Memory.
    # We will take only the *latest* user message from the request.
    last_user_message = next((m.content for m in reversed(request.messages) if m.role == "user"), None)
    
    if not last_user_message:
         # Fallback if no user message found (rare)
         raise HTTPException(status_code=400, detail="No user message found")

    async def event_generator():
        # Stream the response
        async for chunk in with_message_history.astream(
            {"input": last_user_message},
            config={"configurable": {"session_id": session_id}}
        ):
            yield chunk.content

    return StreamingResponse(
        event_generator(), 
        media_type="text/plain", 
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/plain",
            "X-Accel-Buffering": "no",
        }
    )

@app.get("/health")
def health_check():
    return {"status": "ok", "backend": "langchain-mlx-sqlite"}
