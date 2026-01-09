from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
import json
from typing import List, Optional

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and tokenizer
model = None
tokenizer = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512

@app.on_event("startup")
def load_model():
    global model, tokenizer
    print("Loading model...")
    # Load the same model as used in chat.py
    model, tokenizer = load("mlx-community/LFM2-1.2B-8bit")
    print("Model loaded.")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = tokenizer.apply_chat_template(
        [m.dict() for m in request.messages], 
        tokenizer=False, 
        add_generation_prompt=True
    )

    sampler = make_sampler(temp=request.temperature)

    async def event_generator():
        for response in stream_generate(
            model, 
            tokenizer, 
            prompt=prompt, 
            sampler=sampler, 
            max_tokens=request.max_tokens
        ):
            # Yield raw text directly for simplicity
            yield response.text
        
        # No [DONE] signal needed for raw stream, connection close ends it
        # but we can just stop yielding.

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
    return {"status": "ok"}
