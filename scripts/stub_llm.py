from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn  # type: ignore[reportMissingImports]

app = FastAPI()

@app.get("/v1/internal/model/info")
def model_info():
    return {"model_name": "mock-llm", "loaded": True}

@app.get("/v1/internal/model/list")
def model_list():
    return {"model_names": ["mock-llm"]}

class CompReq(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.0

@app.post("/v1/completions")
def completions(_: CompReq):
    return {"id":"cmpl-mock","choices":[{"text":"OK"}],"model":"mock-llm"}

class ChatMsg(BaseModel):
    role: str
    content: str

class ChatReq(BaseModel):
    messages: List[ChatMsg]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.0

@app.post("/v1/chat/completions")
def chat(_: ChatReq):
    return {"id":"chat-mock","choices":[{"message":{"role":"assistant","content":"OK"}}],"model":"mock-llm"}

class GenReq(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.0

@app.post("/api/v1/generate")
def generate(_: GenReq):
    return {"results":[{"text":"OK"}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
