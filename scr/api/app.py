from fastapi import FastAPI
from scr.model import BaseChatModel
import uvicorn
from typing import Optional
from pydantic import BaseModel


app = FastAPI()
chat_model = BaseChatModel()

class ChatRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = "default_user"
    stream: bool = True
@app.post("/chat")
def chat(request: ChatRequest):
    return chat_model.stream_chat(request.user_input, request.session_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
