from fastapi import FastAPI
from scr.model import BaseChatModel
import uvicorn
from typing import Optional, List
from pydantic import BaseModel
from scr.embedding import ChromaClientWrapper
from scr.model.embedding import EmbeddingModel
from scr.chunks.semantic_chunk import SemanticChunker


app = FastAPI()
chat_model = BaseChatModel()
chroma_client = ChromaClientWrapper()
embedding_generator = EmbeddingModel()

class ChatRequest(BaseModel):
    user_input: str
    session_id: Optional[str]
    stream: bool = True
@app.post("/chat")
def chat(request: ChatRequest):
    return chat_model.stream_chat(request.user_input, request.session_id)


class AddCollectionRequest(BaseModel):
    collection_name: str


@app.post("/add_collection")#selfkong
def add_collection(request: AddCollectionRequest):
    return chroma_client.add_collection(request.collection_name)

class SaveChunkRequest(BaseModel):
    collection_name: str
    file_path: str
    #embedding: List[float]


@app.post("/save_chunk")
def save_chunk(request: SaveChunkRequest):
    with open(request.file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    chunker = SemanticChunker(embedding_generator)
    chunks = chunker.get_chunks(
        text=text,
        chunk_method="percentile",
        threshold=90,  # 90th percentile of distances
        buffer_size=1)

    return chroma_client.save_chunk(request.collection_name, chunks, request.embedding)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
