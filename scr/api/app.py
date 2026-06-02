from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
from scr.config import settings
from scr.logging_config import logger
from scr.rag.rag_engine import rag_engine
from scr.vector_store.milvus_store import milvus_store
from scr.embedding.embedding_service import embedding_service
from scr.chunks.semantic_chunk import SemanticChunker
import asyncio
import aiofiles
import os

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="RAG API",
    description="高可用、高并发RAG服务",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

Instrumentator().instrument(app).expose(app)

class ChatRequest(BaseModel):
    user_input: str
    session_id: Optional[str] = "default"
    stream: bool = True
    use_rag: bool = True

class ChatResponse(BaseModel):
    response: str
    session_id: str

class DocumentUploadRequest(BaseModel):
    collection_name: Optional[str] = None
    chunk_size: Optional[int] = 512
    chunk_overlap: Optional[int] = 64

class CollectionStats(BaseModel):
    collection_name: str
    count: int
    indexed: bool

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_window_minutes}minute")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request from session: {request.session_id}")
        
        if request.stream:
            async def generate():
                for chunk in rag_engine.stream_chat(
                    user_input=request.user_input,
                    session_id=request.session_id,
                    use_rag=request.use_rag
                ):
                    yield chunk
            
            return StreamingResponse(generate(), media_type="text/plain")
        else:
            response = rag_engine.chat(
                user_input=request.user_input,
                session_id=request.session_id,
                use_rag=request.use_rag
            )
            return ChatResponse(response=response, session_id=request.session_id)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_document")
@limiter.limit("10/minute")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: Optional[str] = None,
    chunk_size: Optional[int] = 512,
    chunk_overlap: Optional[int] = 64
):
    try:
        logger.info(f"Received document upload: {file.filename}")
        
        async with aiofiles.open(f"/tmp/{file.filename}", "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
        
        with open(f"/tmp/{file.filename}", 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunker = SemanticChunker(embedding_service)
        chunks = chunker.get_chunks(
            text=text,
            chunk_method="percentile",
            threshold=90,
            buffer_size=1
        )
        
        embeddings = embedding_service.generate_embeddings(chunks)
        
        collection = collection_name or settings.milvus_collection_name
        milvus_store.create_collection(collection)
        count = milvus_store.insert_embeddings(collection, chunks, embeddings)
        
        os.remove(f"/tmp/{file.filename}")
        
        logger.info(f"Document processed successfully, {count} chunks inserted")
        return {"status": "success", "chunks_inserted": count, "collection": collection}
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/create_collection")
@limiter.limit("5/minute")
async def create_collection(collection_name: str, dimension: Optional[int] = None):
    try:
        milvus_store.create_collection(collection_name, dimension)
        logger.info(f"Collection created: {collection_name}")
        return {"status": "success", "collection_name": collection_name}
    except Exception as e:
        logger.error(f"Collection creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collection_stats", response_model=CollectionStats)
@limiter.limit("20/minute")
async def get_collection_stats(collection_name: Optional[str] = None):
    try:
        collection = collection_name or settings.milvus_collection_name
        stats = milvus_store.get_collection_stats(collection)
        
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        
        return CollectionStats(
            collection_name=collection,
            count=stats["count"],
            indexed=stats["indexed"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Collection stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear_session")
@limiter.limit("10/minute")
async def clear_session(session_id: str):
    try:
        from scr.rag.rag_engine import RedisChatMessageHistory
        history = RedisChatMessageHistory(session_id)
        history.clear()
        logger.info(f"Session cleared: {session_id}")
        return {"status": "success", "session_id": session_id}
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RAG API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "scr.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=True
    )