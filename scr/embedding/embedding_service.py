import ollama
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from scr.config import settings
from scr.logging_config import logger
from scr.cache.redis_cache import redis_cache
import hashlib

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.ollama_model
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_embedding(self, text: str) -> List[float]:
        cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        cached = redis_cache.get(cache_key)
        
        if cached:
            return cached
        
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            embedding = response["embedding"]
            redis_cache.set(cache_key, embedding)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.generate_embedding(text))
        return embeddings

embedding_service = EmbeddingService()