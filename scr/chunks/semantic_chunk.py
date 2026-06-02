import numpy as np
from typing import List, Optional

class SemanticChunker:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
    
    def get_chunks(self, text: str, chunk_method: str = "percentile", 
                   threshold: float = 90, buffer_size: int = 1, 
                   max_chunk_size: int = 512) -> List[str]:
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
        
        embeddings = self.embedding_service.generate_embeddings(sentences)
        
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        if chunk_method == "percentile":
            cutoff = np.percentile(similarities, threshold)
            split_indices = [i + 1 for i, sim in enumerate(similarities) if sim < cutoff]
        elif chunk_method == "fixed":
            split_indices = list(range(max_chunk_size, len(sentences), max_chunk_size))
        else:
            split_indices = []
        
        chunks = []
        start = 0
        for idx in split_indices:
            chunk = " ".join(sentences[start:idx])
            chunks.append(chunk)
            start = idx
        
        if start < len(sentences):
            chunk = " ".join(sentences[start:])
            chunks.append(chunk)
        
        if buffer_size > 0 and len(chunks) > 1:
            chunks = self._add_buffer(chunks, buffer_size)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        import re
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def _add_buffer(self, chunks: List[str], buffer_size: int) -> List[str]:
        buffered_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                next_chunk = chunks[i + 1] if i + 1 < len(chunks) else ""
                next_sentences = next_chunk.split()[:buffer_size]
                buffered = chunk + " " + " ".join(next_sentences)
            elif i == len(chunks) - 1:
                prev_chunk = chunks[i - 1] if i - 1 >= 0 else ""
                prev_sentences = prev_chunk.split()[-buffer_size:]
                buffered = " ".join(prev_sentences) + " " + chunk
            else:
                prev_chunk = chunks[i - 1]
                next_chunk = chunks[i + 1]
                prev_sentences = prev_chunk.split()[-buffer_size:]
                next_sentences = next_chunk.split()[:buffer_size]
                buffered = " ".join(prev_sentences) + " " + chunk + " " + " ".join(next_sentences)
            
            buffered_chunks.append(buffered.strip())
        
        return buffered_chunks