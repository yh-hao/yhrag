from langchain_ollama import OllamaEmbeddings
from typing import List, Optional, Union
from scr.model.base import EmbeddingGenerator
import numpy as np



class EmbeddingModel(EmbeddingGenerator):
    def __init__(self, model_name: str="qwen3-embedding", keep_alive: int = -1):
        self.model_name = model_name
        self.client = OllamaEmbeddings(model=self.model_name, keep_alive=keep_alive)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        embeddings = np.array(self.client.embed_documents(texts))
        embeddings = np.nan_to_num(embeddings, nan=0.0)
        return embeddings

if __name__ == "__main__":
    embedding_generator = EmbeddingModel()
    print(embedding_generator.generate_embeddings(["我在长沙，我在长沙，今天下雨了"]))