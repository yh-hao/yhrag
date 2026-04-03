from langchain_ollama import OllamaEmbeddings
from typing import List, Optional, Union


class EmbeddingModel:
    def __init__(self, model_name: str="bge-m3:567m", keep_alive: int = -1):
        self.model_name = model_name
        self.client = OllamaEmbeddings(model=self.model_name, keep_alive=keep_alive)

    def get_embedding(self, text: Union[str, List[str]]) -> List[float]:
        return self.client.embed_query(text)

