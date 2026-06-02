import chromadb
from typing import List

class ChromaClientWrapper:

    def __init__(self):
        self.client = chromadb.PersistentClient(path="../store/store.db")

    def add_collection(self, name: str, dimension: int = 1536):
        self.client.get_or_create_collection(name=name)
        return True

    def save_chunk(self, collection_name: str, chunks: List[str], embedding: List[List[float]]):
        self.client.get_or_create_collection(name=collection_name)
        self.client.get_or_create_collection(name=collection_name).add(
            documents=chunks,
            embeddings=embedding,
            metadatas=[{"chunk": chunk} for chunk in chunks],
            ids=[f"doc_id_{i}" for i in range(len(chunks))]
        )
        return True

    def query(self, collection_name: str, query: str, n_results: int = 1):
        results = self.client.get_or_create_collection(name=collection_name).query(
            query_texts=query,
            n_results=n_results,
        )
        return results

if __name__ == "__main__":
    chroma_client = ChromaClientWrapper()
    chroma_client.query("test", "你好")

