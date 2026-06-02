from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from scr.config import settings
from scr.logging_config import logger
import numpy as np

class MilvusVectorStore:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MilvusVectorStore, cls).__new__(cls)
            cls._instance._connect()
        return cls._instance
    
    def _connect(self):
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port,
                timeout=30
            )
            logger.info("Milvus connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def create_collection(self, collection_name: str, dimension: int = None):
        dim = dimension or settings.milvus_dimension
        try:
            if utility.has_collection(collection_name):
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535)
            ]
            
            schema = CollectionSchema(fields=fields, description="RAG collection")
            collection = Collection(name=collection_name, schema=schema)
            
            index_params = {
                "metric_type": "IP",
                "index_type": "HNSW",
                "params": {"M": 8, "efConstruction": 64}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()
            
            logger.info(f"Collection {collection_name} created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def insert_embeddings(self, collection_name: str, texts: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict]] = None):
        try:
            if not utility.has_collection(collection_name):
                self.create_collection(collection_name)
            
            collection = Collection(name=collection_name)
            
            if metadatas is None:
                metadatas = [{} for _ in texts]
            
            import json
            entities = [
                {"embedding": emb, "text": text, "metadata": json.dumps(meta)}
                for emb, text, meta in zip(embeddings, texts, metadatas)
            ]
            
            result = collection.insert(entities)
            collection.flush()
            collection.load()
            
            logger.info(f"Inserted {len(texts)} records into {collection_name}")
            return result.insert_count
        except Exception as e:
            logger.error(f"Failed to insert embeddings: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def search(self, collection_name: str, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        try:
            if not utility.has_collection(collection_name):
                logger.warning(f"Collection {collection_name} does not exist")
                return []
            
            collection = Collection(name=collection_name)
            
            search_params = {
                "metric_type": "IP",
                "params": {"ef": 128}
            }
            
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=None if not filters else self._build_filter_expr(filters),
                output_fields=["text", "metadata"]
            )
            
            import json
            output = []
            for hit in results[0]:
                metadata = json.loads(hit.entity.get("metadata"))
                output.append({
                    "text": hit.entity.get("text"),
                    "metadata": metadata,
                    "distance": hit.distance,
                    "id": hit.id
                })
            
            return output
        except Exception as e:
            logger.error(f"Failed to search embeddings: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception)
    )
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        try:
            if not utility.has_collection(collection_name):
                return {"error": "Collection not found"}
            
            collection = Collection(name=collection_name)
            return {
                "count": collection.num_entities,
                "indexed": True if collection.has_index() else False
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def _build_filter_expr(self, filters: Dict) -> str:
        expr_parts = []
        for key, value in filters.items():
            if isinstance(value, str):
                expr_parts.append(f'{key} == "{value}"')
            else:
                expr_parts.append(f'{key} == {value}')
        return " && ".join(expr_parts)

milvus_store = MilvusVectorStore()