from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from scr.config import settings
from scr.logging_config import logger
from scr.vector_store.milvus_store import milvus_store
from scr.embedding.embedding_service import embedding_service
from scr.cache.redis_cache import redis_cache
import json
import hashlib

class RedisChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.key = f"session:{session_id}:history"
    
    @property
    def messages(self):
        from langchain_core.messages import deserialize_messages
        data = redis_cache.get(self.key)
        if data:
            return deserialize_messages(data)
        return []
    
    def add_message(self, message):
        from langchain_core.messages import serialize_messages
        current = self.messages
        current.append(message)
        redis_cache.set(self.key, serialize_messages(current), ttl=86400)
    
    def clear(self):
        redis_cache.delete(self.key)

class RAGEngine:
    def __init__(self):
        self.chat_model = ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_host,
            keep_alive=-1
        )
        self._build_rag_chain()
    
    def _build_rag_chain(self):
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的AI助手，擅长根据提供的上下文信息回答问题。
请严格按照以下规则回答：
1. 只使用提供的【检索到的上下文】中的信息进行回答
2. 如果上下文信息不足以回答问题，请明确说明
3. 不要编造信息
4. 回答要简洁准确

【检索到的上下文】:
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.base_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个乐于助人的AI助手。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
    
    def _retrieve_documents(self, query: str, collection_name: str = None, top_k: int = 5) -> List[str]:
        collection = collection_name or settings.milvus_collection_name
        
        try:
            query_embedding = embedding_service.generate_embedding(query)
            results = milvus_store.search(collection, query_embedding, top_k)
            
            documents = [result["text"] for result in results]
            logger.info(f"Retrieved {len(documents)} documents for query")
            return documents
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def _format_context(self, documents: List[str]) -> str:
        if not documents:
            return "没有找到相关文档。"
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"文档{i}:\n{doc}\n")
        
        return "\n".join(context_parts)
    
    def _get_rag_chain(self):
        def retrieve_and_format(query: str):
            docs = self._retrieve_documents(query)
            return self._format_context(docs)
        
        return (
            {
                "context": RunnablePassthrough() | retrieve_and_format,
                "input": RunnablePassthrough(),
                "chat_history": RunnablePassthrough()
            }
            | self.rag_prompt
            | self.chat_model
            | StrOutputParser()
        )
    
    def _get_base_chain(self):
        return self.base_prompt | self.chat_model | StrOutputParser()
    
    def get_chat_chain(self, use_rag: bool = True):
        base_chain = self._get_rag_chain() if use_rag else self._get_base_chain()
        
        return RunnableWithMessageHistory(
            base_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )
    
    def _get_session_history(self, session_id: str):
        return RedisChatMessageHistory(session_id)
    
    def chat(self, user_input: str, session_id: str = "default", use_rag: bool = True) -> str:
        cache_key = f"chat:{session_id}:{hashlib.md5(user_input.encode()).hexdigest()}"
        cached = redis_cache.get(cache_key)
        
        if cached:
            logger.info(f"Cache hit for session {session_id}")
            return cached
        
        chain = self.get_chat_chain(use_rag)
        config = {"configurable": {"session_id": session_id}}
        
        try:
            response = chain.invoke({"input": user_input}, config=config)
            redis_cache.set(cache_key, response, ttl=3600)
            return response
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise
    
    def stream_chat(self, user_input: str, session_id: str = "default", use_rag: bool = True):
        chain = self.get_chat_chain(use_rag)
        config = {"configurable": {"session_id": session_id}}
        
        try:
            response = chain.stream({"input": user_input}, config=config)
            for chunk in response:
                yield chunk
        except Exception as e:
            logger.error(f"Stream chat failed: {e}")
            raise

rag_engine = RAGEngine()