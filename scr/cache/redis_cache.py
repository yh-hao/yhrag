import json
import redis
from typing import Any, Optional
from scr.config import settings
from scr.logging_config import logger

class RedisCache:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RedisCache, cls).__new__(cls)
            cls._instance._connect()
        return cls._instance
    
    def _connect(self):
        try:
            self.client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                health_check_interval=30
            )
            self.client.ping()
            logger.info("Redis connection established successfully")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
    
    def get(self, key: str) -> Optional[Any]:
        if not self.client:
            return None
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self.client:
            return False
        try:
            ttl = ttl or settings.cache_ttl_seconds
            self.client.set(key, json.dumps(value), ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        if not self.client:
            return False
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        if not self.client:
            return False
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        if not self.client:
            return None
        try:
            return self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Redis increment error: {e}")
            return None
    
    def get_session_history(self, session_id: str) -> Optional[list]:
        return self.get(f"session:{session_id}:history")
    
    def set_session_history(self, session_id: str, history: list) -> bool:
        return self.set(f"session:{session_id}:history", history, ttl=86400)

redis_cache = RedisCache()