import sys
from loguru import logger
from scr.config import settings

def setup_logging():
    logger.remove()
    
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        enqueue=True
    )
    
    logger.add(
        "logs/rag_api.log",
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        rotation="10 MB",
        retention="7 days",
        enqueue=True
    )
    
    return logger

logger = setup_logging()