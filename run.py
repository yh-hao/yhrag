import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scr.config import settings
from scr.logging_config import logger

def main():
    logger.info("Starting RAG API server...")
    
    try:
        import uvicorn
        
        uvicorn.run(
            "scr.api.app:app",
            host=settings.api_host,
            port=settings.api_port,
            workers=settings.api_workers,
            reload=False,
            access_log=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    main()