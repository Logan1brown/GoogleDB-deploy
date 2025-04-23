"""Logging utilities for TMDB API client."""
import json
import logging
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

# Configure logger
logger = logging.getLogger("tmdb_client")
logger.setLevel(logging.INFO)

# Add console handler if not already added
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(console_handler)

def log_api_call(func: Callable) -> Callable:
    """Decorator to log API calls with timing and response status."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = datetime.now()
        endpoint = kwargs.get('endpoint', 'unknown')
        
        # Log request
        logger.info(f"API Request - Endpoint: {endpoint}")
        
        try:
            result = func(*args, **kwargs)
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Log success
            logger.info(
                f"API Response - Endpoint: {endpoint} - "
                f"Duration: {duration:.2f}s - Status: Success"
            )
            
            return result
            
        except Exception as e:
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Log error
            logger.error(
                f"API Error - Endpoint: {endpoint} - "
                f"Duration: {duration:.2f}s - Error: {str(e)}"
            )
            raise
            
    return wrapper
