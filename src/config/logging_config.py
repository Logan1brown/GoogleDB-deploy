"""Logging configuration for the application."""
import logging
import sys
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent.parent / 'logs'
log_dir.mkdir(exist_ok=True)

def setup_logging(name: str) -> logging.Logger:
    """Set up logging with both file and console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Prevent duplicate handlers
    if not logger.handlers:
        # Console handler with WARNING level (less verbose)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.WARNING)
        console.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        
        # File handler with DEBUG level
        file_handler = logging.FileHandler(
            log_dir / 'sheets_api.log',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        logger.addHandler(console)
        logger.addHandler(file_handler)
    
    return logger
