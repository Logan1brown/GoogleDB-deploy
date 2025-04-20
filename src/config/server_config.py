"""Server configuration and environment variable handling."""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ServerConfig:
    """Configuration for server settings."""
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.port = int(os.getenv('STREAMLIT_PORT', '8501'))
        self.host = os.getenv('STREAMLIT_HOST', 'localhost')
        
        # Additional server settings can be added here
        self.debug = os.getenv('STREAMLIT_DEBUG', 'false').lower() == 'true'
        self.allow_remote = os.getenv('STREAMLIT_ALLOW_REMOTE', 'false').lower() == 'true'

# Create a singleton instance
server_config = ServerConfig()
