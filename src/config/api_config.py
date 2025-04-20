"""API configuration."""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class APIConfig:
    """API configuration settings."""
    tmdb_api_key: str
    
    @classmethod
    def from_env(cls) -> 'APIConfig':
        """Create config from environment variables."""
        tmdb_api_key = os.getenv('TMDB_API_KEY')
        if not tmdb_api_key:
            raise ValueError("TMDB_API_KEY environment variable not set")
        return cls(tmdb_api_key=tmdb_api_key)
