"""TMDB client with retries and error handling."""
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Union
from .tmdb_cache import TMDBCache, cache_response
from .tmdb_logger import log_api_call
from .tmdb_models import TVShow, TVShowDetails, Genre, TVShowSeason
from urllib.parse import quote

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.config.api_config import APIConfig

class TMDBError(Exception):
    """Base exception for TMDB API errors."""
    pass

class TMDBRateLimitError(TMDBError):
    """Raised when rate limit is exceeded."""
    pass

class TMDBAuthenticationError(TMDBError):
    """Raised when authentication fails."""
    pass

def rate_limit(max_per_window: int = 40, window_seconds: int = 10):
    """Decorator to rate limit API calls.
    
    Args:
        max_per_window: Maximum number of calls allowed per window
        window_seconds: Time window in seconds
    """
    def decorator(func):
        last_reset = time.time()
        calls = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_reset, calls
            
            current_time = time.time()
            elapsed = current_time - last_reset
            
            # Reset counter if window has passed
            if elapsed > window_seconds:
                calls = 0
                last_reset = current_time
            
            # Wait if we've hit the limit
            if calls >= max_per_window:
                sleep_time = window_seconds - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                calls = 0
                last_reset = time.time()
            
            calls += 1
            return func(*args, **kwargs)
        return wrapper
    return decorator

class TMDBClient:
    """Wrapper around TMDB API with better error handling."""
    
    BASE_URL = "https://api.themoviedb.org/3"
    
    def __init__(self, api_key: str = None, cache_ttl: int = 24):
        """Initialize the client with config.
        
        Args:
            api_key: Optional API key. If not provided, will try to load from environment.
        """
        if api_key:
            self.api_key = api_key
        else:
            config = APIConfig.from_env()
            self.api_key = config.tmdb_api_key
            
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json;charset=utf-8'
        }
        
        # Initialize cache
        self.cache = TMDBCache(ttl_hours=cache_ttl)
        
    @retry(
        retry=retry_if_exception_type((requests.RequestException, TMDBRateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    @rate_limit(max_per_window=40, window_seconds=10)
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a rate-limited request to TMDB API with retries.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Optional query parameters
            
        Returns:
            JSON response from API
            
        Raises:
            TMDBError: Base class for all TMDB-specific errors
            TMDBRateLimitError: When rate limit is exceeded
            TMDBAuthenticationError: When authentication fails
            requests.RequestException: For other request failures
        """
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 10))
            raise TMDBRateLimitError(f"Rate limit exceeded. Retry after {retry_after} seconds")
        elif response.status_code == 401:
            raise TMDBAuthenticationError("Invalid API key")
        elif response.status_code == 404:
            raise TMDBError(f"Resource not found: {endpoint}")
        
        response.raise_for_status()
        data = response.json()
        
        if 'success' in data and not data['success']:
            raise TMDBError(data.get('status_message', 'Unknown TMDB error'))
            
        return data
    
    def _create_tv_shows(self, data: List[Dict]) -> List[TVShow]:
        """Create TVShow objects from raw data."""
        return [TVShow.model_validate(show) for show in data]
    
    @cache_response(model_type=TVShow)
    @log_api_call
    def search_tv_show(self, query: str, language: str = 'en-US', page: int = 1) -> List[TVShow]:
        """Search for a TV show by name.
        
        Args:
            query: Show name to search for
            language: ISO 639-1 language code (default: en-US)
            page: Page number for paginated results (default: 1)
            
        Returns:
            List of matching TV shows with their details including:
            - id: TMDB show ID
            - name: Show name
            - original_name: Original show name
            - overview: Show description
            - first_air_date: First air date
            - genre_ids: List of genre IDs
        """
        endpoint = "/search/tv"
        params = {
            'query': quote(query),
            'language': language,
            'page': page,
            'include_adult': False
        }
        
        response = self._make_request(endpoint, params)
        return self._create_tv_shows(response.get('results', []))
    
    def _create_tv_show_details(self, data: Dict) -> TVShowDetails:
        """Create TVShowDetails object from raw data."""
        return TVShowDetails.model_validate(data)
    
    @cache_response(model_type=TVShowDetails)
    @log_api_call
    def get_tv_show_details(self, show_id: int, language: str = 'en-US') -> TVShowDetails:
        """Get detailed information about a specific TV show.
        
        Args:
            show_id: TMDB ID of the show
            language: ISO 639-1 language code (default: en-US)
            
        Returns:
            Detailed show information including:
            - id: TMDB show ID
            - name: Show name
            - overview: Show description
            - first_air_date: First air date
            - genres: List of genre objects with id and name
            - status: Show status (e.g., 'Returning Series')
            - type: Show type
            - number_of_seasons: Total seasons
        """
        endpoint = f"/tv/{show_id}"
        params = {
            'language': language,
            'append_to_response': 'content_ratings,keywords'
        }
        response = self._make_request(endpoint, params)
        return self._create_tv_show_details(response)
    
    def _create_genres(self, data: List[Dict]) -> List[Genre]:
        """Create Genre objects from raw data."""
        return [Genre.model_validate(genre) for genre in data]
    
    @cache_response(model_type=Genre)
    @log_api_call
    def get_genre_list(self) -> List[Genre]:
        """Get the list of official TMDB TV genres.
        
        Returns:
            List of genres with their IDs
        """
        endpoint = "/genre/tv/list"
        response = self._make_request(endpoint)
        return self._create_genres(response.get('genres', []))
        
    @cache_response()
    @log_api_call
    def get_tv_show_credits(self, show_id: int, language: str = 'en-US') -> Dict:
        """Get the cast and crew for a TV show.
        
        Args:
            show_id: TMDB ID of the show
            language: ISO 639-1 language code (default: en-US)
            
        Returns:
            Dictionary containing cast and crew information including:
            - cast: List of cast members with character and actor names
            - crew: List of crew members with job titles and departments
        """
        endpoint = f"/tv/{show_id}/credits"
        params = {'language': language}
        response = self._make_request(endpoint, params)
        return response
        
    @cache_response(model_type=TVShowSeason)
    @log_api_call
    def get_tv_show_season(self, show_id: int, season_number: int, language: str = 'en-US') -> TVShowSeason:
        """Get detailed information about a specific TV show season.
        
        Args:
            show_id: TMDB ID of the show
            season_number: Season number to get details for
            language: ISO 639-1 language code (default: en-US)
            
        Returns:
            Season information including episodes list
        """
        endpoint = f"/tv/{show_id}/season/{season_number}"
        params = {'language': language}
        response = self._make_request(endpoint, params)
        return TVShowSeason.model_validate(response)

