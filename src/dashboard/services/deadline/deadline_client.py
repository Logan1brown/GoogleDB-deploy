"""Client for accessing Deadline's WordPress API with retries and error handling."""

import logging
import html
import time
from datetime import datetime
from functools import wraps
from typing import Dict, List, Optional

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logger = logging.getLogger(__name__)

class DeadlineError(Exception):
    """Base exception for Deadline API errors."""
    pass

class DeadlineRateLimitError(DeadlineError):
    """Raised when rate limit is exceeded."""
    pass

class DeadlineAuthenticationError(DeadlineError):
    """Raised when authentication fails."""
    pass

def rate_limit(max_per_window: int = 30, window_seconds: int = 10):
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

class DeadlineClient:
    """Client for accessing Deadline's WordPress API with retries and error handling."""
    
    def __init__(self):
        self.base_url = "https://deadline.com/wp-json/wp/v2"
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json;charset=utf-8'
        }
    
    @retry(
        retry=retry_if_exception_type((requests.RequestException, DeadlineRateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    @rate_limit(max_per_window=30, window_seconds=10)
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make a rate-limited request to Deadline API with retries.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Optional query parameters
            
        Returns:
            JSON response from API
            
        Raises:
            DeadlineError: Base class for all Deadline-specific errors
            DeadlineRateLimitError: When rate limit is exceeded
            DeadlineAuthenticationError: When authentication fails
            requests.RequestException: For other request failures
        """
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 10))
            raise DeadlineRateLimitError(f"Rate limit exceeded. Retry after {retry_after} seconds")
        elif response.status_code == 401:
            raise DeadlineAuthenticationError("Authentication failed")
        elif response.status_code == 404:
            raise DeadlineError(f"Resource not found: {endpoint}")
        
        response.raise_for_status()
        return response.json()
    
    def _get_post(self, post_id: int) -> Optional[Dict]:
        """Get a specific post by ID.
        
        Args:
            post_id: Post ID to retrieve
            
        Returns:
            Post data including title, URL, and date
        """
        try:
            return self._make_request(f"/posts/{post_id}")
        except DeadlineError as e:
            logger.error(f"Deadline API error getting post {post_id}: {e}")
            return None
        except requests.RequestException as e:
            logger.error(f"Request error getting post {post_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting post {post_id}: {e}")
            return None
        
    def search_straight_to_series(self) -> List[Dict]:
        """Search for 'straight to series' articles on Deadline.
        
        Returns:
            List of dicts with article info (title, url, published_date)
        """
        try:
            # Search for articles
            search_results = self._make_request(
                "/search",
                params={
                    "page": 1,
                    "per_page": 50,  # This gets most articles in one request
                    "search": "straight to series",
                    "type": "post"
                }
            )
            
            # Get full post data for each result
            articles = []
            for result in search_results:
                post_data = self._get_post(result['id'])
                if post_data:
                    articles.append({
                        'title': self._clean_title(post_data['title']['rendered']),
                        'url': post_data['link'],
                        'published_date': post_data['date'][:10]  # YYYY-MM-DD
                    })
            
            logger.info(f"Found {len(articles)} articles")
            return articles
            
        except DeadlineError as e:
            logger.error(f"Deadline API error searching articles: {e}")
            return []
        except requests.RequestException as e:
            logger.error(f"Request error searching articles: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error searching articles: {e}")
            return []
    
    def _clean_title(self, title: str) -> str:
        """Clean HTML entities from title."""
        return html.unescape(title)
