"""Cache implementation for TMDB API responses."""
import json
from datetime import datetime, timedelta, date
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

T = TypeVar('T')

class TMDBCache:
    """Simple file-based cache for TMDB API responses."""
    
    def __init__(self, cache_dir: str = ".cache/tmdb", ttl_hours: int = 24):
        """Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours for cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(hours=ttl_hours)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get path for cache key."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, key: str, model_type: Optional[Type[T]] = None) -> Optional[Union[Dict, T, List[T]]]:
        """Get value from cache if not expired.
        
        Args:
            key: Cache key to retrieve
            model_type: Optional Pydantic model type for deserialization
            
        Returns:
            Cached value, deserialized if model_type provided
        """
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return None
            
        try:
            data = json.loads(cache_path.read_text())
            cached_time = datetime.fromisoformat(data["cached_at"])
            
            if datetime.now() - cached_time > self.ttl:
                cache_path.unlink()
                return None
                
            value = data["value"]
            
            # Deserialize with model type if provided
            if model_type:
                if isinstance(value, list):
                    return [model_type.model_validate(item) for item in value]
                return model_type.model_validate(value)
                
            return value
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for JSON storage."""
        if isinstance(value, date):
            return value.isoformat()
        elif isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif hasattr(value, 'model_dump'):
            # Handle Pydantic models
            return self._serialize_value(value.model_dump())
        return value

    def set(self, key: str, value: Any):
        """Set value in cache with current timestamp."""
        serialized = self._serialize_value(value)
        cache_data = {
            "cached_at": datetime.now().isoformat(),
            "value": serialized
        }
        self._get_cache_path(key).write_text(json.dumps(cache_data))
    
    def clear(self):
        """Clear all cached data."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

def cache_response(model_type: Optional[Type[T]] = None) -> Callable:
    """Decorator to cache API responses with optional model type."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg).replace('/', '_').replace('\\', '_') for arg in args)
            key_parts.extend(f"{k}={v}".replace('/', '_').replace('\\', '_') for k, v in sorted(kwargs.items()))
            # Add timestamp to force fresh request
            key_parts.append(str(datetime.now().timestamp()))
            cache_key = "_".join(key_parts)
            
            # Try to get from cache
            cached = self.cache.get(cache_key, model_type=model_type)
            if cached is not None:
                return cached
            
            # Call API and cache result
            result = func(self, *args, **kwargs)
            self.cache.set(cache_key, result)
            return result
            
        return wrapper
    return decorator
        
    return wrapper
