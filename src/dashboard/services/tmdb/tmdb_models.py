"""TMDB data models for request/response validation."""
from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ShowStatus:
    """Valid show statuses for success calculation."""
    RELIABLE = ['Returning Series', 'Ended', 'Canceled']
    IN_DEVELOPMENT = ['Planned', 'In Production', 'Pilot', 'In Development']
    UNRELIABLE = []  # Any status not in RELIABLE or IN_DEVELOPMENT
    
    @classmethod
    def is_reliable(cls, status: str) -> bool:
        """Check if a show status is considered reliable for success metrics."""
        return status in cls.RELIABLE

class Genre(BaseModel):
    """Genre model from TMDB."""
    id: int
    name: str

class TVShow(BaseModel):
    """TV show model from TMDB search results."""
    id: int
    name: str
    original_name: str
    overview: Optional[str] = None
    first_air_date: Optional[date] = None
    genre_ids: List[int] = Field(default_factory=list)
    popularity: float
    vote_average: float = 0.0
    vote_count: int = 0
    
    @field_validator('first_air_date', mode='before')
    @classmethod
    def validate_date(cls, v: Optional[str]) -> Optional[date]:
        """Convert empty string to None for dates."""
        if not v:
            return None
        return v  # Pydantic will convert valid date string to date object
        
    def model_dump_json(self) -> str:
        """Convert model to JSON string."""
        return self.model_dump_json()
        
    def dict(self) -> dict:
        """Convert model to dict for JSON serialization."""
        return self.model_dump()

class Network(BaseModel):
    """Network model from TMDB."""
    id: int
    name: str

class TVShowDetails(TVShow):
    """Detailed TV show model from TMDB."""
    genres: List[Genre] = Field(default_factory=list)
    networks: List[Network] = Field(default_factory=list)
    status: str
    type: str
    number_of_seasons: int
    number_of_episodes: int
    in_production: bool
    languages: List[str] = Field(default_factory=list)
    last_air_date: Optional[date] = None
    seasons: List[TVShowSeason] = Field(default_factory=list)
    
    @field_validator('last_air_date', mode='before')
    @classmethod
    def validate_last_air_date(cls, v: Optional[str]) -> Optional[date]:
        """Convert empty string to None for dates."""
        if not v:
            return None
        return v
    
    def get_genre_names(self) -> List[str]:
        """Get list of genre names."""
        return [genre.name for genre in self.genres]

class Episode(BaseModel):
    """Episode model from TMDB."""
    id: int
    name: str
    episode_number: int
    air_date: Optional[date] = None
    
    @field_validator('air_date', mode='before')
    @classmethod
    def validate_air_date(cls, v: Optional[str]) -> Optional[date]:
        """Convert empty string to None for dates."""
        if not v:
            return None
        return v

class TVShowSeason(BaseModel):
    """Season model from TMDB."""
    id: int
    name: str
    season_number: int
    episode_count: Optional[int] = None
    air_date: Optional[date] = None
    episodes: List[Episode] = Field(default_factory=list)
    
    def get_episode_count(self) -> int:
        """Get episode count either from field or episodes list."""
        if self.episode_count is not None:
            return self.episode_count
        return len(self.episodes)
    
    @field_validator('air_date', mode='before')
    @classmethod
    def validate_air_date(cls, v: Optional[str]) -> Optional[date]:
        """Convert empty string to None for dates."""
        if not v:
            return None
        return v
