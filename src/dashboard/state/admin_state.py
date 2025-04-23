"""
Admin dashboard state management.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from enum import Enum

class MatchStatus(Enum):
    """Status of a TMDB match."""
    PENDING = "pending"  # Needs review
    APPROVED = "approved"  # Match confirmed
    REJECTED = "rejected"  # Match rejected
    MANUAL = "manual"  # Manually matched

@dataclass
class TMDBMatch:
    """Represents a match between our show and TMDB."""
    our_show_id: int
    our_show_title: str
    tmdb_id: int
    tmdb_title: str
    confidence: float
    status: MatchStatus = MatchStatus.PENDING
    notes: str = ""

@dataclass
class AdminState:
    """State for admin dashboard."""
    # Navigation
    current_view: str = "User Management"  # Current selected view
    
    # User Management
    create_user_email: str = ""
    create_user_password: str = ""
    create_user_confirm: str = ""
    create_user_role: str = "viewer"
    
    # Announcements
    announcement_title: str = ""
    announcement_content: str = ""
    announcements: List[dict] = field(default_factory=list)
    
    # TMDB Integration
    tmdb_search_query: str = ""
    tmdb_matches: List[TMDBMatch] = field(default_factory=list)
    selected_match_ids: List[int] = field(default_factory=list)  # For batch operations
    match_filter: MatchStatus = MatchStatus.PENDING  # Filter view by status
    show_low_confidence: bool = False  # Whether to show low confidence matches
    
    # TMDB API metrics
    api_calls_total: int = 0
    api_calls_remaining: int = 40  # TMDB rate limit is 40 calls per 10s
    api_window_reset_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
