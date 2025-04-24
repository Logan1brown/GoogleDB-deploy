"""
Admin dashboard state management.
"""

import streamlit as st
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
    # Required fields (no defaults)
    our_show_id: int
    our_show_title: str
    tmdb_id: int
    name: str
    
    # Optional fields (with defaults)
    our_network: Optional[str] = None
    our_year: Optional[str] = None
    networks: List[str] = field(default_factory=list)
    first_air_date: Optional[str] = None
    last_air_date: Optional[str] = None
    status: str = ""
    episodes_per_season: List[int] = field(default_factory=list)
    executive_producers: List[str] = field(default_factory=list)
    
    # Match confidence scores
    confidence: float = 0.0
    title_score: float = 0.0
    network_score: float = 0.0
    year_score: float = 0.0

@dataclass
class AdminState:
    """State for admin dashboard."""
    current_view: str = "User Management"
    
    # User management
    create_user_email: str = ""
    create_user_password: str = ""
    create_user_confirm: str = ""
    create_user_role: str = "viewer"
    
    # Announcements
    announcement_title: str = ""
    announcement_content: str = ""
    announcements: List[dict] = field(default_factory=list)
    
    # TMDB matching
    tmdb_search_query: str = ""
    tmdb_matches: List[TMDBMatch] = field(default_factory=list)
    selected_match_ids: List[int] = field(default_factory=list)
    match_filter: MatchStatus = MatchStatus.PENDING
    show_low_confidence: bool = False
    our_eps: List[str] = field(default_factory=list)
    
    # API stats
    api_calls_total: int = field(default=0)
    api_calls_remaining: int = field(default=40)
    api_window_reset_time: float = field(default=0.0)
    cache_hits: int = field(default=0)
    cache_misses: int = field(default=0)

def get_admin_state() -> AdminState:
    """Get the current admin state from session state."""
    if 'admin_state' not in st.session_state:
        st.session_state.admin_state = AdminState()
    return st.session_state.admin_state

def update_admin_state(state: AdminState):
    """Update the admin state in session state."""
    st.session_state.admin_state = state

    
    # TMDB Integration
    tmdb_search_query: str = ""
    tmdb_matches: List[TMDBMatch] = field(default_factory=list)
    selected_match_ids: List[int] = field(default_factory=list)  # For batch operations
    match_filter: MatchStatus = MatchStatus.PENDING  # Filter view by status
    show_low_confidence: bool = False  # Whether to show low confidence matches
    
    # TMDB API Metrics
    api_calls_total: int = field(default=0)  # Total API calls made
    api_calls_remaining: int = field(default=40)  # Remaining calls in current window
    api_window_reset_time: float = field(default=0.0)  # When rate limit window resets
    cache_hits: int = field(default=0)  # Number of cache hits
    cache_misses: int = field(default=0)  # Number of cache misses
