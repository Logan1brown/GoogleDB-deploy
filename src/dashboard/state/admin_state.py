"""
Admin dashboard state management.

This module handles state management for the admin dashboard, following the same
patterns used in data entry for consistency across the application.

State is organized hierarchically:
1. AdminState - Top level dashboard state
2. Section states (UserManagementState, TMDBMatchingState, etc.)
3. Component states where needed

All state changes must go through the state management functions to maintain
consistency and traceability.
"""

import streamlit as st
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
from .session import get_page_state, update_page_state

class MatchStatus(Enum):
    """Status of a TMDB match."""
    PENDING = "pending"  # Needs review
    APPROVED = "approved"  # Match confirmed
    REJECTED = "rejected"  # Match rejected
    MANUAL = "manual"  # Manually matched

@dataclass
class TMDBMatchState:
    """State for a single TMDB match."""
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
    ep_score: float = 0.0
    
    # UI state
    expanded: bool = False
    validation_error: Optional[str] = None

@dataclass
class UserManagementState:
    """State for user management section."""
    create_user_email: str = ""
    create_user_password: str = ""
    create_user_confirm: str = ""
    create_user_role: str = "viewer"
    users: List[Dict[str, Any]] = field(default_factory=list)
    selected_user_id: Optional[str] = None
    error_message: Optional[str] = None
    success_message: Optional[str] = None

@dataclass
class AnnouncementState:
    """State for announcements section."""
    title: str = ""
    content: str = ""
    announcements: List[Dict[str, Any]] = field(default_factory=list)
    selected_announcement_id: Optional[int] = None
    error_message: Optional[str] = None
    success_message: Optional[str] = None

@dataclass
class TMDBMatchingState:
    """State for TMDB matching section."""
    search_query: str = ""
    matches: List[TMDBMatchState] = field(default_factory=list)
    selected_match_ids: List[int] = field(default_factory=list)
    match_filter: MatchStatus = MatchStatus.PENDING
    show_low_confidence: bool = False
    our_eps: List[str] = field(default_factory=list)
    last_validation: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    success_message: Optional[str] = None

@dataclass
class APIMetricsState:
    """State for API metrics tracking."""
    calls_total: int = 0
    calls_remaining: int = 40
    window_reset_time: datetime = field(default_factory=datetime.now)
    cache_hits: int = 0
    cache_misses: int = 0

@dataclass
class AdminState:
    """Top-level state for admin dashboard.
    
    Each major section of the dashboard has its own state class to maintain
    clear boundaries and separation of concerns.
    """
    current_view: str = "User Management"
    user_management: UserManagementState = field(default_factory=UserManagementState)
    announcements: AnnouncementState = field(default_factory=AnnouncementState)
    tmdb_matching: TMDBMatchingState = field(default_factory=TMDBMatchingState)
    api_metrics: APIMetricsState = field(default_factory=APIMetricsState)

def get_admin_state() -> AdminState:
    """Get admin dashboard state.
    
    Returns:
        AdminState instance with all section states properly initialized.
    """
    state = get_page_state("admin")
    if "admin" not in state:
        state["admin"] = asdict(AdminState())
    return AdminState(**state["admin"])

def update_admin_state(admin_state: AdminState) -> None:
    """Update admin dashboard state.
    
    This is the ONLY way state should be updated in the admin dashboard.
    Do not modify st.session_state directly.
    
    Args:
        admin_state: New admin state to save
    """
    state = get_page_state("admin")
    state["admin"] = asdict(admin_state)

def clear_section_state(state: AdminState, section: str) -> None:
    """Clear state for a specific section.
    
    Args:
        state: Current admin state to update
        section: Name of section to clear ('User Management', 'Announcements', 'TMDB Matches')
    """
    # Reset section state
    if section == "User Management":
        state.user_management = UserManagementState()
        prefix = "user_"
    elif section == "Announcements":
        state.announcements = AnnouncementState()
        prefix = "announcement_"
    elif section == "TMDB Matches":
        state.tmdb_matching = TMDBMatchingState()
        prefix = "tmdb_"
    
    # Clear section-specific session state
    for key in list(st.session_state.keys()):
        if key.startswith(prefix):
            del st.session_state[key]
    
    update_admin_state(state)

def clear_matching_state(admin_state: AdminState) -> None:
    """Clear TMDB matching state after a successful match.
    
    Args:
        admin_state: Current admin state to update
    """
    clear_section_state(admin_state, "TMDB Matches")

    
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
