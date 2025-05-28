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

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import streamlit as st

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
    our_network: str  # Make network required
    our_year: str  # Make year required
    tmdb_id: int
    name: str
    
    # Optional fields (with defaults)
    networks: List[str] = field(default_factory=list)
    first_air_date: Optional[str] = None
    last_air_date: Optional[str] = None
    status: str = ""
    episodes_per_season: List[int] = field(default_factory=list)
    executive_producers: List[str] = field(default_factory=list)
    our_eps: List[str] = field(default_factory=list)  # Our executive producers
    
    # Match confidence scores
    confidence: float = 0.0
    title_score: float = 0.0
    network_score: float = 0.0
    ep_score: float = 0.0
    
    # Match status
    validated: bool = False
    
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
    validated_show_id: Optional[int] = None  # Track which show was just validated

@dataclass
class RTMatchingState:
    """State for RT matching section.
    
    Tracks state for collecting RT scores for shows.
    """
    error_message: Optional[str] = None
    success_message: Optional[str] = None

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
    rt_matching: RTMatchingState = field(default_factory=RTMatchingState)
