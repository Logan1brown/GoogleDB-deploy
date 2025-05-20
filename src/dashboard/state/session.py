"""
Session state management for the dashboard.
Provides utilities for managing page-scoped state.
"""

__all__ = ['get_page_state', 'update_page_state', 'get_filter_state', 'update_filter_state', 'get_data_entry_state', 'update_data_entry_state', 'get_admin_state', 'update_admin_state']

import streamlit as st
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from src.dashboard.state.show_state import DataEntryState
from src.dashboard.state.admin_state import AdminState, TMDBMatchState, UserManagementState, AnnouncementState, TMDBMatchingState, RTMatchingState

@dataclass
class FilterState:
    """Common filter state used across pages."""
    source_type: Optional[str] = None
    genre: Optional[str] = None

def get_page_state(page_name: str) -> Dict[str, Any]:
    """Get state for a specific page.
    
    Args:
        page_name: Name of the page to get state for
        
    Returns:
        Dictionary containing the page's state
    """
    key = f"state_{page_name}"
    if key not in st.session_state:
        st.session_state[key] = {}
    return st.session_state[key]

def update_page_state(page_name: str, state: Any) -> None:
    """Update state for a specific page.
    
    Args:
        page_name: Name of the page to update state for
        state: New state to set
    """
    key = f"state_{page_name}"
    st.session_state[key] = state

def get_filter_state(page_name: str) -> FilterState:
    """Get filter state for a specific page.
    
    Args:
        page_name: Name of the page to get filter state for
        
    Returns:
        FilterState instance for the page
    """
    state = get_page_state(page_name)
    if "filters" not in state:
        state["filters"] = asdict(FilterState())
    return FilterState(**state["filters"])

def update_filter_state(page_name: str, filters: FilterState) -> None:
    """Update filter state for a specific page.
    
    Args:
        page_name: Name of the page to update filter state for
        filters: New filter state
    """
    state = get_page_state(page_name)
    state["filters"] = asdict(filters)

def get_data_entry_state() -> DataEntryState:
    """Get data entry state.
    
    Returns:
        DataEntryState instance
    """
    state = get_page_state("data_entry")
    if "data_entry" not in state:
        state["data_entry"] = asdict(DataEntryState())
    return DataEntryState(**state["data_entry"])

def update_data_entry_state(data_entry: DataEntryState) -> None:
    """Update data entry state.
    
    Args:
        data_entry: New data entry state
    """
    state = get_page_state("data_entry")
    state["data_entry"] = asdict(data_entry)

def clear_admin_state() -> None:
    """Clear admin dashboard state.
    
    This is useful when the state schema changes and we need to reset to defaults.
    """
    if "admin" in st.session_state:
        del st.session_state["admin"]

def get_admin_state() -> AdminState:
    """Get admin dashboard state.
    
    Returns:
        AdminState instance with all section states properly initialized.
    """
    # Clear state if api_metrics is present (schema migration)
    state = get_page_state("admin")
    if state and "admin" in state and "api_metrics" in state["admin"]:
        clear_admin_state()
        state = get_page_state("admin")
    
    if not state:
        state = {}
    if "admin" not in state:
        admin_state = AdminState()
        admin_state.rt_matching = RTMatchingState()
        state["admin"] = asdict(admin_state)
    
    # Convert nested dicts to proper state objects
    admin_dict = state["admin"]
    if isinstance(admin_dict["user_management"], dict):
        admin_dict["user_management"] = UserManagementState(**admin_dict["user_management"])
    if isinstance(admin_dict["announcements"], dict):
        admin_dict["announcements"] = AnnouncementState(**admin_dict["announcements"])
    if isinstance(admin_dict["tmdb_matching"], dict):
        matching_dict = admin_dict["tmdb_matching"]
        if isinstance(matching_dict.get("matches", []), list):
            matching_dict["matches"] = [TMDBMatchState(**m) if isinstance(m, dict) else m for m in matching_dict["matches"]]
        admin_dict["tmdb_matching"] = TMDBMatchingState(**matching_dict)
    if isinstance(admin_dict["rt_matching"], dict):
        admin_dict["rt_matching"] = RTMatchingState(**admin_dict["rt_matching"])
    
    return AdminState(**admin_dict)

def update_admin_state(admin_state: AdminState) -> None:
    """Update admin dashboard state.
    
    This is the ONLY way state should be updated in the admin dashboard.
    Do not modify st.session_state directly.
    
    Args:
        admin_state: New admin state to save
    """
    # Convert nested state objects to dicts
    admin_dict = asdict(admin_state)
    
    # Update the page state
    state = get_page_state("admin")
    state["admin"] = admin_dict
    update_page_state("admin", state)

def clear_match_session_state(match_id: int):
    """Clear all session state keys for a specific match.
    
    Args:
        match_id: ID of the match to clear state for
    """
    prefix = f"tmdb_match_{match_id}"
    keys_to_clear = [k for k in st.session_state.keys() if k.startswith(prefix)]
    for k in keys_to_clear:
        del st.session_state[k]

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
    elif section == "RT Matches":
        state.rt_matching = RTMatchingState()
        prefix = "rt_"
    
    # Clear section-specific session state
    for key in list(st.session_state.keys()):
        if key.startswith(prefix):
            del st.session_state[key]
    
    update_admin_state(state)

def clear_matching_state(admin_state: AdminState):
    """Clear TMDB matching state after a successful match.
    
    Args:
        admin_state: Current admin state to update
    """
    # Preserve success message if present
    success_message = admin_state.tmdb_matching.success_message
    
    # Clear session state for all matches
    for match in admin_state.tmdb_matching.matches:
        clear_match_session_state(match.our_show_id)
    
    # Reset matching state but restore success message
    admin_state.tmdb_matching = TMDBMatchingState()
    admin_state.tmdb_matching.success_message = success_message
    update_admin_state(admin_state)
