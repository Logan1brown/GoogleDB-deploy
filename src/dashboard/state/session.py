"""
Session state management for the dashboard.
Provides utilities for managing page-scoped state.
"""

__all__ = ['get_page_state', 'update_page_state', 'get_filter_state', 'update_filter_state', 'get_data_entry_state', 'update_data_entry_state', 'get_admin_state', 'update_admin_state']

import streamlit as st
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from src.dashboard.state.show_state import DataEntryState
from src.dashboard.state.admin_state import AdminState

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

def get_admin_state() -> AdminState:
    """Get admin dashboard state.
    
    Returns:
        AdminState instance
    """
    if 'admin_state' not in st.session_state:
        st.session_state.admin_state = asdict(AdminState())
    return AdminState(**st.session_state.admin_state)

def update_admin_state(admin_state: AdminState) -> None:
    """Update admin dashboard state.
    
    Args:
        admin_state: New admin state
    """
    st.session_state.admin_state = asdict(admin_state)
