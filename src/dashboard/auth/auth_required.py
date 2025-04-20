"""Authentication decorator for dashboard pages."""

import streamlit as st
import toml
from pathlib import Path
from functools import wraps
from dashboard.auth.auth_state import init_auth_state, refresh_session
from src.shared.auth import get_user_role

def auth_required(required_roles=None):
    """Decorator to require authentication and specific roles for a page or function.
    Args:
        required_roles: List of roles allowed to access this page. If None, any authenticated user can access.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Initialize auth state
            init_auth_state()
            
            # Check if authenticated
            if not st.session_state.authenticated:
                st.sidebar.error("Please log in to access this page")
                st.stop()
                
            # Try to refresh session
            if not refresh_session():
                st.session_state.authenticated = False
                st.sidebar.error("Session expired. Please log in again")
                st.stop()
                
            # If no specific roles required, allow any authenticated user
            if not required_roles:
                return func(*args, **kwargs)
                
            # Get user role
            role = get_user_role()
            
            # Check if role is allowed
            if role not in required_roles:
                st.error(f"This page requires one of these roles: {', '.join(required_roles)}")
                st.stop()
                
            return func(*args, **kwargs)
            
        return wrapper
    return decorator
