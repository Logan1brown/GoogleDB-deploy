"""Shared authentication module for both dashboard and data entry applications."""

import streamlit as st
from supabase import Client, create_client
import asyncio
import os
from functools import wraps
from typing import Optional, List

# Initialize Supabase client
def init_supabase():
    """Initialize Supabase client with proper URL formatting."""
    try:
        # Get URL and strip any whitespace
        url = st.secrets["SUPABASE_URL"].strip()
        st.error(f"1. Raw URL: {url}")
        
        # Ensure URL is properly formatted
        if not url.startswith('https://'):
            url = f"https://{url}"
        url = url.rstrip('/')
        st.error(f"2. Final URL: {url}")
        
        # Create client
        key = st.secrets["SUPABASE_ANON_KEY"].strip()
        st.error(f"3. Key length: {len(key)}")
        
        # Create async client
        client = create_client(url, key)
        st.info(f"Connected to Supabase at: {url}")
        return client
        
    except Exception as e:
        st.error(f"Failed to initialize Supabase: {str(e)}")
        raise

# Initialize client
supabase = init_supabase()

# Helper function to run async functions
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

def init_auth_state():
    """Initialize authentication state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'refresh_token' not in st.session_state:
        st.session_state.refresh_token = None

def login(email: str, password: str) -> bool:
    """Handle user login.
    
    Args:
        email: User's email
        password: User's password
        
    Returns:
        bool: True if login successful, False otherwise
    """
    try:
        # Get auth token
        auth = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        # Store auth in session
        st.session_state.user = auth.user
        st.session_state.access_token = auth.session.access_token
        st.session_state.refresh_token = auth.session.refresh_token
        st.session_state.authenticated = True
        
        # Update Supabase client with auth token
        supabase.auth.set_session(auth.session.access_token, auth.session.refresh_token)
        
        return True
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        st.session_state.authenticated = False
        return False

def logout():
    """Handle user logout."""
    try:
        supabase.auth.sign_out()
    except:
        pass
    finally:
        st.session_state.user = None
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        st.session_state.authenticated = False

def refresh_session() -> bool:
    """Refresh the user's session if needed.
    
    Returns:
        bool: True if session is valid or was refreshed successfully
    """
    if not st.session_state.authenticated:
        return False
        
    try:
        # Try to refresh the session
        auth = supabase.auth.refresh_session()
        if auth and auth.session:
            st.session_state.access_token = auth.session.access_token
            st.session_state.refresh_token = auth.session.refresh_token
            supabase.auth.set_session(auth.session.access_token, auth.session.refresh_token)
            return True
    except:
        # If refresh fails, log out
        logout()
        return False
        
    return False

def auth_required(func):
    """Decorator to require authentication for a page or function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        init_auth_state()
        
        if not st.session_state.authenticated:
            st.warning("Please log in to access this page")
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")
                
                if submitted:
                    if login(email, password):
                        st.rerun()
            return
            
        if not refresh_session():
            st.error("Session expired. Please log in again.")
            return
            
        return func(*args, **kwargs)
    return wrapper

def get_user_role() -> Optional[str]:
    """Get the current user's role."""
    if not st.session_state.authenticated:
        return None
        
    try:
        # Use service key client for role queries
        client = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_SERVICE_KEY")
        )
        
        response = client.from_('user_roles').select('role').eq('id', st.session_state.user.id).single().execute()
        return response.data['role'] if response.data else None
    except Exception as e:
        return None

def check_role_access(required_roles: List[str]) -> bool:
    """Check if current user has one of the required roles."""
    user_role = get_user_role()
    if not user_role:
        return False
    return user_role in required_roles

def get_supabase_client() -> Optional[Client]:
    """Get an authenticated Supabase client if user is logged in."""
    if not st.session_state.authenticated:
        return None
    return supabase
