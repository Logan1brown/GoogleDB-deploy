"""Shared authentication module for both dashboard and data entry applications."""

import streamlit as st
from supabase import Client, create_client
import asyncio
import os
import time
from functools import wraps
from typing import Optional, List

# Initialize Supabase client
def init_supabase():
    """Initialize Supabase client with proper URL formatting."""
    try:
        # Get URL and strip any whitespace
        url = st.secrets["SUPABASE_URL"].strip()
        
        # Ensure URL is properly formatted
        if not url.startswith('https://'):
            url = f"https://{url}"
        url = url.rstrip('/')
        
        # Basic URL validation
        if not url or len(url) < 10 or ' ' in url:
            raise ValueError("Invalid Supabase URL format")
            
        # Create client
        key = st.secrets["SUPABASE_ANON_KEY"].strip()
        if not key or len(key) < 20:  # Supabase keys are typically long
            raise ValueError("Invalid Supabase anon key format")
            
        # Create client
        client = create_client(url, key)
        
        # Try to get current session if it exists
        try:
            session = client.auth.get_session()
            if session:
                # Set session in Supabase client
                client.auth.set_session(session.access_token, session.refresh_token)
                
                # Update session state
                st.session_state.session = session
                st.session_state.access_token = session.access_token
                st.session_state.refresh_token = session.refresh_token
                st.session_state.authenticated = True
                st.session_state.last_refresh = time.time()
        except:
            # No session or invalid session, that's okay
            pass
            
        return client
        
    except Exception as e:
        st.error(f"Failed to initialize Supabase: {str(e)}")
        st.error("Please check your SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit secrets")
        raise

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
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = None
    if 'session' not in st.session_state:
        st.session_state.session = None

# Helper function to run async functions
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# Initialize auth state first
init_auth_state()

# Initialize Supabase client lazily
def get_supabase():
    """Get or create Supabase client instance for current session."""
    if 'supabase' not in st.session_state:
        st.session_state.supabase = init_supabase()
        
        # Try to restore session if we have tokens
        if (st.session_state.access_token and st.session_state.refresh_token):
            try:
                st.session_state.supabase.auth.set_session(
                    st.session_state.access_token,
                    st.session_state.refresh_token
                )
            except Exception as e:
                st.error(f"Failed to restore session: {str(e)}")
                st.session_state.authenticated = False
                st.session_state.access_token = None
                st.session_state.refresh_token = None
                st.session_state.session = None
                
    return st.session_state.supabase
    
# Get Supabase client
supabase = get_supabase()

# Try to restore session if we have tokens
if (st.session_state.access_token and st.session_state.refresh_token and 
    not st.session_state.authenticated):
    try:
        # Get a new session object
        session = supabase.auth.get_session()
        if session:
            st.session_state.session = session
            st.session_state.authenticated = True
            st.session_state.access_token = session.access_token
            st.session_state.refresh_token = session.refresh_token
            st.session_state.last_refresh = time.time()
    except Exception as e:
        st.error(f"Failed to restore session: {str(e)}")
        st.session_state.authenticated = False
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        st.session_state.session = None

def login(email: str, password: str) -> bool:
    """Handle user login.
    
    Args:
        email: User's email
        password: User's password
        
    Returns:
        bool: True if login successful, False otherwise
    """
    try:
        client = get_supabase()
        if not client:
            st.error("Failed to initialize Supabase client")
            return False
            
        # Attempt sign in
        try:
            auth = client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if not auth or not auth.session:
                st.error("No session returned from authentication")
                return False
                
        except Exception as auth_e:
            st.error(f"Login failed: {str(auth_e)}")
            return False
        
        # Store auth in session
        st.session_state.user = auth.user
        st.session_state.session = auth.session
        st.session_state.access_token = auth.session.access_token
        st.session_state.refresh_token = auth.session.refresh_token
        st.session_state.authenticated = True
        st.session_state.last_refresh = time.time()
        
        # Update Supabase client with auth token
        client.auth.set_session(auth.session.access_token, auth.session.refresh_token)
        return True
        
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        st.error(f"Error type: {type(e)}")
        st.session_state.authenticated = False
        return False

def logout():
    """Handle user logout."""
    try:
        client = get_supabase()
        if client:
            client.auth.sign_out()
    except:
        pass
        
    # Clear session state
    st.session_state.user = None
    st.session_state.session = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.authenticated = False
    
    # Clear Supabase client
    if 'supabase' in st.session_state:
        del st.session_state.supabase

def restore_session() -> bool:
    """Try to restore or refresh the user's session.
    
    Returns:
        bool: True if session was restored or refreshed successfully
    """
    try:
        client = get_supabase()
        if not client:
            return False
            
        # First try to get current session
        session = client.auth.get_session()
        if session:
            st.session_state.session = session
            st.session_state.access_token = session.access_token
            st.session_state.refresh_token = session.refresh_token
            st.session_state.authenticated = True
            st.session_state.last_refresh = time.time()
            return True
            
        # If no session, try to refresh
        auth = client.auth.refresh_session()
        if auth and auth.session:
            st.session_state.session = auth.session
            st.session_state.access_token = auth.session.access_token
            st.session_state.refresh_token = auth.session.refresh_token
            st.session_state.authenticated = True
            st.session_state.last_refresh = time.time()
            return True
            
        return False
            
    except Exception as e:
        st.error(f"Session restoration failed: {str(e)}")
        st.session_state.authenticated = False
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        st.session_state.session = None
        return False
        
def refresh_session() -> bool:
    """Refresh the user's session if needed.
    
    Returns:
        bool: True if session is valid or was refreshed successfully
    """
    if not st.session_state.authenticated:
        return False
        
    try:
        client = get_supabase()
        if not client:
            return False
            
        # Only refresh if we haven't refreshed in the last minute
        current_time = time.time()
        if (st.session_state.last_refresh and 
            current_time - st.session_state.last_refresh < 60):
            return True
            
        # Try to restore/refresh session
        return restore_session()
            
    except Exception as e:
        st.error(f"Session refresh failed: {str(e)}")
        logout()
        return False

def auth_required(func=None, required_roles=None):
    """Decorator to require authentication and optionally specific roles for a page or function.
    
    Can be used as @auth_required or @auth_required(required_roles=['admin'])
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            init_auth_state()
            
            # Try to restore session if we have tokens
            if (st.session_state.access_token and st.session_state.refresh_token and 
                not st.session_state.authenticated):
                restore_session()
            
            # Show login form if not authenticated
            if not st.session_state.authenticated:
                st.warning("Please log in to access this page")
                
                with st.form("login_form"):
                    email = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    submitted = st.form_submit_button("Login")
                    
                    if submitted:
                        if login(email, password):
                            # Force rerun after successful login
                            st.rerun()
                return
            
            # Session is authenticated, try to refresh if needed
            if not refresh_session():
                st.session_state.authenticated = False
                st.error("Session expired. Please log in again.")
                return
            
            if required_roles and not check_role_access(required_roles):
                st.error(f"Access denied. Required roles: {', '.join(required_roles)}")
                return
                
            return f(*args, **kwargs)
        return wrapper
    
    if func:
        return decorator(func)
    return decorator

def get_user_role() -> Optional[str]:
    """Get the current user's role."""
    try:
        if not st.session_state.authenticated:
            return None
            
        # Use service key client for role queries
        client = create_client(
            st.secrets["SUPABASE_URL"],
            st.secrets["SUPABASE_SERVICE_KEY"]
        )
        
        response = client.from_('user_roles').select('role').eq('id', st.session_state.user.id).single().execute()
        return response.data['role'] if response.data else None
    except Exception as e:
        st.error(f"Failed to get user role: {str(e)}")
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
