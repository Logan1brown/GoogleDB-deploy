"""Authentication state management for the dashboard."""

import streamlit as st
from supabase import create_client
import os
from functools import wraps

def get_supabase_client():
    """Get Supabase client with current session."""
    try:
        url = st.secrets["SUPABASE_URL"].strip()
        
        # If it's just the subdomain, append .supabase.co
        if not url.endswith('.supabase.co') and not url.startswith('http'):
            url = f"{url}.supabase.co"
        
        # Ensure https://
        if not url.startswith("https://"):
            url = f"https://{url}"
            
        client = create_client(url, st.secrets["SUPABASE_ANON_KEY"].strip())
        
        # If we have a session, set it
        if st.session_state.get('access_token') and st.session_state.get('refresh_token'):
            client.auth.set_session(
                st.session_state.access_token,
                st.session_state.refresh_token
            )
        
        return client
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {str(e)}")
        st.error(f"URL attempted: {url}")
        raise

def init_auth_state():
    """Initialize authentication state variables."""
    # Initialize state if not present
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'refresh_token' not in st.session_state:
        st.session_state.refresh_token = None
        
    # If we have tokens but not authenticated, try to restore session
    if (not st.session_state.authenticated and 
        st.session_state.access_token and 
        st.session_state.refresh_token):
        refresh_session()

def login(email: str, password: str) -> bool:
    """Handle user login.
    
    Args:
        email: User's email
        password: User's password
        
    Returns:
        bool: True if login successful, False otherwise
    """
    try:
        st.info("1. Starting login process...")
        
        # Get fresh client
        client = get_supabase_client()
        st.info("2. Got Supabase client")
        
        # Get auth token
        try:
            auth = client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            st.info("3. Sign in successful")
        except Exception as auth_error:
            st.error(f"3. Sign in failed: {str(auth_error)}")
            raise
        
        # Store auth in session
        st.session_state.user = auth.user
        st.session_state.access_token = auth.session.access_token
        st.session_state.refresh_token = auth.session.refresh_token
        st.session_state.authenticated = True
        st.info("4. Session state updated")
        
        return True
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        st.session_state.authenticated = False
        return False

def logout():
    """Handle user logout."""
    try:
        # Get fresh client
        client = get_supabase_client()
        client.auth.sign_out()
    except:
        pass  # Ignore errors on signout
    
    # Clear session state
    st.session_state.user = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None
    st.session_state.authenticated = False

def refresh_session():
    """Refresh the user's session if needed."""
    try:
        # Get fresh client
        client = get_supabase_client()
        
        # Try to get current session
        session = client.auth.get_session()
        if session:
            # Session is still valid
            st.session_state.user = session.user
            st.session_state.access_token = session.access_token
            st.session_state.refresh_token = session.refresh_token
            st.session_state.authenticated = True
            return True
            
        # Try to refresh the session
        auth = client.auth.refresh_session()
        if auth and auth.session:
            st.session_state.user = auth.user
            st.session_state.access_token = auth.session.access_token
            st.session_state.refresh_token = auth.session.refresh_token
            st.session_state.authenticated = True
            return True
            
        # No valid session
        logout()
        return False
    except Exception as e:
        print(f"Session refresh error: {e}")
        # If refresh fails, log out
        logout()
        return False
