"""Admin Dashboard Page

This page is only visible to admin users and provides access to administrative functions.
"""

import streamlit as st
import sys
import os
import secrets
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.shared.auth import auth_required, get_user_role, check_role_access
from src.dashboard.state.session import get_admin_state, update_admin_state
from src.dashboard.state.admin_state import MatchStatus
from src.dashboard.services.supabase import get_supabase_client
from supabase import create_client

def get_admin_client():
    """Get Supabase client with service role for admin operations."""
    return create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_SERVICE_KEY")
    )


def get_users() -> List[Dict[str, Any]]:
    """Get list of users with their roles."""
    try:
        # Get admin client
        client = get_admin_client()
        
        # Get users from auth API
        response = client.auth.admin.list_users()
        
        # Get roles from database
        roles_query = client.table('user_roles').select('*').execute()
        db_roles = {role['id']: role['role'] for role in roles_query.data}
        
        # Format user data
        formatted_users = []
        for user in response:
            # Get role from database or default to viewer
            role = db_roles.get(user.id, 'viewer')
            
            formatted_users.append({
                'id': user.id,
                'email': user.email,
                'role': role,
                'created_at': user.created_at,
                'last_sign_in': getattr(user, 'last_sign_in_at', None)
            })
        
        return formatted_users
        
    except Exception as e:
        return []
        
def update_user_role(user_id: str, new_role: str, reason: str) -> bool:
    """Update a user's role or delete the user if new_role is 'delete'."""
    try:
        # Get admin client
        client = get_admin_client()
        
        if new_role == 'delete':
            try:
                # Delete from user_roles first
                client.table('user_roles').delete().eq('id', user_id).execute()
                
                # Then delete from auth
                client.auth.admin.delete_user(user_id)
                
                st.success("User deleted successfully")
            except Exception as delete_error:
                st.error(f"Error during deletion: {str(delete_error)}")
                # Try to rollback user_roles deletion if auth delete failed
                try:
                    client.table('user_roles').upsert({
                        'id': user_id,
                        'role': 'viewer',  # Default to viewer role
                        'updated_by': st.session_state.user.id
                    }).execute()
                except:
                    pass  # Ignore rollback errors
                return False
        else:
            # Update role in user_roles table
            result = client.table('user_roles').upsert({
                'id': user_id,
                'role': new_role,
                'updated_by': st.session_state.user.id
            }).execute()
            
            st.success(f"Role updated to {new_role}")
        
        # Clean up session state
        for key in list(st.session_state.keys()):
            if key.startswith(('role_', 'reason_', 'update_')):
                del st.session_state[key]
        
        return True
        
    except Exception as e:
        st.error(f"Failed to {'delete user' if new_role == 'delete' else 'update role'}: {str(e)}")
        return False

def invite_user(email: str, role: str) -> bool:
    """Invite a new user with specified role using Supabase's invite system."""
    try:
        # Get admin client
        client = get_admin_client()
        
        # Invite user with role in metadata
        response = client.auth.admin.invite_user_by_email(
            email,
            {
                'data': {'initial_role': role}
            }
        )
        user = response.user
        
        # Set initial role in database
        client.table('user_roles').insert({
            'id': user.id,
            'role': role,
            'created_by': st.session_state.user.id,
            'updated_by': st.session_state.user.id
        }).execute()
        
        st.success(f"Invitation email sent to {email}. Note: Limited to 4 invites per hour.")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            st.error("Rate limit reached. Please wait before sending more invitations (limit: 4 per hour).")
        else:
            st.error(f"Failed to invite user: {error_msg}")
        return False

def render_user_management():
    """Render the user management section."""
    # Get all users
    users = get_users()
    
    # User creation form first
    st.subheader("Create New User")
    with st.form("create_user", clear_on_submit=True):
        # Use form_ prefix for form keys to avoid conflicts
        email = st.text_input("Email", key="form_email")
        password = st.text_input("Password", type="password", key="form_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="form_confirm")
        role = st.selectbox("Role", ["viewer", "editor", "admin"], key="form_role")
        submitted = st.form_submit_button("Create User")
        
        if submitted:
            if not email or not password:
                st.error("Please enter both email and password")
            elif password != confirm_password:
                st.error("Passwords do not match")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters")
            else:
                try:
                    # Create user with password
                    client = get_admin_client()
                    response = client.auth.admin.create_user({
                        'email': email,
                        'password': password,
                        'email_confirm': True  # Skip email verification
                    })
                    user = response.user
                    
                    # Set role in user_roles table
                    client.table('user_roles').insert({
                        'id': user.id,
                        'role': role,
                        'created_by': st.session_state.user.id,
                        'updated_by': st.session_state.user.id
                    }).execute()
                    
                    st.success(f"Successfully created user {email} with role {role}")
                    # Form will clear automatically due to clear_on_submit=True
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to create user: {str(e)}")
    
    # Manage existing users
    st.subheader("Manage Users")
    for user in users:
        # Show revoked status in title if applicable
        title = f"{user['email']} - {user['role']}"
        
        with st.expander(title):
            # Show user details
            st.markdown(f"**Current Role:** {user['role']}")
            st.markdown(f"**Last Login:** {user['last_sign_in'] or 'Never'}")
            
            # Only show role management if it's not the current user
            if user['id'] != st.session_state.user.id:
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    new_role = st.selectbox(
                        "New Role",
                        ["admin", "editor", "viewer", "delete"],
                        key=f"role_{user['id']}"
                    )
                    
                    reason = st.text_input(
                        "Reason for change",
                        key=f"reason_{user['id']}",
                        placeholder="Required for role changes"
                    )
                    
                with col2:
                    if st.button("Update", key=f"update_{user['id']}"):
                        if not reason:
                            st.error("Please provide a reason for the role change")
                        elif new_role == 'delete' and not reason:
                            st.error("Please provide a reason for deleting user")
                        elif update_user_role(user['id'], new_role, reason):
                            st.success("Role updated successfully")
                            st.rerun()
                        else:
                            st.error("Failed to update role")


def render_announcements():
    """Render the announcements section."""
    st.subheader("Announcements")
    st.info("Coming soon: Review and manage announcements from Deadline and other sources")


def render_tmdb_matches():
    """Render the TMDB matches section.
    
    This section allows admins to:
    1. Search for shows in TMDB
    2. Review and approve/reject matches
    3. Handle batch operations on matches
    4. Track integration progress
    """
    state = get_admin_state()
    
    # API Metrics
    st.subheader("API Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("API Calls", f"{state.api_calls_total} total", f"{state.api_calls_remaining} remaining")
    with col2:
        cache_ratio = (state.cache_hits / (state.cache_hits + state.cache_misses)) * 100 if state.cache_hits + state.cache_misses > 0 else 0
        st.metric("Cache", f"{state.cache_hits} hits", f"{cache_ratio:.1f}% hit rate")
    with col3:
        reset_in = max(0, state.api_window_reset_time - time.time())
        st.metric("Rate Limit", f"{state.api_calls_remaining}/40", f"Reset in {reset_in:.1f}s")
    
    # Unmatched Shows
    st.subheader("Unmatched Shows")
    
    # Get unmatched shows from database
    supabase = get_supabase_client()
    response = supabase.table('show_details') \
        .select('id', 'title', 'network_name', 'date') \
        .is_('tmdb_id', 'null') \
        .execute()
    
    # Transform response
    unmatched_shows = [{
        'id': show['id'],
        'title': show['title'],
        'year': show['date'].split('-')[0] if show.get('date') else None,
        'network': show.get('network_name')
    } for show in response.data]
    
    if not unmatched_shows:
        st.info("No unmatched shows found!")
        return
        
    # Display shows in a table with buttons
    st.write("Click 'Find Matches' to search TMDB for potential matches:")
    
    for show in unmatched_shows:
        col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
        with col1:
            st.write(show['title'])
        with col2:
            st.write(show.get('network') or 'No network')
        with col3:
            st.write(show.get('year') or 'No year')
        with col4:
            if st.button('Find Matches', key=f"find_matches_{show['id']}"):
                st.session_state.selected_show = show
    
    # Search Interface
    st.subheader("Search TMDB")
    col1, col2 = st.columns([3, 1])
    with col1:
        # Search box with previous query preserved
        state.tmdb_search_query = st.text_input(
            "Search Shows",
            value=state.tmdb_search_query,
            placeholder="Enter show title..."
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("Search", type="primary"):
            # TODO: Implement search
            # 1. Search our database for show
            # 2. Search TMDB API
            # 3. Calculate match confidence
            # 4. Update state.tmdb_matches
            pass
    
    # Filters
    st.subheader("Match Review")
    col1, col2, col3 = st.columns(3)
    with col1:
        # Filter by status
        status_values = [status.value for status in MatchStatus]
        current_index = status_values.index(state.match_filter.value)
        status_filter = st.selectbox(
            "Status",
            status_values,
            index=current_index
        )
        state.match_filter = MatchStatus(status_filter)
    
    with col2:
        # Toggle low confidence matches
        state.show_low_confidence = st.checkbox(
            "Show Low Confidence Matches",
            value=state.show_low_confidence
        )
    
    with col3:
        # Batch operations
        if state.selected_match_ids:
            st.write(f"{len(state.selected_match_ids)} matches selected")
            # TODO: Add batch approve/reject buttons
    
    # Results table
    if state.tmdb_matches:
        # TODO: Implement results table with columns:
        # - Checkbox for selection
        # - Our Show Title
        # - TMDB Title
        # - Confidence Score
        # - Status
        # - Actions (Approve/Reject/View Details)
        pass
    else:
        st.info("Search for shows to see potential matches")
    
    # Progress tracking
    st.subheader("Integration Progress")
    # TODO: Add metrics:
    # - Total shows to match
    # - Shows matched
    # - Pending reviews
    # - Match success rate
    
    # Save state
    update_admin_state(state)


@auth_required(['admin'])
def admin_show():
    """Main function for admin dashboard."""
    try:

        st.title("Admin Dashboard")
        
        # Test admin client access
        client = get_admin_client()
        client.auth.admin.list_users()
        
        # Get admin state
        state = get_admin_state()
        
        # Update view based on radio selection
        state.current_view = st.radio(
            "Select Function",
            ["User Management", "Announcements", "TMDB Matches"],
            horizontal=True
        )
        update_admin_state(state)
        st.divider()
        
        # Render selected section
        if state.current_view == "User Management":
            render_user_management()
        elif state.current_view == "Announcements":
            render_announcements()
        elif state.current_view == "TMDB Matches":
            render_tmdb_matches()
    except Exception as e:
        st.error(f"Admin dashboard error: {str(e)}")

if __name__ == "__main__":
    admin_show()
