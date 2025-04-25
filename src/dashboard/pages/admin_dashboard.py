"""Admin Dashboard Page

This page is only visible to admin users and provides access to administrative functions.
"""

import streamlit as st
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from ..components.tmdb_match_view import render_match_card
from ..components.unmatched_show_view import render_unmatched_show_row, render_unmatched_show_header
from ..services.supabase import get_supabase_client
from ..services.tmdb.match_service import TMDBMatchService
from ..services.tmdb.tmdb_client import TMDBClient
from ..state.admin_state import TMDBMatchState
from ..state.session import get_admin_state, update_admin_state, clear_section_state
from src.shared.auth import auth_required
from supabase import create_client
from dataclasses import dataclass

def validate_match(match: TMDBMatchState) -> bool:
    """Validate a TMDB match and save the data.
    
    Args:
        match: TMDBMatchState object containing match data and UI state
        
    Returns:
        True if validation succeeded, False if there were any errors
    """
    try:
        state = get_admin_state()
        matching = state.tmdb_matching
        metrics = state.api_metrics
        
        # Input validation
        if not match.our_show_id or not isinstance(match.our_show_id, int):
            matching.error_message = "Invalid show ID"
            update_admin_state(state)
            return False
        if not isinstance(match.tmdb_id, int):
            matching.error_message = "Invalid TMDB ID"
            update_admin_state(state)
            return False
            
        client = get_admin_client()
        
        # Check if show exists and doesn't have TMDB ID
        show_response = client.table('shows').select('*').eq('id', match.our_show_id).execute()
        if not show_response.data:
            matching.error_message = f"Show {match.our_show_title} (ID: {match.our_show_id}) not found"
            update_admin_state(state)
            return False
            
        existing_show = show_response.data[0]
        if existing_show.get('tmdb_id'):
            matching.error_message = f"Show {match.our_show_title} already has TMDB ID {existing_show['tmdb_id']}"
            return False
            
        # Handle no-match case
        if match.tmdb_id == -1:
            # Insert into no_tmdb_matches
            no_match_response = client.table('no_tmdb_matches').insert({
                'show_id': match.our_show_id
            }).execute()
            
            if not no_match_response.data:
                st.error("Failed to mark show as no-match")
                return False
            return True

        # Check if TMDB ID already exists in success metrics
        metrics_response = client.table('tmdb_success_metrics').select('id').eq('tmdb_id', match.tmdb_id).execute()
        if metrics_response.data:
            st.error(f"TMDB ID {match.tmdb_id} already exists in success metrics")
            return False
        
        # Get full show details from TMDB
        tmdb_client = TMDBClient()
        try:
            details = tmdb_client.get_tv_show_details(match.tmdb_id)
        except Exception as e:
            st.error(f"Failed to get TMDB details: {str(e)}")
            return
            
        if not details:
            st.error(f"No TMDB details found for ID {match.tmdb_id}")
            return
            
        # Validate required fields
        if not details.status:
            st.error("Missing required field: status")
            return
        
        # Use our data mapper to prepare updates
        from ..services.tmdb.tmdb_data_mapper import map_tmdb_success_metrics, map_tmdb_show_data
        
        # First prepare success metrics data
        metrics_data = map_tmdb_success_metrics(details)
        
        # Validate all required metrics fields exist
        if not all(key in metrics_data for key in ['tmdb_id', 'seasons', 'episodes_per_season', 'status']):
            st.error("Missing required metrics fields")
            return
        
        # Then prepare show updates
        show_updates = map_tmdb_show_data(details, existing_show)
        st.write(f"Show updates to apply: {show_updates}")
        
        # Start transaction
        # First update the show
        update_response = client.table('shows')\
            .update(show_updates)\
            .eq('id', match.our_show_id)\
            .execute()
        st.write(f"Show update response: {update_response.data}")
            
        if not update_response.data or len(update_response.data) == 0:
            st.error(f"Failed to update show {match.our_show_title}")
            return
        
        # Then insert metrics
        metrics_response = client.table('tmdb_success_metrics')\
            .insert(metrics_data)\
            .execute()
            
        if not metrics_response.data or len(metrics_response.data) == 0:
            # Rollback show update
            client.table('shows').update({"tmdb_id": None}).eq('id', match.our_show_id).execute()
            st.error("Failed to insert TMDB metrics")
            return
            
        # Set last validation in matching state
        matching.last_validation = {
            'show_id': match.our_show_id,
            'tmdb_id': match.tmdb_id,
            'timestamp': time.time()
        }
        
        # Set success message and clear UI state
        matching.success_message = f"Successfully validated match for {match.our_show_title}"
        matching.search_query = ""
        matching.matches = []
        matching.validated_show_id = match.our_show_id  # Track which show was validated
        
        # Clear any UI state and update state
        for key in list(st.session_state.keys()):
            if key.startswith('tmdb_'):
                del st.session_state[key]
        
        # Update state
        update_admin_state(state)
        
        return True
        
    except Exception as e:
        st.error(f"Error validating match: {str(e)}")
        # Attempt rollback
        try:
            client.table('shows').update({"tmdb_id": None}).eq('id', match.our_show_id).execute()
        except Exception as rollback_error:
            st.error(f"Failed to rollback changes: {str(rollback_error)}")
            st.error("Manual database check required")
        return False


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
    """
    state = get_admin_state()
    matching = state.tmdb_matching
    metrics = state.api_metrics
    match_service = TMDBMatchService()
    
    # Show any error messages
    if matching.error_message:
        st.error(matching.error_message)
        matching.error_message = None
        update_admin_state(state)
    
    # API Status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("API Calls", f"{metrics.calls_total} total", f"{metrics.calls_remaining} remaining")
    with col2:
        cache_ratio = (metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)) * 100 if metrics.cache_hits + metrics.cache_misses > 0 else 0
        st.metric("Cache", f"{metrics.cache_hits} hits", f"{cache_ratio:.1f}% hit rate")
    with col3:
        reset_in = max(0, (metrics.window_reset_time.timestamp() - datetime.now().timestamp()))
        st.metric("Rate Limit", f"{metrics.calls_remaining}/40", f"Reset in {reset_in:.1f}s")
    
    # Search TMDB
    st.text_input("Search Shows", 
                 value=matching.search_query,
                 placeholder="Enter show title...",
                 key="tmdb_search_bar")
    if not matching.matches or matching.validated_show_id:
        st.info("Search for shows to see potential matches")
    
    # Unmatched Shows section
    st.subheader("Unmatched Shows")
    
    # Get unmatched shows, excluding ones marked as no-match
    supabase = get_supabase_client()
    
    # First get all show_ids that have no matches
    no_match_ids = supabase.table('no_tmdb_matches')\
        .select('show_id')\
        .execute()
    no_match_ids = [row['show_id'] for row in no_match_ids.data]
    
    # Then get all shows without tmdb_id that aren't in no_match_ids
    response = supabase.table('shows')\
        .select(
            'id, title, network_id, date, network_list(network), show_team(name, role_type_id)'
        )\
        .is_('tmdb_id', 'null')\
        .not_.in_('id', no_match_ids)\
        .order('date', desc=True)\
        .execute()
    
    # Convert show_team and network data
    for show in response.data:
        # Handle network name
        network = show.pop('network_list', [None])[0]
        show['network_name'] = network['network'] if network else ''
        
        # Handle team members
        team = show.pop('show_team', [])
        # Include both producers and creators as EPs
        show['team_members'] = [
            {'name': member['name'], 'role': 'Executive Producer'}
            for member in team 
            if member['role_type_id'] in (2, 4)  # 2 = Producer, 4 = Creator
        ]
    unmatched_shows = response.data
    
    # Filter out the show that was just validated
    if matching.validated_show_id:
        unmatched_shows = [show for show in unmatched_shows if show['id'] != matching.validated_show_id]
    
    if not unmatched_shows:
        st.info("No unmatched shows found!")
        return
        
    # Display shows in a table with buttons
    st.write("Click 'Find Matches' to search TMDB for potential matches:")
    
    # Render table header
    render_unmatched_show_header()
    st.markdown("---")
    
    # Create containers for each show's info and potential matches
    for show in unmatched_shows:
        def on_find_matches(show_data):
            try:
                with st.spinner(f"Searching TMDB for {show_data['title']}..."):
                    # Get our EPs first
                    team_members = show_data.get('team_members', [])
                    our_eps = [member['name'] for member in team_members 
                             if member['role'].lower() == 'executive producer']
                    
                    # Get TMDB matches
                    matches = match_service.search_and_match(show_data)
                    
                    if not matches:
                        st.warning("No matches found")
                        return
                    
                    # Store matches and our_eps in state
                    try:
                        matching.matches = []
                        for match in matches:
                            # Convert match to state object with required fields
                            state_obj = TMDBMatchState(
                                our_show_id=show_data['id'],
                                our_show_title=show_data['title'],
                                tmdb_id=match.tmdb_id,
                                name=match.name,
                                networks=match.networks or [],
                                first_air_date=match.first_air_date,
                                executive_producers=match.executive_producers or [],
                                confidence=match.confidence,
                                title_score=match.title_score,
                                network_score=match.network_score
                            )
                            matching.matches.append(state_obj)
                            
                        matching.search_query = show_data['title']
                        matching.our_eps = our_eps
                        matching.last_validation = None  # Clear any previous validation
                    except Exception as e:
                        st.error(f"Error processing matches: {str(e)}")
                        return
                    
                    # Update metrics
                    try:
                        metrics.calls_total += 1
                        metrics.calls_remaining -= 1
                        
                        update_admin_state(state)
                    except Exception as e:
                        st.error(f"Error updating metrics: {str(e)}")
                        return
            except Exception as e:
                st.error(f"Error searching TMDB: {str(e)}")
        
        render_unmatched_show_row(show, on_find_matches)

    
    # No need to show validation result here anymore
    # The success message is shown directly in validate_match
    
    # Match Results
    if matching.matches and not matching.validated_show_id:
        st.subheader(f"Matches for '{matching.search_query}'")
        for match in matching.matches:
            # Add our_eps to match object for template
            match.our_eps = matching.our_eps
            
            # Use template to render match card
            render_match_card(match, validate_match)
    elif matching.success_message:
        # Show success message after container disappears
        st.success(matching.success_message)
        matching.success_message = None
        update_admin_state(state)

    # Save state
    update_admin_state(state)


@auth_required(['admin'])
def admin_show():
    """Main function for admin dashboard.
    
    This is the entry point for the admin dashboard. It handles:
    1. Access control
    2. View selection
    3. Section rendering
    """
    try:
        st.title("Admin Dashboard")
        
        # Test admin client access
        client = get_admin_client()
        client.auth.admin.list_users()
        
        # Get admin state
        state = get_admin_state()
        
        # Update view based on radio selection
        new_view = st.radio(
            "Select Function",
            ["User Management", "Announcements", "TMDB Matches"],
            horizontal=True,
            key="admin_view_selector"
        )
        
        # Only update state if view changed
        if new_view != state.current_view:
            # Clear old section state
            clear_section_state(state, state.current_view)
            # Update view
            state.current_view = new_view
            
        st.divider()
        
        # Render selected section
        if state.current_view == "User Management":
            render_user_management()
        elif state.current_view == "Announcements":
            render_announcements()
        elif state.current_view == "TMDB Matches":
            render_tmdb_matches()
            
    except Exception as e:
        # Debug error info
        st.write("Exception type:", type(e))
        st.write("Exception dir:", dir(e))
        st.write("Exception dict:", e.__dict__ if hasattr(e, '__dict__') else 'No dict')
        st.write("Exception str:", str(e))
        st.write("Exception repr:", repr(e))
        
        # Update error in appropriate section state
        state = get_admin_state()
        error_msg = str(e) if str(e) else "Unknown error occurred"
        
        if state.current_view == "User Management":
            state.user_management.error_message = error_msg
        elif state.current_view == "Announcements":
            state.announcements.error_message = error_msg
        elif state.current_view == "TMDB Matches":
            state.tmdb_matching.error_message = error_msg
        update_admin_state(state)
        
        st.error(f"Admin dashboard error: {error_msg}")

if __name__ == "__main__":
    admin_show()
