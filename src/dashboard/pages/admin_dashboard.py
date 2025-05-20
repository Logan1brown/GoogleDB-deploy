"""Admin Dashboard Page

This page is only visible to admin users and provides access to administrative functions.
"""

import streamlit as st
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from streamlit_searchbox import st_searchbox
from ..components.tmdb_match_view import render_match_card
from ..components.unmatched_show_view import render_unmatched_shows_table
from ..components.announcement_view import render_announcements_list
from ..services.supabase import get_supabase_client
from ..services.tmdb.match_service import TMDBMatchService
from ..services.tmdb.tmdb_client import TMDBClient
from ..services import show_service
from ..services.show_service import search_shows
from ..services.deadline.deadline_client import DeadlineClient
from ..utils.style_config import COLORS, FONTS
from ..state.admin_state import TMDBMatchState
from ..state.session import get_admin_state, update_admin_state, clear_section_state
from ..components.rt_tools.rt_matches import RTMatches
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
        
        # Use TMDBMatchService to validate
        match_service = TMDBMatchService(supabase_client=get_admin_client())
        match_service.validate_match(match)
        
        # Set success message
        matching.success_message = f"Successfully validated match for {match.our_show_title}"
        update_admin_state(state)
        
        # Clear state and rerun
        from ..state.session import clear_matching_state
        clear_matching_state(state)
        st.rerun()
        
        return True
        
    except Exception as e:
        # Log the error
        error_msg = str(e) if str(e) else "Unknown error occurred"
        st.error(error_msg)  # Show error immediately
        
        # Update error in state
        state = get_admin_state()
        matching = state.tmdb_matching
        matching.error_message = error_msg
        
        # Attempt rollback if needed
        try:
            client = get_admin_client()
            client.table('shows').update({"tmdb_id": None}).eq('id', match.our_show_id).execute()
        except Exception as rollback_error:
            st.error(f"Failed to rollback changes: {str(rollback_error)}")
            st.error("Manual database check required")
        
        # Update state
        update_admin_state(state)
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
    st.markdown(f"### Announcements")
    
    # Controls row with consistent spacing
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.caption("Filter")
        filter_status = st.selectbox(
            "##",  # Hidden label
            ["Unreviewed", "Reviewed", "All"],
            index=0,
            label_visibility="collapsed"
        )
    with col2:
        st.caption("Search Shows")
        selected_show = st_searchbox(
            search_shows,
            key="admin_show_search",
            placeholder="Search existing shows...",
            clear_on_submit=True
        )
    with col3:
        st.write("")
        if st.button("Fetch New", type="primary"):
                with st.spinner("Fetching articles..."):
                    deadline = DeadlineClient()
                    articles = deadline.search_straight_to_series()
                    
                    # Add new articles to database
                    client = get_admin_client()
                    new_count = 0
                    for article in articles:
                        try:
                            client.table('announcements').insert({
                                'url': article['url'],
                                'title': article['title'],
                                'published_date': article['published_date'],
                                'reviewed': False
                            }).execute()
                            new_count += 1
                        except Exception as e:
                            if "duplicate key value" not in str(e):
                                st.error(f"Error adding article: {e}")
                    
                    if new_count > 0:
                        st.success(f"Added {new_count} new articles")
                    else:
                        st.info("No new articles found")
                    st.rerun()
    if selected_show:
        st.info(f"✨ Found show: {selected_show} - Add it in Data Entry if needed")
    
    # Add divider before announcements
    st.markdown("---")
    
    # Get announcements based on filter
    client = get_admin_client()
    query = client.table('announcements').select('*')
    
    if filter_status == "Unreviewed":
        query = query.eq('reviewed', False)
    elif filter_status == "Reviewed":
        query = query.eq('reviewed', True)
    
    result = query.order('published_date', desc=True).execute()
    announcements = result.data
    
    if not announcements:
        st.info(f"No {filter_status.lower()} announcements")
        return

    def handle_review(announcement: Dict[str, Any]):
        """Handle review button click"""
        client.table('announcements')\
            .update({
                'reviewed': True,
                'reviewed_at': datetime.now().isoformat()
            })\
            .eq('id', announcement['id'])\
            .execute()
        st.rerun()

    # Render announcements with review callback
    render_announcements_list(announcements, on_review=handle_review)


def render_rt_matches():
    """Render the RT matches section.
    
    This section allows admins to:
    1. Search for shows on RT
    2. Collect and save RT scores
    """
    try:
        # Get admin state
        state = get_admin_state()
        rt_state = state.rt_matching
        
        # Get unmatched shows if needed
        if not rt_state.unmatched_shows:
            client = get_admin_client()
            response = client.from_('shows')\
                .select('id, title')\
                .is_('rt_success_metrics.rt_id', 'null')\
                .order('id')\
                .limit(5)\
                .execute()
                
            rt_state.unmatched_shows = response.data if response else []
            update_admin_state(state)
        
        # Handle score collection
        def handle_scores(data):
            st.session_state.rt_scores = data
            rt_state.success_message = f"Collected scores for {data['title']}"
            update_admin_state(state)
            # TODO: Save to database
        
        # Show messages
        if rt_state.error_message:
            st.error(rt_state.error_message)
            rt_state.error_message = None
            update_admin_state(state)
            
        if rt_state.success_message:
            st.success(rt_state.success_message)
            rt_state.success_message = None
            update_admin_state(state)
        
        # Create and render component
        rt_matches = RTMatches(rt_state.unmatched_shows, handle_scores)
        rt_matches.render()
        
    except Exception as e:
        state = get_admin_state()
        state.rt_matching.error_message = str(e)
        update_admin_state(state)

def render_tmdb_matches():
    """Render the TMDB matches section.
    
    This section allows admins to:
    1. Search for shows in TMDB
    2. Review and approve/reject matches
    """
    st.markdown("### Match TMDB")
    
    state = get_admin_state()
    matching = state.tmdb_matching
    match_service = TMDBMatchService()
    
    # Show any error messages
    if matching.error_message:
        st.error(matching.error_message)
        matching.error_message = None
        update_admin_state(state)
    
    # Get unmatched shows and show count
    unmatched_shows = show_service.get_unmatched_shows()
    
    # Get matched shows count using admin client
    supabase = get_supabase_client()
    matched_response = supabase.table('shows')\
        .select('id', count='exact')\
        .not_.is_('tmdb_id', 'null')\
        .execute()
    matched_count = matched_response.count if matched_response.count is not None else 0
    
    # Display metrics
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.metric("Unmatched Shows", len(unmatched_shows))
        with mcol2:
            st.metric("Matched Shows", matched_count)
    st.write("")
    
    # Add search box with caption
    st.caption("Search for shows to match with TMDB")
    selected_title = st_searchbox(
        search_function=search_shows,
        label="Search existing shows",
        placeholder="Start typing show title (3+ characters)...",
        key="tmdb_search_bar",
        clear_on_submit=False  # Don't clear until we've found matches
    )
    
    if selected_title:
        # Get show data and search for matches
        with st.spinner(f"Searching TMDB for {selected_title}..."):
            try:
                # Get our show data
                show_data = show_service.load_show(selected_title)
                if show_data:
                    # Make sure we have an ID
                    if 'id' not in show_data:
                        st.error(f"No ID found for show {selected_title}")
                        return
                        
                    # Store show ID for matching
                    show_data['show_id'] = show_data['id']  # Add show_id for match service
                    matching.our_show_id = show_data['id']
                    matching.our_show_title = show_data['title']
                    
                    # Get TMDB matches
                    matches = match_service.search_and_match(show_data)
                    
                    if not matches:
                        st.warning("No matches found")
                    else:
                        # Store matches in state
                        matching.matches = matches
                        matching.search_query = show_data['title']
                        matching.our_eps = []  # We'll get EPs from TMDB match
                        matching.last_validation = None  # Clear any previous validation
                        update_admin_state(state)
                        
                        # Only clear search box after successful match
                        if 'tmdb_search_bar' in st.session_state:
                            del st.session_state['tmdb_search_bar']
                else:
                    st.error(f"Could not load show data for {selected_title}")
            except Exception as e:
                st.error(f"Error loading show: {str(e)}")
    
    # Add tip about search
    if not matching.matches or matching.validated_show_id:
        st.info("💡 Search for shows to see potential matches")
    
    # Add divider before unmatched shows
    st.markdown("---")
    
    # Get and display unmatched shows
    unmatched_shows = show_service.get_unmatched_shows()
    
    # Show data is already processed by show_service.process_show_data
    
    # Filter out the show that was just validated
    if matching.validated_show_id:
        unmatched_shows = [show for show in unmatched_shows if show['id'] != matching.validated_show_id]
    
    if not unmatched_shows:
        st.info("No unmatched shows found!")
        return
        
    # Display shows in a table with buttons
    st.write("Click 'Find Matches' to search TMDB for potential matches:")
    
    # Define callback for Find Matches button
    def on_find_matches(show_data):
        try:
            with st.spinner(f"Searching TMDB for {show_data['title']}..."):
                # Get our EPs first
                team_members = show_data.get('team_members', [])
                our_eps = [member['name'] for member in team_members 
                         if member['role'] == 'Executive Producer']
                
                # Get TMDB matches
                matches = match_service.search_and_match(show_data)
                
                if not matches:
                    st.warning("No matches found")
                    return
                
                # Store matches in state
                try:
                    matching.matches = matches  # Use TMDBMatchState objects directly from match_service
                        
                    matching.search_query = show_data['title']
                    matching.our_eps = our_eps
                    matching.last_validation = None  # Clear any previous validation
                except Exception as e:
                    st.error(f"Error processing matches: {str(e)}")
                    return
                
                update_admin_state(state)
        except Exception as e:
            st.error(f"Error searching TMDB: {str(e)}")
    
    # Render table with all shows and inline matches
    render_unmatched_shows_table(unmatched_shows, on_find_matches, matching, validate_match)

    
    # No need to show validation result here anymore
    # The success message is shown directly in validate_match
    
    # Show success message if we just validated
    if matching.success_message:
        st.success(matching.success_message)
        matching.success_message = None
        update_admin_state(state)
    
    # Match Results - handled in unmatched_show_view.py

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
            ["User Management", "Announcements", "TMDB Matches", "RT Matches"],
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
        elif state.current_view == "RT Matches":
            render_rt_matches()
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
