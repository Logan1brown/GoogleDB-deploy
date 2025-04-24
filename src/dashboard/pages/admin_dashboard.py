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
from ..services.supabase import get_supabase_client
from ..services.tmdb.match_service import TMDBMatchService
from ..services.tmdb.tmdb_client import TMDBClient
from ..state.admin_state import get_admin_state, update_admin_state, TMDBMatch
from src.shared.auth import auth_required
from supabase import create_client
from dataclasses import dataclass

def validate_match(match: TMDBMatch):
    """Validate a TMDB match and save the data."""
    try:
        # Input validation
        if not match.our_show_id or not isinstance(match.our_show_id, int):
            st.error("Invalid show ID")
            return
        if not match.tmdb_id or not isinstance(match.tmdb_id, int):
            st.error("Invalid TMDB ID")
            return
            
        client = get_admin_client()
        
        # Check if show exists and doesn't have TMDB ID
        show_response = client.table('shows').select('tmdb_id').eq('id', match.our_show_id).execute()
        if not show_response.data:
            st.error(f"Show {match.our_show_title} (ID: {match.our_show_id}) not found")
            return
        if show_response.data[0].get('tmdb_id'):
            st.error(f"Show {match.our_show_title} already has TMDB ID {show_response.data[0]['tmdb_id']}")
            return
            
        # Check if TMDB ID already exists in success metrics
        metrics_response = client.table('tmdb_success_metrics').select('id').eq('tmdb_id', match.tmdb_id).execute()
        if metrics_response.data:
            st.error(f"TMDB ID {match.tmdb_id} already exists in success metrics")
            return
        
        # Get full show details from TMDB
        tmdb_client = TMDBClient()
        try:
            details = tmdb_client.get_tv_show_details(match.tmdb_id)
            st.write("Raw TMDB response:")
            st.write(f"Number of seasons: {details.number_of_seasons}")
            st.write("Seasons:")
            for season in details.seasons:
                st.write(f"Season {season.season_number}:")
                st.write(f"  Episode count: {season.episode_count}")
                st.write(f"  Air date: {season.air_date}")
                st.write(f"  Overview: {season.overview}")
                st.write("  Raw season data:", vars(season))
        except Exception as e:
            st.error(f"Failed to get TMDB details: {str(e)}")
            return
            
        if not details:
            st.error(f"No TMDB details found for ID {match.tmdb_id}")
            return
        
        # Calculate metrics with validation
        episodes_per_season = []
        total_episodes = 0
        average_episodes = None
        
        try:
            # Get episode counts from each season
            for season in details.seasons:
                if season.season_number > 0:  # Skip season 0 (specials)
                    if season.episode_count:
                        episodes_per_season.append(season.episode_count)
                        st.write(f"Season {season.season_number}: {season.episode_count} episodes")
            
            # Only calculate totals if we have episode counts
            if episodes_per_season:
                total_episodes = sum(episodes_per_season)
                average_episodes = round(total_episodes / len(episodes_per_season), 2)
                st.write(f"Found {len(episodes_per_season)} seasons with episode counts: {episodes_per_season}")
            else:
                st.warning("No episode counts found in TMDB data")
        except (AttributeError, TypeError, ZeroDivisionError) as e:
            st.error(f"Error calculating episode metrics: {str(e)}")
            return
            
        # Validate required fields
        if not details.status:
            st.error("Missing required field: status")
            return
        
        # Start transaction
        # Update show's tmdb_id
        update_response = client.table('shows')\
            .update({"tmdb_id": match.tmdb_id})\
            .eq('id', match.our_show_id)\
            .execute()
            
        if not update_response.data or len(update_response.data) == 0:
            st.error(f"Failed to update show {match.our_show_title} with TMDB ID")
            return
        
        # Prepare and validate metrics data
        metrics_data = {
            "tmdb_id": match.tmdb_id,
            "seasons": details.number_of_seasons,
            "episodes_per_season": episodes_per_season,
            "total_episodes": total_episodes or None,
            "average_episodes": average_episodes,
            "status": details.status,
            "last_air_date": details.last_air_date.isoformat() if details.last_air_date else None
        }
        
        # Validate all metrics data fields
        if not all(key in metrics_data for key in ['tmdb_id', 'seasons', 'episodes_per_season', 'status']):
            client.table('shows').update({"tmdb_id": None}).eq('id', match.our_show_id).execute()
            st.error("Missing required metrics fields")
            return
        
        # Insert into tmdb_success_metrics
        metrics_response = client.table('tmdb_success_metrics')\
            .insert(metrics_data)\
            .execute()
            
        if not metrics_response.data or len(metrics_response.data) == 0:
            # Rollback show update
            client.table('shows').update({"tmdb_id": None}).eq('id', match.our_show_id).execute()
            st.error("Failed to insert TMDB metrics")
            return
            
        st.success(f"Successfully validated match for {match.our_show_title} (ID: {match.our_show_id})")
        
    except Exception as e:
        st.error(f"Error validating match: {str(e)}")
        # Attempt rollback
        try:
            client.table('shows').update({"tmdb_id": None}).eq('id', match.our_show_id).execute()
        except Exception as rollback_error:
            st.error(f"Failed to rollback changes: {str(rollback_error)}")
            st.error("Manual database check required")


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
    match_service = TMDBMatchService()
    
    # API Status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("API Calls", f"{state.api_calls_total} total", f"{state.api_calls_remaining} remaining")
    with col2:
        cache_ratio = (state.cache_hits / (state.cache_hits + state.cache_misses)) * 100 if state.cache_hits + state.cache_misses > 0 else 0
        st.metric("Cache", f"{state.cache_hits} hits", f"{cache_ratio:.1f}% hit rate")
    with col3:
        reset_in = max(0, state.api_window_reset_time - time.time())
        st.metric("Rate Limit", f"{state.api_calls_remaining}/40", f"Reset in {reset_in:.1f}s")
    
    # Search TMDB
    st.text_input("Search Shows", 
                 value=state.tmdb_search_query,
                 placeholder="Enter show title...",
                 key="tmdb_search_bar")
    if not state.tmdb_matches:
        st.info("Search for shows to see potential matches")
    
    # Unmatched Shows section
    st.subheader("Unmatched Shows")
    
    # Get unmatched shows from our view
    supabase = get_supabase_client()
    response = supabase.table('api_tmdb_match') \
        .select('show_id, title, network_name, date, team_members') \
        .execute()
    
    unmatched_shows = response.data
    
    if not unmatched_shows:
        st.info("No unmatched shows found!")
        return
        
    # Display shows in a table with buttons
    st.write("Click 'Find Matches' to search TMDB for potential matches:")
    
    # Create containers for each show's info and potential matches
    for show in unmatched_shows:
        show_container = st.container()
        with show_container:
            # Display show info in columns
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.write(show['title'])
            with col2:
                st.write(show.get('network_name', ''))
            with col3:
                st.write(show.get('date', ''))
            with col4:
                if st.button("Find Matches", key=f"find_{show['show_id']}"):
                    try:
                        with st.spinner(f"Searching TMDB for {show['title']}..."):
                            # Get our EPs first
                            team_members = show.get('team_members', [])
                            our_eps = [member['name'] for member in team_members 
                                     if member['role'].lower() == 'executive producer']
                            
                            # Get TMDB matches
                            matches = match_service.search_and_match(show['title'])
                            
                            if not matches:
                                st.error("No matches found")
                                continue
                            
                            # Store matches and our_eps in state
                            state.tmdb_matches = matches
                            state.tmdb_search_query = show['title']
                            state.our_eps = our_eps
                            update_admin_state(state)
                            
                            # Update metrics
                            state.api_calls_total += 1
                            state.api_calls_remaining -= 1
                            update_admin_state(state)
                    except Exception as e:
                        st.error(f"Error searching TMDB: {str(e)}")
                        continue
    
    # Match Results
    if state.tmdb_matches:
        st.subheader(f"Matches for '{state.tmdb_search_query}'")
        
        for match in state.tmdb_matches:
            with st.expander(f"{match.name}", expanded=True):
                col1, col2 = st.columns(2)
                
                # Our Show Details
                with col1:
                    st.markdown("**Our Show Data**")
                    st.markdown(f"**Title:** {match.our_show_title}")
                    st.markdown(f"**Network:** {match.our_network or 'Unknown'}")
                    st.markdown(f"**Year:** {match.our_year or 'Unknown'}")
                    st.markdown("**Executive Producers:**")
                    if state.our_eps:
                        for ep in state.our_eps:
                            matched = ep in match.executive_producers
                            st.write("🟢" if matched else "⚫", ep)
                    else:
                        st.markdown("*No executive producers listed*")

                # TMDB Details
                with col2:
                    st.markdown("**TMDB Data**")
                    st.markdown(f"**Title:** {match.name}")
                    st.markdown(f"**Network:** {', '.join(match.networks) if match.networks else 'Unknown'}")
                    st.markdown(f"**First Air:** {match.first_air_date or 'Unknown'}")
                    st.markdown("**Executive Producers:**")
                    if match.executive_producers:
                        for ep in match.executive_producers:
                            matched = ep in state.our_eps
                            st.write("🟢" if matched else "⚫", ep)
                    else:
                        st.markdown("*No executive producers found*")
                
                # Match Score Details
                st.markdown("**Match Details**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("Title Match")
                    st.markdown(f"{match.title_score}%")
                with col2:
                    st.markdown("Network Match")
                    st.markdown(f"{match.network_score}%")
                with col3:
                    st.markdown("EP Match")
                    st.markdown(f"{match.ep_score}%")

                # Validate Match Button - centered and prominent
                st.markdown("---")
                col1, col2, col3 = st.columns([1,3,1])
                with col2:
                    if st.button(f"Validate Match ({match.confidence}%)", 
                               key=f"validate_{match.our_show_id}_{match.tmdb_id}",
                               type="primary",
                               use_container_width=True):
                        validate_match(match)

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
