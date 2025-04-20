"""Admin Dashboard Page

This page is only visible to admin users and provides access to administrative functions.
"""

import streamlit as st
import sys
import os
import secrets
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.shared.auth import auth_required
from src.shared.auth import get_user_role
from src.dashboard.utils.style_config import COLORS, FONTS
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
    """Update a user's role."""
    try:
        # Get admin client
        client = get_admin_client()
        
        # Update role in user_roles table
        result = client.table('user_roles').upsert({
            'id': user_id,
            'role': new_role,
            'updated_by': st.session_state.user.id
        }).execute()
        
        # Clean up session state
        for key in list(st.session_state.keys()):
            if key.startswith(('role_', 'reason_', 'update_')):
                del st.session_state[key]
        
        return True
        
    except Exception as e:
        st.error(f"Failed to update role: {str(e)}")
        return False

def invite_user(email: str, role: str) -> bool:
    """Invite a new user with specified role."""
    try:
        # Get admin client
        client = get_admin_client()
        
        # Sign up user with email confirmation
        temp_password = 'temp' + secrets.token_urlsafe(16)
        response = client.auth.sign_up({
            'email': email,
            'password': temp_password,
            'options': {
                'data': {'initial_role': role}
            }
        })
        user = response.user
        
        # Set initial role
        client.table('user_roles').insert({
            'id': user.id,
            'role': role,
            'created_by': st.session_state.user.id,
            'updated_by': st.session_state.user.id
        }).execute()
        
        return True
        
    except Exception as e:
        st.error(f"Failed to invite user: {str(e)}")
        return False

@auth_required(['admin'])
def admin_show():
    """Main function for admin dashboard."""
    try:
        st.title("Admin Dashboard")
        
        # Test admin client access
        client = get_admin_client()
        client.auth.admin.list_users()
        
        # Get all users
        users = get_users()
        
        # Create a table to display users
        st.markdown("### User Management")
        
        # Display each user in an expander
        for user in users:
            # Show revoked status in title if applicable
            title = f"{user['email']} - {'🚫 REVOKED' if user['role'] == 'revoked' else user['role']}"
            
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
                            ["admin", "editor", "viewer", "revoked"],
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
                            elif new_role == 'revoked' and not reason:
                                st.error("Please provide a reason for revoking access")
                            elif update_user_role(user['id'], new_role, reason):
                                st.success("Role updated successfully")
                                st.rerun()
                            else:
                                st.error("Failed to update role")
    
        # Invite new user section
        st.markdown("### Invite New User")
        with st.form("invite_user_form"):
            email = st.text_input("Email")
            role = st.selectbox(
                "Initial Role",
                options=['viewer', 'editor', 'admin'],
                index=0
            )
            submitted = st.form_submit_button("Send Invitation")
            
            if submitted:
                if not email:
                    st.error("Please enter an email address")
                    return
                
                if invite_user(email, role):
                    st.success("Invitation sent successfully")
                    st.rerun()
    except Exception as e:
        st.error(f"Admin dashboard error: {str(e)}")

if __name__ == "__main__":
    admin_show()
