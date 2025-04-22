"""Add a new show to the database."""
import streamlit as st
import sys
import os
from datetime import date

# Add parent directory to path so we can import from the parent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import messaging system
from utils.messages import UserMessage, MessageType, MessageCategory

# Import menu and data functions
from data_entry_menu import menu
from data_entry_app_v3 import (
    load_lookup_data,
    save_show,
    supabase
)

# Initialize session state (this will also load lookups)
if 'lookups' not in st.session_state:
    st.session_state.lookups = {}
if 'show_data' not in st.session_state:
    st.session_state.show_data = {
        'title': '',
        'network_id': None,
        'genre_id': None,
        'subgenre_id': None,
        'source_type_id': None,
        'order_type_id': None,
        'episode_count': 0,
        'date': date.today(),
        'studio_ids': [],
        'new_studios': [],
        'team_members': []
    }
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'num_new_studios' not in st.session_state:
    st.session_state.num_new_studios = 0
if 'num_team_members' not in st.session_state:
    st.session_state.num_team_members = 0

# Debug: Show what's in lookups
st.write("Lookups loaded:")
for key, value in st.session_state.lookups.items():
    st.write(f"{key}: {len(value)} items")

# Show menu
menu()

# Check auth before showing content
if not st.session_state.authenticated:
    st.warning("Please log in to access this page")
    st.stop()

# Load lookup data
try:
    lookups = load_lookup_data()
    if not lookups:
        st.error("Failed to load required data. Please refresh the page.")
        st.stop()
    
    # Store lookups in session state
    st.session_state.lookups = lookups
        
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Page title
st.title("Add New Show")

# Tab indices
TAB_DETAILS = 0
TAB_STUDIOS = 1
TAB_TEAM = 2
TAB_REVIEW = 3

# Helper functions
def format_lookup(key: str):
    """Format lookup values for dropdowns"""
    def _format(x):
        if x is None:
            return f'Select {key.replace("_", " ").title()}...'
        return st.session_state.lookups[key].get(x, str(x))
    return _format

def validate_show_details() -> bool:
    """Validate show details form data.
    Returns:
        bool: True if validation passes, False otherwise
    """
    # Validate genre and subgenre are different
    if st.session_state.genre == st.session_state.subgenre and st.session_state.genre is not None:
        UserMessage.show(
            "Genre and Subgenre must be different",
            MessageType.ERROR,
            MessageCategory.VALIDATION
        )
        return False
    return True

def on_studios_submit():
    """Save studio selections"""
    st.session_state.show_data['studio_ids'] = st.session_state.studio_ids
    st.session_state.show_data['new_studios'] = [
        st.session_state[f'new_studio_{i}']
        for i in range(st.session_state.num_new_studios)
        if st.session_state[f'new_studio_{i}']
    ]
    st.session_state.active_tab = TAB_TEAM

def on_team_submit():
    """Save team members"""
    st.session_state.show_data['team_members'] = [
        {
            'name': st.session_state[f'team_member_{i}_name'],
            'role_ids': st.session_state[f'team_member_{i}_roles']
        }
        for i in range(st.session_state.num_team_members)
        if st.session_state.get(f'team_member_{i}_name') and st.session_state.get(f'team_member_{i}_roles')
    ]
    st.session_state.active_tab = TAB_REVIEW

# Create tabs
tabs = st.tabs(["Show Details", "Studios", "Team Members", "Review"])

# Show Details Tab
with tabs[TAB_DETAILS]:
    st.subheader("Add New Show")
    
    show_form = st.form("show_details_form")
    with show_form:
        # Title
        show_form.text_input(
            "Title",
            key="title",
            value=st.session_state.show_data['title']
        )
        
        # Date and Episode Count
        col1, col2 = show_form.columns(2)
        
        with col1:
            show_form.date_input(
                "Announcement Date",
                key="date",
                value=st.session_state.show_data['date']
            )
        
        with col2:
            show_form.number_input(
                "Episode Count",
                key="episode_count",
                min_value=0,
                value=st.session_state.show_data['episode_count']
            )
        
        # Network, Genre, and Order Type
        col1, col2 = show_form.columns(2)
        
        with col1:
            # Network
            network_options = [{'id': None, 'name': 'Select a network'}]
            network_options.extend(st.session_state.lookups.get('networks', []))
            show_form.selectbox(
                "Network",
                key="network_id",
                options=network_options,
                format_func=lambda x: x['name'],
                index=0
            )
            
            # Genre
            genre_options = [{'id': None, 'name': 'Select a genre'}]
            genre_options.extend(st.session_state.lookups.get('genres', []))
            show_form.selectbox(
                "Genre",
                key="genre_id",
                options=genre_options,
                format_func=lambda x: x['name'],
                index=0
            )
            
            # Subgenre
            subgenre_options = [{'id': None, 'name': 'Select a subgenre'}]
            subgenre_options.extend(st.session_state.lookups.get('subgenres', []))
            show_form.selectbox(
                "Subgenre",
                key="subgenre_id",
                options=subgenre_options,
                format_func=lambda x: x['name'],
                index=0
            )
        
        with col2:
            # Source Type
            source_type_options = [{'id': None, 'name': 'Select a source type'}]
            source_type_options.extend(st.session_state.lookups.get('source_types', []))
            show_form.selectbox(
                "Source Type",
                key="source_type_id",
                options=source_type_options,
                format_func=lambda x: x['name'],
                index=0
            )
            
            # Order Type
            order_type_options = [{'id': None, 'name': 'Select an order type'}]
            order_type_options.extend(st.session_state.lookups.get('order_types', []))
            show_form.selectbox(
                "Order Type",
                key="order_type_id",
                options=order_type_options,
                format_func=lambda x: x['name'],
                index=0
            )
        
        # Submit button
        submitted = show_form.form_submit_button("Continue to Studios")
        
        if submitted:
            if not validate_show_details():
                st.stop()
            
            # Update show data
            st.session_state.show_data.update({
                'title': st.session_state.title,
                'network_id': st.session_state.network,
                'genre_id': st.session_state.genre,
                'subgenre_id': st.session_state.subgenre,
                'source_type_id': st.session_state.source_type,
                'order_type_id': st.session_state.order_type,
                'episode_count': st.session_state.episode_count,
                'date': st.session_state.date
            })
            st.rerun()

# Studios Tab
with tabs[TAB_STUDIOS]:
    st.subheader("Add Studios")
    
    with st.form("studios_form"):
        # Existing studios
        st.multiselect(
            "Select Existing Studios",
            key="studio_ids",
            options=list(st.session_state.lookups['studios'].keys()),
            format_func=format_lookup('studios'),
            default=st.session_state.show_data['studio_ids']
        )
        
        # New studios
        st.number_input(
            "Number of New Studios to Add",
            key="num_new_studios",
            min_value=0,
            value=st.session_state.num_new_studios
        )
        
        # New studio inputs
        new_studios = []
        for i in range(st.session_state.num_new_studios):
            new_studio = st.text_input(f"New Studio {i+1}", key=f"new_studio_{i}")
            if new_studio:
                new_studios.append(new_studio)
        
        if st.form_submit_button("Continue to Team Members", type="primary"):
            # Update show data
            st.session_state.show_data['studio_ids'] = st.session_state.studio_ids
            st.session_state.show_data['new_studios'] = new_studios
            st.session_state.active_tab = TAB_TEAM
            st.rerun()

# Team Members Tab
with tabs[TAB_TEAM]:
    st.subheader("Add Team Members")
    
    with st.form("team_members_form"):
        # Number of team members
        st.number_input(
            "Number of Team Members to Add",
            key="num_team_members",
            min_value=0,
            value=st.session_state.num_team_members
        )
        
        # Team member inputs
        team_members = []
        for i in range(st.session_state.num_team_members):
            st.markdown(f"### Team Member {i+1}")
            
            name = st.text_input(f"Name", key=f"team_member_{i}_name")
            roles = st.multiselect(
                "Roles",
                options=list(st.session_state.lookups['roles'].keys()),
                format_func=format_lookup('roles'),
                key=f"member_roles_{i}"
            )
            
            if name and roles:
                team_members.append({
                    'name': name,
                    'role_ids': roles
                })
        
        if st.form_submit_button("Continue to Review", type="primary"):
            # Update show data
            st.session_state.show_data['team_members'] = team_members
            st.session_state.active_tab = TAB_REVIEW
            st.rerun()

# Review Tab
with tabs[TAB_REVIEW]:
    st.subheader("Review Show Details")
    
    # Show Details
    st.markdown("### 📺 Show Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Title:**", st.session_state.show_data['title'] or "*Not set*")
        st.write("**Network:**", format_lookup('networks')(st.session_state.show_data['network_id']))
        st.write("**Genre:**", format_lookup('genres')(st.session_state.show_data['genre_id']))
        st.write("**Subgenre:**", format_lookup('subgenres')(st.session_state.show_data['subgenre_id']))
    
    with col2:
        st.write("**Source Type:**", format_lookup('source_types')(st.session_state.show_data['source_type_id']))
        st.write("**Order Type:**", format_lookup('order_types')(st.session_state.show_data['order_type_id']))
        st.write("**Episode Count:**", st.session_state.show_data['episode_count'])
        st.write("**Date:**", st.session_state.show_data['date'])
    
    # Studios
    if st.session_state.show_data['studio_ids'] or st.session_state.show_data['new_studios']:
        st.markdown("### 🏢 Studios")
        
        # Existing studios
        if st.session_state.show_data['studio_ids']:
            st.write("**Selected Studios:**")
            for studio_id in st.session_state.show_data['studio_ids']:
                st.write(f"- {format_lookup('studios')(studio_id)}")
        
        # New studios
        if st.session_state.show_data['new_studios']:
            st.write("**New Studios to Add:**")
            for studio in st.session_state.show_data['new_studios']:
                st.write(f"- {studio}")
    
    # Team Members
    if st.session_state.show_data['team_members']:
        st.markdown("### 👥 Team Members")
        for i, member in enumerate(st.session_state.show_data['team_members'], 1):
            st.markdown(f"**Team Member {i}**")
            st.write(f"Name: {member['name']}")
            st.write("Roles:")
            for role_id in member['role_ids']:
                st.write(f"- {format_lookup('roles')(role_id)}")
    
    # Submit button - only shown in Review tab
    if st.button("Submit Show", type="primary"):
        try:
            # Save show to database
            save_show(st.session_state.show_data)
            st.success("Show saved successfully!")
            
            # Reset form
            st.session_state.show_data = {
                'title': '',
                'network_id': None,
                'genre_id': None,
                'subgenre_id': None,
                'source_type_id': None,
                'order_type_id': None,
                'episode_count': 0,
                'date': date.today(),
                'studio_ids': [],
                'new_studios': [],
                'team_members': []
            }
            st.session_state.num_new_studios = 0
            st.session_state.num_team_members = 0
            st.session_state.active_tab = TAB_DETAILS
            st.rerun()
        except Exception as e:
            st.error(f"Error saving show: {str(e)}")
    

