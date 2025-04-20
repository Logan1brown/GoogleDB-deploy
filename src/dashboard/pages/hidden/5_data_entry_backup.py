"""
Data Entry Page

Provides forms for adding and editing show data.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import date, datetime
import difflib
import time

# Add src to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

import streamlit as st
from streamlit_searchbox import st_searchbox
from src.dashboard.services.data_entry import supabase, load_lookup_data, search_shows, save_show, load_show
from src.dashboard.utils.style_config import COLORS, FONTS

@st.cache_data(ttl=3600)
def load_lookup_data() -> Dict[str, List[Dict]]:
    """Load all lookup data from Supabase with caching"""
    lookups = {}
    
    # Load networks
    response = supabase.table('network_list').select('id, network').execute()
    lookups['networks'] = [{'id': n['id'], 'name': n['network']} for n in response.data]
    
    # Load studios
    response = supabase.table('studio_list').select('id, studio, type').execute()
    lookups['studios'] = [{'id': s['id'], 'name': s['studio'], 'type': s['type']} for s in response.data]
    
    # Load genres
    response = supabase.table('genre_list').select('id, genre').execute()
    lookups['genres'] = [{'id': g['id'], 'name': g['genre']} for g in response.data]
    
    # Load subgenres
    response = supabase.table('subgenre_list').select('id, subgenre').execute()
    lookups['subgenres'] = [{'id': s['id'], 'name': s['subgenre']} for s in response.data]
    
    # Load roles
    response = supabase.table('role_types').select('id, role').execute()
    lookups['roles'] = [{'id': r['id'], 'name': r['role']} for r in response.data]
    
    # Load source types
    response = supabase.table('source_types').select('id, type').execute()
    lookups['source_types'] = [{'id': s['id'], 'name': s['type']} for s in response.data]
    
    # Load order types
    response = supabase.table('order_types').select('id, type').execute()
    lookups['order_types'] = [{'id': o['id'], 'name': o['type']} for o in response.data]
    
    return lookups

def reset_show_data():
    """Reset show data to initial state"""
    return {
        'title': '',
        'network_id': None,
        'genre_id': None,
        'subgenre_id': None,
        'source_type_id': None,
        'order_type_id': None,
        'episode_count': 0,
        'date': date.today().strftime('%Y-%m-%d'),
        'description': '',
        'studio_ids': [],
        'new_studios': [],
        'team': []  # Initialize empty team list
    }

# Initialize session state
if 'lookups' not in st.session_state:
    st.session_state.lookups = load_lookup_data()
if 'show_data' not in st.session_state:
    st.session_state.show_data = reset_show_data()
if 'search_text' not in st.session_state:
    st.session_state.search_text = ''
if 'form_title' not in st.session_state:
    st.session_state.form_title = ''
if 'operation' not in st.session_state:
    st.session_state.operation = "Add show"

# No need for explicit form clearing - handled in callbacks

# Landing page
st.title("TV shows data entry")

# Operation selection at the top
st.radio(
    "Select Operation",
    ["Add show", "Edit show", "Remove show"],
    horizontal=True,
    key="operation",
    index=0
)

# Divider
st.markdown("---")

# Search section - show for both Add and Edit
if st.session_state.operation in ["Add show", "Edit show"]:
    st.markdown("### 🔍 Search Shows")
    if st.session_state.operation == "Add show":
        st.markdown("Search before adding a new show to avoid duplicates.")
    else:
        st.markdown("Search for the show you want to edit.")

    # Different UI for Add vs Edit
    if st.session_state.operation == "Add show":
        # Search box with clear visual separation
        st.markdown("<div style='margin-bottom: 1em;'></div>", unsafe_allow_html=True)
        selected = st_searchbox(
            search_shows,
            placeholder="Start typing show title (3+ characters)...",
            label="Show Title",
            key="show_search",
            clear_on_submit=True
        )

        # Status and actions based on search
        current_text = selected if selected else ''
        if current_text and len(current_text.strip()) >= 3:
            matches = search_shows(current_text)
            if matches:
                st.markdown("### Similar Shows Found")
                for match in matches:
                    st.markdown(f"• {match}")
                st.warning("Please check if your show already exists above.")
            else:
                st.success("No similar shows found")
        elif current_text:
            st.info("ℹ️ Please enter at least 3 characters to search.")
    else:  # Edit Show
        # Get all active shows
        response = supabase.table('shows').select('title').eq('active', True).order('title').execute()
        show_titles = [show['title'] for show in response.data]
        
        # Show dropdown to select show
        selected = st.selectbox(
            "Select Show to Edit",
            options=["Select a show..."] + show_titles,
            key="edit_show_select"
        )
        
        if selected and selected != "Select a show...":
            try:
                # Load show data
                show_data = load_show(selected, st.session_state.lookups)
                st.session_state.show_data = show_data
                st.session_state.form_title = show_data['title']
                st.session_state.show_form_started = True
            except Exception as e:
                st.error(f"Error loading show: {str(e)}")
    
    # For Add Show, always show the Add as New button
    if st.session_state.operation == "Add show":
        st.markdown("")
        if st.button("Add new show", type="primary", use_container_width=True):
            st.session_state.show_data = reset_show_data()
            st.session_state.form_title = current_text.strip() if current_text else ""
            st.session_state.show_form_started = True

# For Remove Show operation
if st.session_state.operation == "Remove show":
    st.markdown("### Remove show")
    st.markdown("Search for the show you want to remove.")
    
    selected = st_searchbox(
        search_shows,
        placeholder="Start typing show title (3+ characters)...",
        label="Show Title",
        key="remove_search",
        clear_on_submit=True
    )
    
    if selected:
        st.warning(f"Are you sure you want to remove '{selected}'?")
        if st.button("Confirm Remove", type="primary"):
            # TODO: Implement show removal
            st.error("Remove functionality not yet implemented")
            st.session_state.operation = "Add show"

# Create tabs for form sections
tabs = st.tabs(["Show Details", "Studios", "Team Members", "Review"])

# Show Details Tab
with tabs[0]:
    if st.session_state.get('show_form_started', False):
        # Show appropriate header based on operation
        if st.session_state.operation == "Add show":
            st.markdown("### Add new show")
        else:
            st.markdown(f"### Edit show: {st.session_state.form_title}")
        st.subheader("Show Details")
        
        with st.form("show_details_form"):
            # Title from search
            title = st.text_input(
                "Title",
                value=st.session_state.form_title,  # Use the title from search
                key="show_title_input"
            )
            
            # Create columns for the form layout
            col1, col2 = st.columns(2)
            
            # Left Column
            with col1:
                # Network
                network_options = [(n['id'], n['name']) for n in st.session_state.lookups['networks']]
                network_index = next((i for i, (id, _) in enumerate(network_options) 
                                    if id == st.session_state.show_data.get('network_id')), None)
                network = st.selectbox(
                    "Network",
                    options=network_options,
                    format_func=lambda x: x[1],
                    key="network_dropdown",
                    index=network_index
                )
                
                # Genre
                genre_options = [(g['id'], g['name']) for g in st.session_state.lookups['genres']]
                genre_index = next((i for i, (id, _) in enumerate(genre_options) 
                                  if id == st.session_state.show_data.get('genre_id')), None)
                genre = st.selectbox(
                    "Genre",
                    options=genre_options,
                    format_func=lambda x: x[1],
                    key="genre_dropdown",
                    index=genre_index
                )
                
                # Source Type
                source_type_options = [(s['id'], s['name']) for s in st.session_state.lookups['source_types']]
                source_type_index = next((i for i, (id, _) in enumerate(source_type_options) 
                                        if id == st.session_state.show_data.get('source_type_id')), None)
                source_type = st.selectbox(
                    "Source Type",
                    options=source_type_options,
                    format_func=lambda x: x[1],
                    key="source_type_dropdown",
                    index=source_type_index
                )
                
                # Subgenre
                subgenre_options = [(s['id'], s['name']) for s in st.session_state.lookups['subgenres']]
                subgenre_index = next((i for i, (id, _) in enumerate(subgenre_options) 
                                     if id == st.session_state.show_data.get('subgenre_id')), None)
                subgenre = st.selectbox(
                    "Subgenre",
                    options=subgenre_options,
                    format_func=lambda x: x[1],
                    key="subgenre_dropdown",
                    index=subgenre_index
                )
            
            # Right Column
            with col2:
                # Order Type
                order_type_options = [(o['id'], o['name']) for o in st.session_state.lookups['order_types']]
                order_type_index = next((i for i, (id, _) in enumerate(order_type_options) 
                                       if id == st.session_state.show_data.get('order_type_id')), None)
                order_type = st.selectbox(
                    "Order Type",
                    options=order_type_options,
                    format_func=lambda x: x[1],
                    key="order_type_dropdown",
                    index=order_type_index
                )
                
                # Episode Count
                episode_count = st.number_input(
                    "Episode Count",
                    min_value=0,
                    value=st.session_state.show_data.get('episode_count', 0),
                    key="episode_count_input"
                )
                
                # Date
                air_date = st.date_input(
                    "Date",
                    value=datetime.strptime(st.session_state.show_data['date'], '%Y-%m-%d').date() if st.session_state.show_data.get('date') else date.today(),
                    key="date_input"
                )
                
                # Description
                description = st.text_area(
                    "Description",
                    value=st.session_state.show_data.get('description', ''),
                    height=100,
                    key="description_input"
                )
                
            # Form buttons
            if st.form_submit_button("Save show details", type="primary", use_container_width=True):
                # Create new dict with all existing data
                updated_data = st.session_state.show_data.copy()
                # Preserve team data
                team_data = updated_data.get('team', [])
                # Update only the form fields
                updated_data.update({
                    'title': st.session_state.show_title_input,
                    'network_id': st.session_state.network_dropdown[0] if st.session_state.network_dropdown else None,
                    'genre_id': st.session_state.genre_dropdown[0] if st.session_state.genre_dropdown else None,
                    'subgenre_id': st.session_state.subgenre_dropdown[0] if st.session_state.subgenre_dropdown else None,
                    'source_type_id': st.session_state.source_type_dropdown[0] if st.session_state.source_type_dropdown else None,
                    'order_type_id': st.session_state.order_type_dropdown[0] if st.session_state.order_type_dropdown else None,
                    'episode_count': st.session_state.episode_count_input,
                    'date': st.session_state.date_input.strftime('%Y-%m-%d') if st.session_state.date_input else None,
                    'description': st.session_state.description_input
                })
                # Restore team data
                updated_data['team'] = team_data
                # Replace show data with updated version
                st.session_state.show_data = updated_data
                st.success("Show details saved")
            
    # Studios Tab
    with tabs[1]:
        if st.session_state.get('show_form_started', False):
            st.subheader("Add Studios")
            
            with st.form("studios_form"):
                # Studio selection
                selected_studios = st.multiselect(
                    "Select Studios",
                    [(s['id'], s['name']) for s in st.session_state.lookups['studios']],
                    default=st.session_state.show_data.get('studio_ids', []),
                    format_func=lambda x: x[1] if isinstance(x, tuple) else x
                )
                
                # New studio input
                new_studio = st.text_input(
                    "Add New Studio",
                    key="new_studio"
                )
                
                # Add button for new studio
                if st.form_submit_button("Add New Studio"):
                    if new_studio.strip():
                        new_studios = st.session_state.show_data.get('new_studios', [])
                        if new_studio not in new_studios:
                            new_studios.append(new_studio)
                            st.session_state.show_data['new_studios'] = new_studios
                            st.session_state.new_studio = ""  # Clear input
                    else:
                        st.error("Please enter a studio name")
                
                # Show new studios
                new_studios = st.session_state.show_data.get('new_studios', [])
                if new_studios:
                    st.write("New Studios to be added:")
                    for studio in new_studios:
                        st.write(f"- {studio}")
                
                # Save button
                if st.form_submit_button("Save Studios", type="primary", use_container_width=True):
                    # Update show data with selected studios
                    st.session_state.show_data['studio_ids'] = selected_studios

    # Team Members Tab
    with tabs[2]:
        if st.session_state.get('show_form_started', False):
            st.subheader("Add Team Members")
            
            # Show current team members
            if st.session_state.show_data.get('team'):
                st.markdown("### Current Team Members")
                for member in st.session_state.show_data['team']:
                    # Look up role names from role IDs
                    role_names = ", ".join(
                        next(r['name'] for r in st.session_state.lookups['roles'] if r['id'] == role_id)
                        for role_id in member['roles']
                    )
                    st.write(f"- **{member['name']}** ({role_names})")
        
            # Add some space
            st.write("")
            
            # Team member selection dropdown (outside form)
            st.markdown("### Edit Team Member")
            if st.session_state.show_data.get('team'):
                st.write("Select an existing team member to edit or choose 'Add new member' to add someone new:")
                team_members = [m['name'] for m in st.session_state.show_data['team']]
                selected_member = st.selectbox(
                    "Team Member",
                    options=["Add new member"] + team_members,
                    key="selected_team_member"
                )
            else:
                st.write("Add your first team member:")
                selected_member = "Add new member"
            
            with st.form("team_form"):
                # Create two columns for name and roles
                name_col, roles_col = st.columns(2)
            
            with name_col:
                # Team member name
                default_name = "" if selected_member == "Add new member" else selected_member
                name = st.text_input("Team Member Name", value=default_name)
            
            with roles_col:
                # Role selection
                if selected_member != "Add new member":
                    # Find the member's data
                    member = next((m for m in st.session_state.show_data['team'] 
                                 if m['name'] == selected_member), None)
                    if member:
                        current_role_ids = member['roles']  # Already just IDs
                        default_roles = [r for r in st.session_state.lookups['roles'] 
                                       if r['id'] in current_role_ids]
                    else:
                        default_roles = []
                else:
                    default_roles = []

                # Create a unique key for each member's roles
                role_key = f"team_roles_{selected_member}" if selected_member else "team_roles_new"
                selected_roles = st.multiselect(
                    "Select Roles",
                    options=st.session_state.lookups['roles'],
                    format_func=lambda x: x['name'],
                    default=default_roles,
                    key=role_key
                )
            
            # Form buttons
            if st.form_submit_button("Save team member", type="primary", use_container_width=True):
                if name and selected_roles:
                    name = name.strip()
                    # Create new dict with all existing data
                    updated_data = st.session_state.show_data.copy()
                    
                    # Initialize team list if it doesn't exist
                    if 'team' not in updated_data:
                        updated_data['team'] = []
                    
                    # Remove existing entry for this name if it exists
                    updated_data['team'] = [m for m in updated_data['team'] 
                                           if m['name'].lower() != name.lower()]
                    
                    # Add/update member
                    team_member = {
                        'name': name,
                        'roles': [r['id'] for r in selected_roles]  # Only store role IDs
                    }
                    updated_data['team'].append(team_member)
                    
                    # Replace show data with updated version
                    st.session_state.show_data = updated_data
                    
                    # Show success message
                    role_list = ', '.join(r['name'] for r in selected_roles)
                    action = "Updated" if selected_member != "Add new member" else "Added"
                    st.success(f"{action} {name} with roles: {role_list}")


    # Review Tab
    with tabs[3]:
        st.subheader("Review Show Information")
        
        # Show Details
        st.markdown("### Show Details")
        st.write(f"Title: {st.session_state.show_data.get('title', 'Not set')}")

        
        # Find network name
        network_id = st.session_state.show_data.get('network_id')
        network_name = next((n['name'] for n in st.session_state.lookups.get('networks', []) 
                            if n['id'] == network_id), 'Not selected')
        st.write(f"Network: {network_name}")
        
        # Find genre name
        genre_id = st.session_state.show_data.get('genre_id')
        genre_name = next((g['name'] for g in st.session_state.lookups.get('genres', []) 
                          if g['id'] == genre_id), 'Not selected')
        st.write(f"Genre: {genre_name}")
        
        # Find subgenre name
        subgenre_id = st.session_state.show_data.get('subgenre_id')
        subgenre_name = next((s['name'] for s in st.session_state.lookups.get('subgenres', []) 
                             if s['id'] == subgenre_id), 'Not selected')
        st.write(f"Subgenre: {subgenre_name}")
        
        # Find source type name
        source_type_id = st.session_state.show_data.get('source_type_id')
        source_type_name = next((s['name'] for s in st.session_state.lookups.get('source_types', []) 
                                if s['id'] == source_type_id), 'Not selected')
        st.write(f"Source Type: {source_type_name}")
        
        st.write(f"Episode Count: {st.session_state.show_data.get('episode_count', 0)}")
        st.write(f"Date: {st.session_state.show_data.get('date', 'Not set')}")
        st.write(f"Description: {st.session_state.show_data.get('description', 'Not set')}")

        # Studios
        st.markdown("### Studios")
        selected_studios = [s[0] for s in st.session_state.show_data.get('studio_ids', [])]
        if selected_studios:
            for studio_id in selected_studios:
                studio_name = next((s['name'] for s in st.session_state.lookups['studios'] if s['id'] == studio_id), 'Unknown')
                st.write(f"- {studio_name}")
        else:
            st.write("No studios selected")

        new_studios = st.session_state.show_data.get('new_studios', [])
        if new_studios:
            st.markdown("### New Studios")
            for studio in new_studios:
                st.write(f"- {studio}")
        
        # Team Members
        st.markdown("### Team members")
        team = st.session_state.show_data.get('team', [])
        if team:
            for member in team:
                # Look up role names from role IDs
                role_names = ", ".join(
                    next(r['name'] for r in st.session_state.lookups['roles'] if r['id'] == role_id)
                    for role_id in member['roles']
                )
                st.write(f"- **{member['name']}** ({role_names})")
        else:
            st.write("*No team members added*")

        with st.form("review_form"):
            st.write("Click 'Submit Show' to save all changes to the database.")
            # Form buttons
            # Debug expander above submit button
            with st.expander("🔍 Debug Info"):
                st.write("Current Form State")
                st.write(f"Operation: {st.session_state.operation}\n")
                
                # Show Data Keys
                st.write("Show Data Keys:\n")
                st.write(f"new_studios: {st.session_state.show_data.get('new_studios')}\n")
                
                # Team Members
                st.write("Team Members")
                team = st.session_state.show_data.get('team', [])
                if team:
                    st.write("Full team data:")
                    st.json(team)
                    for member in team:
                        st.write(f"{member['name']} roles:")
                        st.json(member['roles'])
                        st.write(f"Role types: {[type(r) for r in member['roles']]}")
                        st.write(f"Role values: {member['roles']}")
                
                # Form Info
                st.write("Form Info")
                st.write(f"Form title: {st.session_state.form_title}")
            
            def handle_submit():
                try:
                    # Save show data
                    show_id = save_show(st.session_state.show_data, st.session_state.operation)
                    
                    # Show success message inside form context
                    show_details_form.success(f"Show {st.session_state.operation.lower().replace(' ', '')}ed successfully! ID: {show_id}")
                    
                    # Only clear form for Add show
                    if st.session_state.operation == "Add show":
                        st.session_state.show_form_started = False
                        if 'show_data' in st.session_state:
                            del st.session_state['show_data']
                except Exception as e:
                    # Show error inside form context
                    show_details_form.error(f"Error saving show: {str(e)}")
            
        # Submit button with spinner
        with st.spinner("Submitting show..."):
            try:
                st.button("Submit Show", key="submit_show", on_click=handle_submit, type="primary", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error submitting show: {str(e)}")

        # Studios
        st.markdown("### Studios")
        selected_studios = [s[0] for s in st.session_state.show_data.get('studio_ids', [])]
        if selected_studios:
            for studio_id in selected_studios:
                studio_name = next((s['name'] for s in st.session_state.lookups['studios'] if s['id'] == studio_id), 'Unknown')
                st.write(f"- {studio_name}")
        else:
            st.write("No studios selected")

        new_studios = st.session_state.show_data.get('new_studios', [])
        if new_studios:
            st.markdown("### New Studios")
            for studio in new_studios:
                st.write(f"- {studio}")
    
        # Team Members
        st.markdown("### Team members")
        team = st.session_state.show_data.get('team', [])
        if team:
            for member in team:
                # Look up role names from role IDs
                role_names = ", ".join(
                    next(r['name'] for r in st.session_state.lookups['roles'] if r['id'] == role_id)
                    for role_id in member['roles']
                )
                st.write(f"- **{member['name']}** ({role_names})")
        else:
            st.write("*No team members added*")
