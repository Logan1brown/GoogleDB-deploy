"""Data Entry Page

Provides forms for adding and editing show data.
"""

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import time

import streamlit as st
from src.dashboard.utils.timing import time_page
from streamlit_searchbox import st_searchbox
from src.dashboard.utils.style_config import COLORS, FONTS
from src.dashboard.auth.auth_required import auth_required
from src.shared.auth import get_user_role, check_role_access

# Add custom CSS for data entry components
st.markdown("""
<style>
/* Search box styling */
div.stSearchBox > div > div {
    background-color: white;
    border-radius: 4px;
    border-color: rgb(49, 51, 63);
}

/* Form field styling */
div.stTextInput > div > div > input {
    background-color: white;
    border-radius: 4px;
}

/* Warning/error styling */
div.stAlert > div {
    padding: 1em;
    border-radius: 4px;
    margin: 1em 0;
}
</style>
""", unsafe_allow_html=True)

# Add src to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)



from src.dashboard.state.session import get_page_state, update_page_state
from src.dashboard.state.show_state import DataEntryState, ShowFormState
from src.dashboard.utils.style_config import COLORS, FONTS
from src.dashboard.services.show_service import (
    supabase,
    load_lookup_data,
    search_shows,
    load_show,
    save_show,
    remove_show
)

def render_section_header(title: str):
    """Render a section header with consistent styling."""
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">{title}</p>', unsafe_allow_html=True)

def render_landing_page(state: DataEntryState):
    """Render the landing page based on operation."""
    # Show success message if present
    if state.success_message:
        st.success(state.success_message)
        state.success_message = None  # Clear the message
        update_data_entry_state(state)
    
    if state.operation == "Add Show":
        render_section_header("Add New Show")
        selected_show = st_searchbox(
            search_function=search_shows,
            label="Search existing shows",
            placeholder="Start typing show title (3+ characters)...",
            key="add_show_search",
            clear_on_submit=True
        )
        
        if selected_show:
            # Show similar shows as warnings
            matches = search_shows(selected_show)
            if matches:
                st.warning("Similar shows found:")
                for match in matches:
                    st.write(f"- {match}")
                
        # Add New Show button at the bottom
        st.divider()
        if st.button("Add New Show", type="primary"):
            state.form_started = True
            update_data_entry_state(state)
            st.rerun()
                
    elif state.operation == "Edit Show":
        render_section_header("Edit Show")
        selected_show = st_searchbox(
            search_function=search_shows,
            label="Select show to edit",
            placeholder="Start typing show title (3+ characters)...",
            key="edit_show_search",
            clear_on_submit=True
        )
        
        if selected_show:
            # Load show data
            try:
                st.info(f"⏳ Loading show '{selected_show}'... Please wait.")
                show_data = load_show(selected_show)
                if show_data:
                    # Reset state before loading new show
                    state.form_started = True
                    state.operation = "Edit Show"  # Ensure operation is set correctly
                    state.read_only = False  # Make sure form is editable
                    state.show_form = ShowFormState()  # Reset form
                    state.show_form = ShowFormState(**show_data)  # Load new data
                    update_data_entry_state(state)
                    st.rerun()  # Need this to show the form
            except Exception as e:
                st.error(f"Error loading show: {str(e)}")
                
    else:  # Remove Show
        render_section_header("Remove Show")
        selected_show = st_searchbox(
            search_function=search_shows,
            label="Select show to remove",
            placeholder="Start typing show title (3+ characters)...",
            key="remove_show_search",
            clear_on_submit=True
        )
        
        if selected_show:
            # Load show data in read-only mode
            try:
                st.info(f"⏳ Loading show '{selected_show}'... Please wait.")
                show_data = load_show(selected_show)
                if show_data:
                    # Reset state before loading new show
                    state.form_started = True
                    state.operation = "Remove Show"  # Ensure operation is set correctly
                    state.read_only = True  # Make sure form is read-only
                    state.show_form = ShowFormState(**show_data)  # Load show data
                    update_data_entry_state(state)
                    st.rerun()  # Need this to show the form
            except Exception as e:
                st.error(f"Error loading show: {str(e)}")

def get_data_entry_state() -> DataEntryState:
    """Get the data entry state from session state"""
    state = get_page_state("data_entry")
    if not state:
        state = DataEntryState()
    return state

def update_data_entry_state(state: DataEntryState):
    """Update the data entry state in session state"""
    update_page_state("data_entry", state)

# Initialize state
state = get_data_entry_state()

# Initialize lookups if needed
if not state.lookups:
    state.lookups = load_lookup_data()
    update_data_entry_state(state)

# Page title using dashboard style
st.markdown(
    f'<p style="font-family: {FONTS["primary"]["family"]}; '
    f'font-size: {FONTS["primary"]["sizes"]["header"]}px; '
    f'text-transform: uppercase; font-weight: 600; '
    f'letter-spacing: 0.1em; color: {COLORS["accent"]}; '
    f'margin-bottom: 1em;">Data Entry</p>', 
    unsafe_allow_html=True
)

# Operation selection
new_operation = st.radio(
    "Select Operation",
    ["Add Show", "Edit Show", "Remove Show"],
    horizontal=True,
    key="operation_radio",
    index=["Add Show", "Edit Show", "Remove Show"].index(state.operation)
)

# Update operation if changed
if new_operation != state.operation:
    state.operation = new_operation
    state.form_started = False  # Reset form when operation changes
    state.read_only = (new_operation == "Remove Show")
    state.show_form = ShowFormState()  # Reset form data
    update_data_entry_state(state)
    st.rerun()

st.divider()

def render_form_field(field_name: str, field_key: str, value: Any, readonly: bool = False):
    """Render a form field with appropriate behavior"""
    if readonly:
        st.write(f"{field_name}: {value}")
    else:
        st.text_input(field_name, value=value, key=f"form_{field_key}")

def handle_show_details_save():
    """Handle show details form submission"""
    state = get_data_entry_state()
    show_form = state.show_form
    
    # Update form data from session state
    show_form.title = st.session_state.show_title_input
    show_form.network_id = st.session_state.network_dropdown[0]
    show_form.genre_id = st.session_state.genre_dropdown[0] if st.session_state.genre_dropdown else None
    show_form.status_id = st.session_state.status_dropdown[0] if st.session_state.status_dropdown else None
    show_form.subgenres = [s[0] for s in st.session_state.subgenre_dropdown] if st.session_state.subgenre_dropdown else []
    show_form.source_type_id = st.session_state.source_type_dropdown[0] if st.session_state.source_type_dropdown else None
    show_form.order_type_id = st.session_state.order_type_dropdown[0] if st.session_state.order_type_dropdown else None
    show_form.episode_count = st.session_state.episode_count_input
    show_form.date = st.session_state.date_input
    show_form.description = st.session_state.description_input
    
    # Validate required fields
    if not show_form.title:
        state.form_error = "Title is required"
        update_data_entry_state(state)
        return
        
    if not show_form.network_id:
        state.form_error = "Network is required"
        update_data_entry_state(state)
        return
    
    # Check if any subgenre matches the genre
    if show_form.genre_id and show_form.subgenres:
        genre_name = next((g['name'] for g in state.lookups.get('genres', []) if g['id'] == show_form.genre_id), None)
        subgenre_names = [s['name'] for s in state.lookups.get('subgenres', []) if s['id'] in show_form.subgenres]
        
        if genre_name in subgenre_names:
            state.form_error = f"Subgenre cannot be the same as the genre ('{genre_name}')"
            update_data_entry_state(state)
            return
    
    # Clear any previous errors and update state
    state.form_error = None
    update_data_entry_state(state)

def handle_studio_select(selected):
    """Handle studio selection"""
    state = get_data_entry_state()
    show_form = state.show_form
    
    # Replace studios list with selected IDs
    show_form.studios = [s[0] for s in selected]
    update_data_entry_state(state)

def handle_studio_save():
    """Handle adding a new studio"""
    state = get_data_entry_state()
    show_form = state.show_form
    
    # Get new studio name from input
    new_studio = st.session_state.get('new_studio_input', '')
    
    # Validate
    if not new_studio:
        state.form_error = "Studio name is required"
        update_data_entry_state(state)
        return
    
    # Check for duplicates
    if new_studio in show_form.new_studios:
        state.form_error = f"Studio '{new_studio}' already added"
        update_data_entry_state(state)
        return
    
    # Add to new studios list
    show_form.new_studios.append(new_studio)
    
    # Clear input
    if 'new_studio_input' in st.session_state:
        del st.session_state.new_studio_input
    
    update_data_entry_state(state)

def handle_existing_studio_remove(studio_id: int):
    """Handle removing an existing studio"""
    state = get_data_entry_state()
    show_form = state.show_form
    
    # Remove the studio from studios list
    if studio_id in show_form.studios:
        show_form.studios.remove(studio_id)
    
    # Update state
    update_data_entry_state(state)

def handle_studio_remove(studio_name: str):
    """Handle removing a new studio"""
    state = get_data_entry_state()
    show_form = state.show_form
    
    # Remove the studio from new_studios
    show_form.new_studios.remove(studio_name)
    
    # Update state
    update_data_entry_state(state)

def handle_studios_apply():
    """Handle applying studio changes"""
    state = get_data_entry_state()
    show_form = state.show_form
    
    # Clear any previous errors
    state.form_error = None
    
    # Nothing to do here - the studios are already saved
    # when you click Add Selected Studios or Add Studio
    update_data_entry_state(state)

def render_select(label: str, options: List[Tuple], value_id: Optional[int], key: str, required: bool = False, readonly: bool = False):
    """Render a single-select dropdown"""
    index = next((i for i, (id, _) in enumerate(options) if id == value_id), None)
    if not readonly:
        return st.selectbox(
            f"{label}{'*' if required else ''}",
            options=options,
            format_func=lambda x: x[1],
            key=key,
            index=index
        )
    return st.write(f"{label}: {next((name for id, name in options if id == value_id), '')}")

def render_multi_select(label: str, options: List[Tuple], value_ids: List[int], key: str, readonly: bool = False):
    """Render a multi-select dropdown"""
    if not readonly:
        return st.multiselect(
            label,
            options=options,
            format_func=lambda x: x[1],
            key=key,
            default=[opt for opt in options if opt[0] in (value_ids or [])]
        )
    return st.write(f"{label}: {', '.join(name for id, name in options if id in (value_ids or []))}")

def render_show_details(show_form: ShowFormState, lookups: Dict, readonly: bool = False):
    """Render show details tab"""
    st.subheader("Show Details")
    
    with st.form("show_details_form"):
        # Show error message if any
        state = get_data_entry_state()
        if state.form_error:
            st.error(state.form_error)
            state.form_error = None  # Clear after showing
            update_data_entry_state(state)
        
        # Title
        st.text_input(
            "Title*",
            value=show_form.title,
            key="show_title_input",
            disabled=readonly
        )
        
        # Create columns for the form layout
        col1, col2 = st.columns(2)
        
        # Left Column
        with col1:
            # Network (required)
            network_options = [(n['id'], n['name']) for n in lookups.get('networks', [])]
            render_select("Network", network_options, show_form.network_id, 
                         key="network_dropdown", required=True, readonly=readonly)
            
            # Genre
            genre_options = [(g['id'], g['name']) for g in lookups.get('genres', [])]
            render_select("Genre", genre_options, show_form.genre_id,
                         key="genre_dropdown", readonly=readonly)
            
            # Subgenre (multi-select)
            subgenre_options = [(s['id'], s['name']) for s in lookups.get('subgenres', [])]
            render_multi_select("Subgenre", subgenre_options, show_form.subgenres,
                              key="subgenre_dropdown", readonly=readonly)
            
            # Status
            status_options = [(s['id'], s['name']) for s in lookups.get('status_types', [])]
            render_select("Status", status_options, show_form.status_id,
                         key="status_dropdown", readonly=readonly)
            
            # Source Type
            source_type_options = [(s['id'], s['name']) for s in lookups.get('source_types', [])]
            render_select("Source Type", source_type_options, show_form.source_type_id,
                         key="source_type_dropdown", readonly=readonly)
        
        # Right Column
        with col2:
            # Order Type
            order_type_options = [(o['id'], o['name']) for o in lookups.get('order_types', [])]
            render_select("Order Type", order_type_options, show_form.order_type_id,
                         key="order_type_dropdown", readonly=readonly)
            
            # Episode Count
            if not readonly:
                st.number_input(
                    "Episode Count",
                    min_value=0,
                    value=show_form.episode_count,
                    key="episode_count_input"
                )
            else:
                st.write(f"Episode Count: {show_form.episode_count}")
            
            # Date
            if not readonly:
                st.date_input(
                    "Date",
                    value=show_form.date,
                    key="date_input"
                )
            else:
                st.write(f"Date: {show_form.date}")
            
            # Description
            if not readonly:
                st.text_area(
                    "Description",
                    value=show_form.description,
                    height=100,
                    key="description_input"
                )
            else:
                st.write(f"Description: {show_form.description}")
        
        # Submit form to update state
        if not readonly:
            st.form_submit_button(
                "Apply Changes", 
                use_container_width=True,
                on_click=handle_show_details_save,
                type="primary"
            )

def render_studios(show_form: ShowFormState, lookups: Dict, readonly: bool = False):
    """Render studios tab"""
    st.subheader("Studios")
    
    # Show error message if any
    state = get_data_entry_state()
    if state.form_error:
        st.error(state.form_error)
        state.form_error = None
        update_data_entry_state(state)
    
    # Select existing studios
    studio_options = [(s['id'], s['name']) for s in lookups.get('studios', [])]
    # Convert studio IDs to tuples for the multiselect
    default_studios = [(s['id'], s['name']) for s in lookups.get('studios', []) if s['id'] in show_form.studios]
    
    selected = st.multiselect(
        "Select Existing Studios",
        options=studio_options,
        format_func=lambda x: x[1],
        default=default_studios,
        key="studios_dropdown",
        disabled=readonly,
        on_change=lambda: handle_studio_select(st.session_state.studios_dropdown)
    )
    
    # Form 2: Add new studio
    if not readonly:
        with st.form("new_studio_form"):
            col1, col2 = st.columns([3,1])
            with col1:
                st.text_input(
                    "Add New Studio",
                    key="new_studio_input",
                    value="",
                    placeholder="Enter studio name"
                )
            with col2:
                st.write("")
                st.form_submit_button(
                    "Add Studio",
                    on_click=handle_studio_save,
                    type="primary",
                    use_container_width=True
                )
    
    # Import at top of file
    from dashboard.components.list_item import render_list_item
    
    # Show selected and new studios
    if show_form.studios or show_form.new_studios:
        st.markdown("### Selected Studios")
        
        # Show selected existing studios
        if show_form.studios:
            for studio_id in show_form.studios:
                studio = next((s for s in lookups.get('studios', []) if s['id'] == studio_id), None)
                if studio:
                    st.write(studio['name'])
        
        # Show new studios with remove buttons
        for studio in show_form.new_studios:
            col1, col2 = st.columns([10,1])
            with col1:
                st.write(f"{studio} (New)")
            with col2:
                if st.button("✕", key=f"remove_new_studio_{studio}"):
                    handle_studio_remove(studio)
    
    # Apply Changes button
    st.button("Apply Changes", on_click=handle_studios_apply, type="primary", use_container_width=True)


def handle_team_select(selected):
    """Handle team role selection"""
    state = get_data_entry_state()
    st.session_state.team_member_role_types = selected
    update_data_entry_state(state)

def handle_team_save():
    """Handle adding a new team member"""
    state = get_data_entry_state()
    show_form = state.show_form
    name = st.session_state.team_member_name
    roles = st.session_state.team_member_role_types
    
    # Validate
    if not name:
        state.form_error = "Name is required"
        update_data_entry_state(state)
        return
    
    if not roles:
        state.form_error = "At least one role is required"
        update_data_entry_state(state)
        return
    
    # Add team member
    for role in roles:
        show_form.team_members.append({
            'name': name,
            'role_type_id': role[0]  # Extract ID from tuple
        })
    
    # Clear form
    st.session_state.team_member_name = ""
    st.session_state.team_member_role_types = []
    
    # Update state
    update_data_entry_state(state)

def handle_team_remove(index: int):
    """Handle removing a team member"""
    state = get_data_entry_state()
    show_form = state.show_form
    
    # Remove team member
    show_form.team_members.pop(index)
    
    # Update state
    update_data_entry_state(state)

def handle_team_apply():
    """Handle applying team changes"""
    state = get_data_entry_state()
    update_data_entry_state(state)

def render_team(show_form: ShowFormState, lookups: Dict, readonly: bool = False):
    """Render team members tab"""
    st.subheader("Team Members")
    
    # Help text to explain the flow
    st.info("To add a team member: Enter their name, select their role(s), then click Add Member")
    
    # Show error message if any
    state = get_data_entry_state()
    if state.form_error:
        st.error(state.form_error)
        state.form_error = None
        update_data_entry_state(state)
    
    # Form for adding new team member first
    if not readonly:
        with st.form("new_team_member_form"):
            col1, col2 = st.columns([3,1])
            with col1:
                st.text_input(
                    "Add Team Member",
                    key="team_member_name",
                    value=st.session_state.get('team_member_name', ''),
                    placeholder="Enter team member name"
                )
            with col2:
                st.write("")
                st.form_submit_button(
                    "Add Member",
                    on_click=handle_team_save,
                    type="primary",
                    use_container_width=True
                )
    

    # Then select roles
    role_options = [(r['id'], r['name']) for r in lookups.get('role_types', [])]
    # Convert default IDs to tuples
    default_roles = []
    saved_roles = st.session_state.get('team_member_role_types', [])
    if saved_roles:
        role_map = {r[0]: r for r in role_options}
        default_roles = [role_map[role_id] for role_id in saved_roles if role_id in role_map]
    
    st.multiselect(
        "Select Roles",
        options=role_options,
        format_func=lambda x: x[1],
        key="team_member_role_types",
        default=default_roles,
        disabled=readonly,
        on_change=lambda: handle_team_select(st.session_state.team_member_role_types)
    )
    
    # Display existing team members
    if show_form.team_members:
        st.markdown("### Team Members")
        
        for i, member in enumerate(show_form.team_members):
            col1, col2 = st.columns([10,1])
            with col1:
                # Get role name
                role_name = next(
                    (r['name'] for r in lookups.get('role_types', [])
                     if r['id'] == member['role_type_id']),
                    'Unknown'
                )
                st.write(f"{member['name']} - {role_name}")
            
            # Remove button
            if not readonly:
                with col2:
                    if st.button("✕", key=f"remove_team_{i}"):
                        handle_team_remove(i)
    
    # Apply Changes button
    st.button(
        "Apply Changes",
        on_click=handle_team_apply,
        type="primary",
        use_container_width=True,
        key="apply_team_changes"
    )

def handle_submit(show_form: ShowFormState):
    """Handle final form submission"""
    state = get_data_entry_state()
    
    # Only validate form for add/edit operations
    if state.operation != "Remove Show":
        if not validate_show_details(show_form, state.lookups):
            return
        
    # Convert ShowFormState to dict for saving
    show_data = {
        'id': show_form.id,  # Important for edit operations
        'title': show_form.title,
        'description': show_form.description,
        'network_id': show_form.network_id,
        'genre_id': show_form.genre_id,
        'subgenres': show_form.subgenres,
        'source_type_id': show_form.source_type_id,
        'order_type_id': show_form.order_type_id,
        'status_id': show_form.status_id,
        'date': show_form.date.isoformat() if show_form.date else None,
        'episode_count': show_form.episode_count,
        'studios': show_form.studios,
        'new_studios': show_form.new_studios,
        'team_members': show_form.team_members
    }
        
    # Handle show data
    state = get_data_entry_state()
    if state.operation == "Remove Show":
        # Clear search box before removing show
        if 'remove_show_search' in st.session_state:
            del st.session_state['remove_show_search']
            
        newly_removed = remove_show(show_form.id)
        # This is a success case - the show is now inactive (either newly or already)
        state.success_message = f"Show '{show_form.title}' {'marked as inactive' if newly_removed else 'was already inactive'}"
        state.form_started = False
        state.show_form = ShowFormState()
        update_data_entry_state(state)
            
    else:
        try:
            save_show(show_data, operation=state.operation)
            state.success_message = f"Show '{show_form.title}' saved successfully!"
            state.form_started = False
            state.show_form = ShowFormState()
            
            # Clear search box based on operation
            search_key = 'add_show_search' if state.operation == 'Add Show' else 'edit_show_search'
            if search_key in st.session_state:
                del st.session_state[search_key]
                
            update_data_entry_state(state)
            
        except Exception as e:
            st.error(f"Error saving show: {str(e)}")
            return

def render_review(show_form: ShowFormState, lookups: Dict, readonly: bool = False):
    """Render review tab"""
    st.subheader("Review")
    
    # Show summary of entered data
    st.write("### Show Details")
    st.write(f"**Title:** {show_form.title}")
    
    # Get readable names for IDs
    network_name = next((n['name'] for n in lookups.get('networks', []) if n['id'] == show_form.network_id), 'None')
    genre_name = next((g['name'] for g in lookups.get('genres', []) if g['id'] == show_form.genre_id), 'None')
    subgenre_names = [s['name'] for s in lookups.get('subgenres', []) if s['id'] in show_form.subgenres]
    source_type_name = next((s['name'] for s in lookups.get('source_types', []) if s['id'] == show_form.source_type_id), 'None')
    order_type_name = next((o['name'] for o in lookups.get('order_types', []) if o['id'] == show_form.order_type_id), 'None')
    status_name = next((s['name'] for s in lookups.get('status_types', []) if s['id'] == show_form.status_id), 'None')
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Network:** {network_name}")
        st.write(f"**Genre:** {genre_name}")
        st.write(f"**Subgenres:** {', '.join(subgenre_names) or 'None'}")
        st.write(f"**Source Type:** {source_type_name}")
    
    with col2:
        st.write(f"**Order Type:** {order_type_name}")
        st.write(f"**Status:** {status_name}")
        st.write(f"**Episode Count:** {show_form.episode_count}")
        st.write(f"**Announcement Date:** {show_form.date.strftime('%B %d, %Y') if show_form.date else 'None'}")
    
    st.write("")
    st.write(f"**Description**")
    st.write(show_form.description or 'None')
    
    if show_form.studios or show_form.new_studios:
        st.write("")
        st.write("### Studios")
        
        # Show selected existing studios
        if show_form.studios:
            studio_names = [next((s['name'] for s in lookups.get('studios', []) if s['id'] == studio_id), 'Unknown') 
                          for studio_id in show_form.studios]
            for studio in studio_names:
                st.write(f"- {studio}")
        
        # Show new studios
        for studio in show_form.new_studios:
            st.write(f"- {studio} (New)")
    
    if show_form.team_members:
        st.write("")
        st.write("### Team Members")
        
        # Group members by name
        members_by_name = {}
        for member in show_form.team_members:
            name = member.get('name')
            role = next((r['name'] for r in lookups.get('role_types', []) if r['id'] == member.get('role_type_id')), 'Unknown')
            if name not in members_by_name:
                members_by_name[name] = []
            members_by_name[name].append(role)
        
        # Display each person with their roles
        for name, roles in sorted(members_by_name.items()):
            st.write(f"- **{name}** ({', '.join(roles)})")

    
    # Review tab just shows the summary, submit button is handled in main flow

def validate_show_details(show_form: ShowFormState, lookups: Dict) -> bool:
    """Validate show details tab"""
    if not show_form.title:
        st.error("Title is required")
        return False
    if not show_form.network_id:
        st.error("Network is required")
        return False
        
    # Check if any subgenre matches the genre
    if show_form.genre_id and show_form.subgenres:
        genre_name = next((g['name'] for g in lookups.get('genres', []) if g['id'] == show_form.genre_id), None)
        subgenre_names = [s['name'] for s in lookups.get('subgenres', []) if s['id'] in show_form.subgenres]
        
        if genre_name in subgenre_names:
            st.error(f"Subgenre cannot be the same as the genre ('{genre_name}')")
            return False
            
    return True

@time_page
@auth_required(['editor', 'admin'])
def main():
    # Check for editor/admin access
    if not check_role_access(['editor', 'admin']):
        st.error("Access denied. You need editor or admin privileges to use the data entry system.")
        return
        
    # Initialize state
    state = get_data_entry_state()

    if not state.form_started:
        render_landing_page(state)
    else:
        if state.operation == "Remove Show":
            # For remove operations, only show review and confirmation
            st.warning("Please review the show details below and confirm removal. This will hide the show from the active shows list.")
            render_review(state.show_form, state.lookups, state.read_only)
            
            # Show remove button
            st.divider()
            if st.button("Remove Show (Click Twice)", type="primary", use_container_width=True):
                handle_submit(state.show_form)
        else:
            # Form tabs for add/edit
            tab1, tab2, tab3, tab4 = st.tabs(["Show Details", "Studios", "Team Members", "Review"])
            
            with tab1:
                render_show_details(state.show_form, state.lookups, state.read_only)
            
            with tab2:
                render_studios(state.show_form, state.lookups, state.read_only)
            
            with tab3:
                render_team(state.show_form, state.lookups, state.read_only)
            
            with tab4:
                render_review(state.show_form, state.lookups, state.read_only)
            
            # Only show submit button in review tab
            if not state.read_only:
                st.divider()
                submit_label = {
                    "Add Show": "Add Show (Click Twice)",
                    "Edit Show": "Update Show (Click Twice)",
                    "Remove Show": "Remove Show (Click Twice)"
                }.get(state.operation, 'Submit')
                
                if st.button(submit_label, type="primary", use_container_width=True):
                    # Submit form
                    handle_submit(state.show_form)

if __name__ == "__main__":
    main()
