"""
Data Entry Page

Provides forms for adding and editing show data.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from dashboard.services.data_entry import supabase

import streamlit as st
from dataclasses import asdict
from datetime import date
import time

from src.dashboard.state.session import get_page_state
from src.dashboard.state.data_entry import DataEntryState, ShowFormState
from src.dashboard.utils.style_config import COLORS, FONTS
from src.dashboard.services.data_entry import load_lookup_data

# Page title using dashboard style
st.markdown(
    f'<p style="font-family: {FONTS["primary"]["family"]}; '
    f'font-size: {FONTS["primary"]["sizes"]["header"]}px; '
    f'text-transform: uppercase; font-weight: 600; '
    f'letter-spacing: 0.1em; color: {COLORS["accent"]}; '
    f'margin-bottom: 1em;">Data Entry</p>', 
    unsafe_allow_html=True
)

try:
    # Get page state
    state = get_page_state("data_entry")
    if "form" not in state:
        state["form"] = asdict(DataEntryState())
    
    # Initialize lookups if needed
    if not state["form"]["lookups"]:
        state["form"]["lookups"] = load_lookup_data()
    
    # Operation selection
    operation = st.radio(
        "Select Operation",
        ["Add Show", "Edit Show", "Remove Show"],
        horizontal=True,
        key="data_entry_operation"
    )
    
    st.divider()
    
    # Add Show Operation
    if operation == "Add Show":
        show_tabs = st.tabs(["Show Details", "Studios", "Team Members", "Review"])
        
        # Show Details Tab
        with show_tabs[0]:
            # Check if form is loaded
            if "show_form_loaded" not in st.session_state:
                st.session_state.show_form_loaded = False
            
            if not st.session_state.show_form_loaded:
                with st.form("load_form"):
                    st.write("Click 'Start' to begin entering show details")
                    def handle_start():
                        # Initialize form state
                        st.session_state.show_form_loaded = True
                        
                        # Clear any existing form values
                        for key in ['show_network', 'show_genre', 'show_subgenres', 'show_source_type',
                                  'show_order_type', 'show_status_type', 'show_title', 'show_description']:
                            if key in st.session_state:
                                del st.session_state[key]
                    
                    st.form_submit_button("Start", on_click=handle_start)
            else:
                show_form = st.form("show_details_form")
                with show_form:
                    # Mark required fields
                    st.markdown("**Required fields are marked with a \*.**")
                    
                    # Title (required)
                    show_form.text_input(
                        "Title *",
                        key="show_title",
                        value=state["form"]["show_form"]["title"]
                    )
                
                # Network, Genre, and Order Type
                col1, col2 = show_form.columns(2)
                
                with col1:
                    # Network (required)
                    network_options = state["form"]["lookups"].get('networks', [])
                    show_form.selectbox(
                        "Network *",
                        options=network_options,
                        format_func=lambda x: x['name'],
                        key="show_network",
                        index=None
                    )
                    
                    # Genre
                    genre_options = state["form"]["lookups"].get('genres', [])
                    show_form.selectbox(
                        "Genre",
                        options=genre_options,
                        format_func=lambda x: x['name'],
                        key="show_genre",
                        index=None
                    )
                    
                    # Subgenres (multiselect)
                    subgenre_options = state["form"]["lookups"].get('subgenres', [])
                    show_form.multiselect(
                        "Subgenres",
                        options=subgenre_options,
                        format_func=lambda x: x['name'],
                        key="show_subgenres"
                    )
                
                with col2:
                    # Source Type
                    source_type_options = state["form"]["lookups"].get('source_types', [])
                    show_form.selectbox(
                        "Source Type",
                        options=source_type_options,
                        format_func=lambda x: x['name'],
                        key="show_source_type",
                        index=None
                    )
                    
                    # Order Type
                    order_type_options = state["form"]["lookups"].get('order_types', [])
                    show_form.selectbox(
                        "Order Type",
                        options=order_type_options,
                        format_func=lambda x: x['name'],
                        key="show_order_type",
                        index=None
                    )
                    
                    # Status Type
                    status_type_options = state["form"]["lookups"].get('status_types', [])
                    show_form.selectbox(
                        "Status Type",
                        options=status_type_options,
                        format_func=lambda x: x['name'],
                        key="show_status_type",
                        index=None
                    )
                    
                    # Date and Episode Count
                    show_form.date_input(
                        "Announcement Date",
                        key="show_date",
                        value=state["form"]["show_form"].get("date", date.today())
                    )
                    
                    show_form.number_input(
                        "Episode Count",
                        key="show_episode_count",
                        min_value=0,
                        value=state["form"]["show_form"].get("episode_count", 0),
                        step=1
                    )
                    
                    # Description
                    show_form.text_area(
                        "Description",
                        key="show_description",
                        value=state["form"]["show_form"].get("description", ""),
                        height=100
                    )
                
                def handle_save():
                    # Validate form
                    errors = []
                    if not st.session_state.show_title.strip():
                        errors.append("Title is required")
                    if not st.session_state.show_network:
                        errors.append("Network is required")
                    
                    # Check for genre/subgenre overlap
                    if st.session_state.show_genre and st.session_state.show_subgenres:
                        overlapping = [s for s in st.session_state.show_subgenres 
                                      if s["id"] == st.session_state.show_genre["id"]]
                        if overlapping:
                            errors.append(f"'{overlapping[0]['name']}' cannot be both a genre and subgenre")
                    
                    if errors:
                        st.error("\n".join(errors))
                    else:
                        # Update form state
                        state["form"]["show_form"]["title"] = st.session_state.show_title
                        state["form"]["show_form"]["description"] = st.session_state.show_description
                        state["form"]["show_form"]["network_id"] = st.session_state.show_network["id"] if st.session_state.show_network else None
                        state["form"]["show_form"]["genre_id"] = st.session_state.show_genre["id"] if st.session_state.show_genre else None
                        state["form"]["show_form"]["subgenres"] = [s["id"] for s in st.session_state.show_subgenres] if st.session_state.show_subgenres else []
                        state["form"]["show_form"]["source_type_id"] = st.session_state.show_source_type["id"] if st.session_state.show_source_type else None
                        state["form"]["show_form"]["order_type_id"] = st.session_state.show_order_type["id"] if st.session_state.show_order_type else None
                        state["form"]["show_form"]["status_id"] = st.session_state.show_status_type["id"] if st.session_state.show_status_type else None
                        state["form"]["show_form"]["date"] = st.session_state.show_date
                        state["form"]["show_form"]["episode_count"] = st.session_state.show_episode_count
                        # Move to Studios tab
                        state["form"]["active_tab"] = 1
                        
                        # Show success message in the form
                        show_form.success("Show details saved! Please continue to add studios.")
                
                # Add the submit button to the form
                show_form.form_submit_button("Save", on_click=handle_save)
        
        # Studios Tab
        with show_tabs[1]:
            # Add existing studios
            studio_options = state["form"]["lookups"].get('studios', [])
            selected_studios = st.multiselect(
                "Select Studios",
                options=studio_options,
                format_func=lambda x: x['name'],
                key="selected_studios",
                default=state["form"]["show_form"]["studios"]
            )
            
            # Add new studio
            st.divider()
            st.subheader("Add New Studio")
            new_studio_col1, new_studio_col2 = st.columns([3, 1])
            
            with new_studio_col1:
                new_studio = st.text_input(
                    "New Studio Name",
                    key=f"new_studio_{state['form']['num_new_studios']}"
                )
            
            with new_studio_col2:
                if st.button("Add New Studio", key="add_new_studio") and new_studio.strip():
                    if new_studio not in state["form"]["show_form"]["new_studios"]:
                        state["form"]["show_form"]["new_studios"].append(new_studio)
                        state["form"]["num_new_studios"] += 1
                        st.success(f"Added new studio: {new_studio}")
                    else:
                        st.warning("Studio already added")
            
            # Show all selected studios
            st.divider()
            st.subheader("Selected Studios")
            
            # Show existing studios
            for studio in selected_studios:
                st.text(f"• {studio['name']}")
            
            # Show new studios
            for i, studio in enumerate(state["form"]["show_form"]["new_studios"]):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"• [NEW] {studio}")
                with col2:
                    if st.button("Remove", key=f"remove_new_studio_{i}"):
                        state["form"]["show_form"]["new_studios"].remove(studio)
                        st.rerun()
                
            if not selected_studios and not state["form"]["show_form"]["new_studios"]:
                st.text("No studios selected yet")
            
            # Save button
            st.divider()
            if st.button("Save", key="save_studios"):
                # Update selected studios
                state["form"]["show_form"]["studios"] = selected_studios
                
                # Show success and guide to next step
                if not selected_studios and not state["form"]["show_form"]["new_studios"]:
                    st.error("Please add at least one studio")
                else:
                    st.success("Studios saved! Please continue to add team members.")
                    state["form"]["active_tab"] = 2  # Move to Team Members tab
        
        # Team Members Tab
        with show_tabs[2]:
            # Add team member
            st.subheader("Add Team Member")
            member_col1, member_col2, member_col3 = st.columns([2, 2, 1])
            
            with member_col1:
                member_name = st.text_input(
                    "Name",
                    key=f"member_name_{state['form']['num_team_members']}"
                )
            
            with member_col2:
                role_options = state["form"]["lookups"].get('roles', [])
                selected_roles = st.multiselect(
                    "Roles",
                    options=role_options,
                    format_func=lambda x: x['name'],
                    key=f"member_roles_{state['form']['num_team_members']}"
                )
            
            with member_col3:
                if st.button("Add Member", key="add_team_member") and member_name.strip() and selected_roles:
                    new_member = {
                        "name": member_name.strip(),
                        "role_ids": [role["id"] for role in selected_roles],
                        "roles": selected_roles  # Keep full role objects for display
                    }
                    if new_member not in state["form"]["show_form"]["team_members"]:
                        state["form"]["show_form"]["team_members"].append(new_member)
                        role_names = ", ".join(role["name"] for role in selected_roles)
                        st.success(f"Added {member_name} as {role_names}")
                        # Force fields to clear by incrementing the key counter
                        state["form"]["num_team_members"] += 1
                        st.rerun()
                    else:
                        st.warning("Team member already added")
            
            if state["form"]["show_form"]["team_members"]:
                st.divider()
                st.subheader("Team Members")
                
                for i, member in enumerate(state["form"]["show_form"]["team_members"]):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.text(member["name"])
                    with col2:
                        role_names = ", ".join(role["name"] for role in member["roles"])
                        st.text(role_names)
                    with col3:
                        if st.button("Remove", key=f"remove_member_{i}"):
                            state["form"]["show_form"]["team_members"].remove(member)
                            st.rerun()
            
            # Save button
            st.divider()
            if st.button("Save", key="save_team_members"):
                if not state["form"]["show_form"]["team_members"]:
                    st.error("Please add at least one team member")
                else:
                    st.success("Team members saved! Please continue to review.")
                    state["form"]["active_tab"] = 3  # Move to Review tab
        
        # Review Tab
        with show_tabs[3]:
            # Show Details
            st.subheader("Show Details")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f'<p style="color: {COLORS["text"]["secondary"]}; font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["small"]}px;">Title</p>', unsafe_allow_html=True)
                st.text(state["form"]["show_form"]["title"])
                
                st.markdown(f'<p style="color: {COLORS["text"]["secondary"]}; font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["small"]}px;">Network</p>', unsafe_allow_html=True)
                network = next((n["name"] for n in state["form"]["lookups"]["networks"] 
                              if n["id"] == state["form"]["show_form"]["network_id"]), "Not selected")
                st.text(network)
                
                st.markdown(f'<p style="color: {COLORS["text"]["secondary"]}; font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["small"]}px;">Genre</p>', unsafe_allow_html=True)
                genre = next((g["name"] for g in state["form"]["lookups"]["genres"] 
                            if g["id"] == state["form"]["show_form"]["genre_id"]), "Not selected")
                st.text(genre)
                
                st.markdown(f'<p style="color: {COLORS["text"]["secondary"]}; font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["small"]}px;">Subgenres</p>', unsafe_allow_html=True)
                subgenres = [g["name"] for g in state["form"]["lookups"]["subgenres"] 
                            if g["id"] in state["form"]["show_form"]["subgenres"]]
                st.text(", ".join(subgenres) if subgenres else "None")
                
                st.markdown(f'<p style="color: {COLORS["text"]["secondary"]}; font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["small"]}px;">Source Type</p>', unsafe_allow_html=True)
                source = next((s["name"] for s in state["form"]["lookups"]["source_types"] 
                             if s["id"] == state["form"]["show_form"]["source_type_id"]), "Not selected")
                st.text(source)
            
            with col2:
                st.markdown(f'<p style="color: {COLORS["text"]["secondary"]}; font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["small"]}px;">Order Type</p>', unsafe_allow_html=True)
                order = next((o["name"] for o in state["form"]["lookups"]["order_types"] 
                            if o["id"] == state["form"]["show_form"]["order_type_id"]), "Not selected")
                st.text(order)
                
                st.markdown(f'<p style="color: {COLORS["text"]["secondary"]}; font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["small"]}px;">Episode Count</p>', unsafe_allow_html=True)
                st.text(str(state["form"]["show_form"]["episode_count"]))
                
                st.markdown(f'<p style="color: {COLORS["text"]["secondary"]}; font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["small"]}px;">Announcement Date</p>', unsafe_allow_html=True)
                st.text(state["form"]["show_form"]["date"].strftime("%Y-%m-%d"))
                
                st.markdown(f'<p style="color: {COLORS["text"]["secondary"]}; font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["small"]}px;">Description</p>', unsafe_allow_html=True)
                st.text(state["form"]["show_form"]["description"] or "None")
            
            # Studios
            st.divider()
            st.subheader("Studios")
            
            # Existing studios
            for studio in state["form"]["show_form"]["studios"]:
                st.text(f"• {studio['name']}")
            
            # New studios
            for studio in state["form"]["show_form"]["new_studios"]:
                st.text(f"• [NEW] {studio}")
            
            # Team Members
            st.divider()
            st.subheader("Team Members")
            
            for member in state["form"]["show_form"]["team_members"]:
                role_names = ", ".join(role["name"] for role in member["roles"])
                st.text(f"• {member['name']} - {role_names}")
            
            # Submit button
            st.divider()
            def handle_submit():
                # First save any new studios
                studio_ids = [s["id"] for s in state["form"]["show_form"]["studios"]]
                new_studio_ids = []
                for new_studio in state["form"]["show_form"]["new_studios"]:
                    # Check if studio already exists
                    existing = supabase.table('studio_list').select('id').eq('studio', new_studio).execute()
                    if existing.data:
                        new_studio_ids.append(existing.data[0]['id'])
                    else:
                        # Add new studio as production company
                        response = supabase.table('studio_list').insert({
                            'studio': new_studio,
                            'type': 'Production Company'
                        }).execute()
                        new_studio_ids.append(response.data[0]['id'])
                
                # Prepare show data
                show_data = {
                    "title": state["form"]["show_form"]["title"],
                    "description": state["form"]["show_form"]["description"],
                    "network_id": state["form"]["show_form"]["network_id"],
                    "genre_id": state["form"]["show_form"]["genre_id"],
                    "subgenres": state["form"]["show_form"]["subgenres"],
                    "source_type_id": state["form"]["show_form"]["source_type_id"],
                    "order_type_id": state["form"]["show_form"]["order_type_id"],
                    "status_id": state["form"]["show_form"]["status_id"],
                    "episode_count": state["form"]["show_form"]["episode_count"],
                    "date": state["form"]["show_form"]["date"].isoformat(),
                    "active": True,
                    "studios": studio_ids + new_studio_ids
                }
                
                # Check if show exists
                existing = supabase.table('shows').select('id').eq('title', show_data['title']).execute()
                if existing.data:
                    st.error(f"Show with title '{show_data['title']}' already exists")
                    st.stop()
                
                # Insert show
                response = supabase.table('shows').insert(show_data).execute()
                show_id = response.data[0]['id']
                
                # Insert team members
                for member in state["form"]["show_form"]["team_members"]:
                    for role_id in member["role_ids"]:
                        supabase.table('show_team').insert({
                            "show_id": show_id,
                            "name": member["name"],
                            "role_type_id": role_id
                        }).execute()
                
                # Show success message with details
                st.success(f"✅ Show '{show_data['title']}' submitted successfully!")
                st.write(f"📊 Added {len(state['form']['show_form']['team_members'])} team members")
                if new_studio_ids:
                    st.write(f"🏢 Added {len(new_studio_ids)} new studios")
                
                # Clear form after showing success
                st.balloons()
                
                # Reset all form keys to force re-render with default values
                if 'key_counter' not in st.session_state:
                    st.session_state.key_counter = 0
                st.session_state.key_counter += 1
                
                # Clear form state
                state["form"] = asdict(DataEntryState())
                
                # Clear any leftover session state
                for key in list(st.session_state.keys()):
                    if key.startswith('show_'):
                        del st.session_state[key]
                
                # Return to landing page
                st.session_state.show_form_loaded = False
            
            with st.spinner("Submitting show..."):
                try:
                    st.button("Submit Show", key="submit_show", on_click=handle_submit)
                except Exception as e:
                    st.error(f"❌ Error submitting show: {str(e)}")
    
    # Edit Show Operation
    elif operation == "Edit Show":
        if not st.session_state.get('edit_form_loaded', False):
            with st.form('edit_landing_form'):
                st.markdown('Select a show to edit')
                shows = supabase.table('shows').select('id, title').order('title').execute()
                show_options = [{'id': s['id'], 'name': s['title']} for s in shows.data]
                
                st.selectbox(
                    'Show',
                    options=show_options,
                    format_func=lambda x: x['name'],
                    key='edit_show_select',
                    index=None
                )
                
                def handle_edit_start():
                    if st.session_state.edit_show_select:
                        st.session_state.edit_form_loaded = True
                        st.session_state.editing_show_id = st.session_state.edit_show_select['id']
                        # Load show data into state
                        show = supabase.table('shows').select('*').eq('id', st.session_state.editing_show_id).single().execute()
                        state['form']['show_form'] = show.data
                
                st.form_submit_button('Edit Show', on_click=handle_edit_start)
        else:
            show_form = st.form('edit_show_form')
            with show_form:
                # Form fields here (similar to add show but with loaded values)
                st.text_input('Title', value=state['form']['show_form'].get('title', ''), key='edit_show_title')
                
                def handle_save_edit():
                    # Update show in database
                    try:
                        show_data = {
                            'title': st.session_state.edit_show_title,
                            # Add other fields...
                        }
                        supabase.table('shows').update(show_data).eq('id', st.session_state.editing_show_id).execute()
                        show_form.success('Show updated successfully!')
                    except Exception as e:
                        show_form.error(f'Error updating show: {str(e)}')
                
                st.form_submit_button('Save Changes', on_click=handle_save_edit)
            
            # Cancel button outside form to return to landing
            if st.button('Cancel Edit'):
                st.session_state.edit_form_loaded = False
                if 'editing_show_id' in st.session_state:
                    del st.session_state.editing_show_id
    
    # Remove Show Operation
    else:
        st.info("Remove Show functionality coming soon...")

except Exception as e:
    st.error(f"Error in data entry: {str(e)}")
    st.info("Please ensure you are logged in and have proper permissions.")
