import os
from datetime import date
from typing import Dict, List, Tuple
import difflib

import streamlit as st
from streamlit_searchbox import st_searchbox
from supabase import create_client

# Initialize Supabase client
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')
)

# Cache lookup data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_lookup_data() -> Dict[str, List[Dict]]:
    """Load lookup data for dropdowns"""
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
    
    # Load source types
    response = supabase.table('source_types').select('id, type').execute()
    lookups['source_types'] = [{'id': s['id'], 'name': s['type']} for s in response.data]
    
    # Load order types
    response = supabase.table('order_types').select('id, type').execute()
    lookups['order_types'] = [{'id': o['id'], 'name': o['type']} for o in response.data]
    
    # Load status types
    response = supabase.table('status_types').select('id, status').execute()
    lookups['statuses'] = [{'id': s['id'], 'name': s['status']} for s in response.data]
    
    # Load roles
    response = supabase.table('role_types').select('id, role').execute()
    lookups['roles'] = [{'id': r['id'], 'name': r['role']} for r in response.data]
    
    return lookups

# Style constants
COLORS = {
    'text': {
        'primary': 'rgb(49, 51, 63)',
        'secondary': 'rgb(120, 120, 120)'
    },
    'background': '#FFFFFF',
    'accent': 'rgb(55, 83, 109)'
}

# Page config
st.set_page_config(
    page_title="TV Shows Data Entry",
    page_icon="📺",
    layout="wide"
)

# Set page styles
st.markdown("""
<style>
    /* Form labels */
    .stTextInput > label,
    .stTextArea > label,
    .stSelectbox > label,
    .stMultiSelect > label,
    .stDateInput > label,
    .stNumberInput > label {
        font-size: 14px;
        font-weight: bold;
        color: rgb(49, 51, 63);
    }
    
    /* Placeholder text */
    div[data-baseweb="select"] > div:first-child {
        color: rgba(49, 51, 63, 0.4);
        font-size: 16px;
    }
    
    /* Form field text */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div,
    .stMultiSelect > div > div > div,
    div[data-baseweb="select"] div[role="button"] {
        font-size: 16px;
        color: rgb(49, 51, 63);
    }
    
    /* Form field placeholder */
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: rgba(49, 51, 63, 0.4);
    }
    
    /* Headers */
    h1 {
        font-size: 20px;
        font-weight: bold;
        color: rgb(49, 51, 63);
    }
    h2 {
        font-size: 16px;
        font-weight: bold;
        color: rgb(49, 51, 63);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user' not in st.session_state:
    st.session_state.user = None

if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

def login():
    """Handle user login"""
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            try:
                response = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                st.session_state.user = response.user
                st.success("Logged in successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {str(e)}")

def main():
    """Main app logic"""
    if not st.session_state.user:
        st.markdown("# 📺 TV Shows Data Entry")
        st.markdown("Please login to continue")
        login()
        return
    
    # Main content
    st.markdown("# 📺 TV Shows Data Entry")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Add Show", "Edit Show", "Add Team Member"])
    
    with tab1:
        st.markdown("## Add New Show")
        
        # Load lookup data
        lookups = load_lookup_data()
        
        # Initialize session state
        if 'active_section' not in st.session_state:
            st.session_state.active_section = 'show_details'
        
        # Initialize form data if not present
        if 'form_data' not in st.session_state:
            st.session_state.form_data = {
                'title': '',
                'network_id': None,
                'genre_id': None,
                'subgenre_id': None,
                'source_type_id': None,
                'order_type_id': None,
                'status_id': None,
                'description': '',
                'episode_count': None,
                'announcement_date': None,
                'studios': {
                    'studio_ids': [],
                    'new_studios': [],
                    'num_new_studios': 0
                },
                'team': [],
                'tmdb_id': ''
            }
            
        # Title search outside form
        st.write("### Check for Existing Shows")
        
        # Cache show titles
        @st.cache_data(ttl=3600)  # Cache for 1 hour
        def get_show_titles() -> List[str]:
            response = supabase.table('shows').select('title').execute()
            return [show['title'] for show in response.data]
        
        def search_shows(searchterm: str) -> List[str]:
            if not searchterm:
                return []
            searchterm = searchterm.lower()
            shows = get_show_titles()
            # First look for exact substring matches
            exact_matches = [show for show in shows if searchterm in show.lower()]
            if exact_matches:
                return exact_matches[:5]
            # If no substring matches, try fuzzy matching
            return difflib.get_close_matches(searchterm, shows, n=5, cutoff=0.6)
        
        # Initialize session state for search
        if 'search_text' not in st.session_state:
            st.session_state.search_text = ''
        
        # Track search text changes
        if 'show_search' in st.session_state and isinstance(st.session_state.show_search, dict):
            st.session_state.search_text = st.session_state.show_search.get('search', '')
            
        selected_title = st_searchbox(
            search_shows,
            placeholder="Search shows...",
            label="Search Title",
            key="show_search"
        )
        
        # Check if there are any matches
        current_text = st.session_state.search_text.strip() if st.session_state.search_text else ''
        
        # Create a placeholder for status messages
        status_placeholder = st.empty()
        
        # Initialize form title from search if not already set
        if 'form_title' not in st.session_state:
            st.session_state.form_title = ''
        
        # Update form title when search box value changes
        if selected_title:
            st.session_state.form_title = selected_title
            st.session_state.form_data['title'] = selected_title
        elif current_text != st.session_state.get('last_search', ''):
            st.session_state.form_title = current_text
            st.session_state.form_data['title'] = current_text
            st.session_state.last_search = current_text
            
            # Clear status when text is erased
            if not current_text:
                status_placeholder.empty()
        
        # Show warning if there are matches
        if current_text and len(current_text) >= 3:
            matches = search_shows(current_text)
            if matches:
                status_placeholder.warning("⚠️ Similar shows found in database: " + ", ".join(matches))
            else:
                status_placeholder.success("✅ No similar shows found in database.")
        else:
            # Clear the status if search is empty
            status_placeholder.empty()
        
        # Show Details Form
        with st.expander("Show Details", expanded=st.session_state.active_section == 'show_details'):
            st.subheader("Show Details")
            
            # Title from search
            title = st.text_input(
                "Title",
                value=st.session_state.form_data.get('title', st.session_state.get('form_title', '')),
                key="show_title_input",  # Different key from search state
                disabled=False  # Allow editing in case user wants to modify the title slightly
            )
            # Update form_title state with any manual edits
            st.session_state.form_title = title
            
            # Create two columns for the form layout
            left_col, right_col = st.columns(2)
            
            # Left Column
            with left_col:
                # Network
                network_options = [(n['id'], n['name']) for n in lookups['networks']]
                network_index = next((i for i, (id, _) in enumerate(network_options) if id == st.session_state.form_data.get('network_id')), None)
                network = st.selectbox(
                    "Network",
                    options=network_options,
                    format_func=lambda x: x[1],
                    index=network_index,
                    key="network_dropdown"
                )
                st.session_state.form_data['network_id'] = network[0] if network else None
                
                # Genre
                genre_options = [(g['id'], g['name']) for g in lookups['genres']]
                genre_index = next((i for i, (id, _) in enumerate(genre_options) if id == st.session_state.form_data.get('genre_id')), None)
                genre = st.selectbox(
                    "Genre",
                    options=genre_options,
                    format_func=lambda x: x[1],
                    index=genre_index,
                    key="genre_dropdown"
                )
                st.session_state.form_data['genre_id'] = genre[0] if genre else None
                
                # Subgenre
                subgenre_options = [(s['id'], s['name']) for s in lookups['subgenres']]
                subgenre_index = next((i for i, (id, _) in enumerate(subgenre_options) if id == st.session_state.form_data.get('subgenre_id')), None)
                subgenre = st.selectbox(
                    "Subgenre (optional)",
                    options=subgenre_options,
                    format_func=lambda x: x[1],
                    index=subgenre_index,
                    key="subgenre_dropdown"
                )
                st.session_state.form_data['subgenre_id'] = subgenre[0] if subgenre else None
                
                # Source Type
                source_type_options = [(s['id'], s['name']) for s in lookups['source_types']]
                source_type_index = next((i for i, (id, _) in enumerate(source_type_options) if id == st.session_state.form_data.get('source_type_id')), None)
                source_type = st.selectbox(
                    "Source Type",
                    options=source_type_options,
                    format_func=lambda x: x[1],
                    index=source_type_index,
                    key="source_type_dropdown"
                )
                st.session_state.form_data['source_type_id'] = source_type[0] if source_type else None
                
            # Right Column
            with right_col:
                # Episode count
                episode_count = st.number_input(
                    "Episode Count (optional)",
                    min_value=0,
                    value=st.session_state.form_data.get('episode_count', 0) or 0,
                    step=1,
                    help="Leave at 0 if unknown",
                    key="episode_count"
                )
                # Convert 0 to None for database storage
                st.session_state.form_data['episode_count'] = None if episode_count == 0 else episode_count
                
                # Announcement Date
                date = st.date_input(
                    "Announcement Date (optional)",
                    value=None,
                    key="announcement_date"
                )
                
                # Order type selection
                order_type_options = [(o['id'], o['name']) for o in lookups['order_types']]
                order_type_index = next((i for i, (id, _) in enumerate(order_type_options) if id == st.session_state.form_data.get('order_type_id')), None)
                order_type = st.selectbox(
                    "Order Type (optional)",
                    options=order_type_options,
                    format_func=lambda x: x[1],
                    index=order_type_index,
                    key="order_type_dropdown"
                )
                st.session_state.form_data['order_type_id'] = order_type[0] if order_type else None
                
                # Status selection
                status_options = [(s['id'], s['name']) for s in lookups['statuses']]
                status_index = next((i for i, (id, _) in enumerate(status_options) if id == st.session_state.form_data.get('status_id')), None)
                status = st.selectbox(
                    "Status (optional)",
                    options=status_options,
                    format_func=lambda x: x[1],
                    index=status_index,
                    key="status_dropdown"
                )
                st.session_state.form_data['status_id'] = status[0] if status else None
            
            # Description (full width)
            description = st.text_area(
                "Description (optional)",
                value=st.session_state.form_data.get('description', ''),
                key="description"
            )
            st.session_state.form_data['description'] = description
            
            # Store form data
            form_data = {
                'title': title,
                'network_id': network[0] if network else None,
                'genre_id': genre[0] if genre else None,
                'subgenre_id': subgenre[0] if subgenre else None,
                'source_type_id': source_type[0] if source_type else None,
                'order_type_id': order_type[0] if order_type else None,
                'status_id': status[0] if status else None,
                'description': description,
                'episode_count': episode_count,
                'announcement_date': date
            }
            st.session_state.form_data.update(form_data)
            
            with st.form("show_details_form", clear_on_submit=False):
                # Submit button to move to next section
                if st.form_submit_button("Continue", use_container_width=True):
                    if not title:
                        st.error("Please search for a show title first")
                        return
                    # Validate genre and subgenre aren't the same
                    if genre and subgenre and genre[0] == subgenre[0]:
                        st.error("⚠️ Genre and Subgenre cannot be the same")
                        return
                    st.session_state.active_section = 'studios'
                    st.rerun()
        
        # Studios Form
        with st.expander("Studios", expanded=st.session_state.active_section == 'studios'):
            st.subheader("Studios")
            
            # Filter for only type='Studio'
            studio_options = [(s['id'], s['name']) for s in lookups['studios'] if s['type'] == 'Studio']
            
            # Multiselect for existing studios
            selected_studios = st.multiselect(
                "Select Existing Studios",
                options=studio_options,
                format_func=lambda x: x[1],
                key="studio_multiselect"
            )
            
            # Number input for new studios
            num_new_studios = st.number_input(
                "Number of new studios to add",
                min_value=0,
                max_value=5,
                value=0,
                step=1,
                help="Enter a number from 0-5 to add studio name fields",
                key="num_new_studios"
            )
            
            # Text inputs for new studios
            new_studios = []
            for i in range(num_new_studios):
                new_name = st.text_input(
                    f"New Studio {i+1}",
                    key=f"new_studio_{i}",
                    value=st.session_state.get(f"new_studio_{i}", "")
                )
                if new_name and new_name.strip():
                    new_studios.append(new_name.strip())
            
            # Collect form data
            studio_data = {
                'studio_ids': [s[0] for s in selected_studios],
                'new_studios': new_studios,
                'num_new_studios': num_new_studios
            }
            
            with st.form("studios_form", clear_on_submit=False):
                # Store form data
                st.session_state.form_data['studios'] = studio_data
                
                # Submit button to move to next section
                if st.form_submit_button("Continue", use_container_width=True):
                    st.session_state.active_section = 'team'
                    st.rerun()
        
        # Team Members Form
        with st.expander("Team Members", expanded=st.session_state.active_section == 'team'):
            st.subheader("Team Members")
            
            # Number input for team members
            num_new_team = st.number_input(
                "Number of team members to add",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Enter a number from 0-10 to add team member fields",
                key="num_new_team"
            )
            
            # Team member inputs
            team_members = []
            for i in range(num_new_team):
                col1, col2 = st.columns([3, 2])
                
                # Name input
                with col1:
                    name = st.text_input(
                        f"Team Member {i+1} Name",
                        key=f"team_member_{i}_name",
                        value=st.session_state.get(f"team_member_{i}_name", "")
                    )
                
                # Role selection
                with col2:
                    roles = st.multiselect(
                        f"Role(s)",
                        options=[(r['id'], r['name']) for r in lookups['roles']],
                        format_func=lambda x: x[1],
                        key=f"team_member_{i}_roles"
                    )
                
                if name and name.strip():
                    team_members.append({
                        'name': name.strip(),
                        'role_ids': [r[0] for r in roles]
                    })
            
            with st.form("team_form", clear_on_submit=False):
                # Store form data
                st.session_state.form_data['team'] = team_members
                
                # Submit button to move to next section
                if st.form_submit_button("Continue", use_container_width=True):
                    st.session_state.active_section = 'additional'
                    st.rerun()
        
        # Additional Info Form
        with st.expander("Additional Information", expanded=st.session_state.active_section == 'additional'):
            # TMDB ID input
            tmdb_id = st.text_input(
                "TMDB ID",
                help="The Movie Database ID (if known)",
                value=st.session_state.form_data.get('tmdb_id', '')
            )
            with st.form("additional_info_form", clear_on_submit=False):
                # Store form data
                st.session_state.form_data['tmdb_id'] = tmdb_id
                
                # Submit button to move to next section
                if st.form_submit_button("Continue", use_container_width=True):
                    st.session_state.active_section = 'review'
                    st.rerun()
        
        # Check if form is ready for submission
        can_submit = True
        submit_help = ""
        
        # Check title exists and is unique
        title = st.session_state.form_data.get('title')
        if not title:
            can_submit = False
            submit_help = "Enter and save show details first"
        else:
            # Check for duplicate title
            existing = supabase.table('shows').select('id').eq('title', title).execute()
            if existing.data:
                can_submit = False
                submit_help = "A show with this title already exists"
        
        # Check network is selected (required)
        if can_submit and not st.session_state.form_data.get('network_id'):
            can_submit = False
            submit_help = "Select a network in show details"
        
        # Submit button
        submit_clicked = st.button(
            "Submit All", 
            disabled=not can_submit,
            help=submit_help,
            use_container_width=True
        )

        if submit_clicked:
            # Show submission status at top
            status_container = st.empty()
            
            # Validate we have all required data
            missing_sections = []
            if 'title' not in st.session_state.form_data:
                missing_sections.append("Show Details")
            if 'studios' not in st.session_state.form_data:
                missing_sections.append("Studios")
            if 'team' not in st.session_state.form_data:
                missing_sections.append("Team Members")
                
            if missing_sections:
                status_container.error(
                    "⚠️ Cannot save - Missing required sections:\n" + 
                    "\n".join(f"• {section}" for section in missing_sections) + 
                    "\n\nPlease fill out and save all required sections before submitting.")
            else:
                try:
                    # Process studios first
                    studio_data = st.session_state.form_data['studios']
                    studio_ids = [s[0] for s in studio_data['selected_studios']]  # IDs of existing studios
                    
                    # Add new studios
                    for studio_name in studio_data['new_studios']:
                        try:
                            # Check if studio exists (case insensitive)
                            existing = supabase.table('studio_list') \
                                .select('id, studio') \
                                .ilike('studio', studio_name) \
                                .execute()
                            
                            if existing.data:
                                # Studio exists, use its ID
                                studio_ids.append(existing.data[0]['id'])
                                st.success(f"Using existing studio: {existing.data[0]['studio']}")
                            else:
                                # Create new studio
                                response = supabase.table('studio_list').insert({
                                    "studio": studio_name,
                                    "type": "production company"
                                }).execute()
                                if response.data:
                                    studio_ids.append(response.data[0]['id'])
                                    st.success(f"Added new studio: {studio_name}")
                        except Exception as e:
                            st.error(f"Error with studio '{studio_name}': {str(e)}")
                            continue
                    
                    # Format show data
                    show_data = {
                        'title': st.session_state.form_data['title'],
                        'network_id': st.session_state.form_data.get('network_id'),  # Required
                        'genre_id': st.session_state.form_data.get('genre_id'),
                        'subgenres': [st.session_state.form_data.get('subgenre_id')] if st.session_state.form_data.get('subgenre_id') else [],
                        'source_type_id': st.session_state.form_data.get('source_type_id'),
                        'order_type_id': st.session_state.form_data.get('order_type_id'),
                        'status_id': st.session_state.form_data.get('status_id'),
                        'episode_count': st.session_state.form_data.get('episode_count'),
                        'description': st.session_state.form_data.get('description'),
                        'studios': studio_ids,  # Already an array of bigints
                        'tmdb_id': st.session_state.form_data.get('tmdb_id'),
                        'date': st.session_state.form_data.get('announcement_date')
                    }
                    
                    # Validate required fields
                    if not show_data['network_id']:
                        raise ValueError('Network is required')
                    
                    # Check if show with this title already exists
                    existing = supabase.table('shows') \
                        .select('id, title') \
                        .eq('title', show_data['title']) \
                        .execute()
                    
                    if existing.data:
                        raise ValueError(f"A show with title '{show_data['title']}' already exists. Please use a different title.")
                    
                    # Add show to database
                    response = supabase.table('shows').insert(show_data).execute()
                    
                    if response.data:
                        show_id = response.data[0]['id']
                        
                        # Add team members
                        for member in st.session_state.form_data['team']['members']:
                            for role_id in member['role_ids']:
                                supabase.table('show_team').insert({
                                    'show_id': show_id,
                                    'name': member['name'],
                                    'role_type_id': role_id
                                }).execute()
                        
                        show_title = show_data['title']  # Store title before clearing
                        
                        # Show success message in main area (not in form)
                        st.success(f"✅ Successfully added: {show_title}")
                        st.markdown("Show saved to database. Form data preserved.")
                    else:
                        st.error(
                            "⚠️ Show not saved - Database error\n" +
                            "Please try again or contact support if the problem persists.")
                except Exception as e:
                    error_msg = str(e)
                    if "duplicate key" in error_msg.lower():
                        st.error(
                            "⚠️ Show not saved - A show with this title already exists.\n" +
                            "Please check the title and try again."
                        )
                    elif "network" in error_msg.lower():
                        st.error(
                            "⚠️ Show not saved - Network is required.\n" +
                            "Please select a network and try again."
                        )
                    else:
                        st.error(
                            "⚠️ Show not saved - An error occurred:\n" +
                            f"{error_msg}\n" +
                            "Please fix the error and try again."
                        )
                    # Don't clear form data on error so user can fix and retry
                    return

                
                # Create new studio if needed
                studio_id = None
                if studio_selection[0] is None and new_studio_name:
                    # Add new studio
                    response = supabase.table('studios').insert({
                        'name': new_studio_name
                    }).execute()
                    
                    if response.data:
                        studio_id = response.data[0]['id']
                        st.success(f"Created new studio: {new_studio_name}")
                    else:
                        st.error("Failed to create new studio")
                        return
                else:
                    studio_id = studio_selection[0]
                
                # Add show
                show_data = {
                    'title': title,
                    'description': description if description else None,
                    'status_id': status[0] if status else None,
                    'network_id': network[0] if network else None,
                    'studio_id': studio_id,
                    'genre_id': genre[0] if genre else None,
                    'subgenres': [g[0] for g in subgenres] if subgenres else None,
                    'source_type_id': source_type[0] if source_type else None,
                    'order_type_id': order_type[0] if order_type else None,
                    'date': date.isoformat() if date else None,
                    'episode_count': episode_count if episode_count > 0 else None,
                    'tmdb_id': int(tmdb_id) if tmdb_id else None
                }
                
                # Add show and get its ID
                response = supabase.table('shows').insert(show_data).execute()
                
                if response.data:
                    st.success(f"Added show: {title}")
                    # Clear form
                    st.session_state.form_title = ''
                    st.rerun()
                else:
                    st.error("Failed to add show")
    
    with tab2:
        st.markdown("## Edit Existing Show")
        # TODO: Implement edit show form with fuzzy matching
    
    with tab3:
        st.markdown("## Add Team Member")
        # TODO: Implement team member form

if __name__ == "__main__":
    main()
