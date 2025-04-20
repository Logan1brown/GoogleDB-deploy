import os
from datetime import date
from typing import Dict, List, Optional
import streamlit as st
from supabase import create_client, Client

# Initialize Supabase client
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_KEY')
)

# Constants
REQUIRED_FIELDS = ['title', 'network_id', 'genre_id']

# Database Functions
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_lookup_data() -> Dict[str, List[Dict]]:
    """Load all lookup data needed for the forms"""
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

def search_shows(title: str) -> List[str]:
    """Search for shows with similar titles"""
    if len(title) < 3:
        return []
    
    response = supabase.table('shows').select('title').execute()
    return [s['title'] for s in response.data if title.lower() in s['title'].lower()]

def load_all_shows() -> List[Dict]:
    """Get all shows for the edit selector"""
    return supabase.table('shows').select('*').execute().data

def load_show_studios(show_id: int) -> List[Dict]:
    """Load studios for a show"""
    show = supabase.table('shows').select('studios').eq('id', show_id).execute().data[0]
    if not show['studios']:
        return []
    return supabase.table('studio_list').select('id, studio').in_('id', show['studios']).execute().data

def load_show_team(show_id: int) -> List[Dict]:
    """Get team members for a specific show"""
    return supabase.table('show_team').select('name, role_type_id').eq('show_id', show_id).execute().data

def save_show(data: Dict) -> None:
    """Save show data to database"""
    try:
        # Process studios first
        studio_ids = [s[0] for s in data['studios']]  # IDs of existing studios
        
        # Add new studios
        for studio_name in data['new_studios']:
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
            'title': data['title'],
            'network_id': data['network_id'],  # Required
            'genre_id': data['genre_id'],
            'subgenres': [data['subgenre_id']] if data['subgenre_id'] else [],
            'source_type_id': data['source_type_id'],
            'order_type_id': data['order_type_id'],
            'status_id': data['status_id'],
            'episode_count': data['episode_count'],
            'description': data['description'],
            'studios': studio_ids,  # Array of studio IDs
            'tmdb_id': data['tmdb_id'],
            'date': data['announcement_date']
        }
        
        # Validate required fields
        if not show_data['network_id']:
            raise ValueError('Network is required')
        
        # Check if show exists
        existing = supabase.table('shows') \
            .select('id, title') \
            .eq('title', show_data['title']) \
            .execute()
        
        if existing.data:
            raise ValueError(f"A show with title '{show_data['title']}' already exists")
        
        # Add show to database
        response = supabase.table('shows').insert(show_data).execute()
        
        if response.data:
            show_id = response.data[0]['id']
            
            # Add team members
            for member in data['team']:
                for role_id in member['role_ids']:
                    supabase.table('show_team').insert({
                        'show_id': show_id,
                        'name': member['name'],
                        'role_type_id': role_id
                    }).execute()
            
            st.success(f"✅ Successfully added: {show_data['title']}")
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
            st.error(f"⚠️ Error saving show: {error_msg}")

def update_show(show_id: int, data: Dict) -> None:
    """Update an existing show and its related data"""
    try:
        # Process studios first
        studio_ids = [s[0] for s in data['studios']]  # IDs of existing studios
        
        # Add new studios
        for studio_name in data['new_studios']:
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
            'title': data['title'],
            'network_id': data['network_id'],  # Required
            'genre_id': data['genre_id'],
            'subgenres': [data['subgenre_id']] if data['subgenre_id'] else [],
            'source_type_id': data['source_type_id'],
            'order_type_id': data['order_type_id'],
            'status_id': data['status_id'],
            'episode_count': data['episode_count'],
            'description': data['description'],
            'studios': studio_ids,  # Array of studio IDs
            'tmdb_id': data['tmdb_id'],
            'date': data['announcement_date']
        }
        
        # Validate required fields
        if not show_data['network_id']:
            raise ValueError('Network is required')
        
        # Update show
        response = supabase.table('shows').update(show_data).eq('id', show_id).execute()
        
        if response.data:
            # Update team members
            # First delete existing team members
            supabase.table('show_team').delete().eq('show_id', show_id).execute()
            
            # Then add new team members
            for member in data['team']:
                for role_id in member['role_ids']:
                    supabase.table('show_team').insert({
                        'show_id': show_id,
                        'name': member['name'],
                        'role_type_id': role_id
                    }).execute()
            
            st.success(f"✅ Successfully updated: {show_data['title']}")
            st.markdown("Show updated in database. Form data preserved.")
        else:
            st.error(
                "⚠️ Show not updated - Database error\n" +
                "Please try again or contact support if the problem persists.")
    except Exception as e:
        error_msg = str(e)
        if "network" in error_msg.lower():
            st.error(
                "⚠️ Show not updated - Network is required.\n" +
                "Please select a network and try again."
            )
        else:
            st.error(f"⚠️ Error updating show: {error_msg}")

def delete_show(show_id: int) -> None:
    """Delete show and all related data"""
    with supabase.transaction() as txn:
        txn.table('show_team').delete().eq('show_id', show_id).execute()
        txn.table('shows').delete().eq('id', show_id).execute()

def validate_form(data: Dict) -> bool:
    """Validate form data before saving"""
    for field in REQUIRED_FIELDS:
        if not data.get(field):
            st.error(f"Missing required field: {field}")
            return False
    return True

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
    st.set_page_config(
        page_title="Data Entry App",
        layout="wide"
    )
    
    # Check if user is logged in
    if 'user' not in st.session_state:
        login()
        return
    
    # Load lookup data
    lookups = load_lookup_data()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Add Show", "Edit/Remove Show"])
    
    # Add Show Tab
    with tab1:
        # Search section (outside form)
        search_text = st.text_input("Search Shows", key="add_search")
        if search_text:
            matches = search_shows(search_text)
            if matches:
                st.warning("Similar shows found: " + ", ".join(matches))
        
        # Add form
        with st.form("add_show_form"):
            # Show details
            title = st.text_input("Title", value=search_text)
            
            col1, col2 = st.columns(2)
            with col1:
                network = st.selectbox(
                    "Network",
                    options=[(n['id'], n['name']) for n in lookups['networks']],
                    format_func=lambda x: x[1],
                    key="add_network"
                )
                
                genre = st.selectbox(
                    "Genre",
                    options=[(g['id'], g['name']) for g in lookups['genres']],
                    format_func=lambda x: x[1],
                    key="add_genre"
                )
                
                subgenre = st.selectbox(
                    "Subgenre (optional)",
                    options=[(s['id'], s['name']) for s in lookups['subgenres']],
                    format_func=lambda x: x[1],
                    key="add_subgenre"
                )
                
                source_type = st.selectbox(
                    "Source Type",
                    options=[(s['id'], s['name']) for s in lookups['source_types']],
                    format_func=lambda x: x[1],
                    key="add_source_type"
                )
                
                order_type = st.selectbox(
                    "Order Type",
                    options=[(o['id'], o['name']) for o in lookups['order_types']],
                    format_func=lambda x: x[1],
                    key="add_order_type"
                )
            
            with col2:
                announcement_date = st.date_input(
                    "Announcement Date",
                    value=None,
                    key="add_announcement_date"
                )
            
            # Studios section
            st.subheader("Studios")
            
            # Filter for only type='Studio'
            studio_options = [(s['id'], s['name']) for s in lookups['studios'] if s['type'] == 'Studio']
            
            # Multiselect for existing studios
            selected_studios = st.multiselect(
                "Select Existing Studios",
                options=studio_options,
                format_func=lambda x: x[1],
                key="add_studios"
            )
            
            # Number input for new studios
            num_new_studios = st.number_input(
                "Number of new studios to add",
                min_value=0,
                max_value=5,
                value=0,
                step=1,
                help="Enter a number from 0-5 to add studio name fields",
                key="add_num_new_studios"
            )
            
            # Text inputs for new studios
            new_studios = []
            for i in range(num_new_studios):
                new_name = st.text_input(
                    f"New Studio {i+1}",
                    key=f"add_new_studio_{i}",
                    value=st.session_state.get(f"add_new_studio_{i}", "")
                )
                if new_name and new_name.strip():
                    new_studios.append(new_name.strip())
            
            # Team section
            st.subheader("Team Members")
            
            # Number input for team members
            num_team = st.number_input(
                "Number of team members to add",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Enter a number from 0-10 to add team member fields",
                key="add_num_team"
            )
            
            # Text inputs and role selects for team members
            team = []
            for i in range(num_team):
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input(
                        f"Team Member {i+1} Name",
                        key=f"add_team_name_{i}",
                        value=st.session_state.get(f"add_team_name_{i}", "")
                    )
                with col2:
                    roles = st.multiselect(
                        f"Team Member {i+1} Roles",
                        options=[(r['id'], r['name']) for r in lookups['roles']],
                        format_func=lambda x: x[1],
                        key=f"add_team_roles_{i}"
                    )
                if name and name.strip() and roles:
                    team.append({
                        'name': name.strip(),
                        'role_ids': [r[0] for r in roles]
                    })
            
            # Submit button
            submitted = st.form_submit_button("Add Show")
            
            if submitted:
                data = {
                    'title': title,
                    'network_id': network[0] if network else None,
                    'genre_id': genre[0] if genre else None,
                    'subgenre_id': subgenre[0] if subgenre else None,
                    'announcement_date': announcement_date,
                    'studios': studios,
                    'team': team,
                    'source_type_id': source_type[0] if source_type else None,
                    'order_type_id': order_type[0] if order_type else None
                }
                
                if validate_form(data):
                    save_show(data)
                    st.success("Show added successfully!")
    
    # Edit/Remove Show Tab
    with tab2:
        shows = load_all_shows()
        selected = st.selectbox(
            "Select Show",
            options=shows,
            format_func=lambda x: x['title']
        )
        
        if selected:
            # Load existing data
            existing_studios = load_show_studios(selected['id'])
            existing_team = load_show_team(selected['id'])
            
            with st.form("edit_show_form"):
                # Show details
                title = st.text_input("Title", value=selected['title'])
                
                col1, col2 = st.columns(2)
                with col1:
                    network = st.selectbox(
                        "Network",
                        options=[(n['id'], n['name']) for n in lookups['networks']],
                        format_func=lambda x: x[1],
                        index=[i for i, n in enumerate(lookups['networks']) 
                              if n['id'] == selected['network_id']][0]
                    )
                    
                    genre = st.selectbox(
                        "Genre",
                        options=[(g['id'], g['name']) for g in lookups['genres']],
                        format_func=lambda x: x[1],
                        index=[i for i, g in enumerate(lookups['genres']) 
                              if g['id'] == selected['genre_id']][0]
                    )
                    
                    subgenre = st.selectbox(
                        "Subgenre (optional)",
                        options=[(s['id'], s['name']) for s in lookups['subgenres']],
                        format_func=lambda x: x[1],
                        index=[i for i, s in enumerate(lookups['subgenres']) 
                              if s['id'] == selected.get('subgenre_id', None)][0] 
                              if selected.get('subgenre_id') else 0
                    )
                    
                    source_type = st.selectbox(
                        "Source Type",
                        options=[(s['id'], s['name']) for s in lookups['source_types']],
                        format_func=lambda x: x[1],
                        index=[i for i, s in enumerate(lookups['source_types']) 
                              if s['id'] == selected.get('source_type_id', None)][0] 
                              if selected.get('source_type_id') else 0
                    )
                    
                    order_type = st.selectbox(
                        "Order Type",
                        options=[(o['id'], o['name']) for o in lookups['order_types']],
                        format_func=lambda x: x[1],
                        index=[i for i, o in enumerate(lookups['order_types']) 
                              if o['id'] == selected.get('order_type_id', None)][0] 
                              if selected.get('order_type_id') else 0
                    )
                
                with col2:
                    announcement_date = st.date_input(
                        "Announcement Date",
                        value=selected.get('announcement_date', date.today())
                    )
                
                # Studios section
                st.subheader("Studios")
                col1, col2 = st.columns(2)
                with col1:
                    studio_options = [(s['id'], s['name']) for s in lookups['studios']]
                    studio_ids = [s['id'] for s in existing_studios]
                    default_studios = [opt for opt in studio_options if opt[0] in studio_ids]
                    selected_studios = st.multiselect(
                        "Select Existing Studios",
                        options=studio_options,
                        format_func=lambda x: x[1],
                        default=default_studios,
                        key="edit_studios"
                    )
                with col2:
                    num_new_studios = st.number_input(
                        "Number of New Studios",
                        min_value=0,
                        max_value=10,
                        value=0,
                        step=1,
                        key="edit_num_new_studios"
                    )
                    
                new_studios = []
                for i in range(int(num_new_studios)):
                    name = st.text_input(
                        f"New Studio {i+1} Name",
                        key=f"edit_new_studio_{i}"
                    )
                    if name and name.strip():
                        new_studios.append(name.strip())
                
                # Team section
                st.subheader("Team Members")
                num_team = st.number_input(
                    "Number of Team Members",
                    min_value=0,
                    max_value=max(20, len(existing_team)),
                    value=len(existing_team),
                    step=1
                )
                
                team = []
                for i in range(int(num_team)):
                    col1, col2 = st.columns(2)
                    with col1:
                        name = st.text_input(
                            f"Team Member {i+1} Name",
                            value=existing_team[i]['name'] if i < len(existing_team) else "",
                            key=f"edit_team_name_{i}"
                        )
                    with col2:
                        roles = st.multiselect(
                            f"Team Member {i+1} Roles",
                            options=[(r['id'], r['name']) for r in lookups['roles']],
                            format_func=lambda x: x[1],
                            default=[(r['role_type_id'], next(role['name'] 
                                    for role in lookups['roles'] 
                                    if role['id'] == r['role_type_id']))
                                   for r in existing_team if i < len(existing_team)],
                            key=f"edit_team_roles_{i}"
                        )
                    if name and roles:
                        team.append({
                            'name': name,
                            'role_ids': [r[0] for r in roles]
                        })
                
                # Save/Delete buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("Save Changes"):
                        data = {
                            'title': title,
                            'network_id': network[0],
                            'genre_id': genre[0],
                            'subgenre_id': subgenre[0] if subgenre else None,
                            'announcement_date': announcement_date,
                            'studios': [{'id': s[0]} for s in selected_studios],
                            'new_studios': new_studios,
                            'team': team,
                            'source_type_id': source_type[0] if source_type else None,
                            'order_type_id': order_type[0] if order_type else None
                        }
                        
                        if validate_form(data):
                            update_show(selected['id'], data)
                            st.success("Show updated successfully!")
                
                with col2:
                    if st.form_submit_button("Delete Show", type="primary"):
                        delete_show(selected['id'])
                        st.success("Show deleted successfully!")
                        st.rerun()

if __name__ == "__main__":
    main()
