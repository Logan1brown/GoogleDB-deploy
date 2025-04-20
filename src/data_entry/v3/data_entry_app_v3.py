import os
from datetime import date
from typing import Dict, List, Any
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

# Must be the first Streamlit command
st.set_page_config(page_title="Data Entry App", layout="wide")

# Load environment variables from project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Initialize Supabase client
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_KEY')
supabase: Client = create_client(url, key)

# Initialize session state
if 'lookups' not in st.session_state:
    st.session_state.lookups = load_lookup_data()

if 'show_data' not in st.session_state:
    st.session_state.show_data = {
        'title': '',
        'network_id': None,
        'genre_id': None,
        'subgenre_id': None,
        'source_type_id': None,
        'order_type_id': None,
        'episode_count': 0,
        'date': date.today()
    }

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

def init_session_state():
    """Initialize session state with default values"""
    if 'lookups' not in st.session_state:
        st.session_state.lookups = load_lookup_data()
    
    if 'show_data' not in st.session_state:
        st.session_state.show_data = {
            'title': '',
            'network_id': None,
            'genre_id': None,
            'subgenre_id': None,
            'source_type_id': None,
            'order_type_id': None,
            'episode_count': 0,
            'date': date.today()
        }

def save_studios(studio_data: Dict) -> List[int]:
    """Process and save studios, return list of studio IDs"""
    studio_ids = studio_data['studio_ids']
    for studio_name in studio_data['new_studios']:
        # Check existing (case insensitive)
        existing = supabase.table('studio_list').select('id, studio').ilike('studio', studio_name).execute()
        if existing.data:
            studio_ids.append(existing.data[0]['id'])
            st.success(f"Using existing studio: {existing.data[0]['studio']}")
        else:
            # Create new
            response = supabase.table('studio_list').insert({
                "studio": studio_name,
                "type": "production company"
            }).execute()
            if response.data:
                studio_ids.append(response.data[0]['id'])
                st.success(f"Added new studio: {studio_name}")
    return studio_ids

def save_team(show_id: int, team_data: List[Dict]) -> None:
    """Save team members for a show"""
    for member in team_data:
        for role_id in member['role_ids']:
            supabase.table('show_team').insert({
                'show_id': show_id,
                'name': member['name'],
                'role_type_id': role_id
            }).execute()

def save_show(show_data: Dict) -> bool:
    """Save show and related data"""
    try:
        # Process studios first
        studio_ids = save_studios({
            'studio_ids': show_data['studio_ids'],
            'new_studios': show_data['new_studios']
        })
        
        # Format show data for database
        db_show_data = {
            'title': show_data['title'],
            'network_id': show_data['network_id'],
            'genre_id': show_data['genre_id'],
            'subgenres': [show_data['subgenre_id']] if show_data['subgenre_id'] else [],
            'source_type_id': show_data['source_type_id'],
            'order_type_id': show_data['order_type_id'],
            'episode_runtime': show_data['episode_runtime'],
            'start_date': show_data['start_date'],
            'studios': studio_ids
        }
        
        # Validate required fields
        if not db_show_data['network_id']:
            raise ValueError('Network is required')
        
        # Check if show exists
        existing = supabase.table('shows').select('id, title').eq('title', db_show_data['title']).execute()
        if existing.data:
            raise ValueError(f"A show with title '{db_show_data['title']}' already exists")
        
        # Save show
        response = supabase.table('shows').insert(db_show_data).execute()
        if response.data:
            show_id = response.data[0]['id']
            save_team(show_id, show_data['team_members'])
            st.success(f"✅ Successfully added: {db_show_data['title']}")
            return True
        else:
            st.error("⚠️ Show not saved - Database error")
            return False
    except Exception as e:
        error_msg = str(e)
        if "duplicate key" in error_msg.lower():
            st.error("⚠️ Show not saved - A show with this title already exists")
        elif "network" in error_msg.lower():
            st.error("⚠️ Show not saved - Network is required")
        else:
            st.error(f"⚠️ Error saving show: {error_msg}")
        return False

def init_session_state():
    """Initialize session state variables"""
    # Auth state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # Active tab state
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    
    # Current page tracking
    if 'current_page' not in st.session_state:
        st.session_state.current_page = None
        
    # Load lookup data
    if 'lookups' not in st.session_state:
        st.session_state.lookups = load_lookup_data()
    
    # Initialize show data structure if not present
    if 'show_data' not in st.session_state:
        st.session_state.show_data = {
            'title': '',
            'network_id': None,
            'genre_id': None,
            'subgenre_id': None,
            'source_type_id': None,
            'order_type_id': None,
            'episode_runtime': 0,
            'start_date': date.today(),
            'studio_ids': [],
            'new_studios': [],
            'team_members': []
        }
        
    # Initialize counters if not present
    if 'num_new_studios' not in st.session_state:
        st.session_state.num_new_studios = 0
    if 'num_team_members' not in st.session_state:
        st.session_state.num_team_members = 0
    
    # Track current page
    current_page = __file__
    if 'current_page' not in st.session_state:
        st.session_state.current_page = current_page

def login():
    """Handle user login"""
    # If already logged in, return True
    if 'authenticated' in st.session_state and st.session_state.authenticated:
        return True
        
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            try:
                # Get auth token
                auth = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                
                # Store auth in session
                st.session_state.user = auth.user
                st.session_state.access_token = auth.session.access_token
                st.session_state.refresh_token = auth.session.refresh_token
                st.session_state.authenticated = True
                
                # Update Supabase client with auth token
                supabase.auth.set_session(auth.session.access_token, auth.session.refresh_token)
                
                st.success("Logged in successfully!")
                st.rerun()
                return True
            except Exception as e:
                st.error(f"Login failed: {str(e)}")
                st.session_state.authenticated = False
                return False
                
    return False

def render_studio_inputs(num_studios: int) -> List[str]:
    """Render studio input fields"""
    new_studios = []
    for i in range(num_studios):
        name = st.text_input(
            f"New Studio {i+1}",
            key=f"new_studio_{i}",
            value=st.session_state.get(f"new_studio_{i}", "")
        )
        if name and name.strip():
            new_studios.append(name.strip())
    return new_studios

def render_team_inputs(num_team: int, roles: List[Dict]) -> List[Dict]:
    """Render team member input fields"""
    team = []
    for i in range(num_team):
        col1, col2 = st.columns([3, 2])
        with col1:
            name = st.text_input(
                f"Team Member {i+1} Name",
                key=f"team_member_{i}_name"
            )
        with col2:
            member_roles = st.multiselect(
                "Role(s)",
                options=[(r['id'], r['name']) for r in roles],
                format_func=lambda x: x[1],
                key=f"team_member_{i}_roles"
            )
        if name and name.strip() and member_roles:
            team.append({
                'name': name.strip(),
                'role_ids': [r[0] for r in member_roles]
            })
    return team

def main():
    """Main application homepage"""
    # Import menu here to avoid circular imports
    from data_entry_menu import menu
    
    # Initialize session state first
    init_session_state()
    
    # Show menu based on auth state
    menu()
    
    # Handle login if not authenticated
    if not st.session_state.authenticated:
        login()
        return
    
    # Show main content
    st.title("Show Data Entry")
    st.write("""
    Welcome to the Show Data Entry application! Use the sidebar to navigate between pages:
    
    - **Add Show**: Add a new show to the database
    - **Edit Show**: Modify an existing show's details
    - **Remove Show**: Remove a show from the database
    """)
    
    # Show some stats or recent activity
    st.subheader("Recent Activity")
    st.info("Coming soon! This section will show recent changes to the database.")


if __name__ == "__main__":
    main()
