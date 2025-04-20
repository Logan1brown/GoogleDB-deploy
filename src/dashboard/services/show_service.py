"""
Data entry services for interacting with Supabase.
"""

from typing import Dict, List
from datetime import datetime, date
from postgrest import APIError
import streamlit as st
from supabase.client import create_client, Client
import difflib
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_KEY')

if not url or not key:
    raise ValueError("Missing Supabase credentials. Make sure SUPABASE_URL and SUPABASE_SERVICE_KEY are set in .env")

supabase: Client = create_client(url, key)

@st.cache_data(ttl=3600)
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
    
    # Load role types
    response = supabase.table('role_types').select('id, role, active').execute()
    lookups['role_types'] = [{'id': r['id'], 'name': r['role']} for r in response.data]
    
    # Load source types
    response = supabase.table('source_types').select('id, type').execute()
    lookups['source_types'] = [{'id': s['id'], 'name': s['type']} for s in response.data]
    
    # Load order types
    response = supabase.table('order_types').select('id, type').execute()
    lookups['order_types'] = [{'id': o['id'], 'name': o['type']} for o in response.data]
    
    # Load status types
    response = supabase.table('status_types').select('id, status').execute()
    lookups['status_types'] = [{'id': s['id'], 'name': s['status']} for s in response.data]
    
    return lookups

def search_shows(title: str) -> List[str]:
    """Search for shows by title using fuzzy matching"""
    if not title or len(title.strip()) < 3:
        return []

    # Get all existing shows from shows table
    response = supabase.table('shows').select('title').eq('active', True).execute()
    existing_shows = [show['title'] for show in response.data]

    # Find fuzzy matches
    matches = difflib.get_close_matches(title.lower(), [s.lower() for s in existing_shows], n=5, cutoff=0.6)
    
    # Return original case versions of matches
    return [next(s for s in existing_shows if s.lower() == m) for m in matches]

def load_show(title: str, lookups: Dict[str, List[Dict]] = None) -> dict:
    """Load show data for editing"""
    # Get lookups if not provided
    if lookups is None:
        lookups = load_lookup_data()
    
    # Get show details
    response = supabase.table('shows') \
        .select('*') \
        .eq('title', title) \
        .eq('active', True) \
        .single() \
        .execute()
    
    if not response.data:
        raise ValueError(f"Show not found: {title}")
    
    # Get team members
    team_response = supabase.table('show_team') \
        .select('*') \
        .eq('show_id', response.data['id']) \
        .execute()
    
    team_members = [{
        'name': member['name'],
        'role_type_id': member['role_type_id']
    } for member in team_response.data]
    
    # Parse date string into date object if present
    date_str = response.data.get('date')
    if date_str:
        try:
            show_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            show_date = None
    else:
        show_date = None
    
    show_data = {
        'id': int(response.data['id']),  # Convert to int for ShowFormState
        'title': response.data['title'],
        'original_title': response.data['title'],  # Save original title for comparison
        'network_id': response.data['network_id'],
        'genre_id': response.data['genre_id'],
        'subgenres': response.data.get('subgenres', []) if response.data.get('subgenres') else [],
        'source_type_id': response.data.get('source_type_id'),
        'order_type_id': response.data.get('order_type_id'),
        'status_id': response.data.get('status_id'),
        'date': show_date,
        'episode_count': response.data.get('episode_count', 0),
        'description': response.data.get('description', ''),
        'studios': response.data.get('studios', []) if response.data.get('studios') else [],
        'new_studios': [],
        'team_members': team_members
    }
    
    return show_data

def save_show(show_data: dict, operation: str = "Add show"):
    """Save show data to database and handle form reset."""
    # Validate required fields
    required_fields = ['title', 'network_id']
    if not all(show_data.get(field) for field in required_fields):
        raise ValueError(f"Missing required fields: {', '.join(required_fields)}")
    
    # First handle the show data
    if operation == "Edit Show":
        if not show_data.get('id'):
            raise ValueError("Missing show ID for edit operation")
            
        show_id = show_data['id']  # Set show_id before the try block
            
        # Get existing show to verify it exists
        try:
            existing = supabase.table('shows') \
                .select('id,title,active') \
                .eq('id', show_id) \
                .eq('active', True) \
                .execute()
                
            if not existing.data:
                raise ValueError(f"Show with ID {show_id} not found")
                
            if len(existing.data) > 1:
                raise ValueError(f"Multiple shows found with ID {show_id}")
                
        except Exception as e:
            raise
    
    else:
        show_id = None
        
    # Check title uniqueness
    title_check_query = supabase.table('shows') \
        .select('id,title') \
        .eq('title', show_data['title']) \
        .eq('active', True)
        
    if show_id is not None:
        title_check_query = title_check_query.neq('id', show_id)
        
    title_check = title_check_query.execute()
    
    if title_check.data:
        raise ValueError(f"Show title '{show_data['title']}' already exists")
    
    # Handle studios first to get IDs
    studio_ids = show_data.get('studios', [])
    
    # Handle new studios
    new_studios = show_data.get('new_studios', [])
    if new_studios:
        for studio_name in new_studios:
            try:
                # Add new studio as production company
                response = supabase.table('studio_list').insert({
                    'studio': studio_name,
                    'type': 'Production Company'
                }).execute()
                studio_ids.append(response.data[0]['id'])
            except Exception as e:
                raise Exception(f"Error adding studio '{studio_name}': {str(e)}")
    
    # Prepare data for insert/update
    data = {
        'title': show_data['title'],
        'network_id': show_data['network_id'],
        'genre_id': show_data['genre_id'],
        'subgenres': show_data.get('subgenres', []),
        'source_type_id': show_data.get('source_type_id'),
        'order_type_id': show_data.get('order_type_id'),
        'status_id': show_data.get('status_id'),
        'episode_count': show_data.get('episode_count'),
        'description': show_data.get('description', ''),
        'studios': studio_ids
    }
    
    # Handle date - convert to string if it's a date object
    show_date = show_data.get('date')
    if show_date:
        if isinstance(show_date, date):
            data['date'] = show_date.isoformat()
        else:
            data['date'] = show_date

    # Save show
    try:
        if operation == "Edit Show":
            # Update show using ID
            response = supabase.table('shows') \
                .update(data) \
                .eq('id', show_id) \
                .execute()
            if not response.data:
                raise Exception("No data returned from update")
                
            # Delete existing team members for update
            supabase.table('show_team') \
                .delete() \
                .eq('show_id', show_id) \
                .execute()
        else:
            # Insert new show
            response = supabase.table('shows') \
                .insert(data) \
                .execute()
            if not response.data:
                raise Exception("No data returned from insert")
            show_id = response.data[0]['id']
    except Exception as e:
        print(f"Error saving show: {str(e)}")
        raise
    
    # Add team members - one row per role
    if show_data.get('team_members'):
        try:
            # Delete existing team members
            supabase.table('show_team').delete().eq('show_id', show_id).execute()
            
            # Create one row per role
            team_inserts = []
            for member in show_data['team_members']:
                name = member['name'].strip()
                if not name:
                    continue
                
                role_type_id = member['role_type_id']
                if not isinstance(role_type_id, int):
                    continue
                    
                team_inserts.append({
                    'show_id': show_id,
                    'name': name,
                    'role_type_id': role_type_id
                })
            
            if team_inserts:
                supabase.table('show_team').insert(team_inserts).execute()
        except Exception as e:
            raise Exception(f"Error saving team members: {str(e)}")
                
    return show_id


def remove_show(show_id: int) -> bool:
    """Mark a show as inactive in the database using a soft delete approach.
    
    Args:
        show_id: The ID of the show to mark as inactive
        
    Returns:
        bool: True if the show was newly marked as inactive, False if it was already inactive
    """
    if not show_id:
        raise ValueError("Missing show ID for remove operation")
        
    # First check if show exists and get its active status
    show = supabase.table('shows') \
        .select('id,active,title') \
        .eq('id', show_id) \
        .execute()
        
    if not show.data:
        return False  # Show not found, consider it already inactive
        
    show_data = show.data[0]
    if not show_data.get('active'):
        return False  # Show is already inactive
        
    try:
        # Mark show as inactive
        supabase.table('shows') \
            .update({'active': False}) \
            .eq('id', show_id) \
            .execute()
            
        # Mark show_team entries as inactive if they exist
        team_result = supabase.table('show_team') \
            .update({'active': False}) \
            .eq('show_id', show_id) \
            .eq('active', True) \
            .execute()
        print(f"Team update result: {team_result.data}")  # Debug
            
        # Mark show_studios entries as inactive if they exist
        studios_result = supabase.table('show_studios') \
            .update({'active': False}) \
            .eq('show_id', show_id) \
            .eq('active', True) \
            .execute()
        print(f"Studios update result: {studios_result.data}")  # Debug
            
        return True  # Show was newly marked as inactive
            
    except Exception as e:
        print(f"Error in remove_show: {str(e)}")  # Log error but continue
        return True  # Show was marked inactive even if related updates failed
    # Mark show_studios entries as inactive
    try:
        # First check what studios exist
        studios = supabase.table('show_studios') \
            .select('*') \
            .eq('show_id', show_id) \
            .execute()
        print(f"Found studios: {studios.data}")  # Debug
        
        # Then try to update them
        result = supabase.table('show_studios') \
            .update({'active': False}) \
            .eq('show_id', show_id) \
            .execute()
        print(f"Update result: {result.data}")  # Debug
    except Exception as e:
        print(f"Error updating studios: {str(e)}")  # Debug but continue
        
    return True  # Show was newly marked as inactive
