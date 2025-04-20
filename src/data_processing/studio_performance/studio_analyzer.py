"""Studio Performance Analysis.

Analyzes studio performance metrics including:
- Network relationships
- Genre specialization
- Show volume and success rates
"""

from typing import Dict, List
import pandas as pd
import logging
import streamlit as st
logger = logging.getLogger(__name__)
# No longer using Google Sheets
# from src.dashboard.utils.sheets_client import sheets_client


def filter_active_shows(shows_df: pd.DataFrame) -> pd.DataFrame:
    """Filter for active shows if the active column exists.
    
    Args:
        shows_df: DataFrame with show information
        
    Returns:
        DataFrame with only active shows if active column exists,
        otherwise returns original DataFrame
    """
    if 'active' in shows_df.columns:
        return shows_df[shows_df['active']].copy()
    return shows_df


def get_all_studios(shows_df: pd.DataFrame) -> pd.Series:
    """Get all unique studios and their show counts.
    
    Args:
        shows_df: DataFrame with show data
        
    Returns:
        Series with studio names as index and show counts as values
    """
    # Explode studio_names to get one row per show-studio combination
    studio_shows = shows_df.explode('studio_names')
    studio_names = studio_shows['studio_names']
    studio_names = studio_names[studio_names.notna() & (studio_names != '')]
    
    # Filter out studios prefixed with 'Other:'
    studio_names = studio_names[~studio_names.str.startswith('Other:', na=False)]
    
    # Count each studio once per show by dropping duplicates
    studio_shows = pd.DataFrame({'studio_names': studio_names, 'title': studio_shows['title']})
    studio_shows = studio_shows.drop_duplicates()
    
    # Count occurrences of each studio
    return studio_shows['studio_names'].value_counts()

def get_shows_for_studio(shows_df: pd.DataFrame, studio: str, studio_categories_df: pd.DataFrame) -> pd.DataFrame:
    """Get all shows for a specific studio, handling multiple studios per show.
    
    Args:
        shows_df: DataFrame with show information
        studio: Studio name to match
        studio_categories_df: DataFrame with studio categories
        
    Returns:
        DataFrame containing only shows that include this studio
    """
    # Get active studios from categories
    active_studios = set(studio_categories_df[studio_categories_df['active']]['studio'].unique())
    
    # Only proceed if this studio is active
    if studio not in active_studios:
        return pd.DataFrame()
    
    # Explode studio_names to get one row per show-studio combination
    studio_shows = shows_df.explode('studio_names')
    
    # Filter for this studio
    studio_shows = studio_shows[studio_shows['studio_names'] == studio]
    
    # Drop duplicates to count each show once
    matching_shows = studio_shows.drop_duplicates(subset=['title'])
    
    return matching_shows

def analyze_studio_relationships(shows_df: pd.DataFrame, studio_categories_df: pd.DataFrame) -> Dict:
    """Analyze relationships between studios and networks.
    
    Args:
        shows_df: DataFrame with show information including:
            - studio_names: List of studios for each show
            - network_name: Network name
            - active: Whether the show is active
            - status_name: Show status (e.g. Active, Cancelled)
            - tmdb_seasons: Number of seasons
            - tmdb_total_episodes: Total episodes
        
    Returns:
        Dictionary containing:
        - studio_sizes: Number of shows per studio
        - network_relationships: Network distribution by studio
        - total_studios: Total number of unique studios
        - top_studios: List of studios sorted by show count
        - studio_success: Success metrics by studio
    """
    # Filter for active shows if column exists
    shows_df = filter_active_shows(shows_df)
    
    # Get active studios from categories
    active_studios = set(studio_categories_df[studio_categories_df['active']]['studio'].unique())
    
    # Get studio sizes by show count
    studio_sizes = {}
    for studio in active_studios:
        # Use get_shows_for_studio which handles filtering correctly
        studio_shows = get_shows_for_studio(shows_df, studio, studio_categories_df)
        if not studio_shows.empty:
            studio_sizes[studio] = len(studio_shows)
    
    # Get network relationships
    network_relationships = {}
    for studio in studio_sizes.keys():
        studio_shows = get_shows_for_studio(shows_df, studio, studio_categories_df)
        if not studio_shows.empty:
            network_relationships[studio] = studio_shows['network_name'].value_counts().to_dict()
    
    # Calculate success metrics by studio
    studio_success = {}
    for studio in studio_sizes.keys():
        studio_shows = get_shows_for_studio(shows_df, studio, studio_categories_df)
        if not studio_shows.empty:
            # Calculate success metrics
            avg_seasons = studio_shows['tmdb_seasons'].mean()
            avg_episodes = studio_shows['tmdb_total_episodes'].mean()
            
            # Get show status distribution
    # Get indie studios from active studios
    indie_studios = {}
    
    # Cache all studio shows with 2+ shows
    all_studio_shows = {}
    for studio in active_studios:
        studio_shows = get_shows_for_studio(shows_df, studio, studio_categories_df)
        if len(studio_shows) >= 2:
            all_studio_shows[studio] = studio_shows
    
    def is_indie_cat(cat):
        if isinstance(cat, list):
            # Check each element, splitting by comma if needed
            for x in cat:
                if any(part.strip().lower() == 'independent' for part in str(x).split(',')):
                    return True
            return False
        elif isinstance(cat, str):
            # Split by comma and check each part
            return any(part.strip().lower() == 'independent' for part in cat.split(','))
        return False
    
    # Then filter to active indie studios
    active_indies = set(studio_categories_df[
        studio_categories_df['active'] & 
        studio_categories_df['category'].apply(is_indie_cat)
    ]['studio'].unique())
    
    # Process only active indie studios that have 2+ shows
    for studio in active_indies:
        # Check if we have cached shows for this studio
        if studio in all_studio_shows:
            studio_shows = all_studio_shows[studio]
            # Get genres from filtered shows
            genres = []
            for _, show in studio_shows.iterrows():
                if 'genre_name' in show and show['genre_name']:
                    genres.extend([g.strip() for g in str(show['genre_name']).split(',')])
            
            indie_studios[studio] = {
                'show_count': len(studio_shows),
                'shows': studio_shows['title'].tolist(),
                'networks': studio_shows['network_name'].value_counts().to_dict(),
                'genres': list(set(genres))
            }
    
    # Convert studio_sizes to dict and get sorted studios
    studio_sizes_dict = {}
    for studio, count in studio_sizes.items():
        if isinstance(studio, str):  # Only include string keys
            studio_sizes_dict[studio] = count
    
    # Sort studios by show count
    sorted_studios = sorted(
        studio_sizes_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_studios = [studio for studio, _ in sorted_studios]
    
    return {
        'studio_sizes': studio_sizes_dict,
        'network_relationships': network_relationships,
        'total_studios': len(studio_sizes),
        'top_studios': sorted([studio for studio, _ in studio_sizes_dict.items()], 
                            key=lambda x: studio_sizes_dict[x], 
                            reverse=True),
        'indie_studios': indie_studios,
        'top_indies': indie_studios  # Add top_indies for success stories
    }

def get_studio_insights(shows_df: pd.DataFrame, studio: str, studio_categories_df: pd.DataFrame) -> Dict:
    """Get detailed insights for a specific studio.
    
    Args:
        shows_df: DataFrame with show information
        studio: Studio to analyze
        studio_categories_df: DataFrame with studio categories
        
    Returns:
        Dictionary containing:
        - network_partners: Network distribution
        - show_details: Basic show information
        - success_metrics: Success rate metrics
    """
    # Get shows for this studio
    studio_shows = get_shows_for_studio(shows_df, studio, studio_categories_df)
    if studio_shows.empty:
        return {
            'error': f'No shows found for studio: {studio}'
        }
        
    # Get network partners with show counts
    network_partners = studio_shows['network_name'].value_counts().to_dict()
        
    # Calculate success metrics
    success_metrics = {
        'total_shows': len(studio_shows),
        'avg_seasons': studio_shows['tmdb_seasons'].mean(),
        'avg_episodes': studio_shows['tmdb_total_episodes'].mean(),
        'status_distribution': studio_shows['status_name'].value_counts().to_dict()
    }
    
    # Add active metrics if column exists
    if 'active' in studio_shows.columns:
        success_metrics.update({
            'active_shows': len(studio_shows),  # All shows are active since we filtered at the start
            'active_percentage': 100.0
        })
        
    # Get basic show info
    show_details = []
    for _, show in studio_shows.iterrows():
        show_info = {
            'title': show['title'],
            'network_name': show['network_name'],  # Changed from 'network' to match standardized names
            'status': show['status_name'],
            'seasons': show['tmdb_seasons'],
            'episodes': show['tmdb_total_episodes'],
            'genre': show.get('genre_name', 'Unknown')  # Add genre which studio_view expects
        }
        if 'active' in show:
            show_info['active'] = show['active']
        if 'tmdb_last_air_date' in show:
            show_info['last_air_date'] = show['tmdb_last_air_date']
        show_details.append(show_info)
    
    # Get genre distribution
    genres = []
    for show in studio_shows.iterrows():
        if 'genre_name' in show[1] and show[1]['genre_name'] is not None:
            genres.extend([g.strip() for g in show[1]['genre_name'].split(',')])
    top_genres = pd.Series(genres).value_counts().to_dict() if genres else {}
    
    return {
        'network_partners': network_partners,
        'show_details': show_details,
        'success_metrics': success_metrics,
        'show_count': len(studio_shows),  # Add show count which studio_view expects
        'top_genres': top_genres  # Add genre distribution which studio_view expects
    }
