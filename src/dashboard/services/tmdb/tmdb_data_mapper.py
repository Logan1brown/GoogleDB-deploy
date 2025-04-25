"""Map TMDB data to our database schema.

This module handles the mapping of TMDB show data to our internal database schema,
following the rules defined in the TMDB integration documentation.
"""

from typing import Dict, List, Optional
from .tmdb_models import TVShowDetails

def map_tmdb_status(tmdb_status: str) -> Optional[int]:
    """Map TMDB status to our status_id.
    
    Only returns a value if our status_id field is empty.
    
    Args:
        tmdb_status: Status from TMDB API
        
    Returns:
        Our status_id value or None if no mapping
    """
    status_map = {
        'Returning Series': 1,  # active
        'Pilot': 1,             # active
        'Ended': 3,             # ended
        'Canceled': 2,          # canceled
        'Planned': 4,           # development
        # Note: In Production maps to development per spec
        'In Production': 4      # development
    }
    return status_map.get(tmdb_status)

def map_order_type(tmdb_status: str, episodes_per_season: List[int]) -> Optional[int]:
    """Map TMDB data to our order_type_id.
    
    Only returns a value if our order_type_id field is empty.
    
    Args:
        tmdb_status: Status from TMDB API
        episodes_per_season: List of episode counts per season
        
    Returns:
        Our order_type_id value or None if no mapping
    """
    if not episodes_per_season:
        return None
        
    # If it has multiple seasons, it's ongoing regardless of status
    if len(episodes_per_season) > 1:
        return 1  # ongoing
    # If it has one season and is ended, it's limited
    elif len(episodes_per_season) == 1 and tmdb_status == 'Ended':
        return 2  # limited
        
    return None

def get_episode_count(episodes_per_season: List[int]) -> Optional[int]:
    """Get episode count from first season.
    
    This is always written as it's factual data.
    
    Args:
        episodes_per_season: List of episode counts per season
        
    Returns:
        Episode count from first season or None if no seasons
    """
    if episodes_per_season and len(episodes_per_season) > 0:
        return episodes_per_season[0]
    return None

def get_episodes_per_season(tmdb_show: TVShowDetails) -> List[int]:
    """Calculate episodes per season from TMDB show data.
    
    Args:
        tmdb_show: Show details from TMDB
        
    Returns:
        List of episode counts per season, excluding specials
    """
    episodes_per_season = []
    for season in tmdb_show.seasons:
        if season.season_number > 0:  # Skip season 0 (specials)
            if season.episode_count:
                episodes_per_season.append(season.episode_count)
    return episodes_per_season

def map_tmdb_success_metrics(
    tmdb_show: TVShowDetails
) -> Dict[str, any]:
    """Map TMDB show data to our tmdb_success_metrics table.
    
    This stores all raw TMDB data (except executive producers and networks)
    that we might want to use later.
    
    Args:
        tmdb_show: Show details from TMDB
        
    Returns:
        Dict of fields to insert into tmdb_success_metrics table
    """
    # Calculate episode metrics
    episodes_per_season = get_episodes_per_season(tmdb_show)
    total_episodes = 0
    average_episodes = None
    
    if episodes_per_season:
        total_episodes = sum(episodes_per_season)
        average_episodes = round(total_episodes / len(episodes_per_season), 2)
    
    return {
        "tmdb_id": tmdb_show.id,
        "seasons": tmdb_show.number_of_seasons,
        "episodes_per_season": episodes_per_season,
        "total_episodes": total_episodes or None,
        "average_episodes": average_episodes,
        "status": tmdb_show.status,
        "last_air_date": tmdb_show.last_air_date.isoformat() if tmdb_show.last_air_date else None
    }

def map_tmdb_show_data(
    tmdb_show: TVShowDetails,
    existing_show: Dict
) -> Dict[str, any]:
    """Map TMDB show data to our database schema.
    
    Args:
        tmdb_show: Show details from TMDB
        existing_show: Existing show data from our database
        
    Returns:
        Dict of fields to update in our shows table
    """
    updates = {}
    
    # Calculate episodes per season
    episodes_per_season = get_episodes_per_season(tmdb_show)
    
    # Always set tmdb_id as this is the core mapping field
    updates['tmdb_id'] = tmdb_show.id
    
    # Only map status if current status_id is empty
    if not existing_show.get('status_id'):
        status = map_tmdb_status(tmdb_show.status)
        if status:
            updates['status_id'] = status
    
    # Only map order type if current order_type_id is empty
    if not existing_show.get('order_type_id'):
        order_type = map_order_type(
            tmdb_show.status,
            episodes_per_season
        )
        if order_type:
            updates['order_type_id'] = order_type
    
    # Only map description if current description is empty
    if not existing_show.get('description') and tmdb_show.overview:
        updates['description'] = tmdb_show.overview
    
    # Always map episode count as it's factual data
    episode_count = get_episode_count(episodes_per_season)
    if episode_count:
        updates['episode_count'] = episode_count
    
    return updates
