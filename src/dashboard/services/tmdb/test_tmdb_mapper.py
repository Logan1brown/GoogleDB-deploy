"""Test TMDB data mapping logic with real API data."""

import os
import pytest
from datetime import datetime
from .tmdb_client import TMDBClient
from .tmdb_data_mapper import map_tmdb_success_metrics, map_tmdb_show_data

# Test shows with different statuses and episode patterns
TEST_SHOWS = {
    "stranger_things": 66732,  # Multi-season, Returning Series
    "house_of_the_dragon": 94997,  # Single season, Returning Series
    "chernobyl": 87108,  # Single season, Ended
    "ms_marvel": 92782,  # Single season, status varies
    "the_last_of_us": 100088  # Recent show, good for testing
}

TMDB_API_KEY = '8ceb057922d27a34c69e4d3efb8442d2'

@pytest.fixture
def tmdb_client():
    """Get TMDB client."""
    return TMDBClient(api_key=TMDB_API_KEY)

def test_success_metrics_mapping(tmdb_client):
    """Test mapping TMDB data to success metrics with real show data."""
    # Test with Stranger Things (multi-season show)
    details = tmdb_client.get_tv_show_details(TEST_SHOWS["stranger_things"])
    metrics = map_tmdb_success_metrics(details)
    
    # Verify all required fields are present
    assert all(key in metrics for key in [
        'tmdb_id', 'seasons', 'episodes_per_season', 
        'total_episodes', 'average_episodes', 'status', 
        'last_air_date'
    ])
    
    # Verify episode calculations
    assert len(metrics['episodes_per_season']) > 0
    assert metrics['total_episodes'] == sum(metrics['episodes_per_season'])
    assert metrics['average_episodes'] == round(
        metrics['total_episodes'] / len(metrics['episodes_per_season']), 
        2
    )
    
    # Verify date formatting
    if metrics['last_air_date']:
        # Should be ISO format string
        datetime.fromisoformat(metrics['last_air_date'])

def test_show_data_mapping(tmdb_client):
    """Test mapping TMDB data to shows table with real show data."""
    # Test with different show statuses
    for show_name, show_id in TEST_SHOWS.items():
        details = tmdb_client.get_tv_show_details(show_id)
        
        # Test with empty existing show
        empty_show = {}
        updates = map_tmdb_show_data(details, empty_show)
        
        # Should always set these
        assert 'tmdb_id' in updates
        assert updates['tmdb_id'] == show_id
        
        if len(details.seasons) > 0:
            assert 'episode_count' in updates
            assert isinstance(updates['episode_count'], int)
        
        # Test with existing data
        full_show = {
            'status': 'active',
            'order_type': 'ongoing',
            'description': 'Existing description',
            'episode_count': 10
        }
        updates = map_tmdb_show_data(details, full_show)
        
        # Should not override existing data except episode_count
        assert 'status' not in updates
        assert 'order_type' not in updates
        assert 'description' not in updates
        if len(details.seasons) > 0:
            assert 'episode_count' in updates

def test_status_mapping(tmdb_client):
    """Test specific status mapping rules with real show data."""
    # Get a show we know is returning
    details = tmdb_client.get_tv_show_details(TEST_SHOWS["stranger_things"])
    empty_show = {}
    updates = map_tmdb_show_data(details, empty_show)
    
    # Should map to active
    if details.status == "Returning Series":
        assert updates.get('status') == 'active'
    
    # Get a completed limited series
    details = tmdb_client.get_tv_show_details(TEST_SHOWS["chernobyl"])
    updates = map_tmdb_show_data(details, empty_show)
    
    # Should map to ended
    if details.status == "Ended":
        assert updates.get('status') == 'ended'

def test_order_type_mapping(tmdb_client):
    """Test order type mapping rules with real show data."""
    # Multi-season returning show should be ongoing
    details = tmdb_client.get_tv_show_details(TEST_SHOWS["stranger_things"])
    empty_show = {}
    updates = map_tmdb_show_data(details, empty_show)
    
    if (details.status == "Returning Series" and 
        len(details.seasons) > 1):
        assert updates.get('order_type') == 'ongoing'
    
    # Single season ended show should be limited
    details = tmdb_client.get_tv_show_details(TEST_SHOWS["chernobyl"])
    updates = map_tmdb_show_data(details, empty_show)
    
    if (details.status == "Ended" and 
        len(details.seasons) == 1):
        assert updates.get('order_type') == 'limited'
