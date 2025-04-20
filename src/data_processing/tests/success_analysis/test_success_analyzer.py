"""Tests for the SuccessAnalyzer component."""
import pytest
import pandas as pd
import numpy as np

from src.data_processing.success_analysis import SuccessAnalyzer, SuccessConfig


@pytest.fixture
def success_analyzer():
    """Create a SuccessAnalyzer with default config."""
    return SuccessAnalyzer()


@pytest.fixture
def sample_shows():
    """Create a sample DataFrame of shows."""
    return pd.DataFrame([
        # High success - 3 seasons + high episode volume
        {
            'tmdb_id': '1',
            'shows': 'Hit Show',
            'tmdb_status': 'Returning Series',
            'tmdb_seasons': 3,
            'tmdb_avg_eps': 12,  # Gets volume bonus
        },
        # Medium success - 2 seasons + meets min episodes
        {
            'tmdb_id': '2',
            'shows': 'Solid Show',
            'tmdb_status': 'Ended',
            'tmdb_seasons': 2,
            'tmdb_avg_eps': 9,  # Gets base points only
        },
        # Low success - 1 season + low episode count
        {
            'tmdb_id': '3',
            'shows': 'Failed Show',
            'tmdb_status': 'Canceled',
            'tmdb_seasons': 1,
            'tmdb_avg_eps': 6,  # Below minimum
        },
        # Unreliable - In development
        {
            'tmdb_id': '4',
            'shows': 'New Show',
            'tmdb_status': 'In Production',
            'tmdb_seasons': 1,
            'tmdb_avg_eps': None,
        }
    ])


def test_success_calculation(success_analyzer, sample_shows):
    """Test success score calculation for individual shows."""
    # Hit Show: 
    # - 3 seasons (40 + 20 = 60)
    # - High episodes (20 + 20 = 40)
    # - Returning bonus (100 * 1.2 = 120)
    hit_show = sample_shows.iloc[0]
    assert success_analyzer.calculate_success(hit_show) == 120

    # Solid Show:
    # - 2 seasons (40)
    # - Base episodes (20)
    # - No status modifier (60 * 1.0 = 60)
    solid_show = sample_shows.iloc[1]
    assert success_analyzer.calculate_success(solid_show) == 60

    # Failed Show:
    # - 1 season (0)
    # - No episode points (0)
    # - Canceled penalty (0 * 0.8 = 0)
    failed_show = sample_shows.iloc[2]
    assert success_analyzer.calculate_success(failed_show) == 0

    # Unreliable Show: Should return 0
    new_show = sample_shows.iloc[3]
    assert success_analyzer.calculate_success(new_show) == 0


def test_market_analysis(success_analyzer, sample_shows):
    """Test market-wide success analysis."""
    results = success_analyzer.analyze_market(sample_shows)
    
    # Check max and thresholds
    assert results['max_score'] == 120  # Hit Show score
    assert results['high_threshold'] == 120 * 0.8  # 80% of max
    assert results['medium_threshold'] == 120 * 0.5  # 50% of max
    
    # Check show tiers
    shows = results['shows']
    assert shows['1']['tier'] == 'high'  # Hit Show
    assert shows['2']['tier'] == 'medium'  # Solid Show
    assert shows['3']['tier'] == 'low'  # Failed Show
    assert '4' not in shows  # Unreliable show excluded


def test_custom_config():
    """Test SuccessAnalyzer with custom config."""
    config = SuccessConfig(
        SEASON2_VALUE=50,  # Higher value for S2
        ADDITIONAL_SEASON_VALUE=30,  # Higher value for additional seasons
        EPISODE_MIN_THRESHOLD=8,  # Lower episode threshold
        EPISODE_BONUS_THRESHOLD=10,  # Episodes needed for bonus
        EPISODE_BASE_POINTS=30,  # Higher base points
        EPISODE_BONUS_POINTS=30  # Higher bonus points
    )
    analyzer = SuccessAnalyzer(config)
    
    # Create a show that would fail with default config
    show = pd.Series({
        'tmdb_status': 'Returning Series',
        'tmdb_seasons': 3,
        'tmdb_avg_eps': 9  # Gets base points only
    })
    
    # With custom config:
    # - 3 seasons (50 + 30 = 80)
    # - 9 episodes (base points only = 30)
    # - Returning bonus (110 * 1.2 = 132)
    assert analyzer.calculate_success(show) == 132


def test_empty_market():
    """Test handling of empty market data."""
    analyzer = SuccessAnalyzer()
    empty_df = pd.DataFrame(columns=['tmdb_status', 'tmdb_seasons', 'tmdb_avg_eps'])
    
    results = analyzer.analyze_market(empty_df)
    assert results['max_score'] == 0
    assert results['high_threshold'] == 0
    assert results['medium_threshold'] == 0
    assert results['shows'] == {}
