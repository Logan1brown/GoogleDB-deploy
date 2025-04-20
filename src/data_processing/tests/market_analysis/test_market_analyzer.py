"""Tests for MarketAnalyzer."""
import pandas as pd
import pytest

from src.data_processing.market_analysis.market_analyzer import MarketAnalyzer
from src.data_processing.success_analysis.success_analyzer import SuccessConfig


def test_market_analyzer():
    """Test basic market analyzer functionality."""
    # Create test data
    shows_df = pd.DataFrame([
        {
            'title': 'Hit Show',
            'network': 'Netflix',
            'studio': 'Warner Bros',
            'tmdb_id': '1',
            'tmdb_status': 'Returning Series',
            'tmdb_seasons': 3,
            'tmdb_avg_eps': 12,  # Gets volume bonus
        },
        {
            'title': 'Solid Show',
            'network_name': 'Hulu',
            'studio_names': ['Disney'],
            'tmdb_id': '2',
            'tmdb_status': 'Ended',
            'tmdb_seasons': 2,
            'tmdb_avg_eps': 9,  # Gets base points
        },
        {
            'title': 'Failed Show',
            'network_name': 'Amazon',
            'studio_names': ['Sony'],
            'tmdb_id': '3',
            'tmdb_status': 'Canceled',
            'tmdb_seasons': 1,
            'tmdb_avg_eps': 6,  # Below minimum
        }
    ])
    
    # Initialize analyzer
    analyzer = MarketAnalyzer(titles_df=titles_df)
    
    # Check success scores were calculated
    assert 'success_score' in analyzer.titles_df.columns
    assert len(analyzer.titles_df[analyzer.titles_df['success_score'].notna()]) == 3
    
    # Test network distribution
    networks = analyzer.get_network_distribution()
    assert len(networks) == 3
    assert all(n in networks.index for n in ['Netflix', 'Hulu', 'Amazon'])
    
    # Test success by network
    success_by_network = analyzer.get_success_by_network()
    assert len(success_by_network) == 3
    assert success_by_network['Netflix'] > success_by_network['Amazon']


def test_custom_success_config():
    """Test market analyzer with custom success config."""
    shows_df = pd.DataFrame([
        {
            'shows': 'Test Show',
            'network': 'Netflix',
            'studio': 'Warner Bros',
            'tmdb_id': '4',
            'tmdb_status': 'Returning Series',
            'tmdb_seasons': 3,
            'tmdb_avg_eps': 9,
        }
    ])
    
    # Use custom config with higher points
    config = SuccessConfig(
        EPISODE_BASE_POINTS=30,
        EPISODE_BONUS_POINTS=30
    )
    
    analyzer = MarketAnalyzer(shows_df, success_config=config)
    assert analyzer.shows_df['success_score'].iloc[0] > 100  # Higher score with custom config


def test_missing_columns():
    """Test error handling for missing columns."""
    shows_df = pd.DataFrame([
        {
            'name': 'Test Show',
            'network': 'Netflix',  # Missing other required columns
        }
    ])
    
    with pytest.raises(ValueError) as exc:
        MarketAnalyzer(shows_df)
    assert "Missing required columns" in str(exc.value)
