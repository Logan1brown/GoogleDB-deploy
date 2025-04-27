"""Tests for show detail analyzer."""

import pytest
import pandas as pd
from typing import Dict, List

from src.data_processing.show_detail.show_detail_analyzer import ShowDetailAnalyzer, SimilarShow, NetworkAnalysis
from src.data_processing.success_analysis.success_analyzer import SuccessAnalyzer


def create_test_show(
    show_id: int,
    title: str,
    description: str = "A test show",
    announced_date: str = '2023-01-01',
    episode_count: int = None,
    tmdb_id: int = None,
    network_name: str = "Netflix",
    network_type: str = "Streaming",
    parent_company: str = "Netflix Inc.",
    status_name: str = "In Development",
    genre_name: str = "Drama",
    subgenres: List[str] = None,
    source_name: str = "Original",
    order_name: str = "Direct-to-Series",
    tmdb_seasons: int = None,
    tmdb_total_episodes: int = None,
    tmdb_status: str = "Returning Series",
    tmdb_last_air_date: str = None,
    studios: List[str] = None,
    team_members: List[Dict] = None,
    success_score: float = None
) -> pd.Series:
    """Create a test show with default values."""
    return pd.Series({
        'show_id': show_id,
        'title': title,
        'description': description,
        'announced_date': pd.to_datetime(announced_date),
        'episode_count': episode_count,
        'tmdb_id': tmdb_id,
        'network_name': network_name,
        'network_type': network_type,
        'parent_company': parent_company,
        'status_name': status_name,
        'genre_name': genre_name,
        'subgenres': subgenres or [],
        'source_name': source_name,
        'order_name': order_name,
        'tmdb_seasons': tmdb_seasons,
        'tmdb_total_episodes': tmdb_total_episodes,
        'tmdb_status': tmdb_status,
        'tmdb_last_air_date': pd.to_datetime(tmdb_last_air_date) if tmdb_last_air_date else None,
        'studios': studios or [],
        'team_members': team_members or [],
        'success_score': success_score
    })


def test_search_shows():
    """Test show search functionality."""
    analyzer = ShowDetailAnalyzer()
    
    # Create test data
    test_shows = pd.DataFrame([
        create_test_show(1, "Stranger Things", network_name="Netflix", network_type="Streaming", parent_company="Netflix Inc.", success_score=90),
        create_test_show(2, "The Stranger", network_name="HBO", network_type="Premium Cable", parent_company="Warner Bros. Discovery", success_score=80),
        create_test_show(3, "Strange New Worlds", network_name="Paramount+", network_type="Streaming", parent_company="Paramount Global", success_score=85),
        create_test_show(4, "Normal Show", network_name="ABC", network_type="Broadcast", parent_company="Disney", success_score=70)
    ])
    
    # Mock the data loading
    analyzer._shows_df = test_shows
    
    # Test exact match
    results = analyzer.search_shows("Stranger Things")
    assert len(results) == 1
    assert results[0]['title'] == "Stranger Things"
    assert results[0]['network_name'] == "Netflix"
    
    # Test partial match
    results = analyzer.search_shows("Strange")
    assert len(results) == 3
    assert all('Strange' in r['title'] for r in results)
    
    # Test case insensitive
    results = analyzer.search_shows("stranger")
    assert len(results) == 2
    
    # Test no matches
    results = analyzer.search_shows("xyz")
    assert len(results) == 0
    
    # Test limit
    results = analyzer.search_shows("Strange", limit=2)
    assert len(results) == 2


def test_get_show_details():
    """Test getting detailed show information."""
    analyzer = ShowDetailAnalyzer()
    
    # Create test data
    test_shows = pd.DataFrame([
        create_test_show(
            show_id=1,
            title="Test Show",
            genre_name="Drama",
            subgenres=["Crime", "Thriller"],
            network_name="Netflix",
            source_name="Original",
            announced_date='2023-01-01',
            team_members=[{"name": "Alice", "role": "Creator"}, {"name": "Bob", "role": "Executive Producer"}],
            success_score=85,
            tmdb_status="Returning Series",
            episode_count=10,
            tmdb_seasons=1,
            tmdb_total_episodes=10
        )
    ])
    
    # Mock the data loading
    analyzer._shows_df = test_shows
    
    # Test existing show
    details = analyzer.get_show_details(1)
    assert details is not None
    assert details['show_id'] == 1
    assert details['title'] == "Test Show"
    assert details['network_name'] == "Netflix"
    assert details['genre_name'] == "Drama"
    assert "Crime" in details['subgenres']
    assert details['source_name'] == "Original"
    assert details['success_score'] == 85
    assert details['tmdb_status'] == "Returning Series"
    assert details['episode_count'] == 10
    assert details['tmdb_seasons'] == 1
    assert details['tmdb_total_episodes'] == 10
    assert len(details['team_members']) == 2
    
    # Test non-existent show
    details = analyzer.get_show_details(999)
    assert details is None


def test_find_similar_shows():
    """Test finding similar shows from a dataset."""
    analyzer = ShowDetailAnalyzer()
    
    # Create test data
    test_shows = pd.DataFrame([
        create_test_show(
            show_id=1,
            title="Main Show",
            genre_name="Drama",
            subgenres=["Crime"],
            team_members=[{"name": "Alice", "role": "Creator"}, {"name": "Bob", "role": "Executive Producer"}],
            announced_date='2023-01-01'
        ),
        create_test_show(
            show_id=2,
            title="Similar Drama",
            genre_name="Drama",
            subgenres=["Crime"],
            team_members=[{"name": "Alice", "role": "Creator"}],
            success_score=80,
            announced_date='2023-01-01'
        ),
        create_test_show(
            show_id=3,
            title="Different Show",
            genre_name="Comedy",
            subgenres=["Sitcom"],
            team_members=[{"name": "Charlie", "role": "Creator"}],
            success_score=60,
            announced_date='2020-06-15'
        ),
        create_test_show(
            show_id=4,
            title="Another Drama",
            genre_name="Drama",
            team_members=[{"name": "Bob", "role": "Executive Producer"}],
            success_score=70,
            announced_date='2022-03-30'
        )
    ])
    
    # Mock the data loading
    analyzer._shows_df = test_shows
    
    similar_shows = analyzer.find_similar_shows(1)
    
    assert len(similar_shows) > 0
    assert similar_shows[0].show_id == 2  # Most similar show (same genre, subgenre, team member, year)
    assert similar_shows[-1].show_id == 4  # Less similar (same genre but no subgenre, one team member)
    assert all(s.match_score['total'] >= 30 for s in similar_shows)
    
    # Test similarity scores
    most_similar = next(s for s in similar_shows if s.show_id == 2)
    assert most_similar.match_score['genre_score'] == 40  # Same genre (35) + subgenre (5)
    assert most_similar.match_score['team_score'] == 10  # One team member
    assert most_similar.match_score['source_score'] == 20  # Same source
    assert most_similar.match_score['date_score'] == 10  # Same year


def test_analyze_network_patterns():
    """Test network pattern analysis."""
    analyzer = ShowDetailAnalyzer()
    
    similar_shows = [
        SimilarShow(show_id=1, title="Show 1", network_name="Netflix", success_score=80, match_score={'total': 90}),
        SimilarShow(show_id=2, title="Show 2", network_name="Netflix", success_score=60, match_score={'total': 85}),
        SimilarShow(show_id=3, title="Show 3", network_name="HBO", success_score=90, match_score={'total': 80}),
        SimilarShow(show_id=4, title="Show 4", network_name="Hulu", success_score=70, match_score={'total': 75})
    ]
    
    analysis = analyzer.analyze_network_patterns(similar_shows)
    
    assert analysis.similar_show_counts["Netflix"] == 2
    assert analysis.similar_show_counts["HBO"] == 1
    assert analysis.similar_show_counts["Hulu"] == 1
    assert analysis.success_rates["Netflix"] == 70  # (80 + 60) / 2
    assert analysis.success_rates["HBO"] == 90


@pytest.fixture
def mock_supabase(mocker):
    """Mock Supabase client for testing."""
    mock_client = mocker.Mock()
    mock_table = mocker.Mock()
    mock_select = mocker.Mock()
    mock_execute = mocker.Mock()
    
    # Set up the mock chain
    mock_client.table.return_value = mock_table
    mock_table.select.return_value = mock_select
    mock_select.execute.return_value = mock_execute
    
    # Mock the get_client function
    mocker.patch('src.data_processing.show_detail.show_detail_analyzer.get_client', return_value=mock_client)
    
    return mock_execute


def test_fetch_show_data():
    """Test fetching show data from api_show_details."""
    analyzer = ShowDetailAnalyzer()

    # Test successful fetch
    shows_df = analyzer.fetch_show_data()
    
    # Check that we got data back
    assert len(shows_df) > 0
    
    # Check that all required columns are present
    required_columns = [
        'show_id', 'title', 'network_name', 'announced_date', 'genre_name',
        'subgenres', 'source_name', 'team_members', 'tmdb_status',
        'episode_count', 'success_score'
    ]
    print("Actual columns:", shows_df.columns.tolist())
    print("Missing columns:", [col for col in required_columns if col not in shows_df.columns])
    assert all(col in shows_df.columns for col in required_columns)
    
    # Check data types
    assert shows_df['show_id'].dtype == 'int64'
    assert shows_df['title'].dtype == 'object'
    assert shows_df['network_name'].dtype == 'object'
    assert shows_df['genre_name'].dtype == 'object'
    assert shows_df['source_name'].dtype == 'object'
    assert shows_df['tmdb_status'].dtype == 'object'
    assert shows_df['success_score'].dtype == 'float64'


def test_ensure_data_loaded(mocker):
    """Test data loading behavior."""
    analyzer = ShowDetailAnalyzer()
    
    # Mock fetch_show_data
    mock_fetch = mocker.patch.object(
        analyzer,
        'fetch_show_data',
        return_value=pd.DataFrame([{'id': 1, 'title': 'Test Show'}])
    )
    
    # Test initial load
    assert analyzer._shows_df is None
    analyzer._ensure_data_loaded()
    assert analyzer._shows_df is not None
    mock_fetch.assert_called_once()
    
    # Test cached data
    mock_fetch.reset_mock()
    analyzer._ensure_data_loaded()
    mock_fetch.assert_not_called()
    
    # Test forced reload
    analyzer._ensure_data_loaded(force=True)
    mock_fetch.assert_called_once()
