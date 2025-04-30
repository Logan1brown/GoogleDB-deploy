"""Tests for the CompAnalyzer."""

import pytest
import pandas as pd
from src.data_processing.analyze_shows import ShowsAnalyzer
from src.data_processing.comp_analysis import CompAnalyzer


def test_comp_analyzer_integration():
    """Test CompAnalyzer integration with ShowsAnalyzer."""
    
    # Initialize analyzers
    shows_analyzer = ShowsAnalyzer()
    comp_analyzer = CompAnalyzer(shows_analyzer)
    
    # Fetch comp data
    comp_data = comp_analyzer.fetch_comp_data()
    assert isinstance(comp_data, pd.DataFrame)
    
    # Test getting similar shows
    show_id = comp_data.iloc[0]['id']  # Get first show ID
    similar_shows = comp_analyzer.get_similar_shows(show_id, limit=5)
    
    assert len(similar_shows) <= 5
    for show_id, score in similar_shows:
        assert isinstance(show_id, int)
        assert isinstance(score.total, float)
        assert 0 <= score.total <= 86  # Max possible score
        
        # Check score breakdowns
        assert 0 <= score.content_score <= 70
        assert 0 <= score.production_score <= 13
        assert 0 <= score.format_score <= 3
        
        # Individual component checks
        assert 0 <= score.genre_base <= 9
        assert 0 <= score.genre_overlap <= 8
        assert 0 <= score.source_type <= 8
        assert 0 <= score.character_types <= 14
        assert 0 <= score.plot_elements <= 12
        assert 0 <= score.theme_elements <= 13
        assert 0 <= score.tone <= 8
        assert 0 <= score.time_setting <= 4
        assert 0 <= score.location <= 3
        assert 0 <= score.network <= 5
        assert 0 <= score.studio <= 3
        assert 0 <= score.team <= 5
        assert 0 <= score.episodes <= 2
        assert 0 <= score.order_type <= 1


def test_get_field_options():
    """Test getting field options for dropdowns."""
    # Initialize analyzer
    comp_analyzer = CompAnalyzer()
    
    # Time first call (includes data fetch)
    import time
    start = time.time()
    options = comp_analyzer.get_field_options()
    first_call = time.time() - start
    
    # Time cached call
    start = time.time()
    cached_options = comp_analyzer.get_field_options()
    cached_call = time.time() - start
    
    # Time forced refresh
    start = time.time()
    refresh_options = comp_analyzer.get_field_options(force=True)
    refresh_call = time.time() - start
    
    # Print timing results
    print("\n" + "=" * 50)
    print("PERFORMANCE RESULTS:")
    print(f"First call (with fetch): {first_call:.3f}s")
    print(f"Cached call: {cached_call:.3f}s")
    print(f"Forced refresh: {refresh_call:.3f}s")
    print("=" * 50)
    
    # Check that we have all expected fields
    expected_fields = {
        # Content fields
        'genres', 'subgenres', 'source_types', 'character_types',
        'plot_elements', 'thematic_elements', 'tones', 'time_settings',
        'locations',
        # Production fields
        'networks', 'studios',
        # Format fields
        'order_types'
    }
    assert set(options.keys()) == expected_fields
    
    # Check that each field has valid values
    for field, values in options.items():
        assert isinstance(values, list)
        assert len(values) > 0  # Each field should have at least one option
        assert all(v is not None for v in values)  # No None values
        
        # Check that list is sorted
        assert values == sorted(values)
        
        # For array fields, check they're properly flattened
        if field in {'subgenres', 'character_types', 'plot_elements', 
                    'thematic_elements', 'studios'}:
            # No nested lists
            assert all(not isinstance(v, list) for v in values)
            # No duplicates
            assert len(values) == len(set(values))
    
    # Verify cached results match
    assert cached_options == options
    assert refresh_options == options
