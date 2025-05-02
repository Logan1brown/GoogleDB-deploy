"""Tests for the CompView component."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from dataclasses import dataclass
import sys
from types import ModuleType

# Configure pytest marks
pytest.mark.integration = pytest.mark.integration

# Create mock modules
def setup_mock_module(name):
    mock_module = ModuleType(name)
    sys.modules[name] = mock_module
    return mock_module

# Mock required dependencies
mock_success = setup_mock_module('src.data_processing.success_analysis.success_analyzer')
mock_success.SuccessAnalyzer = MagicMock

mock_shows = setup_mock_module('src.data_processing.analyze_shows')
mock_shows.ShowsAnalyzer = MagicMock

mock_supabase = setup_mock_module('src.utils.supabase')
mock_supabase.get_client = MagicMock(return_value=MagicMock())

# Mock streamlit
mock_st = setup_mock_module('streamlit')
for attr in ['selectbox', 'multiselect', 'dataframe', 'markdown', 'expander', 
            'write', 'error', 'info', 'container', 'column', 'number_input',
            'slider', 'radio', 'checkbox', 'text_input', 'button']:
    setattr(mock_st, attr, MagicMock())

# Mock enter/exit for expander
mock_st.expander.return_value.__enter__ = MagicMock()
mock_st.expander.return_value.__exit__ = MagicMock()

# Import after mocking dependencies
from src.dashboard.components.comp_view import (
    get_field_name, get_field_names, get_id_for_name, get_ids_for_names,
    update_single_id_field, update_array_field, update_setting_field,
    update_production_field, update_match_details, create_results_df,
    render_criteria_section, render_results_section
)
from src.data_processing.comp_analysis.comp_analyzer import CompAnalyzer, CompScore

# Test data
MOCK_OPTIONS = [(1, "Option 1"), (2, "Option 2"), (3, "Option 3")]
MOCK_SCORES = {
    "genre_score": 5,
    "studio_score": 3,
    "team_score": 2
}

def test_get_field_name():
    """Test getting field name from options."""
    assert get_field_name(1, MOCK_OPTIONS) == "Option 1"
    assert get_field_name(4, MOCK_OPTIONS, "Not Found") == "Not Found"

def test_get_field_names():
    """Test getting multiple field names from options."""
    assert get_field_names([1, 3], MOCK_OPTIONS) == ["Option 1", "Option 3"]
    assert get_field_names([4], MOCK_OPTIONS, "Unknown") == ["Unknown"]

def test_get_id_for_name():
    """Test getting ID from name."""
    assert get_id_for_name("Option 1", MOCK_OPTIONS) == 1
    assert get_id_for_name("Invalid", MOCK_OPTIONS) is None

def test_get_ids_for_names():
    """Test getting multiple IDs from names."""
    assert get_ids_for_names(["Option 1", "Option 3"], MOCK_OPTIONS) == [1, 3]
    assert get_ids_for_names(["Invalid"], MOCK_OPTIONS) == []

def test_update_single_id_field():
    """Test updating single ID field details."""
    details = {}
    update_single_id_field(
        details, "genre", 1, 1, MOCK_OPTIONS, "genre_score"
    )
    assert details["genre"]["match"] == True
    assert details["genre"]["name1"] == "Option 1"
    assert details["genre"]["name2"] == "Option 1"

def test_update_array_field():
    """Test updating array field details."""
    details = {}
    show_ids = [1, 2]
    selected_ids = [1, 3]
    update_array_field(details, "characters", show_ids, selected_ids, MOCK_OPTIONS)
    assert details["characters"]["matches"] == ["Option 1"]
    assert details["characters"]["mismatches"] == ["Option 2"]
    assert details["characters"]["selected"] == True

def test_update_setting_field():
    """Test updating setting field details."""
    details = {}
    update_setting_field(
        details, 1, 2, 1, 2, MOCK_OPTIONS, MOCK_OPTIONS
    )
    assert details["setting"]["time"] == "Option 1"
    assert details["setting"]["location"] == "Option 2"
    assert details["setting"]["selected"] == True

def test_update_production_field():
    """Test updating production field details."""
    details = {}
    update_production_field(
        details, "studio", [1], [1], MOCK_OPTIONS, MOCK_SCORES, "studio_score"
    )
    assert details["studio"]["match"] == True
    assert details["studio"]["name1"] == "Option 1"
    assert details["studio"]["score"] == 3

def test_update_match_details():
    """Test updating all match details."""
    details = {}
    match = {
        "genre_id": 1,
        "character_type_ids": [1, 2],
        "time_setting_id": 1,
        "location_setting_id": 2,
        "studios": [1]
    }
    criteria = {
        "genre_id": 1,
        "character_type_ids": [1, 3],
        "time_setting_id": 1,
        "location_setting_id": 2,
        "studios": [1]
    }
    display_options = {
        "genre": MOCK_OPTIONS,
        "character_types": MOCK_OPTIONS,
        "time_setting": MOCK_OPTIONS,
        "location_setting": MOCK_OPTIONS,
        "studios": MOCK_OPTIONS
    }
    
    update_match_details(details, match, criteria, display_options, MOCK_SCORES)
    
    # Check genre details
    assert details.get("genre", {}).get("match") == True
    # Check character details
    assert len(details.get("characters", {}).get("matches", [])) == 1
    # Check setting details
    assert details.get("setting", {}).get("selected") == True
    # Check studio details
    assert details.get("studio", {}).get("score") == 3

@pytest.fixture
def mock_comp_analyzer():
    """Create a mock CompAnalyzer with test data."""
    analyzer = MagicMock(spec=CompAnalyzer)
    
    # Mock field options
    field_options = {
        'genre': [(1, 'Action'), (2, 'Comedy')],
        'subgenres': [(1, 'Action'), (2, 'Comedy')],  # Same as genre for testing
        'source_type': [(1, 'Original'), (2, 'Adaptation')],
        'character_types': [(1, 'Hero'), (2, 'Villain')],
        'plot_elements': [(1, 'Adventure'), (2, 'Mystery')],
        'thematic_elements': [(1, 'Friendship'), (2, 'Revenge')],
        'tone': [(1, 'Light'), (2, 'Dark')],
        'time_setting': [(1, 'Present'), (2, 'Future')],
        'location_setting': [(1, 'Earth'), (2, 'Space')],
        'network': [(1, 'Netflix'), (2, 'HBO')],
        'studios': [(1, 'Studio A'), (2, 'Studio B')],
        'order_type': [(1, 'Season'), (2, 'Special')]
    }
    analyzer.get_field_options.return_value = field_options
    
    # Mock display options
    analyzer.get_field_display_options = lambda field: analyzer.get_field_options()[field]
    
    # Mock find_by_criteria
    @dataclass
    class MockCompScore:
        total: float = 90
        content_score: float = 70
        production_score: float = 15
        format_score: float = 5
        
        def to_display_dict(self):
            return {
                'total': self.total,
                'content': self.content_score,
                'production': self.production_score,
                'format': self.format_score
            }
            
        def get_match_details(self):
            return {
                'genre': {'match': True},
                'characters': {'matches': ['Hero']},
                'plot': {'matches': ['Adventure']},
                'themes': {'matches': ['Friendship']},
                'setting': {'time': 'Present', 'location': 'Earth'}
            }
    
    analyzer.find_by_criteria.return_value = [
        {
            'title': 'Test Show 1',
            'success_score': 80,
            'comp_score': MockCompScore(),
            'genre_id': 1,
            'character_type_ids': [1],
            'plot_element_ids': [1],
            'thematic_element_ids': [1],
            'time_setting_id': 1,
            'location_setting_id': 1
        }
    ]
    
    return analyzer

@pytest.mark.integration
def test_render_criteria_section(mock_comp_analyzer):
    """Test rendering criteria section with analyzer data."""
    state = {'criteria': {}}
    
    # Mock all streamlit functions
    patches = [
        patch('streamlit.selectbox', return_value='Action'),
        patch('streamlit.multiselect', return_value=['Hero']),
        patch('streamlit.number_input', return_value=12),
        patch('streamlit.markdown'),
        patch('streamlit.expander'),
        patch('streamlit.write'),
        patch('streamlit.info'),
        patch('streamlit.error'),
        patch('streamlit.container'),
        patch('streamlit.column')
    ]
    
    # Start all patches
    for p in patches:
        p.start()
    
    try:
        render_criteria_section(mock_comp_analyzer, state)
        
        # Verify state updates
        assert state['criteria'].get('genre_id') == 1
        assert state['criteria'].get('character_types_ids') == [1]
        assert state['criteria'].get('episode_count') == 12
    finally:
        # Stop all patches
        for p in patches:
            p.stop()

@pytest.mark.integration
def test_render_results_section(mock_comp_analyzer):
    """Test rendering results section with analyzer data."""
    state = {'criteria': {'genre_id': 1, 'character_types_ids': [1]}}
    
    # Mock all streamlit functions
    patches = [
        patch('streamlit.dataframe'),
        patch('streamlit.markdown'),
        patch('streamlit.expander'),
        patch('streamlit.write'),
        patch('streamlit.info'),
        patch('streamlit.error'),
        patch('streamlit.container'),
        patch('streamlit.column'),
        patch('streamlit.empty'),
        patch('streamlit.spinner')
    ]
    
    # Start all patches
    for p in patches:
        p.start()
    
    try:
        # Reset mock before test
        mock_comp_analyzer.find_by_criteria.reset_mock()
        
        render_results_section(mock_comp_analyzer, state)
        
        # Verify analyzer calls
        assert mock_comp_analyzer.find_by_criteria.call_count == 1
        mock_comp_analyzer.find_by_criteria.assert_called_with(state['criteria'])
        
        # Verify DataFrame creation
        assert patches[0].return_value.call_count >= 1
        df_arg = patches[0].return_value.call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        assert len(df_arg) == 1
        assert df_arg.iloc[0]['Show'] == 'Test Show 1'
        assert df_arg.iloc[0]['Success'] == 80
        assert df_arg.iloc[0]['Total Score'] == 90
    finally:
        # Stop all patches
        for p in patches:
            p.stop()

def test_create_results_df():
    """Test creating results DataFrame."""
    @dataclass
    class MockCompScore:
        total: float = 90
        content_score: float = 70
        production_score: float = 15
        format_score: float = 5
    
    mock_results = [
        {
            'title': 'Show 1',
            'success_score': 80,
            'comp_score': MockCompScore()
        }
    ]
    
    df = create_results_df(mock_results)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]['Show'] == 'Show 1'
    assert df.iloc[0]['Success'] == 80
    assert df.iloc[0]['Total Score'] == 90
