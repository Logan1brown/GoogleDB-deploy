"""Integration tests for CompAnalyzer.

Tests the integration between CompAnalyzer, CompView, and ShowsAnalyzer,
focusing on field names, data types, and scoring calculations.
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from typing import Dict, List, Any

import sys
from unittest.mock import MagicMock

# Mock missing modules
mock_success = MagicMock()
sys.modules['src.data_processing.analyze_success'] = mock_success
mock_success.SuccessAnalyzer = MagicMock

mock_supabase = MagicMock()
sys.modules['src.utils.supabase'] = mock_supabase
mock_supabase.get_client = MagicMock()

# Now import our modules
from src.data_processing.comp_analysis.comp_analyzer import CompAnalyzer, CompScore
from src.data_processing.analyze_shows import ShowsAnalyzer

class TestCompAnalyzerIntegration(unittest.TestCase):
    """Test CompAnalyzer integration with CompView and ShowsAnalyzer."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data that will be reused across tests."""
        # Create mock show data
        cls.mock_show_data = pd.DataFrame({
            'id': [1, 2],
            'title': ['Show A', 'Show B'],
            'genre_id': [1, 1],
            'subgenres': [[1, 2], [2, 3]],
            'source_type_id': [1, 1],
            'character_type_ids': [[1, 2], [2, 3]],
            'plot_element_ids': [[1, 2], [2, 3]],
            'thematic_element_ids': [[1, 2], [2, 3]],
            'tone_id': [1, 2],
            'time_setting_id': [1, 2],
            'location_setting_id': [1, 2],
            'network_id': [1, 2],
            'studios': [[1, 2, 3], [2, 3, 4]],
            'team_member_ids': [[1, 2], [2, 3]],
            'episode_count': [12, 24],
            'order_type_id': [1, 2],
            'tmdb_avg_eps': [12.0, 24.0]
        })
        
        # Create mock reference data
        cls.mock_reference_data = {
            'character_types': pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Character A', 'Character B', 'Character C']
            }),
            'genre': pd.DataFrame({
                'id': [1, 2, 3],
                'genre': ['Genre A', 'Genre B', 'Genre C']
            }),
            'source_type': pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Source A', 'Source B', 'Source C']
            }),
            'location_setting': pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Location A', 'Location B', 'Location C']
            }),
            'network': pd.DataFrame({
                'id': [1, 2, 3],
                'network': ['Network A', 'Network B', 'Network C']
            }),
            'order_type': pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Order A', 'Order B', 'Order C']
            }),
            'plot_elements': pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Plot A', 'Plot B', 'Plot C']
            }),
            'studios': pd.DataFrame({
                'id': [1, 2, 3],
                'studio': ['Studio A', 'Studio B', 'Studio C']
            }),
            'thematic_elements': pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Theme A', 'Theme B', 'Theme C']
            }),
            'tone': pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Tone A', 'Tone B', 'Tone C']
            }),
            'time_setting': pd.DataFrame({
                'id': [1, 2, 3],
                'name': ['Time A', 'Time B', 'Time C']
            })
        }

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock ShowsAnalyzer
        self.shows_analyzer = MagicMock(spec=ShowsAnalyzer)
        self.shows_analyzer.fetch_comp_data.return_value = (
            self.mock_show_data,
            self.mock_reference_data
        )
        
        # Create CompAnalyzer with mock ShowsAnalyzer
        self.comp_analyzer = CompAnalyzer(shows_analyzer=self.shows_analyzer)
        self.comp_analyzer.initialize(force=True)

    def test_field_names(self):
        """Test that all required field names are present in comp data."""
        required_fields = {
            'genre_id', 'subgenres', 'source_type_id', 'character_type_ids',
            'plot_element_ids', 'thematic_element_ids', 'tone_id',
            'time_setting_id', 'location_setting_id', 'network_id',
            'studios', 'team_member_ids', 'episode_count', 'order_type_id'
        }
        
        missing_fields = required_fields - set(self.comp_analyzer.comp_data.columns)
        self.assertEqual(missing_fields, set(), f"Missing required fields: {missing_fields}")

    def test_array_field_types(self):
        """Test that array fields are properly converted to Python lists."""
        array_fields = [
            'subgenres', 'character_type_ids', 'plot_element_ids',
            'thematic_element_ids', 'studios', 'team_member_ids'
        ]
        
        for field in array_fields:
            with self.subTest(field=field):
                values = self.comp_analyzer.comp_data[field].values
                self.assertTrue(
                    all(isinstance(x, list) for x in values),
                    f"Field {field} contains non-list values"
                )

    def test_id_field_types(self):
        """Test that ID fields contain integers."""
        id_fields = [
            'genre_id', 'source_type_id', 'tone_id', 'time_setting_id',
            'location_setting_id', 'network_id', 'order_type_id'
        ]
        
        for field in id_fields:
            with self.subTest(field=field):
                values = self.comp_analyzer.comp_data[field].values
                self.assertTrue(
                    all(isinstance(x, (int, np.integer)) or pd.isna(x) for x in values),
                    f"Field {field} contains non-integer values"
                )

    def test_score_calculation(self):
        """Test that score calculations produce valid results."""
        # Test perfect match (same show)
        source = self.comp_analyzer.comp_data.iloc[0]
        score = self.comp_analyzer._calculate_score(source, source)
        
        self.assertIsInstance(score, CompScore)
        self.assertGreaterEqual(score.total(), 0)
        self.assertLessEqual(score.total(), 100)
        
        # Verify score components
        self.assertEqual(score.content_score(), 82)  # Perfect content match
        self.assertEqual(score.production_score(), 13)  # Perfect production match
        self.assertEqual(score.format_score(), 5)  # Perfect format match

    def test_field_options(self):
        """Test that field options are correctly formatted for CompView."""
        options = self.comp_analyzer.get_field_options()
        
        # Check all required fields are present
        required_fields = {
            'character_types', 'genre', 'subgenres', 'source_type',
            'plot_elements', 'thematic_elements', 'tone', 'time_setting',
            'location_setting', 'network', 'studios', 'order_type'
        }
        
        self.assertEqual(
            set(options.keys()),
            required_fields,
            "Missing required field options"
        )
        
        # Check option format
        for field, field_options in options.items():
            with self.subTest(field=field):
                # Options should be list of (id, name) tuples
                self.assertTrue(all(
                    isinstance(opt, tuple) and len(opt) == 2
                    for opt in field_options
                ))
                # First element should be int, second should be str
                self.assertTrue(all(
                    isinstance(opt[0], int) and isinstance(opt[1], str)
                    for opt in field_options
                ))

    def test_comparison_details(self):
        """Test that comparison details are correctly formatted for CompView."""
        details = self.comp_analyzer.get_comparison_details(1, 2)
        
        # Check structure
        required_keys = {'scores', 'match_details', 'source', 'target'}
        self.assertEqual(set(details.keys()), required_keys)
        
        # Check scores
        self.assertIsInstance(details['scores']['total'], float)
        self.assertIsInstance(details['scores']['content'], float)
        self.assertIsInstance(details['scores']['production'], float)
        self.assertIsInstance(details['scores']['format'], float)
        
        # Check match details structure
        match_types = {'content', 'production', 'format'}
        self.assertEqual(set(details['match_details'].keys()), match_types)
        
        # Check source/target structure
        for show in ['source', 'target']:
            self.assertIn('id', details[show])
            self.assertIn('title', details[show])
            self.assertIn('fields', details[show])
            
            # Check field values are lists of strings
            for field_values in details[show]['fields'].values():
                self.assertIsInstance(field_values, list)
                self.assertTrue(all(isinstance(v, str) for v in field_values))

if __name__ == '__main__':
    unittest.main()
