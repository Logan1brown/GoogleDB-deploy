"""Configuration for Show Optimizer scoring and analysis.

This module defines all configurable parameters for the Show Optimizer system,
including scoring weights, confidence thresholds, performance settings, and
suggestion impact thresholds.

Key configuration categories:

1. Scoring Weights:
   - Criteria weights (genre, source type, etc.)
   - Component weights (audience, critics, longevity)
   - Network compatibility parameters

2. Confidence Settings:
   - Sample size thresholds for confidence levels
   - Degradation factors for partial criteria

3. Performance Settings:
   - Caching parameters
   - Incremental calculation thresholds

4. Suggestion Impact Settings:
   - Impact thresholds for generating suggestions
   - Prioritization weights
"""

from typing import Dict, Any


class OptimizerConfig:
    """Configuration for Show Optimizer scoring and analysis."""
    
    # Scoring weights for different criteria
    SCORING_WEIGHTS = {
        'criteria': {
            'genre': 15,
            'source_type': 10,
            'character_types': 12,
            'plot_elements': 10,
            'thematic_elements': 10,
            'tone': 9,
            'time_setting': 4,
            'location_setting': 4,
            'network': 5,
            'studios': 4,
            'team_members': 5,
            'order_type': 1
        },
        'components': {
            'audience': 0.4,  # 40% of total score
            'critics': 0.3,   # 30% of total score
            'longevity': 0.3  # 30% of total score
        },
        'network_compatibility': {
            'content_match': 0.7,     # 70% of network compatibility score
            'success_history': 0.3    # 30% of network compatibility score
        }
    }
    
    # Confidence thresholds for sample sizes
    CONFIDENCE = {
        'minimum_sample': 5,           # Minimum shows needed for any analysis
        'low_confidence': 10,          # Below this is low confidence
        'medium_confidence': 25,       # Below this is medium confidence
        'high_confidence': 50,         # Above this is high confidence
        'degradation_factor': 0.9      # Factor to reduce confidence per missing criteria
    }
    
    # Performance settings
    PERFORMANCE = {
        'cache_duration': 3600,        # Cache duration in seconds (1 hour)
        'max_cached_combinations': 100, # Maximum number of cached criteria combinations
        'incremental_threshold': 0.7    # Threshold for incremental vs full recalculation
    }
    
    # Suggestion impact settings
    SUGGESTIONS = {
        'minimum_impact': 0.05,        # Minimum impact score to generate suggestion (5%)
        'high_impact_threshold': 0.15, # Threshold for high impact suggestions (15%)
        'max_suggestions': 5,          # Maximum suggestions to return
        'network_specific_ratio': 0.6   # Ratio of network-specific to general suggestions
    }
    
    # Threshold values for various comparisons
    THRESHOLDS = {
        'significant_difference': 0.1,  # 10% difference is considered significant
        'major_difference': 0.2,        # 20% difference is considered major
        'minimum_compatibility': 0.3,   # Minimum compatibility score for recommendations
        'strong_compatibility': 0.7     # Strong compatibility threshold
    }
    
    # Complementary criteria that work well together
    COMPLEMENTARY_CRITERIA = {
        'genre': {
            # Comedy
            '3': [
                {'criteria_type': 'tone', 'criteria_value': 1, 'impact_score': 0.1, 'name': 'Light-hearted'},
                {'criteria_type': 'plot_elements', 'criteria_value': 12, 'impact_score': 0.08, 'name': 'Fish Out of Water'},
                {'criteria_type': 'character_types', 'criteria_value': 5, 'impact_score': 0.07, 'name': 'Quirky Ensemble'}
            ],
            # Crime
            '4': [
                {'criteria_type': 'tone', 'criteria_value': 3, 'impact_score': 0.12, 'name': 'Gritty'},
                {'criteria_type': 'plot_elements', 'criteria_value': 3, 'impact_score': 0.09, 'name': 'Investigation'},
                {'criteria_type': 'character_types', 'criteria_value': 2, 'impact_score': 0.08, 'name': 'Troubled Detective'}
            ],
            # Drama
            '6': [
                {'criteria_type': 'tone', 'criteria_value': 5, 'impact_score': 0.11, 'name': 'Emotional'},
                {'criteria_type': 'thematic_elements', 'criteria_value': 7, 'impact_score': 0.09, 'name': 'Family Dynamics'},
                {'criteria_type': 'character_types', 'criteria_value': 8, 'impact_score': 0.08, 'name': 'Complex Protagonist'}
            ]
        },
        'tone': {
            # Light-hearted
            '1': [
                {'criteria_type': 'genre', 'criteria_value': 3, 'impact_score': 0.1, 'name': 'Comedy'},
                {'criteria_type': 'plot_elements', 'criteria_value': 12, 'impact_score': 0.08, 'name': 'Fish Out of Water'}
            ],
            # Gritty
            '3': [
                {'criteria_type': 'genre', 'criteria_value': 4, 'impact_score': 0.12, 'name': 'Crime'},
                {'criteria_type': 'plot_elements', 'criteria_value': 3, 'impact_score': 0.09, 'name': 'Investigation'}
            ]
        }
    }
    
    @classmethod
    def get_criteria_weight(cls, criteria_name: str) -> float:
        """Get the weight for a specific criteria.
        
        Args:
            criteria_name: Name of the criteria to get weight for
            
        Returns:
            Weight value for the criteria
        """
        return cls.SCORING_WEIGHTS['criteria'].get(criteria_name, 0)
    
    @classmethod
    def get_component_weight(cls, component_name: str) -> float:
        """Get the weight for a specific component.
        
        Args:
            component_name: Name of the component to get weight for
            
        Returns:
            Weight value for the component
        """
        return cls.SCORING_WEIGHTS['components'].get(component_name, 0)
    
    @classmethod
    def get_confidence_level(cls, sample_size: int) -> str:
        """Get the confidence level based on sample size.
        
        Args:
            sample_size: Number of samples in the analysis
            
        Returns:
            Confidence level as string: 'none', 'low', 'medium', or 'high'
        """
        if sample_size < cls.CONFIDENCE['minimum_sample']:
            return 'none'
        elif sample_size < cls.CONFIDENCE['low_confidence']:
            return 'low'
        elif sample_size < cls.CONFIDENCE['medium_confidence']:
            return 'medium'
        else:
            return 'high'
    
    @classmethod
    def calculate_confidence_score(cls, sample_size: int, criteria_count: int, 
                               total_criteria: int) -> float:
        """Calculate a confidence score (0-1) based on sample size and criteria completeness.
        
        Args:
            sample_size: Number of samples in the analysis
            criteria_count: Number of criteria specified
            total_criteria: Total number of possible criteria
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from sample size
        if sample_size < cls.CONFIDENCE['minimum_sample']:
            base_confidence = 0
        elif sample_size >= cls.CONFIDENCE['high_confidence']:
            base_confidence = 1.0
        else:
            # Linear scaling between minimum and high confidence
            base_confidence = (sample_size - cls.CONFIDENCE['minimum_sample']) / \
                             (cls.CONFIDENCE['high_confidence'] - cls.CONFIDENCE['minimum_sample'])
            
        # Apply degradation for missing criteria
        criteria_factor = 1.0
        if total_criteria > 0:
            missing_criteria = total_criteria - criteria_count
            for _ in range(missing_criteria):
                criteria_factor *= cls.CONFIDENCE['degradation_factor']
                
        return base_confidence * criteria_factor
