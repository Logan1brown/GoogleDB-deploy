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
    
    # Criteria importance classification for flexible matching
    CRITERIA_IMPORTANCE = {
        # Essential criteria (must match for any meaningful analysis)
        'genre': 'essential',
        
        # Core criteria (strongly influence show identity)
        'subgenres': 'core',        # Refined genre classification
        'plot_elements': 'core',    # Fundamental to show identity 
        'character_types': 'core',  # Key character archetypes 
        'source_type': 'core',      # Original material source 
        
        # Primary criteria (important but can be relaxed if needed)
        'tone': 'primary',             # Show's emotional tone 
        'thematic_elements': 'primary', # Important themes and messages
        'location_setting': 'primary', # Geographic setting 
        'time_setting': 'primary',    # Time period setting 
        
        # Secondary criteria (contextual factors, less critical for matching)
        'network': 'secondary',         # Distribution channel
        'studios': 'secondary',         # Production companies
        'team_members': 'secondary',    # Creative talent
        'order_type': 'secondary'       # How show was ordered/developed
    }
    
    # Importance category weights for flexible matching
    IMPORTANCE_WEIGHTS = {
        'essential': 10.0, # Essential remains highest
        'core': 7.0,       # 7x secondary weight
        'primary': 3.5,    # 3.5x secondary weight
        'secondary': 1.0   # Base weight
    }
    
    # Match level definitions and confidence mapping
    MATCH_LEVELS = {
        1: {
            'name': 'All criteria matched',
            'confidence': 'high',
            'min_quality': 0.9
        },
        2: {
            'name': 'All but one secondary criterion matched',
            'confidence': 'medium',
            'min_quality': 0.7
        },
        3: {
            'name': 'Core and primary criteria matched',
            'confidence': 'low',
            'min_quality': 0.5
        },
        4: {
            'name': 'Only core criteria matched',
            'confidence': 'very_low',
            'min_quality': 0.3
        }
    }
    
    # Confidence thresholds for sample sizes
    CONFIDENCE = {
        'minimum_sample': 5,           # Minimum shows needed for any analysis
        'low_confidence': 10,          # Below this is low confidence
        'medium_confidence': 25,       # Below this is medium confidence
        'high_confidence': 50,         # Above this is high confidence
        'degradation_factor': 0.9,     # Factor to reduce confidence per missing criteria
        'match_level_factor': {        # Confidence adjustment factors by match level
            1: 1.0,                    # No reduction for level 1 (all criteria)
            2: 0.9,                    # 10% reduction for level 2 (all but one secondary)
            3: 0.7,                    # 30% reduction for level 3 (core + primary)
            4: 0.5                     # 50% reduction for level 4 (core only)
        }
    }
    
    # Performance settings
    PERFORMANCE = {
        'cache_duration': 3600,        # Cache duration in seconds (1 hour)
        'success_threshold': 0.6,      # Default threshold for success rate calculation
        'min_criteria_coverage': 0.5,  # Minimum criteria coverage for valid analysis
        'min_confidence_score': 0.3,   # Minimum confidence score for valid results
        'max_cached_combinations': 100, # Maximum number of cached criteria combinations
        'incremental_threshold': 0.7    # Threshold for incremental vs full recalculation
    }
    
    @classmethod
    def calculate_confidence_score(cls, sample_size: int, criteria_count: int, total_criteria: int, match_level: int) -> float:
        """Calculate a confidence score based on sample size, criteria coverage, and match level.
        
        Args:
            sample_size: Number of shows in the sample
            criteria_count: Number of criteria used in the match
            total_criteria: Total number of available criteria
            match_level: Match level used (1-4)
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from sample size
        if sample_size <= cls.CONFIDENCE['minimum_sample']:
            base_confidence = 0.1
        elif sample_size <= cls.CONFIDENCE['low_confidence']:
            base_confidence = 0.3
        elif sample_size <= cls.CONFIDENCE['medium_confidence']:
            base_confidence = 0.6
        elif sample_size <= cls.CONFIDENCE['high_confidence']:
            base_confidence = 0.8
        else:
            base_confidence = 1.0
            
        # Adjust for criteria coverage
        criteria_coverage = criteria_count / total_criteria if total_criteria > 0 else 0
        coverage_factor = max(criteria_coverage, cls.PERFORMANCE['min_criteria_coverage'])
        
        # Adjust for match level degradation
        match_level_factor = cls.CONFIDENCE['match_level_factor'].get(match_level, 0.5)
        
        # Calculate final confidence score
        confidence_score = base_confidence * coverage_factor * match_level_factor
        
        return min(1.0, max(0.1, confidence_score))  # Clamp between 0.1 and 1.0
    
    @classmethod
    def get_confidence_level(cls, sample_size: int, match_level: int) -> str:
        """Get a confidence level string based on sample size and match level.
        
        Args:
            sample_size: Number of shows in the sample
            match_level: Match level used (1-4)
            
        Returns:
            Confidence level string ('none', 'very_low', 'low', 'medium', 'high')
        """
        if sample_size < cls.CONFIDENCE['minimum_sample']:
            return 'none'
            
        # Base level from sample size
        if sample_size < cls.CONFIDENCE['low_confidence']:
            base_level = 'very_low'
        elif sample_size < cls.CONFIDENCE['medium_confidence']:
            base_level = 'low'
        elif sample_size < cls.CONFIDENCE['high_confidence']:
            base_level = 'medium'
        else:
            base_level = 'high'
            
        # Adjust for match level
        if match_level == 1:
            # No adjustment for level 1
            return base_level
        elif match_level == 2:
            # Reduce by one level for level 2
            if base_level == 'high':
                return 'medium'
            elif base_level == 'medium':
                return 'low'
            else:
                return 'very_low'
        elif match_level == 3:
            # Reduce by two levels for level 3
            if base_level == 'high':
                return 'low'
            else:
                return 'very_low'
        else:  # match_level == 4 or higher
            # Always very_low for level 4+
            return 'very_low'
    
    # Suggestion impact settings
    SUGGESTIONS = {
        'minimum_impact': 0.05,        # Minimum impact score to generate suggestion (5%)
        'high_impact_threshold': 0.15, # Threshold for high impact suggestions (15%)
        'max_suggestions': 5,          # Maximum suggestions to return
        'network_specific_ratio': 0.6   # Ratio of network-specific to general suggestions
    }
    
    # Fallback recommendation system settings
    FALLBACK_SYSTEM = {
        # Criteria relaxation settings
        'relaxation': {
            'min_matches_before_fallback': 10,  # Minimum matches needed before fallback is triggered
            'min_confidence_before_fallback': 'low',  # Minimum confidence level before fallback is triggered
            'relaxation_tiers': [
                'secondary',  # First relax secondary criteria
                'primary',    # Then primary criteria if needed
                'core'        # Only relax core criteria as last resort (never relax essential)
            ],
            'max_criteria_to_relax': 2,  # Maximum criteria to relax in a single tier
            'min_sample_increase_factor': 2.0  # Minimum factor by which sample size must increase to justify relaxation
        },
        
        # Weighted relevance scoring
        'relevance': {
            'min_relevance_score': 0.5,  # Minimum relevance score for inclusion in results
            'high_relevance_threshold': 0.8,  # Threshold for high relevance
            'weight_multipliers': {  # Multipliers for importance categories in relevance scoring
                'essential': 1.0,  # Essential criteria always fully weighted
                'core': 0.9,      # Core criteria slightly reduced
                'primary': 0.7,   # Primary criteria moderately reduced
                'secondary': 0.5  # Secondary criteria significantly reduced
            }
        }
    }
    
    # Threshold values for various comparisons
    THRESHOLDS = {
        'significant_difference': 0.1,  # 10% difference is considered significant
        'major_difference': 0.2,        # 20% difference is considered major
        'minimum_compatibility': 0.3,   # Minimum compatibility score for recommendations
        'strong_compatibility': 0.7,    # Strong compatibility threshold
        'success_threshold': 0.6        # Threshold for considering a show successful (60%)
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
    def get_confidence_level(cls, sample_size: int, match_level: int = 1) -> str:
        """Get the confidence level based on sample size and match level.
        
        Args:
            sample_size: Number of samples in the analysis
            match_level: Match level used (1-4, where 1 is highest)
            
        Returns:
            Confidence level as string: 'none', 'very_low', 'low', 'medium', or 'high'
        """
        if sample_size < cls.CONFIDENCE['minimum_sample']:
            return 'none'
            
        # Adjust thresholds based on match level
        level_factor = cls.CONFIDENCE['match_level_factor'].get(match_level, 0.5)
        adjusted_low = cls.CONFIDENCE['low_confidence'] / level_factor
        adjusted_medium = cls.CONFIDENCE['medium_confidence'] / level_factor
        adjusted_high = cls.CONFIDENCE['high_confidence'] / level_factor
        
        if match_level >= 4:  # Core criteria only - add very_low confidence level
            if sample_size < adjusted_low:
                return 'very_low'
                
        if sample_size < adjusted_low:
            return 'low'
        elif sample_size < adjusted_medium:
            return 'medium'
        else:
            return 'high'
    
    @classmethod
    def calculate_confidence_score(cls, sample_size: int, criteria_count: int, 
                               total_criteria: int, match_level: int = 1) -> float:
        """Calculate a confidence score (0-1) based on sample size, criteria completeness, and match level.
        
        Args:
            sample_size: Number of samples in the analysis
            criteria_count: Number of criteria specified
            total_criteria: Total number of possible criteria
            match_level: Match level used (1-4, where 1 is highest)
            
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
        
        # Apply match level adjustment
        match_level_factor = cls.CONFIDENCE['match_level_factor'].get(match_level, 0.5)
                
        return base_confidence * criteria_factor * match_level_factor
