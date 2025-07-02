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
    
    # Version information
    VERSION = "1.0.28"
    
    # Debug message counter to limit output volume
    _debug_message_count = 0
    
    # Debug mode flag - will be controlled by session state
    DEBUG_MODE = False
    
    # Maximum number of debug messages to show - limited to avoid overwhelming the UI
    MAX_DEBUG_MESSAGES = 100
    
    # Debug categories to filter messages
    DEBUG_CATEGORIES = {
        'matcher': False,  # Disable matcher debug messages - too verbose
        'analyzer': True,
        'scorer': True,
        'network': False,  # Disable network debug messages - too verbose
        'recommendation': True,
        'general': True
    }
    
    @staticmethod
    def debug(message, category='general', force=False):
        """Print debug message only if DEBUG_MODE is True or force is True.
        
        Args:
            message: Debug message to print
            category: Debug category for filtering messages
            force: If True, print regardless of DEBUG_MODE setting
        """
        import streamlit as st
        import inspect
        import os
        
        # Check if we should show this message based on mode and category
        should_show = force or (OptimizerConfig.DEBUG_MODE and 
                              (category == 'general' or 
                               OptimizerConfig.DEBUG_CATEGORIES.get(category, True)))
        
        # Check if we've exceeded the maximum number of messages
        if should_show:
            if OptimizerConfig._debug_message_count < OptimizerConfig.MAX_DEBUG_MESSAGES:
                # Get caller information
                frame = inspect.currentframe().f_back
                filename = frame.f_code.co_filename
                lineno = frame.f_lineno
                function = frame.f_code.co_name
                
                # Get just the filename without the path
                short_filename = os.path.basename(filename)
                
                # Format the debug message with source information
                debug_msg = f"DEBUG [{short_filename}:{lineno} in {function}] {category}: {message}"
                
                st.write(debug_msg)
                OptimizerConfig._debug_message_count += 1
            elif OptimizerConfig._debug_message_count == OptimizerConfig.MAX_DEBUG_MESSAGES:
                st.write(f"DEBUG: Maximum debug messages ({OptimizerConfig.MAX_DEBUG_MESSAGES}) reached. Suppressing further output.")
                OptimizerConfig._debug_message_count += 1
            
    @staticmethod
    def set_debug_mode(enabled=True):
        """Set the debug mode flag.
        
        Args:
            enabled: Whether to enable debug mode
        """
        OptimizerConfig.DEBUG_MODE = enabled
        # Reset message counter when toggling debug mode
        OptimizerConfig._debug_message_count = 0
    
    # Verbose debug mode - set to True to enable detailed debug output
    # When False, only critical debug information is shown
    # VERBOSE_DEBUG flag removed
    
    # Maximum number of results to return from matcher
    MAX_RESULTS = 100
    
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
    
    # Match level definitions based on exact number of missing criteria
    # Each match level corresponds directly to the number of missing criteria
    # Level 1 = 0 missing criteria (all match)
    # Level 2 = 1 missing criterion
    # Level 3 = 2 missing criteria
    # And so on...
    
    # Helper functions for match level configuration
    @classmethod
    def get_confidence_for_diff(cls, diff):
        """Get confidence level based on criteria difference."""
        if diff == 0:
            return 'high'
        elif diff == 1:
            return 'medium'
        elif diff == 2:
            return 'medium-low'
        elif diff <= 4:
            return 'low'
        else:
            return 'minimal'
    
    @classmethod
    def get_quality_for_diff(cls, diff):
        """Get quality score based on criteria difference."""
        if diff == 0:
            return 0.9
        elif diff == 1:
            return 0.8
        elif diff == 2:
            return 0.7
        elif diff == 3:
            return 0.6
        elif diff == 4:
            return 0.5
        else:
            return max(0.3, 1.0 - (diff * 0.15))  # Decreases with more missing criteria
    
    @classmethod
    def get_match_level_factor(cls, diff):
        """Get confidence adjustment factor based on criteria difference."""
        if diff == 0:
            return 1.0  # No reduction for exact matches
        elif diff == 1:
            return 0.9  # 10% reduction for missing 1 criterion
        elif diff == 2:
            return 0.8  # 20% reduction for missing 2 criteria
        elif diff == 3:
            return 0.7  # 30% reduction for missing 3 criteria
        elif diff == 4:
            return 0.5  # 50% reduction for missing 4 criteria
        else:
            return max(0.2, 1.0 - (diff * 0.15))  # Decreases with more missing criteria
    
    @classmethod
    def get_match_level_config(cls, diff):
        """Get match level configuration for a specific criteria difference."""
        return {
            'criteria_diff': diff,
            'confidence': cls.get_confidence_for_diff(diff),
            'min_quality': cls.get_quality_for_diff(diff),
            'factor': cls.get_match_level_factor(diff)
        }
    
    # Match levels are generated dynamically based on criteria differences
    # We initialize with level 1 (exact match) and others are created as needed
    MATCH_LEVELS = {}
    
    # Default match level to use when no specific level is specified
    DEFAULT_MATCH_LEVEL = 1  # Level 1 = exact match (all criteria)
    
    # Initialize the default match level
    @classmethod
    def _initialize_default_match_level(cls):
        """Initialize the default match level in MATCH_LEVELS."""
        cls.ensure_match_level_exists(cls.DEFAULT_MATCH_LEVEL)
    
    @classmethod
    def ensure_match_level_exists(cls, level):
        """Ensure that a match level configuration exists for the given level.
        
        Args:
            level: The match level (1 = exact match, 2 = missing 1 criterion, etc.)
        """
        if level not in cls.MATCH_LEVELS:
            # Calculate the criteria difference (level 1 = 0 diff, level 2 = 1 diff, etc.)
            diff = level - 1
            cls.MATCH_LEVELS[level] = cls.get_match_level_config(diff)
        return cls.MATCH_LEVELS[level]
    
    # Confidence thresholds for sample sizes
    CONFIDENCE = {
        'minimum_sample': 1,           # Minimum shows needed for any analysis (lowered from 5)
        'low_confidence': 10,          # Below this is low confidence
        'medium_confidence': 25,       # Below this is medium confidence
        'high_confidence': 50,         # Above this is high confidence
        'degradation_factor': 0.9      # Factor to reduce confidence per missing criteria
    }
    
    # Confidence score to level mapping thresholds
    CONFIDENCE_SCORE_LEVELS = {
        'none': 0.0,       # No confidence (score = 0)
        'low': 0.3,       # Low confidence threshold
        'medium': 0.7,    # Medium confidence threshold
        'high': 1.0       # High confidence threshold (anything above 0.7)
    }
    
    @classmethod
    def map_confidence_score_to_level(cls, confidence_score: float) -> str:
        """Map a confidence score to a confidence level string.
        
        Args:
            confidence_score: Confidence score between 0 and 1
            
        Returns:
            Confidence level string ('none', 'low', 'medium', 'high')
        """
        if confidence_score == 0:
            return 'none'
        elif confidence_score < cls.CONFIDENCE_SCORE_LEVELS['low']:
            return 'low'
        elif confidence_score < cls.CONFIDENCE_SCORE_LEVELS['medium']:
            return 'medium'
        else:
            return 'high'
    
    # Suggestion impact settings
    SUGGESTIONS = {
        'minimum_impact': 0.001,       # Minimum impact score to generate suggestion (0.1%)
        'high_impact_threshold': 0.15, # Threshold for high impact suggestions (15%)
        'max_suggestions': 20,         # Maximum suggestions to return (increased from 5 to 20)
        'network_specific_ratio': 0.6,  # Ratio of network-specific to general suggestions
        'max_unselected_fields': 5     # Maximum number of unselected fields to process for 'add' recommendations
    }
    
    # Success calculation settings
    SUCCESS = {
        'threshold': 0.6,              # Default threshold for success rate calculation
        'min_data_points': 3,          # Minimum data points needed for success calculation (lowered from 5 to 3)
        'confidence_factor': 0.8       # Confidence factor for success probability
    }
    
    # Network analysis settings
    NETWORK = {
        'default_limit': 5,            # Default limit for network recommendations
        'min_compatibility': 0.3,      # Minimum compatibility score for network inclusion
        'high_compatibility': 0.7      # Threshold for high compatibility networks
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
            'min_relevance_score': 0.3,  # Minimum relevance score for inclusion in results (lowered to allow more relaxed matches)
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
        'significant_difference': 0.01,  # 1% difference is considered significant (lowered from 5% to allow more recommendations)
        'major_difference': 0.2,        # 20% difference is considered major
        'minimum_compatibility': 0.3,   # Minimum compatibility score for recommendations
        'strong_compatibility': 0.7,    # Strong compatibility threshold
        # Removed duplicate success_threshold - now only in SUCCESS dictionary
        'network_difference': 0.001     # Minimum difference for network-specific recommendations (lowered to 0.1%)
    }
    
    # Network tier thresholds for compatibility scoring
    NETWORK_TIERS = {
        'excellent': 0.85,  # Threshold for excellent network compatibility
        'good': 0.7,       # Threshold for good network compatibility
        'fair': 0.5        # Threshold for fair network compatibility
        # Note: poor tier is anything below fair threshold
    }
    
    # UI color scheme for consistent visualization
    UI_COLORS = {
        'success': 'green',      # Used for high scores, positive impacts
        'warning': 'orange',     # Used for medium scores, neutral impacts
        'danger': 'red',         # Used for low scores, negative impacts
        'neutral': 'gray',       # Used for unknown or NA values
        'default_text': 'black'  # Default text color
    }
    
    # Default values for fallbacks and missing data
    DEFAULT_VALUES = {
        'impact_score': 0.5,     # Default impact score when no data available
        'confidence': 'low',     # Default confidence level when sample size is limited
        'min_recommendations': 3,  # Minimum number of recommendations to generate
        'fallback_sample_size': 0,  # Default sample size for fallback recommendations
        'fallback_impact_score': 0.1  # Default impact score for fallback recommendations
    }
    
    # Recommendation type display names
    RECOMMENDATION_TYPES = {
        'add': "Missing Criteria",
        'remove': "Consider Removing",
        'change': "Successful Pattern",
        'consider': "Consider Adding",
        'network_keep': "Network Strengths",
        'network_change': "Network Adjustments"
    }
    
    # Confidence level display names
    CONFIDENCE_DISPLAY = {
        'high': "High confidence",
        'medium': "Medium confidence",
        'low': "Low confidence",
        'very_low': "Very low confidence",
        'none': "Insufficient data"
    }
    
    @classmethod
    def get_match_level_description(cls, level):
        """Generate a human-readable description of a match level.
        
        Args:
            level: The match level (1 = exact match, 2 = missing 1 criterion, etc.)
            
        Returns:
            A string description of the match level
        """
        # Match level directly corresponds to criteria differences + 1
        # Level 1 = 0 differences, Level 2 = 1 difference, etc.
        diff = level - 1
        
        if diff == 0:
            return "All criteria matched"
        elif diff == 1:
            return f"Missing {diff} criterion"
        else:
            return f"Missing {diff} criteria"
    
    # Longevity scoring configuration
    LONGEVITY_SCORING = {
        # Status scores for different show statuses
        'status_scores': {
            'Returning Series': 100, 
            'Ended': 75, 
            'Canceled': 25,
            'In Production': 50, 
            'Pilot': 10, 
            'In Development': 5
        },
        # Component weights for calculating overall longevity score
        'component_weights': {
            'seasons': 0.4,
            'episodes': 0.4,
            'status': 0.2
        },
        # Season scoring parameters
        'season_scoring': {
            'base_score_per_season': 25,     # Score per season for first 2 seasons
            'bonus_threshold': 2,            # Seasons needed before bonus points
            'bonus_base': 50,                # Base score after reaching bonus threshold
            'bonus_per_season': 10,          # Additional points per season after threshold
            'max_score': 100                 # Maximum possible score
        },
        # Episode scoring parameters
        'episode_scoring': {
            'high_threshold': 10,            # Episodes needed for maximum score
            'medium_threshold': 5,           # Episodes needed for medium score
            'high_score': 100,              # Score for high threshold
            'medium_base': 50,              # Base score for medium threshold
            'medium_per_episode': 10,       # Points per episode above medium threshold
            'base_score_per_episode': 10    # Score per episode below medium threshold
        }
    }
    
    # Score normalization factors
    SCORE_NORMALIZATION = {
        'popcornmeter': 100.0,  # Divide by this to normalize to 0-1 scale
        'tomatometer': 100.0,   # Divide by this to normalize to 0-1 scale
        'success_filter_min': 0.0  # Minimum value for success score to be considered valid
    }
    
    # Required and optional columns for score calculations
    REQUIRED_COLUMNS = {
        'base': ['show_id', 'title'],  # Basic identification fields required by all calculators
        'success': [],  # No additional required columns beyond base
        'audience': [],  # No additional required columns beyond base
        'critics': [],   # No additional required columns beyond base
        'longevity': []  # No additional required columns beyond base
    }
    
    OPTIONAL_COLUMNS = {
        'success': ['success_score'],
        'audience': ['popcornmeter', 'popcornmeter_count'],
        'critics': ['tomatometer'],
        'longevity': ['tmdb_seasons', 'tmdb_total_episodes', 'tmdb_status']
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
        # Ensure the match level exists and get its factor
        cls.ensure_match_level_exists(match_level)
        level_factor = cls.MATCH_LEVELS[match_level]['factor']
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
        # Ensure the match level exists and get its factor
        cls.ensure_match_level_exists(match_level)
        match_level_factor = cls.MATCH_LEVELS[match_level]['factor']
                
        return base_confidence * criteria_factor * match_level_factor


# Initialize the default match level when the module is loaded
OptimizerConfig._initialize_default_match_level()
