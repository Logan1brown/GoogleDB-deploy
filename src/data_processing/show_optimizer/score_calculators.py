from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable
import pandas as pd
import ast
import streamlit as st
from abc import ABC, abstractmethod

from .optimizer_config import OptimizerConfig

__all__ = [
    'ComponentScore',
    'NetworkMatch',
    'ScoreCalculationError',
    'ScoreCalculator',
    'SuccessScoreCalculator',
    'AudienceScoreCalculator',
    'CriticsScoreCalculator',
    'LongevityScoreCalculator',
    'NetworkScoreCalculator'
]

@dataclass
class NetworkMatch:
    """Network match information with success metrics.
    
    A data container for network matching results with compatibility and success scores.
    Uses OptimizerConfig for default confidence values.
    """
    network_id: int = 0
    network_name: str = ''
    compatibility_score: Optional[float] = None  # 0-1 score of how well the network matches criteria, None if N/A
    success_probability: Optional[float] = None  # 0-1 probability of success on this network, None if N/A
    sample_size: int = 0  # Number of shows in the sample
    confidence: str = 'none'  # Confidence level (none, very_low, low, medium, high)
    details: Dict[str, Any] = field(default_factory=dict)  # Detailed breakdown of score
    
    def __post_init__(self):
        """Validate and set default values from OptimizerConfig."""
        # Set minimum compatibility threshold if not provided
        if self.compatibility_score is not None and self.compatibility_score < OptimizerConfig.THRESHOLDS['minimum_compatibility']:
            self.details['below_threshold'] = True


@dataclass
class ComponentScore:
    """Success score for a component (audience, critics, longevity)."""
    component: str = ''  # audience, critics, or longevity
    score: Optional[float] = None  # 0-1 score, None if N/A
    sample_size: int = 0
    confidence: str = 'none'  # none, low, medium, high
    details: Dict[str, Any] = field(default_factory=dict)  # Detailed breakdown of score

class ScoreCalculationError(Exception):
    """Base exception for score calculation errors."""
    pass

class ScoreCalculator:
    """Base class for all score calculators.
    
    This class provides common functionality for score calculation, including
    data validation and preparation.
    """
    component_name = "base"  # Override in subclasses
    required_columns = []   # Override in subclasses
    optional_columns = []   # Override in subclasses
    data_column = None      # Override in subclasses
    
    def __init__(self, component_name=None):
        """Initialize the score calculator.
        
        Args:
            component_name: Optional name to override the class attribute
        """
        if component_name:
            self.component_name = component_name
    
    def validate_and_prepare_data(self, shows: pd.DataFrame, required_columns: List[str], optional_columns: List[str], data_column: str = None, filter_condition: Callable = None) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, Any]]:
        """Validate and prepare data for calculation.
        
        Args:
            shows: DataFrame of shows
            required_columns: List of required columns
            optional_columns: List of optional columns
            data_column: Column to check for data presence
            filter_condition: Optional function to filter shows
            
        Returns:
            Tuple of (is_success, filtered_shows, result_info)
            - result_info: Dictionary with validation results and metadata for error reporting
        """
        # Initialize result info
        result_info = {
            'component': self.component_name,
            'error': None,
            'warning': None,
            'sample_size': 0,
            'data_coverage': 0
        }
        
        # Check if shows DataFrame is valid
        if shows is None or shows.empty:
            result_info['error'] = 'No shows data provided'
            return False, None, result_info
            
        # Use provided parameters or class attributes
        required_cols = required_columns if required_columns is not None else self.required_columns
        optional_cols = optional_columns if optional_columns is not None else self.optional_columns
        data_col = data_column if data_column is not None else self.data_column
        
        # Check for required columns
        missing_columns = [col for col in required_cols if col not in shows.columns]
        if missing_columns:
            result_info['error'] = f"Missing required columns for {self.component_name}: {missing_columns}"
            return False, None, result_info
        
        # Check for optional columns and note which ones are missing
        missing_optional = [col for col in optional_cols if col not in shows.columns]
        if missing_optional and len(missing_optional) == len(optional_cols):
            result_info['warning'] = f"All optional columns missing for {self.component_name}: {missing_optional}"
        
        # Check if key data column is present
        if data_col and data_col not in shows.columns:
            result_info['warning'] = f"Data column '{data_col}' not present for {self.component_name} score calculation"
            return False, None, result_info
        
        # Apply filter condition if provided, otherwise filter for non-null values in data_column
        if shows.empty:
            result_info['warning'] = f"No shows provided for {self.component_name} score calculation"
            return False, None, result_info
            
        if filter_condition:
            valid_shows = shows[filter_condition(shows)]
        elif data_col and data_col in shows.columns:
            valid_shows = shows[shows[data_col].notna()]
        else:
            valid_shows = shows
            
        # Calculate sample size and coverage
        sample_size = len(valid_shows)
        if sample_size is None:
            sample_size = OptimizerConfig.DEFAULT_VALUES['fallback_sample_size']
        result_info['sample_size'] = sample_size
        result_info['data_coverage'] = sample_size / len(shows) if len(shows) > 0 else 0
        
        # Handle case where no shows have valid data
        if sample_size == 0:
            result_info['warning'] = f"No shows with valid data found for {self.component_name} score"
            return False, None, result_info
            
        return True, valid_shows, result_info
        
    def get_confidence_level(self, sample_size: int) -> str:
        """Get confidence level based on sample size.
        
        Uses OptimizerConfig.CONFIDENCE thresholds to determine confidence level.
        
        Args:
            sample_size: Number of samples
            
        Returns:
            Confidence level string ('none', 'very_low', 'low', 'medium', 'high')
        """
        # Use OptimizerConfig's get_confidence_level method for consistency
        # Default match_level of 1 is appropriate for basic confidence calculation
        return OptimizerConfig.get_confidence_level(sample_size, match_level=1)
    
    @abstractmethod
    def calculate(self, shows: pd.DataFrame, threshold: float = None) -> ComponentScore:
        """Calculate score for the component.
        
        Args:
            shows: DataFrame of shows to calculate score for
            
        Returns:
            ComponentScore object with score and confidence information
        """
        pass

class SuccessScoreCalculator(ScoreCalculator):
    """Calculate score based on success metrics."""
    
    def __init__(self):
        super().__init__(component_name='success')

    def calculate(self, shows: pd.DataFrame, threshold: float = None) -> ComponentScore:
        # Debug information about the shows DataFrame
        if OptimizerConfig.DEBUG_MODE:
            st.write(f"Debug: SuccessScoreCalculator received {len(shows)} shows")
            if 'success_score' in shows.columns:
                non_null_count = shows['success_score'].notna().sum()
                st.write(f"Debug: Found {non_null_count} shows with non-null success_score values")
                if non_null_count > 0:
                    min_score = shows['success_score'].min()
                    max_score = shows['success_score'].max()
                    mean_score = shows['success_score'].mean()
                    st.write(f"Debug: Success score range: min={min_score}, max={max_score}, mean={mean_score}")
            else:
                st.write("Debug: 'success_score' column not found in shows DataFrame")
                st.write(f"Debug: Available columns: {list(shows.columns)}")
        
        # Get required and optional columns from config
        required_columns = OptimizerConfig.REQUIRED_COLUMNS['base'] + OptimizerConfig.REQUIRED_COLUMNS['success']
        optional_columns = OptimizerConfig.OPTIONAL_COLUMNS['success']
        
        # Custom filter condition for success scores - must be present AND greater than minimum threshold
        def success_filter(df):
            return (df['success_score'].notna()) & (df['success_score'] > OptimizerConfig.SCORE_NORMALIZATION['success_filter_min'])
        
        # Validate and prepare data using the helper method
        is_success, valid_shows, result_info = self.validate_and_prepare_data(
            shows, required_columns, optional_columns, 
            data_column='success_score', filter_condition=success_filter
        )
        
        # Handle validation failures and warnings
        if not is_success:
            if result_info['error']:
                st.error(result_info['error'])
            if result_info['warning']:
                st.warning(result_info['warning'])
            # Create a ComponentScore with all required fields and default values
            details = {'error': result_info['error'] or result_info['warning']}
            return ComponentScore(
                component=self.component_name,
                score=None,
                sample_size=result_info.get('sample_size', 0),
                confidence='none',
                details=details
            )
        
        # Calculate average success score from valid shows
        avg_score = valid_shows['success_score'].mean()
        # Normalize to 0-1 if on a 0-100 scale (assume >1 means 0-100 scale)
        if avg_score > 1.0:
            avg_score = avg_score / 100.0
        # Determine confidence level based on sample size
        sample_size = result_info.get('sample_size', None)
        if sample_size is None:
            sample_size = OptimizerConfig.DEFAULT_VALUES['fallback_sample_size']
        
        # Set confidence level based on sample size
        confidence = self.get_confidence_level(sample_size)

        # Create detailed breakdown
        details = {
            'success_score': avg_score,
            'data_coverage': result_info['data_coverage'],
            'valid_shows': result_info['sample_size'],
            'total_shows': len(shows),
            'min_score': valid_shows['success_score'].min() if result_info['sample_size'] > 0 else None,
            'max_score': valid_shows['success_score'].max() if result_info['sample_size'] > 0 else None
        }
        
        # Ensure all fields have proper values
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )

class AudienceScoreCalculator(ScoreCalculator):
    """Calculate audience score."""

    def __init__(self):
        super().__init__(component_name='audience')

    def calculate(self, shows: pd.DataFrame, threshold: float = None) -> ComponentScore:
        # Get required and optional columns from config
        required_columns = OptimizerConfig.REQUIRED_COLUMNS['base'] + OptimizerConfig.REQUIRED_COLUMNS['audience']
        optional_columns = OptimizerConfig.OPTIONAL_COLUMNS['audience']
        
        # Validate and prepare data using the helper method
        is_success, valid_shows, result_info = self.validate_and_prepare_data(
            shows, required_columns, optional_columns, data_column='popcornmeter'
        )
        
        # Handle validation failures and warnings
        if not is_success:
            if result_info['error']:
                st.error(result_info['error'])
            if result_info['warning']:
                st.warning(result_info['warning'])
            return ComponentScore(
                component=self.component_name,
                score=None,
                sample_size=result_info['sample_size'],
                confidence='none',
                details={'error': result_info['error'] or result_info['warning']}
            )
        
        # Calculate average popcornmeter score and normalize to 0-1
        avg_score = valid_shows['popcornmeter'].mean() / OptimizerConfig.SCORE_NORMALIZATION['popcornmeter']
        # Ensure sample_size is always defined
        sample_size = result_info.get('sample_size', None)
        if sample_size is None:
            sample_size = OptimizerConfig.DEFAULT_VALUES['fallback_sample_size']
        confidence = self.get_confidence_level(sample_size)
        
        # Create detailed breakdown
        details = {
            'popcornmeter': avg_score,
            'data_coverage': result_info['data_coverage'],
            'valid_shows': result_info['sample_size'],
            'total_shows': len(shows)
        }
        
        # Add popcornmeter_count information if available
        if 'popcornmeter_count' in valid_shows.columns:
            avg_count = valid_shows['popcornmeter_count'].mean() if valid_shows['popcornmeter_count'].notna().any() else None
            details['avg_popcornmeter_count'] = avg_count
        
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=result_info['sample_size'],
            confidence=confidence,
            details=details
        )

class CriticsScoreCalculator(ScoreCalculator):
    """Calculate critics score using tomatometer."""

    def __init__(self):
        super().__init__(component_name='critics')

    def calculate(self, shows: pd.DataFrame, threshold: float = None) -> ComponentScore:
        # Get required and optional columns from config
        required_columns = OptimizerConfig.REQUIRED_COLUMNS['base'] + OptimizerConfig.REQUIRED_COLUMNS['critics']
        optional_columns = OptimizerConfig.OPTIONAL_COLUMNS['critics']
        
        # Validate and prepare data using the helper method
        is_success, valid_shows, result_info = self.validate_and_prepare_data(
            shows, required_columns, optional_columns, data_column='tomatometer'
        )
        
        # Handle validation failures and warnings
        if not is_success:
            # Display appropriate messages
            if result_info['error']:
                st.error(result_info['error'])
            if result_info['warning']:
                st.warning(result_info['warning'])
                
            # Return null score with no confidence, consistent with other calculators
            warning_message = result_info['error'] or result_info['warning'] or 'Unknown validation issue'
            return ComponentScore(
                component=self.component_name,
                score=None,  # Return None instead of hardcoded value
                sample_size=0,
                confidence='none',
                details={'error': warning_message}
            )
        
        # Calculate average tomatometer score and normalize to 0-1
        avg_score = valid_shows['tomatometer'].mean() / OptimizerConfig.SCORE_NORMALIZATION['tomatometer']
        # Ensure sample_size is always defined
        sample_size = result_info.get('sample_size', None)
        if sample_size is None:
            sample_size = OptimizerConfig.DEFAULT_VALUES['fallback_sample_size']
        confidence = self.get_confidence_level(sample_size)
        
        # Create detailed breakdown
        details = {
            'tomatometer_avg_score': avg_score,
            'data_coverage': result_info['data_coverage'],
            'sample_size': result_info['sample_size'],
            'total_shows': len(shows)
        }
        
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=result_info['sample_size'],
            confidence=confidence,
            details=details
        )

class LongevityScoreCalculator(ScoreCalculator):
    """Calculate longevity score using TMDB metrics."""

    def __init__(self):
        super().__init__(component_name='longevity')

    def _calculate_season_score(self, seasons: float) -> float:
        """Calculate score based on number of seasons using config parameters.
        
        Args:
            seasons: Number of seasons
            
        Returns:
            Score between 0-100
        """
        # Get season scoring parameters from config
        config = OptimizerConfig.LONGEVITY_SCORING['season_scoring']
        
        if seasons >= config['bonus_threshold']:
            # Apply bonus scoring for shows with seasons above threshold
            return min(config['max_score'], 
                       config['bonus_base'] + (seasons - config['bonus_threshold']) * config['bonus_per_season'])
        
        # Base scoring for shows with fewer seasons
        return seasons * config['base_score_per_season']

    def _calculate_episode_score(self, episodes: float) -> float:
        """Calculate score based on number of episodes using config parameters.
        
        Args:
            episodes: Number of episodes
            
        Returns:
            Score between 0-100
        """
        # Get episode scoring parameters from config
        config = OptimizerConfig.LONGEVITY_SCORING['episode_scoring']
        
        if episodes >= config['high_threshold']:
            # Maximum score for shows with many episodes
            return config['high_score']
        elif episodes >= config['medium_threshold']:
            # Medium scoring for shows with moderate number of episodes
            return config['medium_base'] + (episodes - config['medium_threshold']) * config['medium_per_episode']
        
        # Base scoring for shows with few episodes
        return episodes * config['base_score_per_episode']

    def calculate(self, shows: pd.DataFrame, threshold: float = None) -> ComponentScore:
        # Get required and optional columns from config
        required_columns = OptimizerConfig.REQUIRED_COLUMNS['base'] + OptimizerConfig.REQUIRED_COLUMNS['longevity']
        optional_columns = OptimizerConfig.OPTIONAL_COLUMNS['longevity']
        
        # Validate and prepare data using the helper method
        is_success, valid_shows, result_info = self.validate_and_prepare_data(
            shows, required_columns, optional_columns, 
            # Need all three columns to be present for calculation
            filter_condition=lambda df: df['tmdb_seasons'].notna() & 
                                      df['tmdb_total_episodes'].notna() & 
                                      df['tmdb_status'].notna()
        )
        
        # Handle validation failures and warnings
        if not is_success:
            if result_info['error']:
                st.error(result_info['error'])
            if result_info['warning']:
                st.warning(result_info['warning'])
            return ComponentScore(
                component=self.component_name,
                score=None,
                sample_size=0,
                confidence='none',
                details={'error': result_info['error'] or result_info['warning']}
            )
        
        # Get status scores and component weights from config
        status_scores = OptimizerConfig.LONGEVITY_SCORING['status_scores']
        component_weights = OptimizerConfig.LONGEVITY_SCORING['component_weights']
        
        # Calculate individual component scores
        valid_shows['season_score_val'] = valid_shows['tmdb_seasons'].apply(self._calculate_season_score)
        valid_shows['episode_score_val'] = valid_shows['tmdb_total_episodes'].apply(self._calculate_episode_score)
        valid_shows['status_score_val'] = valid_shows['tmdb_status'].map(status_scores).fillna(0)
        
        # Calculate weighted longevity score using config weights
        valid_shows['longevity_score_calc'] = (
            valid_shows['season_score_val'] * component_weights['seasons'] +
            valid_shows['episode_score_val'] * component_weights['episodes'] +
            valid_shows['status_score_val'] * component_weights['status']
        ) / 100  # Convert to 0-1 scale
        
        avg_score = valid_shows['longevity_score_calc'].mean()
        # Ensure sample_size is always defined
        sample_size = result_info.get('sample_size', None)
        if sample_size is None:
            sample_size = OptimizerConfig.DEFAULT_VALUES['fallback_sample_size']
        confidence = self.get_confidence_level(sample_size)
        
        details = {
            'avg_seasons': valid_shows['tmdb_seasons'].mean(),
            'avg_episodes': valid_shows['tmdb_total_episodes'].mean(),
            'status_distribution': valid_shows['tmdb_status'].value_counts(normalize=True).mul(100).round(1).to_dict(),
            'data_coverage': result_info['data_coverage'],
            'sample_size': result_info['sample_size'],
            'total_shows': len(shows)
        }
        
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=result_info['sample_size'],
            confidence=confidence,
            details=details
        )


class NetworkScoreCalculator:
    """Calculates network compatibility and success scores for a set of criteria."""
    
    def __init__(self):
        """Initialize the calculator.
        
        Note: This class no longer depends on CriteriaScorer to prevent redundant matching operations.
        """
        self._integrated_data = None
        # No longer storing matching_shows as an instance variable
        
    def set_integrated_data(self, integrated_data: Dict[str, pd.DataFrame]) -> None:
        """Set the integrated data to use for network scoring.
        
        Args:
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
        """
        self._integrated_data = integrated_data
        
        # Extract network data from reference_data if available
        if 'reference_data' in integrated_data and 'network' in integrated_data['reference_data']:
            # Make network data directly accessible for compatibility with existing code
            if 'networks' not in self._integrated_data:
                self._integrated_data['networks'] = integrated_data['reference_data']['network']
            
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug: Found {len(integrated_data['reference_data']['network'])} networks in reference data")
        else:
            if st.session_state.get('debug_mode', False):
                st.write("Debug: No network data found in reference_data")
        
    def calculate_network_scores(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame) -> List[NetworkMatch]:
        """Calculate network compatibility and success scores for a set of criteria.
        
        Args:
            criteria: Dictionary of criteria to match against
            matching_shows: DataFrame of shows matching the criteria
            
        Returns:
            List of NetworkMatch objects with compatibility and success scores
        """
        try:
            # matching_shows is now a required parameter, no need to check if it's None
                
            # Check if we have network_id in the matching shows
            if 'network_id' not in matching_shows.columns:
                # Try to find an alternative network ID column
                network_id_alternatives = ['network', 'network_name']
                found_alternative = False
                
                for alt_col in network_id_alternatives:
                    if alt_col in matching_shows.columns:
                        if st.session_state.get('debug_mode', False):
                            st.write(f"Debug: Using alternative column '{alt_col}' for network identification")
                        # Create a temporary network_id column
                        matching_shows['network_id'] = matching_shows[alt_col]
                        found_alternative = True
                        break
                
                if not found_alternative:
                    st.warning("No network_id column in matching shows. Cannot calculate network scores.")
                    if st.session_state.get('debug_mode', False):
                        st.write("Debug: Missing network_id column in matching shows")
                        st.write(f"Debug: Available columns: {matching_shows.columns.tolist()}")
                    return []
                
            # Get unique network IDs from matching shows
            network_ids = matching_shows['network_id'].dropna().unique()
            
            # Check if we found any network IDs
            if len(network_ids) == 0:
                st.warning("No valid network IDs found in matching shows")
                if st.session_state.get('debug_mode', False):
                    st.write("Debug: No valid network IDs found in matching shows")
                return []
                
            # Debug message about network IDs found
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug: Found {len(network_ids)} unique network IDs in matching shows")
            
            # Initialize results list and network matches list
            results = []
            network_matches = []
            
            # Get unique networks from the matching shows
            if 'network_id' in matching_shows.columns:
                network_ids = matching_shows['network_id'].dropna().unique()
                
            # Process each network ID
            for network_id in network_ids:
                # Ensure network_id is an integer for proper lookup
                if isinstance(network_id, str) and network_id.isdigit():
                    network_id = int(network_id)
                elif isinstance(network_id, float):
                    network_id = int(network_id)
                    
                # Get network name directly from field manager - no fallbacks
                if hasattr(self, 'field_manager') and self.field_manager:
                    network_name = self.field_manager.get_name('network', network_id)
                else:
                    # If field manager is not available, use a default name
                    network_name = f"Network {network_id}"
                    
                    # Debug output
                    if st.session_state.get('debug_mode', False):
                        st.write(f"Debug: No field manager available for network ID {network_id}, using default name")
                        st.write(f"Debug: NetworkScoreCalculator attributes: {dir(self)}")
                        st.write(f"Debug: NetworkScoreCalculator has field_manager: {hasattr(self, 'field_manager')}")
                        st.write(f"Debug: NetworkScoreCalculator field_manager type: {type(getattr(self, 'field_manager', None))}")
                        st.write(f"Debug: NetworkScoreCalculator field_manager dir: {dir(getattr(self, 'field_manager', None)) if hasattr(self, 'field_manager') else 'N/A'}")
                        st.write("---")
                
                # Get shows for this network
                network_shows = matching_shows[matching_shows['network_id'] == network_id]
                
                # Calculate match quality and confidence info
                sample_size = len(network_shows)
                match_level = 1  # Direct match
                
                # Calculate a more meaningful match quality based on how well shows match on this network
                # This uses a nuanced approach that considers match levels of individual shows
                
                # Base calculation on match levels if available
                if 'match_level' in network_shows.columns:
                    # Calculate weighted average match quality based on match levels
                    # Level 1 = perfect match (100%), Level 2 = missing 1 criterion (lower %), etc.
                    total_weight = 0
                    weighted_match_quality = 0
                    
                    # Get total number of criteria from the criteria dictionary
                    total_criteria = len(criteria) if isinstance(criteria, dict) else 3  # Default to 3 if unknown
                    
                    # Calculate weighted match quality for each show
                    for _, show in network_shows.iterrows():
                        # Get match level (1 = perfect match, higher = more criteria missing)
                        level = show.get('match_level', 1)
                        
                        # Calculate match quality for this show (decreases as level increases)
                        # Level 1: 100%, Level 2: 67%, Level 3: 33%, etc. based on criteria matched
                        criteria_matched = max(0, total_criteria - (level - 1))
                        show_match_quality = criteria_matched / total_criteria if total_criteria > 0 else 0
                        
                        # Weight by show's success score if available
                        show_weight = show.get('success_score', 1.0)
                        
                        # Add to weighted average
                        weighted_match_quality += show_match_quality * show_weight
                        total_weight += show_weight
                    
                    # Calculate final weighted average
                    if total_weight > 0:
                        match_quality = weighted_match_quality / total_weight
                    else:
                        match_quality = 0.3  # Default base value
                        
                    # Ensure we have a reasonable minimum value
                    match_quality = max(0.3, match_quality)
                    
                    # Debug output to help diagnose issues
                    if st.session_state.get('debug_mode', False):
                        st.write(f"Debug: Network {network_name} match quality calculation:")
                        st.write(f"- Shows: {sample_size}")
                        st.write(f"- Match levels: {network_shows['match_level'].value_counts().to_dict() if 'match_level' in network_shows.columns else 'N/A'}")
                        st.write(f"- Final match quality: {match_quality:.2f}")
                
                # Fallback to simpler calculation if match_level not available
                else:
                    if 'success_score' in network_shows.columns:
                        # Use average success score as a component of match quality
                        avg_success = network_shows['success_score'].mean()
                        # Scale to ensure we get a reasonable distribution between 0.3-1.0
                        match_quality = max(0.3, min(1.0, avg_success))
                    else:
                        # Calculate based on sample size relative to total matching shows
                        relative_size = min(1.0, sample_size / max(1, len(matching_shows)))
                        
                        # Add a genre-based factor to differentiate networks
                        genre_factor = 0.0
                        if 'genre' in criteria and network_shows.shape[0] > 0:
                            target_genre = criteria.get('genre')
                            if target_genre and 'genre' in network_shows.columns:
                                # Calculate percentage of shows on this network that match the target genre
                                genre_matches = network_shows[network_shows['genre'] == target_genre].shape[0]
                                genre_factor = min(0.3, genre_matches / max(1, network_shows.shape[0]) * 0.3)
                        
                        # Combine relative size with genre factor for a more differentiated score
                        match_quality = 0.3 + (0.4 * relative_size) + genre_factor
                
                # Create confidence info
                # Calculate confidence score (0-1) based on sample size and other factors
                confidence_score = OptimizerConfig.calculate_confidence_score(
                    sample_size=sample_size,
                    match_level=match_level,
                    criteria_count=len(criteria),
                    total_criteria=len(OptimizerConfig.CRITERIA_IMPORTANCE)
                )
                
                # Map the confidence score to a confidence level string
                confidence_level = OptimizerConfig.map_confidence_score_to_level(confidence_score)
                
                # Create a proper NetworkMatch object instead of a dictionary
                # Initialize with empty details dictionary to ensure it's properly created
                network_match_obj = NetworkMatch(
                    network_id=network_id,
                    network_name=network_name,
                    compatibility_score=match_quality,  # Use match_quality as compatibility score
                    success_probability=None,  # Will be calculated later
                    sample_size=sample_size,
                    confidence=confidence_level
                )
                
                # Add details separately to ensure the dictionary is properly initialized
                network_match_obj.details = {
                    'criteria': criteria,
                    'match_level': match_level,
                    'match_quality': match_quality,
                    'confidence_score': confidence_score,
                    'matching_shows': network_shows
                }
                
                
                network_matches.append(network_match_obj)
            
            # If we didn't find any networks in the matching shows, log a message
            if not network_matches and st.session_state.get('debug_mode', False):
                st.write("Debug: No networks found in matching shows")
                # Check if there are any network IDs in the data
                if 'network_id' in matching_shows.columns:
                    network_id_counts = matching_shows['network_id'].value_counts()
                    if not network_id_counts.empty:
                        st.write(f"Debug: Network ID distribution: {network_id_counts.to_dict()}")
                
            # Add a debug message about the number of network matches
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug: Found {len(network_matches)} network matches using {len(matching_shows)} matching shows")

            
            # Process each network match to calculate success probability
            for i, network_match in enumerate(network_matches):
                # Access attributes directly from NetworkMatch object
                network_id = network_match.network_id
                network_name = network_match.network_name
                matching_shows = network_match.details.get('matching_shows', pd.DataFrame())
                count = network_match.sample_size
                
                # Create confidence_info dictionary from details
                confidence_info = {
                    'score': network_match.details.get('confidence_score'),
                    'level': network_match.confidence,
                    'sample_size': network_match.sample_size,
                    'match_level': network_match.details.get('match_level', 1),  # Direct match
                    'match_quality': network_match.details.get('match_quality')
                }
                
                # Calculate weighted compatibility score using config weights
                # Access match_quality directly from the details dictionary
                match_quality = network_match.details.get('match_quality') if isinstance(network_match.details, dict) else None
                
                # Debug output to help diagnose issues
                if st.session_state.get('debug_mode', False):
                    st.write(f"Debug: network_match.details type: {type(network_match.details)}")
                    st.write(f"Debug: match_quality from details: {match_quality}")
                
                compatibility_score = self._calculate_weighted_compatibility_score(
                    match_quality=match_quality,
                    success_history=None  # NetworkMatch objects don't have success_history
                )
                
                # Always try to calculate success probability with available data
                # Instead of relying on potentially hardcoded values
                success_rate = None
                
                # First check if we have enough shows for this network
                if not matching_shows.empty:
                    # Check if we have enough shows for a reliable calculation
                    if count >= OptimizerConfig.CONFIDENCE['minimum_sample']:
                        # Calculate success rate using our method
                        success_rate, confidence_info = self.calculate_success_rate(matching_shows, confidence_info=confidence_info)
                        
                        # Debug output in debug mode only
                        if st.session_state.get('debug_mode', False):
                            st.write(f"Debug: Calculated success rate for {network_name}: {success_rate} with {count} shows")
                    else:
                        # Not enough shows for reliable calculation
                        if st.session_state.get('debug_mode', False):
                            st.write(f"Debug: Insufficient sample size for {network_name}: {count} shows (min required: {OptimizerConfig.CONFIDENCE['minimum_sample']})")
                else:
                    # No matching shows for this network
                    if st.session_state.get('debug_mode', False):
                        st.write(f"Debug: No matching shows for {network_name}")
                
                # Get confidence level from config based on sample size and match level
                match_level = confidence_info.get('match_level', 1)
                confidence = OptimizerConfig.get_confidence_level(count, match_level) if count > 0 else 'none'
                
                # Create a details dictionary with all required fields
                details_dict = {
                    'criteria': criteria,
                    'match_level': confidence_info.get('match_level', 1),
                    'match_quality': confidence_info.get('match_quality', None)
                }
                
                # Only add confidence_info if it's not None to avoid potential issues
                if confidence_info is not None:
                    details_dict['confidence_info'] = confidence_info
                
                # Update the existing NetworkMatch object with success probability and confidence
                network_match.success_probability = success_rate if success_rate is not None else None
                network_match.confidence = confidence
                
                # Update details dictionary with additional information
                network_match.details.update(details_dict)
                if confidence_info is not None:
                    network_match.details['confidence_info'] = confidence_info
                
                # Debug output in debug mode
                if st.session_state.get('debug_mode', False):
                    st.write(f"Debug: Updated network match for {network_name} with success probability: {success_rate}")
                    st.write(f"Debug: Network match now has compatibility_score: {network_match.compatibility_score}, success_probability: {network_match.success_probability}")
                    
                # No need to create a new object, we're updating the existing one in the list
                
            # Add final debug output about the network matches
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug: Final network_matches count: {len(network_matches)}")
                if network_matches:
                    st.write(f"Debug: First network match: {network_matches[0].network_name} (ID: {network_matches[0].network_id})")
                    st.write(f"Debug: First network match compatibility: {network_matches[0].compatibility_score}")
                    st.write(f"Debug: First network match success probability: {network_matches[0].success_probability}")
                    
            # Sort network matches by compatibility score (descending)
            network_matches.sort(key=lambda x: x.compatibility_score if x.compatibility_score is not None else -1, reverse=True)
                
            # Return the network_matches list that we've been updating
            return network_matches

        except ValueError as ve:
            # Use consistent error handling - st.write for debug, st.error for user-facing
            st.write(f"Debug: Network score calculation error: {str(ve)}")
            st.error("Unable to calculate network scores. Please check your criteria and try again.")
            return []
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug: Unexpected error in network score calculation: {str(e)}")
                import traceback
                st.write(f"Debug: Traceback: {traceback.format_exc()}")
            return []
            
    def calculate_success_rate(self, shows: pd.DataFrame, confidence_info: Dict[str, Any] = None, threshold: Optional[float] = None) -> Tuple[Optional[float], Dict[str, Any]]:
        """Calculate success rate for a set of shows with confidence information.
        
        Args:
            shows: DataFrame of shows to calculate success rate for
            confidence_info: Optional dictionary of confidence information to update
            threshold: Optional success threshold (0-1)
            
        Returns:
            Tuple of (success_rate, confidence_info)
        """
        # Initialize confidence_info if not provided
        if confidence_info is None:
            confidence_info = {}
            
        # Check if success_score is present
        if 'success_score' not in shows.columns:
            # Success score column missing from shows
            if st.session_state.get('debug_mode', False):
                st.write("Debug: Shows data missing required 'success_score' field for success rate calculation")
            
            if 'id' not in shows.columns:
                if st.session_state.get('debug_mode', False):
                    st.write("Debug: Shows data missing required 'id' field for success score calculation")
                # Don't raise an error, just return None to indicate we can't calculate success rate
                confidence_info['success_rate'] = None
                confidence_info['success_count'] = 0
                confidence_info['total_count'] = 0
                return None, confidence_info
            
            # We can't calculate success rate without success_score column
            # Don't try to fetch criteria data here as that would trigger redundant matching
            if st.session_state.get('debug_mode', False):
                st.write("Debug: Cannot calculate success rate without success_score column")
            
            confidence_info['success_rate'] = None
            confidence_info['success_count'] = 0
            confidence_info['total_count'] = 0
            return None, confidence_info
                    
            # No need for additional checks here as we've already raised exceptions for failure cases
                
        # Filter out shows with zero or missing success scores
        valid_shows = shows[shows['success_score'].notna() & (shows['success_score'] > 0)]
            
        if valid_shows.empty:
            # No valid shows after filtering
            # Use DEFAULT_VALUES instead of FALLBACK_VALUES which doesn't exist
            default_success_rate = 0.0  # Don't use hardcoded values
            confidence_info['success_rate'] = None  # Use None to indicate no data
            confidence_info['success_count'] = 0
            confidence_info['total_count'] = 0
            return None, confidence_info
        
        # Use default threshold if none provided
        if threshold is None:
            threshold = OptimizerConfig.THRESHOLDS['success_threshold']
        
        # Get success score range and distribution
        min_score = valid_shows['success_score'].min()
        max_score = valid_shows['success_score'].max()
        mean_score = valid_shows['success_score'].mean()
        
        # Normalize threshold if scores are on 0-100 scale
        normalized_threshold = threshold
        
        # Check if scores need normalization (0-100 scale)
        if max_score > 1.0:  # If scores are on 0-100 scale
            normalized_threshold = threshold * 100
        elif threshold > 1.0:  # If threshold is on 0-100 scale but scores are on 0-1 scale
            normalized_threshold = threshold / 100
        
        # Count successful shows (those with score >= threshold)
        successful = valid_shows[valid_shows['success_score'] >= normalized_threshold]
        success_count = len(successful)
        total_count = len(valid_shows)
        
        # Calculate success rate based on success count and total count
        success_rate = success_count / total_count if total_count > 0 else 0.0
        
        # Update confidence info with success metrics
        confidence_info['success_rate'] = success_rate
        confidence_info['success_count'] = success_count
        confidence_info['total_count'] = total_count
        confidence_info['min_score'] = float(min_score)
        confidence_info['max_score'] = float(max_score)
        confidence_info['mean_score'] = float(mean_score)
        
        # Adjust confidence based on sample size and match level
        sample_size = total_count
        match_level = confidence_info.get('match_level', 1)
        
        # Use OptimizerConfig to determine confidence level
        confidence_info['level'] = OptimizerConfig.get_confidence_level(sample_size, match_level)
        
        # Calculate confidence score using config parameters
        confidence_score = OptimizerConfig.calculate_confidence_score(
            sample_size=sample_size,
            criteria_count=confidence_info.get('criteria_count', 1),
            total_criteria=confidence_info.get('total_criteria', 1),
            match_level=match_level
        )
        confidence_info['score'] = confidence_score

        return success_rate, confidence_info

    def _calculate_weighted_compatibility_score(self, match_quality: float, success_history: Optional[float] = None) -> float:
        """Calculate weighted compatibility score using configuration weights.
        
        Args:
            match_quality: Content match quality score (0-1)
            success_history: Optional success history score (0-1)
            
        Returns:
            Weighted compatibility score (0-1)
        """
        # Get weights from config
        content_match_weight = OptimizerConfig.SCORING_WEIGHTS['network_compatibility']['content_match']
        success_history_weight = OptimizerConfig.SCORING_WEIGHTS['network_compatibility']['success_history']
        
        # Debug output to help diagnose issues
        if st.session_state.get('debug_mode', False):
            st.write(f"Debug: Calculating compatibility score with match_quality={match_quality}, success_history={success_history}")
        
        # If success history is not available, use only content match with adjusted weight
        if success_history is None:
            # Use match_quality directly as the score, with a scaling factor to ensure distribution
            # This ensures we don't get uniform scores for all networks
            return match_quality
            
        # Calculate weighted score
        weighted_score = (match_quality * content_match_weight) + (success_history * success_history_weight)
        
        # Debug output for the final score
        if st.session_state.get('debug_mode', False):
            st.write(f"Debug: Calculated weighted score: {weighted_score}")
            
        # Ensure score is in 0-1 range
        return max(0.0, min(1.0, weighted_score))
    
    def batch_calculate_success_rates(self, criteria_list: List[Dict[str, Any]], matching_shows_list: List[pd.DataFrame] = None) -> List[Tuple[Optional[float], Dict[str, Any]]]:
        """
        Batch calculate success rates for multiple criteria with confidence information.

        Args:
            criteria_list: List of criteria dictionaries
            matching_shows_list: Required list of DataFrames, each containing shows matching the corresponding criteria.

        Returns:
            List of tuples (success_rate, confidence_info) in the same order as criteria_list
        """
        results = []
        # We now require matching_shows_list to be provided
        if matching_shows_list is None:
            if st.session_state.get('debug_mode', False):
                st.write("Debug: No matching_shows_list provided to batch_calculate_success_rates")
            # Return empty results for all criteria
            return [(None, {'error': 'No matching shows provided'})] * len(criteria_list)
            
        if len(criteria_list) != len(matching_shows_list):
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug: Mismatch between criteria list ({len(criteria_list)}) and matching shows list ({len(matching_shows_list)})")
            return [(None, {'error': 'criteria/matching_shows_list length mismatch'})] * len(criteria_list)
            
        for criteria, matching_shows in zip(criteria_list, matching_shows_list):
            try:
                if matching_shows is None or matching_shows.empty:
                    confidence_info = {
                        'sample_size': 0,
                        'match_level': OptimizerConfig.DEFAULT_MATCH_LEVEL,
                        'match_quality': OptimizerConfig.ensure_match_level_exists(OptimizerConfig.DEFAULT_MATCH_LEVEL)['min_quality'],
                        'level': 'none',  # Use string directly instead of accessing CONFIDENCE_LEVELS
                        'success_rate': None
                    }
                    results.append((None, confidence_info))
                    continue
                    
                success_rate, confidence_info = self.calculate_success_rate(matching_shows)
                results.append((success_rate, confidence_info))
            except Exception as e:
                if st.session_state.get('debug_mode', False):
                    st.write(f"Debug: Error calculating success rate: {str(e)}")
                    
                error_confidence = {
                    'level': 'none',  # Use string directly
                    'score': 0.0,  # No fallback value, use 0.0 to indicate no data
                    'match_quality': 0.0,  # No fallback value, use 0.0 to indicate no data
                    'sample_size': 0,
                    'match_level': OptimizerConfig.DEFAULT_MATCH_LEVEL,
                    'success_rate': None,
                    'success_count': 0,
                    'total_count': 0,
                    'error': str(e)
                }
                results.append((None, error_confidence))
                
        return results
