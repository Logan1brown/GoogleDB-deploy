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
    
    def validate_and_prepare_data(self, shows: pd.DataFrame, required_columns=None, optional_columns=None, 
                                  data_column=None, filter_condition=None) -> Tuple[bool, Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Validate and prepare data for score calculation.
        
        Args:
            shows: DataFrame of shows
            required_columns: List of required column names (overrides class attribute)
            optional_columns: List of optional column names (overrides class attribute)
            data_column: Key data column to check presence of (typically the main score/metric column)
            filter_condition: Optional callable that takes a DataFrame and returns a boolean Series for filtering
                             Default is to filter for non-null values in data_column
            
        Returns:
            Tuple of (is_success, valid_shows, result_info)
            - is_success: True if validation succeeded and data is ready for scoring
            - valid_shows: Filtered DataFrame with valid data, or None if validation failed
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
            return ComponentScore(
                component=self.component_name,
                score=None,
                sample_size=result_info['sample_size'],
                confidence='none',
                details={'error': result_info['error'] or result_info['warning']}
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
        self._matching_shows = None
        
    def set_integrated_data(self, integrated_data: Dict[str, pd.DataFrame]) -> None:
        """Set the integrated data to use for network scoring.
        
        Args:
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
        """
        self._integrated_data = integrated_data
        
    def set_matching_shows(self, matching_shows: pd.DataFrame) -> None:
        """Set the matching shows to use for network scoring.
        
        Args:
            matching_shows: DataFrame of matching shows
        """
        self._matching_shows = matching_shows
    
    def calculate_network_scores(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame = None) -> List[NetworkMatch]:
        """Calculate network compatibility and success scores for a set of criteria.
        
        Args:
            criteria: Dictionary containing criteria for network scoring
            matching_shows: DataFrame of shows matching the criteria (optional, will use self._matching_shows if not provided)
            
        Returns:
            List of NetworkMatch objects with compatibility and success scores
        """
        try:
            # Prepare results list
            results = []
            
            # Use the matching_shows parameter if provided, otherwise use the stored matching_shows
            if matching_shows is not None:
                self._matching_shows = matching_shows
            
            if self._matching_shows is None or self._matching_shows.empty:
                st.error("No matching shows available for network scoring. Make sure to provide matching_shows.")
                return []
                
            # We still need integrated_data for additional context
            if self._integrated_data is None or 'shows' not in self._integrated_data:
                st.error("No integrated data available for network scoring. Make sure to call set_integrated_data first.")
                return []
                
            criteria_data = self._integrated_data['shows']
            # Only show debug output in debug mode
            if st.session_state.get('debug_mode', False):
                st.write("Using matching shows and integrated data for network scoring")
            
            # We'll use the provided matching_shows directly instead of triggering a new matching operation
            # This prevents redundant matching operations
            
            # Check if we have network data in the integrated data
            if 'networks' not in self._integrated_data:
                if st.session_state.get('debug_mode', False):
                    st.write("Debug: No network data available in integrated_data")
                st.error("Network data is not available. Please check your data sources.")
                return []
                
            # Get network data from integrated data
            network_data = self._integrated_data['networks']
            
            # Process network matches directly using the matching shows
            # This is a simplified version of what find_network_matches would do
            network_matches = []
            
            # Get unique networks from the matching shows
            if 'network_id' in self._matching_shows.columns:
                network_ids = self._matching_shows['network_id'].dropna().unique()
                
                for network_id in network_ids:
                    # Get network name from network data
                    network_name = network_data[network_data['id'] == network_id]['name'].iloc[0] if not network_data.empty else f"Network {network_id}"
                    
                    # Get shows for this network
                    network_shows = self._matching_shows[self._matching_shows['network_id'] == network_id]
                    
                    # Calculate match quality and confidence info
                    sample_size = len(network_shows)
                    match_level = 1  # Direct match
                    match_quality = 1.0  # Perfect match quality for direct network matches
                    
                    # Create confidence info
                    confidence_info = OptimizerConfig.create_confidence_info(
                        sample_size=sample_size,
                        match_level=match_level,
                        criteria_count=len(criteria),
                        total_criteria=len(OptimizerConfig.CRITERIA_IMPORTANCE)
                    )
                    confidence_info['match_quality'] = match_quality
                    
                    network_matches.append({
                        'network_id': network_id,
                        'network_name': network_name,
                        'matching_shows': network_shows,
                        'sample_size': sample_size,
                        'confidence_info': confidence_info,
                        'match_quality': match_quality
                    })
            
            # If we didn't find any networks in the matching shows, log a message
            if not network_matches and st.session_state.get('debug_mode', False):
                st.write("Debug: No networks found in matching shows")
                
            # Add a debug message about the number of network matches
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug: Found {len(network_matches)} network matches using {len(self._matching_shows)} matching shows")

            
            # Process each network match
            for network_match in network_matches:
                network_id = network_match['network_id']
                network_name = network_match['network_name']
                matching_shows = network_match['matching_shows']
                count = network_match['sample_size']
                confidence_info = network_match['confidence_info']
                
                # Calculate weighted compatibility score using config weights
                compatibility_score = self._calculate_weighted_compatibility_score(
                    match_quality=network_match['match_quality'],
                    success_history=network_match.get('success_history', None)
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
                
                # Create NetworkMatch object
                network_match_obj = NetworkMatch(
                    network_id=int(network_id),
                    network_name=network_name,
                    compatibility_score=compatibility_score,
                    success_probability=success_rate if success_rate is not None else None,
                    sample_size=count if not matching_shows.empty else 0,
                    confidence=confidence,
                    details={
                        'criteria': criteria,
                        'confidence_info': confidence_info,
                        'match_level': confidence_info.get('match_level', 1),
                        'match_quality': confidence_info.get('match_quality', None)
                    }
                )
                
                results.append(network_match_obj)
                
            return results

        except ValueError as ve:
            # Use consistent error handling - st.write for debug, st.error for user-facing
            st.write(f"Debug: Network score calculation error: {str(ve)}")
            st.error("Unable to calculate network scores. Please check your criteria and try again.")
            return []
        except Exception as e:
            st.write(f"Debug: Unexpected error in network score calculation: {str(e)}")
            st.error("An unexpected error occurred while calculating network scores.")
            return []
    
    def calculate_success_rate(self, shows: pd.DataFrame, threshold: Optional[float] = None, 
                               confidence_info: Optional[Dict[str, Any]] = None) -> Tuple[Optional[float], Dict[str, Any]]:
        """Calculate the success rate for a set of shows with confidence information.
        
        Args:
            shows: DataFrame of shows
            threshold: Success threshold (shows with score >= threshold are considered successful)
            confidence_info: Optional confidence information from flexible matching
                
        Returns:
            Tuple of (success_rate, confidence_info)
            - success_rate: Success rate (0-1) or None if success_score is missing
            - confidence_info: Dictionary with confidence metrics
        """
        # Initialize confidence info if not provided
        if confidence_info is None:
            sample_size = len(shows) if not shows.empty else 0
            match_level = OptimizerConfig.DEFAULT_MATCH_LEVEL
            
            # Use OptimizerConfig's standardized method to create confidence info
            confidence_info = OptimizerConfig.create_confidence_info(
                sample_size=sample_size,
                match_level=match_level,
                criteria_count=1,
                total_criteria=1
            )
                
        # Check if we have shows to analyze
        if shows.empty:
            # No shows to analyze
            confidence_info['success_rate'] = None
            confidence_info['success_count'] = 0
            confidence_info['total_count'] = 0
            return None, confidence_info
                
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
        
        # If success history is not available, use only content match with config weight
        if success_history is None:
            # Use content match with weight adjustment from config
            content_only_weight = OptimizerConfig.SCORING_WEIGHTS['network_compatibility'].get('content_only', 1.0)
            return match_quality * content_only_weight
            
        # Calculate weighted score
        weighted_score = (match_quality * content_match_weight) + (success_history * success_history_weight)
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
