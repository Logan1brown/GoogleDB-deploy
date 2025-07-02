from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple, Callable
import pandas as pd
import ast
import streamlit as st
from abc import ABC, abstractmethod

from .optimizer_config import OptimizerConfig
from .optimizer_data_contracts import CriteriaDict, ConfidenceInfo, IntegratedData, NetworkMatch

# Empty placeholder function for compatibility with existing code
def debug_warning(message):
    pass

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
    
    This class provides common functionality for all score calculators, including:
    - Input validation
    - Data preparation
    - Confidence level calculation
    - Error handling
    
    All score calculators should inherit from this class and implement at minimum:
    - calculate_scores: Main entry point that returns a structured dictionary with scores and confidence
    
    Data Contracts:
    - Input: All calculators should accept criteria (Dict) and matching_shows (DataFrame)
    - Output: All calculators should return a Dict with structured results and confidence info
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
            shows: DataFrame of shows to validate and filter
            required_columns: List of required columns that must be present
            optional_columns: List of optional columns that enhance calculation
            data_column: Column to check for data presence
            filter_condition: Optional function to filter shows
            
        Returns:
            Tuple of (is_success, filtered_shows, result_info) where:
            - is_success: Boolean indicating if validation passed
            - filtered_shows: DataFrame of valid shows or None if validation failed
            - result_info: Dictionary with validation results and metadata
        """
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
        required_cols = required_columns or self.required_columns
        optional_cols = optional_columns or self.optional_columns
        data_col = data_column or self.data_column
        
        # Check for required columns
        missing_columns = [col for col in required_cols if col not in shows.columns]
        if missing_columns:
            result_info['error'] = f"Missing required columns for {self.component_name}: {missing_columns}"
            return False, None, result_info
        
        # Check for optional columns
        missing_optional = [col for col in optional_cols if col not in shows.columns]
        if missing_optional and len(missing_optional) == len(optional_cols):
            result_info['warning'] = f"All optional columns missing for {self.component_name}: {missing_optional}"
        
        # Check if key data column is present
        if data_col and data_col not in shows.columns:
            result_info['warning'] = f"Data column '{data_col}' not present for {self.component_name} score calculation"
            return False, None, result_info
        
        # Filter shows based on condition or data column presence
        if filter_condition:
            valid_shows = shows[filter_condition(shows)]
        elif data_col and data_col in shows.columns:
            valid_shows = shows[shows[data_col].notna()]
        else:
            valid_shows = shows
            
        # Calculate sample size and coverage
        sample_size = len(valid_shows)
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
        
    def calculate_scores(self, criteria: CriteriaDict, matching_shows: pd.DataFrame) -> Dict[str, Any]:
        """Calculate scores for the given criteria and matching shows.
        
        This is the main entry point for score calculation, providing a unified interface
        across all score calculators. Subclasses can override this method for custom behavior,
        but the default implementation should work for most calculators.
        
        Args:
            criteria: Dictionary of criteria to match against conforming to CriteriaDict.
            matching_shows: DataFrame of shows matching the criteria.
                
        Returns:
            A dictionary containing:
            - component_score: ComponentScore object with score information
            - confidence: Dictionary with overall confidence information
            - error: Error message if calculation failed (only present if there was an error)
        """
        result = {
            'confidence': {
                'level': OptimizerConfig.CONFIDENCE_LEVELS['none'],
                'score': 0.0
            }
        }
        
        try:
            # Calculate component score using the specific calculator's implementation
            component_score = self.calculate(matching_shows)
            result['component_score'] = component_score
            
            # Update confidence information based on component score
            if component_score and component_score.confidence:
                result['confidence'] = {
                    'level': component_score.confidence,
                    'score': component_score.details.get('confidence_score', 0.0) or 0.0
                }
        except Exception as e:
            result['error'] = f'Error calculating {self.component_name} scores: {str(e)}'
            
        return result

class SuccessScoreCalculator(ScoreCalculator):
    """Calculate score based on success metrics.
    
    This calculator produces a ComponentScore object containing:
    - component: The name of the component ('success')
    - score: A float between 0-1 representing the success score
    - sample_size: The number of shows used to calculate the score
    - confidence: A string indicating the confidence level ('high', 'medium', 'low', 'none')
    - details: A dictionary containing additional information about the score calculation
    """
    
    def __init__(self):
        super().__init__(component_name='success')
        
    # Using the base class calculate_scores method

    def calculate(self, shows: pd.DataFrame, threshold: float = None) -> ComponentScore:
        """Calculate success score based on success metrics in the shows DataFrame.
        
        Args:
            shows: DataFrame of shows with success metrics. Must contain 'success_score' column.
            threshold: Optional threshold value (not currently used).
            
        Returns:
            ComponentScore object with success score information and confidence level.
            If validation fails, returns ComponentScore with score=None and error details.
        """
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
        
        # Handle validation failures
        if not is_success:
            # Return error ComponentScore
            details = {'error': result_info['error'] or result_info['warning']}
            return ComponentScore(
                component=self.component_name,
                score=None,
                sample_size=result_info.get('sample_size', 0),
                confidence=OptimizerConfig.CONFIDENCE_LEVELS['none'],
                details=details
            )
        
        # Calculate average success score from valid shows, weighted by match level if available
        if 'match_level' in valid_shows.columns:
            # Apply match level weighting
            weighted_scores = []
            match_level_counts = {}
            
            # Group by match level and apply weighting
            for level, level_shows in valid_shows.groupby('match_level'):
                level_count = len(level_shows)
                match_level_counts[level] = level_count
                
                # Get weight factor for this match level
                weight_factor = OptimizerConfig.get_match_level_factor(level - 1)  # level-1 = criteria diff
                
                # Calculate weighted score for this level
                level_score = level_shows['success_score'].mean() * weight_factor
                weighted_scores.append(level_score * level_count)
                
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Match level {level}: {level_count} shows, weight={weight_factor:.2f}, score={level_score:.4f}", category='scoring')
            
            # Calculate weighted average
            total_shows = sum(match_level_counts.values())
            if total_shows > 0:
                avg_score = sum(weighted_scores) / total_shows
                
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Weighted avg score: {avg_score:.4f} across {total_shows} shows in {len(match_level_counts)} match levels", category='scoring')
            else:
                avg_score = valid_shows['success_score'].mean()
        else:
            # Fall back to simple average if match_level not available
            avg_score = valid_shows['success_score'].mean()
            
        # Normalize to 0-1 if on a 0-100 scale
        if avg_score > 1.0:
            avg_score = avg_score / 100.0
            
        # Get sample size and determine confidence level
        sample_size = result_info['sample_size']
        confidence = self.get_confidence_level(sample_size)

        # Create detailed breakdown
        details = {
            'success_score': avg_score,
            'data_coverage': result_info['data_coverage'],
            'valid_shows': sample_size,
            'total_shows': len(shows),
            'min_score': valid_shows['success_score'].min() if sample_size > 0 else None,
            'max_score': valid_shows['success_score'].max() if sample_size > 0 else None
        }
        
        # Add match level distribution to details if available
        if 'match_level' in valid_shows.columns:
            match_level_dist = {f'level_{level}': count for level, count in match_level_counts.items()}
            details['match_level_distribution'] = match_level_dist
        
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
        
    # Using the base class calculate_scores method


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
        
    # Using the base class calculate_scores method


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
        """Calculate longevity score for a set of shows.
        
        Args:
            shows: DataFrame of shows to calculate longevity score for. Must contain
                'tmdb_seasons', 'tmdb_total_episodes', and 'tmdb_status' columns.
            threshold: Optional threshold (unused in this calculator).
            
        Returns:
            ComponentScore object with longevity score and details.
        """
        # Get required and optional columns from config
        required_columns = OptimizerConfig.REQUIRED_COLUMNS['base'] + OptimizerConfig.REQUIRED_COLUMNS['longevity']
        optional_columns = OptimizerConfig.OPTIONAL_COLUMNS['longevity']
        
        # Validate and prepare data using the helper method
        is_success, valid_shows, result_info = self.validate_and_prepare_data(
            shows, required_columns, optional_columns, 
            filter_condition=lambda df: df['tmdb_seasons'].notna() & 
                                      df['tmdb_total_episodes'].notna() & 
                                      df['tmdb_status'].notna()
        )
        
        # Handle validation failures
        if not is_success:
            return ComponentScore(
                component=self.component_name,
                score=None,
                sample_size=0,
                confidence=OptimizerConfig.CONFIDENCE_LEVELS['none'],
                details={'error': result_info['error'] or result_info['warning']}
            )
        
        # Get scoring configuration
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
        
        # Calculate final score and confidence
        avg_score = valid_shows['longevity_score_calc'].mean()
        sample_size = result_info['sample_size']
        confidence = self.get_confidence_level(sample_size)
        
        # Prepare detailed results
        details = {
            'avg_seasons': valid_shows['tmdb_seasons'].mean(),
            'avg_episodes': valid_shows['tmdb_total_episodes'].mean(),
            'status_distribution': valid_shows['tmdb_status'].value_counts(normalize=True).mul(100).round(1).to_dict(),
            'data_coverage': result_info['data_coverage'],
            'sample_size': sample_size,
            'total_shows': len(shows)
        }
        
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
        
    # Using the base class calculate_scores method


class NetworkScoreCalculator(ScoreCalculator):
    """Calculates network compatibility and success scores for a set of criteria.
    
    Uses the matching_shows DataFrame as the single source of truth for network compatibility analysis.
    
    This calculator produces a list of NetworkMatch objects, each containing:
    - network_id: The ID of the network
    - network_name: The name of the network
    - compatibility_score: A score from 0-1 indicating how well the criteria match the network
    - success_probability: A score from 0-1 indicating the likelihood of success on this network
    - sample_size: The number of shows used to calculate the scores
    - confidence: A string indicating the confidence level ('high', 'medium', 'low', 'none')
    - details: A dictionary containing additional information about the match
    """
    

        
    def calculate_scores(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all network scores for the given criteria and matching shows.
        
        This is the main entry point for network score calculation, providing a unified interface
        that aligns with the CriteriaScorer.calculate_scores method.
        
        Args:
            criteria: Dictionary of criteria to match against. Must contain valid criteria fields.
            matching_shows: DataFrame of shows matching the criteria. Must contain at minimum
                'network_id' and 'match_level' columns.
                
        Returns:
            A dictionary containing:
            - network_matches: List of NetworkMatch objects with compatibility and success scores
            - confidence: Dictionary with overall confidence information
            - error: Error message if calculation failed (only present if there was an error)
        """
        result = {
            'network_matches': [],
            'confidence': {
                'level': OptimizerConfig.CONFIDENCE_LEVELS['none'],
                'score': 0.0
            }
        }
        
        try:
            # Calculate network scores
            network_matches = self.calculate_network_scores(criteria, matching_shows)
            result['network_matches'] = network_matches
            
            # Calculate overall confidence based on network matches
            if network_matches:
                confidence_levels = [match.confidence for match in network_matches]
                confidence_scores = [match.details.get('confidence_score', 0.0) for match in network_matches if match.details]
                
                confidence_level = self._get_highest_confidence_level(confidence_levels)
                confidence_score = max(confidence_scores) if confidence_scores else 0.0
                
                result['confidence'] = {
                    'level': confidence_level,
                    'score': confidence_score
                }
        except Exception as e:
            result['error'] = f'Error calculating network scores: {str(e)}'
            
        return result
    
    def __init__(self):
        """Initialize the calculator.
        
        Note: This class uses only the matching_shows DataFrame as the data source to prevent redundant matching operations.
        """
        super().__init__()
        self.component_name = 'network'
        self.required_columns = ['network_id', 'match_level']
        self.field_manager = None  # Will be set by CriteriaScorer if available
        
    def calculate_network_scores(self, criteria: CriteriaDict, matching_shows: pd.DataFrame) -> List[NetworkMatch]:
        """Calculate network compatibility and success scores for a set of criteria.
        
        Args:
            criteria: Dictionary of criteria to match against conforming to CriteriaDict.
            matching_shows: DataFrame of shows matching the criteria. Must contain at minimum
                'network_id' and 'match_level' columns.
            
        Returns:
            List of NetworkMatch objects with compatibility and success scores.
        """
        # Validate and prepare data using the base class method
        is_success, valid_shows, result_info = self.validate_and_prepare_data(
            shows=matching_shows,
            required_columns=['network_id', 'match_level'],
            optional_columns=[],
            data_column='network_id'
        )
        
        if not is_success or valid_shows is None or valid_shows.empty:
            return []
            
        # Get unique network IDs
        network_ids = valid_shows['network_id'].dropna().unique()
        if len(network_ids) == 0:
            return []
            
        # Initialize network matches list
        network_matches = []
        
        # Process each network ID
        for network_id in network_ids:
            # Normalize network_id to integer if possible
            if isinstance(network_id, (str, float)) and str(network_id).replace('.', '', 1).isdigit():
                network_id = int(float(network_id))
                
            # Get network name
            network_name = self.field_manager.get_name('network', network_id) if hasattr(self, 'field_manager') and self.field_manager else f"Network {network_id}"
                
            # Get shows for this network
            network_shows = valid_shows[valid_shows['network_id'] == network_id]
            if network_shows.empty:
                continue
                
            # Calculate match quality metrics
            sample_size = len(network_shows)
            match_level = network_shows['match_level'].min()  # Best match level
            total_criteria = len(criteria) if isinstance(criteria, dict) else 3
            
            # Calculate weighted match quality
            total_weight = 0
            weighted_match_quality = 0
            
            for _, show in network_shows.iterrows():
                level = show['match_level'] if 'match_level' in show else 1
                criteria_matched = max(0, total_criteria - (level - 1))
                show_match_quality = criteria_matched / total_criteria if total_criteria > 0 else 0
                show_weight = show['success_score'] if 'success_score' in show else 1.0
                
                weighted_match_quality += show_match_quality * show_weight
                total_weight += show_weight
            
            if total_weight <= 0:
                continue
                
            match_quality = weighted_match_quality / total_weight
            
            # Calculate confidence score
            confidence_score = OptimizerConfig.calculate_confidence_score(
                sample_size=sample_size,
                match_level=match_level,
                criteria_count=len(criteria),
                total_criteria=len(OptimizerConfig.CRITERIA_IMPORTANCE)
            )
            
            confidence_level = OptimizerConfig.map_confidence_score_to_level(confidence_score)
            
            # Create NetworkMatch object
            network_match = NetworkMatch(
                network_id=network_id,
                network_name=network_name,
                compatibility_score=match_quality,
                success_probability=None,
                sample_size=sample_size,
                confidence=confidence_level,
                details={
                    'criteria': criteria,
                    'match_level': match_level,
                    'match_quality': match_quality,
                    'confidence_score': confidence_score,
                    'matching_shows': network_shows
                }
            )
            
            network_matches.append(network_match)
        
        # Calculate success probability for each network match
        for network_match in network_matches:
            matching_shows = network_match.details.get('matching_shows', pd.DataFrame())
            count = network_match.sample_size
            
            # Create confidence_info dictionary
            confidence_info = {
                'score': network_match.details.get('confidence_score', 0.0),
                'level': network_match.confidence,
                'sample_size': network_match.sample_size,
                'match_level': network_match.details.get('match_level', 1),
                'match_quality': network_match.details.get('match_quality', 0.0)
            }
            
            # Calculate weighted compatibility score
            match_quality = network_match.details.get('match_quality', 0.0)
            compatibility_score = self._calculate_weighted_compatibility_score(
                match_quality=match_quality,
                success_history=None
            )
            
            # Calculate success rate if enough samples
            success_rate = None
            if not matching_shows.empty and count >= OptimizerConfig.CONFIDENCE['minimum_sample']:
                success_rate, confidence_info = self.calculate_success_rate(matching_shows, confidence_info=confidence_info)
            
            # Update the network match
            if success_rate is not None:
                network_match.success_probability = success_rate
            
            if compatibility_score is not None:
                network_match.compatibility_score = compatibility_score
                
            # Get confidence level - safely extract match_level from confidence_info
            # Extract match_level from confidence_info dictionary
            match_level = confidence_info.get('match_level', 1) if confidence_info else 1
                
            confidence = OptimizerConfig.get_confidence_level(count, match_level) if count > 0 else OptimizerConfig.CONFIDENCE_LEVELS['none']
            network_match.confidence = confidence
            
            # Update details
            network_match.details.update({
                'match_level': match_level,  # Use the already extracted match_level value
                'match_quality': confidence_info.get('match_quality', 0.0),
                'confidence_info': confidence_info
            })
        
        # Sort network matches by compatibility score (descending)
        network_matches.sort(key=lambda x: x.compatibility_score if x.compatibility_score is not None else -1, reverse=True)
        
        # Return the list of network matches
        return network_matches
            
    def calculate_success_rate(self, shows: pd.DataFrame, confidence_info: Optional[ConfidenceInfo] = None, threshold: Optional[float] = None) -> Tuple[Optional[float], ConfidenceInfo]:
        """Calculate success rate for a set of shows with confidence information.
        
        Args:
            shows: DataFrame of shows to calculate success rate for. Must contain 'success_score' column.
            confidence_info: Optional dictionary of confidence information to update conforming to ConfidenceInfo.
            threshold: Optional success threshold (0-1). Defaults to OptimizerConfig.SUCCESS['threshold'].
            
        Returns:
            Tuple of (success_rate, confidence_info) where:
                - success_rate: Float between 0-1 representing the success rate, or None if calculation failed
                - confidence_info: Dictionary with calculation details conforming to ConfidenceInfo
        """
        # Initialize or use provided confidence_info
        confidence_info = confidence_info or {}
            
        # Check if success_score is present
        if 'success_score' not in shows.columns or shows.empty:
            confidence_info.update({
                'success_rate': None,
                'success_count': 0,
                'total_count': 0,
                'error': "Missing success_score column" if 'success_score' not in shows.columns else "No shows available",
                'level': OptimizerConfig.CONFIDENCE_LEVELS['none']
            })
            return None, confidence_info
            
        # Use provided threshold or default
        success_threshold = threshold if threshold is not None else OptimizerConfig.SUCCESS['threshold']
            
        # Calculate success metrics
        total_count = len(shows)
        success_count = len(shows[shows['success_score'] >= success_threshold])
        success_rate = success_count / total_count
        
        # Update confidence info with success metrics
        confidence_info.update({
            'success_rate': success_rate,
            'success_count': success_count,
            'total_count': total_count
        })
        
        # Calculate confidence score based on sample size - extract match_level from dictionary
        match_level = confidence_info.get('match_level', OptimizerConfig.DEFAULT_MATCH_LEVEL) if confidence_info else OptimizerConfig.DEFAULT_MATCH_LEVEL
            
        confidence_score = OptimizerConfig.calculate_confidence_score(
            sample_size=total_count,
            match_level=match_level,
            criteria_count=confidence_info.get('criteria_count', 0),
            total_criteria=len(OptimizerConfig.CRITERIA_IMPORTANCE)
        )
        
        # Update confidence info with confidence score and level
        confidence_info.update({
            'score': confidence_score,
            'level': OptimizerConfig.map_confidence_score_to_level(confidence_score)
        })
        
        return success_rate, confidence_info
            
    def _calculate_weighted_compatibility_score(self, match_quality: Optional[float] = None, success_history: Optional[float] = None) -> float:
        """Calculate weighted compatibility score using config weights.
        
        Combines match quality and success history scores using weights from OptimizerConfig
        to produce a single compatibility score.
        
        Args:
            match_quality: Match quality score (0-1) indicating criteria match quality
            success_history: Success history score (0-1) indicating past success
            
        Returns:
            Weighted compatibility score (0-1) representing overall compatibility
        """
        # Get weights from config
        weights = OptimizerConfig.NETWORK_COMPATIBILITY_WEIGHTS
        
        # Use provided values or defaults
        match_quality = match_quality or 0.0
        success_history = success_history or 0.0
            
        # Calculate weighted score and ensure it's in 0-1 range
        weighted_score = (match_quality * weights['content_match']) + \
                         (success_history * weights['success_history'])
                
        return max(0.0, min(1.0, weighted_score))
    
    def _get_highest_confidence_level(self, confidence_levels: List[str]) -> str:
        """Get the highest confidence level from a list of confidence levels.
        
        Args:
            confidence_levels: List of confidence level strings ('high', 'medium', 'low', 'none')
            
        Returns:
            The highest confidence level as a string from OptimizerConfig.CONFIDENCE_LEVELS
        """
        if not confidence_levels:
            return OptimizerConfig.CONFIDENCE_LEVELS['none']
        
        # Map valid confidence levels to their numeric values
        level_values = []
        for level in confidence_levels:
            level_key = level.lower() if isinstance(level, str) else 'none'
            if level_key in OptimizerConfig.CONFIDENCE_LEVEL_VALUES:
                level_values.append((level_key, OptimizerConfig.CONFIDENCE_LEVEL_VALUES[level_key]))
            
        # Return highest confidence level or default to 'none'
        if not level_values:
            return OptimizerConfig.CONFIDENCE_LEVELS['none']
            
        # Find the key with the maximum value
        max_level_key = max(level_values, key=lambda x: x[1])[0]
        return OptimizerConfig.CONFIDENCE_LEVELS[max_level_key]
    
    def batch_calculate_success_rates(self, criteria_list: List[CriteriaDict], matching_shows_list: List[pd.DataFrame] = None) -> List[Tuple[Optional[float], ConfidenceInfo]]:
        """Batch calculate success rates for multiple criteria with confidence information.

        Args:
            criteria_list: List of criteria dictionaries conforming to CriteriaDict.
            matching_shows_list: List of DataFrames with shows matching the corresponding criteria.
                Each DataFrame must contain 'success_score' column for success rate calculation.

        Returns:
            List of tuples (success_rate, confidence_info) in the same order as criteria_list,
            where confidence_info conforms to ConfidenceInfo.
        """
        # Return empty list for invalid input
        if not criteria_list:
            return []
            
        # Create standard error response helper function
        def create_error_response(error_message: str) -> Tuple[None, ConfidenceInfo]:
            return (None, {
                'error': error_message,
                'level': OptimizerConfig.CONFIDENCE_LEVELS['none'],
                'score': 0.0,
                'sample_size': 0,
                'success_rate': None,
                'success_count': 0,
                'total_count': 0
            })
            
        # Validate inputs
        if matching_shows_list is None:
            error_response = create_error_response('No matching shows provided')
            return [error_response] * len(criteria_list)
            
        if len(criteria_list) != len(matching_shows_list):
            error_response = create_error_response('Length mismatch between criteria_list and matching_shows_list')
            return [error_response] * len(criteria_list)
            
        # Process each criteria and matching shows pair
        results = []
        for criteria, matching_shows in zip(criteria_list, matching_shows_list):
            try:
                if matching_shows is None or matching_shows.empty:
                    results.append(create_error_response('No matching shows available'))
                    continue
                    
                success_rate, confidence_info = self.calculate_success_rate(matching_shows)
                results.append((success_rate, confidence_info))
            except Exception as e:
                results.append(create_error_response(f'Error calculating success rate: {str(e)}'))
                
        return results
