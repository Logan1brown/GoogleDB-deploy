from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import ast
import streamlit as st
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .optimizer_config import OptimizerConfig
from .optimizer_matcher import Matcher

__all__ = [
    'ComponentScore',
    'NetworkMatch',
    'ScoreCalculationError',
    'ScoreCalculator',
    'SuccessScoreCalculator',
    'AudienceScoreCalculator',
    'CriticsScoreCalculator',
    'LongevityScoreCalculator',
    'NetworkScoreCalculator',
    'MatchingCalculator'
]

@dataclass
class NetworkMatch:
    """Network match information with success metrics."""
    network_id: int
    network_name: str
    compatibility_score: Optional[float]  # 0-1 score of how well the network matches criteria, None if N/A
    success_probability: Optional[float]  # 0-1 probability of success on this network, None if N/A
    sample_size: int  # Number of shows in the sample
    confidence: str = 'none'  # Confidence level (none, very_low, low, medium, high)
    details: Dict[str, Any] = field(default_factory=dict)  # Detailed breakdown of score


@dataclass
class ComponentScore:
    """Success score for a component (audience, critics, longevity)."""
    component: str  # audience, critics, or longevity
    score: Optional[float]  # 0-1 score, None if N/A
    sample_size: int
    confidence: str  # none, low, medium, high
    details: Dict[str, Any] = field(default_factory=dict)  # Detailed breakdown of score

class ScoreCalculationError(Exception):
    """Base exception for score calculation errors."""
    pass

class ScoreCalculator(ABC):
    """Abstract base class for score calculations."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
    
    @abstractmethod
    def calculate(self, shows: pd.DataFrame) -> 'ComponentScore': # Forward reference ComponentScore
        """Calculate the component score."""
        pass
    
    def _get_confidence(self, sample_size: int) -> str:
        """Get confidence level based on sample size."""
        return OptimizerConfig.get_confidence_level(sample_size)

class SuccessScoreCalculator(ScoreCalculator):
    """Calculate score based on success metrics."""
    
    def __init__(self):
        super().__init__(component_name='success')

    def calculate(self, shows: pd.DataFrame) -> ComponentScore:
        if shows.empty:
            raise ScoreCalculationError(f"Cannot calculate {self.component_name} score with empty shows DataFrame")
        
        if 'success_score' not in shows.columns:
            raise ScoreCalculationError(f"success_score column not found in shows data for {self.component_name} score")

        # Filter out shows with missing success scores AND shows with a score of 0
        # Shows with a score of 0 are typically those that haven't aired yet or have unreliable data
        valid_shows = shows[(shows['success_score'].notna()) & (shows['success_score'] > 0)]
        sample_size = len(valid_shows)
        
        if sample_size == 0:
            raise ScoreCalculationError(f"No shows with valid success_score data found for {self.component_name}")
        
        avg_score = valid_shows['success_score'].mean()
        confidence = self._get_confidence(sample_size)
        
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details={'success_score': avg_score}
        )

class AudienceScoreCalculator(ScoreCalculator):
    """Calculate audience score."""

    def __init__(self):
        super().__init__(component_name='audience')

    def calculate(self, shows: pd.DataFrame) -> ComponentScore:
        if shows.empty:
            raise ScoreCalculationError(f"Cannot calculate {self.component_name} score with empty shows DataFrame")
            
        if 'popcornmeter' not in shows.columns:
            raise ScoreCalculationError(f"popcornmeter column not found in shows data for {self.component_name} score")
            
        valid_shows = shows[shows['popcornmeter'].notna()]
        sample_size = len(valid_shows)
        
        if sample_size == 0:
            raise ScoreCalculationError(f"No shows with valid popcornmeter data found for {self.component_name} score")
            
        avg_score = valid_shows['popcornmeter'].mean() / 100  # Normalize to 0-1
        confidence = self._get_confidence(sample_size)
        
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details={'popcornmeter': avg_score}
        )

class CriticsScoreCalculator(ScoreCalculator):
    """Calculate critics score using tomatometer."""

    def __init__(self):
        super().__init__(component_name='critics')

    def calculate(self, shows: pd.DataFrame) -> ComponentScore:
        if shows.empty:
            # Return a default score with very low confidence instead of raising an error
            st.warning(f"No shows available for {self.component_name} score calculation")
            return ComponentScore(
                component=self.component_name,
                score=0.5,  # Neutral score
                sample_size=0,
                confidence='none',
                details={'warning': 'No shows available', 'sample_size': 0}
            )
            
        if 'tomatometer' not in shows.columns:
            # Return a default score with very low confidence instead of raising an error
            st.warning(f"tomatometer column not found for {self.component_name} score calculation")
            return ComponentScore(
                component=self.component_name,
                score=0.5,  # Neutral score
                sample_size=0,
                confidence='none',
                details={'warning': 'Missing tomatometer data', 'sample_size': 0}
            )
            
        valid_shows = shows[shows['tomatometer'].notna()]
        sample_size = len(valid_shows)
        
        if sample_size == 0:
            # Return a default score with very low confidence instead of raising an error
            st.warning(f"No shows with valid tomatometer data found for {self.component_name} score")
            return ComponentScore(
                component=self.component_name,
                score=0.5,  # Neutral score
                sample_size=0,
                confidence='none',
                details={'warning': 'No valid tomatometer data', 'sample_size': 0}
            )
            
        avg_score = valid_shows['tomatometer'].mean() / 100.0  # Normalize to 0-1
        confidence = self._get_confidence(sample_size)
        
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details={'tomatometer_avg_score': avg_score, 'sample_size': sample_size}
        )

class LongevityScoreCalculator(ScoreCalculator):
    """Calculate longevity score using TMDB metrics."""

    def __init__(self):
        super().__init__(component_name='longevity')

    def _calculate_season_score(self, seasons: float) -> float:
        if seasons >= 2:
            return min(100, 50 + (seasons - 2) * 10)
        return seasons * 25

    def _calculate_episode_score(self, episodes: float) -> float:
        if episodes >= 10:
            return 100
        elif episodes >= 5:
            return 50 + (episodes - 5) * 10
        return episodes * 10

    def calculate(self, shows: pd.DataFrame) -> ComponentScore:
        if shows.empty:
            raise ScoreCalculationError(f"Cannot calculate {self.component_name} score with empty shows DataFrame")
        
        required_columns = ['tmdb_seasons', 'tmdb_total_episodes', 'tmdb_status']
        missing_columns = [col for col in required_columns if col not in shows.columns]
        if missing_columns:
            raise ScoreCalculationError(
                f"Missing required columns for {self.component_name} score: {', '.join(missing_columns)}. "
                f"Available columns: {list(shows.columns)}"
            )
        
        valid_shows = shows.dropna(subset=required_columns).copy() # Use .copy() to avoid SettingWithCopyWarning
        sample_size = len(valid_shows)
        
        if sample_size == 0:
            raise ScoreCalculationError(f"No shows with valid TMDB metrics found for {self.component_name} score")
        
        status_scores = {
            'Returning Series': 100, 'Ended': 75, 'Canceled': 25,
            'In Production': 50, 'Pilot': 10, 'In Development': 5
        }
        
        valid_shows['season_score_val'] = valid_shows['tmdb_seasons'].apply(self._calculate_season_score)
        valid_shows['episode_score_val'] = valid_shows['tmdb_total_episodes'].apply(self._calculate_episode_score)
        valid_shows['status_score_val'] = valid_shows['tmdb_status'].map(status_scores).fillna(0)
        
        valid_shows['longevity_score_calc'] = (
            valid_shows['season_score_val'] * 0.4 +
            valid_shows['episode_score_val'] * 0.4 +
            valid_shows['status_score_val'] * 0.2
        ) / 100  # Convert to 0-1 scale
        
        avg_score = valid_shows['longevity_score_calc'].mean()
        confidence = self._get_confidence(sample_size)
        
        details = {
            'avg_seasons': valid_shows['tmdb_seasons'].mean(),
            'avg_episodes': valid_shows['tmdb_total_episodes'].mean(),
            'status_distribution': valid_shows['tmdb_status'].value_counts(normalize=True).mul(100).round(1).to_dict(),
            'sample_size': sample_size
        }
        
        return ComponentScore(
            component=self.component_name,
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )


class NetworkScoreCalculator:
    """Calculates network compatibility and success scores for a set of criteria."""
    
    def __init__(self, criteria_scorer):
        """Initialize the calculator with a reference to the CriteriaScorer.
        
        Args:
            criteria_scorer: The CriteriaScorer instance to use for calculations
        """
        self.criteria_scorer = criteria_scorer
    
    def calculate_network_scores(self, criteria_str) -> List[NetworkMatch]:
        """Calculate network compatibility and success scores for a set of criteria.
        
        Args:
            criteria_str: String representation of criteria dictionary or a dict
            
        Returns:
            List of NetworkMatch objects with compatibility and success scores
        """
        try:
            criteria: Dict[str, Any]
            if isinstance(criteria_str, str):
                try:
                    criteria = ast.literal_eval(criteria_str)
                except (SyntaxError, ValueError) as parse_err:
                    st.warning(f"Invalid criteria string format for network scores: '{criteria_str}'. Error: {parse_err}")
                    st.error(f"Invalid format for criteria input: {str(parse_err)}. Please check the input string.")
                    raise ValueError(f"Invalid criteria string format: {str(parse_err)}") from parse_err
            elif isinstance(criteria_str, dict):
                criteria = criteria_str
            else:
                err_msg = f"Invalid criteria type: {type(criteria_str)}. Must be a dict or a string representation of a dict."
                st.error(err_msg)
                raise ValueError(err_msg)
                
            # Get all networks from the data
            data = self.criteria_scorer.fetch_criteria_data()
            networks = data[['network_id', 'network_name']].drop_duplicates().dropna()
            
            # Prepare results list
            results = []
            
            # For each network, calculate compatibility and success probability
            for _, network in networks.iterrows():
                network_id = network['network_id']
                network_name = network['network_name']
                
                # Skip if network_id is not valid
                if pd.isna(network_id) or pd.isna(network_name):
                    continue
                    
                # Create network-specific criteria
                network_criteria = criteria.copy()
                network_criteria['network'] = int(network_id)
                
                # Get shows matching both criteria and network
                # Use the matching calculator if available, otherwise fall back to direct method
                confidence_info = {
                    'level': 'none',
                    'score': 0.0,
                    'match_quality': 0.0,
                    'sample_size': 0,
                    'match_level': 1
                }
                
                if hasattr(self.criteria_scorer, '_matching_calculator'):
                    # First ensure we have criteria data with success metrics
                    if self.criteria_scorer._matching_calculator._criteria_data is None:
                        self.criteria_scorer._matching_calculator._criteria_data = self.criteria_scorer.fetch_criteria_data(force_refresh=False)
                    # Use flexible matching to get best possible results
                    matching_shows, count, confidence_info = self.criteria_scorer._matching_calculator.get_matching_shows(network_criteria, flexible=True)
                else:
                    # Fall back to original implementation
                    matching_shows, count = self.criteria_scorer._get_matching_shows(network_criteria)
                    confidence_info['sample_size'] = count
                    confidence_info['level'] = OptimizerConfig.get_confidence_level(count, 1)
                
                # Calculate compatibility score (0-1)
                # Use match quality from confidence info if available, no fallback
                compatibility_score = confidence_info.get('match_quality', None)  # No default, will be None if missing
                
                # Calculate success probability if we have enough shows
                if not matching_shows.empty and count >= OptimizerConfig.CONFIDENCE['minimum_sample']:
                    # Check if we have success_score in the matching_shows
                    if 'success_score' not in matching_shows.columns and hasattr(self.criteria_scorer, '_matching_calculator'):
                        # Try to merge success_score from criteria_data
                        if self.criteria_scorer._matching_calculator._criteria_data is not None and 'success_score' in self.criteria_scorer._matching_calculator._criteria_data.columns:
                            matching_shows = matching_shows.merge(
                                self.criteria_scorer._matching_calculator._criteria_data[['id', 'success_score']], 
                                on='id', 
                                how='left'
                            )
                    
                    # Use the matching calculator if available, otherwise fall back to direct method
                    if hasattr(self.criteria_scorer, '_matching_calculator'):
                        success_rate, confidence_info = self.criteria_scorer._matching_calculator.calculate_success_rate(matching_shows, confidence_info=confidence_info)
                    else:
                        success_rate = self.criteria_scorer._calculate_success_rate(matching_shows)
                    confidence = confidence_info.get('level', OptimizerConfig.get_confidence_level(count, confidence_info.get('match_level', 1)))
                else:
                    success_rate = None
                    confidence = 'none'
                
                # Create NetworkMatch object
                network_match = NetworkMatch(
                    network_id=int(network_id),
                    network_name=network_name,
                    compatibility_score=compatibility_score,
                    success_probability=success_rate if success_rate is not None else None,  # Use None instead of 0.0
                    sample_size=count if not matching_shows.empty else 0,
                    confidence=confidence,
                    details={
                        'criteria': criteria,
                        'network_criteria': network_criteria,
                        'confidence_info': confidence_info,
                        'match_level': confidence_info.get('match_level', 1),
                        'match_quality': confidence_info.get('match_quality', None)  # No default
                    }
                )
                
                results.append(network_match)
                
            return results

        except ValueError as ve:
            st.error(f"Calculation Error (Network Scores): {str(ve)}")
            raise
        except Exception as e:
            st.error(f"Optimizer Error: An unexpected error occurred while calculating network scores. Details: {e}")
            raise


class MatchingCalculator:
    """Handles show matching and success rate calculations."""
    
    def __init__(self, criteria_scorer: 'CriteriaScorer'):
        """Initialize the MatchingCalculator.
        
        Args:
            criteria_scorer: CriteriaScorer instance to use for data access
        """
        self.criteria_scorer = criteria_scorer
        self._criteria_data = None  # Cache for criteria data
        self.matcher = Matcher(self.criteria_scorer.field_manager)
        
    def get_criteria_for_match_level(self, criteria: Dict[str, Any], match_level: int) -> Dict[str, Any]:
        """Get a subset of criteria for a specific match level.
        
        Match levels:
        1 - All criteria (strict matching)
        2 - All but one secondary criterion
        3 - Core and primary criteria only
        4 - Only essential and core criteria
        
        Args:
            criteria: Dictionary of criteria
            match_level: Match level (1-4)
            
        Returns:
            Dictionary of criteria for the specified match level
        """
        # If match level is 1, use all criteria
        if match_level == 1:
            return criteria.copy()
            
        # Classify criteria by importance
        classified = self.criteria_scorer.field_manager.classify_criteria_by_importance(criteria)
        
        # For match level 4, use only essential and core criteria
        if match_level == 4:
            result = {}
            result.update(classified['essential'])
            result.update(classified['core'])
            return result
            
        # For match level 3, use essential, core, and primary criteria
        if match_level == 3:
            result = {}
            result.update(classified['essential'])
            result.update(classified['core'])
            result.update(classified['primary'])
            return result
            
        # For match level 2, use all criteria except one secondary criterion (if any)
        if match_level == 2 and classified['secondary']:
            result = {}
            result.update(classified['essential'])
            result.update(classified['core'])
            result.update(classified['primary'])
            
            # Add all but one secondary criterion
            secondary_items = list(classified['secondary'].items())
            for i, (field, value) in enumerate(secondary_items):
                if i < len(secondary_items) - 1:  # Skip the last one
                    result[field] = value
            return result
        
        # Default to all criteria if match level is invalid or no secondary criteria for level 2
        return criteria.copy()
        
    def calculate_match_confidence(self, shows: pd.DataFrame, match_level: int, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics for a match result.
        
        Args:
            shows: DataFrame of matched shows
            match_level: Match level used (1-4)
            criteria: Original criteria dictionary
            
        Returns:
            Dictionary with confidence metrics:
            - level: Confidence level string ('none', 'very_low', 'low', 'medium', 'high')
            - score: Confidence score (0-1)
            - match_quality: Quality of the match based on match level (0-1)
            - sample_size: Number of shows in the sample
            - actual_match_level: The actual match level based on criteria validation
        """
        # Get sample size
        sample_size = len(shows)
        
        # Calculate criteria coverage
        total_criteria = len(OptimizerConfig.CRITERIA_IMPORTANCE)
        criteria_count = len(criteria)
        
        # Calculate match quality based on match level
        match_quality = OptimizerConfig.MATCH_LEVELS.get(match_level, {}).get('min_quality', 0.3)
        
        # Calculate confidence score using OptimizerConfig
        confidence_score = OptimizerConfig.calculate_confidence_score(
            sample_size, criteria_count, total_criteria, match_level)
        
        # Get confidence level string
        confidence_level = OptimizerConfig.get_confidence_level(sample_size, match_level)
        
        # Validate the actual match level by checking if all criteria are truly matched
        # This is especially important for array fields like character_types
        actual_match_level = match_level
        
        # For logging purposes
        import logging
        logger = logging.getLogger(__name__)
        
        # Only perform validation if we have shows and claiming exact match (level 1)
        if not shows.empty and match_level == 1:
            # Get array field mapping to check array fields properly
            array_field_mapping = self.criteria_scorer.field_manager.get_array_field_mapping()
            
            # Check each criterion to see if it's actually matched
            for field_name, value in criteria.items():
                # Skip empty criteria
                if value is None or (isinstance(value, list) and not value):
                    continue
                    
                # Handle array fields differently
                if isinstance(value, list):
                    # Get the correct column name for this field
                    if field_name in array_field_mapping:
                        field_column = array_field_mapping[field_name]
                    elif f"{field_name}_ids" in shows.columns:
                        field_column = f"{field_name}_ids"
                    else:
                        field_column = field_name
                        
                    # Check if this column exists in the data
                    if field_column not in shows.columns:
                        logger.warning(f"Field '{field_column}' not found in shows data")
                        actual_match_level = 2  # Downgrade to level 2 if field is missing
                        continue
                        
                    # Sample the first row to check data format
                    sample = shows[field_column].iloc[0] if not shows.empty else None
                    
                    # Check if all shows actually match this array criterion
                    value_set = set(value)
                    if isinstance(sample, list):
                        # For each show, check if any of the criteria values match
                        all_match = shows[field_column].apply(
                            lambda x: isinstance(x, list) and bool(value_set.intersection(x))
                        ).all()
                    else:
                        # If not stored as lists, check if any match the values
                        all_match = shows[field_column].isin(value).any(axis=0)
                        
                    # If not all shows match this criterion, downgrade the match level
                    if not all_match:
                        logger.info(f"Not all shows match array criterion '{field_name}', downgrading match level")
                        actual_match_level = 2  # Downgrade to level 2
                        break
                        
                # Handle scalar fields
                else:
                    # Determine the actual column name
                    field_id = f"{field_name}_id" if f"{field_name}_id" in shows.columns else field_name
                    
                    # Check if this column exists in the data
                    if field_id not in shows.columns:
                        logger.warning(f"Field '{field_id}' not found in shows data")
                        actual_match_level = 2  # Downgrade to level 2 if field is missing
                        continue
                        
                    # Check if all shows match this scalar criterion
                    if not (shows[field_id] == value).all():
                        logger.info(f"Not all shows match scalar criterion '{field_name}', downgrading match level")
                        actual_match_level = 2  # Downgrade to level 2
                        break
        
        return {
            'level': confidence_level,
            'score': confidence_score,
            'match_quality': match_quality,
            'sample_size': sample_size,
            'match_level': actual_match_level,  # Use the validated match level
            'original_match_level': match_level,  # Keep track of the original level
            'match_level_name': OptimizerConfig.MATCH_LEVELS.get(actual_match_level, {}).get('name', f'Level {actual_match_level}')
        }
    
    def get_matching_shows(self, criteria: Dict[str, Any], flexible: bool = True) -> Tuple[pd.DataFrame, int, Dict[str, Any]]:
        """Get shows matching the given criteria with flexible matching support.
        
        Args:
            criteria: Dictionary of criteria
            flexible: Whether to use flexible matching
            
        Returns:
            Tuple of (matching_shows, match_count, confidence_info)
            - matching_shows: DataFrame of matching shows with 'match_level' column
            - match_count: Total number of matching shows across all levels
            - confidence_info: Dictionary with confidence metrics and match level counts
        """
        import streamlit as st
        
        # Get criteria data, only force refresh if we don't have it yet
        if self._criteria_data is None:
            data = self.criteria_scorer.fetch_criteria_data(force_refresh=False)
            self._criteria_data = data
        else:
            data = self._criteria_data
        
        if data.empty:
            st.error("Empty criteria data from fetch_criteria_data")
            raise ValueError("No criteria data available")
        
        # Get array fields and mapping from field_manager
        array_field_mapping = self.criteria_scorer.field_manager.get_array_field_mapping()
        array_fields = list(array_field_mapping.keys())
        
        # Clean up criteria - remove None or empty values to make matching more lenient
        clean_criteria = {}
        
        for field_name, value in criteria.items():
            # Skip None values and empty lists
            if value is None:
                continue
            if isinstance(value, list) and len(value) == 0:
                continue
            
            # Handle array fields
            if field_name in array_fields:
                # Make sure array field values are always lists
                if not isinstance(value, list):
                    clean_criteria[field_name] = [value]
                else:
                    clean_criteria[field_name] = value
            else:
                # Don't map field names here - let FieldManager handle it
                # This prevents double mapping (e.g., network -> network_id -> network_id_id)
                clean_criteria[field_name] = value
        
        # If we have no valid criteria after cleaning, return all shows
        if not clean_criteria:
            confidence_info = {
                'level': 'high',
                'score': 1.0,
                'match_quality': 1.0,
                'sample_size': len(data),
                'match_level': 1,
                'match_level_name': 'All shows (no criteria)'
            }
            return data, len(data), confidence_info
        
        # If flexible matching is disabled, just do a regular match
        if not flexible:
            try:
                matched_shows, match_count = self.matcher._match_shows(clean_criteria, data)
                
                # Ensure success_score is present
                if not matched_shows.empty and 'success_score' not in matched_shows.columns:
                    if 'success_score' in data.columns and 'id' in matched_shows.columns and 'id' in data.columns:
                        matched_shows = matched_shows.merge(data[['id', 'success_score']], on='id', how='left')
                
                confidence_info = self.calculate_match_confidence(matched_shows, 1, clean_criteria)
                return matched_shows, match_count, confidence_info
            except Exception as e:
                st.error(f"Optimizer Error: An error occurred during show matching. Details: {e}")
                return pd.DataFrame(), 0, {'level': 'none', 'score': 0, 'match_quality': 0, 'sample_size': 0, 'match_level': 0}
        
        # Flexible matching - try different match levels
        min_sample_size = OptimizerConfig.CONFIDENCE['minimum_sample']
        # We'll collect matches from all levels
        all_matches_by_level = {}
        match_counts = {}
        best_level = 0
        confidence_info = {}
        
        # Try each match level, starting with the strictest (level 1)
        import logging
        logger = logging.getLogger(__name__)
        
        # Log the criteria we're trying to match
        logger.info(f"Attempting to match criteria: {clean_criteria}")
        
        # First, collect matches from all levels
        for level in range(1, 5):
            try:
                # Get criteria for this match level
                level_criteria = self.get_criteria_for_match_level(clean_criteria, level)
                
                # Skip if we have no criteria at this level
                if not level_criteria:
                    logger.info(f"No criteria for match level {level}, skipping")
                    continue
                
                # Log what criteria we're using for this level
                logger.info(f"Match level {level} using criteria: {level_criteria}")
                
                # Match shows using the level-specific criteria
                matched_shows, match_count = self.matcher._match_shows(level_criteria, data)
                logger.info(f"Match level {level} found {match_count} shows")
                
                # Store these matches with their level
                if not matched_shows.empty:
                    # Add a match_level column to identify the source level
                    matched_shows['match_level'] = level
                    all_matches_by_level[level] = matched_shows
                    match_counts[level] = match_count
                
                # Calculate confidence metrics for this match
                level_confidence = self.calculate_match_confidence(matched_shows, level, clean_criteria)
                
                # Track the best level for confidence info
                # We'll use the best level that has at least min_sample_size matches
                if match_count >= min_sample_size and (best_level == 0 or level < best_level):
                    best_level = level
                    confidence_info = level_confidence
                    logger.info(f"Setting best confidence level to {level} with {match_count} shows")
                
                # If we have no confidence info yet but have some matches, use this level
                if not confidence_info and match_count > 0:
                    confidence_info = level_confidence
                    best_level = level
                    logger.info(f"Setting initial confidence level to {level} with {match_count} shows")
                
                # If we have enough exact matches, we can skip other levels
                if level == 1 and match_count >= 100:
                    logger.info(f"Found {match_count} exact matches (level 1), which is enough to fill display. Skipping other levels.")
                    break
                    
            except Exception as e:
                logger.warning(f"Match level {level} failed: {e}")
                st.warning(f"Match level {level} failed: {e}")
                continue
        
        # If we found no matches at any level, return empty
        if not all_matches_by_level:
            return pd.DataFrame(), 0, {'level': 'none', 'score': 0, 'match_quality': 0, 'sample_size': 0, 'match_level': 0}
        
        # Create a prioritized list of shows, filling up to 100 slots
        # Start with level 1 (exact matches), then level 2, etc.
        prioritized_shows = pd.DataFrame()
        total_shows = 0
        max_shows = 100  # Maximum number of shows to display
        
        # Track which shows we've already included (by ID or title)
        seen_shows = set()
        
        # Debug the match counts by level
        logger.info(f"Match counts by level: {match_counts}")
        for level_num, level_df in all_matches_by_level.items():
            logger.info(f"Level {level_num} has {len(level_df)} shows with columns: {list(level_df.columns)}")
        
        # Process each match level in priority order
        for level in range(1, 5):
            if level in all_matches_by_level and not all_matches_by_level[level].empty:
                df = all_matches_by_level[level]
                logger.info(f"Processing level {level} with {len(df)} shows")
                
                # Validate the match level for each show to ensure accuracy
                # This is especially important for array fields like character_types
                if level == 1:  # Only validate exact matches (level 1)
                    # Calculate actual match level for each show
                    confidence_info = self.calculate_match_confidence(df, level, clean_criteria)
                    actual_match_level = confidence_info.get('match_level', level)
                    
                    # If the actual match level is different from the claimed level,
                    # update the match_level column in the DataFrame
                    if actual_match_level != level:
                        logger.info(f"Updating match level from {level} to {actual_match_level} for some shows")
                        df['match_level'] = actual_match_level
                        
                        # Move these shows to their correct level bucket
                        if actual_match_level not in all_matches_by_level:
                            all_matches_by_level[actual_match_level] = df
                        else:
                            all_matches_by_level[actual_match_level] = pd.concat(
                                [all_matches_by_level[actual_match_level], df], ignore_index=True)
                        
                        # Skip these shows at the current level since they've been moved
                        continue
                
                # Deduplicate based on ID or title
                if 'id' in df.columns:
                    # Filter out shows we've already seen by ID
                    new_shows = df[~df['id'].isin(seen_shows)]
                    logger.info(f"After ID deduplication: {len(new_shows)} shows remain")
                    if not new_shows.empty:
                        # Update our tracking set with these IDs
                        seen_shows.update(new_shows['id'].tolist())
                elif 'title' in df.columns:
                    # Filter out shows we've already seen by title
                    new_shows = df[~df['title'].isin(seen_shows)]
                    logger.info(f"After title deduplication: {len(new_shows)} shows remain")
                    if not new_shows.empty:
                        # Update our tracking set with these titles
                        seen_shows.update(new_shows['title'].tolist())
                else:
                    # No way to deduplicate, just use all shows
                    new_shows = df
                    logger.info("No ID or title column for deduplication, using all shows")
                
                # Calculate how many more shows we can add
                slots_remaining = max_shows - total_shows
                
                if slots_remaining <= 0:
                    # We've already filled all slots
                    break
                    
                # Take only as many shows as we have slots for
                if len(new_shows) > slots_remaining:
                    new_shows = new_shows.head(slots_remaining)
                
                # Add these shows to our prioritized list
                prioritized_shows = pd.concat([prioritized_shows, new_shows], ignore_index=True)
                total_shows += len(new_shows)
                
                logger.info(f"Added {len(new_shows)} shows from match level {level}. Total now: {total_shows}/{max_shows}")
                
                # If we've filled all slots, we're done
                if total_shows >= max_shows:
                    break
        
        # Ensure success_score is present in the prioritized shows
        if 'success_score' not in prioritized_shows.columns:
            if 'success_score' in data.columns and 'id' in prioritized_shows.columns and 'id' in data.columns:
                prioritized_shows = prioritized_shows.merge(data[['id', 'success_score']], on='id', how='left')
        
        # Update confidence info with match counts by level
        confidence_info['match_counts_by_level'] = match_counts
        confidence_info['total_matches'] = sum(match_counts.values()) if match_counts else 0
        
        # Return the prioritized shows, total count across all levels, and confidence info
        return prioritized_shows, sum(match_counts.values()) if match_counts else 0, confidence_info
    
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
            confidence_info = {
                'level': 'medium',
                'score': 0.5,
                'match_quality': 1.0,
                'sample_size': len(shows) if not shows.empty else 0,
                'match_level': 1,
                'match_level_name': 'Standard match'
            }
                
        # Check if we have shows to analyze
        if shows.empty:
            # No shows to analyze
            confidence_info['success_rate'] = None
            confidence_info['success_count'] = 0
            confidence_info['total_count'] = 0
            return None, confidence_info
                
        # Check if success_score is present
        if 'success_score' not in shows.columns:
            # success_score column missing from shows
            # Try to get it from criteria_data if available
            if self._criteria_data is not None and 'success_score' in self._criteria_data.columns:
                # We have criteria_data with success_score, try to merge
                if 'id' in shows.columns and 'id' in self._criteria_data.columns:
                    # Merge success_score from criteria_data
                    shows = shows.merge(self._criteria_data[['id', 'success_score']], on='id', how='left')
                else:
                    # Can't merge, no common key
                    confidence_info['success_rate'] = None
                    confidence_info['success_count'] = 0
                    confidence_info['total_count'] = 0
                    return None, confidence_info
            else:
                # No criteria_data or no success_score in criteria_data
                confidence_info['success_rate'] = None
                confidence_info['success_count'] = 0
                confidence_info['total_count'] = 0
                return None, confidence_info
                
        # Check again if success_score is present after potential merge
        if 'success_score' not in shows.columns:
            # Still missing success_score
            confidence_info['success_rate'] = None
            confidence_info['success_count'] = 0
            confidence_info['total_count'] = 0
            return None, confidence_info
                
        # Filter out shows with zero or missing success scores
        valid_shows = shows[shows['success_score'].notna() & (shows['success_score'] > 0)]
            
        if valid_shows.empty:
            # No valid shows after filtering
            confidence_info['success_rate'] = 0.0
            confidence_info['success_count'] = 0
            confidence_info['total_count'] = 0
            return 0.0, confidence_info
        
        # Use default threshold if none provided
        if threshold is None:
            threshold = OptimizerConfig.PERFORMANCE['success_threshold']
        
        # Get success score range and distribution
        min_score = valid_shows['success_score'].min()
        max_score = valid_shows['success_score'].max()
        mean_score = valid_shows['success_score'].mean()
        
        # Normalize threshold if scores are on 0-100 scale
        normalized_threshold = threshold
        
        # Check if scores need normalization (0-100 scale)
        if max_score > 1.0:  # If scores are on 0-100 scale
            normalized_threshold = threshold * 100
        elif threshold > 1.0:  # If scores are already on 0-1 scale but threshold is on 0-100 scale
            normalized_threshold = threshold / 100
        
        # Count successful shows (those with score >= threshold)
        successful = valid_shows[valid_shows['success_score'] >= normalized_threshold]
        success_count = len(successful)
        total_count = len(valid_shows)
        
        success_rate = success_count / total_count if total_count > 0 else 0.0
        
        # Update confidence info with success metrics
        confidence_info['success_rate'] = success_rate
        confidence_info['success_count'] = success_count
        confidence_info['total_count'] = total_count
        confidence_info['min_score'] = float(min_score)
        confidence_info['max_score'] = float(max_score)
        confidence_info['mean_score'] = float(mean_score)
        
        # Adjust confidence based on sample size
        if total_count < OptimizerConfig.CONFIDENCE['minimum_sample']:
            confidence_info['level'] = 'low'
            confidence_info['score'] *= 0.7  # Reduce confidence for small samples
        
        return success_rate, confidence_info

    def batch_calculate_success_rates(self, criteria_list: List[Dict[str, Any]], flexible: bool = True) -> List[Tuple[Optional[float], Dict[str, Any]]]:
        """Batch calculate success rates for multiple criteria with confidence information.
        
        Args:
            criteria_list: List of criteria dictionaries
            flexible: Whether to use flexible matching (try different match levels)
            
        Returns:
            List of tuples (success_rate, confidence_info) in the same order as criteria_list
        """
        # Ensure we have criteria data with success metrics, but don't force refresh
        if self._criteria_data is None:
            try:
                self._criteria_data = self.criteria_scorer.fetch_criteria_data(force_refresh=False)
            except Exception as e:
                st.error(f"Error fetching criteria data: {e}")
            
        results = []
        for criteria in criteria_list:
            try:
                # Get matching shows with flexible matching
                shows, count, confidence_info = self.get_matching_shows(criteria, flexible=flexible)
                
                # Calculate success rate with confidence information
                success_rate, confidence_info = self.calculate_success_rate(shows, confidence_info=confidence_info)
                
                # Add the result with confidence info
                results.append((success_rate, confidence_info))
            except Exception as e:
                st.warning(f"Error calculating success rate for criteria: {e}")
                # Return None with basic confidence info for errors
                error_confidence = {
                    'level': 'none',
                    'score': 0.0,
                    'match_quality': 0.0,
                    'sample_size': 0,
                    'match_level': 0,
                    'success_rate': None,
                    'success_count': 0,
                    'total_count': 0,
                    'error': str(e)
                }
                results.append((None, error_confidence))
        
        return results
