"""CriteriaScorer: Calculates raw success scores for show criteria.

This component calculates success rates for different criteria based on
historical show data. It integrates with SuccessAnalyzer for base success metrics
and provides data to CriteriaAnalyzer for pattern recognition.

Key concepts:

1. Success Scoring:
   - Calculate success rates for each criteria value (genre, source type, etc.)
   - Analyze network-specific success patterns
   - Weight different success components (audience, critics, longevity)

2. Data Flow:
   - Get criteria from FieldManager
   - Get success metrics from SuccessAnalyzer
   - Calculate raw scores based on OptimizerConfig weights
   - Provide scores to CriteriaAnalyzer

3. Performance:
   - Cache common calculations
   - Support incremental updates for real-time UI
   - Handle partial criteria sets
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
import pandas as pd
import numpy as np
import logging
from functools import lru_cache, partial
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .optimizer_config import OptimizerConfig
from .field_manager import FieldManager

logger = logging.getLogger(__name__)

__all__ = ['CriteriaScorer', 'ComponentScore', 'NetworkMatch', 'ScoreCalculationError']

@dataclass
class ComponentScore:
    """Success score for a component (audience, critics, longevity)."""
    component: str  # audience, critics, or longevity
    score: float  # 0-1 score
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
    def calculate(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate the component score."""
        pass
    
    def _get_confidence(self, sample_size: int) -> str:
        """Get confidence level based on sample size."""
        return OptimizerConfig.get_confidence_level(sample_size)

class SuccessScoreCalculator(ScoreCalculator):
    """Calculate score based on success metrics."""
    
    def calculate(self, shows: pd.DataFrame) -> ComponentScore:
        if shows.empty:
            raise ScoreCalculationError(f"Cannot calculate {self.component_name} score with empty shows DataFrame")
        
        # Filter shows with success metrics
        valid_shows = shows[shows['success_score'].notna()]
        sample_size = len(valid_shows)
        
        if sample_size == 0:
            raise ScoreCalculationError(f"No shows with valid success_score data found for {self.component_name}")
        
        # Calculate metrics
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
    
    def calculate(self, shows: pd.DataFrame) -> ComponentScore:
        if shows.empty:
            raise ScoreCalculationError("Cannot calculate audience score with empty shows DataFrame")
            
        if 'popcornmeter' not in shows.columns:
            raise ScoreCalculationError("popcornmeter column not found in shows data")
            
        # Filter shows with audience metrics
        valid_shows = shows[shows['popcornmeter'].notna()]
        sample_size = len(valid_shows)
        
        if sample_size == 0:
            raise ScoreCalculationError("No shows with valid popcornmeter data found")
            
        # Calculate average popcornmeter score (normalized to 0-1)
        avg_score = valid_shows['popcornmeter'].mean() / 100  # Normalize to 0-1
        confidence = self._get_confidence(sample_size)
        
        return ComponentScore(
            component='audience',
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details={'popcornmeter': avg_score}
        )

# Similar classes for CriticsScoreCalculator and LongevityScoreCalculator would follow
# ...


@dataclass
class NetworkMatch:
    """Network match information with success metrics."""
    network_id: int
    network_name: str
    compatibility_score: float  # 0-1 score of how well the network matches criteria
    success_probability: float  # 0-1 probability of success on this network
    sample_size: int  # Number of shows in the sample
    confidence: str  # none, low, medium, high

    details: Dict[str, Any] = field(default_factory=dict)  # Detailed breakdown of score


class CriteriaScorer:
    """Calculates raw success scores for show criteria."""
    
    def __init__(self, shows_analyzer: ShowsAnalyzer, success_analyzer: SuccessAnalyzer):
        """Initialize the criteria scorer.
        
        Args:
            shows_analyzer: ShowsAnalyzer instance for show data
            success_analyzer: SuccessAnalyzer instance for success metrics
        """
        self.shows_analyzer = shows_analyzer
        self.success_analyzer = success_analyzer
        
        # Get reference data from ShowsAnalyzer using fetch_comp_data
        try:
            comp_df, reference_data = shows_analyzer.fetch_comp_data()
            self.field_manager = FieldManager(reference_data)
        except Exception as e:
            logger.error(f"Error initializing FieldManager: {e}")
            # Initialize with empty reference data as fallback
            self.field_manager = FieldManager({})
            
        self.criteria_data = None
        self.last_update = None
        self.cache_duration = OptimizerConfig.PERFORMANCE['cache_duration']
        
    def fetch_criteria_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch show criteria data from ShowsAnalyzer.
        
        Args:
            force_refresh: Whether to force a refresh of the data
            
        Returns:
            DataFrame with show criteria data
            
        Raises:
            ValueError: If data cannot be fetched or is invalid
        """
        import streamlit as st
        
        # Check cache first
        current_time = datetime.now()
        if not force_refresh and self.criteria_data is not None and \
           self.last_update is not None and \
           (current_time - self.last_update) < timedelta(seconds=self.cache_duration):
            logger.debug("Using cached criteria data")
            return self.criteria_data
            
        logger.debug("Fetching fresh criteria data")
        
        try:
            # Get show criteria data from ShowsAnalyzer
            comp_df, reference_data = self.shows_analyzer.fetch_comp_data()
            
            if comp_df.empty:
                raise ValueError("No data returned from ShowsAnalyzer.fetch_comp_data()")
                
            # Get success metrics from SuccessAnalyzer
            success_df = self.success_analyzer.fetch_success_data()
            
            if not success_df.empty:
                # Success data is indexed by show_id, so we need to merge on that
                # Create a temporary copy of comp_df with show_id as index for merging
                comp_df_indexed = comp_df.set_index('id', drop=False)
                
                # Get all required success metrics from success_df
                required_metrics = ['success_score', 'popcornmeter', 'tomatometer', 'has_rt']
                available_metrics = [col for col in required_metrics if col in success_df.columns]
                
                if not available_metrics:
                    logger.error("No required metrics found in success data")
                else:
                    # Merge the success metrics into comp_df
                    success_metrics = success_df[available_metrics]
                    comp_df_indexed = comp_df_indexed.join(success_metrics, how='left')
                    
                    # Reset the index and update comp_df
                    comp_df = comp_df_indexed.reset_index(drop=True)
                    
                    # Log which metrics were added
                    for metric in available_metrics:
                        logger.debug(f"Added {metric} from SuccessAnalyzer to {comp_df[metric].notna().sum()} shows")
            else:
                logger.error("No success data available from SuccessAnalyzer")
                
            # Update field manager with new reference data
            self.field_manager = FieldManager(reference_data)
            
            # Validate required columns
            if 'id' not in comp_df.columns:
                raise ValueError("Show ID column 'id' not found in criteria data")
                
            # Clean up any duplicate columns from merge
            duplicate_cols = [col for col in comp_df.columns if col.endswith('_orig') or col.endswith('_success')]
            if duplicate_cols:
                logger.warning(f"Duplicate columns created during merge: {duplicate_cols}")
                # Clean up duplicate columns by keeping the non-null version
                processed_base_cols = set()  # Track which base columns we've already processed
                
                for col in duplicate_cols:
                    if col.endswith('_orig'):
                        base_col = col[:-5]  # Remove _orig suffix
                        success_col = f"{base_col}_success"
                        
                        # Skip if we've already processed this base column
                        if base_col in processed_base_cols:
                            continue
                        processed_base_cols.add(base_col)
                        
                        # Only process if both versions exist
                        if success_col in comp_df.columns:
                            # Keep the non-null version of the data
                            comp_df[base_col] = comp_df[base_col].fillna(comp_df[success_col])
                            comp_df = comp_df.drop(columns=[success_col])
                        comp_df = comp_df.drop(columns=[col])
            
            # Verify that all required columns exist after cleanup
            required_columns = ['id']  # Add other required columns as needed
            for col in required_columns:
                if col not in comp_df.columns:
                    logger.error(f"Required column '{col}' missing after duplicate cleanup")
                    # Try to recover from suffix columns if they still exist
                    orig_col = f"{col}_orig"
                    success_col = f"{col}_success"
                    
                    if orig_col in comp_df.columns:
                        comp_df[col] = comp_df[orig_col]
                        logger.debug(f"Recovered '{col}' from '{orig_col}'")
                        comp_df = comp_df.drop([orig_col], axis=1, errors='ignore')
                    elif success_col in comp_df.columns:
                        comp_df[col] = comp_df[success_col]
                        logger.debug(f"Recovered '{col}' from '{success_col}'")
                        comp_df = comp_df.drop([success_col], axis=1, errors='ignore')
            
            # Drop rows with missing success scores to make issues visible
            if 'success_score' in comp_df.columns:
                missing_scores = comp_df['success_score'].isna().sum()
                if missing_scores > 0:
                    logger.debug(f"Dropping {missing_scores} shows with missing success scores")
                    comp_df = comp_df.dropna(subset=['success_score'])
            
            # Log the columns in the merged data
            logger.debug(f"Merged criteria data columns: {list(comp_df.columns)}")
            logger.debug(f"Merged data has {len(comp_df)} rows")
            
            # Check if we have the required metrics columns
            for col in ['popcornmeter', 'tomatometer']:
                if col not in comp_df.columns:
                    logger.warning(f"'{col}' column missing from merged data")
                else:
                    logger.debug(f"'{col}' column present in merged data")
                    logger.debug(f"'{col}' non-null count: {comp_df[col].notna().sum()}")
            
            # Store the data and update timestamp
            self.criteria_data = comp_df
            self.last_update = datetime.now()
            
            # Cache the criteria data
            self._criteria_data = self.criteria_data
            
            return self.criteria_data
            
        except Exception as e:
            st.error(f"Error fetching criteria data: {str(e)}")
            raise
        
        return self.criteria_data
    
    def _get_matching_shows(self, criteria: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Get shows matching the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Tuple of (DataFrame of matching shows with success metrics, count of matches)
        """
        data = self.fetch_criteria_data()
        if data.empty:
            logger.error("Empty criteria data from fetch_criteria_data")
            raise ValueError("No criteria data available")
        
        # Get array fields and mapping from field_manager
        array_field_mapping = self.field_manager.get_array_field_mapping()
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
                clean_criteria[field_name] = value
        
        # If we have no valid criteria after cleaning, return all shows
        if not clean_criteria:
            logger.warning("No valid criteria after cleaning, returning all shows")
            return data, len(data)
        
        # Use FieldManager to match shows against criteria
        try:
            matched_shows, match_count = self.field_manager.match_shows(clean_criteria, data)
            
            if matched_shows.empty:
                # Provide a more specific error message with the criteria that failed to match
                criteria_str = ", ".join([f"{k}: {v}" for k, v in clean_criteria.items()])
                logger.error(f"No shows matched the criteria: {criteria_str}")
                
                # Log the criteria and available columns for debugging
                logger.warning(f"No shows matched criteria: {clean_criteria}")
                logger.info(f"Available columns: {list(data.columns)}")
                
                # Return empty DataFrame with zero matches
                # The calling code should handle this appropriately
                return matched_shows, 0
                
            return matched_shows, match_count
        except Exception as e:
            logger.error(f"Error matching shows: {e}", exc_info=True)
            # Return empty DataFrame with zero matches
            # The calling code should handle this appropriately
            return pd.DataFrame(), 0   
    def _calculate_success_rate(self, shows: pd.DataFrame, threshold: Optional[float] = None) -> Optional[float]:
        """Calculate the success rate for a set of shows.
        
        Args:
            shows: DataFrame of shows
            threshold: Success threshold (shows with score >= threshold are considered successful)
            
        Returns:
            Success rate (0-1) or None if success_score is missing
        """
        # Use threshold from OptimizerConfig if not provided
        if threshold is None:
            from .optimizer_config import OptimizerConfig
            threshold = OptimizerConfig.THRESHOLDS['success_threshold']
        import streamlit as st
        
        if shows.empty:
            logger.error("Empty shows DataFrame provided to _calculate_success_rate")
            return None
        
        if 'success_score' not in shows.columns:
            logger.error("'success_score' column missing from shows data")
            return None
        
        # Filter out shows with missing success scores
        shows_with_scores = shows[shows['success_score'].notna()]
        
        if len(shows_with_scores) == 0:
            logger.error("No shows with valid success scores found")
            raise ValueError("No shows with valid success scores available")
        
        # Get success score range and distribution for logging
        min_score = shows_with_scores['success_score'].min()
        max_score = shows_with_scores['success_score'].max()
        mean_score = shows_with_scores['success_score'].mean()
        
        logger.debug(f"Success score range: {min_score} to {max_score}, mean: {mean_score}")
        
        # Normalize threshold if scores are on 0-100 scale
        normalized_threshold = threshold
        normalized_scores = shows_with_scores['success_score'].copy()
        
        # Check if scores need normalization (0-100 scale)
        if max_score > 1.0:  # If scores are on 0-100 scale
            normalized_threshold = threshold * 100
        else:  # If scores are already on 0-1 scale but threshold is on 0-100 scale
            if threshold > 1.0:
                normalized_threshold = threshold / 100
        
        # Count successful shows (those with score >= threshold)
        successful = shows_with_scores[shows_with_scores['success_score'] >= normalized_threshold]
        success_count = len(successful)
        total_count = len(shows_with_scores)
        
        logger.debug(f"Success rate calculation - {success_count} successful shows out of {total_count} total (threshold: {threshold})")
        
        success_rate = success_count / total_count
        logger.debug(f"Success rate: {success_rate:.4f} ({success_rate*100:.1f}%)")
        
        return success_rate
    
    @lru_cache(maxsize=32)
    def calculate_criteria_impact(self, base_criteria: Dict[str, Any], field_name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Calculate the impact of criteria values on success rate.
        
        Args:
            base_criteria: The base criteria to compare against.
            field_name: Optional name of the field to analyze. If None, analyzes all fields.
            
        Returns:
            A dictionary mapping field names to dictionaries of option IDs to impact scores.
        """
        try:
            # Get the field manager's array field mapping
            array_field_mapping = self.field_manager.get_array_field_mapping()
            array_fields = list(array_field_mapping.keys())
            
            # Get base success rate
            base_shows, base_match_count = self._get_matching_shows(base_criteria)
            
            if base_shows.empty:
                st.error("ERROR: No shows matched the base criteria")
                raise ValueError("Cannot calculate impact scores with no matching shows")
                
            if base_match_count < OptimizerConfig.CONFIDENCE['minimum_sample']:
                st.error(f"ERROR: Insufficient sample size for base criteria: {base_match_count} shows")
                raise ValueError(f"Cannot calculate impact scores with insufficient sample size ({base_match_count} shows)")
            
            base_rate = self._calculate_success_rate(base_shows)
            
            if base_rate is None:
                st.warning("WARNING: Unable to calculate base success rate - success_score data missing")
                return {}
            
            if base_rate == 0:
                st.error("ERROR: Base success rate is zero")
                raise ValueError("Cannot calculate impact scores with zero base success rate")
            
            impact_scores = {}
            
            # Determine which fields to process
            fields_to_process = [field_name] if field_name else self.field_manager.FIELD_CONFIGS.keys()
            
            for current_field in fields_to_process:
                # Skip fields already in base criteria
                if current_field in base_criteria:
                    continue
                
                is_array_field = current_field in array_fields
                options = self.field_manager.get_options(current_field)
                
                # Prepare batch criteria for all options
                batch_criteria = []
                option_data = []
                
                for option in options:
                    new_criteria = base_criteria.copy()
                    if is_array_field:
                        new_criteria[current_field] = [option.id]
                    else:
                        new_criteria[current_field] = option.id
                    batch_criteria.append(new_criteria)
                    option_data.append((option.id, option.name))
                
                # Batch calculate success rates
                success_rates = self._batch_calculate_success_rates(batch_criteria)
                
                # Process results
                field_impact = {}
                field_sample_sizes = {}
                
                # Get sample sizes for each option
                for i, criteria_set in enumerate(batch_criteria):
                    # Get the option ID and name
                    option_id, option_name = option_data[i]
                    rate = success_rates[i]
                    
                    if rate is not None:
                        # Get the sample size for this criteria
                        option_shows, match_count = self._get_matching_shows(criteria_set)
                        
                        # Calculate impact as relative change in success rate
                        impact = (rate - base_rate) / base_rate if base_rate != 0 else 0
                        
                        # Ensure option_id is hashable (not a dict or list)
                        if isinstance(option_id, (dict, list)):
                            # Convert to a string representation for dict/list option_ids
                            hashable_id = str(option_id)
                            logger.warning(f"Converting unhashable option_id to string: {option_id} -> {hashable_id}")
                        else:
                            hashable_id = option_id
                            
                        # Store both impact and sample size
                        field_impact[hashable_id] = {"impact": impact, "sample_size": match_count}
                
                if field_impact:  # Only add if we have valid impacts
                    impact_scores[current_field] = field_impact
                    
            if not impact_scores:
                raise ValueError("Could not calculate any impact scores with the given criteria")
                
            return impact_scores
            
        except Exception as e:
            st.error(f"Error calculating criteria impact: {str(e)}")
            raise
    
    def calculate_component_scores(self, criteria: Dict[str, Any]) -> Dict[str, 'ComponentScore']:
        """Calculate component scores for a set of criteria.
        
        Args:
            criteria: Dictionary of criteria to calculate scores for
            
        Returns:
            Dictionary mapping component names to ComponentScore objects
            
        Raises:
            ValueError: If no component scores could be calculated
        """
        import streamlit as st
        
        # Get matching shows first
        try:
            matching_shows, match_count = self._get_matching_shows(criteria)
            if matching_shows.empty:
                st.error("No matching shows found for the given criteria")
                raise ValueError("No matching shows found for the given criteria")
                
            component_scores = {}
            
            # Calculate each component score with individual error handling
            for component in ['audience', 'critics', 'longevity']:
                try:
                    score_method = getattr(self, f'_calculate_{component}_score')
                    score = score_method(matching_shows)
                    component_scores[component] = score
                except Exception as e:
                    logger.error(f"Error calculating {component} score: {str(e)}", exc_info=True)
            
            # Check if we have all required component scores
            required_components = ['audience', 'critics', 'longevity']
            missing_components = [c for c in required_components if c not in component_scores]
            
            if missing_components:
                logger.error(f"Failed to calculate required component scores: {missing_components}")
                raise ValueError(f"Failed to calculate required component scores: {', '.join(missing_components)}")
            
            logger.debug(f"Final component scores: {component_scores}")
            return component_scores
            
        except Exception as e:
            st.error(f"Error in calculate_component_scores: {str(e)}")
            raise
    
    def _calculate_audience_score(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate audience score for a set of shows using popcornmeter.
        
        Args:
            shows: DataFrame of shows with 'popcornmeter' column
            
        Returns:
            ComponentScore for audience
            
        Raises:
            ValueError: If popcornmeter data is missing or empty
        """
        import streamlit as st
        
        if shows.empty:
            logger.error("Empty shows DataFrame provided to _calculate_audience_score")
            raise ValueError("Cannot calculate audience score with empty shows DataFrame")
        
        # Check if popcornmeter column exists
        if 'popcornmeter' not in shows.columns:
            logger.error(f"Popcornmeter column missing from shows data. Available columns: {list(shows.columns)}")
            raise ValueError("popcornmeter column is required for audience score calculation")
            
        # Filter shows with audience metrics
        audience_shows = shows[shows['popcornmeter'].notna()]
        sample_size = len(audience_shows)
        
        if sample_size == 0:
            logger.error("No shows with valid popcornmeter data found")
            raise ValueError("No shows with valid popcornmeter data available")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Calculate average popcornmeter score (normalized to 0-1)
        avg_popcorn = audience_shows['popcornmeter'].mean() / 100
        
        # Prepare score details
        details = {
            'popcornmeter': avg_popcorn
        }
        
        return ComponentScore(
            component='audience',
            score=avg_popcorn,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
        
    def _calculate_critics_score(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate critics score for a set of shows using tomatometer.
        
        Args:
            shows: DataFrame of shows with 'tomatometer' column
            
        Returns:
            ComponentScore for critics
            
        Raises:
            ValueError: If tomatometer data is missing or empty
        """
        import streamlit as st
        
        if shows.empty:
            logger.error("Empty shows DataFrame provided to _calculate_critics_score")
            raise ValueError("Cannot calculate critics score with empty shows DataFrame")
        
        # Check if tomatometer column exists
        if 'tomatometer' not in shows.columns:
            logger.error(f"Tomatometer column missing from shows data. Available columns: {list(shows.columns)}")
            raise ValueError("tomatometer column is required for critics score calculation")
            
        # Filter shows with critics metrics
        critics_shows = shows[shows['tomatometer'].notna()]
        sample_size = len(critics_shows)
        
        if sample_size == 0:
            logger.error("No shows with valid tomatometer data found")
            raise ValueError("No shows with valid tomatometer data available")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Calculate average critics score (normalized to 0-1)
        avg_score = critics_shows['tomatometer'].mean() / 100.0
        
        # Prepare score details
        details = {
            'tomatometer': avg_score,
            'sample_size': sample_size
        }
        
        return ComponentScore(
            component='critics',
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
        

    def _calculate_longevity_score(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate longevity score for a set of shows using TMDB metrics.
        
        Longevity is calculated based on:
        - Number of seasons (weight: 40%)
        - Number of episodes (weight: 40%)
        - Show status (weight: 20%)
        
        Args:
            shows: DataFrame containing show data with TMDB metrics
            
        Returns:
            ComponentScore for longevity (0-1 scale)
            
        Raises:
            ValueError: If required TMDB metrics are missing or empty
        """
        import streamlit as st
        
        if shows.empty:
            st.error("DEBUG ERROR: Empty shows DataFrame provided to _calculate_longevity_score")
            raise ValueError("Cannot calculate longevity score with empty shows DataFrame")
        
        # Check for required columns
        required_columns = ['tmdb_seasons', 'tmdb_total_episodes', 'tmdb_status']
        missing_columns = [col for col in required_columns if col not in shows.columns]
        if missing_columns:
            st.error(f"DEBUG ERROR: Missing required columns for longevity calculation: {missing_columns}")
            st.error("DEBUG ERROR: Available columns: " + str(list(shows.columns)))
            raise ValueError(f"Missing required columns for longevity calculation: {', '.join(missing_columns)}")
        
        # Filter shows with required metrics
        valid_shows = shows.dropna(subset=required_columns)
        sample_size = len(valid_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid TMDB metrics found for longevity calculation")
            raise ValueError("No shows with valid TMDB metrics available for longevity calculation")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Calculate season score (0-100 scale)
        def calculate_season_score(seasons):
            if seasons >= 2:
                return min(100, 50 + (seasons - 2) * 10)  # 50 for 2 seasons, +10 per additional season, max 100
            return seasons * 25  # 25 for 1 season, 0 for 0 seasons
        
        # Calculate episode score (0-100 scale)
        def calculate_episode_score(episodes):
            if episodes >= 10:
                return 100
            elif episodes >= 5:
                return 50 + (episodes - 5) * 10  # 50 for 5 episodes, +10 per episode up to 100
            return episodes * 10  # 10 points per episode up to 5
        
        # Calculate status score (0-100 scale)
        status_scores = {
            'Returning Series': 100,
            'Ended': 75,
            'Canceled': 25,
            'In Production': 50,
            'Pilot': 10,
            'In Development': 5
        }
        
        # Calculate scores for each show
        valid_shows = valid_shows.copy()
        valid_shows['season_score'] = valid_shows['tmdb_seasons'].apply(calculate_season_score)
        valid_shows['episode_score'] = valid_shows['tmdb_total_episodes'].apply(calculate_episode_score)
        valid_shows['status_score'] = valid_shows['tmdb_status'].map(status_scores).fillna(0)
        
        # Calculate weighted average (40% season, 40% episode, 20% status)
        valid_shows['longevity_score'] = (
            valid_shows['season_score'] * 0.4 +
            valid_shows['episode_score'] * 0.4 +
            valid_shows['status_score'] * 0.2
        ) / 100  # Convert to 0-1 scale
        
        # Calculate average longevity score across all shows
        avg_score = valid_shows['longevity_score'].mean()
        
        # Prepare score details
        details = {
            'avg_seasons': valid_shows['tmdb_seasons'].mean(),
            'avg_episodes': valid_shows['tmdb_total_episodes'].mean(),
            'status_distribution': valid_shows['tmdb_status'].value_counts().to_dict(),
            'sample_size': sample_size
        }
        
        return ComponentScore(
            component='longevity',
            score=avg_score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
        

    def _batch_calculate_success_rates(self, criteria_list: List[Dict[str, Any]]) -> List[Optional[float]]:
        """Batch calculate success rates for multiple criteria.
        
        Args:
            criteria_list: List of criteria dictionaries
            
        Returns:
            List of success rates in the same order as criteria_list, with None for missing data
        """
        results = []
        for criteria in criteria_list:
            try:
                shows, count = self._get_matching_shows(criteria)
                if not shows.empty and count >= OptimizerConfig.CONFIDENCE['minimum_sample']:
                    success_rate = self._calculate_success_rate(shows)
                    results.append(success_rate)  # This might be None if success_score is missing
                else:
                    results.append(None)
            except Exception as e:
                import streamlit as st
                import traceback
                st.error(f"Error calculating success rate: {str(e)}")
                st.error(f"Traceback: {traceback.format_exc()}")
                results.append(None)
                
        return results
    
    def get_criteria_confidence(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence levels for criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary with confidence information
        """
        return self.field_manager.calculate_confidence(criteria)

    def analyze_criteria_confidence(self, criteria: Dict[str, Any]) -> Dict[str, str]:
        """Analyze confidence levels for criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary of confidence levels by component
        """
        return self.field_manager.calculate_confidence(criteria)
        
    def calculate_network_scores(self, criteria_str: str) -> List[NetworkMatch]:
        """Calculate network compatibility and success scores for a set of criteria.
        
        Args:
            criteria_str: String representation of criteria dictionary
            
        Returns:
            List of NetworkMatch objects with compatibility and success scores
        """
        import streamlit as st
        import ast
        import traceback
        
        try:
            # Parse criteria string back to dictionary if needed
            if isinstance(criteria_str, str):
                criteria = ast.literal_eval(criteria_str)
            else:
                criteria = criteria_str
                
            # Get all networks from the data
            data = self.fetch_criteria_data()
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
                matching_shows, count = self._get_matching_shows(network_criteria)
                
                # Calculate compatibility score (0-1)
                # Simple version: percentage of criteria that match the network's typical shows
                compatibility_score = 0.5  # Default medium compatibility
                
                # Calculate success probability if we have enough shows
                if not matching_shows.empty and count >= OptimizerConfig.CONFIDENCE['minimum_sample']:
                    success_rate = self._calculate_success_rate(matching_shows)
                    confidence = OptimizerConfig.get_confidence_level(count)
                else:
                    success_rate = None
                    confidence = 'none'
                
                # Create NetworkMatch object
                network_match = NetworkMatch(
                    network_id=int(network_id),
                    network_name=network_name,
                    compatibility_score=compatibility_score,
                    success_probability=success_rate if success_rate is not None else 0.0,
                    sample_size=count if not matching_shows.empty else 0,
                    confidence=confidence,
                    details={
                        'criteria': network_criteria,
                        'matching_shows': count if not matching_shows.empty else 0
                    }
                )
                
                results.append(network_match)
                
            return results
            
        except Exception as e:
            st.error(f"Error in calculate_network_scores: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")
            return []
