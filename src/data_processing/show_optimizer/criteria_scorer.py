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
            logger.warning("popcornmeter column not found, falling back to success score")
            return SuccessScoreCalculator('audience').calculate(shows)
            
        # Filter shows with audience metrics
        valid_shows = shows[shows['popcornmeter'].notna()]
        sample_size = len(valid_shows)
        
        if sample_size == 0:
            logger.warning("No shows with valid popcornmeter data, falling back to success score")
            return SuccessScoreCalculator('audience').calculate(shows)
            
        # Calculate metrics
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


@dataclass
class ComponentScore:
    """Success score for a component (audience, critics, longevity)."""
    component: str  # audience, critics, or longevity
    score: float  # 0-1 score
    sample_size: int
    confidence: str  # none, low, medium, high
    details: Dict[str, float] = field(default_factory=dict)  # Detailed breakdown of score


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
            st.write("DEBUG: Using cached criteria data")
            return self.criteria_data
            
        st.write("DEBUG: Fetching fresh criteria data")
        
        try:
            # Get show criteria data from ShowsAnalyzer
            comp_df, reference_data = self.shows_analyzer.fetch_comp_data()
            
            if comp_df.empty:
                raise ValueError("No data returned from ShowsAnalyzer.fetch_comp_data()")
                
            # Update field manager with new reference data
            self.field_manager = FieldManager(reference_data)
            
            # Validate required columns
            if 'id' not in comp_df.columns:
                raise ValueError("Show ID column 'id' not found in criteria data")
                
            # Clean up any duplicate columns from merge
            duplicate_cols = [col for col in comp_df.columns if col.endswith('_orig') or col.endswith('_success')]
            if duplicate_cols:
                st.write(f"DEBUG WARNING: Duplicate columns created during merge: {duplicate_cols}")
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
                    st.error(f"DEBUG ERROR: Required column '{col}' missing after duplicate cleanup")
                    # Try to recover from suffix columns if they still exist
                    orig_col = f"{col}_orig"
                    success_col = f"{col}_success"
                    
                    if orig_col in comp_df.columns:
                        comp_df[col] = comp_df[orig_col]
                        st.write(f"DEBUG: Recovered '{col}' from '{orig_col}'")
                        comp_df = comp_df.drop([orig_col], axis=1, errors='ignore')
                    elif success_col in comp_df.columns:
                        comp_df[col] = comp_df[success_col]
                        st.write(f"DEBUG: Recovered '{col}' from '{success_col}'")
                        comp_df = comp_df.drop([success_col], axis=1, errors='ignore')
            
            # Drop rows with missing success scores to make issues visible
            if 'success_score' in comp_df.columns:
                missing_scores = comp_df['success_score'].isna().sum()
                if missing_scores > 0:
                    st.write(f"DEBUG: Dropping {missing_scores} shows with missing success scores")
                    comp_df = comp_df.dropna(subset=['success_score'])
            
            # Log the columns in the merged data
            st.write(f"DEBUG: Merged criteria data columns: {list(comp_df.columns)}")
            st.write(f"DEBUG: Merged data has {len(comp_df)} rows")
            
            # Check if we have the required metrics columns
            for col in ['popcornmeter', 'tomatometer']:
                if col not in comp_df.columns:
                    st.write(f"DEBUG WARNING: '{col}' column missing from merged data")
                else:
                    st.write(f"DEBUG: '{col}' column present in merged data")
                    st.write(f"DEBUG: '{col}' non-null count: {comp_df[col].notna().sum()}")
            
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
    
    def _get_matching_shows(self, criteria: Dict[str, Any]) -> pd.DataFrame:
        """Get shows matching the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            DataFrame of matching shows with success metrics
        """
        import streamlit as st
        
        # Fetch the latest criteria data
        data = self.fetch_criteria_data()
        if data.empty:
            st.error("DEBUG ERROR: Empty criteria data from fetch_criteria_data")
            raise ValueError("No criteria data available")
        
        st.write(f"DEBUG: Available columns in data: {list(data.columns)}")
        
        # Get array fields and mapping from field_manager
        array_field_mapping = self.field_manager.get_array_field_mapping()
        array_fields = list(array_field_mapping.keys())
        
        for field_name in criteria.keys():
            if field_name in array_fields:
                # Make sure array field values are always lists
                if not isinstance(criteria[field_name], list):
                    criteria[field_name] = [criteria[field_name]]
                # Let field_manager handle the mapping
        
        # Use FieldManager to match shows against criteria
        matched_shows, match_count = self.field_manager.match_shows(criteria, data)
        
        if matched_shows.empty:
            st.error(f"ERROR: No shows matched the criteria")
            return matched_shows, 0
            
        return matched_shows, match_count   
    def _calculate_success_rate(self, shows: pd.DataFrame, threshold: float = 0.6) -> float:
        """Calculate the success rate for a set of shows.
        
        Args:
            shows: DataFrame of shows
            threshold: Success threshold (shows with score >= threshold are considered successful)
            
        Returns:
            Success rate (0-1)
        """
        import streamlit as st
        
        if shows.empty:
            st.error("DEBUG ERROR: Empty shows DataFrame provided to _calculate_success_rate")
            raise ValueError("Cannot calculate success rate with empty shows DataFrame")
        
        if 'success_score' not in shows.columns:
            st.error("DEBUG ERROR: 'success_score' column missing from shows data")
            raise ValueError("'success_score' column required for success rate calculation")
        
        # Filter out shows with missing success scores
        shows_with_scores = shows[shows['success_score'].notna()]
        
        if len(shows_with_scores) == 0:
            st.error("DEBUG ERROR: No shows with valid success scores found")
            raise ValueError("No shows with valid success scores available")
        
        # Get success score range and distribution for debugging
        min_score = shows_with_scores['success_score'].min()
        max_score = shows_with_scores['success_score'].max()
        mean_score = shows_with_scores['success_score'].mean()
        
        st.write(f"DEBUG: Success score range: {min_score} to {max_score}, mean: {mean_score}")
        
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
        
        st.write(f"DEBUG: Success rate calculation - {success_count} successful shows out of {total_count} total (threshold: {threshold})")
        
        success_rate = success_count / total_count
        st.write(f"DEBUG: Success rate: {success_rate:.4f} ({success_rate*100:.1f}%)")
        
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
        import streamlit as st
        
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
                for (option_id, option_name), rate in zip(option_data, success_rates):
                    if rate is not None:
                        # Calculate impact as relative change in success rate
                        impact = (rate - base_rate) / base_rate if base_rate != 0 else 0
                        field_impact[option_id] = impact
                
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
            matching_shows = self._get_matching_shows(criteria)
            if matching_shows.empty:
                st.error("No matching shows found for the given criteria")
                raise ValueError("No matching shows found for the given criteria")
                
            component_scores = {}
            
            # Calculate each component score with individual error handling
            for component in ['audience', 'critics', 'longevity']:
                try:
                    st.write(f"DEBUG: Calculating {component} score")
                    score_method = getattr(self, f'_calculate_{component}_score')
                    score = score_method(matching_shows)
                    st.write(f"DEBUG: {component.capitalize()} score result: {score}")
                    component_scores[component] = score
                except Exception as e:
                    st.error(f"DEBUG ERROR: Error calculating {component} score: {str(e)}")
                    import traceback
                    st.error(f"DEBUG ERROR: {component.capitalize()} score traceback: {traceback.format_exc()}")
            
            # Check if we have at least one component score
            if not component_scores:
                st.error("DEBUG ERROR: Failed to calculate any component scores")
                raise ValueError("Failed to calculate any component scores")
            
            # If we're missing any component scores but have success_score, create fallback scores
            if 'success_score' in matching_shows.columns:
                for component in ['audience', 'critics', 'longevity']:
                    if component not in component_scores:
                        st.write(f"DEBUG: Creating fallback {component} score from success_score")
                        fallback_method = getattr(self, f'_calculate_{component}_score_from_success', None)
                        if fallback_method:
                            component_scores[component] = fallback_method(matching_shows)
            
            st.write(f"DEBUG: Final component scores: {component_scores}")
            return component_scores
            
        except Exception as e:
            st.error(f"Error in calculate_component_scores: {str(e)}")
            raise
    
    def _calculate_audience_score(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate audience score for a set of shows.
        
        Args:
            shows: DataFrame of shows
            
        Returns:
            ComponentScore for audience
        """
        import streamlit as st
        
        if shows.empty:
            st.error("DEBUG ERROR: Empty shows DataFrame provided to _calculate_audience_score")
            raise ValueError("Cannot calculate audience score with empty shows DataFrame")
        
        # Check if popcornmeter column exists
        if 'popcornmeter' not in shows.columns:
            st.error("DEBUG ERROR: Popcornmeter column missing from shows data")
            st.error("DEBUG ERROR: Available columns: " + str(list(shows.columns)))
            
            # Try to use success_score as a fallback if available
            if 'success_score' in shows.columns:
                st.write("DEBUG: Using success_score as fallback for audience score calculation")
                # Filter shows with success metrics
                audience_shows = shows[shows['success_score'].notna()]
                sample_size = len(audience_shows)
                
                if sample_size == 0:
                    st.error("DEBUG ERROR: No shows with valid success_score data found")
                    raise ValueError("No shows with valid success metrics available for audience score")
                
                # Calculate confidence level
                confidence = OptimizerConfig.get_confidence_level(sample_size)
                
                # Use success_score as audience score (assuming it's already normalized to 0-1)
                avg_score = audience_shows['success_score'].mean()
                
                # Calculate audience engagement metrics
                details = {'success_score': avg_score}
                
                # Calculate overall audience score
                score = avg_score
                
                return ComponentScore(
                    component='audience',
                    score=score,
                    sample_size=sample_size,
                    confidence=confidence,
                    details=details
                )
            else:
                raise ValueError("No metrics available for audience score calculation")
            
        # Filter shows with audience metrics
        audience_shows = shows[shows['popcornmeter'].notna()]
        sample_size = len(audience_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid popcornmeter data found")
            # Try to use success_score as a fallback
            if 'success_score' in shows.columns:
                st.write("DEBUG: Using success_score as fallback for audience score calculation")
                return self._calculate_audience_score_from_success(shows)
            else:
                raise ValueError("No shows with valid popcornmeter data available")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Calculate average popcornmeter score (normalized to 0-1)
        avg_popcorn = audience_shows['popcornmeter'].mean() / 100
        
        # Calculate audience engagement metrics if available
        details = {'popcornmeter': avg_popcorn}
        
        # Calculate overall audience score
        score = avg_popcorn
        
        return ComponentScore(
            component='audience',
            score=score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
        
    def _calculate_audience_score_from_success(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate audience score using success_score as a fallback.
        
        Args:
            shows: DataFrame of shows
            
        Returns:
            ComponentScore for audience
        """
        import streamlit as st
        
        # Filter shows with success metrics
        audience_shows = shows[shows['success_score'].notna()]
        sample_size = len(audience_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid success_score data found")
            raise ValueError("No shows with valid success metrics available for audience score")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Use success_score as audience score (assuming it's already normalized to 0-1)
        avg_score = audience_shows['success_score'].mean()
        
        # Calculate audience engagement metrics
        details = {'success_score': avg_score}
        
        # Calculate overall audience score
        score = avg_score
        
        return ComponentScore(
            component='audience',
            score=score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
    
    def _calculate_critics_score(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate critics score for a set of shows.
        
        Args:
            shows: DataFrame of shows
            
        Returns:
            ComponentScore for critics
        """
        import streamlit as st
        
        if shows.empty:
            st.error("DEBUG ERROR: Empty shows DataFrame provided to _calculate_critics_score")
            raise ValueError("Cannot calculate critics score with empty shows DataFrame")
        
        # Check if tomatometer column exists
        if 'tomatometer' not in shows.columns:
            st.error("DEBUG ERROR: Tomatometer column missing from shows data")
            st.error("DEBUG ERROR: Available columns: " + str(list(shows.columns)))
            
            # Try to use success_score as a fallback if available
            if 'success_score' in shows.columns:
                st.write("DEBUG: Using success_score as fallback for critics score calculation")
                return self._calculate_critics_score_from_success(shows)
            else:
                raise ValueError("No metrics available for critics score calculation")
            
        # Filter shows with critics metrics
        critics_shows = shows[shows['tomatometer'].notna()]
        sample_size = len(critics_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid tomatometer data found")
            # Try to use success_score as a fallback
            if 'success_score' in shows.columns:
                st.write("DEBUG: Using success_score as fallback for critics score calculation")
                return self._calculate_critics_score_from_success(shows)
            else:
                raise ValueError("No shows with valid tomatometer data available")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Calculate average critics score (normalized to 0-1)
        avg_score = critics_shows['tomatometer'].mean() / 100.0
        
        # Calculate basic critics details
        details = {
            'tomatometer': avg_score,
            'sample_size': sample_size
        }
        
        # Calculate overall critics score
        score = avg_score
        
        return ComponentScore(
            component='critics',
            score=score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
        
    def _calculate_critics_score_from_success(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate critics score using success_score as a fallback.
        
        Args:
            shows: DataFrame of shows
            
        Returns:
            ComponentScore for critics
        """
        import streamlit as st
        
        # Filter shows with success metrics
        critics_shows = shows[shows['success_score'].notna()]
        sample_size = len(critics_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid success_score data found")
            raise ValueError("No shows with valid success metrics available for critics score")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Use success_score as critics score (assuming it's already normalized to 0-1)
        avg_score = critics_shows['success_score'].mean()
        
        # Calculate critics engagement metrics
        details = {'success_score': avg_score}
        
        # Calculate overall critics score
        score = avg_score
        
        return ComponentScore(
            component='critics',
            score=score,
            sample_size=len(shows),
            confidence=OptimizerConfig.get_confidence_level(len(shows)),
            details={'avg_score': avg_score}
        )
        
    def _calculate_longevity_score(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate longevity score for a set of shows.
        
        Args:
            shows: DataFrame containing show data with required columns
            
        Returns:
            ComponentScore for longevity
        """
        import streamlit as st
        
        required_columns = ['success_score']
        missing_columns = [col for col in required_columns if col not in shows.columns]
        
        if missing_columns:
            st.error(f"DEBUG ERROR: Missing columns for longevity calculation: {missing_columns}")
            st.error("DEBUG ERROR: Available columns: " + str(list(shows.columns)))
            return self._calculate_longevity_score_from_success(shows)
            
        # Filter shows with success metrics
        longevity_shows = shows[shows['success_score'].notna()]
        sample_size = len(longevity_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid success_score data found")
            return self._calculate_longevity_score_from_success(shows)
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Use success_score as longevity score (assuming it's already normalized to 0-1)
        avg_score = longevity_shows['success_score'].mean()
        
        # Calculate basic longevity details
        details = {'success_score': avg_score}
        
        # Calculate overall longevity score
        score = avg_score
        
        return ComponentScore(
            component='longevity',
            score=score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
        
    def _calculate_longevity_score_from_success(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate longevity score using success_score as a fallback.
        
        Args:
            shows: DataFrame of shows
            
        Returns:
            ComponentScore for longevity
        """
        import streamlit as st
        
        # Filter shows with success metrics
        longevity_shows = shows[shows['success_score'].notna()]
        sample_size = len(longevity_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid success_score data found")
            raise ValueError("No shows with valid success metrics available for longevity score")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Use success_score as longevity score (assuming it's already normalized to 0-1)
        avg_score = longevity_shows['success_score'].mean()
        
        # Calculate basic longevity details
        details = {'success_score': avg_score}
        
        # Calculate overall longevity score
        score = avg_score
        
        return ComponentScore(
            component='longevity',
            score=score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )

    def _batch_calculate_success_rates(self, criteria_list: List[Dict[str, Any]]) -> List[float]:
        """Batch calculate success rates for multiple criteria.
        
        Args:
            criteria_list: List of criteria dictionaries
            
        Returns:
            List of success rates in the same order as criteria_list
        """
        results = []
        for criteria in criteria_list:
            try:
                shows, count = self._get_matching_shows(criteria)
                if not shows.empty and count >= OptimizerConfig.CONFIDENCE['minimum_sample']:
                    results.append(self._calculate_success_rate(shows))
                else:
                    results.append(None)
            except Exception as e:
                import streamlit as st
                st.error(f"Error calculating success rate: {str(e)}")
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
