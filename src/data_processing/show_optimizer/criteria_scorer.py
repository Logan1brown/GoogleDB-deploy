"""CriteriaScorer: Calculates success rates for show criteria based on historical data.

Integrates with SuccessAnalyzer for metrics and provides data to CriteriaAnalyzer.
Handles success scoring, network analysis, and component weighting with caching.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import ast
import streamlit as st
import time
from functools import lru_cache
from datetime import datetime, timedelta

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .field_manager import FieldManager
from .optimizer_config import OptimizerConfig
from .score_calculators import ComponentScore, ScoreCalculationError, NetworkMatch, NetworkScoreCalculator, MatchingCalculator
from .score_calculators import SuccessScoreCalculator, AudienceScoreCalculator, CriticsScoreCalculator, LongevityScoreCalculator

SCORE_CALCULATORS_CLASSES = {
    'success': SuccessScoreCalculator,
    'audience': AudienceScoreCalculator,
    'critics': CriticsScoreCalculator,
    'longevity': LongevityScoreCalculator,
}

__all__ = ['CriteriaScorer', 'ComponentScore', 'ScoreCalculationError']



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
        self.criteria_data = None
        self.last_update = None
        self.cache_duration = OptimizerConfig.PERFORMANCE['cache_duration']
        self._normalization_performed = False  # Track if normalization has been performed
        
        # Initialize field manager
        self.field_manager = FieldManager({})
        
        # Initialize the MatchingCalculator
        from .score_calculators import MatchingCalculator
        self._matching_calculator = MatchingCalculator(self)
        
        # Get reference data from ShowsAnalyzer using fetch_comp_data
        try:
            comp_df, reference_data = shows_analyzer.fetch_comp_data()
            self.field_manager = FieldManager(reference_data)
            # Initialize the matching calculator
            self._matching_calculator = MatchingCalculator(self)
        except Exception as e:
            st.error(f"Optimizer Initialization Error: Could not initialize FieldManager due to: {e}. Some features may not work correctly or data may be incomplete.")
            # Initialize with empty reference data as fallback
            self.field_manager = FieldManager({})
            
        self.criteria_data = None
        self.last_update = None
        self.cache_duration = OptimizerConfig.PERFORMANCE['cache_duration']
        
    def fetch_criteria_data(self, force_refresh=False):
        """Fetch criteria data for matching and scoring.
        
        Args:
            force_refresh: Force refresh of data, ignoring cache
            
        Returns:
            DataFrame of criteria data with success metrics
        """
        current_time = datetime.now()
        
        # Use cached data if available and not forcing refresh
        if not force_refresh and self.criteria_data is not None and self.last_update is not None:
            if (current_time - self.last_update) < timedelta(seconds=self.cache_duration):
                return self.criteria_data
        
        try:
            # Get show criteria data from ShowsAnalyzer
            comp_df, reference_data = self.shows_analyzer.fetch_comp_data()
            
            if comp_df.empty:
                st.error("CRITICAL: No base data returned from ShowsAnalyzer for CriteriaScorer. Optimizer cannot function.")
                raise ValueError("No data returned from ShowsAnalyzer.fetch_comp_data()")
            
            # Get success metrics from SuccessAnalyzer
            success_df = self.success_analyzer.fetch_success_data()
            
            # Reset index if show_id is the index
            if not success_df.empty and success_df.index.name == 'show_id':
                success_df = success_df.reset_index()
            
            # Define success metrics to merge from success_df
            success_metrics_to_integrate = ['success_score', 'rt_score', 'imdb_score', 'metacritic_score']
            
            # Determine which metrics are available in success_df
            available_metrics_in_success_df = []
            if not success_df.empty:
                available_metrics_in_success_df = [col for col in success_metrics_to_integrate if col in success_df.columns]
            
            # Merge success metrics if available
            if available_metrics_in_success_df and not success_df.empty:
                try:
                    # Merge success metrics into comp_df
                    comp_df = pd.merge(
                        comp_df,
                        success_df[['show_id'] + available_metrics_in_success_df],
                        left_on='id',
                        right_on='show_id',
                        how='left'
                    )
                    
                    # Drop the redundant show_id column from the merge
                    if 'show_id' in comp_df.columns:
                        comp_df = comp_df.drop(columns=['show_id'])
                    
                except Exception as e:
                    st.error(f"Optimizer Data Error: Failed to merge success metrics. Details: {e}")
            
            elif success_df.empty:
                st.warning("Optimizer Data Note: Success metrics data is currently empty. Scores and analyses will not include these metrics.")
            else:  # success_df not empty, but available_metrics_in_success_df is empty
                st.warning(f"Optimizer Data Note: Key success metrics ({success_metrics_to_integrate}) were not found in the available success data. Scores and analyses will not include these specific metrics.")
            
            # Normalize success_score to 0-1 range if it exists and is on 0-100 scale
            # Only perform normalization if it hasn't been done already
            if 'success_score' in comp_df.columns and not self._normalization_performed:
                # Check if any non-NaN scores are > 1, indicating a 0-100 scale
                if comp_df['success_score'].notna().any() and (comp_df.loc[comp_df['success_score'].notna(), 'success_score'] > 1).any():
                    st.info("Data Normalization: 'success_score' was detected on a 0-100 scale and has been normalized to 0-1.")
                    comp_df['success_score'] = comp_df['success_score'] / 100.0
                    self._normalization_performed = True  # Mark normalization as performed
            
            # Update field manager with new reference data
            self.field_manager = FieldManager(reference_data)
            
            # Validate required columns
            if 'id' not in comp_df.columns:
                st.error("CRITICAL: Show ID column 'id' not found in criteria data after processing. Optimizer cannot function.")
                raise ValueError("Show ID column 'id' not found in criteria data")
            
            # Update cache
            self.criteria_data = comp_df
            self.last_update = current_time
            
            st.toast(f"Optimizer data refreshed. Shape: {self.criteria_data.shape}")
            return self.criteria_data
            
        except ValueError as ve:
            st.error(f"Data Error in Optimizer: {str(ve)}")  # Cloud visible error
            raise
        except Exception as e:
            st.error("An unexpected error occurred in the Optimizer while fetching data.")  # Cloud visible error
            raise
        
    def _get_matching_shows(self, criteria: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Get shows matching the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Tuple of (DataFrame of matching shows with success metrics, count of matches)
        """
        # Delegate to the MatchingCalculator
        return self._matching_calculator.get_matching_shows(criteria)
    def _calculate_success_rate(self, shows: pd.DataFrame, threshold: Optional[float] = None) -> Optional[float]:
        """Calculate the success rate for a set of shows.
        
        Args:
            shows: DataFrame of shows
            threshold: Success threshold (shows with score >= threshold are considered successful)
            
        Returns:
            Success rate (0-1) or None if success_score is missing
        """
        # Delegate to the MatchingCalculator
        return self._matching_calculator.calculate_success_rate(shows, threshold)

   
    def _batch_calculate_success_rates(self, criteria_list: List[Dict[str, Any]]) -> List[Optional[float]]:
        """Batch calculate success rates for multiple criteria.
        
        Args:
            criteria_list: List of criteria dictionaries
            
        Returns:
            List of success rates in the same order as criteria_list, with None for missing data
        """
        # Delegate to the MatchingCalculator
        return self._matching_calculator.batch_calculate_success_rates(criteria_list)
    
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
                raise ValueError("Cannot calculate impact scores with no matching shows")
                
            if base_match_count < OptimizerConfig.CONFIDENCE['minimum_sample']:
                raise ValueError(f"Cannot calculate impact scores with insufficient sample size ({base_match_count} shows). Minimum required: {OptimizerConfig.CONFIDENCE['minimum_sample']}")
            
            base_rate = self._calculate_success_rate(base_shows)
            
            if base_rate is None:
                st.warning("WARNING: Unable to calculate base success rate - success_score data missing")
                return {}
            
            if base_rate == 0:
                raise ValueError("Base success rate is zero, cannot calculate meaningful impact scores.")
            
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
                        # Ensure option.id is treated as a list item for array fields
                        new_criteria[current_field] = [option.id] if not isinstance(option.id, list) else option.id
                    else:
                        new_criteria[current_field] = option.id
                    batch_criteria.append(new_criteria)
                    option_data.append((option.id, option.name))
                
                # Batch calculate success rates
                success_rates = self._batch_calculate_success_rates(batch_criteria)
                
                # Process results
                field_impact = {}
                
                for i, criteria_set in enumerate(batch_criteria):
                    option_id, option_name = option_data[i]
                    rate = success_rates[i]
                    
                    if rate is not None:
                        option_shows, match_count = self._get_matching_shows(criteria_set)
                        impact = (rate - base_rate) / base_rate if base_rate != 0 else 0
                        
                        # option_id is guaranteed to be an int from FieldManager's FieldOption.id, which is hashable.
                        # No conversion or special handling is needed for its use as a dictionary key.
                        field_impact[option_id] = {"impact": impact, "sample_size": match_count, "option_name": option_name}
                
                if field_impact:
                    impact_scores[current_field] = field_impact
                    
            if not impact_scores and fields_to_process:
                # Only raise if we attempted to process fields but got no results.
                # If fields_to_process was empty (e.g. field_name was already in base_criteria), this is not an error.
                raise ValueError("Could not calculate any impact scores with the given criteria configuration.")
                
            return impact_scores

        except ValueError as ve:
            st.error(str(ve)) # Surface our specific ValueErrors
            raise
        except Exception as e:
            st.error(f"Error calculating criteria impact: {str(e)}")
            raise
    
    def calculate_component_scores(self, criteria: Dict[str, Any]) -> Dict[str, ComponentScore]:
        """
        Calculate component scores for a set of criteria.

        Args:
            criteria: Dictionary of criteria to calculate scores for

        Returns:
            Dictionary mapping component names to ComponentScore objects
        """
        component_scores: Dict[str, ComponentScore] = {}

        try:
            matching_shows, match_count = self._get_matching_shows(criteria)

            if matching_shows.empty:
                st.warning(f"No matching shows found for criteria: {criteria}. Cannot calculate component scores.")
                return {}

            # Using general minimum_sample from OptimizerConfig
            if match_count < OptimizerConfig.CONFIDENCE['minimum_sample']:
                st.warning(
                    f"Insufficient sample size ({match_count}) for criteria: {criteria} "
                    f"to calculate reliable component scores. "
                    f"Minimum required: {OptimizerConfig.CONFIDENCE['minimum_sample']}"
                )
                return {}

        except Exception as e:
            st.error(f"Optimizer Error: Failed to get matching shows for component score calculation. Criteria: {criteria}. Details: {e}")
            return {}

        # Ensure field_manager is initialized (should be in __init__)
        if not hasattr(self, 'field_manager') or self.field_manager is None:
            st.error("Optimizer Critical Error: FieldManager is not initialized in CriteriaScorer. This may affect score calculations like longevity.")
            # Depending on how critical, could return {} here or let individual calculators fail.

        calculators = [
            SuccessScoreCalculator(component_name="success"),
            AudienceScoreCalculator(component_name="audience"),
            CriticsScoreCalculator(component_name="critics"),
            LongevityScoreCalculator(component_name="longevity", field_manager=self.field_manager)
        ]

        for calculator in calculators:
            try:
                score_component = calculator.calculate(matching_shows.copy())  # Pass a copy to avoid modification issues
                if score_component:  # Ensure a component score object was returned
                    component_scores[calculator.component_name] = score_component
                else:
                    st.warning(
                        f"Score calculation for component '{calculator.component_name}' did not return a score object. "
                        "This component will be excluded."
                    )
            except ScoreCalculationError as e:
                st.warning(
                    f"Could not calculate score for component '{calculator.component_name}': {e}. "
                    "This component will be excluded from the results."
                )
            except Exception as e:
                st.error(
                    f"An unexpected error occurred while calculating score for component '{calculator.component_name}': {e}. "
                    "This component will be excluded from the results."
                )

        if not component_scores:
            st.warning("No component scores could be calculated for the given criteria after attempting all components.")
            return {}

        return component_scores




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
        # Use the NetworkScoreCalculator to calculate network scores
        if not hasattr(self, '_network_calculator'):
            self._network_calculator = NetworkScoreCalculator(self)
        return self._network_calculator.calculate_network_scores(criteria_str)
