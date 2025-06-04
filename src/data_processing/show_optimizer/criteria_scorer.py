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
from .score_calculators import ComponentScore, ScoreCalculationError, NetworkMatch, NetworkScoreCalculator
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
        
        # Get reference data from ShowsAnalyzer using fetch_comp_data
        try:
            comp_df, reference_data = shows_analyzer.fetch_comp_data()
            self.field_manager = FieldManager(reference_data)
        except Exception as e:
            st.error(f"Optimizer Initialization Error: Could not initialize FieldManager due to: {e}. Some features may not work correctly or data may be incomplete.")
            # Initialize with empty reference data as fallback
            self.field_manager = FieldManager({})
        
        # Initialize the matching calculator after field_manager is set up
        # Import here to avoid circular imports
        from .score_calculators import MatchingCalculator
        # Create an instance and store it
        self._matching_calculator = MatchingCalculator(self)
        # Ensure it has access to the field_manager
        # (Field manager access check removed)
        
    def fetch_criteria_data(self, force_refresh=False):
        """Fetch criteria data for matching and scoring.
        
        Args:
            force_refresh: Force refresh of data, ignoring cache
            
        Returns:
            DataFrame of criteria data with success metrics
        """
        # Fetch criteria data with optional force refresh
        current_time = datetime.now()
        
        # Use cached data if available and not forcing refresh
        if not force_refresh and self.criteria_data is not None and self.last_update is not None:
            if (current_time - self.last_update) < timedelta(seconds=self.cache_duration):
                # Using cached criteria data
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
            # Exclude tmdb_seasons and tmdb_status which already exist in comp_df
            # to avoid column conflicts during merge
            success_metrics_to_include = ['success_score', 'popcornmeter', 'tomatometer', 'tmdb_total_episodes']
            
            # Merge success metrics if available
            if not success_df.empty:
                try:
                    # Only include non-conflicting columns to avoid suffix issues
                    # Always include show_id for the join
                    columns_to_merge = ['show_id'] + success_metrics_to_include
                    available_columns = [col for col in columns_to_merge if col in success_df.columns]
                    
                    # Handle columns that might cause suffix issues
                    # Instead of dropping columns, we'll rename them after the merge
                    
                    # Merge success metrics into comp_df
                    comp_df = pd.merge(
                        comp_df,
                        success_df[available_columns],
                        left_on='id',
                        right_on='show_id',
                        how='left'
                    )
                    
                    # With our selective column approach, we shouldn't have any suffix issues
                    
                    # Check for required component calculator columns
                    for col in ['popcornmeter', 'tomatometer']:
                        if col not in comp_df.columns:
                            pass  # Column is missing from merged data
                    
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
            
            # Debug output for success metrics
            # Final criteria data ready for use
            
            st.toast(f"Optimizer data refreshed. Shape: {self.criteria_data.shape}")
            return self.criteria_data
            
        except ValueError as ve:
            st.error(f"Data Error in Optimizer: {str(ve)}")  # Cloud visible error
            raise
        except Exception as e:
            st.error("An unexpected error occurred in the Optimizer while fetching data.")  # Cloud visible error
            raise
        
    def _get_matching_shows(self, criteria: Dict[str, Any], flexible: bool = True) -> Tuple[pd.DataFrame, int, Dict[str, Any]]:
        """Get shows matching the given criteria with flexible matching support.
        
        Args:
            criteria: Dictionary of criteria
            flexible: Whether to use flexible matching (try different match levels)
            
        Returns:
            Tuple of (DataFrame of matching shows, count of matches, confidence info)
        """
        # Delegate to the MatchingCalculator
        return self._matching_calculator.get_matching_shows(criteria, flexible=flexible)
    def _calculate_success_rate(self, shows: pd.DataFrame, threshold: Optional[float] = None, confidence_info: Optional[Dict[str, Any]] = None) -> Tuple[Optional[float], Dict[str, Any]]:
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
        # Delegate to the MatchingCalculator
        return self._matching_calculator.calculate_success_rate(shows, threshold, confidence_info)

   
    def _batch_calculate_success_rates(self, criteria_list: List[Dict[str, Any]], flexible: bool = True) -> List[Tuple[Optional[float], Dict[str, Any]]]:
        """Batch calculate success rates for multiple criteria with confidence information.
        
        Args:
            criteria_list: List of criteria dictionaries
            flexible: Whether to use flexible matching (try different match levels)
            
        Returns:
            List of tuples (success_rate, confidence_info) in the same order as criteria_list
        """
        # Delegate to the MatchingCalculator
        return self._matching_calculator.batch_calculate_success_rates(criteria_list, flexible=flexible)
    
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
            # Get matching shows and ensure they have all required columns for scoring
            matching_shows, match_count = self._get_matching_shows(criteria)
            
            # Log the number of matching shows found
            st.write(f"Found {match_count} matching shows")

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
                
            # Ensure all required columns are present in the data
            required_columns = {
                'success': ['success_score'],
                'audience': ['popcornmeter'],
                'critics': ['tomatometer'],
                'longevity': ['tmdb_seasons', 'tmdb_total_episodes', 'tmdb_status']
            }
            
            # Check if we have all required columns and log any missing ones
            for component, columns in required_columns.items():
                missing = [col for col in columns if col not in matching_shows.columns]
                if missing:
                    st.warning(f"Missing columns for {component} score calculation: {missing}")

        except Exception as e:
            st.error(f"Optimizer Error: Failed to get matching shows for component score calculation. Criteria: {criteria}. Details: {e}")
            return {}

        # Ensure field_manager is initialized (should be in __init__)
        if not hasattr(self, 'field_manager') or self.field_manager is None:
            st.error("Optimizer Critical Error: FieldManager is not initialized in CriteriaScorer. This may affect score calculations.")

        # Initialize the calculators
        calculators = [
            SuccessScoreCalculator(),
            AudienceScoreCalculator(),
            CriticsScoreCalculator(),
            LongevityScoreCalculator()
        ]
        
        # Calculate scores for each component
        for calculator in calculators:
            try:
                # Check if required columns for this calculator are present
                if calculator.component_name == 'success' and 'success_score' not in matching_shows.columns:
                    st.error(f"Missing success_score column for {calculator.component_name} calculator")
                elif calculator.component_name == 'audience' and 'popcornmeter' not in matching_shows.columns:
                    st.error(f"Missing popcornmeter column for {calculator.component_name} calculator")
                elif calculator.component_name == 'critics' and 'tomatometer' not in matching_shows.columns:
                    st.error(f"Missing tomatometer column for {calculator.component_name} calculator")
                elif calculator.component_name == 'longevity':
                    missing = [col for col in ['tmdb_seasons', 'tmdb_total_episodes', 'tmdb_status'] if col not in matching_shows.columns]
                    if missing:
                        st.error(f"Missing {missing} columns for {calculator.component_name} calculator")
                
                # Calculate the component score
                score_component = calculator.calculate(matching_shows.copy())  # Pass a copy to avoid modification issues
                if score_component:  # Ensure a component score object was returned
                    component_scores[calculator.component_name] = score_component
            except Exception as e:
                st.error(f"Failed to calculate {calculator.component_name} score: {str(e)}")
                # Print stack trace for debugging
                import traceback
                st.error(f"Stack trace: {traceback.format_exc()}")

        if not component_scores:
            st.warning("No component scores could be calculated for the given criteria after attempting all components.")
            return {}

        # Component scores calculation complete
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
