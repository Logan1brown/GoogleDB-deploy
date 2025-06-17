"""CriteriaScorer: Calculates component scores for shows based on provided data.

Responsible for calculating various component scores (success, audience, critics, longevity)
based on integrated data provided by orchestrator components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import pandas as pd
import numpy as np
import ast
import streamlit as st
import time
from datetime import datetime, timedelta

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
    """Calculates component scores for show criteria based on provided data.
    
    This class is responsible for calculating various component scores including success,
    audience, critics, and longevity scores. It delegates the actual score calculations to
    specialized calculator classes while providing orchestration and result aggregation.
    """
    
    def __init__(self, field_manager: FieldManager, matcher=None):
        """Initialize the criteria scorer.
        
        Args:
            field_manager: FieldManager instance for field mapping and validation
            matcher: Optional Matcher instance for finding matches
        """
        self.field_manager = field_manager
        self.matcher = matcher  # Matcher instance for finding matches
        self.network_analyzer = None  # Will be set by ShowOptimizer after initialization
        
    def calculate_success_rate(self, shows: pd.DataFrame = None, threshold: Optional[float] = None, integrated_data: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[Optional[float], Dict[str, Any]]:
        """Public method to calculate the success rate for a set of shows.
        
        Delegates to the private _calculate_success_rate method.
        
        Args:
            shows: DataFrame of shows (if None or empty, will use integrated_data['shows'] if provided)
            threshold: Optional success threshold
            integrated_data: Optional dict of integrated DataFrames (e.g., {'shows': DataFrame})
        Returns:
            Tuple of success rate and confidence information
        """
        return self._calculate_success_rate(shows, threshold, integrated_data=integrated_data)

    def _calculate_success_rate(self, shows: pd.DataFrame = None, threshold: Optional[float] = None, confidence_info: Optional[Dict[str, Any]] = None, integrated_data: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[Optional[float], Dict[str, Any]]:
        """Calculate the success rate for a set of shows with confidence information.
        
        Delegates success rate calculation to the SuccessScoreCalculator.
        
        Args:
            shows: DataFrame of shows (if None or empty, will use integrated_data['shows'] if provided)
            threshold: Optional success threshold
            confidence_info: Optional confidence information
            integrated_data: Optional dict of integrated DataFrames (e.g., {'shows': DataFrame})
        Returns:
            Tuple of success rate and confidence information
        """
        # Use integrated_data['shows'] if shows is None or empty and integrated_data is provided
        if (shows is None or shows.empty) and integrated_data is not None and 'shows' in integrated_data and not integrated_data['shows'].empty:
            shows = integrated_data['shows']
                
        if shows is None or shows.empty:
            if OptimizerConfig.DEBUG_MODE:
                st.warning("No shows available for success rate calculation")
            if confidence_info is None:
                confidence_info = {'level': 'none', 'score': 0.0}
            return None, confidence_info
        
        # Delegate to SuccessScoreCalculator
        calculator = SuccessScoreCalculator()
        
        # Use the calculator's validate_and_prepare_data method with the same filter condition as in calculate
        def success_filter(df):
            return (df['success_score'].notna()) & (df['success_score'] > OptimizerConfig.SCORE_NORMALIZATION['success_filter_min'])
        
        is_valid, validated_data, validation_info = calculator.validate_and_prepare_data(
            shows, 
            required_columns=['success_score'],
            optional_columns=[],
            data_column='success_score',
            filter_condition=success_filter
        )
        
        if not is_valid or validated_data is None or validated_data.empty:
            if OptimizerConfig.DEBUG_MODE:
                st.warning(f"No valid success score data found: {validation_info.get('error', 'Unknown error')}")
            if confidence_info is None:
                confidence_info = {'level': 'none', 'score': 0.0, 'error': validation_info.get('error', 'Unknown error')}
            else:
                confidence_info['error'] = validation_info.get('error', 'Unknown error')
            return None, confidence_info
        
        # Use the provided threshold or default from config
        if threshold is None:
            threshold = OptimizerConfig.SUCCESS['threshold']
        
        # Let the calculator handle the actual calculation
        try:
            component_score = calculator.calculate(validated_data, threshold=threshold)
            
            if component_score is None:
                if OptimizerConfig.DEBUG_MODE:
                    st.warning("Failed to calculate success score")
                if confidence_info is None:
                    confidence_info = {'level': 'none', 'score': 0.0, 'error': 'Failed to calculate success score'}
                else:
                    confidence_info['error'] = 'Failed to calculate success score'
                return None, confidence_info
            
        except Exception as e:
            st.error(f"Error in success score calculation: {str(e)}")
            if OptimizerConfig.DEBUG_MODE:
                import traceback
                st.write(f"Error details: {traceback.format_exc()}")
            # Create a basic confidence_info dictionary with error details
            if confidence_info is None:
                confidence_info = {'level': 'none', 'score': 0.0, 'error': f'Exception during calculation: {str(e)}'}
            else:
                confidence_info['error'] = f'Exception during calculation: {str(e)}'
            return None, confidence_info
        
        # Extract success rate and metadata from component score
        success_rate = component_score.score
        
        # Update confidence information
        if confidence_info is None:
            confidence_info = {}
        
        # Merge metadata from component score into confidence info
        for key, value in component_score.details.items():
            confidence_info[key] = value
        
        return success_rate, confidence_info

   
    def _batch_calculate_success_rates(self, criteria_list: List[Dict[str, Any]], matching_shows_list: Optional[List[pd.DataFrame]] = None) -> List[Optional[float]]:
        """
        Calculate success rates for a batch of criteria using provided matching shows.

        Args:
            criteria_list: List of criteria dictionaries
            matching_shows_list: Optional list of DataFrames containing shows matching each criteria. If not provided, raises an error (explicit matching is required).
        Returns:
            List of success rates (one for each criteria/matching shows pair)
        """
        if matching_shows_list is None:
            st.error("No matching_shows_list provided to _batch_calculate_success_rates. This method requires explicit matching.")
            return [None] * len(criteria_list)
        if len(criteria_list) != len(matching_shows_list):
            st.error(f"Mismatch between criteria list ({len(criteria_list)}) and matching shows list ({len(matching_shows_list)})")
            return [None] * len(criteria_list)
        
        results = []
        
        # Create a single SuccessScoreCalculator instance to reuse
        calculator = SuccessScoreCalculator()
        
        for i, (criteria, matching_shows) in enumerate(zip(criteria_list, matching_shows_list)):
            try:
                if matching_shows is None or matching_shows.empty:
                    results.append(None)
                    continue
                    
                # Use the calculator to calculate the success rate with the proper threshold
                threshold = OptimizerConfig.PERFORMANCE.get('success_threshold', None)
                
                try:
                    # Ensure we pass the correct parameters to calculate
                    component_score = calculator.calculate(matching_shows, threshold=threshold)
                    
                    if component_score is None:
                        results.append(None)
                    else:
                        # Extract just the success rate value
                        results.append(component_score.score)
                except Exception as calc_error:
                    # Handle specific calculation errors
                    st.warning(f"Error in success rate calculation: {str(calc_error)}")
                    results.append(None)
                    
            except Exception as e:
                st.error(f"Error calculating success rate for criteria {criteria}: {str(e)}")
                results.append(None)
                
        return results
    
    def calculate_criteria_impact(self, base_criteria: Dict[str, Any], base_matching_shows: pd.DataFrame, option_matching_shows_map: Optional[Dict[str, Dict[int, pd.DataFrame]]] = None, field_name: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Calculate the impact of criteria values on success rate.

        Args:
            base_criteria: The base criteria to compare against.
            base_matching_shows: DataFrame of shows matching the base criteria.
            option_matching_shows_map: Optional mapping of field names to option IDs to matching shows. If provided, these pre-matched shows will be used for impact calculation (recommended for batch/impact analysis).
            field_name: Optional name of the field to analyze. If None, analyzes all fields.
            
        Returns:
            A dictionary mapping field names to dictionaries of option IDs to impact scores.
        """
        try:
            # Debug output for input parameters
            if OptimizerConfig.DEBUG_MODE and OptimizerConfig.VERBOSE_DEBUG:
                st.write(f"Debug: calculate_criteria_impact called with {len(base_criteria)} criteria")
                st.write(f"Debug: Base criteria keys: {list(base_criteria.keys())}")
                st.write(f"Debug: Base matching shows: {len(base_matching_shows)} rows")
                if field_name:
                    st.write(f"Debug: Analyzing specific field: {field_name}")
            
            # Let the field manager handle array field identification
            array_field_mapping = self.field_manager.get_array_field_mapping()
            array_fields = list(array_field_mapping.keys())
            
            if OptimizerConfig.DEBUG_MODE and OptimizerConfig.VERBOSE_DEBUG:
                if OptimizerConfig.VERBOSE_DEBUG:
                    st.write(f"Debug: Array fields: {array_fields}")
                    st.write(f"Debug: Field manager has {len(self.field_manager.FIELD_CONFIGS)} field configs")
            
            # Use the provided base matching shows
            base_match_count = len(base_matching_shows) if not base_matching_shows.empty else 0
            
            if base_matching_shows.empty:
                if OptimizerConfig.DEBUG_MODE:
                    # This is a critical error worth showing even without verbose debug
                    st.write(f"Debug: Cannot calculate impact scores - no matching shows")
                raise ValueError("Cannot calculate impact scores with no matching shows")
                
            if base_match_count < OptimizerConfig.CONFIDENCE['minimum_sample']:
                if OptimizerConfig.DEBUG_MODE:
                    # This is a critical error worth showing even without verbose debug
                    st.write(f"Debug: Cannot calculate impact scores - insufficient sample size ({base_match_count} shows)")
                raise ValueError(f"Cannot calculate impact scores with insufficient sample size ({base_match_count} shows). Minimum required: {OptimizerConfig.CONFIDENCE['minimum_sample']}")
            
            # Calculate base success rate using the provided shows
            if OptimizerConfig.DEBUG_MODE and OptimizerConfig.VERBOSE_DEBUG:
                if OptimizerConfig.VERBOSE_DEBUG:
                    st.write(f"Debug: Calculating base success rate for {len(base_matching_shows)} shows")
                    if 'success_score' in base_matching_shows.columns:
                        st.write(f"Debug: Success score stats: min={base_matching_shows['success_score'].min()}, max={base_matching_shows['success_score'].max()}, mean={base_matching_shows['success_score'].mean()}")
                else:
                    st.write("Debug: No success_score column in base_matching_shows")
            
            base_rate, base_info = self._calculate_success_rate(base_matching_shows)
            
            if OptimizerConfig.DEBUG_MODE and OptimizerConfig.VERBOSE_DEBUG:
                if OptimizerConfig.VERBOSE_DEBUG:
                    st.write(f"Debug: Base success rate calculation result: {base_rate}")
                    st.write(f"Debug: Base success info: {base_info}")
            
            if base_rate is None:
                if OptimizerConfig.DEBUG_MODE:
                    st.write("Debug: Unable to calculate base success rate - success_score data missing")
                st.warning("WARNING: Unable to calculate base success rate - success_score data missing")
                return {}
            
            impact_scores = {}
            
            # Determine which fields to process
            fields_to_process = [field_name] if field_name else self.field_manager.FIELD_CONFIGS.keys()
            
            if OptimizerConfig.DEBUG_MODE and OptimizerConfig.VERBOSE_DEBUG:
                if OptimizerConfig.VERBOSE_DEBUG:
                    st.write(f"Debug: Base success rate: {base_rate:.2f}")
                    st.write(f"Debug: Fields to process: {list(fields_to_process)}")
            
            def make_hashable(val):
                """Convert any value to a hashable type for dictionary keys.
                
                Uses field_manager's knowledge of array fields to properly handle list values.
                """
                if isinstance(val, dict):
                    # Convert dict to a sorted tuple of (key, value) pairs
                    return tuple(sorted((k, make_hashable(v)) for k, v in val.items()))
                    
                if isinstance(val, list):
                    # For empty lists, return an empty tuple
                    if not val:
                        return tuple()
                    # For lists of primitive values, sort and convert to tuple
                    if all(isinstance(v, (int, float, str, bool)) for v in val):
                        return tuple(sorted(val))
                    # For lists of complex values, convert each item and then sort
                    return tuple(sorted(make_hashable(v) for v in val))
                    
                if isinstance(val, tuple):
                    return tuple(make_hashable(v) for v in val)
                    
                # Ensure we return a hashable type
                if isinstance(val, (str, int, float, bool, type(None))):
                    return val
                    
                # Last resort - convert to string
                return str(val)
            
            for current_field in fields_to_process:
                # Process both fields in base criteria (for Remove/Change) and not in base criteria (for Add)
                
                # Use field_manager to determine if this is an array field
                is_array_field = self.field_manager.get_field_type(current_field) == 'array'
                options = self.field_manager.get_options(current_field)
                
                if OptimizerConfig.DEBUG_MODE and OptimizerConfig.VERBOSE_DEBUG:
                        st.write(f"Debug: Processing field {current_field} (array: {is_array_field})")
                        st.write(f"Debug: Found {len(options)} options for field {current_field}")
                
                # Prepare batch criteria for all options
                batch_criteria = []
                option_data = []
                recommendation_types = []
                
                # Check if this field is already in the base criteria
                field_in_base = current_field in base_criteria
                current_value = base_criteria.get(current_field) if field_in_base else None
                
                if field_in_base:
                    # For fields already in criteria, we'll calculate both Remove and Change recommendations
                    
                    if OptimizerConfig.DEBUG_MODE and OptimizerConfig.VERBOSE_DEBUG:
                            st.write(f"Debug: Field {current_field} is in base criteria with value {current_value}")
                    
                    # 1. First, create a "Remove" recommendation by removing this field
                    remove_criteria = base_criteria.copy()
                    del remove_criteria[current_field]
                    batch_criteria.append(remove_criteria)
                    option_data.append(('remove', 'Remove ' + current_field))
                    recommendation_types.append('remove')
                    
                    # 2. Then, create "Change" recommendations for each alternative option
                    for option in options:
                        # Skip the current value
                        if is_array_field and isinstance(current_value, list) and option.id in current_value:
                            continue
                        elif not is_array_field and option.id == current_value:
                            continue
                            
                        # Create a new criteria with this option
                        new_criteria = base_criteria.copy()
                        
                        # For array fields, we replace the current value with a new array
                        # For scalar fields, we simply replace the value
                        if is_array_field and isinstance(current_value, list):
                            # Replace the array with a new array containing just this option
                            option_key = int(option.id)
                            new_criteria[current_field] = [option_key]
                        else:
                            # Replace the scalar value
                            option_key = int(option.id)
                            new_criteria[current_field] = option_key
                            
                        batch_criteria.append(new_criteria)
                        option_data.append((option_key, self.field_manager.get_name(current_field, option.id)))
                        recommendation_types.append('change')
                else:
                    # For fields not in criteria, create "Add" recommendations for each option
                        
                    for option in options:
                        # Create a new criteria with this option
                        new_criteria = base_criteria.copy()
                        
                        # For array fields, always use list of ints
                        # For scalar fields, use the int directly
                        if is_array_field:
                            option_key = int(option.id)
                            new_criteria[current_field] = [option_key]
                        else:
                            option_key = int(option.id)
                            new_criteria[current_field] = option_key
                            
                        batch_criteria.append(new_criteria)
                        option_data.append((option_key, self.field_manager.get_name(current_field, option.id)))
                        recommendation_types.append('add')
                
                # Process each option using the provided option_matching_shows_map or generate it if not provided
                field_impact = {}
                
                # If option_matching_shows_map is not provided, we need to generate it
                if not option_matching_shows_map or current_field not in option_matching_shows_map:
                    
                    # Generate option matching shows map for this field
                    field_options_map = {}
                    
                    # For each option in option_data, run the matcher to get matching shows
                    for i, (option_id, option_name) in enumerate(option_data):
                        try:
                            # Create criteria for this option
                            option_criteria = base_criteria.copy()
                            
                            # Set the criteria value based on field type
                            is_array_field = self.field_manager.get_field_type(current_field) == 'array'
                            if is_array_field:
                                option_criteria[current_field] = [option_id]
                            else:
                                option_criteria[current_field] = option_id
                                
                            # Get matching shows for this option
                            # Pass None as the data parameter to use the original integrated data in the matcher
                            # This ensures we're not filtering an already filtered dataset
                            option_shows, _, _ = self._get_matching_shows(option_criteria, None)
                            
                            # Store in field_options_map
                            field_options_map[option_id] = option_shows
                            
                            # Only show debug output for zero matches as this is critical information
                            if OptimizerConfig.DEBUG_MODE and len(option_shows) == 0 and current_field in base_criteria:
                                st.write(f"Debug: No matches for {current_field}={option_name} (conflicts with base criteria)")
                            elif OptimizerConfig.DEBUG_MODE and len(option_shows) == 0 and OptimizerConfig.VERBOSE_DEBUG:
                                    st.write(f"Debug: No matches for {current_field}={option_name}")
                                
                        except Exception as e:
                            if OptimizerConfig.DEBUG_MODE:
                                # Keep error messages visible as they're important
                                st.write(f"Debug: Error generating matching shows for {current_field}={option_name}: {str(e)}")
                            # Skip this option
                            continue
                else:
                    # Use the provided option_matching_shows_map
                    field_options_map = option_matching_shows_map[current_field]
                    
                    # Convert all keys in field_options_map to hashable for safe lookup
                    hashable_field_options_map = {}
                    for k, v in field_options_map.items():
                        hashable_key = make_hashable(k)
                        hashable_field_options_map[hashable_key] = v
                    
                    # Only show debug output if there are no options at all (critical issue)
                    if OptimizerConfig.DEBUG_MODE and len(field_options_map) == 0:
                        # This is a critical error worth showing even without verbose debug
                        st.write(f"Debug: Field {current_field} has no options available")
                        
                    # Verify that we have matching shows for at least some options
                    has_valid_options = False
                    for k, v in hashable_field_options_map.items():
                        if v is not None and not (isinstance(v, pd.DataFrame) and v.empty):
                            has_valid_options = True
                            break
                            
                    if not has_valid_options and OptimizerConfig.DEBUG_MODE:
                        # This is a critical error worth showing even without verbose debug
                        st.write(f"Debug: No valid matching shows found for any option in field {current_field}")
                        continue
                    
                    for i, (option_id, option_name) in enumerate(option_data):
                        # Make the option_id hashable for lookup
                        hashable_option_id = make_hashable(option_id)
                        
                        # Skip if we don't have matching shows for this option
                        if hashable_option_id not in hashable_field_options_map:
                            # No debug output needed for individual missing options
                            continue
                            
                        # Get the matching shows for this option
                        option_shows = hashable_field_options_map[hashable_option_id]
                        
                        if option_shows is None or option_shows.empty:
                            continue
                            
                        # Calculate success rate for this option
                        option_rate, option_confidence = self._calculate_success_rate(option_shows)
                        
                        if option_rate is not None:
                            match_count = len(option_shows)
                            impact = (option_rate - base_rate) / base_rate if base_rate != 0 else 0
                            
                            # Get the recommendation type for this option
                            rec_type = recommendation_types[i] if i < len(recommendation_types) else 'add'
                            
                            # Use field_manager to get the proper option name
                            if option_id != 'remove':
                                try:
                                    # For numeric option IDs, get the name from field_manager
                                    if isinstance(option_id, (int, float)):
                                        display_name = self.field_manager.get_name(current_field, int(option_id))
                                    else:
                                        display_name = option_name
                                except Exception as e:
                                    if OptimizerConfig.DEBUG_MODE:
                                        # Keep error messages visible as they're important
                                        st.write(f"Debug: Error getting option name for {current_field}={option_id}: {str(e)}")
                                    display_name = option_name
                            else:
                                display_name = option_name
                            
                            if OptimizerConfig.DEBUG_MODE and OptimizerConfig.VERBOSE_DEBUG:
                                    st.write(f"Debug: Impact for {current_field}={display_name}: {impact:.2f} (sample size: {match_count})")
                            
                            field_impact[option_id] = {
                                "impact": impact, 
                                "sample_size": match_count, 
                                "option_name": display_name,
                                "success_rate": option_rate,
                                "recommendation_type": rec_type
                            }
                    
                # For accurate impact calculation, we need to find matching shows for each criteria option
                # combined with the base criteria (for adding criteria) or with criteria removed (for removing criteria)
                matching_shows_list = []
                
                if self.matcher:
                    # For each criteria option, we want to see how it affects the base criteria
                    for crit in batch_criteria:
                        try:
                            # This is the key difference - we're using the combined criteria
                            # (base + new option) to find matching shows, not just the option alone
                            option_shows, _, _ = self._get_matching_shows(crit)
                            matching_shows_list.append(option_shows)
                        except Exception as e:
                            # If matching fails for this option, use an empty DataFrame
                            matching_shows_list.append(pd.DataFrame())
                else:
                    # If no matcher is available, we can't calculate impact scores
                    for _ in batch_criteria:
                        matching_shows_list.append(pd.DataFrame())
                
                rates = self._batch_calculate_success_rates(batch_criteria, matching_shows_list)
                for i, (option_id, option_name) in enumerate(option_data):
                    option_rate = rates[i]
                    if option_rate is not None:
                        # Use the actual matching shows count for this option
                        match_count = len(matching_shows_list[i]) if i < len(matching_shows_list) and not matching_shows_list[i].empty else 0
                        # Calculate impact based on difference from base rate
                        impact = (option_rate - base_rate) / base_rate if base_rate != 0 else 0
                        # Get the recommendation type for this option
                        rec_type = recommendation_types[i] if i < len(recommendation_types) else 'add'
                        
                        field_impact[option_id] = {
                            "impact": impact,
                            "sample_size": match_count,
                            "option_name": option_name,
                            "success_rate": option_rate,
                            "recommendation_type": rec_type
                        }
                if field_impact:
                    impact_scores[current_field] = field_impact
                    
            if not impact_scores and fields_to_process:
                # Check if option_matching_shows_map was provided
                if option_matching_shows_map is None:
                    # Instead of showing an error, just return empty results and continue
                    if OptimizerConfig.DEBUG_MODE:
                        st.write("No option matching shows map available for impact score calculation - skipping")
                    return {}
                else:
                    # Only log a warning if we attempted to process fields but got no results
                    # If fields_to_process was empty (e.g. field_name was already in base_criteria), this is not an error
                    if OptimizerConfig.DEBUG_MODE:
                        st.write("Could not calculate any impact scores with the given criteria configuration - continuing")
                    return {}
                
            return impact_scores

        except ValueError as ve:
            st.error(str(ve)) # Surface our specific ValueErrors
            raise
        except Exception as e:
            st.error(f"Error calculating criteria impact: {str(e)}")
            raise
    
    def calculate_component_scores(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame, confidence_info: Dict[str, Any], integrated_data: Dict[str, pd.DataFrame] = None) -> Dict[str, ComponentScore]:
        """
        Calculate component scores for the given criteria using provided matched shows.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: DataFrame of shows matching the criteria
            confidence_info: Confidence information from matching
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            
        Returns:
            Dictionary mapping component names to ComponentScore objects
        """
        component_scores: Dict[str, ComponentScore] = {}

        try:
            # First check if we have any matching shows
            if matching_shows is None or matching_shows.empty:
                st.warning(f"No matching shows found for criteria: {criteria}. Cannot calculate component scores.")
                return {}
            
            # Calculate match count
            match_count = len(matching_shows)
            
            # Using general minimum_sample from OptimizerConfig
            if match_count < OptimizerConfig.CONFIDENCE['minimum_sample']:
                st.warning(f"Insufficient sample size ({match_count}) for criteria: {criteria} to calculate reliable component scores. Minimum required: {OptimizerConfig.CONFIDENCE['minimum_sample']}")
                # We'll continue with the calculation, but with a warning
            
            # Get match level from confidence info
            actual_match_level = confidence_info.get('match_level')
            if actual_match_level is None:
                st.warning("Incomplete confidence info provided to calculate_component_scores: missing match_level")
                # Don't add a fallback, let the error propagate naturally
                
            original_match_level = confidence_info.get('original_match_level', actual_match_level)
            
            if actual_match_level != original_match_level and st.session_state.get('debug_mode', False):
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"Note: Match level adjusted from {original_match_level} to {actual_match_level} for component score calculation")
                
            # If we have array fields like character_types in our criteria, make sure they're properly matched
            # This helps ensure component scores are calculated based on shows that actually match the criteria
                array_fields = [field for field, value in criteria.items() if isinstance(value, list) and value]
                if array_fields and actual_match_level > 1 and st.session_state.get('debug_mode', False):
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"Note: Array criteria matching relaxed to level {actual_match_level} for fields: {array_fields}")
            
            # Ensure all required data is available in the matching_shows DataFrame
            # This prevents each calculator from having to find matches again
            required_columns = {
                'success': ['success_score'],
                'audience': ['tmdb_vote_average', 'tmdb_vote_count'],
                'critics': ['tomatometer'],
                'longevity': ['tmdb_seasons', 'tmdb_total_episodes', 'tmdb_status']
            }
            
            # Log once at the beginning rather than for each calculator
            if st.session_state.get('debug_mode', False):
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"Using {match_count} matched shows for all component score calculations")

        except Exception as e:
            st.error(f"Optimizer Error: Failed to prepare for component score calculation. Criteria: {criteria}. Details: {e}")
            return {}

        # Ensure field_manager is initialized (should be in __init__)
        if not hasattr(self, 'field_manager') or self.field_manager is None:
            st.error("Optimizer Critical Error: FieldManager is not initialized in CriteriaScorer. This may affect score calculations.")

        # Initialize and use calculators for each component
        calculators = [
            SuccessScoreCalculator(),
            AudienceScoreCalculator(),
            CriticsScoreCalculator(),
            LongevityScoreCalculator()
        ]
        
        # Calculate scores for each component using the SAME matching_shows DataFrame
        # This prevents redundant matching operations
        for calculator in calculators:
            try:
                # Pass the same matching_shows to all calculators
                score_component = calculator.calculate(matching_shows.copy())  # Pass a copy to avoid modification issues
                if score_component:  # Ensure a component score object was returned
                    component_scores[calculator.component_name] = score_component
            except Exception as e:
                st.warning(f"Failed to calculate {calculator.component_name} score: {str(e)} - will display N/A")
                # Create a placeholder component score with None value
                component_scores[calculator.component_name] = ComponentScore(
                    component=calculator.component_name,
                    score=None,  # None will be displayed as N/A
                    sample_size=0,
                    confidence='none',
                    details={'status': 'calculation_error', 'error': str(e)}
                )

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
        
    def _get_matching_shows(self, criteria: Dict[str, Any], data: pd.DataFrame = None, flexible: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        """Get shows matching the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            data: Optional DataFrame to use instead of integrated data
            flexible: Whether to use flexible matching (fallback to fewer criteria)
            
        Returns:
            Tuple of (matching_shows, confidence_info, match_info)
        """
        if self.matcher is None:
            return pd.DataFrame(), {'level': 'none', 'score': 0.0}, {'match_level': 0}
            
        # Use find_matches_with_fallback to get matches
        matching_shows, confidence_info = self.matcher.find_matches_with_fallback(
            criteria, data, flexible=flexible)
            
        # Extract match level from confidence info
        match_level = confidence_info.get('match_level', 0) if confidence_info else 0
        match_info = {'match_level': match_level}
        
        return matching_shows, confidence_info, match_info
    
    def calculate_network_scores(self, criteria: Dict[str, Any], matching_shows: pd.DataFrame) -> List[NetworkMatch]:
        """Calculate network compatibility and success scores for a set of criteria.
        
        Args:
            criteria: Dictionary of criteria
            matching_shows: Pre-matched shows DataFrame
            
        Returns:
            List of NetworkMatch objects with compatibility and success scores
        """
        criteria_str = str(criteria) if isinstance(criteria, dict) else criteria
        
        if not hasattr(self, '_network_calculator'):
            self._network_calculator = NetworkScoreCalculator()
            
        # Pass the field_manager to the network calculator if needed
        if hasattr(self, 'field_manager') and not hasattr(self._network_calculator, 'field_manager'):
            self._network_calculator.field_manager = self.field_manager
            
        return self._network_calculator.calculate_network_scores(criteria_str, matching_shows)
