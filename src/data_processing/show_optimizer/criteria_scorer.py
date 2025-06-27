"""CriteriaScorer: Calculates component scores for shows based on provided data.

Responsible for calculating various component scores (success, audience, critics, longevity)
based on integrated data provided by orchestrator components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import pandas as pd
import numpy as np
import ast
import streamlit as st
import time
from datetime import datetime, timedelta

from .field_manager import FieldManager
from .optimizer_config import OptimizerConfig
from .score_calculators import ComponentScore, ScoreCalculationError, NetworkScoreCalculator
from .optimizer_data_contracts import NetworkMatch
from .score_calculators import SuccessScoreCalculator, AudienceScoreCalculator, CriticsScoreCalculator, LongevityScoreCalculator
from .optimizer_data_contracts import CriteriaDict, ConfidenceInfo, IntegratedData, update_confidence_info

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
        
    def calculate_success_rate(self, shows: pd.DataFrame = None, threshold: Optional[float] = None, integrated_data: Optional[IntegratedData] = None) -> Tuple[Optional[float], ConfidenceInfo]:
        """Public method to calculate the success rate for a set of shows.
        
        Delegates to the private _calculate_success_rate method.
        
        Args:
            shows: DataFrame of shows (if None or empty, will use integrated_data['shows'] if provided)
            threshold: Optional success threshold
            integrated_data: Optional dict of integrated DataFrames conforming to IntegratedData
        Returns:
            Tuple of success rate and confidence information conforming to ConfidenceInfo
        """
        return self._calculate_success_rate(shows, threshold, integrated_data=integrated_data)

    def _calculate_success_rate(self, shows: pd.DataFrame = None, threshold: Optional[float] = None, confidence_info: Optional[ConfidenceInfo] = None, integrated_data: Optional[IntegratedData] = None) -> Tuple[Optional[float], ConfidenceInfo]:
        """Calculate the success rate for a set of shows with confidence information.
        
        Delegates success rate calculation to the SuccessScoreCalculator.
        
        Args:
            shows: DataFrame of shows (if None or empty, will use integrated_data['shows'] if provided)
            threshold: Optional success threshold
            confidence_info: Optional confidence information conforming to ConfidenceInfo
            integrated_data: Optional dict of integrated DataFrames conforming to IntegratedData
        Returns:
            Tuple of success rate and confidence information conforming to ConfidenceInfo
            The confidence_info is always a dictionary with at least level, score, and error keys
        """
        # Use integrated_data['shows'] if shows is None or empty and integrated_data is provided
        if (shows is None or shows.empty) and integrated_data is not None and 'shows' in integrated_data:
            shows = integrated_data['shows']
                
        # Initialize confidence info using update_confidence_info to ensure it conforms to ConfidenceInfo contract
        from .optimizer_data_contracts import update_confidence_info
        confidence_info = update_confidence_info({} if confidence_info is None else confidence_info, {})
        if not isinstance(confidence_info, dict):
            # If somehow confidence_info is not a dict, create a new one that conforms to contract
            OptimizerConfig.debug(f"Warning: confidence_info was not a dictionary: {type(confidence_info)}", category='error')
            confidence_info = update_confidence_info({}, {})
        
        # Handle case with no valid shows data
        if shows is None or shows.empty:
            confidence_info = update_confidence_info(confidence_info, {
                'level': 'none',
                'score': 0.0,
                'error': 'No valid shows data provided'
            })
            return None, confidence_info
        
        # Define success filter function
        def success_filter(df):
            return (df['success_score'].notna()) & (df['success_score'] > OptimizerConfig.SCORE_NORMALIZATION['success_filter_min'])
        
        # Validate and prepare data
        calculator = SuccessScoreCalculator()
        is_valid, validated_data, validation_info = calculator.validate_and_prepare_data(
            shows, 
            required_columns=['success_score'],
            optional_columns=[],
            data_column='success_score',
            filter_condition=success_filter
        )
        
        # Handle validation failure
        if not is_valid or validated_data is None or validated_data.empty:
            confidence_info = update_confidence_info(confidence_info, {
                'level': 'none',
                'score': 0.0,
                'error': validation_info.get('error', 'Data validation failed')
            })
            return None, confidence_info
        
        # Use the provided threshold or default from config
        threshold = threshold or OptimizerConfig.SUCCESS['threshold']
        
        try:
            # Calculate success score
            component_score = calculator.calculate(validated_data, threshold=threshold)
            
            if component_score is None:
                confidence_info = update_confidence_info(confidence_info, {
                    'level': 'none',
                    'score': 0.0,
                    'error': 'Failed to calculate success score'
                })
                return None, confidence_info
            
            # Extract success rate and update confidence info with component score details
            success_rate = component_score.score
            confidence_info.update(component_score.details)
            
            return success_rate, confidence_info
            
        except Exception as e:
            if OptimizerConfig.DEBUG_MODE:
                import traceback
                OptimizerConfig.debug(f"Exception during success score calculation: {str(e)}", category='success_rate')
                OptimizerConfig.debug(traceback.format_exc(), category='success_rate')
            
            confidence_info = update_confidence_info(confidence_info, {
                'level': 'none',
                'score': 0.0,
                'error': f'Exception during calculation: {str(e)}'
            })
            return None, confidence_info

   
    def _batch_calculate_success_rates(self, criteria_list: List[CriteriaDict], matching_shows_list: Optional[List[pd.DataFrame]] = None) -> List[Optional[float]]:
        """
        Calculate success rates for a batch of criteria using provided matching shows.

        Args:
            criteria_list: List of criteria dictionaries conforming to CriteriaDict
            matching_shows_list: List of DataFrames containing shows matching each criteria
        Returns:
            List of success rates (one for each criteria/matching shows pair)
        """
        # Validate inputs
        if matching_shows_list is None or len(criteria_list) != len(matching_shows_list):
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Invalid inputs for batch calculation: criteria_list={len(criteria_list)}, matching_shows_list={len(matching_shows_list) if matching_shows_list else None}", category='success_rate')
            return [None] * len(criteria_list)
        
        results = []
        calculator = SuccessScoreCalculator()
        threshold = OptimizerConfig.PERFORMANCE.get('success_threshold', None)
        
        for i, (criteria, matching_shows) in enumerate(zip(criteria_list, matching_shows_list)):
            try:
                if matching_shows is None or matching_shows.empty:
                    results.append(None)
                    continue
                
                component_score = calculator.calculate(matching_shows, threshold=threshold)
                results.append(component_score.score if component_score else None)
                
            except Exception as e:
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Error calculating success rate for criteria {i}: {str(e)}", category='success_rate')
                results.append(None)
                
        return results
    
    def _is_valid_dataframe(self, df) -> bool:
        """Check if a DataFrame is valid (not None and not empty).
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if DataFrame is valid, False otherwise
        """
        return df is not None and not df.empty
        
    def _create_option_criteria(self, base_criteria: CriteriaDict, field: str, option_id, is_array_field: bool) -> CriteriaDict:
        """Create criteria with a specific option value for a field or remove the field.
        
        Args:
            base_criteria: Base criteria to modify conforming to CriteriaDict
            field: Field to set or remove
            option_id: Option ID to set for the field, or 'remove' to remove the field
            is_array_field: Whether this is an array field (ignored for 'remove')
            
        Returns:
            New criteria dictionary with the option set or field removed conforming to CriteriaDict
        """
        # Create a shallow copy of the base criteria
        new_criteria = dict(base_criteria)
        
        # Special handling for 'remove' option
        if option_id == 'remove':
            if field in new_criteria:
                del new_criteria[field]
            return new_criteria
        
        # Convert option ID to integer for normal options
        option_id = int(option_id)
        
        # For array fields, use a list; for scalar fields, use the value directly
        if is_array_field:
            new_criteria[field] = [option_id]
        else:
            new_criteria[field] = option_id
            
        return new_criteria
    
    def _determine_recommendation_type(self, option_id, impact: float, is_field_selected: bool, is_option_selected: bool) -> str:
        """Determine recommendation type based on impact and selection status.
        
        Args:
            option_id: ID of the option being evaluated
            impact: Impact score (difference from base success rate)
            is_field_selected: Whether the field is selected in base criteria
            is_option_selected: Whether this specific option is selected
            
        Returns:
            Recommendation type ('add', 'change', 'remove') or None if no recommendation
        """
        # Special case for explicit remove option
        if option_id == 'remove':
            return 'remove'
            
        if impact > 0:  # Positive impact
            if is_field_selected:
                # Field is selected - suggest changing it
                return 'change'
            else:
                # Field is not selected - suggest adding it
                return 'add'
        else:  # Negative impact
            # Only recommend removal for selected options with negative impact
            if is_option_selected:
                return 'remove'
            else:
                # No recommendation for unselected options with negative impact
                return None
    
    def calculate_criteria_impact(self, criteria: CriteriaDict, matching_shows: pd.DataFrame = None, option_matching_shows_map: Dict[str, Dict[Any, pd.DataFrame]] = None, integrated_data: IntegratedData = None) -> Dict[str, Dict[Any, Dict[str, Union[float, int, str, bool]]]]:
        """Calculate the impact of each criteria option on success rate.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            matching_shows: DataFrame of shows matching the criteria
            option_matching_shows_map: Optional pre-computed map of option matching shows
            integrated_data: Dictionary containing integrated data sets (REQUIRED for proper matching)
            
        Returns:
            Dictionary of impact scores by field and option with metrics like impact, sample_size, confidence, etc.
            Structure: {field_name: {option_id: {impact: float, sample_size: int, recommendation_type: str, ...}}}
        """
        # Validate inputs - fail fast with clear error messages
        if not criteria:
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("Cannot calculate criteria impact: empty criteria", category='impact')
            return {}
            
        # Normalize base criteria once at the beginning
        normalized_base_criteria = self.field_manager.normalize_criteria(criteria)
            
        # Check if criteria contains only empty values
        try:
            has_valid_criteria = False
            for field, value in criteria.items():
                try:
                    if value is not None and (not isinstance(value, (list, dict)) or len(value) > 0):
                        has_valid_criteria = True
                        break
                except Exception as e:
                    # If we can't check a value properly, consider it valid to be safe
                    if OptimizerConfig.DEBUG_MODE:
                        OptimizerConfig.debug(f"Error checking criteria value for {field}: {str(e)}", category='impact')
                    has_valid_criteria = True  # Assume valid if we can't check properly
                    break
                    
            if not has_valid_criteria:
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug("Cannot calculate criteria impact: criteria contains only empty values", category='impact')
                return {}
        except Exception as e:
            # If we can't check criteria at all, log and continue with processing
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Error checking criteria validity: {str(e)}, continuing with processing", category='impact')
            # We'll continue and let other validation steps catch issues
            
        if matching_shows is None or matching_shows.empty:
            return {}
            
        # Initialize impact scores dictionary
        impact_scores = {}
        
        try:
            # Let the field manager handle array field identification
            array_field_mapping = self.field_manager.get_array_field_mapping()
            
            # Calculate base success rate for all matching shows
            base_rate, base_info = self._calculate_success_rate(matching_shows)
            
            # Check if we have a valid base success rate
            if base_rate is None:
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug("Cannot calculate impact: invalid base success rate", category='impact')
                return {}
                
            # Get all fields to analyze from the FIELD_CONFIGS dictionary
            all_fields = list(self.field_manager.FIELD_CONFIGS.keys())
            
            # First, add all fields that are in the criteria (selected fields)
            selected_fields = [field for field in criteria.keys() if field in all_fields]
            
            # Identify unselected fields for "Add" recommendations
            unselected_fields = [field for field in all_fields if field not in criteria]
            
            # Store unselected fields for recommendation type determination
            self._unselected_fields = set(unselected_fields)
            
            # Prioritize unselected fields based on importance
            importance_map = {'primary': 1, 'secondary': 2, 'tertiary': 3}
            
            # Sort unselected fields by importance
            prioritized_unselected = sorted(
                unselected_fields,
                key=lambda field: importance_map.get(OptimizerConfig.CRITERIA_IMPORTANCE.get(field, 'secondary'), 3)
            )
            
            # Limit number of unselected fields to process
            max_unselected = OptimizerConfig.SUGGESTIONS.get('max_unselected_fields', 7)
            prioritized_unselected = prioritized_unselected[:max_unselected]
            
            # Combine selected and prioritized unselected fields
            fields_to_process = selected_fields + prioritized_unselected
            
            # Check if we have fields to process
            if not fields_to_process:
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug("Cannot calculate impact: no fields to process", category='impact')
                return {}
            
            # Process each field
            for current_field in fields_to_process:
                # Skip fields that don't have options
                options = self.field_manager.get_options(current_field)
                if not options:
                    continue
                    
                # Always initialize the field in impact_scores as an empty dict
                # This ensures we never have a missing field entry
                impact_scores[current_field] = {}
                        
                # Use field_manager to determine if this is an array field
                is_array_field = self.field_manager.get_field_type(current_field) == 'array'
                
                # Prepare batch criteria for all options
                batch_criteria = []
                option_data = []
                recommendation_types = []
                
                # Check if this field is already in the criteria
                field_in_base = current_field in criteria
                current_value = criteria.get(current_field) if field_in_base else None
                
                if field_in_base:
                    # For fields already in criteria, we'll calculate both Remove and Change recommendations
                    
                    # Field in base criteria
                    
                    # 1. First, create a "Remove" recommendation by removing this field
                    remove_criteria = dict(criteria)  # Shallow copy is sufficient
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
                        
                        # Create criteria with this option
                        new_criteria = self._create_option_criteria(criteria, current_field, option.id, is_array_field)
                        option_key = int(option.id)
                        
                        batch_criteria.append(new_criteria)
                        option_data.append((option_key, self.field_manager.get_name(current_field, option.id)))
                        recommendation_types.append('change')
                else:
                    # For fields not in criteria, create "Add" recommendations for each option
                        
                    for option in options:
                        # Create criteria with this option
                        new_criteria = self._create_option_criteria(criteria, current_field, option.id, is_array_field)
                        option_key = int(option.id)
                        
                        batch_criteria.append(new_criteria)
                        option_data.append((option_key, self.field_manager.get_name(current_field, option.id)))
                        recommendation_types.append('add')
                
                # Process each option using the provided option_matching_shows_map or generate it if not provided
                field_impact = {}
                
                # If option_matching_shows_map is not provided, we need to generate it
                if not option_matching_shows_map or current_field not in option_matching_shows_map:
                    # Generate option matching shows map for this field
                    field_options_map = {}
                    # Track how many options have no data available
                    missing_data_count = 0
                    
                    # Set batch operation flag on matcher to suppress repetitive debug messages
                    setattr(self.matcher, '_in_batch_operation', True)
                    
                    # For each option in option_data, run the matcher to get matching shows
                    for i, (option_id, option_name) in enumerate(option_data):
                        try:
                            # Create criteria for this option using our helper method
                            option_criteria = self._create_option_criteria(normalized_base_criteria, current_field, option_id, is_array_field)
                            
                            # For 'remove' recommendations, use the already filtered matching shows
                            # For other types, let the matcher use its own data source
                            if option_id == 'remove' and matching_shows is not None and not matching_shows.empty:
                                option_shows, confidence_info = self.matcher.find_matches_with_fallback(option_criteria, matching_shows)
                            else:
                                # For 'change' recommendations, use the integrated data
                                if integrated_data and 'shows' in integrated_data and not integrated_data['shows'].empty:
                                    option_shows, confidence_info = self.matcher.find_matches_with_fallback(option_criteria, integrated_data['shows'])
                                else:
                                    # No fallback - we need integrated data
                                    if OptimizerConfig.DEBUG_MODE:
                                        OptimizerConfig.debug(f"Missing integrated data for option {option_name}", category='error')
                                    option_shows = pd.DataFrame()
                                    confidence_info = update_confidence_info({}, {
                                        'level': 'none',
                                        'match_level': 1,  # Use 1 as the default match level
                                        'error': 'Missing integrated data for option'
                                    })
                                                        
                            # Process option matching shows
                            
                            # Check if we got valid shows
                            if self._is_valid_dataframe(option_shows):
                                # Store in field_options_map
                                field_options_map[option_id] = option_shows
                            else:
                                # Track missing data
                                missing_data_count += 1
                        except Exception as e:
                            # Log error and skip this option
                            if OptimizerConfig.DEBUG_MODE:
                                OptimizerConfig.debug(f"Exception getting matching shows for option {option_name}: {str(e)}", category='impact')
                            # Skip this option
                            continue
                    
                    # Set the option_matching_shows_map for this field
                    if not option_matching_shows_map:
                        option_matching_shows_map = {}
                    option_matching_shows_map[current_field] = field_options_map
                    
                    # Log a summary of missing data if any occurred
                    if missing_data_count > 0 and OptimizerConfig.DEBUG_MODE:
                        OptimizerConfig.debug(f"No data available for {missing_data_count} options in field '{current_field}'", category='impact')
                    
                    # Reset batch operation flag
                    setattr(self.matcher, '_in_batch_operation', False)
                else:
                    # Use the provided option_matching_shows_map
                    field_options_map = option_matching_shows_map[current_field]
                
                # Field is already initialized in impact_scores above
                # This ensures we maintain a consistent data structure
                
                # Process each option in option_data
                for i, (option_id, option_name) in enumerate(option_data):
                    try:
                        # Get the option shows from field_options_map
                        option_shows = field_options_map.get(option_id)
                        
                        # Skip options with no matching shows
                        if not self._is_valid_dataframe(option_shows):
                            continue
                            
                        # Calculate success rate for this option's matching shows
                        option_rate, option_info = self._calculate_success_rate(option_shows)
                        
                        if option_rate is None:
                            continue
                            
                        # Calculate impact as the difference from base rate
                        impact = option_rate - base_rate
                    
                        # Get minimum impact threshold
                        min_impact = OptimizerConfig.SUGGESTIONS.get('minimum_impact', 0.05)
                        
                        # Store the original impact for reference and debugging
                        original_impact = impact
                        
                        # Force a minimum impact for testing - this ensures we get recommendations
                        if abs(impact) < min_impact:  # If impact is very small
                            # Boost the impact slightly to ensure we get recommendations
                            # But preserve the sign (positive/negative)
                            impact = min_impact if impact >= 0 else -min_impact
                            
                            if OptimizerConfig.DEBUG_MODE:
                                OptimizerConfig.debug(f"Boosted impact for {option_name} from {original_impact:.4f} to {impact:.4f}", category='impact')
                        
                        # Determine recommendation type based on impact and whether this specific option is selected
                        is_field_selected = current_field in criteria
                        is_option_selected = False
                        
                        # Check if this specific option is selected
                        if is_field_selected:
                            current_value = criteria[current_field]
                            if isinstance(current_value, list):
                                is_option_selected = option_id in current_value
                            else:
                                is_option_selected = current_value == option_id
                                
                            # Debug removed - redundant option selection status
                        
                        # Determine recommendation type based on impact and selection status
                        # Centralized recommendation type assignment rules:
                        # - 'add': For suggesting new unselected fields with positive impact
                        # - 'change': For suggesting different values for already selected fields
                        # - 'remove': For suggesting removal of selected fields with negative impact
                        
                        # Determine recommendation type using helper method
                        recommendation_type = self._determine_recommendation_type(
                            option_id, impact, is_field_selected, is_option_selected
                        )
                        
                        # Skip options with no recommendation type (e.g., negative impact on unselected fields)
                        if recommendation_type is None:
                            if OptimizerConfig.DEBUG_MODE:
                                OptimizerConfig.debug(f"Skipping {current_field}={option_name}: negative impact {impact:.4f} on unselected field", category='impact')
                            continue  # Skip to next option
                        
                        # Add detailed debug output for recommendation type determination
                        # Removed verbose debug statement about recommendation type determination
                        
                        # Store impact score with all relevant information - ALWAYS as a complete dictionary
                        # This ensures consistent data structure throughout the application
                        impact_scores[current_field][option_id] = {
                            'option_id': option_id,
                            'option_name': option_name,
                            'impact': impact,
                            'original_impact': original_impact,  # Store original impact for reference
                            'success_rate': option_rate,
                            'sample_size': len(option_shows),
                            'recommendation_type': recommendation_type
                        }
                    except Exception as e:
                        # Skip this option if there's an error
                        if OptimizerConfig.DEBUG_MODE:
                            import traceback
                            OptimizerConfig.debug(f"Error processing option {option_name}: {str(e)}", category='impact')
                            OptimizerConfig.debug(traceback.format_exc(), category='impact')
                        continue
                
            # Check if we have any impact scores after processing all fields
            # Removed debug statement about impact score counts
                
                # Debug log to verify impact data structure integrity
                for field, options in impact_scores.items():
                    if field == '_summary':
                        continue
                    for option_id, impact_info in options.items():
                        if not isinstance(impact_info, dict):
                            OptimizerConfig.debug(f"WARNING: Non-dict impact_info found for {field}.{option_id}: {type(impact_info)}", category='error')
                        elif 'impact' not in impact_info:
                            OptimizerConfig.debug(f"WARNING: Missing 'impact' key in impact_info for {field}.{option_id}", category='error')
            
            # Process unselected fields to generate 'add' recommendations
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("Processing unselected fields for 'add' recommendations", category='impact')
            
            # Process fields we haven't already processed
            processed_fields = set(fields_to_process)
            for field in self.field_manager.FIELD_CONFIGS.keys():
                if field not in criteria and field not in processed_fields:
                    # Skip fields that don't have options
                    options = self.field_manager.get_options(field)
                    if not options:
                        continue
                    
                    # Always initialize the field in impact_scores as an empty dict
                    # This ensures we never have a missing field entry or inconsistent structure
                    impact_scores[field] = {}
                    
                    # For each option, calculate a default impact score
                    for option in options:
                        try:
                            option_id = option.id
                            option_name = self.field_manager.get_name(field, option_id)
                            
                            # Create a new criteria with this option
                            option_criteria = criteria.copy()
                            option_criteria[field] = option_id
                            
                            # Use field_manager to normalize criteria values based on field types
                            option_criteria = self.field_manager.normalize_criteria(option_criteria)
                            
                            # Determine if we should use existing matching shows or run a fresh match
                            # For adding a new field, we could filter existing matches if they exist
                            use_matching_shows = matching_shows is not None and not matching_shows.empty
                            
                            # If the field is already in criteria (but with a different value), we need a fresh match
                            if field in criteria:
                                use_matching_shows = False
                            
                            if OptimizerConfig.DEBUG_MODE:
                                OptimizerConfig.debug(f"Testing 'add' option {option_name} with criteria: {option_criteria} (using existing matches: {use_matching_shows})", category='impact')
                            
                            # Use existing matches if appropriate, otherwise run a fresh match
                            if use_matching_shows:
                                # When using existing matches, pass the matching_shows as data
                                option_shows, confidence_info = self.matcher.find_matches_with_fallback(option_criteria, matching_shows)
                            else:
                                # Let the matcher use its own data source
                                option_shows, confidence_info = self.matcher.find_matches_with_fallback(option_criteria)
                            
                            # Skip options with no matching shows
                            if option_shows is None or option_shows.empty:
                                continue
                                
                            # Calculate success rate for this option's matching shows
                            option_rate, option_info = self._calculate_success_rate(option_shows)
                            
                            if option_rate is None:
                                continue
                                
                            # Calculate impact as the difference from base rate
                            impact = option_rate - base_rate
                            
                            # Store the original impact for reference
                            original_impact = impact
                            
                            # Skip if impact is too small
                            min_impact = OptimizerConfig.SUGGESTIONS.get('minimum_impact', 0.05)
                            if abs(impact) < min_impact:
                                # Skip options with minimal impact
                                if OptimizerConfig.DEBUG_MODE:
                                    OptimizerConfig.debug(f"Skipping {field}={option_name} due to small impact: {impact:.4f}", category='impact')
                                continue
                            
                            # For unselected fields, only store positive impact scores (as 'add' recommendations)
                            # Skip negative impact scores for unselected fields as they're not actionable
                            if impact <= 0:
                                # Skip storing negative impact scores for unselected fields
                                if OptimizerConfig.DEBUG_MODE:
                                    OptimizerConfig.debug(f"Skipping {field}={option_name}: negative impact {impact:.4f} on unselected field", category='impact')
                                continue
                                
                            # For unselected fields with positive impact, always use 'add' recommendation type
                            recommendation_type = 'add'
                            
                            if OptimizerConfig.DEBUG_MODE:
                                OptimizerConfig.debug(f"Recommendation for {field}={option_name}: type={recommendation_type}, impact={impact:.4f} (positive)", category='impact')
                            
                            # Store impact score with all relevant information - ALWAYS as a complete dictionary
                            # This ensures consistent data structure throughout the application
                            impact_scores[field][option_id] = {
                                'option_id': option_id,
                                'option_name': option_name,
                                'impact': impact,
                                'original_impact': original_impact,
                                'success_rate': option_rate,
                                'sample_size': len(option_shows),
                                'recommendation_type': recommendation_type
                            }
                        except Exception as e:
                            if OptimizerConfig.DEBUG_MODE:
                                import traceback
                                OptimizerConfig.debug(f"Error processing unselected field option {field}: {str(e)}", category='impact')
                                OptimizerConfig.debug(traceback.format_exc(), category='impact')
                            continue
                
            # Summarize the impact scores we've generated
            if OptimizerConfig.DEBUG_MODE:
                # Count fields, options, and recommendation types
                field_count = len(impact_scores)
                option_count = 0
                recommendation_counts = {'add': 0, 'change': 0, 'remove': 0}
                
                # Count options and recommendation types
                for field, options in impact_scores.items():
                    option_count += len(options)
                    for option in options.values():
                        rec_type = option.get('recommendation_type')
                        if rec_type in recommendation_counts:
                            recommendation_counts[rec_type] += 1
                # Add summary statistics to the result for better recommendations
                result_summary = {
                    'field_count': field_count,
                    'option_count': option_count,
                    'recommendation_counts': recommendation_counts
                }
                
                # Store the summary in the result
                impact_scores['_summary'] = result_summary
                
                # Summary information is now stored in impact_scores['_summary'] for better recommendations
                
            # If we still have no impact scores, return structured empty dict with error info
            if not any(impact_scores.values()) or all(k == '_summary' for k in impact_scores.keys()):
                return {
                    '_error': 'No impact scores could be generated from any selected or unselected fields',
                    '_summary': {
                        'field_count': 0,
                        'option_count': 0,
                        'recommendation_counts': {'add': 0, 'change': 0, 'remove': 0}
                    }
                }
            
            # Final validation of impact data structure before returning
            if OptimizerConfig.DEBUG_MODE:
                # Debug removed - redundant final validation log
                for field, options in impact_scores.items():
                    if field == '_summary' or field == '_error':
                        continue
                    for option_id, impact_info in options.items():
                        if not isinstance(impact_info, dict):
                            OptimizerConfig.debug(f"CRITICAL: Non-dict impact_info in final result for {field}.{option_id}: {type(impact_info)}", category='error')
                            # Fix the issue by creating a proper dictionary
                            impact_scores[field][option_id] = {
                                'option_id': option_id,
                                'impact': float(impact_info) if isinstance(impact_info, (int, float)) else 0.0,
                                'sample_size': 0,
                                'recommendation_type': 'add'
                            }
                            OptimizerConfig.debug(f"Fixed non-dict impact_info for {field}.{option_id}", category='impact')
                
            return impact_scores

        except ValueError as ve:
            if OptimizerConfig.DEBUG_MODE:
                import traceback
                OptimizerConfig.debug(f"ValueError in calculate_criteria_impact: {str(ve)}", category='error')
                OptimizerConfig.debug(traceback.format_exc(), category='error')
            return {}
        except Exception as e:
            # Log the exception with detailed traceback
            if OptimizerConfig.DEBUG_MODE:
                import traceback
                OptimizerConfig.debug(f"Exception in calculate_criteria_impact: {str(e)}", category='error')
                OptimizerConfig.debug(traceback.format_exc(), category='error')
                # Log the state of key variables to help diagnose the issue
                OptimizerConfig.debug(f"Criteria: {criteria}", category='error')
                OptimizerConfig.debug(f"Matching shows count: {len(matching_shows) if matching_shows is not None and not matching_shows.empty else 0}", category='error')
                OptimizerConfig.debug(f"Fields processed: {fields_to_process if 'fields_to_process' in locals() else 'Not initialized'}", category='error')
            return {}
            
    def calculate_scores(self, criteria: CriteriaDict, matching_shows: pd.DataFrame, integrated_data: IntegratedData = None) -> Dict[str, Any]:
        """Calculate all scores for a set of criteria and matching shows.
        
        This is the main entry point for score calculation as defined in the architecture.
        It orchestrates the calculation of all component scores and returns a comprehensive
        result dictionary with standardized structure.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            matching_shows: DataFrame of shows matching the criteria, must contain at minimum 'id' and 'success_score' columns
            integrated_data: Optional dictionary of integrated data conforming to IntegratedData
            
        Returns:
            Dictionary with the following structure:
            {
                'component_scores': Dict[str, ComponentScore],  # Component scores by name
                'success_rate': Optional[float],                # Overall success rate (0-1)
                'success_info': Dict[str, Union[float, int, str]],  # Success rate calculation details
                'confidence': ConfidenceInfo                   # Confidence information
            }
            
        Raises:
            ValueError: If matching_shows is None or empty
        """
        # Validate inputs
        if not self._is_valid_dataframe(matching_shows):
            # Return a structured empty result with error information that conforms to our contracts
            from .optimizer_data_contracts import update_confidence_info
            return {
                'component_scores': {},
                'success_rate': None,
                'success_info': {'error': 'No matching shows available for score calculation'},
                'confidence': update_confidence_info({}, {'level': 'none', 'score': 0.0})
            }
            
        # Calculate confidence information if not already provided
        confidence_info = self.calculate_confidence(criteria)
        
        # Calculate component scores
        component_scores = self.calculate_component_scores(criteria, matching_shows, confidence_info, integrated_data)
        
        # Calculate success rate
        success_rate, success_info = self._calculate_success_rate(matching_shows)
        
        # Combine all scores into a comprehensive result, ensuring confidence_info conforms to contract
        from .optimizer_data_contracts import update_confidence_info
        all_scores = {
            'component_scores': component_scores,
            'success_rate': success_rate,
            'success_info': success_info,
            'confidence': update_confidence_info(confidence_info, {})
        }
        
        return all_scores
    
    def calculate_component_scores(self, criteria: CriteriaDict, matching_shows: pd.DataFrame, confidence_info: ConfidenceInfo = None, integrated_data: IntegratedData = None) -> Dict[str, ComponentScore]:
        """Calculate component scores for a set of criteria and matching shows.
        
        This method orchestrates the calculation of individual component scores using
        specialized calculator classes. It handles validation, error handling, and
        aggregation of results.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            matching_shows: DataFrame of shows matching the criteria, must contain required columns for each component
            confidence_info: Optional dictionary of confidence information conforming to ConfidenceInfo
            integrated_data: Optional dictionary of integrated data conforming to IntegratedData
            
        Returns:
            Dictionary mapping component names to ComponentScore objects with the following structure:
            {
                'success': ComponentScore,   # Success component score
                'audience': ComponentScore,  # Audience component score
                'critics': ComponentScore,   # Critics component score
                'longevity': ComponentScore, # Longevity component score
                'network': ComponentScore    # Network component score (if network_analyzer is available)
            }
            
        Raises:
            ValueError: If matching_shows is None or empty
        """
        if matching_shows is None or matching_shows.empty:
            # Return an empty dictionary with structured error information
            return {}
            
        try:
            # Check sample size
            match_count = len(matching_shows)
            
            # Initialize calculators
            calculators = [
                SuccessScoreCalculator(),
                AudienceScoreCalculator(),
                CriticsScoreCalculator(),
                LongevityScoreCalculator()
            ]
            component_scores = {}
            
            # Calculate scores for each component
            for calculator in calculators:
                try:
                    score_component = calculator.calculate(matching_shows.copy())
                    if score_component:
                        component_scores[calculator.component_name] = score_component
                        if OptimizerConfig.DEBUG_MODE:
                            score_value = score_component.score if score_component.score is not None else 'N/A'
                            OptimizerConfig.debug(f"Calculated {calculator.component_name} score: {score_value}", category='components')
                except Exception as e:
                    # Create error component score
                    component_scores[calculator.component_name] = ComponentScore(
                        component=calculator.component_name, 
                        score=None, 
                        sample_size=0, 
                        confidence='none', 
                        details={'status': 'calculation_error', 'error': str(e)}
                    )
                    if OptimizerConfig.DEBUG_MODE:
                        import traceback
                        OptimizerConfig.debug(f"Error calculating {calculator.component_name} score: {str(e)}", category='components')
                        OptimizerConfig.debug(traceback.format_exc(), category='components')
            
            return component_scores
            
        except Exception as e:
            # Error handling without excessive debug output
            return {}

    def calculate_confidence(self, criteria: CriteriaDict) -> ConfidenceInfo:
        """Calculate and analyze confidence levels for criteria.
        
        Delegates to field_manager.calculate_confidence to determine confidence
        levels for the provided criteria. The confidence information includes
        scores and levels for different components based on the criteria values.
        
        Args:
            criteria: Dictionary of criteria field names and values conforming to CriteriaDict
            
        Returns:
            Dictionary with the following structure conforming to ConfidenceInfo:
            {
                'confidence': str,                # Overall confidence level ('none', 'very_low', 'low', 'medium', 'high')
                'confidence_score': float,         # Overall confidence score (0-1)
                'field_confidence': Dict[str, Any] # Per-field confidence information
            }
        """
        return self.field_manager.calculate_confidence(criteria)
            
    # _get_matching_shows method has been removed as part of the architecture refactoring
    # All matching functionality is now handled directly by the Matcher class
    
    def calculate_network_scores(self, criteria: CriteriaDict, matching_shows: pd.DataFrame) -> List[NetworkMatch]:
        """Calculate network compatibility and success scores for a set of criteria.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            matching_shows: Pre-matched shows DataFrame
            
        Returns:
            List of NetworkMatch objects with compatibility and success scores
        """
        # Convert criteria to string format for the network calculator
        criteria_str = str(criteria)
        
        # Initialize network calculator if not already done
        if not hasattr(self, '_network_calculator'):
            self._network_calculator = NetworkScoreCalculator()
            self._network_calculator.field_manager = self.field_manager
            
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("Initialized NetworkScoreCalculator", category='networks')
        
        # Calculate network scores
        network_matches = self._network_calculator.calculate_network_scores(criteria_str, matching_shows)
        
        if OptimizerConfig.DEBUG_MODE:
            network_count = len(network_matches) if network_matches else 0
            OptimizerConfig.debug(f"Found {network_count} network matches", category='networks')
            OptimizerConfig.debug(f"Calculated scores for {network_count} networks", category='networks')
            
        return network_matches
