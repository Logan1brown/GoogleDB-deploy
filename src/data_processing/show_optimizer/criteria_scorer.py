"""CriteriaScorer: Calculates component scores for shows based on provided data.

Responsible for calculating various component scores (success, audience, critics, longevity)
based on integrated data provided by orchestrator components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union, Set, Callable, TypedDict
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict

# Local imports
from .optimizer_config import OptimizerConfig
from .field_manager import FieldManager, CriteriaDict
from .optimizer_matcher import Matcher
from .optimizer_data_contracts import (
    IntegratedData, NetworkMatch, CriteriaDict, 
    ConfidenceInfo, update_confidence_info
)
from .score_calculators import (
    SuccessScoreCalculator, AudienceScoreCalculator,
    CriticsScoreCalculator, LongevityScoreCalculator,
    ComponentScore, ScoreCalculationError, NetworkScoreCalculator
)

# Type aliases
ImpactScores = Dict[str, Dict[Any, Dict[str, Any]]]
OptionMatchingShowsMap = Dict[str, Dict[Any, pd.DataFrame]]

class ImpactAnalysisSummary(TypedDict):
    field_count: int
    option_count: int
    recommendation_counts: Dict[str, int]

@dataclass
class ImpactAnalysisResult:
    """Structured result of criteria impact analysis.
    
    Attributes:
        criteria_impacts: Dictionary mapping criteria types to their impact data
        summary: Summary statistics about the analysis
        error: Optional error message if the analysis failed
    """
    criteria_impacts: Dict[str, Dict[Any, Dict[str, Any]]]
    summary: ImpactAnalysisSummary
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary for backward compatibility."""
        result = dict(self.criteria_impacts)
        result['_summary'] = dict(self.summary)
        if self.error:
            result['_error'] = self.error
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImpactAnalysisResult':
        """Create an ImpactAnalysisResult from a legacy dictionary format."""
        criteria_impacts = {k: v for k, v in data.items() 
                          if k not in ('_summary', '_error')}
        summary = data.get('_summary', {
            'field_count': 0,
            'option_count': 0,
            'recommendation_counts': {'add': 0, 'change': 0, 'remove': 0}
        })
        error = data.get('_error')
        return cls(criteria_impacts, summary, error)

# Score calculator classes mapping

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
    
    # Recommendation type constants for standardization across components
    REC_TYPE_ADD = 'add'
    REC_TYPE_CHANGE = 'change'
    REC_TYPE_REMOVE = 'remove'
    
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
        # Fast path: Check for empty shows first to avoid unnecessary processing
        if shows is None or not isinstance(shows, pd.DataFrame) or len(shows) == 0:
            # Use integrated_data['shows'] if shows is empty and integrated_data is provided
            if integrated_data is not None and 'shows' in integrated_data and len(integrated_data['shows']) > 0:
                shows = integrated_data['shows']
            else:
                # Initialize confidence info using update_confidence_info to ensure it conforms to ConfidenceInfo contract
                from .optimizer_data_contracts import update_confidence_info
                return None, update_confidence_info({} if confidence_info is None else confidence_info, {
                    'level': 'none',
                    'score': 0.0,
                    'error': 'No valid shows data provided'
                })
        
        # Initialize confidence info using update_confidence_info to ensure it conforms to ConfidenceInfo contract
        from .optimizer_data_contracts import update_confidence_info
        
        # Process confidence info once
        confidence_info = update_confidence_info({} if confidence_info is None else confidence_info, {})
        if not isinstance(confidence_info, dict):
            # If somehow confidence_info is not a dict, create a new one that conforms to contract
            confidence_info = update_confidence_info({}, {})
        
        # Handle case with no valid shows data - use len() instead of .empty for better performance
        if len(shows) == 0:
            confidence_info = update_confidence_info(confidence_info, {
                'level': 'none',
                'score': 0.0,
                'error': 'No valid shows data provided'
            })
            return None, confidence_info
        
        # Pre-calculate the minimum threshold value to avoid dictionary lookup in the filter
        min_threshold = OptimizerConfig.SCORE_NORMALIZATION['success_filter_min']
        
        # Fast path: Check if success_score column exists before validation
        if 'success_score' not in shows.columns:
            confidence_info = update_confidence_info(confidence_info, {
                'level': 'none',
                'score': 0.0,
                'error': 'Missing required column: success_score'
            })
            return None, confidence_info
            
        # Fast path: Pre-filter valid rows before full validation
        valid_rows = (shows['success_score'].notna()) & (shows['success_score'] > min_threshold)
        if not valid_rows.any():
            confidence_info = update_confidence_info(confidence_info, {
                'level': 'none',
                'score': 0.0,
                'error': 'No valid success scores found'
            })
            return None, confidence_info
        
        # Validate and prepare data - with pre-filtered data
        calculator = SuccessScoreCalculator()
        is_valid, validated_data, validation_info = calculator.validate_and_prepare_data(
            shows[valid_rows], 
            required_columns=['success_score'],
            optional_columns=[],
            data_column='success_score',
            filter_condition=None  # Already filtered
        )
        
        # Handle validation failure - use len() instead of .empty for better performance
        if not is_valid or validated_data is None or (isinstance(validated_data, pd.DataFrame) and len(validated_data) == 0):
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
            return None, update_confidence_info(confidence_info, {
                'level': 'none',
                'score': 0.0,
                'error': f'Exception during calculation: {str(e)}'
            })

   
    def _batch_calculate_success_rates(self, criteria_list: List[CriteriaDict], matching_shows_list: Optional[List[pd.DataFrame]] = None) -> List[Optional[float]]:
        """
        Calculate success rates for a batch of criteria using provided matching shows.

        Args:
            criteria_list: List of criteria dictionaries conforming to CriteriaDict
            matching_shows_list: List of DataFrames containing shows matching each criteria
        Returns:
            List of success rates (one for each criteria/matching shows pair)
        """
        if not criteria_list or not matching_shows_list or len(criteria_list) != len(matching_shows_list):
            return [None] * len(criteria_list)
        
        results = []
        calculator = SuccessScoreCalculator()
        # Use SUCCESS['threshold'] as the single source of truth for success threshold
        threshold = OptimizerConfig.SUCCESS.get('threshold', 0.6)
        
        for i, (criteria, matching_shows) in enumerate(zip(criteria_list, matching_shows_list)):
            try:
                if matching_shows is None or matching_shows.empty:
                    results.append(None)
                    continue
                
                component_score = calculator.calculate(matching_shows, threshold=threshold)
                results.append(component_score.score if component_score else None)
                
            except Exception as e:
                results.append(None)
                
        return results
    
    def _is_valid_dataframe(self, df) -> bool:
        """Check if a DataFrame is valid (not None and not empty).
        
        Args:
            df: DataFrame to check
            
        Returns:
            True if DataFrame is valid, False otherwise
        """
        # Use len() instead of .empty for better performance
        # .empty property has to check the shape which is slower
        return df is not None and len(df) > 0
        
    def _create_option_criteria(self, base_criteria: CriteriaDict, field: str, option_id, is_array_field: bool) -> CriteriaDict:
        """Create criteria with a specific option value for a field or remove the field.
        
        Args:
            base_criteria: Base criteria to modify conforming to CriteriaDict
            field: Field to set or remove (should be a database column name/ID)
            option_id: Option ID to set for the field, or 'remove' to remove the field
            is_array_field: Whether this is an array field (ignored for 'remove')
            
        Returns:
            New criteria dictionary with the option set or field removed conforming to CriteriaDict
        """
        # Create a shallow copy of the base criteria
        # Use dict comprehension instead of dict() for better performance
        new_criteria = {k: v for k, v in base_criteria.items()}
        
        # Special handling for 'remove' option
        if option_id == 'remove':
            if field in new_criteria:
                del new_criteria[field]
            return new_criteria
        
        # Convert option ID to integer for normal options
        # Only convert if it's not already an integer to avoid unnecessary conversion
        if not isinstance(option_id, int) and option_id != 'remove':
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
            
        if option_id == 'remove':
            return self.REC_TYPE_REMOVE
            
        if impact > 0:
            if is_field_selected:
                if is_option_selected:
                    return None
                else:
                    return self.REC_TYPE_CHANGE
            else:
                return self.REC_TYPE_ADD
        else:
            # Only recommend removal for selected options with negative impact
            if is_option_selected:
                return self.REC_TYPE_REMOVE
            else:
                # No recommendation for unselected options with negative impact
                return None
    
    def calculate_criteria_impact(self, criteria: CriteriaDict, matching_shows: pd.DataFrame = None, option_matching_shows_map: Dict[str, Dict[Any, pd.DataFrame]] = None, integrated_data: IntegratedData = None, fields_to_analyze: List[str] = None) -> ImpactAnalysisResult:
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
            return ImpactAnalysisResult(
                criteria_impacts={},
                summary={
                    'field_count': 0,
                    'option_count': 0,
                    'recommendation_counts': {'add': 0, 'change': 0, 'remove': 0}
                },
                error='Cannot calculate criteria impact: empty criteria'
            )
            
        try:
            # Normalize base criteria once at the beginning
            normalized_base_criteria = self.field_manager.normalize_criteria(criteria)
        except ValueError as e:
            # Re-raise with more context about the validation error
            raise ValueError(f"Invalid criteria format: {str(e)}") from e
            
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
                    has_valid_criteria = True  # Assume valid if we can't check properly
                    break
                    
            if not has_valid_criteria:
                return ImpactAnalysisResult(
                    criteria_impacts={},
                    summary={
                        'field_count': 0,
                        'option_count': 0,
                        'recommendation_counts': {'add': 0, 'change': 0, 'remove': 0}
                    },
                    error='Cannot calculate criteria impact: criteria contains only empty values'
                )
        except Exception as e:
            # If we can't check criteria at all, continue with processing
            # We'll continue and let other validation steps catch issues
            pass
            
        if matching_shows is None or matching_shows.empty:
            return ImpactAnalysisResult(
                criteria_impacts={},
                summary={
                    'field_count': 0,
                    'option_count': 0,
                    'recommendation_counts': {'add': 0, 'change': 0, 'remove': 0}
                },
                error='Cannot calculate criteria impact: no matching shows'
            )
            
        # Initialize impact scores dictionary
        # Ensure all keys are strings to prevent 'criteria_type' errors later
        impact_scores = {}
        
        try:
            # Let the field manager handle array field identification
            array_field_mapping = self.field_manager.get_array_field_mapping()
            
            # Calculate base success rate for all matching shows
            base_rate, base_info = self._calculate_success_rate(matching_shows)
            
            # Check if we have a valid base success rate
            if base_rate is None:
                return ImpactAnalysisResult(
                    criteria_impacts={},
                    summary={
                        'field_count': 0,
                        'option_count': 0,
                        'recommendation_counts': {'add': 0, 'change': 0, 'remove': 0}
                    },
                    error='Cannot calculate impact: invalid base success rate'
                )
                
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
            
            # If fields_to_analyze is provided, use it to filter the selected fields
            # This allows us to limit expensive calculations for change/remove recommendations
            # while still calculating all add recommendations
            if fields_to_analyze:
                # Only limit selected fields (for change/remove), not unselected fields (for add)
                selected_fields = [field for field in selected_fields if field in fields_to_analyze]
                
            # Combine selected and prioritized unselected fields
            fields_to_process = selected_fields + prioritized_unselected
            
            # Check if we have fields to process
            if not fields_to_process:
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
                    recommendation_types.append(self.REC_TYPE_REMOVE)
                    

                    
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
                        recommendation_types.append(self.REC_TYPE_CHANGE)
                else:
                    # For fields not in criteria, create "Add" recommendations for each option
                    for option in options:
                        # Create criteria with this option
                        new_criteria = self._create_option_criteria(criteria, current_field, option.id, is_array_field)
                        option_key = int(option.id)
                        option_name = self.field_manager.get_name(current_field, option.id)
                        
                        # Debug statement removed to reduce verbosity
                        batch_criteria.append(new_criteria)
                        option_data.append((option_key, option_name))
                        # We'll determine the actual recommendation type after calculating impact
                        # For now, mark these as potential ADD recommendations
                        recommendation_types.append(self.REC_TYPE_ADD)
                
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
                    
                    # Pre-check integrated data validity once to avoid repeated checks
                    has_valid_integrated_data = integrated_data and 'shows' in integrated_data and len(integrated_data['shows']) > 0
                    has_valid_matching_shows = matching_shows is not None and len(matching_shows) > 0
                    
                    # Prepare data source once
                    data_source = None
                    if has_valid_integrated_data:
                        data_source = integrated_data['shows']
                    elif has_valid_matching_shows:
                        data_source = matching_shows
                    
                    # Process all option criteria in a single loop to reduce overhead
                    for i, option_criteria in enumerate(batch_criteria):
                        try:
                            # Get the option data
                            option_id, option_name = option_data[i]
                            
                            # Use the prepared data source from earlier
                            if data_source is not None:
                                # Find matching shows for this option
                                option_shows, confidence_info = self.matcher.find_matches_with_fallback(option_criteria, data_source)
                            else:
                                # Skip if no valid data source
                                field_options_map[option_id] = pd.DataFrame()
                                confidence_info = ConfidenceInfo({
                                    'match_level': 1,  # Use 1 as the default match level
                                    'error': 'Missing integrated data for option'
                                })
                                                        
                            # Process option matching shows
                            
                            # Check if we got valid shows - inline check for better performance
                            if option_shows is not None and isinstance(option_shows, pd.DataFrame) and len(option_shows) > 0:
                                # Store in field_options_map
                                field_options_map[option_id] = option_shows
                            else:
                                # Track missing data
                                missing_data_count += 1
                        except Exception as e:
                            # Skip this option on error
                            continue
                    
                    # Set the option_matching_shows_map for this field
                    if not option_matching_shows_map:
                        option_matching_shows_map = {}
                    option_matching_shows_map[current_field] = field_options_map
                    
                    # Track missing data count but don't log it
                    
                    # Reset batch operation flag
                    setattr(self.matcher, '_in_batch_operation', False)
                else:
                    # Use the provided option_matching_shows_map
                    field_options_map = option_matching_shows_map[current_field]
                
                # Field is already initialized in impact_scores above
                # This ensures we maintain a consistent data structure
                
                # Prepare batch processing for success rate calculations
                valid_options = []
                valid_option_shows = []
                option_id_map = {}
                
                # Collect all valid options and their matching shows for batch processing
                for i, (option_id, option_name) in enumerate(option_data):
                    # Get the option shows from field_options_map
                    option_shows = field_options_map.get(option_id)
                    
                    # Skip options with no matching shows - inline check for better performance
                    if option_shows is None or (isinstance(option_shows, pd.DataFrame) and len(option_shows) == 0):
                        continue
                        
                    # Add to batch processing lists
                    valid_options.append((option_id, option_name))
                    valid_option_shows.append(option_shows)
                    option_id_map[len(valid_options) - 1] = option_id
                
                # Batch calculate success rates for all valid options
                if valid_options:
                    option_rates = self._batch_calculate_success_rates([{}] * len(valid_options), valid_option_shows)
                else:
                    option_rates = []
                
                # Process the batch results
                for i, option_rate in enumerate(option_rates):
                    if option_rate is None:
                        continue
                    
                    try:
                        option_id = option_id_map[i]
                        option_name = next(name for oid, name in valid_options if oid == option_id)
                        
                        # Calculate impact as the difference from base rate
                        impact = option_rate - base_rate
                        
                        # Store the original impact for reference
                        original_impact = impact
                        
                        # Special debug for tracking recommendation flow
                        rec_type = recommendation_types[i] if i < len(recommendation_types) else "unknown"
                        
                        # Get minimum impact threshold
                        min_impact = OptimizerConfig.SUGGESTIONS['minimum_impact']
                        
                        # Skip options with impact below minimum threshold
                        if abs(impact) < min_impact:  # If impact is too small
                            continue
                        
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
                        
                        # Determine recommendation type using helper method
                        recommendation_type = self._determine_recommendation_type(
                            option_id, impact, is_field_selected, is_option_selected
                        )
                        
                        # Skip options with no recommendation type
                        if recommendation_type is None:
                            continue  # Skip to next option
                        
                        # Store impact score with all relevant information
                        # Get the sample size safely - we've already validated option_shows is not None above
                        # This avoids redundant None check for better performance
                        sample_size = len(option_shows)
                        
                        impact_scores[current_field][option_id] = {
                            'option_id': option_id,
                            'option_name': option_name,
                            'impact': impact,
                            'original_impact': original_impact,
                            'success_rate': option_rate,
                            'sample_size': sample_size,
                            'recommendation_type': recommendation_type
                        }
                    except Exception as e:
                        # Skip this option if there's an error
                        continue
                
            # Check if we have any impact scores after processing all fields
            # Process impact scores
            
            # Debug statements for impact scores removed to reduce verbosity
            
            # Process unselected fields to generate 'add' recommendations
            # Debug statements removed to reduce verbosity
            
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
                            
                            # Always use the integrated data for unselected fields to ensure consistent comparison
                            if integrated_data and 'shows' in integrated_data and not integrated_data['shows'].empty:
                                # Use the full dataset from integrated_data for testing different criteria
                                option_shows, confidence_info = self.matcher.find_matches_with_fallback(option_criteria, integrated_data['shows'])
                            else:
                                # Let the matcher use its own data source as fallback
                                option_shows, confidence_info = self.matcher.find_matches_with_fallback(option_criteria)
                            
                
                            # Skip options with no matching shows - use optimized check
                            if not self._is_valid_dataframe(option_shows):
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
                            min_impact = OptimizerConfig.SUGGESTIONS['minimum_impact']
                            if abs(impact) < min_impact:
                                # Skip options with minimal impact
                                continue
                            
                            # For unselected fields, only store positive impact scores (as 'add' recommendations)
                            # Skip negative impact scores for unselected fields as they're not actionable
                            if impact <= 0:
                                # Skip storing negative impact scores for unselected fields
                                continue
                                
                            # For unselected fields with positive impact, always use 'add' recommendation type
                            recommendation_type = self.REC_TYPE_ADD
                            
                            # Store impact score with all relevant information - ALWAYS as a complete dictionary
                            # This ensures consistent data structure throughout the application
                            # Pre-calculate sample size to avoid calling len() multiple times
                            sample_size = len(option_shows)
                            
                            impact_scores[field][option_id] = {
                                'option_id': option_id,
                                'option_name': option_name,
                                'impact': impact,
                                'original_impact': original_impact,
                                'success_rate': option_rate,
                                'sample_size': sample_size,
                                'recommendation_type': recommendation_type
                            }
                        except Exception as e:
                            continue
                
            # Summarize the impact scores we've generated
            # Count fields, options, and recommendation types
            field_count = 0
            option_count = 0
            recommendation_counts = {'add': 0, 'change': 0, 'remove': 0}
            
            # Count fields, options, and recommendation types
            for field, options in impact_scores.items():
                if field in ('_summary', '_error'):
                    continue
                field_count += 1
                option_count += len(options)
                for option in options.values():
                    rec_type = option.get('recommendation_type')
                    if rec_type in recommendation_counts:
                        recommendation_counts[rec_type] += 1
            
            # Create summary statistics
            result_summary = {
                'field_count': field_count,
                'option_count': option_count,
                'recommendation_counts': recommendation_counts
            }
            
            # No longer storing summary in impact_scores for better performance
                
            # If we still have no impact scores, return empty result with error
            if not any(impact_scores.values()) or all(k == '_summary' for k in impact_scores.keys()):
                return ImpactAnalysisResult(
                    criteria_impacts={},
                    summary={
                        'field_count': 0,
                        'option_count': 0,
                        'recommendation_counts': {'add': 0, 'change': 0, 'remove': 0}
                    },
                    error='No impact scores could be generated from any selected or unselected fields'
                )
            
            # Prepare the result with proper typing
            result = ImpactAnalysisResult(
                criteria_impacts={
                    k: v for k, v in impact_scores.items() 
                    if k != '_summary' and k != '_error'
                },
                summary={
                    'field_count': 0,
                    'option_count': 0,
                    'recommendation_counts': {'add': 0, 'change': 0, 'remove': 0}
                }
            )
            
            # Update summary if available
            if '_summary' in impact_scores:
                result.summary.update(impact_scores['_summary'])
                        
            return result

        except ValueError as ve:
            error_msg = f"ValueError in calculate_criteria_impact: {str(ve)}"
            return ImpactAnalysisResult(
                criteria_impacts={},
                summary={
                    'field_count': 0,
                    'option_count': 0,
                    'recommendation_counts': {'add': 0, 'change': 0, 'remove': 0}
                },
                error=error_msg
            )
        except Exception as e:
            error_msg = f"Unexpected error in calculate_criteria_impact: {str(e)}"
            return ImpactAnalysisResult(
                criteria_impacts={},
                summary={
                    'field_count': 0,
                    'option_count': 0,
                    'recommendation_counts': {'add': 0, 'change': 0, 'remove': 0}
                },
                error=error_msg
            )

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
            return {
                'success': ComponentScore(score=0.0, sample_size=0, confidence='none', error='No matching shows'),
                'audience': ComponentScore(score=0.0, sample_size=0, confidence='none', error='No matching shows'),
                'critics': ComponentScore(score=0.0, sample_size=0, confidence='none', error='No matching shows'),
                'longevity': ComponentScore(score=0.0, sample_size=0, confidence='none', error='No matching shows')
            }
            
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
                except Exception as e:
                    # Create error component score
                    component_scores[calculator.component_name] = ComponentScore(
                        component=calculator.component_name, 
                        score=None, 
                        sample_size=0, 
                        confidence='none', 
                        details={'status': 'calculation_error', 'error': str(e)}
                    )
            
            return component_scores
            
        except Exception as e:
            # Error handling
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
        
        # Calculate network scores
        network_matches = self._network_calculator.calculate_network_scores(criteria_str, matching_shows)
        
        return network_matches
