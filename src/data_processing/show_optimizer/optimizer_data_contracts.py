"""Data contracts for the Show Optimizer.

This module defines explicit data contracts using TypedDict for dictionaries
that are passed between components in the Show Optimizer. These contracts
serve as documentation and enable static type checking.

Key contracts defined:
- CriteriaDict: Structure of criteria dictionaries used for show matching
- ConfidenceInfo: Structure of confidence information for match quality
- IntegratedData: Structure of integrated data dictionary with dataframes
"""

from typing import Dict, List, TypedDict, Optional, Any, Union, Tuple, Callable
try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired
from dataclasses import dataclass, field
import pandas as pd

from .optimizer_config import OptimizerConfig

__all__ = [
    'CriteriaDict', 'ConfidenceInfo', 'IntegratedData', 'NetworkMatch',
    'SuccessFactor', 'RecommendationItem', 'FieldValueData',
    'create_success_rate_data', 'create_field_value_key', 'parse_field_value_key',
    'create_field_value_data', 'update_confidence_info'
]


class CriteriaDict(TypedDict):
    """Explicit contract for criteria dictionary used throughout the optimizer.
    
    This defines the expected structure of criteria dictionaries passed between components.
    All fields are optional since criteria can be partially specified.
    """
    genre: NotRequired[List[str]]
    network_id: NotRequired[Union[int, List[int]]]
    studio_id: NotRequired[Union[int, List[int]]]
    source_type_id: NotRequired[Union[int, List[int]]]
    character_type_ids: NotRequired[List[int]]
    plot_element_ids: NotRequired[List[int]]
    theme_element_ids: NotRequired[List[int]]
    tone_ids: NotRequired[List[int]]
    time_setting_ids: NotRequired[List[int]]
    location_setting_ids: NotRequired[List[int]]
    episode_count: NotRequired[int]
    order_type_id: NotRequired[int]
    team_ids: NotRequired[List[int]]


class ConfidenceInfo(TypedDict):
    """Explicit contract for confidence information dictionary.
    
    This defines the expected structure of confidence info dictionaries used to
    track match quality and confidence levels throughout the analysis pipeline.
    """
    level: str  # 'high', 'medium', 'low', 'very_low', or 'none'
    match_level: NotRequired[int]  # Match level (1=exact, 2=close, 3=partial)
    match_count: NotRequired[int]  # Number of matching shows
    sample_size: NotRequired[int]  # Sample size used for calculations
    max_match_level: NotRequired[float]  # Maximum match level in the sample
    min_match_level: NotRequired[float]  # Minimum match level in the sample
    mean_match_level: NotRequired[float]  # Mean match level in the sample
    level_counts: NotRequired[Dict[int, int]]  # Counts of each match level
    error: NotRequired[str]  # Error message if applicable


class IntegratedData(TypedDict):
    """Explicit contract for integrated data dictionary.
    
    This defines the expected structure of the integrated data dictionary
    that contains all the dataframes used throughout the analysis pipeline.
    """
    shows: pd.DataFrame  # Main shows dataframe
    networks: NotRequired[pd.DataFrame]  # Networks dataframe
    studios: NotRequired[pd.DataFrame]  # Studios dataframe
    genres: NotRequired[pd.DataFrame]  # Genres dataframe
    success_metrics: NotRequired[pd.DataFrame]  # Success metrics dataframe
    team: NotRequired[pd.DataFrame]  # Team dataframe


# Helper functions for working with these contracts

def update_confidence_info(base_info: Dict[str, Any], updates: Dict[str, Any]) -> ConfidenceInfo:
    """Update confidence info with new values while maintaining the contract.
    
    Args:
        base_info: Existing confidence info dictionary
        updates: New values to update
        
    Returns:
        Updated ConfidenceInfo dictionary
    """
    # Create a copy to avoid modifying the original
    result = dict(base_info) if base_info else {}
    
    # Update with new values
    for key, value in updates.items():
        result[key] = value
    
    # Ensure the required fields are present
    if 'level' not in result:
        result['level'] = 'none'
        
    if 'match_level' not in result:
        result['match_level'] = 1  # Default to exact match (level 1)
    
    return result


def validate_criteria(criteria: Dict[str, Any]) -> CriteriaDict:
    """Validate and standardize criteria dictionary.
    
    Args:
        criteria: Raw criteria dictionary from user input or other sources
        
    Returns:
        Standardized CriteriaDict with validated fields
    """
    # Create a new dictionary that will match our CriteriaDict structure
    validated: Dict[str, Any] = {}
    
    # Validate and standardize each field if present
    if 'genre' in criteria:
        # Ensure genre is always a list
        if isinstance(criteria['genre'], str):
            validated['genre'] = [criteria['genre']]
        else:
            validated['genre'] = criteria['genre']
    
    # Handle network_id which can be single value or list
    if 'network_id' in criteria:
        if isinstance(criteria['network_id'], list):
            validated['network_id'] = criteria['network_id']
        else:
            validated['network_id'] = criteria['network_id']
    
    # Handle studio_id which can be single value or list
    if 'studio_id' in criteria:
        if isinstance(criteria['studio_id'], list):
            validated['studio_id'] = criteria['studio_id']
        else:
            validated['studio_id'] = criteria['studio_id']
    
    # Handle source_type_id which can be single value or list
    if 'source_type_id' in criteria:
        if isinstance(criteria['source_type_id'], list):
            validated['source_type_id'] = criteria['source_type_id']
        else:
            validated['source_type_id'] = criteria['source_type_id']
    
    # Ensure all *_ids fields are lists
    for field in ['character_type_ids', 'plot_element_ids', 'theme_element_ids', 
                 'tone_ids', 'time_setting_ids', 'location_setting_ids', 'team_ids']:
        if field in criteria:
            if not isinstance(criteria[field], list):
                validated[field] = [criteria[field]]
            else:
                validated[field] = criteria[field]
    
    # Simple scalar fields
    for field in ['episode_count', 'order_type_id']:
        if field in criteria:
            validated[field] = criteria[field]
    
    # Copy any other fields that might be used by extensions
    for key, value in criteria.items():
        if key not in validated:
            validated[key] = value
    
    return validated


def validate_integrated_data(data: Dict[str, pd.DataFrame]) -> IntegratedData:
    """Validate integrated data dictionary and ensure it has required fields.
    
    Args:
        data: Dictionary of dataframes from various sources
        
    Returns:
        Validated IntegratedData dictionary
        
    Raises:
        ValueError: If required 'shows' dataframe is missing or empty
    """
    if 'shows' not in data or data['shows'].empty:
        raise ValueError("Integrated data must contain a non-empty 'shows' dataframe")
    
    # Create a new dictionary that will match our IntegratedData structure
    validated: Dict[str, pd.DataFrame] = {'shows': data['shows']}
    
    # Copy optional dataframes if present
    for field in ['networks', 'studios', 'genres', 'success_metrics', 'team']:
        if field in data and not data[field].empty:
            validated[field] = data[field]
    
    return validated


class FieldValueData(TypedDict):
    """Explicit contract for field value data.
    
    This defines the expected structure for field values throughout the optimizer,
    ensuring consistent handling of different field types (scalar, list, etc.).
    """
    field_name: str  # Name of the field (e.g., 'genre', 'subgenres')
    value: Any  # The actual value (could be string, int, list, etc.)
    display_value: str  # String representation for display
    is_array: NotRequired[bool]  # Whether this is an array/list field
    selected: NotRequired[bool]  # Whether this value is selected in criteria


class FieldValueSuccessRate(TypedDict):
    """Explicit contract for field value success rate data.
    
    This defines the expected structure of success rate information for a specific
    field value, used in network analysis and component analysis.
    """
    field_name: str  # Name of the field (e.g., 'genre', 'subgenres')
    value_name: str  # String representation of the value for display
    original_value_name: NotRequired[str]  # Original value name before cleaning
    value: Any  # The actual value (could be string, int, list, etc.)
    rate: float  # Success rate (0.0 to 1.0)
    sample_size: int  # Number of shows used to calculate rate
    has_data: bool  # Whether there is sufficient data for this value
    matching_shows: NotRequired[List[str]]  # List of matching show titles


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
            self.compatibility_score = OptimizerConfig.THRESHOLDS['minimum_compatibility']


class RecommendationItem(TypedDict):
    """Explicit contract for recommendation items.
    
    This defines the expected structure of recommendations generated by
    the recommendation engine and displayed to the user.
    """
    field: str  # Field name (e.g., 'genre', 'subgenres')
    impact: float  # Impact score (-1.0 to 1.0)
    confidence: str  # Confidence level (e.g., 'high', 'medium', 'low')
    recommendation_type: str  # Type of recommendation ('add', 'change', 'remove')
    explanation: str  # Explanation text
    # Fields for suggested values
    suggested_value: Any  # Suggested value for the field
    suggested_name: str  # Display name for the suggested value
    # Optional fields
    sample_size: NotRequired[int]  # Number of shows used in calculation
    current_value: NotRequired[Any]  # Current value if recommendation_type is 'change'
    current_name: NotRequired[str]  # Display name for the current value
    is_network_specific: NotRequired[bool]  # Whether this is network-specific
    metadata: NotRequired[Dict[str, Any]]  # Additional metadata for the recommendation
    # Legacy fields - kept for backward compatibility
    option: NotRequired[Any]  # Legacy field for option value
    option_name: NotRequired[str]  # Legacy field for option display name


# Helper functions for working with field values

def create_success_rate_data(field_name: str, value: Any, rate: float, sample_size: int, value_name: Optional[str] = None) -> Dict[str, Any]:
    """Create a standardized success rate data structure.
    
    This ensures consistent data structure between network-specific and overall success rates.
    
    Args:
        field_name: Name of the field (e.g., 'genre_id', 'tone_id')
        value: The field value
        rate: Success rate as a float between 0 and 1
        sample_size: Number of samples used to calculate the rate
        value_name: Optional human-readable name for the value
        
    Returns:
        A dictionary with standardized structure for success rate data
    """
    return {
        'field_name': field_name,
        'value': value,
        'value_name': value_name if value_name is not None else str(value),
        'rate': rate,
        'sample_size': sample_size
    }

def create_field_value_key(field_name: str, value: Any) -> str:
    """Create a standardized dictionary key for field value success rates.
    
    Handles different value types, including lists, to ensure keys are always strings.
    
    Args:
        field_name: Name of the field (e.g., 'genre', 'subgenres')
        value: The field value, which could be a string, int, list, etc.
        
    Returns:
        A string key in the format "field_name:value_string"
    """
    try:
        # Convert value to a string representation suitable for a dictionary key
        if value is None:
            value_str = "None"
        elif isinstance(value, list):
            # Handle empty lists
            if not value:
                value_str = "empty_list"
            else:
                # Always convert list elements to strings to avoid unhashable type errors
                # This ensures we can handle any type of list content
                try:
                    # Convert each element to string first, then sort
                    string_elements = [str(v) for v in value]
                    value_str = str(sorted(string_elements))
                except Exception:
                    # If any error occurs, fall back to a simple string representation
                    value_str = f"list_{hash(str(value))}"
        elif isinstance(value, dict):
            # Handle dictionaries by converting to sorted items string
            value_str = str(sorted([(str(k), str(v)) for k, v in value.items()]))
        else:
            # Handle all other types
            value_str = str(value)
            
        return f"{field_name}:{value_str}"
    except Exception as e:
        # Fallback for any unexpected errors
        OptimizerConfig.debug(f"Error creating field value key for {field_name}: {str(e)}", category='data')
        return f"{field_name}:error_{hash(str(field_name) + str(type(value)))}"


def parse_field_value_key(key: str) -> Tuple[str, Any]:
    """Extract field name and value from a formatted key.
    
    This is the inverse operation of create_field_value_key and ensures consistent
    parsing of field-value keys throughout the codebase.
    
    Args:
        key: A key in the format 'field_name:value'
        
    Returns:
        Tuple of (field_name, field_value)
    """
    if ':' not in key:
        return key, None
        
    field_name, field_value = key.split(':', 1)
    
    # Try to convert numeric values back to their original type
    if field_value.replace('.', '', 1).isdigit():
        # Convert to float if it has a decimal point
        if '.' in field_value:
            field_value = float(field_value)
        else:
            # Otherwise convert to int
            field_value = int(field_value)
    elif field_value == 'None':
        field_value = None
    elif field_value == 'empty_list':
        field_value = []
        
    return field_name, field_value


def create_field_value_data(field_name: str, value: Any, is_selected: bool = False) -> FieldValueData:
    """Create a standardized FieldValueData dictionary.
    
    Args:
        field_name: Name of the field (e.g., 'genre', 'subgenres')
        value: The field value, which could be a string, int, list, etc.
        is_selected: Whether this value is selected in the current criteria
        
    Returns:
        A FieldValueData dictionary with standardized structure
    """
    is_array = isinstance(value, (list, tuple))
    display_value = ', '.join(map(str, value)) if is_array else str(value)
    
    return {
        'field_name': field_name,
        'value': value,
        'display_value': display_value,
        'is_array': is_array,
        'selected': is_selected
    }


def create_success_rate(field_name: str, value: Any, rate: float, 
                       sample_size: int, matching_shows: List[str] = None) -> FieldValueSuccessRate:
    """Create a standardized FieldValueSuccessRate dictionary.
    
    Args:
        field_name: Name of the field (e.g., 'genre', 'subgenres')
        value: The field value, which could be a string, int, list, etc.
        rate: Success rate (0.0 to 1.0)
        sample_size: Number of shows used to calculate rate
        matching_shows: Optional list of matching show titles
        
    Returns:
        A FieldValueSuccessRate dictionary with standardized structure
    """
    # Create a standardized string representation of the value
    if isinstance(value, list):
        value_name = ', '.join(map(str, sorted(value)))
    else:
        value_name = str(value)
    
    result: FieldValueSuccessRate = {
        'field_name': field_name,
        'value_name': value_name,
        'value': value,
        'success_rate': rate,  # Use 'success_rate' key consistently throughout the codebase
        'sample_size': sample_size,
        'has_data': sample_size > 0
    }
    
    if matching_shows:
        result['matching_shows'] = matching_shows
    
    return result
