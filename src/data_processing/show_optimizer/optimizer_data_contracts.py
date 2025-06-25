"""Data contracts for the Show Optimizer.

This module defines explicit data contracts using TypedDict for dictionaries
that are passed between components in the Show Optimizer. These contracts
serve as documentation and enable static type checking.

Key contracts defined:
- CriteriaDict: Structure of criteria dictionaries used for show matching
- ConfidenceInfo: Structure of confidence information for match quality
- IntegratedData: Structure of integrated data dictionary with dataframes
"""

from typing import Dict, List, Optional, Tuple, Any, Union, TypedDict, NotRequired
import pandas as pd


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
    match_level: NotRequired[float]  # Average match level (1=exact, 2=close, 3=partial)
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

def create_default_confidence_info() -> ConfidenceInfo:
    """Create a default confidence info dictionary with standard values.
    
    Returns:
        Default ConfidenceInfo dictionary
    """
    return {
        'level': 'none',
        'match_level': 0.0,
        'match_count': 0,
        'sample_size': 0
    }


def update_confidence_info(base_info: Dict[str, Any], updates: Dict[str, Any]) -> ConfidenceInfo:
    """Update confidence info with new values while maintaining the contract.
    
    Args:
        base_info: Existing confidence info dictionary
        updates: New values to update
        
    Returns:
        Updated ConfidenceInfo dictionary
    """
    # Start with default values if base_info is empty
    if not base_info:
        result = create_default_confidence_info()
    else:
        # Create a copy to avoid modifying the original
        result = dict(base_info)
    
    # Update with new values
    for key, value in updates.items():
        result[key] = value
    
    # Ensure the required 'level' field is present
    if 'level' not in result:
        result['level'] = 'none'
    
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


class NetworkMatch(TypedDict):
    """Explicit contract for network match data.
    
    This defines the expected structure of network match information used
    for ranking and displaying network compatibility.
    """
    network_id: int  # Network ID
    name: str  # Network name
    compatibility_score: float  # Overall compatibility score (0.0 to 1.0)
    success_probability: float  # Probability of success (0.0 to 1.0)
    confidence: str  # Confidence level (e.g., 'high', 'medium', 'low')
    sample_size: int  # Number of shows used in calculation
    tier: NotRequired[int]  # Tier assignment (1=best, 2=good, 3=average, etc.)
    tier_name: NotRequired[str]  # Display name for the tier
    explanation: NotRequired[str]  # Explanation of the compatibility score
    matching_shows: NotRequired[List[str]]  # List of matching show titles


class RecommendationItem(TypedDict):
    """Explicit contract for recommendation items.
    
    This defines the expected structure of recommendations generated by
    the recommendation engine and displayed to the user.
    """
    field: str  # Field name (e.g., 'genre', 'subgenres')
    option: Any  # Option value (could be string, int, list, etc.)
    option_name: str  # Display name for the option
    impact: float  # Impact score (-1.0 to 1.0)
    confidence: str  # Confidence level (e.g., 'high', 'medium', 'low')
    recommendation_type: str  # Type of recommendation ('add', 'change', 'remove')
    explanation: str  # Explanation text
    sample_size: NotRequired[int]  # Number of shows used in calculation
    current_value: NotRequired[Any]  # Current value if recommendation_type is 'change'
    is_network_specific: NotRequired[bool]  # Whether this is network-specific


# Helper functions for working with field values

def create_field_value_key(field_name: str, value: Any) -> str:
    """Create a standardized dictionary key for field value success rates.
    
    Handles different value types, including lists, to ensure keys are always strings.
    
    Args:
        field_name: Name of the field (e.g., 'genre', 'subgenres')
        value: The field value, which could be a string, int, list, etc.
        
    Returns:
        A string key in the format "field_name:value_string"
    """
    # Convert value to a string representation suitable for a dictionary key
    if isinstance(value, list):
        # Sort the list to ensure consistent key generation
        value_str = str(sorted(value))
    else:
        value_str = str(value)
        
    return f"{field_name}:{value_str}"


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
        'rate': rate,
        'sample_size': sample_size,
        'has_data': sample_size > 0
    }
    
    if matching_shows:
        result['matching_shows'] = matching_shows
    
    return result
