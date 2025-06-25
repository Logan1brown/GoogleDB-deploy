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
