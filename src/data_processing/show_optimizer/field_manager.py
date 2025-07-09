"""Field Manager for Show Optimizer.

This module handles field options, validation, field mapping, and confidence calculation for the Show Optimizer.
It serves as the single authority for field mapping, validation, and confidence data in the system.

Key responsibilities:
- Field name mapping (e.g., 'network' to 'network_id')
- Field type identification (scalar vs array)
- Field value validation
- Field option management
- Sample size tracking and confidence calculation
- Criteria confidence analysis

Key components:
- FieldConfig: Configuration for a field (table, id field, name field)
- FieldOption: A single option for a field with ID and display name
- FieldValidation: Validation rules for a field
- FieldManager: Main class that manages field options, mapping, and confidence data
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import streamlit as st

from .optimizer_config import OptimizerConfig
from .optimizer_data_contracts import (
    CriteriaDict, ConfidenceInfo, IntegratedData, 
    FieldValueData, FieldValueSuccessRate,
    create_field_value_data, create_field_value_key
)


@dataclass
class FieldConfig:
    """Configuration for a field in the optimizer system."""
    table_name: str  # Reference table name
    id_field: str    # Field containing the ID
    name_field: str  # Field containing display name
    is_array: bool = False  # Whether field contains multiple values


@dataclass
class FieldOption:
    """A single option for a field with its ID and display name.
    
    For team members, all_ids contains all IDs that map to this name.
    For other fields, all_ids is None.
    """
    id: int
    name: str
    all_ids: Optional[List[int]] = None


@dataclass
class FieldValidation:
    """Validation rules for a field."""
    required: bool = False  # Whether the field is required
    min_values: int = 0     # Minimum number of values (for array fields)
    max_values: int = 0     # Maximum number of values (for array fields, 0 = unlimited)
    allowed_values: Optional[List[int]] = None  # List of allowed values (if restricted)
    dependent_on: Optional[str] = None  # Field this field depends on
    dependency_values: Optional[List[int]] = None  # Values of dependent field that enable this field


class FieldManager:
    """Manages field options, validation, and field mapping for Show Optimizer.
    
    This class serves as the single authority for field mapping and validation in the system.
    It provides methods for mapping field names, determining field types, and validating field values.
    """
    
    # Field configuration mapping
    FIELD_CONFIGS = {
        'character_types': FieldConfig('character_type_types', 'id', 'name', True),
        'genre': FieldConfig('genre_list', 'id', 'genre'),
        'subgenres': FieldConfig('genre_list', 'id', 'genre', True),
        'source_type': FieldConfig('source_types', 'id', 'type'),
        'plot_elements': FieldConfig('plot_element_types', 'id', 'name', True),
        'thematic_elements': FieldConfig('thematic_element_types', 'id', 'name', True),
        'tone': FieldConfig('tone_types', 'id', 'name'),
        'time_setting': FieldConfig('time_setting_types', 'id', 'name'),
        'location_setting': FieldConfig('location_setting_types', 'id', 'name'),
        'network': FieldConfig('network_list', 'id', 'network'),
        'studios': FieldConfig('studio_list', 'id', 'studio', True),
        'team_members': FieldConfig('api_show_comp_data', 'team_member_ids', 'team_member_names', True),
        'order_type': FieldConfig('order_types', 'id', 'type')
    }
    
    # Field validation rules
    FIELD_VALIDATIONS = {
        'genre': FieldValidation(required=True),
        'network': FieldValidation(required=False),
        'character_types': FieldValidation(min_values=0, max_values=3),
        'plot_elements': FieldValidation(min_values=0, max_values=3),
        'thematic_elements': FieldValidation(min_values=0, max_values=3),
        'team_members': FieldValidation(min_values=0, max_values=3),
        'studios': FieldValidation(min_values=0, max_values=2)
    }
    
    def __init__(self, reference_data: Dict[str, pd.DataFrame]):
        """Initialize the field manager.
        
        Args:
            reference_data: Dictionary of reference data tables for field options and validation
        """
        self.reference_data = reference_data
        self.options: Dict[str, List[FieldOption]] = {}
        self.sample_sizes: Dict[str, Dict[int, int]] = {}  # Field -> Value -> Count
        self._load_options()
        self._calculate_sample_sizes()
        
    def _load_options(self):
        """Load all field options from reference data."""
        try:
            for field_name, config in self.FIELD_CONFIGS.items():
                try:
                    # Special handling for different data sources
                    if field_name == 'subgenres':
                        df = self.reference_data['subgenres']
                    elif field_name == 'team_members':
                        df = self.reference_data[field_name]
                    else:
                        if field_name not in self.reference_data:
                            continue
                        df = self.reference_data[field_name]
                    
                    # Special handling for team members
                    if field_name == 'team_members':
                        # Use dictionary to maintain unique entries by ID
                        unique_members = {}
                        
                        # Process each team member
                        for _, row in df.iterrows():
                            team_member_ids = row.get('team_member_ids', [])
                            team_member_names = row.get('team_member_names', [])
                            
                            # Skip if either array is empty or not a list
                            if not isinstance(team_member_ids, list) or not isinstance(team_member_names, list) \
                                or len(team_member_ids) == 0 or len(team_member_names) == 0:
                                continue
                                
                            # Process each team member
                            for id, name in zip(team_member_ids, team_member_names):
                                # Convert name to string and check if empty
                                name_str = str(name).strip()
                                if len(name_str) > 0:
                                    if name_str not in unique_members:
                                        unique_members[name_str] = []
                                    unique_members[name_str].append(int(id))
                        
                        # Convert dictionary to list of options
                        options = []
                        for name, ids in unique_members.items():
                            # Use first ID as primary and store all IDs
                            options.append(FieldOption(id=ids[0], name=name, all_ids=ids))
                        self.options[field_name] = sorted(options, key=lambda x: x.name)
                    else:
                        clean_members = {}
                        for _, row in df.iterrows():
                            if pd.notna(row[config.id_field]) and pd.notna(row[config.name_field]):
                                id = int(row[config.id_field])
                                name = str(row[config.name_field])
                                clean_name = self._normalize_name(name)
                                if isinstance(clean_name, str) and len(clean_name) > 0:
                                    if clean_name not in clean_members:
                                        clean_members[clean_name] = id
                                    elif isinstance(clean_members[clean_name], int):
                                        # Convert to list if we have a duplicate
                                        clean_members[clean_name] = [clean_members[clean_name], id]
                                    else:
                                        # Add to existing list
                                        clean_members[clean_name].append(id)
                                        
                        # Create options with first ID for display but store all IDs
                        options = []
                        for name, id_or_ids in clean_members.items():
                            if isinstance(id_or_ids, list):
                                opt = FieldOption(id=id_or_ids[0], name=name)
                                opt.all_ids = id_or_ids
                            else:
                                opt = FieldOption(id=id_or_ids, name=name)
                                opt.all_ids = [id_or_ids]
                            options.append(opt)
                        
                        self.options[field_name] = sorted(options, key=lambda x: str(x.name) if x.name is not None else '')
                except Exception as e:
                    st.error(f"Error loading options for {field_name}: {str(e)}")
                    # Create empty options for this field to avoid errors later
                    self.options[field_name] = []
        except Exception as e:
            st.error(f"Error loading field options: {str(e)}")
            # Initialize empty options to avoid errors later
            self.options = {}
    
    def _calculate_sample_sizes(self):
        """Calculate sample sizes for each field value.
        
        This is used to provide confidence indicators for criteria selection.
        """
        try:
            # Get the main shows data
            shows_data = self.reference_data.get('shows', pd.DataFrame())
            if shows_data.empty:
                # Silent handling - no need to spam UI with this message
                return
                
            # Calculate sample sizes for each field
            for field_name, config in self.FIELD_CONFIGS.items():
                try:
                    self.sample_sizes[field_name] = {}
                    
                    # Skip fields not in shows data
                    if field_name not in shows_data.columns and f"{field_name}_id" not in shows_data.columns:
                        continue
                        
                    # Handle array vs scalar fields
                    if config.is_array:
                        # For array fields, count occurrences of each value
                        for _, row in shows_data.iterrows():
                            values = row.get(field_name, [])
                            if isinstance(values, list):
                                for value in values:
                                    if isinstance(value, (int, float)) and not np.isnan(value):
                                        value_int = int(value)
                                        self.sample_sizes[field_name][value_int] = self.sample_sizes[field_name].get(value_int, 0) + 1
                    else:
                        # For scalar fields, count occurrences of each value
                        field_id = f"{field_name}_id" if f"{field_name}_id" in shows_data.columns else field_name
                        if field_id in shows_data.columns:
                            value_counts = shows_data[field_id].value_counts().to_dict()
                            for value, count in value_counts.items():
                                if isinstance(value, (int, float)) and not np.isnan(value):
                                    self.sample_sizes[field_name][int(value)] = count
                except Exception as e:
                    st.error(f"Error calculating sample sizes for {field_name}: {str(e)}")
                    # Initialize empty sample sizes for this field to avoid errors later
                    self.sample_sizes[field_name] = {}
        except Exception as e:
            st.error(f"Error calculating sample sizes: {str(e)}")
            # Initialize empty sample sizes to avoid errors later
            self.sample_sizes = {}
    
    def _normalize_name(self, name: str) -> str:
        """Normalize a name for consistent matching.
        
        Handles variations like:
        - David E. Kelley vs David E Kelley
        - John Smith Jr. vs John Smith Jr
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized name for matching
        """
        # Remove extra spaces and periods from middle initials
        return ' '.join(part.strip(' .') for part in name.split())
    
    def get_options(self, field_name: str) -> List[FieldOption]:
        """Get all options for a field.
        
        Args:
            field_name: Name of the field to get options for
            
        Returns:
            List of FieldOption objects
        """
        return self.options.get(field_name, [])
    
    def get_display_options(self, field_name: str) -> List[Tuple[int, str]]:
        """Get (id, name) tuples for display.
        
        Args:
            field_name: Name of the field to get display options for
            
        Returns:
            List of (id, name) tuples sorted by name
        """
        return [(opt.id, opt.name) for opt in self.get_options(field_name)]
    
    def get_name(self, field_name: str, id: int) -> str:
        """Get display name for an ID.
        
        Args:
            field_name: Name of the field to get name for
            id: ID to get name for
            
        Returns:
            Display name for the ID
        """
        try:
            for opt in self.get_options(field_name):
                if opt.id == id:
                    return opt.name
            return f'Unknown ({id})'
        except Exception as e:
            st.error(f"Error getting name for {field_name} ID {id}: {str(e)}")
            return f'Error ({id})'
    
    def get_id_for_name(self, field_name: str, name: str) -> Optional[int]:
        """Get ID for a display name.
        
        Args:
            field_name: Name of the field to get ID for
            name: Display name to look up
            
        Returns:
            ID if found, None otherwise
        """
        if not name:
            return None
            
        try:
            for opt in self.get_options(field_name):
                if opt.name == name:
                    return opt.id
            return None
        except Exception as e:
            st.error(f"Error getting ID for {field_name} name '{name}': {str(e)}")
            return None
    
    def get_ids_for_names(self, field_name: str, names: List[str]) -> List[int]:
        """Get IDs for display names.
        
        Args:
            field_name: Name of the field to get IDs for
            names: List of display names to look up
            
        Returns:
            List of IDs for the given names
        """
        if not names:
            return []
            
        try:
            # For team members, get all IDs for each name
            if field_name == 'team_members':
                all_ids = []
                for name in names:
                    # Find the option with this name
                    opt = next((opt for opt in self.get_options('team_members') 
                              if opt.name == name), None)
                    if opt and hasattr(opt, 'all_ids'):
                        all_ids.extend(opt.all_ids)
                    elif opt:
                        all_ids.append(opt.id)
                return all_ids
            
            # For other fields, just take the ID for each name
            result = []
            for name in names:
                id = self.get_id_for_name(field_name, name)
                if id is not None:
                    result.append(id)
            return result
        except Exception as e:
            st.error(f"Error getting IDs for {field_name} names: {str(e)}")
            return []
    
    def get_validations(self) -> Dict[str, FieldValidation]:
        """Get validation rules for all fields.
        
        Returns:
            Dictionary mapping field names to FieldValidation objects
        """
        return self.FIELD_VALIDATIONS
    
    def create_field_value_data(self, field_name: str, value: Any, is_selected: bool = False) -> FieldValueData:
        """Create a standardized FieldValueData dictionary for a field value.
        
        This ensures consistent handling of field values throughout the system,
        especially for list-type fields that need special handling.
        
        Args:
            field_name: Name of the field (e.g., 'genre', 'subgenres')
            value: The field value, which could be a string, int, list, etc.
            is_selected: Whether this value is selected in the current criteria
            
        Returns:
            A FieldValueData dictionary with standardized structure
        """
        # Check if this is an array field
        is_array = False
        if field_name in self.FIELD_CONFIGS:
            is_array = self.FIELD_CONFIGS[field_name].is_array
        
        # For array fields, ensure value is a list
        if is_array and not isinstance(value, (list, tuple)):
            value = [value] if value is not None else []
        
        # Create a standardized display value
        if is_array and isinstance(value, (list, tuple)):
            # For list fields, join the display names
            display_names = []
            for item in value:
                name = self.get_name_for_id(field_name, item) if isinstance(item, int) else str(item)
                if name:
                    display_names.append(name)
            display_value = ', '.join(display_names) if display_names else ''
        else:
            # For scalar fields, use the name for the ID or the string value
            display_value = self.get_name_for_id(field_name, value) if isinstance(value, int) else str(value)
        
        # Return a standardized FieldValueData dictionary
        return {
            'field_name': field_name,
            'value': value,
            'display_value': display_value,
            'is_array': is_array,
            'selected': is_selected
        }
    
    def create_success_rate_data(self, field_name: str, value: Any, rate: float, 
                               sample_size: int, matching_shows: List[str] = None) -> FieldValueSuccessRate:
        """Create a standardized FieldValueSuccessRate dictionary for a field value.
        
        This ensures consistent handling of success rates throughout the system,
        especially for list-type fields that need special handling.
        
        Args:
            field_name: Name of the field (e.g., 'genre', 'subgenres')
            value: The field value, which could be a string, int, list, etc.
            rate: Success rate (0.0 to 1.0)
            sample_size: Number of shows used to calculate rate
            matching_shows: Optional list of matching show titles
            
        Returns:
            A FieldValueSuccessRate dictionary with standardized structure
        """
        # Check if this is an array field
        is_array = False
        if field_name in self.FIELD_CONFIGS:
            is_array = self.FIELD_CONFIGS[field_name].is_array
        
        # Create a standardized value name
        if is_array and isinstance(value, (list, tuple)):
            # For list fields, join the display names
            display_names = []
            for item in value:
                name = self.get_name_for_id(field_name, item) if isinstance(item, int) else str(item)
                if name:
                    display_names.append(name)
            value_name = ', '.join(sorted(display_names)) if display_names else ''
        else:
            # For scalar fields, use the name for the ID or the string value
            value_name = self.get_name_for_id(field_name, value) if isinstance(value, int) else str(value)
        
        # Create the success rate dictionary
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
    
    def validate_criteria(self, criteria: CriteriaDict) -> List[str]:
        """Validate criteria against field options and return validation errors.
        
        Args:
            criteria: Dictionary of criteria to validate conforming to CriteriaDict
            
        Returns:
            List of validation error messages (empty if valid)
        """
        try:
            errors = []
            
            # Check required fields
            for field_name, validation in self.FIELD_VALIDATIONS.items():
                if validation.required and field_name not in criteria:
                    errors.append(f"{field_name} is required")
                    
            # Check array field constraints
            for field_name, value in criteria.items():
                if field_name not in self.FIELD_VALIDATIONS:
                    continue
                    
                validation = self.FIELD_VALIDATIONS[field_name]
                config = self.FIELD_CONFIGS.get(field_name)
                
                if config and config.is_array:
                    # Check array size constraints
                    if not isinstance(value, list):
                        value = [value]
                        
                    if validation.min_values > 0 and len(value) < validation.min_values:
                        errors.append(f"{field_name} requires at least {validation.min_values} values")
                        
                    if validation.max_values > 0 and len(value) > validation.max_values:
                        errors.append(f"{field_name} allows at most {validation.max_values} values")
                        
                # Check allowed values
                if validation.allowed_values is not None:
                    if isinstance(value, list):
                        for v in value:
                            if v not in validation.allowed_values:
                                errors.append(f"{v} is not a valid value for {field_name}")
                    elif value not in validation.allowed_values:
                        errors.append(f"{value} is not a valid value for {field_name}")
                        
                # Check dependencies
                if validation.dependent_on and validation.dependency_values:
                    dependent_field = validation.dependent_on
                    if dependent_field not in criteria:
                        errors.append(f"{field_name} requires {dependent_field} to be specified")
                    else:
                        dependent_value = criteria[dependent_field]
                        if dependent_value not in validation.dependency_values:
                            errors.append(f"{field_name} requires {dependent_field} to be {validation.dependency_values}")
            
            return errors
        except Exception as e:
            st.error(f"Error validating criteria: {str(e)}")
            return [f"Validation error: {str(e)}"]
    
    def get_sample_size(self, field_name: str, value_key: Union[int, List[int]]) -> int:
        """Get the sample size for a field value.
        
        Args:
            field_name: Name of the field
            value_key: Value or list of values to get sample size for
            
        Returns:
            Sample size for the field value(s)
        """
        try:
            if field_name not in self.sample_sizes:
                return 0

            # Convert list to tuple for caching (lists aren't hashable)
            if isinstance(value_key, list):
                # For multiple values, return the minimum sample size
                # This represents the most restrictive case
                sizes = [self.sample_sizes[field_name].get(v, 0) for v in value_key]
                return min(sizes) if sizes else 0
            else:
                return self.sample_sizes[field_name].get(value_key, 0)
        except Exception as e:
            st.error(f"Error getting sample size for {field_name}: {str(e)}")
            return 0

    def calculate_confidence(self, criteria: CriteriaDict) -> ConfidenceInfo:
        """Calculate confidence levels for criteria.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            
        Returns:
            Dictionary with confidence information conforming to ConfidenceInfo:
            - level: Overall confidence level ('none', 'low', 'medium', 'high')
            - score: Confidence score between 0 and 1
            - sample_size: Estimated sample size for the criteria
            - fields: Dictionary mapping fields to their individual confidence levels
        """
        try:
            # Calculate sample sizes for each field
            field_samples = {}
            sample_sizes = []
            
            # Process all fields at once to avoid repeated dictionary lookups
            for field, value in criteria.items():
                if field in self.sample_sizes:
                    # Get sample size for this field
                    sample_size = self.get_sample_size(field, value)
                    sample_sizes.append(sample_size)
                    
                    # Store field-specific confidence info
                    field_samples[field] = {
                        'sample_size': sample_size,
                        'confidence': OptimizerConfig.get_confidence_level(sample_size, match_level=1)
                    }
                
            # Estimate overall sample size (use the most restrictive field)
            # This is a conservative estimate
            overall_sample = min(sample_sizes) if sample_sizes else 0
            
            # Calculate overall confidence
            criteria_count = len(criteria)
            total_criteria = len(self.FIELD_CONFIGS)
            confidence_score = OptimizerConfig.calculate_confidence_score(
                overall_sample, criteria_count, total_criteria, match_level=1)
            
            # Map confidence score to level using OptimizerConfig
            confidence_level = OptimizerConfig.map_confidence_score_to_level(confidence_score)
                
            # Ensure we conform to the ConfidenceInfo contract
            return {
                'level': confidence_level,  # Use 'level' instead of 'overall' to match contract
                'score': confidence_score,
                'sample_size': overall_sample,
                'match_level': 1,  # Default match level as required by contract
                'fields': field_samples
            }
        except Exception as e:
            st.error(f"Error calculating confidence: {str(e)}")
            # Return default values with no confidence
            return {
                'overall': 'none',
                'score': 0.0,
                'sample_size': 0,
                'fields': {}
            }
    
    def normalize_criteria(self, criteria: CriteriaDict) -> CriteriaDict:
        """Normalize criteria for consistent processing.
        
        This method ensures that:
        1. List values with single items are converted to scalar values for scalar fields
        2. Scalar values are converted to lists for array fields
        3. Numeric IDs are properly typed as integers for all fields
        4. All field names are strings
        
        Args:
            criteria: Dictionary of criteria to normalize conforming to CriteriaDict
            
        Returns:
            Normalized criteria dictionary conforming to CriteriaDict
            
        Raises:
            ValueError: If any field name is not a string
        """
        if not criteria:
            return {}
            
        # Validate field names in FIELD_CONFIGS
        for field in self.FIELD_CONFIGS:
            if not isinstance(field, str):
                raise ValueError(f"Invalid field name in FIELD_CONFIGS: {field!r} is not a string")
        
        # Fields that should have integer IDs
        numeric_id_fields = list(self.FIELD_CONFIGS.keys())  # All fields use numeric IDs
        
        normalized_criteria = {}
        
        for key, value in criteria.items():
            # Validate field name is a string
            if not isinstance(key, str):
                raise ValueError(f"Field name must be a string, got {key!r} of type {type(key).__name__}")
                
            # Skip None values
            if value is None:
                continue
                
            # Get field type
            field_type = self.get_field_type(key)
            
            # Convert single-item lists to scalar values for scalar fields
            if isinstance(value, list) and len(value) == 1 and field_type == 'scalar':
                normalized_criteria[key] = value[0]
            # Convert scalar values to lists for array fields
            elif not isinstance(value, list) and field_type == 'array':
                normalized_criteria[key] = [value]
            else:
                normalized_criteria[key] = value
            
            # Ensure numeric IDs are integers for scalar fields
            if field_type == 'scalar' and key in numeric_id_fields:
                if not isinstance(normalized_criteria[key], int):
                    try:
                        # Only convert if it's a string that represents a number
                        if isinstance(normalized_criteria[key], str) and normalized_criteria[key].isdigit():
                            normalized_criteria[key] = int(normalized_criteria[key])
                    except (ValueError, TypeError) as e:
                        # Debug statement removed to reduce verbosity
                        pass
                    except (ValueError, TypeError):
                        # If conversion fails, keep the original value
                        # Debug statement removed to reduce verbosity
                        pass
            
            # Ensure array field values are lists of integers where appropriate
            if field_type == 'array' and isinstance(normalized_criteria[key], list) and key in numeric_id_fields:
                try:
                    # Convert string IDs to integers if needed
                    normalized_criteria[key] = [int(item) if isinstance(item, str) and item.isdigit() else item 
                                               for item in normalized_criteria[key]]
                except (ValueError, TypeError):
                    # If conversion fails, keep the original values
                    # Debug statement removed to reduce verbosity
                    pass
        
        # Debug statement removed to reduce verbosity
            
        return normalized_criteria
    
    def get_array_field_mapping(self) -> Dict[str, str]:
        """Get the mapping from array field names to their column names in the DataFrame.
        
        Returns:
            Dictionary mapping array field names to column names.
        """
        return {
            'character_types': 'character_type_ids',
            'plot_elements': 'plot_element_ids',
            'thematic_elements': 'thematic_element_ids',
            'team_members': 'team_member_ids',
            'subgenres': 'subgenres',  # This one doesn't have _ids suffix
            'studios': 'studios'       # This one doesn't have _ids suffix
        }
        
    def get_field_column_name(self, field_name: str, data_columns: List[str]) -> Optional[str]:
        """Get the actual column name for a field in a DataFrame.
        
        This method maps the field name to its corresponding column name and verifies
        that the column exists in the provided data_columns list.
        
        Args:
            field_name: Field name to get column name for
            data_columns: List of available columns in the DataFrame
            
        Returns:
            Column name if found, None otherwise
        """
        try:
            # Use the central mapping method to get the correct column name
            mapped_name = self.map_field_name(field_name)
            
            # Verify the mapped column exists in the data
            if mapped_name in data_columns:
                return mapped_name
            
            # If the mapped column doesn't exist, return None
            return None
        except Exception as e:
            # Return None on error
            return None
        
    def get_criteria_importance(self, field_name: str) -> str:
        """Get the importance level of a criteria field.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Importance level: 'essential', 'core', 'primary', or 'secondary'
        """
        try:
            # Access the CRITERIA_IMPORTANCE dictionary from OptimizerConfig
            criteria_importance = getattr(OptimizerConfig, 'CRITERIA_IMPORTANCE', {})
            return criteria_importance.get(field_name, 'secondary')
        except Exception as e:
            st.error(f"Error determining importance for field '{field_name}': {str(e)}")
            # Default to secondary importance as fallback
            return 'secondary'
    
    def classify_criteria_by_importance(self, criteria: CriteriaDict) -> Dict[str, CriteriaDict]:
        """Classify criteria by their importance levels.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            
        Returns:
            Dictionary with keys 'essential', 'core', 'primary', 'secondary',
            each containing a dictionary of criteria at that importance level conforming to CriteriaDict
        """
        try:
            classified = {
                'essential': {},
                'core': {},
                'primary': {},
                'secondary': {}
            }
            
            for field_name, value in criteria.items():
                try:
                    importance = self.get_criteria_importance(field_name)
                    classified[importance][field_name] = value
                except Exception as e:
                    st.error(f"Error classifying field '{field_name}': {str(e)}")
                    # Default to secondary importance if there's an error
                    classified['secondary'][field_name] = value
                
            return classified
        except Exception as e:
            st.error(f"Error classifying criteria by importance: {str(e)}")
            # Return empty classification as fallback
            return {
                'essential': {},
                'core': {},
                'primary': {},
                'secondary': criteria  # Put all criteria in secondary as fallback
            }
    
    # Field mapping methods
    
    def map_field_name(self, field_name: str, data_columns: List[str] = None) -> str:
        """Map a field name to its corresponding column name in the DataFrame.
        
        This is the central method for field name mapping in the system.
        It handles the conversion of user-facing field names to database column names.
        
        Args:
            field_name: Field name to map
            data_columns: Optional list of available columns to validate against
            
        Returns:
            Mapped column name for the field
        """
        try:
            # Special case for array fields which have a standard mapping
            array_field_mapping = self.get_array_field_mapping()
            if field_name in array_field_mapping:
                mapped_name = array_field_mapping[field_name]
                return mapped_name
            
            # For scalar fields, use _id suffix which is common in the database
            field_type = self.get_field_type(field_name)
            if field_type == 'scalar':
                return f"{field_name}_id"
            
            # If we get here, it's not a recognized array field or scalar field
            # Return the original field name as a last resort
            return field_name
        except Exception as e:
            # Return the original field name on error
            return field_name
    
    def standardize_field_name(self, field_name: str) -> str:
        """Standardize a field name by removing common suffixes.
        
        This ensures consistent field name mapping between network rates and criteria keys.
        
        Args:
            field_name: Name of the field to standardize (e.g., 'genre_id', 'genre_name')
            
        Returns:
            The standardized field name (e.g., 'genre')
        """
        try:
            # Normalize field name by removing common suffixes
            if field_name.endswith('_id') or field_name.endswith('_ids'):
                return field_name[:-3] if field_name.endswith('_id') else field_name[:-4]
            elif field_name.endswith('_name') or field_name.endswith('_names'):
                return field_name[:-5] if field_name.endswith('_name') else field_name[:-6]
            return field_name
        except Exception as e:
            # If any error occurs, return the original field name
            return field_name
    
    def has_field(self, field_name: str) -> bool:
        """Check if a field exists in the field manager.
        
        Args:
            field_name: Name of the field to check
            
        Returns:
            True if the field exists, False otherwise
        """
        try:
            # Check direct field name
            if field_name in self.FIELD_CONFIGS:
                return True
                
            # Check normalized field name (without _id, _ids, _name, _names suffixes)
            base_field = field_name
            if field_name.endswith('_id') or field_name.endswith('_ids'):
                base_field = field_name[:-3] if field_name.endswith('_id') else field_name[:-4]
            elif field_name.endswith('_name') or field_name.endswith('_names'):
                base_field = field_name[:-5] if field_name.endswith('_name') else field_name[:-6]
                
            return base_field in self.FIELD_CONFIGS
        except Exception as e:
            # Error checking if field exists - silently continue
            return False
    
    def get_field_type(self, field_name: str) -> str:
        """Get the field type (array or scalar).
        
        Args:
            field_name: Name of the field
            
        Returns:
            Field type ('array' or 'scalar')
        """
        try:
            # Normalize field name if needed
            base_field = field_name
            if field_name.endswith('_id') or field_name.endswith('_ids'):
                base_field = field_name[:-3] if field_name.endswith('_id') else field_name[:-4]
            elif field_name.endswith('_name') or field_name.endswith('_names'):
                base_field = field_name[:-5] if field_name.endswith('_name') else field_name[:-6]
                
            if base_field in self.FIELD_CONFIGS and self.FIELD_CONFIGS[base_field].is_array:
                return 'array'
                
            # Default to scalar
            return 'scalar'
        except Exception as e:
            st.error(f"Error determining field type for {field_name}: {str(e)}")
            # Default to scalar as fallback
            return 'scalar'
    
    def validate_field_name(self, field_name: str) -> bool:
        """Validate if a field name is recognized by the system.
        
        Args:
            field_name: Field name to validate
            
        Returns:
            True if field name is valid, False otherwise
        """
        try:
            # Check if field is in FIELD_CONFIGS
            if field_name in self.FIELD_CONFIGS:
                return True
                
            # Check if field is in array_field_mapping
            array_field_mapping = self.get_array_field_mapping()
            if field_name in array_field_mapping:
                return True
                
            return False
        except Exception as e:
            st.error(f"Error validating field name '{field_name}': {str(e)}")
            # Return False as fallback to be safe
            return False
    
    def validate_field_value(self, field_name: str, value: Any) -> Tuple[bool, str]:
        """Validate a field value against validation rules.
        
        Args:
            field_name: Field name to validate value for
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if value is valid, False otherwise
            - error_message: Error message if value is invalid, empty string otherwise
        """
        try:
            # Check if field has validation rules
            if field_name not in self.FIELD_VALIDATIONS:
                return True, ""
                
            validation = self.FIELD_VALIDATIONS[field_name]
            field_type = self.get_field_type(field_name)
            
            # Check for None or empty values
            if value is None:
                if validation.required:
                    return False, f"{field_name} is required"
                return True, ""
                
            # Handle array fields
            if field_type == 'array':
                # Ensure value is a list
                if not isinstance(value, list):
                    value = [value]
                    
                # Check array size constraints
                if validation.min_values > 0 and len(value) < validation.min_values:
                    return False, f"{field_name} requires at least {validation.min_values} values"
                    
                if validation.max_values > 0 and len(value) > validation.max_values:
                    return False, f"{field_name} allows at most {validation.max_values} values"
                    
                # Check allowed values
                if validation.allowed_values is not None:
                    for v in value:
                        if v not in validation.allowed_values:
                            return False, f"{v} is not a valid value for {field_name}"
            else:  # Handle scalar fields
                # Check allowed values
                if validation.allowed_values is not None and value not in validation.allowed_values:
                    return False, f"{value} is not a valid value for {field_name}"
                    
            # Check dependencies
            # Note: This would typically be checked at a higher level when all criteria are available
            
            return True, ""
        except Exception as e:
            st.error(f"Error validating field value for {field_name}: {str(e)}")
            # Return as invalid with error message
            return False, f"Validation error: {str(e)}"
