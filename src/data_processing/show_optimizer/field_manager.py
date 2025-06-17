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
            # Debug: Check what reference data keys are available
            st.write(f"DEBUG: Available reference data keys: {list(self.reference_data.keys())}")
            
            for field_name, config in self.FIELD_CONFIGS.items():
                try:
                    # Debug: Check if processing this field
                    st.write(f"DEBUG: Processing field options for {field_name}")
                    
                    # Special handling for different data sources
                    if field_name == 'subgenres':
                        df = self.reference_data['subgenres']
                    elif field_name == 'team_members':
                        df = self.reference_data[field_name]
                    else:
                        if field_name not in self.reference_data:
                            st.write(f"DEBUG: Missing reference data for {field_name}")
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
    
    def get_validations(self) -> Dict[str, FieldValidation]:
        """Get validation rules for all fields.
        
        Returns:
            Dictionary mapping field names to FieldValidation objects
        """
        return self.FIELD_VALIDATIONS
    
    def validate_criteria(self, criteria: Dict[str, Any]) -> List[str]:
        """Validate criteria against field validation rules.
        
        Args:
            criteria: Dictionary of criteria to validate
            
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

    def calculate_confidence(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence levels for criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary with confidence information:
            - overall: Overall confidence level ('none', 'low', 'medium', 'high')
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
                
            return {
                'overall': confidence_level,
                'score': confidence_score,
                'sample_size': overall_sample,
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
    
    def normalize_criteria(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize criteria for consistent processing.
        
        This method ensures that:
        1. List values with single items are converted to scalar values for scalar fields
        2. Numeric IDs like genre are properly typed as integers
        
        Args:
            criteria: Dictionary of criteria to normalize
            
        Returns:
            Normalized criteria dictionary
        """
        if not criteria:
            return {}
            
        normalized_criteria = {}
        
        for key, value in criteria.items():
            # Skip None values
            if value is None:
                continue
                
            # Convert single-item lists to scalar values for scalar fields
            if isinstance(value, list) and len(value) == 1 and self.get_field_type(key) == 'scalar':
                normalized_criteria[key] = value[0]
            else:
                normalized_criteria[key] = value
            
            # Ensure numeric IDs are integers
            if key == 'genre' and not isinstance(normalized_criteria[key], int):
                try:
                    normalized_criteria[key] = int(normalized_criteria[key])
                except (ValueError, TypeError):
                    # If conversion fails, keep the original value
                    pass
        
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
        
        This method checks multiple possible column names and returns the first one
        that exists in the provided data_columns list.
        
        Args:
            field_name: Field name to get column name for
            data_columns: List of available columns in the DataFrame
            
        Returns:
            Column name if found, None otherwise
        """
        try:
            # First try the mapped name
            mapped_name = self.map_field_name(field_name)
            if mapped_name in data_columns:
                return mapped_name
                
            # Then try the original name
            if field_name in data_columns:
                return field_name
                
            # For array fields, try with _ids suffix
            if self.get_field_type(field_name) == 'array' and f"{field_name}_ids" in data_columns:
                return f"{field_name}_ids"
                
            # If we get here, the field doesn't exist in the data
            return None
        except Exception as e:
            st.error(f"Error getting column name for field '{field_name}': {str(e)}")
            # Return None as fallback
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
    
    def classify_criteria_by_importance(self, criteria: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Classify criteria by their importance levels.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary with keys 'essential', 'core', 'primary', 'secondary',
            each containing a dictionary of criteria at that importance level
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
                if data_columns is None or mapped_name in data_columns:
                    return mapped_name
            
            # For scalar fields, try with _id suffix which is common in the database
            if field_name not in array_field_mapping:
                field_id = f"{field_name}_id"
                if data_columns is None or field_id in data_columns:
                    return field_id
            
            # If no special mapping or the mapped field doesn't exist in data_columns,
            # return the original field name
            return field_name
        except Exception as e:
            st.error(f"Error mapping field name '{field_name}': {str(e)}")
            # Return the original field name as fallback
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
