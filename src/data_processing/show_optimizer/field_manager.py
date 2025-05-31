"""Field management for Show Optimizer.

This module handles field options, validation, and show matching for the Show Optimizer.
It builds on patterns from CompAnalyzer's FieldManager but adds:
- Progressive validation for partial criteria sets
- Incremental show matching for real-time feedback
- Sample size tracking for confidence indicators

Key components:
- FieldConfig: Configuration for a field (table, id field, name field)
- FieldOption: A single option for a field with ID and display name
- FieldValidation: Validation rules for a field
- FieldManager: Main class that manages field options and matching
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import pandas as pd
import numpy as np
import logging
from functools import lru_cache

from ..analyze_shows import ShowsAnalyzer
from .optimizer_config import OptimizerConfig

logger = logging.getLogger(__name__)


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
    """Manages field options, validation, and show matching for Show Optimizer."""
    
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
            reference_data: Dictionary of reference data from ShowsAnalyzer
        """
        self.reference_data = reference_data
        self.options: Dict[str, List[FieldOption]] = {}
        self.sample_sizes: Dict[str, Dict[int, int]] = {}  # Field -> Value -> Count
        self._load_options()
        self._calculate_sample_sizes()
        
    def _load_options(self):
        """Load all field options from reference data."""
        for field_name, config in self.FIELD_CONFIGS.items():
            # Special handling for different data sources
            if field_name == 'subgenres':
                df = self.reference_data['subgenres']
            elif field_name == 'team_members':
                df = self.reference_data[field_name]
            else:
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
    
    def _calculate_sample_sizes(self):
        """Calculate sample sizes for each field value.
        
        This is used to provide confidence indicators for criteria selection.
        """
        # Get the main shows data
        shows_data = self.reference_data.get('shows', pd.DataFrame())
        if shows_data.empty:
            logger.warning("No shows data available for sample size calculation")
            return
            
        # Calculate sample sizes for each field
        for field_name, config in self.FIELD_CONFIGS.items():
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
                value_counts = shows_data[field_id].value_counts().to_dict()
                for value, count in value_counts.items():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        self.sample_sizes[field_name][int(value)] = count
    
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
        for opt in self.get_options(field_name):
            if opt.id == id:
                return opt.name
        return f'Unknown ({id})'
    
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
                        errors.append(f"{field_name} is only valid when {dependent_field} is one of {validation.dependency_values}")
        
        return errors
    
    @lru_cache(maxsize=128)
    def get_sample_size(self, field_name: str, value_key: Any) -> int:
        """Get the sample size for a field value.
        
        Args:
            field_name: Name of the field
            value_key: Field value or tuple of values (for lists)
            
        Returns:
            Number of shows with the specified value(s)
        """
        if field_name not in self.sample_sizes:
            return 0
            
        # Convert list to tuple for caching (lists aren't hashable)
        if isinstance(value_key, list):
            value = tuple(value_key)
            # For multiple values, return the minimum sample size
            # This represents the most restrictive case
            sizes = [self.sample_sizes[field_name].get(v, 0) for v in value]
            return min(sizes) if sizes else 0
        else:
            return self.sample_sizes[field_name].get(value_key, 0)
    
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
                    'confidence': OptimizerConfig.get_confidence_level(sample_size)
                }
            
        # Estimate overall sample size (use the most restrictive field)
        # This is a conservative estimate
        overall_sample = min(sample_sizes) if sample_sizes else 0
        
        # Calculate overall confidence
        criteria_count = len(criteria)
        total_criteria = len(self.FIELD_CONFIGS)
        confidence_score = OptimizerConfig.calculate_confidence_score(
            overall_sample, criteria_count, total_criteria)
        
        # Map confidence score to level
        if confidence_score == 0:
            confidence_level = 'none'
        elif confidence_score < 0.3:
            confidence_level = 'low'
        elif confidence_score < 0.7:
            confidence_level = 'medium'
        else:
            confidence_level = 'high'
            
        return {
            'overall': confidence_level,
            'score': confidence_score,
            'sample_size': overall_sample,
            'fields': field_samples
        }
    
    def match_shows(self, criteria: Dict[str, Any], shows_data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Match shows against the given criteria.
        
        Args:
            criteria: Dictionary of criteria to match against
            shows_data: DataFrame of shows data
            
        Returns:
            Tuple of (matched_shows, sample_size) where:
            - matched_shows: DataFrame of shows matching the criteria
            - sample_size: Number of shows in the match
        """
        if shows_data.empty or not criteria:
            return shows_data, 0
            
        # Start with all shows
        matches = shows_data.copy()
        
        # Pre-process criteria to avoid repeated lookups
        array_fields = {}
        scalar_fields = {}
        
        for field_name, value in criteria.items():
            if field_name not in self.FIELD_CONFIGS:
                continue
                
            config = self.FIELD_CONFIGS[field_name]
            if config.is_array:
                array_fields[field_name] = value
            else:
                # For scalar fields, determine the actual column name
                field_id = f"{field_name}_id" if f"{field_name}_id" in matches.columns else field_name
                scalar_fields[field_id] = value
        
        # Process array fields (these require apply functions)
        for field_name, value in array_fields.items():
            if isinstance(value, list):
                # Multiple values: any show containing any of the values matches
                # Use vectorized operations where possible
                value_set = set(value)  # Convert to set for faster lookups
                mask = matches[field_name].apply(
                    lambda x: isinstance(x, list) and bool(value_set.intersection(x)))
            else:
                # Single value: any show containing the value matches
                mask = matches[field_name].apply(
                    lambda x: isinstance(x, list) and value in x)
                    
            # Apply filter
            matches = matches[mask]
        
        # Process scalar fields (these can use vectorized operations)
        for field_id, value in scalar_fields.items():
            if isinstance(value, list):
                # Multiple values: any show with any of the values matches
                mask = matches[field_id].isin(value)
            else:
                # Single value: exact match
                mask = matches[field_id] == value
                
            # Apply filter
            matches = matches[mask]
            
        return matches, len(matches)
