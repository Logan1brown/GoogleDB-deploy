"""Matcher component for Show Optimizer.

This module handles all show matching logic based on criteria, implementing
flexible matching across different match levels with confidence calculation.

Key responsibilities:
- Search for shows matching criteria with different levels of flexibility
- Progressively relax criteria when needed to find sufficient matches
- Track the quality/confidence of different match types
- Return a prioritized list of matches for scoring and analysis
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import streamlit as st
import logging

from .optimizer_config import OptimizerConfig

# Configure logger but use st.write for user-facing messages
logger = logging.getLogger(__name__)


class Matcher:
    """Handles show matching with flexible match levels and confidence calculation.
    
    This class provides a unified interface for searching shows based on criteria,
    with support for flexible matching, fallbacks, and confidence metrics.
    """
    
    def __init__(self, field_manager):
        """Initialize the Matcher.
        
        Args:
            field_manager: FieldManager instance for field mapping and validation
        """
        self.field_manager = field_manager
        self._criteria_data = None  # Cache for criteria data
        
    def set_criteria_data(self, criteria_data: pd.DataFrame):
        """Set the criteria data for matching.
        
        Args:
            criteria_data: DataFrame of shows with criteria data
        """
        self._criteria_data = criteria_data.copy() if criteria_data is not None else None
    
    def find_matches(self, criteria: Dict[str, Any], data: pd.DataFrame = None, 
                     min_sample_size: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Main entry point for finding matches with automatic fallback strategies.
        
        This method orchestrates the search process, trying different strategies
        in order of decreasing precision until sufficient matches are found.
        
        Args:
            criteria: Dictionary of criteria to match against
            data: DataFrame of shows to match against (uses cached data if None)
            min_sample_size: Minimum number of matches required (defaults to OptimizerConfig.CONFIDENCE['minimum_sample'])
            
        Returns:
            Tuple of (matching_shows, match_info)
        """
        # Use config for minimum sample size if not specified
        if min_sample_size is None:
            min_sample_size = OptimizerConfig.CONFIDENCE['minimum_sample']
        
        # Use cached data if none provided
        if data is None:
            if self._criteria_data is None:
                st.error("No criteria data available and none provided")
                return pd.DataFrame(), {'level': 'none', 'score': 0, 'match_quality': 0, 'sample_size': 0}
            data = self._criteria_data
        
        # We'll collect matches from all levels
        all_matches_by_level = {}
        match_counts = {}
        best_level = 0
        confidence_info = {}
        
        # Try each match level defined in OptimizerConfig
        for level in sorted(OptimizerConfig.MATCH_LEVELS.keys()):
            try:
                # Get criteria for this match level
                level_criteria = self.get_criteria_for_match_level(criteria, level)
                
                # Skip if we have no criteria at this level
                if not level_criteria:
                    st.write(f"No criteria for match level {level}, skipping")
                    continue
                
                # Match shows using the level-specific criteria
                matched_shows, match_count = self._match_shows(level_criteria, data)
                level_name = OptimizerConfig.MATCH_LEVELS[level]['name']
                st.write(f"Match level {level} ({level_name}) found {match_count} shows")
                
                # Store these matches with their level
                if not matched_shows.empty:
                    # Add a match_level column to identify the source level
                    matched_shows['match_level'] = level
                    all_matches_by_level[level] = matched_shows
                    match_counts[level] = match_count
                    
                    # If this is the first level with matches, or it has enough matches, use it
                    if best_level == 0 or match_count >= min_sample_size:
                        best_level = level
                        
                        # If we have enough matches, we can stop
                        if match_count >= min_sample_size:
                            break
            except Exception as e:
                st.error(f"Error matching at level {level}: {str(e)}")
        
        # If we didn't find any matches, return empty DataFrame
        if best_level == 0:
            st.warning("No matches found at any level")
            return pd.DataFrame(), {'level': 'none', 'score': 0, 'match_quality': 0, 'sample_size': 0}
        
        # Get the best matches
        best_matches = all_matches_by_level[best_level]
        
        # Calculate confidence metrics
        confidence_info = self.calculate_match_confidence(
            best_matches, best_level, criteria)
            
        # Add match level counts to confidence info
        confidence_info['match_counts'] = match_counts
        
        return best_matches, confidence_info
    
    def get_criteria_for_match_level(self, criteria: Dict[str, Any], match_level: int) -> Dict[str, Any]:
        """Get a subset of criteria for a specific match level.
        
        Match levels:
        1 - All criteria (strict matching)
        2 - All but one secondary criterion
        3 - Core and primary criteria only
        4 - Only essential and core criteria
        
        Args:
            criteria: Dictionary of criteria
            match_level: Match level (1-4)
            
        Returns:
            Dictionary of criteria for the specified match level
        """
        # If match level is 1, use all criteria
        if match_level == 1:
            return criteria.copy()
            
        # Classify criteria by importance
        classified = self.field_manager.classify_criteria_by_importance(criteria)
        
        # For match level 4, use only essential and core criteria
        if match_level == 4:
            result = {}
            result.update(classified['essential'])
            result.update(classified['core'])
            return result
            
        # For match level 3, use essential, core, and primary criteria
        if match_level == 3:
            result = {}
            result.update(classified['essential'])
            result.update(classified['core'])
            result.update(classified['primary'])
            return result
            
        # For match level 2, use all criteria except one secondary criterion (if any)
        if match_level == 2 and classified['secondary']:
            result = {}
            result.update(classified['essential'])
            result.update(classified['core'])
            result.update(classified['primary'])
            
            # Add all but one secondary criterion
            secondary_items = list(classified['secondary'].items())
            for i, (field, value) in enumerate(secondary_items):
                if i < len(secondary_items) - 1:  # Skip the last one
                    result[field] = value
            return result
        
        # Default to all criteria if match level is invalid or no secondary criteria for level 2
        return criteria.copy()
    
    def _match_shows(self, criteria: Dict[str, Any], data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Match shows against criteria.
        
        Args:
            criteria: Dictionary of criteria
            data: DataFrame of shows to match against
            
        Returns:
            Tuple of (matched_shows, match_count)
        """
        if data.empty:
            return pd.DataFrame(), 0
            
        # Start with all shows
        matched = data.copy()
        
        # Get array fields and mapping from field_manager
        array_field_mapping = self.field_manager.get_array_field_mapping()
        array_fields = list(array_field_mapping.keys())
        
        # Clean up criteria - remove None or empty values
        clean_criteria = {}
        for field_name, value in criteria.items():
            # Skip None values and empty lists
            if value is None or (isinstance(value, list) and not value):
                continue
            clean_criteria[field_name] = value
        
        # Apply each criterion
        for field_name, value in clean_criteria.items():
            # Handle array fields (genres, keywords, etc.)
            if field_name in array_fields:
                # Get the correct column name for this field
                if field_name in array_field_mapping:
                    field_column = array_field_mapping[field_name]
                elif f"{field_name}_ids" in matched.columns:
                    field_column = f"{field_name}_ids"
                else:
                    field_column = field_name
                
                # Check if this column exists in the data
                if field_column not in matched.columns:
                    st.warning(f"Field '{field_column}' not found in shows data")
                    continue
                
                # Sample the first row to check data format
                sample = matched[field_column].iloc[0] if not matched.empty else None
                
                # Apply the filter based on data format
                if isinstance(sample, list):
                    # For list-type columns, check if any value matches
                    value_set = set(value) if isinstance(value, list) else {value}
                    matched = matched[matched[field_column].apply(
                        lambda x: isinstance(x, list) and bool(set(x).intersection(value_set))
                    )]
                else:
                    # For scalar columns, check if the value is in the list
                    if isinstance(value, list):
                        matched = matched[matched[field_column].isin(value)]
                    else:
                        matched = matched[matched[field_column] == value]
            
            # Handle scalar fields
            else:
                # Determine the actual column name
                field_id = self.field_manager.get_field_id(field_name)
                
                # Check if this column exists in the data
                if field_id not in matched.columns:
                    st.warning(f"Field '{field_id}' not found in shows data")
                    continue
                
                # Apply the filter
                if isinstance(value, list):
                    matched = matched[matched[field_id].isin(value)]
                else:
                    matched = matched[matched[field_id] == value]
        
        return matched, len(matched)
    
    def calculate_match_confidence(self, shows: pd.DataFrame, match_level: int, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics for a match result.
        
        Args:
            shows: DataFrame of matched shows
            match_level: Match level used (1-4)
            criteria: Original criteria dictionary
            
        Returns:
            Dictionary with confidence metrics:
            - level: Confidence level string ('none', 'very_low', 'low', 'medium', 'high')
            - score: Confidence score (0-1)
            - match_quality: Quality of the match based on match level (0-1)
            - sample_size: Number of shows in the sample
            - actual_match_level: The actual match level based on criteria validation
        """
        # Get sample size
        sample_size = len(shows)
        
        # Calculate criteria coverage
        total_criteria = len(OptimizerConfig.CRITERIA_IMPORTANCE)
        criteria_count = len(criteria)
        
        # Calculate match quality based on match level
        match_quality = OptimizerConfig.MATCH_LEVELS.get(match_level, {}).get('min_quality', 0.3)
        
        # Calculate confidence score using OptimizerConfig
        confidence_score = OptimizerConfig.calculate_confidence_score(
            sample_size, criteria_count, total_criteria, match_level)
        
        # Determine confidence level based on score
        if confidence_score < OptimizerConfig.CONFIDENCE['very_low_threshold']:
            confidence_level = 'none'
        elif confidence_score < OptimizerConfig.CONFIDENCE['low_threshold']:
            confidence_level = 'very_low'
        elif confidence_score < OptimizerConfig.CONFIDENCE['medium_threshold']:
            confidence_level = 'low'
        elif confidence_score < OptimizerConfig.CONFIDENCE['high_threshold']:
            confidence_level = 'medium'
        else:
            confidence_level = 'high'
        
        # Validate the actual match level by checking if all shows match all criteria
        actual_match_level = match_level
        
        # Only validate if we have shows and the match level is 1 (strict)
        if not shows.empty and match_level == 1:
            # Get array fields and mapping from field_manager
            array_field_mapping = self.field_manager.get_array_field_mapping()
            
            # Check each criterion
            for field_name, value in criteria.items():
                # Skip None values and empty lists
                if value is None or (isinstance(value, list) and not value):
                    continue
                    
                # Handle array fields differently
                if isinstance(value, list):
                    # Get the correct column name for this field
                    if field_name in array_field_mapping:
                        field_column = array_field_mapping[field_name]
                    elif f"{field_name}_ids" in shows.columns:
                        field_column = f"{field_name}_ids"
                    else:
                        field_column = field_name
                        
                    # Check if this column exists in the data
                    if field_column not in shows.columns:
                        st.warning(f"Field '{field_column}' not found in shows data")
                        actual_match_level = 2  # Downgrade to level 2 if field is missing
                        continue
                        
                    # Sample the first row to check data format
                    sample = shows[field_column].iloc[0] if not shows.empty else None
                    
                    # Check if all shows actually match this array criterion
                    value_set = set(value)
                    if isinstance(sample, list):
                        # For each show, check if any of the criteria values match
                        all_match = shows[field_column].apply(
                            lambda x: isinstance(x, list) and bool(value_set.intersection(x))
                        ).all()
                    else:
                        # If not stored as lists, check if any match the values
                        all_match = shows[field_column].isin(value).any(axis=0)
                        
                    # If not all shows match this criterion, downgrade the match level
                    if not all_match:
                        st.write(f"Not all shows match array criterion '{field_name}', downgrading match level")
                        actual_match_level = 2  # Downgrade to level 2
                        break
                        
                # Handle scalar fields
                else:
                    # Determine the actual column name
                    field_id = self.field_manager.get_field_id(field_name)
                    
                    # Check if this column exists in the data
                    if field_id not in shows.columns:
                        st.warning(f"Field '{field_id}' not found in shows data")
                        actual_match_level = 2  # Downgrade to level 2 if field is missing
                        continue
                        
                    # Check if all shows match this scalar criterion
                    if not (shows[field_id] == value).all():
                        st.write(f"Not all shows match scalar criterion '{field_name}', downgrading match level")
                        actual_match_level = 2  # Downgrade to level 2
                        break
        
        return {
            'level': confidence_level,
            'score': confidence_score,
            'match_quality': match_quality,
            'sample_size': sample_size,
            'match_level': actual_match_level,  # Use the validated match level
            'original_match_level': match_level,  # Keep track of the original level
            'match_level_name': OptimizerConfig.MATCH_LEVELS.get(actual_match_level, {}).get('name', f'Level {actual_match_level}')
        }
