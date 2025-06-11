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
import numpy as np
import streamlit as st

from .optimizer_config import OptimizerConfig


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
        
    def _get_data(self, data=None):
        """Get data for matching, falling back to cached criteria data if None.
        
        Args:
            data: DataFrame of shows to match against (uses cached data if None)
            
        Returns:
            DataFrame of shows, or empty DataFrame if no data available
        """
        if data is None:
            if self._criteria_data is None:
                st.error("No criteria data available and none provided")
                return pd.DataFrame()
            return self._criteria_data
        return data
        
    def _empty_confidence_info(self) -> Dict[str, Any]:
        """Create an empty confidence info dictionary.
        
        Returns:
            Empty confidence info dictionary
        """
        return {
            'level': 'none',
            'sample_size': 0,
            'match_quality': 0.0,
            'match_level': 0
        }
        
    def _get_match_level_description(self, match_level: int) -> str:
        """Generate a human-readable description for a match level.
        
        Args:
            match_level: The match level (1 = exact match, 2 = missing 1 criterion, etc.)
            
        Returns:
            A human-readable description of the match level
        """
        # Match level directly corresponds to criteria differences + 1
        # Level 1 = 0 differences, Level 2 = 1 difference, etc.
        diff = match_level - 1
        
        if diff == 0:
            return "All criteria matched"
        elif diff == 1:
            return f"Missing {diff} criterion"
        else:
            return f"Missing {diff} criteria"
        
    def set_criteria_data(self, criteria_data: pd.DataFrame):
        """Set the criteria data for matching.
        
        Args:
            criteria_data: DataFrame of shows with criteria data
        """
        self._criteria_data = criteria_data.copy() if criteria_data is not None else None
    
    def find_matches(self, criteria: Dict[str, Any], data: pd.DataFrame = None, 
                     min_sample_size: int = None, flexible: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
        
        # Use helper method to get data
        data = self._get_data(data)
        if data.empty:
            return pd.DataFrame(), self._empty_confidence_info()
        
        # We'll collect matches from all levels
        all_matches_by_level = {}
        match_counts = {}
        best_level = 0
        confidence_info = {}
        
        # Determine how many criteria we have to work with
        total_criteria = len(criteria)
        
        # Define the maximum number of criteria we're willing to drop
        # This is based on the total number of criteria
        max_criteria_to_drop = total_criteria - 1  # Keep at least 1 criterion
        
        if flexible:
            # For flexible matching, start with a higher level (more missing criteria)
            # but don't go beyond our max_criteria_to_drop
            start_level = min(total_criteria, 5)  # Default to level 5 if possible
            levels_to_try = [start_level]
        else:
            # Try each possible match level in order, from exact match to progressively fewer criteria
            # Level 1 = exact match, Level 2 = missing 1 criterion, etc.
            levels_to_try = list(range(1, max_criteria_to_drop + 2))
            
        for level in levels_to_try:
            try:
                # Get criteria for this match level
                level_criteria = self.get_criteria_for_match_level(criteria, level)
                
                # Skip if we have no criteria at this level
                if not level_criteria:
                    # Debug output removed: No criteria for match level
                    continue
                
                # Match shows using the level-specific criteria
                matched_shows, match_count = self._match_shows(level_criteria, data)
                
                # Ensure the match level exists in OptimizerConfig.MATCH_LEVELS
                OptimizerConfig.ensure_match_level_exists(level)
                
                # Generate level name dynamically based on criteria difference
                diff = level - 1  # Level 1 = 0 differences, Level 2 = 1 difference, etc.
                level_name = f"Missing {diff} criteria" if diff > 0 else "All criteria matched"
                # Debug output removed: Match level found shows
                
                # Store these matches with their level
                if not matched_shows.empty:
                    # Add a match_level column to identify the source level
                    matched_shows['match_level'] = level
                    all_matches_by_level[level] = matched_shows
                    match_counts[level] = match_count
                    
                    # If this is the first level with matches, or it has enough matches, use it
                    if best_level == 0 or match_count >= min_sample_size:
                        best_level = level
                        confidence_info = self.calculate_match_confidence(matched_shows, level, criteria)
                        
                        # If we have enough matches, stop searching
                        if match_count >= min_sample_size:
                            break
            except Exception as e:
                st.error(f"Error in match level {level}: {e}")
        
        # If we didn't find any matches at any level
        if not all_matches_by_level:
            st.error("No matches found at any level")
            return pd.DataFrame(), self._empty_confidence_info()
        
        # Combine shows from all match levels, starting with best match level
        result_shows = pd.DataFrame()
        for level in sorted(all_matches_by_level.keys()):
            if level in all_matches_by_level:
                if result_shows.empty:
                    result_shows = all_matches_by_level[level]
                else:
                    # Only add shows that aren't already included (avoid duplicates)
                    existing_titles = set(result_shows['title']) if 'title' in result_shows.columns else set()
                    new_shows = all_matches_by_level[level]
                    if 'title' in new_shows.columns:
                        new_shows = new_shows[~new_shows['title'].isin(existing_titles)]
                    result_shows = pd.concat([result_shows, new_shows], ignore_index=True)
        
        # Add match level counts to confidence info
        confidence_info['match_counts'] = match_counts
        
        # Ensure the match level exists in OptimizerConfig.MATCH_LEVELS
        OptimizerConfig.ensure_match_level_exists(best_level)
        
        # Generate level name dynamically using helper method
        confidence_info['match_level_name'] = self._get_match_level_description(best_level)
        confidence_info['confidence_level'] = confidence_info.get('level', 'none')  # Ensure confidence_level is set for fallback logic
        
        return result_shows, confidence_info
        
    def find_matches_with_fallback(self, criteria: Dict[str, Any], data: pd.DataFrame = None, min_sample_size: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced version of find_matches that incorporates sophisticated fallback logic.
        
        This method starts with exact matching (level 1) and then progressively relaxes
        criteria by removing one criterion at a time until it reaches the target sample size.
        The process:
        1. Start with exact matches (level 1 = all criteria match)
        2. If insufficient, try missing 1 criterion (level 2)
        3. If still insufficient, try missing 2 criteria (level 3)
        4. Continue relaxing criteria until we either:
           - Reach the target sample size (OptimizerConfig.MAX_RESULTS)
           - Have tried all possible match levels (down to 1 remaining criterion)
        
        Args:
            criteria: Dictionary of criteria to match against
            data: DataFrame of shows to match against (uses cached data if None)
            min_sample_size: Minimum number of matches required
            
        Returns:
            Tuple of (matching_shows, match_info) with combined matches from all levels needed
        """
        # Use config values for sample sizes if not specified
        if min_sample_size is None:
            min_sample_size = OptimizerConfig.CONFIDENCE['minimum_sample']
        
        target_sample_size = OptimizerConfig.MAX_RESULTS
        
        # Initialize result variables
        all_matches = pd.DataFrame()
        best_confidence_info = {}
        all_match_counts = {}
        unique_titles = set()
        total_unique_matches = 0
        
        # Get data for matching
        data = self._get_data(data)
        if data.empty:
            return pd.DataFrame(), self._empty_confidence_info()
        
        # Determine how many criteria we have to work with
        total_criteria = len(criteria)
        
        # Define the maximum number of criteria we're willing to drop
        # This is based on the total number of criteria but with a reasonable limit
        max_possible_drop = total_criteria - 1  # Maximum possible criteria to drop
        
        # For high-quality matches, don't drop more than 50% of criteria or 5 criteria, whichever is less
        # This ensures we get relevant matches that still resemble the original concept
        high_quality_max_drop = min(max_possible_drop, min(int(total_criteria * 0.5), 5))
        
        # For additional matches, we can be more permissive but still keep at least 1 criterion
        # This allows us to find more matches while still having some relevance
        max_criteria_to_drop = max_possible_drop
        
        # Ensure we always keep at least 2 criteria for high-quality matches if possible
        if total_criteria > 2 and high_quality_max_drop > total_criteria - 2:
            # Get criteria for this match level
            level_criteria = self.get_criteria_for_match_level(criteria, level)
            if not level_criteria:
                continue
                
            # Match shows using the level-specific criteria
            level_matches, match_count = self._match_shows(level_criteria, data)
            
            # Log attempt at each level
            level_desc = self._get_match_level_description(level)
            if level_matches.empty:
                st.write(f"Tried level {level} ({level_desc}) - No matches found")
                continue
            else:
                st.write(f"Tried level {level} ({level_desc}) - Found {match_count} matches")
            
            # Add match_level to the matches
            level_matches['match_level'] = level
            
            # Calculate match quality as a percentage based on criteria retained
            match_quality = round(((total_criteria - (level - 1)) / total_criteria) * 100)
            level_matches['match_quality'] = match_quality
            
            # Add description of the match level
            level_matches['match_level_desc'] = self._get_match_level_description(level)
                
                # Calculate match quality percentage based on criteria retained
                criteria_retained = total_criteria - (level - 1)
                match_quality_pct = round((criteria_retained / total_criteria) * 100)
                
                st.write(f"Found {new_unique_count} new matches at level {level} ({level_desc}). Match quality: {match_quality_pct}%. Total unique matches: {total_unique_matches}")
            
            # If we've reached the target sample size, we can stop
            if total_unique_matches >= target_sample_size:
                break
        
        # If we still didn't find any matches at any level
        if all_matches.empty:
            return pd.DataFrame(), self._empty_confidence_info()
        
        # Prepare confidence info for the combined results
        confidence_info = best_confidence_info.copy() if best_confidence_info else self._empty_confidence_info()
        confidence_info['match_counts_by_level'] = all_match_counts
        confidence_info['total_unique_matches'] = total_unique_matches
        
        # Add a summary of the match levels we tried and how many matches we found at each level
        level_summaries = []
        for level, count in all_match_counts.items():
            level_desc = self._get_match_level_description(level)
            level_summaries.append(f"{level_desc}: {count} matches")
        
        confidence_info['match_level_summary'] = level_summaries
        
        # Sort by match_level (ascending) and success_score (descending)
        if 'success_score' in all_matches.columns:
            all_matches = all_matches.sort_values(by=['match_level', 'success_score'], 
                                                 ascending=[True, False])
        else:
            all_matches = all_matches.sort_values(by=['match_level'], ascending=[True])
        
        # Always limit to MAX_RESULTS
        if len(all_matches) > OptimizerConfig.MAX_RESULTS:
            all_matches = all_matches.head(OptimizerConfig.MAX_RESULTS)
        
        return all_matches, confidence_info
    
    def find_network_matches(self, criteria: Dict[str, Any], data: pd.DataFrame = None, matching_shows: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Find shows matching criteria for each available network.
        
        This method:
        1. Gets all unique networks from the data
        2. For each network, adds the network ID to the criteria
        3. Uses flexible matching to find shows matching both criteria and network
        4. Returns match results for each network with confidence information
        
        Args:
            criteria: Base criteria to match against
            data: DataFrame of shows to match against (uses cached data if None)
            matching_shows: Pre-filtered DataFrame of shows that match the criteria (optional)
            
        Returns:
            List of dictionaries with network information and matching results
        """
        # Use helper method to get data
        data = self._get_data(data)
        if data.empty:
            return []
            
        # If matching_shows is provided, use it to filter the data
        if matching_shows is not None and not matching_shows.empty:
            # Debug output removed: Using pre-filtered set of matching shows
            pass  # We'll use matching_shows to filter network-specific matches later
        
        # Extract all unique networks from the data
        try:
            networks = data[['network_id', 'network_name']].drop_duplicates().dropna()
            
            # Prepare results list
            results = []
            
            # For each network, find matching shows
            for _, network in networks.iterrows():
                network_id = network['network_id']
                network_name = network['network_name']
                
                # Skip if network_id is not valid
                if pd.isna(network_id) or pd.isna(network_name):
                    continue
                    
                # Create network-specific criteria
                network_criteria = criteria.copy()
                network_criteria['network'] = int(network_id)
                
                # Use flexible matching to get best possible results
                try:
                    # If we have pre-filtered matching_shows, use them to filter network-specific matches
                    if matching_shows is not None and not matching_shows.empty:
                        # Filter the matching_shows by this network
                        network_matching_shows = matching_shows[matching_shows['network_id'] == network_id].copy()
                        
                        if not network_matching_shows.empty:
                            # We already have matching shows for this network, use them directly
                            matching_shows_for_network = network_matching_shows
                            confidence_info = {
                                'match_level': 1,  # Exact match since we're using pre-filtered shows
                                'sample_size': len(matching_shows_for_network),
                                'match_quality': 1.0,  # Perfect match quality
                                'criteria_coverage': 1.0  # All criteria covered
                            }
                        else:
                            # No matching shows for this network in our pre-filtered set
                            matching_shows_for_network = pd.DataFrame()
                            confidence_info = self._empty_confidence_info()
                            confidence_info['match_level'] = 0
                    else:
                        # No pre-filtered shows, use regular matching
                        matching_shows_for_network, confidence_info = self.find_matches(
                            network_criteria, 
                            data=data, 
                            min_sample_size=OptimizerConfig.CONFIDENCE['minimum_sample']
                        )
                    
                    # Store the results
                    results.append({
                        'network_id': int(network_id),
                        'network_name': network_name,
                        'matching_shows': matching_shows_for_network,
                        'sample_size': confidence_info.get('sample_size', 0),
                        'confidence_info': confidence_info,
                        'match_quality': confidence_info.get('match_quality', 0.0)
                    })
                    
                except Exception as e:
                    st.error(f"Error matching network {network_name} (ID: {network_id}): {str(e)}")
                    # Add empty result to maintain network in results
                    empty_confidence = self._empty_confidence_info()
                    empty_confidence['match_level'] = 0  # Add match_level for network results
                    
                    results.append({
                        'network_id': int(network_id),
                        'network_name': network_name,
                        'matching_shows': pd.DataFrame(),
                        'sample_size': 0,
                        'confidence_info': empty_confidence,
                        'match_quality': 0.0
                    })
            
            return results
            
        except Exception as e:
            st.error(f"Error in network matching: {str(e)}")
            return []
    
    def _match_shows(self, criteria: Dict[str, Any], data: pd.DataFrame = None) -> Tuple[pd.DataFrame, int]:
        """Match shows based on criteria.
        
        Args:
            criteria: Dictionary of criteria
            data: DataFrame of shows to match against
            
        Returns:
            Tuple of (matching_shows, match_count)
        """
        # Use helper method to get data
        data = self._get_data(data)
        if data.empty:
            st.error("Empty criteria data provided")
            return pd.DataFrame(), 0
            
        # Clean up criteria - remove None or empty values to make matching more lenient
        clean_criteria = {}
        
        # Get array fields and mapping from field_manager
        array_field_mapping = self.field_manager.get_array_field_mapping()
        array_fields = list(array_field_mapping.keys())
        
        for field_name, value in criteria.items():
            # Skip empty criteria
            if value is None or (isinstance(value, list) and not value):
                continue
                
            # Use field_manager to determine the field type
            field_type = self.field_manager.get_field_type(field_name)
            
            # Handle array fields
            if field_type == 'array':
                # Make sure array field values are always lists
                if not isinstance(value, list):
                    clean_criteria[field_name] = [value]
                else:
                    clean_criteria[field_name] = value
            else:
                # Don't map field names here - let FieldManager handle it
                clean_criteria[field_name] = value
        
        # If we have no valid criteria after cleaning, return all shows
        if not clean_criteria:
            return data, len(data)
            
        try:
            # Start with all shows
            matches = data.copy()
            
            # Separate array fields from scalar fields
            scalar_fields = {}
            array_fields_to_filter = {}
            
            for field_name, value in clean_criteria.items():
                # Use field_manager to determine the field type
                field_type = self.field_manager.get_field_type(field_name)
                
                if field_type == 'array':
                    array_fields_to_filter[field_name] = value
                else:
                    # For scalar fields, determine the actual column name using field_manager
                    field_id = self.field_manager.map_field_name(field_name, matches.columns)
                    scalar_fields[field_id] = value
            
            # Process array fields (these require apply functions)
            for field_name, value in array_fields_to_filter.items():
                # Use field_manager to get the correct column name
                field_column = self.field_manager.get_field_column_name(field_name, matches.columns)
                
                # If column doesn't exist, skip this field
                if field_column is None:
                    st.error(f"Field '{field_name}' not found in data columns")
                    continue
                    
                # Check if the column contains lists or is itself a list
                sample = matches[field_column].iloc[0] if not matches.empty else None
                
                # Convert value to set for faster lookups
                value_set = set(value)  
                
                # Handle different data formats
                if isinstance(sample, list):
                    # If the column contains lists, use list intersection
                    mask = matches[field_column].apply(
                        lambda x: isinstance(x, list) and bool(value_set.intersection(x)))
                else:
                    # If the column isn't storing lists, use standard filtering
                    mask = matches[field_column].isin(value)
                    
                # Apply filter
                matches = matches[mask]
            
            # Process scalar fields (these can use vectorized operations)
            for field_id, value in scalar_fields.items():
                # Check if field exists in data
                if field_id not in matches.columns:
                    st.error(f"Field '{field_id}' not found in data columns")
                    continue
                    
                if isinstance(value, list):
                    # Multiple values: any show with any of the values matches
                    mask = matches[field_id].isin(value)
                else:
                    # Single value: exact match
                    mask = matches[field_id] == value
                    
                # Apply filter
                matches = matches[mask]
                
            # Matching complete
            return matches, len(matches)
        except Exception as e:
            st.error(f"Error matching shows: {e}")
            return pd.DataFrame(), 0
    
    def get_criteria_for_match_level(self, criteria: Dict[str, Any], match_level: int) -> Dict[str, Any]:
        """Get criteria adjusted for a specific match level.
        
        Match levels now directly correspond to the number of criteria differences:
        - Level 1: All criteria match (0 differences)
        - Level 2: Missing 1 criterion
        - Level 3: Missing 2 criteria
        - Level 4: Missing 3 criteria
        - And so on...
        
        This implementation dynamically calculates which criteria to include based on
        the exact number of criteria differences specified by the match level.
        
        Args:
            criteria: Dictionary of criteria
            match_level: Match level (corresponds to criteria differences + 1)
            
        Returns:
            Criteria dictionary adjusted for the match level
        """
        # Special case for only 1 criterion - always include it
        if len(criteria) == 1:
            return criteria.copy()
            
        # If match level is 1, use all criteria (exact match)
        if match_level == 1:
            return criteria.copy()
            
        # Calculate criteria difference based on match level
        # Match level 1 = 0 differences, match level 2 = 1 difference, etc.
        criteria_diff = match_level - 1
        
        # Special case for extreme fallback (very few criteria)
        if criteria_diff >= len(criteria):
            # Keep at least one criterion (most important one)
            result = {}
            
            # Always include genre if it exists
            if 'genre' in criteria:
                result['genre'] = criteria['genre']
                return result
                
            # Classify criteria by importance
            classified = self.field_manager.classify_criteria_by_importance(criteria)
            
            # Try to find at least one criterion to keep, in order of importance
            if classified['essential']:
                field, value = next(iter(classified['essential'].items()))
                result[field] = value
                return result
            elif classified['core']:
                field, value = next(iter(classified['core'].items()))
                result[field] = value
                return result
            elif classified['primary']:
                field, value = next(iter(classified['primary'].items()))
                result[field] = value
                return result
            elif classified['secondary']:
                field, value = next(iter(classified['secondary'].items()))
                result[field] = value
                return result
            else:
                # Last resort: take the first criterion
                field, value = next(iter(criteria.items()))
                result[field] = value
                return result
        
        # Get the total number of criteria
        total_criteria = len(criteria)
        
        # Calculate how many criteria to include
        criteria_to_include = max(1, total_criteria - criteria_diff)
        
        # Classify criteria by importance
        classified = self.field_manager.classify_criteria_by_importance(criteria)
        
        # Build result by adding criteria in order of importance until we reach the target count
        result = {}
        remaining_slots = criteria_to_include
        
        # Always include essential criteria first
        result.update(classified['essential'])
        remaining_slots -= len(classified['essential'])
        
        # Add core criteria if we have slots left
        if remaining_slots > 0 and classified['core']:
            core_to_add = min(len(classified['core']), remaining_slots)
            core_items = list(classified['core'].items())[:core_to_add]
            for field, value in core_items:
                result[field] = value
            remaining_slots -= core_to_add
        
        # Add primary criteria if we have slots left
        if remaining_slots > 0 and classified['primary']:
            primary_to_add = min(len(classified['primary']), remaining_slots)
            primary_items = list(classified['primary'].items())[:primary_to_add]
            for field, value in primary_items:
                result[field] = value
            remaining_slots -= primary_to_add
        
        # Add secondary criteria if we have slots left
        if remaining_slots > 0 and classified['secondary']:
            secondary_to_add = min(len(classified['secondary']), remaining_slots)
            secondary_items = list(classified['secondary'].items())[:secondary_to_add]
            for field, value in secondary_items:
                result[field] = value
        
        # Always prioritize certain critical fields for better matching quality
        # Genre is especially important for show matching
        if 'genre' in criteria and 'genre' not in result:
            result['genre'] = criteria['genre']
            
        # Network is also important for show matching if it exists
        if 'network' in criteria and 'network' not in result and len(result) < criteria_to_include + 1:
            result['network'] = criteria['network']
            
        # Format/type is important for show matching if it exists
        if 'format' in criteria and 'format' not in result and len(result) < criteria_to_include + 1:
            result['format'] = criteria['format']
            
        return result
    
    def calculate_match_confidence(self, shows: pd.DataFrame, match_level: int, 
                                  criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics for a set of matched shows.
        
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
            - match_level: The actual match level based on criteria validation
        """
        # Get sample size
        sample_size = len(shows)
        
        # Calculate criteria coverage
        total_criteria = len(OptimizerConfig.CRITERIA_IMPORTANCE)
        criteria_count = len(criteria)
        
        # Calculate match quality based on criteria difference
        # Level 1 = 0 differences, Level 2 = 1 difference, etc.
        criteria_diff = match_level - 1
        
        # Use the OptimizerConfig helper method to get quality score
        match_quality = OptimizerConfig.get_quality_for_diff(criteria_diff)
        
        # Calculate confidence score using OptimizerConfig
        confidence_score = OptimizerConfig.calculate_confidence_score(
            sample_size, criteria_count, total_criteria, match_level)
        
        # Get confidence level string
        confidence_level = OptimizerConfig.get_confidence_level(sample_size, match_level)
        
        # Validate the actual match level by checking if all criteria are truly matched
        # This is especially important for array fields like character_types
        actual_match_level = match_level
        
        # Only perform validation if we have shows and claiming exact match (level 1)
        if not shows.empty and match_level == 1:
            # Get array field mapping to check array fields properly
            array_field_mapping = self.field_manager.get_array_field_mapping()
            
            # Check each criterion to see if it's actually matched
            for field_name, value in criteria.items():
                # Skip empty criteria
                if value is None or (isinstance(value, list) and not value):
                    continue
                    
                # Handle array fields differently
                if isinstance(value, list):
                    # Use field_manager to get the correct column name
                    field_column = self.field_manager.get_field_column_name(field_name, shows.columns)
                        
                    # Check if this column exists in the data
                    if field_column not in shows.columns:
                        st.error(f"Field '{field_column}' not found in shows data")
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
                        # Debug output removed: Not all shows match array criterion
                        actual_match_level = 2  # Downgrade to level 2
                        break
                else:  # Handle scalar fields
                    # Use field_manager to determine the actual column name
                    field_id = self.field_manager.map_field_name(field_name, shows.columns)
                        
                    # Check if this column exists in the data
                    if field_id not in shows.columns:
                        st.error(f"Field '{field_id}' not found in shows data")
                        actual_match_level = 2  # Downgrade to level 2 if field is missing
                        continue
                            
                    # Check if all shows match this scalar criterion
                    if not (shows[field_id] == value).all():
                        # Debug output removed: Not all shows match scalar criterion
                        actual_match_level = 2  # Downgrade to level 2
                        break
        
        return {
            'level': confidence_level,
            'score': confidence_score,
            'match_quality': match_quality,
            'sample_size': sample_size,
            'match_level': actual_match_level,  # Use the validated match level
            'original_match_level': match_level,  # Keep track of the original level
            # Generate level name dynamically based on criteria difference
            'match_level_name': self._get_match_level_description(actual_match_level)
        }
        
    def _get_relaxed_criteria(self, criteria: Dict[str, Any], relaxation_tier: str) -> List[Dict[str, Any]]:
        """Generate sets of relaxed criteria by removing criteria of the specified importance tier.
        
        Args:
            criteria: Dictionary of criteria to relax
            relaxation_tier: Importance tier to relax ('secondary', 'primary', 'core')
                
        Returns:
            List of dictionaries with relaxed criteria sets
        """
        # Find criteria that belong to the specified relaxation tier
        relaxable_criteria = []
        for criteria_type, criteria_value in criteria.items():
            # Skip if criteria type is not in the importance mapping
            if criteria_type not in OptimizerConfig.CRITERIA_IMPORTANCE:
                continue
                    
            # Get the importance level of this criterion
            importance = OptimizerConfig.CRITERIA_IMPORTANCE[criteria_type]
            
            # Add to relaxable list if it matches the tier and is not essential
            if importance == relaxation_tier and importance != 'essential':
                relaxable_criteria.append(criteria_type)
        
        # If no relaxable criteria found, return empty list
        if not relaxable_criteria:
            return []
                
        # Limit the number of criteria to relax based on config
        max_to_relax = min(len(relaxable_criteria), 
                          OptimizerConfig.FALLBACK_SYSTEM['relaxation']['max_criteria_to_relax'])
        
        # Generate relaxed criteria sets by removing one criterion at a time
        relaxed_criteria_sets = []
        
        # First try relaxing individual criteria
        for criteria_type in relaxable_criteria:
            # Create a copy without this criterion
            relaxed = {k: v for k, v in criteria.items() if k != criteria_type}
            relaxed_criteria_sets.append({
                'criteria': relaxed,
                'relaxed_type': criteria_type,
                'relaxed_value': criteria[criteria_type],
                'relaxed_name': self._get_criteria_name(criteria_type, criteria[criteria_type])
            })
            
        # If allowed to relax more than one criterion and we have enough criteria,
        # try combinations of two criteria to relax
        if max_to_relax >= 2 and len(relaxable_criteria) >= 2:
            for i, type1 in enumerate(relaxable_criteria):
                for type2 in relaxable_criteria[i+1:]:
                    # Create a copy without these two criteria
                    relaxed = {k: v for k, v in criteria.items() if k != type1 and k != type2}
                    relaxed_criteria_sets.append({
                        'criteria': relaxed,
                        'relaxed_type': f"{type1},{type2}",
                        'relaxed_value': f"{criteria[type1]},{criteria[type2]}",
                        'relaxed_name': f"{self._get_criteria_name(type1, criteria[type1])} and "
                                      f"{self._get_criteria_name(type2, criteria[type2])}"
                    })
        
        return relaxed_criteria_sets
        
    def _calculate_relevance_score(self, original_criteria: Dict[str, Any], matched_show: pd.Series) -> float:
        """Calculate a weighted relevance score for a show based on how well it matches the original criteria.
        
        Args:
            original_criteria: Dictionary of original criteria
            matched_show: Series representing a show from the database
                
        Returns:
            Relevance score between 0 and 1
        """
        total_weight = 0
        weighted_score = 0
        
        # Get the weight multipliers from config
        weight_multipliers = OptimizerConfig.FALLBACK_SYSTEM['relevance']['weight_multipliers']
        
        # For each criterion in the original criteria
        for criteria_type, criteria_value in original_criteria.items():
            # Skip if criteria type is not in the importance mapping
            if criteria_type not in OptimizerConfig.CRITERIA_IMPORTANCE:
                continue
                    
            # Get the importance level and base weight for this criterion
            importance = OptimizerConfig.CRITERIA_IMPORTANCE[criteria_type]
            base_weight = OptimizerConfig.IMPORTANCE_WEIGHTS.get(importance, 1.0)
            
            # Apply the weight multiplier based on importance
            weight = base_weight * weight_multipliers.get(importance, 1.0)
            total_weight += weight
            
            # Check if the show matches this criterion
            if criteria_type in matched_show:
                # For list-type fields (e.g., genres, plot_elements)
                if isinstance(matched_show[criteria_type], list) or isinstance(matched_show[criteria_type], np.ndarray):
                    if criteria_value in matched_show[criteria_type]:
                        weighted_score += weight
                # For scalar fields
                elif matched_show[criteria_type] == criteria_value:
                    weighted_score += weight
        
        # Calculate final relevance score
        if total_weight > 0:
            return weighted_score / total_weight
        else:
            return 0.0
            
    def _get_criteria_name(self, criteria_type: str, criteria_value: Any) -> str:
        """Get the display name for a criteria value.
        
        Args:
            criteria_type: Type of criteria
            criteria_value: Value of criteria
                
        Returns:
            Display name for the criteria value
        """
        # Try to get options from field manager
        try:
            options = self.field_manager.get_options(criteria_type)
            
            # Find the option with this ID
            for option in options:
                if option.get('id') == criteria_value or str(option.get('id')) == str(criteria_value):
                    return option.get('name', str(criteria_value))
        except Exception:
            pass
                
        # Default if not found
        return f"{criteria_type}={criteria_value}"