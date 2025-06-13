"""Matcher component for Show Optimizer.

This module handles all show matching logic based on criteria, implementing
flexible matching across different match levels with confidence calculation.

Key responsibilities:
- Search for shows matching criteria with different levels of flexibility
- Progressively relax criteria when needed to find sufficient matches
- Track the quality/confidence of different match types
- Return a prioritized list of matches for scoring and analysis
"""

from typing import Dict, List, Optional, Tuple, Any, Set
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
        
    def _get_data(self, data=None):
        """Get data for matching.
        
        Args:
            data: DataFrame of shows to match against
            
        Returns:
            DataFrame of shows, or empty DataFrame if no data available
        """
        if data is None or data.empty:
            return pd.DataFrame()
        return data
        
    def _empty_confidence_info(self) -> Dict[str, Any]:
        """Create an empty confidence info dictionary.
        
        Returns:
            Empty confidence info dictionary with default values
        """
        return {
            'level': 'none',  # Use string directly instead of non-existent CONFIDENCE_LEVELS
            'score': 0.0,
            'match_quality': 0.0,
            'sample_size': 0,
            'criteria_coverage': 0.0,
            'match_level': 0  # Maintain backward compatibility
        }
        
    def _calculate_criteria_coverage(self, criteria: Dict[str, Any], shows: pd.DataFrame) -> Tuple[float, int]:
        """Calculate criteria coverage and missing criteria count.
        
        Args:
            criteria: Dictionary of criteria to check coverage for
            shows: DataFrame of shows to check against
            
        Returns:
            Tuple of (criteria_coverage as float between 0 and 1, missing_criteria_count as int)
        """
        if not criteria or shows.empty:
            return 0.0, len(criteria)
            
        # Get total number of criteria
        total_criteria = len(criteria)
        if total_criteria == 0:
            return 1.0, 0  # No criteria means 100% coverage and 0 missing criteria
            
        # Count how many criteria are actually present in the shows
        matched_criteria = 0
        
        for key, value in criteria.items():
            # Skip empty or None values
            if value is None or (isinstance(value, (list, str)) and not value):
                continue
                
            # Check if this criterion is present in the shows
            if key in shows.columns:
                # For array fields, check if any show has the value
                if isinstance(value, list):
                    for item in value:
                        # For array columns in DataFrame, need special handling
                        if any(shows[key].apply(lambda x: item in x if isinstance(x, list) else False)):
                            matched_criteria += 1
                            break
                else:
                    # For scalar values, check if any show matches exactly
                    if any(shows[key] == value):
                        matched_criteria += 1
        
        # Calculate coverage ratio and missing criteria count
        coverage = matched_criteria / total_criteria if total_criteria > 0 else 1.0
        missing_criteria = total_criteria - matched_criteria
        
        return coverage, missing_criteria
        
    def _calculate_match_quality(self, missing_criteria_count: int) -> float:
        """Calculate match quality based on number of missing criteria.
        
        Args:
            missing_criteria_count: Number of criteria that weren't matched
            
        Returns:
            Match quality score (0-1)
        """
        # Use the config's programmatic approach to get quality based on missing criteria count
        # This is directly aligned with how OptimizerConfig.get_quality_for_diff works
        return OptimizerConfig.get_quality_for_diff(missing_criteria_count)
        
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
        
    def find_matches_with_fallback(self, criteria: Dict[str, Any], data: pd.DataFrame = None, min_sample_size: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Find shows matching criteria, with fallback to more permissive criteria if needed.
        
        This method will progressively relax the matching criteria until either:
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
        
        # Initialize result variables with proper columns
        all_matches = pd.DataFrame(columns=['match_level', 'match_quality', 'match_level_desc'])
        best_confidence_info = {}
        all_match_counts = {}
        unique_titles = set()
        total_unique_matches = 0
        
        # Get data for matching
        data = self._get_data(data)
        if data.empty:
            # Return an empty DataFrame with the required columns
            return pd.DataFrame(columns=['match_level', 'match_quality', 'match_level_desc', 'title']), self._empty_confidence_info()
        
        # Determine how many criteria we have to work with
        total_criteria = len(criteria)
        
        # Define the maximum number of criteria we're willing to drop
        # This is based on the total number of criteria but with a reasonable limit
        max_possible_drop = total_criteria - 1  # Maximum possible criteria to drop
        
        # Try each possible match level in order, from exact match to progressively fewer criteria
        for level in range(1, max_possible_drop + 2):
            # Get criteria for this match level
            level_criteria = self.get_criteria_for_match_level(criteria, level)
            if not level_criteria:
                continue
                
            # Match shows using the level-specific criteria
            level_matches, match_count = self._match_shows(level_criteria, data)
            
            # Skip if no matches at this level
            if level_matches.empty:
                continue
            
            # Always ensure match_level column exists
            level_matches['match_level'] = level
            all_match_counts[level] = match_count
            
            # Calculate match quality as a percentage based on criteria retained
            criteria_retained = total_criteria - (level - 1)
            match_quality_pct = round((criteria_retained / total_criteria) * 100)
            level_matches['match_quality'] = match_quality_pct
            
            # Add description of the match level
            level_desc = self._get_match_level_description(level)
            level_matches['match_level_desc'] = level_desc
            
            # Calculate confidence for this level if it's the first with matches
            if not best_confidence_info:
                best_confidence_info = self.calculate_match_confidence(level_matches, level, criteria)
            
            # Filter out shows we've already found at better match levels
            new_matches = level_matches[~level_matches['title'].isin(unique_titles)]
            new_unique_count = len(new_matches)
            
            # Process new matches if any were found
            
            # Add new matches to our results
            if all_matches.empty:
                all_matches = new_matches
            else:
                all_matches = pd.concat([all_matches, new_matches], ignore_index=True)
            
            # Update our list of unique titles
            unique_titles.update(new_matches['title'].tolist())
            total_unique_matches += new_unique_count
            
            # If we've found enough matches, stop looking
            if total_unique_matches >= target_sample_size:
                break
        
        # If we still didn't find any matches at any level
        if all_matches.empty:
            # Create an empty DataFrame with the required columns
            # Include all columns that will be used downstream
            empty_df = pd.DataFrame(columns=['match_level', 'match_quality', 'match_level_desc', 'title'])
            return empty_df, self._empty_confidence_info()
        
        # Prepare confidence info for the combined results
        confidence_info = best_confidence_info.copy() if best_confidence_info else self._empty_confidence_info()
        confidence_info['match_counts_by_level'] = all_match_counts
        confidence_info['total_unique_matches'] = total_unique_matches
        
        # Add a summary of the match levels we tried and how many matches we found at each level
        level_summaries = {}
        for level, count in all_match_counts.items():
            level_desc = self._get_match_level_description(level)
            level_summaries[level_desc] = count
        
        confidence_info['match_level_summary'] = level_summaries
        
        # Sort by match_level (ascending) - no success score sorting during selection
        if not all_matches.empty:
            # The match_level column is guaranteed to exist since we set it above
            all_matches = all_matches.sort_values(by=['match_level'], ascending=[True])
        
        # Apply prioritized sampling within each match level if we have more than MAX_RESULTS
        if len(all_matches) > OptimizerConfig.MAX_RESULTS:
            # Define a function to prioritize shows with RT and TMDB data
            def prioritize_shows(group_df):
                # Calculate the target sample size for this group
                # Use ceiling division to avoid getting fewer than MAX_RESULTS total shows
                target_size = min(len(group_df), max(1, int(np.ceil(OptimizerConfig.MAX_RESULTS * len(group_df) / len(all_matches)))))
                
                # Check if we have RT and TMDB columns to prioritize
                has_rt = 'rt_score' in group_df.columns
                has_tmdb = 'tmdb_score' in group_df.columns
                
                if has_rt and has_tmdb:
                    # First prioritize shows with both RT and TMDB data
                    both_data = group_df[(group_df['rt_score'].notna()) & (group_df['tmdb_score'].notna())]
                    if len(both_data) >= target_size:
                        return both_data.head(target_size)
                    
                    # If we need more, add shows with at least one data source
                    remaining = target_size - len(both_data)
                    one_data = group_df[(group_df['rt_score'].notna()) | (group_df['tmdb_score'].notna())]
                    one_data = one_data[~one_data.index.isin(both_data.index)]
                    
                    if len(both_data) + len(one_data) >= target_size:
                        return pd.concat([both_data, one_data.head(remaining)])
                    
                    # If we still need more, add remaining shows
                    remaining = target_size - len(both_data) - len(one_data)
                    no_data = group_df[(group_df['rt_score'].isna()) & (group_df['tmdb_score'].isna())]
                    return pd.concat([both_data, one_data, no_data.head(remaining)])
                
                elif has_rt:
                    # Prioritize shows with RT data
                    rt_data = group_df[group_df['rt_score'].notna()]
                    if len(rt_data) >= target_size:
                        return rt_data.head(target_size)
                    
                    # If we need more, add shows without RT data
                    remaining = target_size - len(rt_data)
                    no_rt = group_df[group_df['rt_score'].isna()]
                    return pd.concat([rt_data, no_rt.head(remaining)])
                
                elif has_tmdb:
                    # Prioritize shows with TMDB data
                    tmdb_data = group_df[group_df['tmdb_score'].notna()]
                    if len(tmdb_data) >= target_size:
                        return tmdb_data.head(target_size)
                    
                    # If we need more, add shows without TMDB data
                    remaining = target_size - len(tmdb_data)
                    no_tmdb = group_df[group_df['tmdb_score'].isna()]
                    return pd.concat([tmdb_data, no_tmdb.head(remaining)])
                
                # Fall back to simple sampling
                return df.sample(min(target_size, len(df)), random_state=42)
            
            # Try to sample by match level groups, but with safety checks
            try:
                # Group and sample
                sampled_matches = all_matches.groupby('match_level').apply(prioritize_shows_safe).reset_index(drop=True)
                
                # If we still have too many, sort by match level and take the top ones
                if len(sampled_matches) > OptimizerConfig.MAX_RESULTS:
                    sampled_matches = sampled_matches.sort_values(by=['match_level'], ascending=[True])
                    sampled_matches = sampled_matches.head(OptimizerConfig.MAX_RESULTS)
            except Exception as e:
                # If groupby fails, fall back to simple sampling
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"Debug: Sampling by match_level failed: {str(e)}")
                    st.write("Debug: Falling back to simple sampling")
                
                # Sort if possible, otherwise just sample
                try:
                    all_matches = all_matches.sort_values(by=['match_level'], ascending=[True])
                    sampled_matches = all_matches.head(OptimizerConfig.MAX_RESULTS)
                except:
                    # Last resort: just take a sample
                    sampled_matches = all_matches.sample(min(OptimizerConfig.MAX_RESULTS, len(all_matches)), random_state=42)
            
            all_matches = sampled_matches
        # If we have fewer matches than MAX_RESULTS, keep them all
        
        return all_matches, confidence_info
    
    # find_network_matches method removed - functionality now handled directly by NetworkAnalyzer.rank_networks_by_compatibility
    
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
            if OptimizerConfig.DEBUG_MODE:
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
                if OptimizerConfig.DEBUG_MODE:
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
                if OptimizerConfig.DEBUG_MODE:
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
                        if OptimizerConfig.DEBUG_MODE:
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
                        if OptimizerConfig.DEBUG_MODE:
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