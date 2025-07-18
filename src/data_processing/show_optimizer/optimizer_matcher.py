"""Matcher component for Show Optimizer.

This module handles all show matching logic based on criteria, implementing
flexible matching across different match levels with confidence calculation.

Key responsibilities:
- Search for shows matching criteria with different levels of flexibility
- Progressively relax criteria when needed to find sufficient matches
- Track the quality/confidence of different match types
- Return a prioritized list of matches for scoring and analysis
"""

from typing import Dict, List, Optional, Tuple, Any, Set, Union
import pandas as pd
import numpy as np

from .optimizer_config import OptimizerConfig
from .optimizer_data_contracts import CriteriaDict, ConfidenceInfo, update_confidence_info


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
        self._criteria_data = None
        
    def _get_data(self, data=None):
        """Get data for matching.
        
        Args:
            data: DataFrame of shows to match against
            
        Returns:
            DataFrame of shows, or empty DataFrame if no data available
        """
        if data is None or data.empty:
            # Use stored criteria data if available
            if self._criteria_data is not None and not self._criteria_data.empty:

                return self._criteria_data
            return pd.DataFrame()
        return data
        
    # _empty_confidence_info method removed - using update_confidence_info from optimizer_data_contracts instead
        
    def get_criteria_for_match_level(self, criteria: CriteriaDict, match_level: int) -> CriteriaDict:
        """Get a subset of criteria based on match level.
        
        Match levels now directly correspond to the number of criteria differences:
        - Level 1: All criteria match (0 differences)
        - Level 2: Missing 1 criterion
        - Level 3: Missing 2 criteria
        - Level 4: Missing 3 criteria
        - And so on...
        
        This implementation dynamically calculates which criteria to include based on
        the exact number of criteria differences specified by the match level.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            match_level: Match level (corresponds to criteria differences + 1)
            
        Returns:
            Criteria dictionary adjusted for the match level conforming to CriteriaDict
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
                
            # Try to find at least one criterion to keep, in order of importance
            classified = self.field_manager.classify_criteria_by_importance(criteria)
            
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
        
    def _calculate_criteria_coverage(self, criteria: CriteriaDict, shows: pd.DataFrame) -> Tuple[float, int]:
        """Calculate how well the shows match the original criteria.
        
        Args:
            criteria: Dictionary of criteria to check coverage for conforming to CriteriaDict
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
        
    def find_matches_with_fallback(self, criteria: CriteriaDict, data: pd.DataFrame = None, min_sample_size: int = None) -> Tuple[pd.DataFrame, ConfidenceInfo]:
        """Find shows matching criteria, with fallback to more permissive criteria if needed.
        
        This method will progressively relax the matching criteria until either:
            - Reach the target sample size (OptimizerConfig.MAX_RESULTS)
            - Have tried all possible match levels (down to 1 remaining criterion)
        
        Args:
            criteria: Dictionary of criteria to match against conforming to CriteriaDict
            data: DataFrame of shows to match against (uses cached data if None)
            min_sample_size: Minimum number of matches required
            
        Returns:
            Tuple of (matching_shows, match_info) with combined matches from all levels needed,
            where match_info conforms to ConfidenceInfo
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
        data_to_match = self._get_data(data)
        if len(data_to_match) == 0:
            # Only output debug message if we're not in a batch operation (criteria_scorer.calculate_criteria_impact)
            # This reduces noise in the logs during impact calculations

            # Return an empty DataFrame with the required columns and explicit confidence info
            empty_confidence = update_confidence_info({}, {
                'level': 'none',
                'match_level': 1,  # Use 1 as the default match level
                'error': 'No data available for matching'
            })
            return pd.DataFrame(columns=['match_level', 'match_quality', 'match_level_desc', 'title']), empty_confidence
        
        # Determine how many criteria we have to work with
        total_criteria = len(criteria)
        
        # Define the maximum number of criteria we're willing to drop
        # This is based on the total number of criteria but with a reasonable limit
        max_possible_drop = total_criteria - 1  # Maximum possible criteria to drop
        
        # Try each possible match level in order, from exact match to progressively fewer criteria
        # Pre-compute all level criteria to avoid redundant computation
        level_criteria_map = {}
        for level in range(1, max_possible_drop + 2):
            # Get criteria for this match level
            level_criteria = self.get_criteria_for_match_level(criteria, level)
            if level_criteria:  # Only store non-empty criteria
                level_criteria_map[level] = level_criteria
                
        # Process each level in order
        for level in sorted(level_criteria_map.keys()):
            level_criteria = level_criteria_map[level]
            # Match shows using the level-specific criteria
            level_matches, match_count = self._match_shows(level_criteria, data_to_match)
            
            # Skip if no matches at this level - use len() instead of .empty for better performance
            if len(level_matches) == 0:
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
            
            # Add new matches to our results - use len() instead of .empty for better performance
            if len(all_matches) == 0:
                all_matches = new_matches
            else:
                # Fast path for common case - if all required columns are present, skip column checks
                if 'match_level' in all_matches.columns and 'match_quality' in all_matches.columns and 'match_level_desc' in all_matches.columns:
                    # Use list of DataFrames for better performance with concat
                    all_matches = pd.concat([all_matches, new_matches], ignore_index=True)
                else:
                    # Pre-check required columns once before loop to avoid repeated lookups
                    required_columns = ['match_level', 'match_quality', 'match_level_desc']
                    all_matches_columns = set(all_matches.columns)
                    
                    # Only process columns that need to be added
                    missing_columns = [col for col in required_columns if col not in all_matches_columns]
                    
                    # Add missing columns efficiently
                    for col in missing_columns:
                        # Use a default value based on the column
                        if col == 'match_level':
                            all_matches[col] = 1  # Default to best match level
                        elif col == 'match_quality':
                            all_matches[col] = 100  # Default to perfect quality
                        elif col == 'match_level_desc':
                            all_matches[col] = self._get_match_level_description(1)
                    
                    # Now concatenate with all required columns present
                    all_matches = pd.concat([all_matches, new_matches], ignore_index=True)
            
            # Update unique titles set with new matches - use values for better performance
            if len(new_matches) > 0:
                unique_titles.update(new_matches['title'].values)
                total_unique_matches += new_unique_count
            
            # Only apply early termination for relaxed criteria (level > 1)
            # This ensures we find ALL exact matches before moving to relaxed criteria
            if level > 1 and total_unique_matches >= target_sample_size:
                # Early termination with sufficient matches
                break
        
        # If we still didn't find any matches at any level - use len() instead of .empty for better performance
        if len(all_matches) == 0:
            # Create an empty DataFrame with the required columns
            # Include all columns that will be used downstream
            empty_df = pd.DataFrame(columns=['match_level', 'match_quality', 'match_level_desc', 'title'])
            empty_confidence = update_confidence_info({}, {
                'level': 'none',
                'match_level': 1,  # Use 1 as the default match level
                'error': 'No matches found at any level'
            })
            return empty_df, empty_confidence
        
        # Prepare confidence info for the combined results
        confidence_info = best_confidence_info.copy() if best_confidence_info else update_confidence_info({}, {
            'level': 'none',
            'match_level': 1,  # Use 1 as the default match level
            'error': 'No confidence info available'
        })
        confidence_info['match_counts_by_level'] = all_match_counts
        confidence_info['total_unique_matches'] = total_unique_matches
        
        # Add a summary of the match levels we tried and how many matches we found at each level
        level_summaries = {}
        for level, count in all_match_counts.items():
            level_desc = self._get_match_level_description(level)
            level_summaries[level_desc] = count
        
        confidence_info['match_level_summary'] = level_summaries
        
        # Sort by match_level (ascending) - no success score sorting during selection
        # Use len() instead of .empty for better performance
        if len(all_matches) > 0:
            # The match_level column is guaranteed to exist since we set it above
            all_matches = all_matches.sort_values(by=['match_level'], ascending=[True])
        # Apply prioritized sampling within each match level if we have more than MAX_RESULTS
        if len(all_matches) > OptimizerConfig.MAX_RESULTS:
            # Define a function to prioritize shows with RT and TMDB data
            def prioritize_shows(group_df):
                """Prioritize shows with better metrics data.
                
                Args:
                    group_df: DataFrame of shows to prioritize
                    
                Returns:
                    DataFrame of prioritized shows, limited to target size
                """
                # Handle empty DataFrame case
                if group_df is None or len(group_df) == 0:
                    return pd.DataFrame()
                    
                # Calculate the target sample size for this group
                # Use ceiling division to avoid getting fewer than MAX_RESULTS total shows
                total_matches_count = len(all_matches)
                target_size = min(len(group_df), max(1, int(np.ceil(OptimizerConfig.MAX_RESULTS * len(group_df) / total_matches_count))))
                
                # Check if we have RT and TMDB columns to prioritize
                has_rt = 'rt_score' in group_df.columns
                has_tmdb = 'tmdb_score' in group_df.columns
                
                try:
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
                except Exception:
                    # If any error occurs in the prioritization logic, fall back to simple sampling
                    pass
                # Falling back to simple sampling
                return group_df.sample(min(target_size, len(group_df)), random_state=42) if len(group_df) > 0 else group_df
            
            # Ensure match_level column exists before grouping or sorting
            if 'match_level' not in all_matches.columns:
                # Add match_level column with default value (1 = best match)
                all_matches['match_level'] = 1
            
            # Try to sample by match level groups, but with safety checks
            try:
                # TIERED SAMPLING STRATEGY:
                # 1. Keep ALL perfect matches (match_level=1)
                # 2. Fill remaining slots (up to MAX_RESULTS) with next match levels
                # 3. Within each level, sort by RT/TMDB data inclusion
                
                # First, separate by match level and sort them (1 is best)
                match_levels = sorted(all_matches['match_level'].unique())
                remaining_slots = OptimizerConfig.MAX_RESULTS
                result_dfs = []
                
                # Define a function to sort by RT/TMDB data inclusion (no filtering)
                def sort_by_metrics(df):
                    """Sort shows by RT and TMDB data inclusion.
                    
                    Args:
                        df: DataFrame of shows to sort
                        
                    Returns:
                        DataFrame of sorted shows (no filtering)
                    """
                    # Check if we have RT and TMDB columns
                    has_rt = 'rt_score' in df.columns
                    has_tmdb = 'tmdb_score' in df.columns
                    
                    if not has_rt and not has_tmdb:
                        return df
                        
                    # Create a priority column for sorting
                    # 3: Has both RT and TMDB
                    # 2: Has RT only
                    # 1: Has TMDB only
                    # 0: Has neither
                    if has_rt and has_tmdb:
                        df['metrics_priority'] = 0
                        df.loc[(df['rt_score'].notna()) & (df['tmdb_score'].notna()), 'metrics_priority'] = 3
                        df.loc[(df['rt_score'].notna()) & (df['tmdb_score'].isna()), 'metrics_priority'] = 2
                        df.loc[(df['rt_score'].isna()) & (df['tmdb_score'].notna()), 'metrics_priority'] = 1
                    elif has_rt:
                        df['metrics_priority'] = 0
                        df.loc[df['rt_score'].notna(), 'metrics_priority'] = 2
                    elif has_tmdb:
                        df['metrics_priority'] = 0
                        df.loc[df['tmdb_score'].notna(), 'metrics_priority'] = 1
                    
                    # Sort by priority (descending)
                    return df.sort_values(by=['metrics_priority'], ascending=[False])
                
                # Process each match level in order
                for level in match_levels:
                    level_matches = all_matches[all_matches['match_level'] == level].copy()
                    
                    # For perfect matches (level 1), keep ALL of them
                    if level == 1:
                        # Keep ALL perfect matches - they are critical for accurate recommendations
                        perfect_count = len(level_matches)
                        
                        # Sort by RT/TMDB data inclusion
                        level_matches = sort_by_metrics(level_matches)
                        
                        # Keep all perfect matches
                        result_dfs.append(level_matches)
                        
                        # Update remaining slots
                        remaining_slots -= perfect_count
                    else:
                        # For non-perfect matches, sort by RT/TMDB data inclusion
                        level_matches = sort_by_metrics(level_matches)
                        
                        # Only take what fits in remaining slots
                        if remaining_slots > 0:
                            slots_to_use = min(remaining_slots, len(level_matches))
                            # Take matches up to the available slots
                            result_dfs.append(level_matches.head(slots_to_use))
                            remaining_slots -= slots_to_use
                        else:
                            # No slots left
                            break
                
                # Combine all match levels, with perfect matches first
                sampled_matches = pd.concat(result_dfs) if result_dfs else pd.DataFrame()
            except Exception as e:
                # If groupby fails, fall back to simple sampling
                # Falling back to simple sampling
                
                # Sort if possible, otherwise just sample
                try:
                    # Make sure match_level exists before trying to sort by it
                    if 'match_level' not in all_matches.columns:
                        # Add match_level column with default value (1 = best match)
                        all_matches['match_level'] = 1
                    
                    # TIERED FALLBACK SAMPLING STRATEGY:
                    # 1. Keep ALL perfect matches (match_level=1)
                    # 2. Fill remaining slots (up to MAX_RESULTS) with next match levels
                    # 3. Within each level, sort by RT/TMDB data inclusion if possible
                    
                    # First, separate by match level and sort them (1 is best)
                    match_levels = sorted(all_matches['match_level'].unique())
                    remaining_slots = OptimizerConfig.MAX_RESULTS
                    result_dfs = []
                    
                    # Process each match level in order
                    for level in match_levels:
                        level_matches = all_matches[all_matches['match_level'] == level].copy()
                        
                        # For perfect matches (level 1), keep ALL of them
                        if level == 1:
                            # Keep ALL perfect matches - they are critical for accurate recommendations
                            perfect_count = len(level_matches)
                            
                            # Try to sort by RT/TMDB if those columns exist
                            try:
                                has_rt = 'rt_score' in level_matches.columns
                                has_tmdb = 'tmdb_score' in level_matches.columns
                                
                                if has_rt or has_tmdb:
                                    # Create a priority column
                                    level_matches['metrics_priority'] = 0
                                    
                                    if has_rt and has_tmdb:
                                        level_matches.loc[(level_matches['rt_score'].notna()) & 
                                                         (level_matches['tmdb_score'].notna()), 'metrics_priority'] = 3
                                        level_matches.loc[(level_matches['rt_score'].notna()) & 
                                                         (level_matches['tmdb_score'].isna()), 'metrics_priority'] = 2
                                        level_matches.loc[(level_matches['rt_score'].isna()) & 
                                                         (level_matches['tmdb_score'].notna()), 'metrics_priority'] = 1
                                    elif has_rt:
                                        level_matches.loc[level_matches['rt_score'].notna(), 'metrics_priority'] = 2
                                    elif has_tmdb:
                                        level_matches.loc[level_matches['tmdb_score'].notna(), 'metrics_priority'] = 1
                                    
                                    # Sort by priority
                                    level_matches = level_matches.sort_values(by=['metrics_priority'], ascending=[False])
                            except Exception as e:
                                # If sorting fails, just continue with unsorted matches
                                pass
                            
                            # Keep all perfect matches in fallback mode
                            result_dfs.append(level_matches)
                            
                            # Update remaining slots
                            remaining_slots -= perfect_count
                        else:
                            # For non-perfect matches, only take what fits in remaining slots
                            if remaining_slots > 0:
                                slots_to_use = min(remaining_slots, len(level_matches))
                                # Take matches up to the available slots in fallback mode
                                result_dfs.append(level_matches.head(slots_to_use))
                                remaining_slots -= slots_to_use
                            else:
                                # No slots left
                                break
                    
                    # Combine all match levels, with perfect matches first
                    sampled_matches = pd.concat(result_dfs) if result_dfs else pd.DataFrame()
                except Exception as e:
                    # If sorting fails, just sample randomly
                    sampled_matches = all_matches.sample(min(OptimizerConfig.MAX_RESULTS, len(all_matches)), random_state=42)
            
            # Use the sampled matches as our final result
            all_matches = sampled_matches
            
        return all_matches, confidence_info

    def _match_shows(self, criteria: CriteriaDict, data: pd.DataFrame = None) -> Tuple[pd.DataFrame, int]:
        """Match shows based on criteria.
        
        Args:
            criteria: Dictionary of criteria conforming to CriteriaDict
            data: DataFrame of shows to match against
            
        Returns:
            Tuple of (matching_shows, match_count)
        """
        # Use helper method to get data
        data = self._get_data(data)
        # Use len() instead of .empty for better performance
        if len(data) == 0:
            # Silent handling for empty data - happens frequently during matching
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
        
        # Start with a copy of the data as our matches - use .copy(deep=False) for better performance
        # since we're only filtering rows, not modifying columns
        matches = data.copy(deep=False)
        
        # Separate array and scalar fields for different processing
        array_fields_to_process = {}
        scalar_fields = {}
        
        # Process each criterion
        for field_name, value in clean_criteria.items():
            # Use field_manager to map the field name to the correct column name
            field_column = self.field_manager.get_field_column_name(field_name, matches.columns)
            
            # Skip if the field doesn't exist in the data
            if field_column not in matches.columns:
                # Skip debug output to reduce noise
                continue
                
            # Check if this is an array field - ONLY use field_manager's determination
            # This ensures consistency with how fields are processed elsewhere
            is_array = self.field_manager.get_field_type(field_name) == 'array'
            
            # Special handling for subgenres which is always an array field
            if field_name == 'subgenres':
                is_array = True
                
            # Array field detection for genre and subgenres
            
            if is_array:
                # For array fields, we need to check if any value matches
                if isinstance(value, list):
                    value_set = set(value)
                    # If the column contains lists, use list intersection
                    # For each show's array field, convert to set and check intersection with criteria values
                    if len(matches) > 0:
                        sample = matches[field_column].iloc[0]
                        
                        # Use different strategies based on dataset size for better performance
                        if len(matches) <= 100:  # Small dataset optimization
                            if isinstance(sample, list):
                                # Fast path for small datasets with list columns
                                mask = pd.Series([len(value_set.intersection(set(x))) > 0 if isinstance(x, list) else False 
                                                 for x in matches[field_column]], index=matches.index)
                            else:
                                # Fallback with type checking for small datasets
                                mask = pd.Series([isinstance(x, list) and len(value_set.intersection(set(x))) > 0 
                                                 for x in matches[field_column]], index=matches.index)
                        else:
                            # For larger datasets, use apply which is optimized for pandas
                            if isinstance(sample, list):
                                # Cache the set conversion to avoid repeated conversions
                                # This significantly improves performance for large datasets
                                mask = matches[field_column].apply(
                                    lambda x: len(value_set.intersection(set(x) if isinstance(x, list) else set())) > 0)
                            else:
                                # Fallback with type checking
                                mask = matches[field_column].apply(
                                    lambda x: isinstance(x, list) and len(value_set.intersection(set(x))) > 0)
                    else:
                        # Empty DataFrame case
                        mask = pd.Series([], dtype=bool)
                else:
                    # Single value in array field
                    if len(matches) > 0:
                        sample = matches[field_column].iloc[0]
                        
                        # Use different strategies based on dataset size
                        if len(matches) <= 100:  # Small dataset optimization
                            if isinstance(sample, list):
                                # Fast path for small datasets with list columns
                                mask = pd.Series([value in x if isinstance(x, list) else False 
                                                 for x in matches[field_column]], index=matches.index)
                            else:
                                # Fallback with type checking for small datasets
                                mask = pd.Series([isinstance(x, list) and value in x 
                                                 for x in matches[field_column]], index=matches.index)
                        else:
                            # For larger datasets, use vectorized operations where possible
                            try:
                                # Try using numpy vectorization for better performance
                                if isinstance(sample, list):
                                    # Convert to numpy array for faster processing if possible
                                    mask = matches[field_column].apply(lambda x: value in x if isinstance(x, list) else False)
                                else:
                                    # Fallback with type checking
                                    mask = matches[field_column].apply(lambda x: isinstance(x, list) and value in x)
                            except Exception:
                                # If numpy fails, fall back to pandas apply
                                mask = matches[field_column].apply(lambda x: isinstance(x, list) and value in x)
                    else:
                        # Empty DataFrame case
                        mask = pd.Series([], dtype=bool)
            else:
                # For scalar fields - use vectorized operations for better performance
                if isinstance(value, list):
                    # Multiple values: any show with any of the values matches
                    # Use isin which is already optimized for pandas
                    mask = matches[field_column].isin(value)
                else:
                    # Single value: exact match
                    # For small datasets, use numpy for faster comparison if possible
                    if len(matches) > 100:
                        try:
                            # Try to use numpy for faster comparison
                            import numpy as np
                            # Convert to numpy array for faster comparison
                            col_values = matches[field_column].values
                            mask = pd.Series(col_values == value, index=matches.index)
                        except Exception:
                            # Fall back to pandas if numpy fails
                            mask = matches[field_column] == value
                    else:
                        # For small datasets, pandas comparison is fast enough
                        mask = matches[field_column] == value
                    
            # Apply filter
            matches = matches[mask]
            
        # Matching complete
        match_count = len(matches)
        
        return matches, match_count
    
    def calculate_match_confidence(self, shows: pd.DataFrame, match_level: int, 
                                  criteria: CriteriaDict) -> ConfidenceInfo:
        """Calculate confidence metrics for a set of matched shows.
        
        Args:
            shows: DataFrame of matched shows
            match_level: Match level used (1-4)
            criteria: Original criteria dictionary conforming to CriteriaDict
            
        Returns:
            Dictionary with confidence metrics conforming to ConfidenceInfo:
            - level: Confidence level string ('none', 'very_low', 'low', 'medium', 'high')
            - score: Confidence score (0-1)
            - match_quality: Quality of the match based on match level (0-1)
            - sample_size: Number of shows in the sample
            - match_level: The actual match level based on criteria validation
        """
        # Fast path: If shows is empty, return minimal confidence
        if shows is None or len(shows) == 0:
            return {
                'level': 'none',
                'score': 0.0,
                'match_quality': 0.0,
                'sample_size': 0,
                'match_level': match_level
            }
        
        # Get sample size - already checked above
        sample_size = len(shows)
        
        # Calculate criteria coverage - use cached value if available
        total_criteria = getattr(self, '_cached_total_criteria', None)
        if total_criteria is None:
            total_criteria = len(OptimizerConfig.CRITERIA_IMPORTANCE)
            setattr(self, '_cached_total_criteria', total_criteria)
            
        criteria_count = len(criteria)
        
        # Calculate match quality based on criteria difference
        # Level 1 = 0 differences, Level 2 = 1 difference, etc.
        criteria_diff = match_level - 1
        
        # Use cached quality scores if possible
        quality_cache = getattr(self, '_quality_cache', {})
        match_quality = quality_cache.get(criteria_diff)
        if match_quality is None:
            match_quality = OptimizerConfig.get_quality_for_diff(criteria_diff)
            if not hasattr(self, '_quality_cache'):
                setattr(self, '_quality_cache', {})
            quality_cache = getattr(self, '_quality_cache')
            quality_cache[criteria_diff] = match_quality
        
        # Cache key for confidence calculations
        cache_key = (sample_size, criteria_count, match_level)
        
        # Use cached confidence values if possible
        confidence_cache = getattr(self, '_confidence_cache', {})
        cached_values = confidence_cache.get(cache_key)
        
        if cached_values:
            confidence_score, confidence_level = cached_values
        else:
            # Calculate confidence score using OptimizerConfig
            confidence_score = OptimizerConfig.calculate_confidence_score(
                sample_size, criteria_count, total_criteria, match_level)
            
            # Get confidence level string
            confidence_level = OptimizerConfig.get_confidence_level(sample_size, match_level)
            
            # Cache the results
            if not hasattr(self, '_confidence_cache'):
                setattr(self, '_confidence_cache', {})
            confidence_cache = getattr(self, '_confidence_cache')
            confidence_cache[cache_key] = (confidence_score, confidence_level)
            
            # Limit cache size to prevent memory leaks
            if len(confidence_cache) > 1000:  # Arbitrary limit
                # Simple strategy: clear half the cache when it gets too large
                keys_to_remove = list(confidence_cache.keys())[:500]
                for key in keys_to_remove:
                    confidence_cache.pop(key, None)
        
        # Validate the actual match level by checking if all criteria are truly matched
        # This is especially important for array fields like character_types
        actual_match_level = match_level
        
        # Only perform validation if we have shows and claiming exact match (level 1)
        if len(shows) > 0 and match_level == 1:
            # Fast path: Use cached validation results if available
            # Convert criteria.items() to a hashable form by handling list values
            hashable_criteria = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in criteria.items())
            validation_cache_key = (frozenset(hashable_criteria), tuple(sorted(shows.columns)))
            validation_cache = getattr(self, '_validation_cache', {})
            cached_level = validation_cache.get(validation_cache_key)
            
            if cached_level is not None:
                actual_match_level = cached_level
            else:
                # Define a helper function to validate criteria and determine match level
                def validate_criteria():
                    # Pre-compute field column mapping once for all criteria
                    field_columns = {}
                    for field_name in criteria.keys():
                        field_columns[field_name] = self.field_manager.get_field_column_name(field_name, shows.columns)
                    
                    # Check each criterion to see if it's actually matched
                    for field_name, value in criteria.items():
                        # Skip empty criteria
                        if value is None or (isinstance(value, list) and not value):
                            continue
                        
                        # Get the mapped column name
                        field_column = field_columns.get(field_name)
                        
                        # Check if column exists in the data
                        if field_column not in shows.columns:
                            return 2  # Downgrade to level 2 if field is missing
                        
                        # Handle array fields differently
                        if isinstance(value, list):
                            # Pre-compute value set once
                            value_set = set(value)
                            
                            # Sample the first row to check data format
                            sample = shows[field_column].iloc[0] if len(shows) > 0 else None
                            
                            # Check if all shows match this array criterion
                            if isinstance(sample, list):
                                # Optimize for small datasets
                                if len(shows) <= 100:
                                    # For small datasets, iterate directly
                                    for idx, row_value in enumerate(shows[field_column]):
                                        if not (isinstance(row_value, list) and bool(value_set.intersection(row_value))):
                                            return 2  # Downgrade to level 2
                                else:
                                    # For larger datasets, use apply with optimized lambda
                                    all_match = shows[field_column].apply(
                                        lambda x: isinstance(x, list) and not value_set.isdisjoint(x)
                                    ).all()
                                    if not all_match:
                                        return 2  # Downgrade to level 2
                            else:
                                # For non-list columns, use vectorized operations
                                all_match = shows[field_column].isin(value).any(axis=0)
                                if not all_match:
                                    return 2  # Downgrade to level 2
                        else:
                            # Handle scalar fields
                            # Use the pre-computed field column mapping
                            field_id = field_columns.get(field_name)
                            
                            # Check if column exists
                            if field_id not in shows.columns:
                                return 2  # Downgrade to level 2 if field is missing
                            
                            # Optimize equality check based on dataset size
                            if len(shows) > 100:
                                try:
                                    # Use numpy's vectorized equality check for large datasets
                                    import numpy as np
                                    if not np.all(shows[field_id].values == value):
                                        return 2  # Downgrade to level 2
                                except:
                                    # Fallback to pandas if numpy approach fails
                                    if not (shows[field_id] == value).all():
                                        return 2  # Downgrade to level 2
                            else:
                                # For small datasets, pandas is fast enough
                                if not (shows[field_id] == value).all():
                                    return 2  # Downgrade to level 2
                    
                    # If we got here, all criteria matched
                    return 1
                
                # Run the validation and get the actual match level
                actual_match_level = validate_criteria()
                
                # Cache the validation result
                if not hasattr(self, '_validation_cache'):
                    setattr(self, '_validation_cache', {})
                validation_cache = getattr(self, '_validation_cache')
                validation_cache[validation_cache_key] = actual_match_level
                
                # Limit cache size to prevent memory leaks
                if len(validation_cache) > 1000:  # Arbitrary limit
                    # Simple strategy: clear half the cache when it gets too large
                    keys_to_remove = list(validation_cache.keys())[:500]
                    for key in keys_to_remove:
                        validation_cache.pop(key, None)
        
        # Create the confidence info dictionary
        confidence_info = {
            'level': confidence_level,
            'score': confidence_score,
            'match_quality': match_quality,
            'sample_size': sample_size,
            'match_level': actual_match_level,  # Use the validated match level
            'original_match_level': match_level,  # Keep track of the original level
            # Generate level name dynamically based on criteria difference
            'match_level_name': self._get_match_level_description(actual_match_level)
        }
        

            
        return confidence_info
        
    def _get_relaxed_criteria(self, criteria: CriteriaDict, relaxation_tier: str) -> List[CriteriaDict]:
        """Generate sets of relaxed criteria by removing criteria of the specified importance tier.
        
        Args:
            criteria: Dictionary of criteria to relax conforming to CriteriaDict
            relaxation_tier: Importance tier to relax ('secondary', 'primary', 'core')
                
        Returns:
            List of dictionaries with relaxed criteria sets conforming to CriteriaDict
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
        
    def _calculate_relevance_score(self, original_criteria: CriteriaDict, matched_show: pd.Series) -> float:
        """Calculate a weighted relevance score for a show based on how well it matches the original criteria.
        
        Args:
            original_criteria: Dictionary of original criteria conforming to CriteriaDict
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