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
        
    def _empty_confidence_info(self):
        """Return standard empty confidence info dictionary.
        
        Returns:
            Dictionary with default confidence metrics
        """
        return {
            'level': 'none', 
            'score': 0, 
            'match_quality': 0, 
            'sample_size': 0
        }
        
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
        
        # Use helper method to get data
        data = self._get_data(data)
        if data.empty:
            return pd.DataFrame(), self._empty_confidence_info()
        
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
        
        # Use the best level we found
        result_shows = all_matches_by_level[best_level]
        
        # Add match level counts to confidence info
        confidence_info['match_counts'] = match_counts
        confidence_info['match_level_name'] = OptimizerConfig.MATCH_LEVELS[best_level]['name']
        confidence_info['confidence_level'] = confidence_info.get('level', 'none')  # Ensure confidence_level is set for fallback logic
        
        return result_shows, confidence_info
        
    def find_matches_with_fallback(self, criteria: Dict[str, Any], data: pd.DataFrame = None,
                                min_sample_size: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Enhanced version of find_matches that incorporates sophisticated fallback logic.
        
        This method first tries the standard matching approach and then applies additional
        fallback strategies if the results are insufficient. The fallback logic includes:
        1. Relaxing secondary criteria
        2. Relaxing primary criteria if needed
        3. Relaxing core criteria as a last resort
        
        Args:
            criteria: Dictionary of criteria to match against
            data: DataFrame of shows to match against (uses cached data if None)
            min_sample_size: Minimum number of matches required
            
        Returns:
            Tuple of (matching_shows, match_info) with enhanced fallback matches if needed
        """
        # First try standard matching
        matched_shows, confidence_info = self.find_matches(criteria, data, min_sample_size)
        match_count = len(matched_shows)
        
        # Check if fallback is needed based on match count and confidence
        min_matches = OptimizerConfig.FALLBACK_SYSTEM['relaxation']['min_matches_before_fallback']
        min_confidence = OptimizerConfig.FALLBACK_SYSTEM['relaxation']['min_confidence_before_fallback']
        confidence_level = confidence_info.get('confidence_level', 'none')
        
        # Define confidence level hierarchy for comparison
        confidence_levels = ['none', 'very_low', 'low', 'medium', 'high']
        
        # If we have enough matches and confidence is high enough, no fallback needed
        if match_count >= min_matches and confidence_levels.index(confidence_level) >= confidence_levels.index(min_confidence):
            return matched_shows, confidence_info
            
        st.write(f"Insufficient matches ({match_count}) or confidence ({confidence_level}), trying fallback strategies...")
        
        # Get relaxation tiers from config
        relaxation_tiers = OptimizerConfig.FALLBACK_SYSTEM['relaxation']['relaxation_tiers']
        best_fallback_matches = pd.DataFrame()
        best_fallback_count = 0
        best_fallback_confidence = {}
        best_fallback_relevance = 0
        
        # Try each relaxation tier in order
        for relaxation_tier in relaxation_tiers:
            st.write(f"Trying relaxation tier: {relaxation_tier}")
            
            # Get sets of relaxed criteria for this tier
            relaxed_criteria_sets = self._get_relaxed_criteria(criteria, relaxation_tier)
            
            # If no relaxable criteria in this tier, continue to next tier
            if not relaxed_criteria_sets:
                st.write(f"No relaxable criteria in tier: {relaxation_tier}")
                continue
                
            # Try each relaxed criteria set
            for relaxed_set in relaxed_criteria_sets:
                relaxed_criteria = relaxed_set['criteria']
                relaxed_type = relaxed_set['relaxed_type']
                relaxed_name = relaxed_set['relaxed_name']
                
                st.write(f"Trying relaxed criteria by removing: {relaxed_name}")
                
                # Get matching shows with relaxed criteria
                relaxed_matches, relaxed_confidence = self.find_matches(relaxed_criteria, data, min_sample_size)
                relaxed_count = len(relaxed_matches)
                
                # Check if relaxation improved the situation significantly
                if relaxed_count >= match_count * OptimizerConfig.FALLBACK_SYSTEM['relaxation']['min_sample_increase_factor']:
                    st.write(f"Relaxing '{relaxed_name}' increased matches from {match_count} to {relaxed_count}")
                    
                    # Calculate relevance scores for the new matches
                    relevance_scores = []
                    for _, show in relaxed_matches.iterrows():
                        score = self._calculate_relevance_score(criteria, show)
                        relevance_scores.append(score)
                    
                    # Add relevance scores to the DataFrame
                    relaxed_matches['relevance_score'] = relevance_scores
                    
                    # Filter by minimum relevance score
                    min_relevance = OptimizerConfig.FALLBACK_SYSTEM['relevance']['min_relevance_score']
                    relevant_matches = relaxed_matches[relaxed_matches['relevance_score'] >= min_relevance]
                    
                    # If we have relevant matches and they're better than our current best
                    if not relevant_matches.empty:
                        avg_relevance = relevant_matches['relevance_score'].mean()
                        
                        # Update best fallback if this is better
                        if len(relevant_matches) > best_fallback_count or \
                           (len(relevant_matches) == best_fallback_count and avg_relevance > best_fallback_relevance):
                            best_fallback_matches = relevant_matches
                            best_fallback_count = len(relevant_matches)
                            best_fallback_confidence = relaxed_confidence
                            best_fallback_relevance = avg_relevance
                            
                            # Add fallback info to confidence info
                            best_fallback_confidence['fallback_info'] = {
                                'relaxed_type': relaxed_type,
                                'relaxed_name': relaxed_name,
                                'original_count': match_count,
                                'fallback_count': best_fallback_count,
                                'avg_relevance': avg_relevance
                            }
                            
                            st.write(f"Found {best_fallback_count} relevant matches with average relevance {avg_relevance:.2f}")
                            
                            # If we found a very good fallback, stop searching
                            if best_fallback_count >= min_sample_size * 2 and avg_relevance >= 0.8:
                                st.write("Found excellent fallback matches, stopping search")
                                break
                
            # If we found good fallback matches in this tier, no need to try more tiers
            if best_fallback_count >= min_sample_size:
                st.write(f"Found sufficient fallback matches in tier {relaxation_tier}, stopping search")
                break
                
        # Return the best fallback matches if we found any, otherwise return original matches
        if not best_fallback_matches.empty and best_fallback_count > match_count:
            st.write(f"Using fallback matches: {best_fallback_count} shows with relevance {best_fallback_relevance:.2f}")
            return best_fallback_matches, best_fallback_confidence
        else:
            st.write("No better fallback matches found, using original matches")
            return matched_shows, confidence_info
    
    def find_network_matches(self, criteria: Dict[str, Any], data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """Find shows matching criteria for each available network.
        
        This method:
        1. Gets all unique networks from the data
        2. For each network, adds the network ID to the criteria
        3. Uses flexible matching to find shows matching both criteria and network
        4. Returns match results for each network with confidence information
        
        Args:
            criteria: Base criteria to match against
            data: DataFrame of shows to match against (uses cached data if None)
            
        Returns:
            List of dictionaries with network information and matching results
        """
        # Use helper method to get data
        data = self._get_data(data)
        if data.empty:
            return []
        
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
                    matching_shows, confidence_info = self.find_matches(
                        network_criteria, 
                        data=data, 
                        min_sample_size=OptimizerConfig.CONFIDENCE['minimum_sample']
                    )
                    
                    # Store the results
                    results.append({
                        'network_id': int(network_id),
                        'network_name': network_name,
                        'matching_shows': matching_shows,
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
        
        Match levels are defined in OptimizerConfig.MATCH_LEVELS:
        1 - Exact match with all criteria
        2 - Match with all primary criteria, relaxed secondary criteria
        3 - Match with only high-importance criteria
        4 - Match with minimal essential criteria
        
        Args:
            criteria: Dictionary of criteria
            match_level: Match level (1-4)
            
        Returns:
            Criteria dictionary adjusted for the match level
        """
        # Validate match level against config
        if match_level not in OptimizerConfig.MATCH_LEVELS:
            st.error(f"Invalid match level {match_level}, using all criteria")
            return criteria.copy()
            
        # If match level is 1, use all criteria
        if match_level == 1:
            return criteria.copy()
            
        # Classify criteria by importance using field_manager
        classified = self.field_manager.classify_criteria_by_importance(criteria)
        
        # Apply match level rules based on OptimizerConfig.MATCH_LEVELS
        result = {}
        
        # Always include essential criteria at all levels
        result.update(classified['essential'])
        
        # For level 4, use only essential and core criteria
        if match_level == 4:
            result.update(classified['core'])
            return result
            
        # For level 3, use essential, core, and primary criteria
        if match_level == 3:
            result.update(classified['core'])
            result.update(classified['primary'])
            return result
            
        # For level 2, use all criteria except one secondary criterion (if any)
        if match_level == 2:
            result.update(classified['core'])
            result.update(classified['primary'])
            
            # Add all but one secondary criterion if any exist
            if classified['secondary']:
                secondary_items = list(classified['secondary'].items())
                for i, (field, value) in enumerate(secondary_items):
                    if i < len(secondary_items) - 1:  # Skip the last one
                        result[field] = value
            return result
        
        # Default to all criteria if match level is invalid
        return criteria.copy()
    
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
        
        # Calculate match quality based on match level
        match_quality = OptimizerConfig.MATCH_LEVELS.get(match_level, {}).get('min_quality', 0.3)
        
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
                        st.write(f"Not all shows match array criterion '{field_name}', downgrading match level")
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