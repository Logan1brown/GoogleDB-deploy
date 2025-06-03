"""CriteriaScorer: Calculates raw success scores for show criteria.

This component calculates success rates for different criteria based on
historical show data. It integrates with SuccessAnalyzer for base success metrics
and provides data to CriteriaAnalyzer for pattern recognition.

Key concepts:

1. Success Scoring:
   - Calculate success rates for each criteria value (genre, source type, etc.)
   - Analyze network-specific success patterns
   - Weight different success components (audience, critics, longevity)

2. Data Flow:
   - Get criteria from FieldManager
   - Get success metrics from SuccessAnalyzer
   - Calculate raw scores based on OptimizerConfig weights
   - Provide scores to CriteriaAnalyzer

3. Performance:
   - Cache common calculations
   - Support incremental updates for real-time UI
   - Handle partial criteria sets
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import pandas as pd
import numpy as np
import logging
from functools import lru_cache
from datetime import datetime, timedelta

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .optimizer_config import OptimizerConfig
from .field_manager import FieldManager

logger = logging.getLogger(__name__)


@dataclass
class NetworkMatch:
    """Network match information with success metrics."""
    network_id: int
    network_name: str
    compatibility_score: float  # 0-1 score of how well the network matches criteria
    success_probability: float  # 0-1 probability of success on this network
    sample_size: int  # Number of shows in the sample
    confidence: str  # none, low, medium, high


@dataclass
class ComponentScore:
    """Success score for a component (audience, critics, longevity)."""
    component: str  # audience, critics, or longevity
    score: float  # 0-1 score
    sample_size: int
    confidence: str  # none, low, medium, high
    details: Dict[str, float] = field(default_factory=dict)  # Detailed breakdown of score


class CriteriaScorer:
    """Calculates raw success scores for show criteria."""
    
    def __init__(self, shows_analyzer: ShowsAnalyzer, success_analyzer: SuccessAnalyzer):
        """Initialize the criteria scorer.
        
        Args:
            shows_analyzer: ShowsAnalyzer instance for show data
            success_analyzer: SuccessAnalyzer instance for success metrics
        """
        self.shows_analyzer = shows_analyzer
        self.success_analyzer = success_analyzer
        
        # Get reference data from ShowsAnalyzer using fetch_comp_data
        try:
            comp_df, reference_data = shows_analyzer.fetch_comp_data()
            self.field_manager = FieldManager(reference_data)
        except Exception as e:
            logger.error(f"Error initializing FieldManager: {e}")
            # Initialize with empty reference data as fallback
            self.field_manager = FieldManager({})
            
        self.criteria_data = None
        self.last_update = None
        self.cache_duration = OptimizerConfig.PERFORMANCE['cache_duration']
        
    def fetch_criteria_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch criteria data with success metrics.
        
        Args:
            force_refresh: Whether to force a refresh of the data
            
        Returns:
            DataFrame with criteria data and success metrics
        """
        import streamlit as st
        
        # Check if we need to refresh the data
        current_time = datetime.now()
        if (self.criteria_data is None or 
            force_refresh or 
            self.last_update is None or 
            (current_time - self.last_update).total_seconds() > self.cache_duration):
            
            st.write("DEBUG: Fetching fresh criteria data with success metrics")
            
            # Get show data from ShowsAnalyzer using fetch_comp_data
            # This returns both the show data and reference data
            comp_df, _ = self.shows_analyzer.fetch_comp_data(force=force_refresh)
            
            if comp_df.empty:
                st.error("DEBUG ERROR: Empty data returned from ShowsAnalyzer")
                raise ValueError("No show data available from ShowsAnalyzer")
                
            # Create base criteria data
            self.criteria_data = comp_df.copy()
            
            # Fetch success data from SuccessAnalyzer
            st.write("DEBUG: Fetching success data from SuccessAnalyzer")
            success_data = self.success_analyzer.fetch_success_data()
            
            if success_data.empty:
                st.error("DEBUG ERROR: Empty success data returned from SuccessAnalyzer")
                raise ValueError("No success data available from SuccessAnalyzer")
            
            st.write(f"DEBUG: SuccessAnalyzer returned data with {len(success_data)} rows")
            st.write(f"DEBUG: SuccessAnalyzer data columns: {list(success_data.columns)}")
            
            # Check if show_id is in the columns
            st.write(f"DEBUG: 'show_id' in columns: {'show_id' in success_data.columns}")
            
            if len(success_data) > 0:
                st.write(f"DEBUG: Sample success score: {success_data['success_score'].iloc[0] if 'success_score' in success_data.columns else 'No success_score column'}")
                st.write(f"DEBUG: Success score range: {success_data['success_score'].min() if 'success_score' in success_data.columns else 'N/A'} to {success_data['success_score'].max() if 'success_score' in success_data.columns else 'N/A'}")
                st.write(f"DEBUG: Success score distribution: {success_data['success_score'].value_counts().head(3) if 'success_score' in success_data.columns else 'N/A'}")
            
            
            # Create a mapping of show_id to success_score
            success_scores = {}
            
            # Check if success_score column exists
            has_success_score = 'success_score' in success_data.columns
            
            if not has_success_score:
                st.error("DEBUG ERROR: 'success_score' column missing from SuccessAnalyzer data")
                raise ValueError("'success_score' column required for success score calculation")
            
            # Check if show_id is the index (SuccessAnalyzer sets show_id as index)
            is_show_id_index = success_data.index.name == 'show_id'
            st.write(f"DEBUG: Index name: {success_data.index.name}")
            
            # Map show_id to success_score
            if is_show_id_index:
                # If show_id is the index, use the index for mapping
                st.write("DEBUG: Using DataFrame index as show_id for mapping success scores")
                for idx, row in success_data.iterrows():
                    # Normalize success score to 0-1 range if it's on a 0-100 scale
                    # Check if the score is already normalized (0-1) or needs normalization (0-100)
                    score = row['success_score']
                    
                    # Check for NaN or None values
                    if pd.isna(score):
                        continue
                        
                    # Normalize if needed
                    if score > 1.0:  # If score is on 0-100 scale
                        success_scores[idx] = score / 100.0
                    else:  # If score is already on 0-1 scale
                        success_scores[idx] = score
            else:
                # If show_id is not the index, check if it's a column
                has_show_id = 'show_id' in success_data.columns
                if not has_show_id:
                    st.error("DEBUG ERROR: 'show_id' column missing and not set as index in SuccessAnalyzer data")
                    raise ValueError("'show_id' column or index required for success score mapping")
                
                st.write("DEBUG: Using 'show_id' column for mapping success scores")
                for _, row in success_data.iterrows():
                    # Normalize success score to 0-1 range if it's on a 0-100 scale
                    # Check if the score is already normalized (0-1) or needs normalization (0-100)
                    score = row['success_score']
                    
                    # Check for NaN or None values
                    if pd.isna(score):
                        continue
                        
                    # Normalize if needed
                    if score > 1.0:  # If score is on 0-100 scale
                        success_scores[row['show_id']] = score / 100.0
                    else:  # If score is already on 0-1 scale
                        success_scores[row['show_id']] = score
            
            if not success_scores:
                st.error("DEBUG ERROR: No valid success scores found in SuccessAnalyzer data")
                raise ValueError("Success scores could not be extracted from SuccessAnalyzer data")
                
            st.write(f"DEBUG: Extracted {len(success_scores)} success scores from SuccessAnalyzer data")
            
            # Apply success metrics to criteria data
            # The criteria data uses 'id' while success data uses 'show_id' as index
            # They are the same value, just different column names/structure
            # Create a copy of success_data with show_id as a column (from index)
            success_data_with_id = success_data.reset_index()
            
            # Get required columns for component score calculations
            required_columns = ['show_id', 'success_score', 'popcornmeter', 'tomatometer', 
                                'tmdb_seasons', 'tmdb_episodes', 'tmdb_status']
            
            # Check which required columns are available in success data
            # First ensure show_id is always included since it might be the index
            if 'show_id' not in success_data_with_id.columns and success_data_with_id.index.name == 'show_id':
                success_data_with_id = success_data_with_id.reset_index()
                
            available_columns = [col for col in required_columns if col in success_data_with_id.columns]
            missing_columns = [col for col in required_columns if col not in success_data_with_id.columns]
            
            st.write(f"DEBUG: Available success metrics columns: {available_columns}")
            if missing_columns:
                st.write(f"DEBUG WARNING: Missing success metrics columns: {missing_columns}")
                st.write("DEBUG: This may affect component score calculations")
            
            # Always ensure show_id and success_score are included
            if 'show_id' not in available_columns:
                st.error("DEBUG ERROR: 'show_id' column missing from success data after reset_index")
                # This shouldn't happen if reset_index worked correctly, but just in case
                if hasattr(success_data_with_id.index, 'name') and success_data_with_id.index.name == 'show_id':
                    success_data_with_id = success_data_with_id.reset_index()
                else:
                    # Last resort - create show_id from index
                    success_data_with_id['show_id'] = success_data_with_id.index
                available_columns.append('show_id')
            
            if 'success_score' not in available_columns:
                st.error("DEBUG ERROR: 'success_score' column missing from success data")
                raise ValueError("'success_score' column required for success score calculation")
            
            # Check if we have at least one audience metric (popcornmeter)
            if 'popcornmeter' not in available_columns:
                st.write("DEBUG WARNING: 'popcornmeter' column missing from success data")
                st.write("DEBUG: Will use success_score as fallback for audience score calculation")
            
            # Check if we have at least one critics metric (tomatometer)
            if 'tomatometer' not in available_columns:
                st.write("DEBUG WARNING: 'tomatometer' column missing from success data")
                st.write("DEBUG: Will use success_score as fallback for critics score calculation")
            
            # Check if we have longevity metrics
            longevity_columns = ['tmdb_seasons', 'tmdb_status']
            missing_longevity = [col for col in longevity_columns if col not in available_columns]
            if missing_longevity:
                st.write(f"DEBUG WARNING: Missing longevity metrics: {missing_longevity}")
                st.write("DEBUG: Will use success_score as fallback for longevity score calculation")
            
            # Merge success metrics into criteria data
            st.write("DEBUG: Merging success metrics into criteria data")
            st.write(f"DEBUG: Criteria data before merge: {len(self.criteria_data)} rows")
            
            # Use explicit suffixes to better track duplicate columns
            self.criteria_data = pd.merge(
                self.criteria_data,
                success_data_with_id[available_columns],
                left_on='id',
                right_on='show_id',
                how='left',
                suffixes=('_orig', '_success')
            )
            
            st.write(f"DEBUG: Criteria data after merge: {len(self.criteria_data)} rows")
            st.write(f"DEBUG: Merge success: {self.criteria_data['success_score'].notna().sum()} rows with success scores")
            
            # Log the merge results for each key column
            for col in available_columns:
                if col in self.criteria_data.columns:
                    non_null_count = self.criteria_data[col].notna().sum()
                    st.write(f"DEBUG: '{col}' column has {non_null_count} non-null values after merge")
                    if non_null_count == 0:
                        st.error(f"DEBUG ERROR: '{col}' column has all null values after merge")
                else:
                    st.error(f"DEBUG ERROR: '{col}' column missing from merged data")
            
            # Check for duplicate columns that might have been created during the merge
            duplicate_cols = [col for col in self.criteria_data.columns if col.endswith('_orig') or col.endswith('_success')]
            if duplicate_cols:
                st.write(f"DEBUG WARNING: Duplicate columns created during merge: {duplicate_cols}")
                # Clean up duplicate columns by keeping the non-null version
                processed_base_cols = set()  # Track which base columns we've already processed
                
                for col in duplicate_cols:
                    if col.endswith('_orig'):
                        base_col = col[:-5]  # Remove _orig suffix
                        success_col = f"{base_col}_success"
                        
                        # Skip if we've already processed this base column
                        if base_col in processed_base_cols:
                            continue
                        processed_base_cols.add(base_col)
                        
                        # Only process if both versions exist
                        if success_col in self.criteria_data.columns:
                            # Keep the version with more non-null values
                            orig_count = self.criteria_data[col].notna().sum()
                            success_count = self.criteria_data[success_col].notna().sum()
                            
                            # Create consolidated column
                            if success_count > 0:  # Prefer success data when available
                                self.criteria_data[base_col] = self.criteria_data[success_col].fillna(self.criteria_data[col])
                                st.write(f"DEBUG: Using '{success_col}' (filled with '{col}') for '{base_col}' ({success_count} success values, {orig_count} orig values)")
                            else:
                                self.criteria_data[base_col] = self.criteria_data[col]
                                st.write(f"DEBUG: Using '{col}' for '{base_col}' ({orig_count} non-null values)")
                            
                            # Drop the duplicate columns
                            self.criteria_data = self.criteria_data.drop([col, success_col], axis=1, errors='ignore')
                
                # Verify that all required columns exist after cleanup
                for col in required_columns:
                    if col not in self.criteria_data.columns:
                        st.error(f"DEBUG ERROR: Required column '{col}' missing after duplicate cleanup")
                        # Try to recover from suffix columns if they still exist
                        orig_col = f"{col}_orig"
                        success_col = f"{col}_success"
                        
                        if orig_col in self.criteria_data.columns:
                            self.criteria_data[col] = self.criteria_data[orig_col]
                            st.write(f"DEBUG: Recovered '{col}' from '{orig_col}'")
                            self.criteria_data = self.criteria_data.drop([orig_col], axis=1, errors='ignore')
                        elif success_col in self.criteria_data.columns:
                            self.criteria_data[col] = self.criteria_data[success_col]
                            st.write(f"DEBUG: Recovered '{col}' from '{success_col}'")
                            self.criteria_data = self.criteria_data.drop([success_col], axis=1, errors='ignore')
            
            # Drop rows with missing success scores to make issues visible
            missing_scores = self.criteria_data['success_score'].isna().sum()
            if missing_scores > 0:
                st.write(f"DEBUG: Dropping {missing_scores} shows with missing success scores")
                self.criteria_data = self.criteria_data.dropna(subset=['success_score'])
                
            # Log the columns in the merged data
            st.write(f"DEBUG: Merged criteria data columns: {list(self.criteria_data.columns)}")
            st.write(f"DEBUG: Merged data has {len(self.criteria_data)} rows")
            
            # Check if we have the required metrics columns
            for col in ['popcornmeter', 'tomatometer']:
                if col not in self.criteria_data.columns:
                    st.write(f"DEBUG WARNING: '{col}' column missing from merged data")
                else:
                    st.write(f"DEBUG: '{col}' column present in merged data")
                    st.write(f"DEBUG: '{col}' non-null count: {self.criteria_data[col].notna().sum()}")
            
            # Cache the criteria data
            self._criteria_data = self.criteria_data
            
            # Update the last update timestamp
            self.last_update = current_time
        
        return self.criteria_data
    
    def _get_matching_shows(self, criteria: Dict[str, Any]) -> pd.DataFrame:
        """Get shows matching the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            DataFrame of matching shows with success metrics
        """
        import streamlit as st
        
        # Fetch the latest criteria data
        data = self.fetch_criteria_data()
        if data.empty:
            st.error("DEBUG ERROR: Empty criteria data from fetch_criteria_data")
            raise ValueError("No criteria data available")
        
        st.write(f"DEBUG: Available columns in data: {list(data.columns)}")
        
        # Log array fields in criteria
        array_fields = ['character_types', 'plot_elements', 'thematic_elements', 'team_members', 'subgenres', 'studios']
        
        for field_name in criteria.keys():
            if field_name in array_fields:
                st.write(f"DEBUG: Processing array field '{field_name}' with value {criteria[field_name]}")
                # Let field_manager handle the mapping
        
        # Use FieldManager to match shows against criteria
        matched_shows, match_count = self.field_manager.match_shows(criteria, data)
        
        if matched_shows.empty:
            st.error(f"DEBUG ERROR: No shows matched the criteria {criteria}")
            return matched_shows, 0
            
        st.write(f"DEBUG: Found {match_count} shows matching the criteria")
        return matched_shows, match_count   
    def _calculate_success_rate(self, shows: pd.DataFrame, threshold: float = 0.6) -> float:
        """Calculate the success rate for a set of shows.
        
        Args:
            shows: DataFrame of shows
            threshold: Success threshold (shows with score >= threshold are considered successful)
            
        Returns:
            Success rate (0-1)
        """
        import streamlit as st
        
        if shows.empty:
            st.error("DEBUG ERROR: Empty shows DataFrame provided to _calculate_success_rate")
            raise ValueError("Cannot calculate success rate with empty shows DataFrame")
        
        if 'success_score' not in shows.columns:
            st.error("DEBUG ERROR: 'success_score' column missing from shows data")
            raise ValueError("'success_score' column required for success rate calculation")
        
        # Filter out shows with missing success scores
        shows_with_scores = shows[shows['success_score'].notna()]
        
        if len(shows_with_scores) == 0:
            st.error("DEBUG ERROR: No shows with valid success scores found")
            raise ValueError("No shows with valid success scores available")
        
        # Get success score range and distribution for debugging
        min_score = shows_with_scores['success_score'].min()
        max_score = shows_with_scores['success_score'].max()
        mean_score = shows_with_scores['success_score'].mean()
        
        st.write(f"DEBUG: Success score range: {min_score} to {max_score}, mean: {mean_score}")
        
        # Normalize threshold if scores are on 0-100 scale
        normalized_threshold = threshold
        normalized_scores = shows_with_scores['success_score'].copy()
        
        # Check if scores need normalization (0-100 scale)
        if max_score > 1.0:  # If scores are on 0-100 scale
            normalized_threshold = threshold * 100
        else:  # If scores are already on 0-1 scale but threshold is on 0-100 scale
            if threshold > 1.0:
                normalized_threshold = threshold / 100
        
        # Count successful shows (those with score >= threshold)
        successful = shows_with_scores[shows_with_scores['success_score'] >= normalized_threshold]
        success_count = len(successful)
        total_count = len(shows_with_scores)
        
        st.write(f"DEBUG: Success rate calculation - {success_count} successful shows out of {total_count} total (threshold: {threshold})")
        
        success_rate = success_count / total_count
        st.write(f"DEBUG: Success rate: {success_rate:.4f} ({success_rate*100:.1f}%)")
        
        return success_rate
    
    @lru_cache(maxsize=32)
    def calculate_network_scores(self, criteria_key: str) -> List[NetworkMatch]:
        """Calculate network compatibility and success scores for criteria.
        
        Args:
            criteria_key: String representation of criteria dictionary for caching
            
        Returns:
            List of NetworkMatch objects sorted by success probability
        """
        # Convert string key back to dictionary
        criteria = eval(criteria_key)
        
        # Get matching shows for the criteria
        matching_shows = self._get_matching_shows(criteria)
        if matching_shows.empty:
            return []
        
        # Get all networks
        networks = self.field_manager.get_options('network')
        network_matches = []
        
        # Calculate scores for each network
        for network in networks:
            network_id = network.id
            network_name = network.name
            
            # Get shows on this network
            network_shows = matching_shows[matching_shows['network_id'] == network_id]
            sample_size = len(network_shows)
            
            # Skip networks with insufficient data
            if sample_size < OptimizerConfig.CONFIDENCE['minimum_sample']:
                continue
            
            # Calculate success rate for this network
            success_rate = self._calculate_success_rate(network_shows)
            
            # Calculate compatibility score
            # This measures how well the network aligns with the criteria
            # We use the network's historical success rate with similar shows
            compatibility_score = 0.0
            
            # Get the network's typical shows
            all_network_shows = self.criteria_data[self.criteria_data['network_id'] == network_id]
            
            if not all_network_shows.empty:
                # For each criteria, calculate how often the network produces shows with that criteria
                weights = {}
                for field, value in criteria.items():
                    if field in self.field_manager.FIELD_CONFIGS:
                        config = self.field_manager.FIELD_CONFIGS[field]
                        
                        if config.is_array:
                            # For array fields, check what percentage of the network's shows have this value
                            if isinstance(value, list):
                                field_matches = all_network_shows[all_network_shows[field].apply(
                                    lambda x: isinstance(x, list) and any(v in x for v in value)
                                )]
                            else:
                                field_matches = all_network_shows[all_network_shows[field].apply(
                                    lambda x: isinstance(x, list) and value in x
                                )]
                        else:
                            # For scalar fields, check exact matches
                            field_id = f"{field}_id" if f"{field}_id" in all_network_shows.columns else field
                            if isinstance(value, list):
                                field_matches = all_network_shows[all_network_shows[field_id].isin(value)]
                            else:
                                field_matches = all_network_shows[all_network_shows[field_id] == value]
                        
                        # Calculate weight based on how common this criteria is for the network
                        weight = len(field_matches) / len(all_network_shows) if len(all_network_shows) > 0 else 0
                        weights[field] = weight * OptimizerConfig.get_criteria_weight(field)
                
                # Calculate weighted compatibility score
                total_weight = sum(OptimizerConfig.get_criteria_weight(field) for field in criteria.keys() 
                                  if field in self.field_manager.FIELD_CONFIGS)
                
                if total_weight > 0:
                    compatibility_score = sum(weights.values()) / total_weight
            
            # Calculate confidence level
            confidence = OptimizerConfig.get_confidence_level(sample_size)
            
            # Create NetworkMatch object
            network_match = NetworkMatch(
                network_id=network_id,
                network_name=network_name,
                compatibility_score=compatibility_score,
                success_probability=success_rate,
                sample_size=sample_size,
                confidence=confidence
            )
            
            network_matches.append(network_match)
        
        # Sort by success probability (descending)
        network_matches.sort(key=lambda x: x.success_probability, reverse=True)
        
        return network_matches
    
    def calculate_component_scores(self, criteria: Dict[str, Any]) -> Dict[str, ComponentScore]:
        """Calculate component scores for a set of criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary mapping component names to ComponentScore objects
        """
        import streamlit as st
        
        st.write("DEBUG: Starting calculate_component_scores")
        
        # Get matching shows
        st.write("DEBUG: Getting matching shows for component scores")
        matching_shows = self._get_matching_shows(criteria)
        
        if matching_shows.empty:
            st.error("DEBUG ERROR: No matching shows found for the given criteria")
            raise ValueError("No matching shows found for the given criteria")
        
        st.write(f"DEBUG: Found {len(matching_shows)} matching shows for component scores")
        st.write(f"DEBUG: Matching shows columns: {list(matching_shows.columns)}")
        
        # Check for required success metrics columns
        required_metrics = ['success_score']
        audience_metrics = ['popcornmeter']
        critics_metrics = ['tomatometer']
        longevity_metrics = ['tmdb_seasons', 'tmdb_status']
        
        # Check which metrics are available
        missing_required = [col for col in required_metrics if col not in matching_shows.columns]
        missing_audience = [col for col in audience_metrics if col not in matching_shows.columns]
        missing_critics = [col for col in critics_metrics if col not in matching_shows.columns]
        missing_longevity = [col for col in longevity_metrics if col not in matching_shows.columns]
        
        # Log available metrics
        st.write(f"DEBUG: Required metrics status: {['Missing: ' + m for m in missing_required] if missing_required else 'All available'}")
        st.write(f"DEBUG: Audience metrics status: {['Missing: ' + m for m in missing_audience] if missing_audience else 'All available'}")
        st.write(f"DEBUG: Critics metrics status: {['Missing: ' + m for m in missing_critics] if missing_critics else 'All available'}")
        st.write(f"DEBUG: Longevity metrics status: {['Missing: ' + m for m in missing_longevity] if missing_longevity else 'All available'}")
        
        # Initialize component scores dictionary
        component_scores = {}
        
        # Calculate each component score with individual error handling
        # Audience score
        try:
            st.write("DEBUG: Calculating audience score")
            audience_score = self._calculate_audience_score(matching_shows)
            st.write(f"DEBUG: Audience score result: {audience_score}")
            component_scores['audience'] = audience_score
        except Exception as e:
            st.error(f"DEBUG ERROR: Error calculating audience score: {str(e)}")
            import traceback
            st.error(f"DEBUG ERROR: Audience score traceback: {traceback.format_exc()}")
            # Continue with other scores instead of failing completely
        
        # Critics score
        try:
            st.write("DEBUG: Calculating critics score")
            critics_score = self._calculate_critics_score(matching_shows)
            st.write(f"DEBUG: Critics score result: {critics_score}")
            component_scores['critics'] = critics_score
        except Exception as e:
            st.error(f"DEBUG ERROR: Error calculating critics score: {str(e)}")
            import traceback
            st.error(f"DEBUG ERROR: Critics score traceback: {traceback.format_exc()}")
            # Continue with other scores
        
        # Longevity score
        try:
            st.write("DEBUG: Calculating longevity score")
            longevity_score = self._calculate_longevity_score(matching_shows)
            st.write(f"DEBUG: Longevity score result: {longevity_score}")
            component_scores['longevity'] = longevity_score
        except Exception as e:
            st.error(f"DEBUG ERROR: Error calculating longevity score: {str(e)}")
            import traceback
            st.error(f"DEBUG ERROR: Longevity score traceback: {traceback.format_exc()}")
            # Continue with other scores
        
        # Check if we have at least one component score
        if not component_scores:
            st.error("DEBUG ERROR: Failed to calculate any component scores")
            raise ValueError("Failed to calculate any component scores")
        
        # If we're missing any component scores but have success_score, create fallback scores
        if 'success_score' in matching_shows.columns:
            if 'audience' not in component_scores:
                st.write("DEBUG: Creating fallback audience score from success_score")
                component_scores['audience'] = self._calculate_audience_score_from_success(matching_shows)
                
            if 'critics' not in component_scores:
                st.write("DEBUG: Creating fallback critics score from success_score")
                component_scores['critics'] = self._calculate_critics_score_from_success(matching_shows)
                
            if 'longevity' not in component_scores:
                st.write("DEBUG: Creating fallback longevity score from success_score")
                component_scores['longevity'] = self._calculate_longevity_score_from_success(matching_shows)
        
        st.write(f"DEBUG: Final component scores: {component_scores}")
        return component_scores
    
    def _calculate_audience_score(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate audience score for a set of shows.
        
        Args:
            shows: DataFrame of shows
            
        Returns:
            ComponentScore for audience
        """
        import streamlit as st
        
        if shows.empty:
            st.error("DEBUG ERROR: Empty shows DataFrame provided to _calculate_audience_score")
            raise ValueError("Cannot calculate audience score with empty shows DataFrame")
        
        # Check if popcornmeter column exists
        if 'popcornmeter' not in shows.columns:
            st.error("DEBUG ERROR: Popcornmeter column missing from shows data")
            st.error("DEBUG ERROR: Available columns: " + str(list(shows.columns)))
            
            # Try to use success_score as a fallback if available
            if 'success_score' in shows.columns:
                st.write("DEBUG: Using success_score as fallback for audience score calculation")
                # Filter shows with success metrics
                audience_shows = shows[shows['success_score'].notna()]
                sample_size = len(audience_shows)
                
                if sample_size == 0:
                    st.error("DEBUG ERROR: No shows with valid success_score data found")
                    raise ValueError("No shows with valid success metrics available for audience score")
                
                # Calculate confidence level
                confidence = OptimizerConfig.get_confidence_level(sample_size)
                
                # Use success_score as audience score (assuming it's already normalized to 0-1)
                avg_score = audience_shows['success_score'].mean()
                
                # Calculate audience engagement metrics
                details = {'success_score': avg_score}
                
                # Calculate overall audience score
                score = avg_score
                
                return ComponentScore(
                    component='audience',
                    score=score,
                    sample_size=sample_size,
                    confidence=confidence,
                    details=details
                )
            else:
                raise ValueError("No metrics available for audience score calculation")
            
        # Filter shows with audience metrics
        audience_shows = shows[shows['popcornmeter'].notna()]
        sample_size = len(audience_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid popcornmeter data found")
            # Try to use success_score as a fallback
            if 'success_score' in shows.columns:
                st.write("DEBUG: Using success_score as fallback for audience score calculation")
                return self._calculate_audience_score_from_success(shows)
            else:
                raise ValueError("No shows with valid popcornmeter data available")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Calculate average popcornmeter score (normalized to 0-1)
        avg_popcorn = audience_shows['popcornmeter'].mean() / 100
        
        # Calculate audience engagement metrics if available
        details = {'popcornmeter': avg_popcorn}
        
        # Calculate overall audience score
        score = avg_popcorn
        
        return ComponentScore(
            component='audience',
            score=score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
        
    def _calculate_audience_score_from_success(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate audience score using success_score as a fallback.
        
        Args:
            shows: DataFrame of shows
            
        Returns:
            ComponentScore for audience
        """
        import streamlit as st
        
        # Filter shows with success metrics
        audience_shows = shows[shows['success_score'].notna()]
        sample_size = len(audience_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid success_score data found")
            raise ValueError("No shows with valid success metrics available for audience score")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Use success_score as audience score (assuming it's already normalized to 0-1)
        avg_score = audience_shows['success_score'].mean()
        
        # Calculate audience engagement metrics
        details = {'success_score': avg_score}
        
        # Calculate overall audience score
        score = avg_score
        
        return ComponentScore(
            component='audience',
            score=score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
    
    def _calculate_critics_score(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate critics score for a set of shows.
        
        Args:
            shows: DataFrame of shows
            
        Returns:
            ComponentScore for critics
        """
        import streamlit as st
        
        if shows.empty:
            st.error("DEBUG ERROR: Empty shows DataFrame provided to _calculate_critics_score")
            raise ValueError("Cannot calculate critics score with empty shows DataFrame")
        
        # Check if tomatometer column exists
        if 'tomatometer' not in shows.columns:
            st.error("DEBUG ERROR: Tomatometer column missing from shows data")
            st.error("DEBUG ERROR: Available columns: " + str(list(shows.columns)))
            
            # Try to use success_score as a fallback if available
            if 'success_score' in shows.columns:
                st.write("DEBUG: Using success_score as fallback for critics score calculation")
                return self._calculate_critics_score_from_success(shows)
            else:
                raise ValueError("No metrics available for critics score calculation")
            
        # Filter shows with critics metrics
        critics_shows = shows[shows['tomatometer'].notna()]
        sample_size = len(critics_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid tomatometer data found")
            # Try to use success_score as a fallback
            if 'success_score' in shows.columns:
                st.write("DEBUG: Using success_score as fallback for critics score calculation")
                return self._calculate_critics_score_from_success(shows)
            else:
                raise ValueError("No shows with valid tomatometer data available")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Calculate average tomatometer score (normalized to 0-1)
        avg_tomato = critics_shows['tomatometer'].mean() / 100
        
        # Calculate critics metrics if available
        details = {'tomatometer': avg_tomato}
        
        # Calculate overall critics score
        score = avg_tomato
        
        return ComponentScore(
            component='critics',
            score=score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
        
    def _calculate_critics_score_from_success(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate critics score using success_score as a fallback.
        
        Args:
            shows: DataFrame of shows
            
        Returns:
            ComponentScore for critics
        """
        import streamlit as st
        
        # Filter shows with success metrics
        critics_shows = shows[shows['success_score'].notna()]
        sample_size = len(critics_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid success_score data found")
            raise ValueError("No shows with valid success metrics available for critics score")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Use success_score as critics score (assuming it's already normalized to 0-1)
        avg_score = critics_shows['success_score'].mean()
        
        # Calculate critics engagement metrics
        details = {'success_score': avg_score}
        
        # Calculate overall critics score
        score = avg_score
        
        return ComponentScore(
            component='critics',
            score=score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
        
    def _calculate_longevity_score(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate longevity score for a set of shows.
        
        Args:
            shows: DataFrame of shows
            
        Returns:
            ComponentScore for longevity
        """
        import streamlit as st
        
        if shows.empty:
            st.error("DEBUG ERROR: Empty shows DataFrame provided to _calculate_longevity_score")
            raise ValueError("Cannot calculate longevity score with empty shows DataFrame")
        
        # Check required columns
        required_columns = ['tmdb_seasons', 'tmdb_status']
        missing_columns = [col for col in required_columns if col not in shows.columns]
        
        if missing_columns:
            st.error(f"DEBUG ERROR: Missing columns for longevity calculation: {missing_columns}")
            st.error("DEBUG ERROR: Available columns: " + str(list(shows.columns)))
            
            # Try to use success_score as a fallback if available
            if 'success_score' in shows.columns:
                st.write("DEBUG: Using success_score as fallback for longevity score calculation")
                return self._calculate_longevity_score_from_success(shows)
            else:
                raise ValueError(f"Missing required columns for longevity score calculation: {missing_columns}")
        
        # Filter shows with longevity metrics
        longevity_shows = shows[shows['tmdb_seasons'].notna()]
        sample_size = len(longevity_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid tmdb_seasons data found")
            # Try to use success_score as a fallback
            if 'success_score' in shows.columns:
                st.write("DEBUG: Using success_score as fallback for longevity score calculation")
                return self._calculate_longevity_score_from_success(shows)
            else:
                raise ValueError("No shows with valid tmdb_seasons data available")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Calculate average seasons
        avg_seasons = longevity_shows['tmdb_seasons'].mean()
        
        # Calculate renewal rate (shows with > 1 season)
        renewal_rate = len(longevity_shows[longevity_shows['tmdb_seasons'] > 1]) / sample_size
        
        # Calculate multi-season rate (shows with > 2 seasons)
        multi_season_rate = len(longevity_shows[longevity_shows['tmdb_seasons'] > 2]) / sample_size
        
        # Calculate status distribution
        status_counts = longevity_shows['tmdb_status'].value_counts(normalize=True).to_dict()
        active_rate = status_counts.get('Returning Series', 0)
        
        # Calculate longevity details
        details = {
            'avg_seasons': avg_seasons / 10,  # Normalize to 0-1 (assuming 10 seasons is max)
            'renewal_rate': renewal_rate,
            'multi_season_rate': multi_season_rate,
            'active_rate': active_rate
        }
        
        # Calculate overall longevity score (weighted average of metrics)
        score = (
            0.3 * details['avg_seasons'] +
            0.3 * details['renewal_rate'] +
            0.2 * details['multi_season_rate'] +
            0.2 * details['active_rate']
        )
        
        return ComponentScore(
            component='longevity',
            score=score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
        
    def _calculate_longevity_score_from_success(self, shows: pd.DataFrame) -> ComponentScore:
        """Calculate longevity score using success_score as a fallback.
        
        Args:
            shows: DataFrame of shows
            
        Returns:
            ComponentScore for longevity
        """
        import streamlit as st
        
        # Filter shows with success metrics
        longevity_shows = shows[shows['success_score'].notna()]
        sample_size = len(longevity_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid success_score data found")
            raise ValueError("No shows with valid success metrics available for longevity score")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Use success_score as longevity score (assuming it's already normalized to 0-1)
        avg_score = longevity_shows['success_score'].mean()
        
        # Calculate basic longevity details
        details = {'success_score': avg_score}
        
        # Calculate overall longevity score
        score = avg_score
        
        return ComponentScore(
            component='longevity',
            score=score,
            sample_size=sample_size,
            confidence=confidence,
            details=details
        )
    
    def calculate_criteria_impact(self, base_criteria: Dict[str, Any]) -> Dict[str, Dict[int, float]]:
        """Calculate the impact of each criteria value on success.
        
        Args:
            base_criteria: Base criteria to calculate impact relative to
            
        Returns:
            Dictionary mapping criteria to dictionaries mapping values to impact scores
        """
        import streamlit as st
        
        st.write("DEBUG: Starting calculate_criteria_impact")
        
        # Fields that support multiple values
        array_fields = ['character_types', 'plot_elements', 'thematic_elements', 'team_members', 'subgenres', 'studios']
        # Track error counts for reporting
        error_counts = {field: 0 for field in array_fields}
        
        try:
            # Get base success rate
            st.write(f"DEBUG: Getting base shows for criteria: {base_criteria}")
            base_shows = self._get_matching_shows(base_criteria)
            
            if base_shows.empty:
                st.error("DEBUG ERROR: No matching shows found for base criteria")
                raise ValueError("No matching shows found for base criteria")
            
            st.write(f"DEBUG: Found {len(base_shows)} base shows")
            
            base_rate = self._calculate_success_rate(base_shows)
            st.write(f"DEBUG: Base success rate: {base_rate}")
            
            if base_rate == 0:
                st.error("DEBUG ERROR: Base success rate is zero")
                raise ValueError("Cannot calculate impact scores with zero base success rate")
            
            impact_scores = {}
            
            # For each criteria field, calculate impact of different values
            field_count = 0
            for field_name in self.field_manager.FIELD_CONFIGS.keys():
                try:
                    # Skip fields already in base criteria
                    if field_name in base_criteria:
                        continue
                    
                    # Handle array fields specially
                    is_array_field = field_name in array_fields
                    
                    # We'll still calculate impact for array fields, but we need to be aware
                    # that they might use different column names in the data
                    
                    field_impact = {}
                    options = self.field_manager.get_options(field_name)
                    
                    st.write(f"DEBUG: Calculating impact for field {field_name} with {len(options)} options")
                    option_count = 0
                    
                    for option in options:
                        try:
                            # Skip this option if we've had too many errors with this field
                            if error_counts.get(field_name, 0) > 3:
                                continue
                            
                            # Create a new criteria with just this option
                            new_criteria = base_criteria.copy()
                            
                            # For array fields, use the original field name in criteria, we'll transform it in _get_matching_shows
                            if is_array_field:
                                new_criteria[field_name] = [option.id]
                            else:
                                new_criteria[field_name] = option.id
                            
                            # Get success rate with this option
                            option_shows = self._get_matching_shows(new_criteria)
                            
                            # Check if we got any matching shows
                            if option_shows.empty:
                                st.error(f"DEBUG ERROR: No shows matched the criteria {new_criteria}")
                                continue
                                
                            if len(option_shows) < OptimizerConfig.CONFIDENCE['minimum_sample']:
                                st.write(f"DEBUG: Insufficient sample size for {field_name}={option.name} (id={option.id})")
                                continue
                            
                            try:
                                option_rate = self._calculate_success_rate(option_shows)
                            except Exception as e:
                                st.error(f"DEBUG ERROR: Error calculating success rate for {field_name}={option.name}: {str(e)}")
                                continue
                            
                            # Calculate impact as relative change in success rate
                            impact = (option_rate - base_rate) / base_rate
                            
                            # Store impact score
                            field_impact[option.id] = impact
                            option_count += 1
                        except Exception as e:
                            st.error(f"DEBUG ERROR: Error calculating impact for {field_name}={option.name}: {str(e)}")
                            continue
                    
                    if field_impact:
                        impact_scores[field_name] = field_impact
                        field_count += 1
                        st.write(f"DEBUG: Added {option_count} impact scores for field {field_name}")
                except Exception as e:
                    st.error(f"DEBUG ERROR: Error processing field {field_name}: {str(e)}")
                    continue
            
            # Check if we have any impact scores
            if not impact_scores:
                st.error("DEBUG ERROR: No impact scores could be calculated")
                
                # If we have a genre, we can create a placeholder impact score for visualization
                # but we clearly mark it as a placeholder
                if 'genre' in base_criteria:
                    genre_id = base_criteria['genre']
                    genre_name = self.field_manager.get_option_name('genre', genre_id)
                    st.error(f"DEBUG ERROR: Using placeholder impact score for genre {genre_name} (id={genre_id})")
                    impact_scores['genre'] = {genre_id: 0.0}  # Neutral impact (no effect)
                    st.error("DEBUG ERROR: This is a placeholder and does not represent real data")
            else:
                st.write(f"DEBUG: Successfully calculated impact scores for {field_count} fields")
                
            return impact_scores
        except Exception as e:
            st.error(f"DEBUG ERROR: Error in calculate_criteria_impact: {str(e)}")
            import traceback
            st.error(f"DEBUG ERROR: Traceback: {traceback.format_exc()}")
            
            # Return an empty dictionary instead of a default
            # This will make it clear that no impact scores could be calculated
            return {}
    
    def get_criteria_confidence(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence levels for criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary with confidence information
        """
        return self.field_manager.calculate_confidence(criteria)
