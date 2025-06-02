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
            success_data = self.success_analyzer.fetch_success_data()
            
            if success_data.empty:
                st.error("DEBUG ERROR: Empty success data returned from SuccessAnalyzer")
                raise ValueError("No success data available from SuccessAnalyzer")
            
            # Create a mapping of show_id to success_score
            success_scores = {}
            for idx, row in success_data.iterrows():
                if 'id' in row and 'success_score' in row:
                    # Normalize success score to 0-1 range (from 0-100)
                    success_scores[row['id']] = row['success_score'] / 100.0
            
            if not success_scores:
                st.error("DEBUG ERROR: No valid success scores found in SuccessAnalyzer data")
                raise ValueError("Success scores could not be extracted from SuccessAnalyzer data")
            
            # Apply success scores to criteria data
            def get_success_score(show_id):
                if not pd.notna(show_id) or show_id not in success_scores:
                    st.write(f"DEBUG: No success score found for show ID: {show_id}")
                    return None
                return success_scores[show_id]
            
            self.criteria_data['success_score'] = self.criteria_data['id'].apply(get_success_score)
            
            # Drop rows with missing success scores to make issues visible
            missing_scores = self.criteria_data['success_score'].isna().sum()
            if missing_scores > 0:
                st.write(f"DEBUG: Dropping {missing_scores} shows with missing success scores")
                self.criteria_data = self.criteria_data.dropna(subset=['success_score'])
            
            self.last_update = current_time
        
        return self.criteria_data
    
    def _get_matching_shows(self, criteria: Dict[str, Any]) -> pd.DataFrame:
        """Get shows matching the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            DataFrame of matching shows with success metrics
        """
        # Fetch the latest criteria data
        data = self.fetch_criteria_data()
        if data.empty:
            return pd.DataFrame()
        
        # Use FieldManager to match shows against criteria
        matched_shows, _ = self.field_manager.match_shows(criteria, data)
        return matched_shows
    
    def _calculate_success_rate(self, shows: pd.DataFrame, threshold: float = 0.6) -> float:
        """Calculate the success rate for a set of shows.
        
        Args:
            shows: DataFrame of shows
            threshold: Success threshold (shows with score >= threshold are considered successful)
            
        Returns:
            Success rate (0-1)
        """
        if shows.empty:
            return 0.0
        
        # Count shows with success score above threshold
        successful = shows[shows['success_score'] >= threshold]
        return len(successful) / len(shows)
    
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
        """Calculate success scores for different components.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary mapping components to ComponentScore objects
        """
        import streamlit as st
        
        # Get matching shows for the criteria
        matching_shows = self._get_matching_shows(criteria)
        if matching_shows.empty:
            logger.error("No matching shows found for criteria")
            st.error("No matching shows found for the selected criteria")
            raise ValueError("No matching shows found for criteria")
        
        component_scores = {}
        
        # Calculate audience score
        audience_score = self._calculate_audience_score(matching_shows)
        component_scores['audience'] = audience_score
        
        # Calculate critics score
        critics_score = self._calculate_critics_score(matching_shows)
        component_scores['critics'] = critics_score
        
        # Calculate longevity score
        longevity_score = self._calculate_longevity_score(matching_shows)
        component_scores['longevity'] = longevity_score
        
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
            raise ValueError("Popcornmeter column required for audience score calculation")
            
        # Filter shows with audience metrics
        audience_shows = shows[shows['popcornmeter'].notna()]
        sample_size = len(audience_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid popcornmeter data found")
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
            raise ValueError("Tomatometer column required for critics score calculation")
            
        # Filter shows with critics metrics
        critics_shows = shows[shows['tomatometer'].notna()]
        sample_size = len(critics_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid tomatometer data found")
            raise ValueError("No shows with valid tomatometer data available")
        
        # Calculate confidence level
        confidence = OptimizerConfig.get_confidence_level(sample_size)
        
        # Calculate average tomatometer score (normalized to 0-1)
        avg_tomato = critics_shows['tomatometer'].mean() / 100
        
        # Calculate critics engagement metrics if available
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
        for column in required_columns:
            if column not in shows.columns:
                st.error(f"DEBUG ERROR: {column} column missing from shows data")
                raise ValueError(f"{column} column required for longevity score calculation")
        
        # Filter shows with longevity metrics
        longevity_shows = shows[shows['tmdb_seasons'].notna()]
        sample_size = len(longevity_shows)
        
        if sample_size == 0:
            st.error("DEBUG ERROR: No shows with valid tmdb_seasons data found")
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
    
    def calculate_criteria_impact(self, base_criteria: Dict[str, Any]) -> Dict[str, Dict[int, float]]:
        """Calculate the impact of each criteria value on success.
        
        Args:
            base_criteria: Base criteria to calculate impact relative to
            
        Returns:
            Dictionary mapping criteria to dictionaries mapping values to impact scores
        """
        import streamlit as st
        
        # Known problematic field types that need special handling
        array_fields = ['character_types', 'plot_elements', 'thematic_elements', 'team_members']
        error_counts = {field: 0 for field in array_fields}
        
        try:
            # Get base success rate
            base_shows = self._get_matching_shows(base_criteria)
            if base_shows.empty:
                return {}
            
            base_rate = self._calculate_success_rate(base_shows)
            if base_rate == 0:
                return {}
            
            impact_scores = {}
            
            # For each criteria field, calculate impact of different values
            for field_name in self.field_manager.FIELD_CONFIGS.keys():
                try:
                    # Skip fields already in base criteria
                    if field_name in base_criteria:
                        continue
                    
                    # Skip array fields that are known to cause errors
                    if field_name in array_fields:
                        continue
                    
                    field_impact = {}
                    options = self.field_manager.get_options(field_name)
                    
                    for option in options:
                        try:
                            # Create new criteria with this option added
                            new_criteria = base_criteria.copy()
                            new_criteria[field_name] = option.id
                            
                            # Get success rate with this option
                            option_shows = self._get_matching_shows(new_criteria)
                            if len(option_shows) < OptimizerConfig.CONFIDENCE['minimum_sample']:
                                continue
                            
                            option_rate = self._calculate_success_rate(option_shows)
                            
                            # Calculate impact as relative change in success rate
                            impact = (option_rate - base_rate) / base_rate
                            
                            # Store impact score
                            field_impact[option.id] = impact
                        except Exception:
                            # Silent error handling for individual options
                            continue
                    
                    if field_impact:
                        impact_scores[field_name] = field_impact
                except Exception as e:
                    # Silent error handling for all fields
                    continue
            
            # If we couldn't calculate any impacts, add a default one for genre
            if not impact_scores and 'genre' in base_criteria:
                genre_id = base_criteria['genre']
                impact_scores['genre'] = {genre_id: 0.5}  # Default middle impact
                
            return impact_scores
        except Exception as e:
            # Return a minimal impact score dictionary
            if 'genre' in base_criteria:
                genre_id = base_criteria['genre']
                return {'genre': {genre_id: 0.5}}  # Default middle impact
            return {}
    
    def get_criteria_confidence(self, criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence levels for criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary with confidence information
        """
        return self.field_manager.calculate_confidence(criteria)
