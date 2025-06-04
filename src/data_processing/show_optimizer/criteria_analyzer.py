"""CriteriaAnalyzer: Analyzes success patterns across criteria.

This component analyzes success patterns across all criteria, identifies which
criteria contribute most to success, and evaluates show-network alignment.
It builds on the raw scoring calculations from CriteriaScorer.

Key concepts:

1. Pattern Recognition:
   - Analyze success patterns across all criteria
   - Calculate overall success rates for each content element
   - Calculate network-specific success rates for each criteria
   - Identify which criteria contribute most to success

2. Network Analysis:
   - Evaluate show-network alignment scores
   - Rank networks by success probability
   - Group networks into tiers based on compatibility

3. Performance:
   - Cache analysis results
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
from .criteria_scorer import CriteriaScorer, NetworkMatch, ComponentScore

logger = logging.getLogger(__name__)


@dataclass
class NetworkTier:
    """A tier of networks with similar compatibility."""
    tier_name: str  # e.g., "Excellent Match", "Good Match", "Fair Match"
    networks: List[NetworkMatch]
    min_score: float
    max_score: float


@dataclass
class SuccessFactor:
    """A success factor identified from analysis."""
    criteria_type: str  # e.g., "genre", "character_types"
    criteria_value: int  # ID of the value
    criteria_name: str   # Display name
    impact_score: float  # Impact on success (-1 to 1)
    confidence: str      # none, low, medium, high
    sample_size: int
    matching_titles: List[str] = field(default_factory=list)  # List of show titles matching this criteria


class CriteriaAnalyzer:
    """Analyzes success patterns across criteria."""
    
    def __init__(self, shows_analyzer: ShowsAnalyzer, success_analyzer: SuccessAnalyzer):
        """Initialize the criteria analyzer.
        
        Args:
            shows_analyzer: ShowsAnalyzer instance for show data
            success_analyzer: SuccessAnalyzer instance for success metrics
        """
        self.shows_analyzer = shows_analyzer
        self.success_analyzer = success_analyzer
        self.criteria_scorer = CriteriaScorer(shows_analyzer, success_analyzer)
        self.last_analysis = None
        self.cache_duration = OptimizerConfig.PERFORMANCE['cache_duration']
        
    def find_matching_networks(self, criteria: Dict[str, Any], 
                              min_confidence: str = 'low') -> Dict[str, List[NetworkMatch]]:
        """Find networks matching the given criteria, grouped into tiers.
        
        Args:
            criteria: Dictionary of criteria
            min_confidence: Minimum confidence level to include (none, low, medium, high)
            
        Returns:
            Dictionary mapping tier names to lists of NetworkMatch objects
        """
        # Get network matches from CriteriaScorer
        network_matches = self.criteria_scorer.calculate_network_scores(str(criteria))
        
        # Filter by confidence
        confidence_levels = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
        min_confidence_level = confidence_levels.get(min_confidence, 0)
        
        filtered_matches = [
            match for match in network_matches
            if confidence_levels.get(match.confidence, 0) >= min_confidence_level
        ]
        
        if not filtered_matches:
            return {}
        
        # Define tier thresholds from OptimizerConfig
        tier_thresholds = OptimizerConfig.NETWORK_TIERS
        
        # Group networks into tiers
        tiers = {}
        for tier_name, threshold in tier_thresholds.items():
            tier_matches = [
                match for match in filtered_matches
                if match.success_probability >= threshold
            ]
            
            # Remove networks already in higher tiers
            for existing_tier in tiers.values():
                tier_matches = [m for m in tier_matches if m not in existing_tier]
                
            if tier_matches:
                tiers[tier_name] = tier_matches
        
        return tiers
    
    def analyze_components(self, criteria: Dict[str, Any]) -> Dict[str, ComponentScore]:
        """Analyze components (audience, critics, longevity) for the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary mapping component names to ComponentScore objects
        """
        return self.criteria_scorer.calculate_component_scores(criteria)
    
    def identify_success_factors(self, criteria: Dict[str, Any], limit: int = 10) -> List[SuccessFactor]:
        """Identify the factors that contribute most to success.
        
        Args:
            criteria: Criteria defining the show concept
            limit: Maximum number of factors to return
            
        Returns:
            List of SuccessFactor objects sorted by impact
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Convert to SuccessFactor objects
        success_factors = []
        
        try:
            # Calculate criteria impact from CriteriaScorer
            impact_data = self.criteria_scorer.calculate_criteria_impact(criteria)
            
            # Process each criteria type
            for criteria_type, values in impact_data.items():
                # Process at most 5 values per criteria type to reduce processing
                processed_count = 0
                
                # Convert any unhashable keys to strings first
                hashable_values = {}
                for k, v in values.items():
                    if isinstance(k, (dict, list)):
                        hashable_values[str(k)] = v
                    else:
                        hashable_values[k] = v
                
                for value_id, impact_data in hashable_values.items():
                    if processed_count >= 5:  # Limit processing per criteria type
                        break
                        
                    try:
                        # Extract impact and sample size from the impact data
                        # Handle both old format (float) and new format (dict with impact and sample_size)
                        if isinstance(impact_data, dict) and 'impact' in impact_data:
                            impact = impact_data['impact']
                            sample_size = impact_data.get('sample_size', 0)
                        else:
                            # Backward compatibility for old format
                            impact = impact_data
                            sample_size = 0
                        
                        # Check if value_id is a dictionary or other unhashable type
                        if isinstance(value_id, (dict, list)):
                            # Convert to a string representation for unhashable types
                            # This is a workaround for the unhashable type error
                            value_id_str = str(value_id)
                            name = value_id_str
                            # Store the string representation for use in the SuccessFactor
                            criteria_value = value_id_str
                        else:
                            # Get the name for this criteria value
                            field_manager = self.criteria_scorer.field_manager
                            options = field_manager.get_options(criteria_type)
                            
                            # Find the option with this ID
                            name = str(value_id)  # Default if not found
                            for option in options:
                                if option.id == value_id:
                                    name = option.name
                                    break
                            
                            # For hashable types, use the original value
                            criteria_value = value_id
                        
                        # Set confidence based on sample size
                        confidence = "low"
                        if sample_size >= 30:
                            confidence = "high"
                        elif sample_size >= 10:
                            confidence = "medium"
                        
                        # Get matching titles for this criteria
                        matching_titles = []
                        try:
                            # Create a single-criteria dictionary for this factor
                            single_criteria = {criteria_type: criteria_value}
                            
                            # Get matching shows for just this criteria
                            matching_shows, _ = self.criteria_scorer._get_matching_shows(single_criteria)
                            
                            # Extract titles if available
                            if not matching_shows.empty and 'title' in matching_shows.columns:
                                matching_titles = matching_shows['title'].tolist()
                                # Limit to 100 titles
                                if len(matching_titles) > 100:
                                    matching_titles = matching_titles[:100]
                        except Exception as e:
                            # Silent error handling for title extraction
                            logger.warning(f"Could not extract matching titles for {criteria_type}={name}: {str(e)}")
                                
                        factor = SuccessFactor(
                            criteria_type=criteria_type,
                            criteria_value=criteria_value,
                            criteria_name=name,
                            impact_score=impact,
                            confidence=confidence,
                            sample_size=sample_size,
                            matching_titles=matching_titles
                        )
                        success_factors.append(factor)
                        processed_count += 1
                    except Exception:
                        # Silent error handling for individual values
                        continue
        except Exception as e:
            # Only log the main error, not individual processing errors
            logger.error(f"Error identifying success factors: {e}")
            # Create a default success factor as fallback
            if 'genre' in criteria:
                genre_id = criteria['genre']
                genre_name = "Unknown Genre"
                try:
                    field_manager = self.criteria_scorer.field_manager
                    options = field_manager.get_options('genre')
                    for option in options:
                        if option.id == genre_id:
                            genre_name = option.name
                            break
                except Exception:
                    pass
                
                default_factor = SuccessFactor(
                    criteria_type="genre",
                    criteria_value=genre_id,
                    criteria_name=genre_name,
                    impact_score=0.5,  # Default middle impact
                    confidence="low",
                    sample_size=0,
                    matching_titles=[]
                )
                success_factors = [default_factor]
        
        # Sort by absolute impact (both positive and negative factors are important)
        success_factors.sort(key=lambda x: abs(x.impact_score), reverse=True)
        
        # Return top factors
        return success_factors[:limit]
    
    def rank_networks_by_compatibility(self, criteria: Dict[str, Any], 
                                     limit: int = 10) -> List[NetworkMatch]:
        """Rank networks by compatibility with the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            limit: Maximum number of networks to return
            
        Returns:
            List of NetworkMatch objects sorted by compatibility score
        """
        # Get network matches from CriteriaScorer
        network_matches = self.criteria_scorer.calculate_network_scores(str(criteria))
        
        # Sort by compatibility score
        network_matches.sort(key=lambda x: x.compatibility_score, reverse=True)
        
        # Return top networks
        return network_matches[:limit]
    
    def analyze_criteria_confidence(self, criteria: Dict[str, Any]) -> Dict[str, str]:
        """Analyze confidence levels for each criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary mapping criteria names to confidence levels
        """
        return self.criteria_scorer.get_criteria_confidence(criteria)
    
    def get_network_specific_success_rates(self, criteria: Dict[str, Any], 
                                         network_id: int) -> Dict[str, float]:
        """Get network-specific success rates for each criteria.
        
        Args:
            criteria: Dictionary of criteria
            network_id: ID of the network to analyze
            
        Returns:
            Dictionary mapping criteria names to success rates
        """
        # Fetch criteria data
        data = self.criteria_scorer.fetch_criteria_data()
        if data.empty:
            return {}
        
        # Filter to this network
        network_data = data[data['network_id'] == network_id]
        if network_data.empty:
            return {}
        
        # Calculate success rates for each criteria
        success_rates = {}
        
        for field_name, value in criteria.items():
            if field_name not in self.criteria_scorer.field_manager.FIELD_CONFIGS:
                continue
                
            config = self.criteria_scorer.field_manager.FIELD_CONFIGS[field_name]
            
            # Filter shows with this criteria
            if config.is_array:
                # Use the same array field mapping as in match_shows
                array_field_mapping = self.criteria_scorer.field_manager.get_array_field_mapping()
                
                # Get the correct column name for this field
                if field_name in array_field_mapping:
                    field_id = array_field_mapping[field_name]
                else:
                    # If not in mapping, skip this field
                    import streamlit as st
                    st.error(f"ERROR: Field '{field_name}' not found in array field mapping")
                    success_rates[field_name] = 0.0
                    continue
                
                # Check if the column exists in the DataFrame
                if field_id not in network_data.columns:
                    import streamlit as st
                    st.error(f"ERROR: Column '{field_id}' for field '{field_name}' not found in data")
                    success_rates[field_name] = 0.0
                    continue
                
                # Check data format (list or scalar)
                sample = network_data[field_id].iloc[0] if not network_data.empty else None
                
                if isinstance(value, list):
                    # Multiple values
                    value_set = set(value)
                    if isinstance(sample, list):
                        # If column contains lists, use intersection
                        matching_shows = network_data[network_data[field_id].apply(
                            lambda x: isinstance(x, list) and bool(value_set.intersection(x))
                        )]
                    else:
                        # If column doesn't contain lists, use isin
                        matching_shows = network_data[network_data[field_id].isin(value)]
                else:
                    # Single value
                    if isinstance(sample, list):
                        # If column contains lists, check if value is in list
                        matching_shows = network_data[network_data[field_id].apply(
                            lambda x: isinstance(x, list) and value in x
                        )]
                    else:
                        # If column doesn't contain lists, use equality
                        matching_shows = network_data[network_data[field_id] == value]
            else:
                # Scalar field
                field_id = f"{field_name}_id" if f"{field_name}_id" in network_data.columns else field_name
                if isinstance(value, list):
                    matching_shows = network_data[network_data[field_id].isin(value)]
                else:
                    matching_shows = network_data[network_data[field_id] == value]
            
            # Calculate success rate and sample size
            if not matching_shows.empty:
                # Calculate success rate using threshold from OptimizerConfig
                from .optimizer_config import OptimizerConfig
                success_threshold = OptimizerConfig.THRESHOLDS['success_threshold']
                success_count = matching_shows[matching_shows['success_score'] >= success_threshold].shape[0]
                total_count = matching_shows.shape[0]
                
                # Get matching show titles (up to 100)
                matching_titles = []
                if 'title' in matching_shows.columns:
                    matching_titles = matching_shows['title'].tolist()
                    # Limit to 100 titles
                    if len(matching_titles) > 100:
                        matching_titles = matching_titles[:100]
                
                if total_count > 0:
                    success_rate = success_count / total_count
                    success_rates[field_name] = {
                        'rate': success_rate,
                        'sample_size': total_count,
                        'has_data': True,
                        'matching_titles': matching_titles
                    }
                else:
                    success_rates[field_name] = {
                        'rate': None,
                        'sample_size': 0,
                        'has_data': False,
                        'matching_titles': []
                    }
            else:
                # No matching shows
                success_rates[field_name] = {
                    'rate': None,
                    'sample_size': 0,
                    'has_data': False,
                    'matching_titles': []
                }
        
        return success_rates
    
    def get_overall_success_rate(self, criteria: Dict[str, Any]) -> Tuple[float, str]:
        """Calculate the overall success rate for the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Tuple of (success_rate, confidence)
        """
        # Get matching shows and count
        matching_shows, match_count = self.criteria_scorer._get_matching_shows(criteria)
        
        if matching_shows.empty or match_count == 0:
            # Return None instead of 0.0 to indicate no data available
            return None, 'none'
        
        # Calculate success rate
        success_rate = self.criteria_scorer._calculate_success_rate(matching_shows)
        
        # Calculate confidence
        confidence = OptimizerConfig.get_confidence_level(match_count)
        
        return success_rate, confidence
