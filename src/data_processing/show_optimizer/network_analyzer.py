"""NetworkAnalyzer: Specialized analyzer for network compatibility and recommendations.

Handles all network-related analysis in the Show Optimizer, including:
1. Network Compatibility Analysis:
   - Calculate network compatibility scores
   - Rank networks based on criteria
   - Group networks into tiers based on compatibility

2. Network Recommendations:
   - Generate network-specific recommendations
   - Analyze network-specific success patterns
   - Identify optimal network features

3. Network Categorization:
   - Classify networks into tiers
   - Analyze network categories and types
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

from ..analyze_shows import ShowsAnalyzer
from ..success_analysis import SuccessAnalyzer
from .optimizer_config import OptimizerConfig
from .field_manager import FieldManager
from .criteria_scorer import CriteriaScorer
from .score_calculators import NetworkMatch, NetworkScoreCalculator

from .optimizer_matcher import Matcher


@dataclass
class NetworkTier:
    """A tier of networks with similar compatibility."""
    tier_name: str  # e.g., "Excellent Match", "Good Match", "Fair Match"
    networks: List[NetworkMatch]
    min_score: float
    max_score: float


class NetworkAnalyzer:
    """Specialized analyzer for network compatibility and recommendations."""
    
    def __init__(self, criteria_scorer: CriteriaScorer, field_manager: FieldManager = None, matcher: Optional[Matcher] = None):
        """Initialize the network analyzer.
        
        Args:
            criteria_scorer: CriteriaScorer instance for scoring calculations
            field_manager: Optional FieldManager instance for field mapping
            matcher: Optional Matcher instance for finding matches
            matcher: Optional Matcher instance for finding network matches
        """
        self.criteria_scorer = criteria_scorer
        self.field_manager = field_manager or criteria_scorer.field_manager
        
        # Initialize NetworkScoreCalculator with the field_manager
        self.network_score_calculator = NetworkScoreCalculator()
        self.network_score_calculator.field_manager = self.field_manager
        

        
        # Initialize matcher for network analysis if not provided
        if matcher is None:
            self.matcher = Matcher(self.field_manager)
        else:
            self.matcher = matcher
        
        # Initialize integrated data
        self._integrated_data = None
        
    def set_integrated_data(self, integrated_data: Dict[str, pd.DataFrame]) -> None:
        """Set the integrated data to use for network analysis.
        
        Args:
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
        """
        self._integrated_data = integrated_data
        
        # Also set the integrated data in the network score calculator
        if hasattr(self.network_score_calculator, 'set_integrated_data'):
            self.network_score_calculator.set_integrated_data(integrated_data)
            
    def set_matching_shows(self, matching_shows: pd.DataFrame):
        """Set the matching shows for network analysis.
        
        Args:
            matching_shows: DataFrame of shows matching the criteria
        """
        self._matching_shows = matching_shows
        
        # Note: We no longer need to set matching_shows in the network_score_calculator
        # as we now pass it directly to calculate_network_scores
        
    def rank_networks_by_compatibility(self, criteria: Dict[str, Any], 
                                       integrated_data: Dict[str, pd.DataFrame],
                                       matching_shows: pd.DataFrame,
                                       limit: int = None) -> List[NetworkMatch]:
        """Rank networks by compatibility with the given criteria.
        
        Args:
            criteria: Dictionary of criteria
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            matching_shows: DataFrame of shows matching the criteria
            limit: Maximum number of networks to return
            
        Returns:
            List of NetworkMatch objects sorted by compatibility score
        """
        try:
            # Validate inputs
            if matching_shows is None or matching_shows.empty:
                return []
                
            if integrated_data is None:
                return []
                
            # Set the integrated data in the network score calculator
            self.set_integrated_data(integrated_data)
            
            network_count = 0
            if 'network_id' in matching_shows.columns:
                network_count = matching_shows['network_id'].nunique()
            
            # Pass matching_shows directly to NetworkScoreCalculator
            network_matches = self.network_score_calculator.calculate_network_scores(criteria, matching_shows=matching_shows)
            
            # Sort by compatibility score (descending)
            network_matches.sort(key=lambda x: x.compatibility_score if x.compatibility_score is not None else -1, reverse=True)
            
            # Use config for default limit if not specified
            if limit is None:
                limit = OptimizerConfig.NETWORK['default_limit']
                
            # Return top networks
            return network_matches[:limit]
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"Debug: Error ranking networks: {str(e)}")
            st.error(f"Error ranking networks: {str(e)}")
            return []
    
    def get_network_tiers(self, criteria: Dict[str, Any], 
                         integrated_data: Dict[str, pd.DataFrame],
                         matching_shows: pd.DataFrame = None,
                         min_confidence: str = 'low') -> Dict[str, NetworkTier]:
        """Group networks into tiers based on compatibility with criteria.
        
        Args:
            criteria: Dictionary of criteria
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            matching_shows: DataFrame of shows matching the criteria (for scoring consistency)
            min_confidence: Minimum confidence level to include (none, low, medium, high)
            
        Returns:
            Dictionary mapping tier names to NetworkTier objects
        """
        try:
            # Validate inputs
            if integrated_data is None:
                return {}
                
            if matching_shows is None or matching_shows.empty:
                pass
            
            # Set the integrated data in the network score calculator
            self.set_integrated_data(integrated_data)
            
            # Get network matches from NetworkScoreCalculator (always use matching_shows if provided)
            network_matches = self.network_score_calculator.calculate_network_scores(criteria, matching_shows=matching_shows)
            
            # Filter by confidence
            confidence_levels = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
            min_confidence_level = confidence_levels.get(min_confidence.lower(), 0)
            
            # More efficient list comprehension instead of loop
            filtered_matches = [
                match for match in network_matches
                if confidence_levels.get(match.confidence.lower(), 0) >= min_confidence_level
            ]
            
            # Sort by compatibility score (descending)
            filtered_matches.sort(key=lambda x: x.compatibility_score if x.compatibility_score is not None else -1, reverse=True)
            
            # Group into tiers
            tiers = {}
            
            # Get tier thresholds directly from OptimizerConfig
            tier_thresholds = OptimizerConfig.NETWORK_TIERS
            
            # Use a more efficient approach to create tiers
            tier_definitions = [
                ('excellent', 'Excellent Match', tier_thresholds['excellent'], 1.0),
                ('good', 'Good Match', tier_thresholds['good'], tier_thresholds['excellent']),
                ('fair', 'Fair Match', tier_thresholds['fair'], tier_thresholds['good']),
                ('poor', 'Poor Match', 0.0, tier_thresholds['fair'])
            ]
            
            # Create tiers using the definitions
            for tier_id, tier_name, min_score, max_score in tier_definitions:
                matches = [m for m in filtered_matches if min_score <= m.compatibility_score < max_score]
                if matches:
                    tiers[tier_id] = NetworkTier(
                        tier_name=tier_name,
                        networks=matches,
                        min_score=min_score,
                        max_score=max_score
                    )

            return tiers
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"Debug: Error grouping networks into tiers: {str(e)}")
            st.error(f"Error grouping networks into tiers: {str(e)}")
            # Return empty tiers dictionary on error
            return {}
    
    def get_network_specific_success_rates(self, criteria: Dict[str, Any], network_id: int, integrated_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Get success rates for specific criteria for a given network.
        
        Args:
            criteria: Dictionary of criteria
            network_id: ID of the network to analyze
            integrated_data: Dictionary of integrated data frames from ShowOptimizer
            
        Returns:
            Dictionary mapping criteria names to success rate information
        """
        try:
            # Set the integrated data in the network score calculator
            self.set_integrated_data(integrated_data)
            
            # Get the shows data from integrated_data
            data = integrated_data.get('shows')
            if data is None or data.empty:
                st.warning("No shows data available in integrated data for network analysis")
                return {}
            
            # Filter to this network
            network_data = data[data['network_id'] == network_id] if 'network_id' in data.columns else pd.DataFrame()
            if network_data.empty:
                return {}
            
            # Calculate success rates for each criteria
            success_rates = {}
            
            for field_name, value in criteria.items():
                if field_name not in self.field_manager.FIELD_CONFIGS:
                    continue
                    
                config = self.field_manager.FIELD_CONFIGS[field_name]
                
                # Filter shows with this criteria
                if config.is_array:
                    # Use the array field mapping from field_manager
                    array_field_mapping = self.field_manager.get_array_field_mapping()
                    
                    # Get the correct column name for this field
                    if field_name in array_field_mapping:
                        field_id = array_field_mapping[field_name]
                    else:
                        st.error(f"Field '{field_name}' not found in array field mapping")
                        success_rates[field_name] = {
                            'rate': None,
                            'sample_size': 0,
                            'has_data': False,
                            'matching_titles': []
                        }
                        continue
                    
                    # Check if the column exists in the DataFrame
                    if field_id not in network_data.columns:
                        st.error(f"Column '{field_id}' for field '{field_name}' not found in data")
                        success_rates[field_name] = {
                            'rate': None,
                            'sample_size': 0,
                            'has_data': False,
                            'matching_titles': []
                        }
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
                    # Scalar field - use field_manager to map field name
                    field_id = self.field_manager.map_field_name(field_name, network_data.columns)
                    if isinstance(value, list):
                        matching_shows = network_data[network_data[field_id].isin(value)]
                    else:
                        matching_shows = network_data[network_data[field_id] == value]
                
                # Calculate success rate and sample size
                # Check if matching_shows is a DataFrame and not empty
                is_empty = True  # Default to empty unless proven otherwise
                
                # First check the type to avoid attribute errors
                if matching_shows is None:
                    is_empty = True
                elif isinstance(matching_shows, pd.DataFrame):
                    is_empty = matching_shows.empty
                elif isinstance(matching_shows, dict):
                    is_empty = len(matching_shows) == 0
                elif isinstance(matching_shows, list):
                    is_empty = len(matching_shows) == 0
                
                if not is_empty:
                    # Calculate success rate using threshold from OptimizerConfig
                    success_threshold = OptimizerConfig.THRESHOLDS['success_threshold']
                    success_count = matching_shows[matching_shows['success_score'] >= success_threshold].shape[0]
                    total_count = matching_shows.shape[0]
                    
                    # Get matching show titles (up to MAX_RESULTS)
                    matching_titles = []
                    if 'title' in matching_shows.columns:
                        matching_titles = matching_shows['title'].tolist()
                        # Limit to MAX_RESULTS titles
                        if len(matching_titles) > OptimizerConfig.MAX_RESULTS:
                            matching_titles = matching_titles[:OptimizerConfig.MAX_RESULTS]
                    
                    if total_count > 0:
                        success_rate = success_count / total_count
                        success_rates[field_name] = {
                            'rate': success_rate,
                            'sample_size': total_count,
                            'has_data': True,
                            'matching_titles': matching_titles,
                            'matching_shows': matching_shows
                        }
                    else:
                        success_rates[field_name] = {
                            'rate': None,
                            'sample_size': 0,
                            'has_data': False,
                            'matching_titles': [],
                            'matching_shows': pd.DataFrame()  # Always include matching_shows key
                        }
                else:
                    # No matching shows
                    success_rates[field_name] = {
                        'rate': None,
                        'sample_size': 0,
                        'has_data': False,
                        'matching_titles': [],
                        'matching_shows': pd.DataFrame()  # Always include matching_shows key
                    }
            
            return success_rates
        except Exception as e:
            st.error(f"Error calculating network-specific success rates: {str(e)}")
            return {}
    
    def get_network_recommendations(self, criteria: Dict[str, Any], 
                                    network: NetworkMatch, 
                                    concept_analyzer=None) -> List[Dict[str, Any]]:
        """Generate network-specific recommendations using the RecommendationEngine from ConceptAnalyzer.
        
        Args:
            criteria: Dictionary of criteria
            network: Target network
            concept_analyzer: ConceptAnalyzer instance that contains the RecommendationEngine
            
        Returns:
            List of recommendation dictionaries
        """
        try:
            if concept_analyzer is None or not hasattr(concept_analyzer, 'recommendation_engine'):
                if OptimizerConfig.DEBUG_MODE:
                    st.write("Warning: ConceptAnalyzer not provided or missing RecommendationEngine. Cannot generate network recommendations.")
                return []
                
            # Get the RecommendationEngine from the ConceptAnalyzer
            recommendation_engine = concept_analyzer.recommendation_engine
                
            # Use the RecommendationEngine to generate network-specific recommendations
            # Get the integrated data from the shows analyzer
            integrated_data = self.shows_analyzer.get_integrated_data() if hasattr(self, 'shows_analyzer') else {}
            
            # Get matching shows from the criteria scorer if available
            matching_shows = None
            if hasattr(self, 'criteria_scorer') and self.criteria_scorer is not None:
                try:
                    # Use the _get_matching_shows method directly from criteria_scorer
                    matching_shows, _, _ = self.criteria_scorer._get_matching_shows(criteria)
                except Exception as e:
                    st.error(f"Error getting matching shows: {str(e)}")
                    matching_shows = None
            
            # Ensure matching_shows is a DataFrame
            if matching_shows is None or not isinstance(matching_shows, pd.DataFrame):
                matching_shows = pd.DataFrame()
            
            # Call with all required parameters
            recommendations = recommendation_engine.generate_network_specific_recommendations(
                criteria=criteria, 
                network=network,
                matching_shows=matching_shows,
                integrated_data=integrated_data
            )
            
            # Convert Recommendation objects to dictionaries for compatibility
            recommendation_dicts = []
            for rec in recommendations:
                recommendation_dicts.append({
                    'recommendation_type': rec.recommendation_type,
                    'criteria_type': rec.criteria_type,
                    'current_value': rec.current_value,
                    'suggested_value': rec.suggested_value,
                    'suggested_name': rec.suggested_name,
                    'impact_score': rec.impact_score,
                    'confidence': rec.confidence,
                    'explanation': rec.explanation
                })
            
            return recommendation_dicts
        except Exception as e:
            st.error(f"Error generating network recommendations: {str(e)}")
            return []
    
    def _get_criteria_name(self, criteria_type: str, criteria_value: Any) -> str:
        """Get the display name for a criteria value.
        
        Args:
            criteria_type: Type of criteria (e.g., 'genre', 'format')
            criteria_value: Value of the criteria (ID)
            
        Returns:
            Display name for the criteria value
        """
        try:
            # Get options for this criteria type
            options = self.field_manager.get_options(criteria_type)
            
            # Find the option with matching ID
            for option in options:
                if option.id == criteria_value:
                    return option.name
            
            # If not found, return the raw value
            return str(criteria_value)
        except Exception as e:
            st.error(f"Error getting criteria name: {str(e)}")
            return str(criteria_value)
    
    def _calculate_success_rate_with_confidence(self, criteria: Dict[str, Any], 
                                              min_sample_size: int = 10) -> Tuple[float, str]:
        """Calculate success rate with confidence level.
        
        Args:
            criteria: Dictionary of criteria
            min_sample_size: Minimum sample size for confidence
            
        Returns:
            Tuple of (success_rate, confidence_level)
        """
        try:
            # Use the criteria scorer to get matching shows
            matching_shows, count, confidence_info = self.criteria_scorer._matching_calculator.get_matching_shows(criteria)
            
            # Check if matching_shows is empty
            is_empty = False
            if isinstance(matching_shows, pd.DataFrame):
                is_empty = matching_shows.empty
            elif isinstance(matching_shows, dict):
                is_empty = len(matching_shows) == 0
            elif isinstance(matching_shows, list):
                is_empty = len(matching_shows) == 0
            else:
                # For other types, assume it's empty (safer approach)
                is_empty = True
                
            if is_empty or count < min_sample_size:
                return None, 'none'
            
            # Calculate success rate
            success_threshold = OptimizerConfig.THRESHOLDS['success_threshold']
            
            # Handle different types of matching_shows for success rate calculation
            if isinstance(matching_shows, pd.DataFrame):
                if 'success_score' in matching_shows.columns:
                    success_count = matching_shows[matching_shows['success_score'] >= success_threshold].shape[0]
                    success_rate = success_count / count if count > 0 else None
                else:
                    # No success_score column
                    success_rate = None
            else:
                # Can't calculate success rate with non-DataFrame
                success_rate = None
            
            # Determine confidence level
            confidence = confidence_info.get('level', OptimizerConfig.get_confidence_level(count, confidence_info.get('match_level', 1)))
            
            return success_rate, confidence
        except Exception as e:
            st.error(f"Error calculating success rate: {str(e)}")
            return None, 'none'
