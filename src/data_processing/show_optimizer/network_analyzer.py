"""NetworkAnalyzer: Specialized analyzer for network compatibility.

Handles network-related analysis in the Show Optimizer, focusing on:
1. Network Compatibility Analysis:
   - Calculate network compatibility scores directly from matching shows
   - Rank networks based on compatibility with show criteria
   - Group networks into tiers based on compatibility thresholds

2. Network-Specific Success Analysis:
   - Calculate success rates for specific networks
   - Analyze network preferences for different criteria
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
import streamlit as st

from .optimizer_config import OptimizerConfig
from .field_manager import FieldManager
from .criteria_scorer import CriteriaScorer


@dataclass
class NetworkMatch:
    """Network match data structure with compatibility and success metrics."""
    network_id: int
    network_name: str
    compatibility_score: float
    success_probability: Optional[float] = None
    sample_size: int = 0
    confidence: str = "none"


class NetworkAnalyzer:
    """Specialized analyzer for network compatibility using a single data stream approach."""
    
    def __init__(self, criteria_scorer: CriteriaScorer, field_manager: FieldManager = None):
        """Initialize the network analyzer.
        
        Args:
            criteria_scorer: CriteriaScorer instance for score calculations
            field_manager: Optional FieldManager instance for field mapping
        """
        self.criteria_scorer = criteria_scorer
        self.field_manager = field_manager or criteria_scorer.field_manager
        
    def get_display_name(self, field_name: str, value: Any) -> str:
        """Wrapper method to handle legacy calls to get_display_name.
        This redirects to the proper field_manager.get_name method.
        
        Args:
            field_name: Name of the field
            value: Value to get name for
            
        Returns:
            Display name for the value
        """
        try:
            if self.field_manager:
                return self.field_manager.get_name(field_name, value)
            return str(value)
        except Exception as e:
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"Debug: Error in get_display_name wrapper for {field_name}={value}: {str(e)}")
            return str(value)
        
    def rank_networks_by_compatibility(self, matching_shows: pd.DataFrame, limit: int = None) -> List[NetworkMatch]:
        """Rank networks by compatibility using only the matching shows DataFrame.
        
        Args:
            matching_shows: DataFrame of shows matching the criteria with match_level column
            limit: Maximum number of networks to return
            
        Returns:
            List of NetworkMatch objects sorted by compatibility score
        """
        try:
            # Validate inputs
            if matching_shows is None or matching_shows.empty:
                if OptimizerConfig.DEBUG_MODE:
                    st.write("Debug: No matching shows provided for network ranking")
                return []
            
            # Check if network_id column exists
            if 'network_id' not in matching_shows.columns:
                if OptimizerConfig.DEBUG_MODE:
                    st.write("Debug: No network_id column in matching shows")
                return []
            
            # Calculate network compatibility scores directly from matching shows
            network_scores = {}
            
            # Group by network_id and calculate compatibility scores
            for network_id, network_shows in matching_shows.groupby('network_id'):
                # Skip if no shows for this network
                if network_shows.empty:
                    continue
                    
                # Get network name
                network_name = "Unknown Network"
                if self.field_manager and hasattr(self.field_manager, 'get_network_name'):
                    network_name = self.field_manager.get_network_name(network_id) or "Unknown Network"
                
                # Calculate compatibility score based on match levels
                # Lower match_level is better (1 is exact match)
                if 'match_level' in network_shows.columns:
                    avg_match_level = network_shows['match_level'].mean()
                    # Convert to score (1 is best, lower match_level = higher score)
                    compatibility_score = max(0.1, 1.0 - ((avg_match_level - 1) * 0.2))
                else:
                    # Default if no match_level column
                    compatibility_score = 0.5
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"Debug: No match_level column for network {network_id}, using default score")
                
                # Calculate success probability if success_score column exists
                success_probability = None
                sample_size = len(network_shows)
                confidence = "none"
                
                if 'success_score' in network_shows.columns:
                    # Calculate average success score
                    success_probability = network_shows['success_score'].mean()
                    
                    # Determine confidence based on sample size and match level using OptimizerConfig
                    avg_match_level = network_shows['match_level'].mean() if 'match_level' in network_shows.columns else 1
                    confidence = OptimizerConfig.get_confidence_level(sample_size, int(avg_match_level))
                
                # Create NetworkMatch object
                network_match = NetworkMatch(
                    network_id=network_id,
                    network_name=network_name,
                    compatibility_score=compatibility_score,
                    success_probability=success_probability,
                    sample_size=sample_size,
                    confidence=confidence
                )
                
                network_scores[network_id] = network_match
            
            # Convert to list and sort by compatibility score (descending)
            network_matches = list(network_scores.values())
            network_matches.sort(key=lambda x: x.compatibility_score if x.compatibility_score is not None else -1, reverse=True)
            
            # Use config for default limit if not specified
            if limit is None:
                limit = OptimizerConfig.NETWORK.get('default_limit', 10)
                
            # Return top networks
            return network_matches[:limit]
        except Exception as e:
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"Debug: Error ranking networks: {str(e)}")
            return []
            
    def group_networks_into_tiers(self, network_matches: List[NetworkMatch]) -> Dict[str, List[NetworkMatch]]:
        """Group networks into tiers based on compatibility score.
        
        Args:
            network_matches: List of NetworkMatch objects sorted by compatibility score
            
        Returns:
            Dictionary of tier_name -> list of NetworkMatch objects
        """
        try:
            # Validate inputs
            if not network_matches:
                return {}
                
            # Initialize tiers dictionary with empty lists for each tier
            tiers = {tier_name: [] for tier_name in OptimizerConfig.NETWORK_TIERS.keys()}
            # Add 'poor' tier if not in config
            if 'poor' not in tiers:
                tiers['poor'] = []
            
            # Get sorted tier thresholds for comparison (highest to lowest)
            sorted_tiers = sorted(
                [(name, threshold) for name, threshold in OptimizerConfig.NETWORK_TIERS.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Assign networks to tiers
            for network in network_matches:
                score = network.compatibility_score
                assigned = False
                
                # Check each tier threshold in order
                for tier_name, threshold in sorted_tiers:
                    if score >= threshold:
                        tiers[tier_name].append(network)
                        assigned = True
                        break
                        
                # If not assigned to any tier, put in 'poor' tier
                if not assigned:
                    tiers['poor'].append(network)
            
            return tiers
            
        except Exception as e:
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"Debug: Error grouping networks into tiers: {str(e)}")
            return {}
    
    def get_network_tiers(self, matching_shows: pd.DataFrame, min_confidence: str = 'low') -> Dict[str, List[NetworkMatch]]:
        """Group networks into tiers based on compatibility using matching shows.
        
        Args:
            matching_shows: DataFrame of shows matching the criteria with match_level column
            min_confidence: Minimum confidence level to include (none, low, medium, high)
            
        Returns:
            Dictionary mapping tier names to lists of NetworkMatch objects
        """
        try:
            # Validate inputs
            if matching_shows is None or matching_shows.empty:
                if OptimizerConfig.DEBUG_MODE:
                    st.write("Debug: No matching shows provided for network tiers")
                return {}
            
            # Get network matches using the simplified approach
            network_matches = self.rank_networks_by_compatibility(matching_shows)
            
            # Filter by confidence using OptimizerConfig
            # Map confidence levels to numeric values for comparison
            confidence_order = {
                'none': 0,
                'minimal': 1,
                'very_low': 2,
                'low': 3,
                'medium': 4,
                'high': 5
            }
            min_confidence_level = confidence_order.get(min_confidence.lower(), 0)
            
            filtered_matches = []
            for network in network_matches:
                network_confidence = confidence_order.get(network.confidence, 0)
                if network_confidence >= min_confidence_level:
                    filtered_matches.append(network)
            
            # Use the simplified group_networks_into_tiers method
            return self.group_networks_into_tiers(filtered_matches)
        except Exception as e:
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"Debug: Error getting network tiers: {str(e)}")
            return {}
    
    def get_network_specific_success_rates(self, matching_shows: pd.DataFrame, network_id: int) -> Dict[str, Dict[str, Any]]:
        """Get success rates for specific criteria for a given network using matching shows.
        
        Args:
            matching_shows: DataFrame of shows matching the criteria with match_level and success_score columns
            network_id: ID of the network to analyze
            
        Returns:
            Dictionary mapping criteria names to success rate information
        """
        try:
            # Validate inputs
            if matching_shows is None or matching_shows.empty:
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"Debug: No matching shows provided for network {network_id} success rates")
                return {}
                
            # Filter to this network
            network_shows = matching_shows[matching_shows['network_id'] == network_id] if 'network_id' in matching_shows.columns else pd.DataFrame()
            if network_shows.empty:
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"Debug: No shows for network {network_id} in matching shows")
                return {}
                
            # Check if success_score column exists
            if 'success_score' not in network_shows.columns:
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"Debug: No success_score column in matching shows for network {network_id}")
                return {}
            
            # For network-specific success rates, we analyze the columns in the matching_shows DataFrame
            # that are relevant for the network's success metrics
            success_rates = {}
            
            # Get unique columns that might represent criteria (exclude standard columns)
            standard_columns = {'network_id', 'match_level', 'success_score', 'title', 'show_id'}
            criteria_columns = [col for col in network_shows.columns if col not in standard_columns]
            
            # Calculate success rate for each criteria column
            success_threshold = OptimizerConfig.THRESHOLDS['success_threshold']
            
            # Filter columns to only include those that the field_manager can handle
            valid_criteria_columns = []
            for column in criteria_columns:
                # Check if this is a field the field_manager knows about
                base_field = column
                if column.endswith('_id') or column.endswith('_ids'):
                    base_field = column[:-3] if column.endswith('_id') else column[:-4]
                elif column.endswith('_name') or column.endswith('_names'):
                    base_field = column[:-5] if column.endswith('_name') else column[:-6]
                    
                try:
                    # Check if field_manager has this field
                    if self.field_manager and self.field_manager.has_field(base_field):
                        valid_criteria_columns.append(column)
                    else:
                        if OptimizerConfig.DEBUG_MODE:
                            st.write(f"Debug: Skipping column {column} - not in field_manager")
                except Exception as e:
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"Debug: Error checking field {column}: {str(e)}")
                    # Skip this column
                    continue
                    
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"Debug: Processing {len(valid_criteria_columns)} valid columns out of {len(criteria_columns)} total")
                
            for column in valid_criteria_columns:
                # Get unique values for this column
                try:
                    # Skip columns with all null values
                    if network_shows[column].isna().all():
                        continue
                        
                    # Get unique non-null values
                    unique_values = network_shows[column].dropna().unique()
                    
                    # Skip if no unique values
                    if len(unique_values) == 0:
                        continue
                        
                    # For each unique value, calculate success rate
                    for value in unique_values:
                        # Skip null values
                        if pd.isna(value):
                            continue
                            
                        # Handle array fields differently
                        is_array_field = False
                        if isinstance(value, (list, tuple)) or (
                            isinstance(value, str) and column.endswith('_names') and '[' in value
                        ):
                            is_array_field = True
                            
                        # For array fields, we need to check if the value is in the array
                        if is_array_field:
                            # Skip array fields for now as they need special handling
                            if OptimizerConfig.DEBUG_MODE:
                                st.write(f"Debug: Skipping array field {column} with value {value}")
                            continue
                        else:
                            # For scalar fields, we can do a direct comparison
                            value_shows = network_shows[network_shows[column] == value]
                        
                        # Skip if no shows
                        if value_shows.empty:
                            continue
                            
                        # Calculate success rate
                        success_count = value_shows[value_shows['success_score'] >= success_threshold].shape[0]
                        total_count = value_shows.shape[0]
                        
                        if total_count > 0:
                            success_rate = success_count / total_count
                            
                            # Get field name for display
                            field_name = column
                            
                            # Get value name for display
                            value_name = str(value)
                            if self.field_manager:
                                try:
                                    # Use the get_display_name wrapper which calls field_manager.get_name
                                    value_name = self.get_display_name(column, value)
                                except Exception as e:
                                    if OptimizerConfig.DEBUG_MODE:
                                        st.write(f"Debug: Error getting option name for {column}={value}: {str(e)}")
                                    # Keep the default string value
                            
                            # Create a key that combines field and value
                            key = f"{field_name}:{value_name}"
                            
                            # Get matching show titles (up to MAX_RESULTS)
                            matching_titles = []
                            if 'title' in value_shows.columns:
                                matching_titles = value_shows['title'].tolist()
                                # Limit to MAX_RESULTS titles
                                if len(matching_titles) > OptimizerConfig.MAX_RESULTS:
                                    matching_titles = matching_titles[:OptimizerConfig.MAX_RESULTS]
                            
                            # Calculate confidence using OptimizerConfig
                            avg_match_level = value_shows['match_level'].mean() if 'match_level' in value_shows.columns else 1
                            confidence = OptimizerConfig.get_confidence_level(total_count, int(avg_match_level))
                            
                            # Store the results
                            success_rates[key] = {
                                'field_name': field_name,
                                'value': value,
                                'value_name': value_name,
                                'rate': success_rate,
                                'sample_size': total_count,
                                'has_data': True,
                                'confidence': confidence,
                                'matching_titles': matching_titles,
                                'matching_shows': value_shows
                            }
                except Exception as e:
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"Debug: Error calculating success rate for column {column}: {str(e)}")
                    continue
            
            return success_rates
        except Exception as e:
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"Debug: Error calculating network-specific success rates: {str(e)}")
            return {}
    
    def get_network_recommendations(self, matching_shows: pd.DataFrame, 
                                    network: NetworkMatch, 
                                    concept_analyzer=None) -> List[Dict[str, Any]]:
        """Generate network-specific recommendations using the RecommendationEngine from ConceptAnalyzer.
        
        Args:
            matching_shows: DataFrame of shows matching the criteria with match_level column
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
                
            # Validate matching_shows
            if matching_shows is None or not isinstance(matching_shows, pd.DataFrame) or matching_shows.empty:
                if OptimizerConfig.DEBUG_MODE:
                    st.write(f"Debug: No matching shows provided for network recommendations for {network.network_name}")
                return []
            
            # Filter matching_shows to this network if needed
            network_shows = matching_shows
            if 'network_id' in matching_shows.columns:
                network_shows = matching_shows[matching_shows['network_id'] == network.network_id]
                
                if network_shows.empty:
                    if OptimizerConfig.DEBUG_MODE:
                        st.write(f"Debug: No shows for network {network.network_name} in matching shows")
                    return []
            
            # Extract criteria from matching_shows if possible
            # This is more reliable than passing potentially stale criteria
            criteria = {}
            if hasattr(matching_shows, 'criteria') and isinstance(matching_shows.criteria, dict):
                criteria = matching_shows.criteria
            
            # Call with all required parameters
            recommendations = recommendation_engine.generate_network_specific_recommendations(
                network=network,
                matching_shows=network_shows,
                criteria=criteria
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
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"Debug: Error generating network recommendations: {str(e)}")
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
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"Debug: Error getting criteria name: {str(e)}")
            return str(criteria_value)
    
    def _calculate_success_rate_with_confidence(self, matching_shows: pd.DataFrame, 
                                               min_sample_size: int = 10) -> Tuple[float, str]:
        """Calculate success rate with confidence level.
        
        Args:
            matching_shows: DataFrame of shows matching the criteria with match_level and success_score columns
            min_sample_size: Minimum sample size for confidence
            
        Returns:
            Tuple of (success_rate, confidence_level)
        """
        try:
            # Validate input
            if matching_shows is None or not isinstance(matching_shows, pd.DataFrame) or matching_shows.empty:
                return None, 'none'
                
            # Get sample size
            count = len(matching_shows)
            if count < min_sample_size:
                return None, 'none'
            
            # Calculate success rate
            success_threshold = OptimizerConfig.THRESHOLDS['success_threshold']
            
            # Calculate success rate if success_score column exists
            if 'success_score' in matching_shows.columns:
                success_count = matching_shows[matching_shows['success_score'] >= success_threshold].shape[0]
                success_rate = success_count / count if count > 0 else None
            else:
                # No success_score column
                if OptimizerConfig.DEBUG_MODE:
                    st.write("Debug: No success_score column in matching_shows for success rate calculation")
                success_rate = None
            
            # Calculate average match level for confidence
            avg_match_level = 1
            if 'match_level' in matching_shows.columns:
                avg_match_level = matching_shows['match_level'].mean()
            
            # Determine confidence level using OptimizerConfig
            confidence = OptimizerConfig.get_confidence_level(count, int(avg_match_level))
            
            return success_rate, confidence
        except Exception as e:
            if OptimizerConfig.DEBUG_MODE:
                st.write(f"Debug: Error calculating success rate: {str(e)}")
            return None, 'none'
