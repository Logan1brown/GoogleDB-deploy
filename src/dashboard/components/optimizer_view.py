"""OptimizerView: Bridge between ShowOptimizer and UI.

This component serves as the bridge between the ShowOptimizer analysis engine
and the UI layer, handling initialization, state management, and result formatting.

Key responsibilities:

1. Initialization:
   - Initialize ShowOptimizer and related components
   - Cache field options and display options in state
   - Manage initialization status

2. Analysis Execution:
   - Run analysis with normalized criteria
   - Handle errors and provide user feedback
   - Store results in state

3. Result Formatting:
   - Format recommendations for UI display
   - Generate user-friendly explanations
   - Format success probability and confidence information
   - Format component scores and network matches
   - Prepare data structures for UI components
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import streamlit as st

from src.data_processing.show_optimizer.optimizer_config import OptimizerConfig
from src.data_processing.show_optimizer.criteria_scorer import ComponentScore
from src.data_processing.show_optimizer.network_analyzer import NetworkMatch
from src.data_processing.show_optimizer.recommendation_engine import SuccessFactor, Recommendation
from src.data_processing.show_optimizer.field_manager import FieldManager
from src.data_processing.show_optimizer.optimizer_concept_analyzer import OptimizationSummary


class OptimizerView:
    """Bridge between ShowOptimizer and UI, handling initialization, state management, and result formatting."""
    
    def __init__(self, show_optimizer=None):
        """Initialize the optimizer view.
        
        Args:
            show_optimizer: Optional ShowOptimizer instance. If not provided,
                           one will be created during initialization.
        """
        self.optimizer = show_optimizer
        self.field_manager = None
        self.criteria_scorer = None
        self.config = OptimizerConfig
        self.initialized = False
        
    def initialize(self, state=None):
        """Initialize the ShowOptimizer and related components.
        
        Args:
            state: Optional Streamlit state object for storing session data
            
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Create ShowOptimizer if not provided
            if self.optimizer is None:
                # Import here to avoid circular imports
                from src.data_processing.show_optimizer.show_optimizer import ShowOptimizer
                self.optimizer = ShowOptimizer()
            
            # Initialize the optimizer
            if not self.optimizer.initialize():
                st.error("Failed to initialize Show Optimizer")
                return False
                
            # Get field manager and criteria scorer from optimizer
            self.field_manager = self.optimizer.field_manager
            self.criteria_scorer = self.optimizer.criteria_scorer
            
            # Cache field options in state if provided
            if state is not None:
                # Get field options from field manager
                field_options = {}
                display_options = {}
                
                for field_name in self.field_manager.FIELD_CONFIGS.keys():
                    field_options[field_name] = self.field_manager.get_options(field_name)
                    display_options[field_name] = self.field_manager.get_display_options(field_name)
                
                # Store in state
                state['field_options'] = field_options
                state['display_options'] = display_options
            
            self.initialized = True
            return True
            
        except Exception as e:
            st.error(f"Error initializing Show Optimizer: {str(e)}")
            return False
    

    def run_analysis(self, state, criteria=None):
        """Run analysis on the provided criteria and update state with results.
        
        Args:
            state: Streamlit state object for storing session data
            criteria: Optional criteria to analyze. If not provided, will use criteria from state.
            
        Returns:
            True if analysis was successful, False otherwise
        """
        # Get criteria from state if not provided
        if criteria is None:
            criteria = state.get('criteria', {})
        
        # Check if criteria is empty
        if not criteria:
            st.info("Select criteria to analyze your concept.")
            # Clear any previous results when criteria are empty
            state['matching_shows'] = pd.DataFrame()
            state['network_matches'] = []
            state['component_scores'] = {}
            state['success_probability'] = None
            state['recommendations'] = []
            state['summary'] = None
            st.session_state['matching_shows'] = pd.DataFrame()
            st.session_state['optimizer_summary'] = None
            return True
            
        # Check if optimizer is initialized
        if not self.initialized:
            # Try to initialize if not already initialized
            if not self.initialize(state):
                st.error("Show Optimizer is not initialized. Please refresh the page and try again.")
                return False
        
        # Check if field_manager is available
        if not self.field_manager:
            st.error("Field manager is not available. Please refresh the page.")
            return False
        
        try:
            # Run the analysis
            with st.spinner("Analyzing concept..."):
                # Normalize criteria using field_manager
                normalized_criteria = self.field_manager.normalize_criteria(criteria)
                
                # Check if criteria_scorer is initialized
                if not hasattr(self.optimizer, 'criteria_scorer'):
                    st.warning("Criteria scorer not initialized properly.")
                    return False
                    
                # In the refactored architecture, we don't need to fetch criteria data separately
                # The ShowOptimizer handles data fetching and integration
                
                # Run the actual analysis
                summary = self.optimizer.analyze_concept(normalized_criteria)
                
                # Store results in state
                if summary:
                    # Format the summary for UI display
                    formatted_data = self.format_optimization_summary(summary)
                    
                    # Attach the formatted data to the summary for easy access
                    summary.formatted_data = formatted_data
                    
                    # No backward compatibility or fallbacks - rely solely on top_networks and formatted_data
                    
                    state['summary'] = summary
                    # Also store in session state for persistence
                    st.session_state["optimizer_summary"] = summary
                    return True
                else:
                    st.error("Analysis failed to produce results.")
                    return False
                    
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            import traceback
            st.write(f"Error details: {traceback.format_exc()}")
            return False
        
    def format_optimization_summary(self, summary: OptimizationSummary) -> OptimizationSummary:
        """Format the optimization summary for display.
        
        Args:
            summary: The optimization summary to format
            
        Returns:
            The formatted optimization summary with attached formatted_data
        """
        try:
            # Initialize formatted_data if it doesn't exist
            if not hasattr(summary, 'formatted_data'):
                summary.formatted_data = {}
                
            # Format network matches if available
            if hasattr(summary, 'top_networks') and summary.top_networks:
                summary.formatted_data['networks'] = self._format_network_matches(summary.top_networks)
                
            # Format component scores if available
            if hasattr(summary, 'component_scores') and summary.component_scores:
                summary.formatted_data['component_scores'] = self._format_component_scores(summary.component_scores)
                
            # Format match quality if available
            if hasattr(summary, 'match_quality') and summary.match_quality:
                summary.formatted_data['match_quality'] = self._format_match_quality(summary.match_quality)
                
            # Format success factors if available
            if hasattr(summary, 'success_factors') and summary.success_factors:
                summary.formatted_data['success_factors'] = self._format_success_factors(summary.success_factors)
                
            # Format recommendations if available
            if hasattr(summary, 'recommendations') and summary.recommendations:
                summary.formatted_data['recommendations'] = self._format_recommendations(summary.recommendations)
            
            # Format success probability if available
            if hasattr(summary, 'overall_success_probability'):
                probability = summary.overall_success_probability
                confidence = summary.confidence if hasattr(summary, 'confidence') else self.config.DEFAULT_VALUES['confidence']
                
                # Format the success probability
                if probability is not None:
                    display = f"{probability:.0%}"
                    confidence_display = self.config.CONFIDENCE_DISPLAY.get(confidence, confidence.capitalize())
                    subtitle = f"Confidence: {confidence_display}"
                else:
                    display = "N/A"
                    subtitle = "Data unavailable"
                
                summary.formatted_data['success_probability'] = {
                    "value": probability,
                    "display": display,
                    "confidence": confidence,
                    "subtitle": subtitle
                }
            
            return summary
            
        except Exception as e:
            st.error(f"Error formatting optimization summary: {str(e)}")
            import traceback
            st.write(f"Error details: {traceback.format_exc()}")
            return summary
    
    def _format_success_probability(self, probability: Optional[float], confidence: str) -> Dict[str, Any]:
        """Format success probability for display.
        
        Args:
            probability: Success probability (0-1) or None
            confidence: Confidence level (none, low, medium, high)
            
        Returns:
            Dictionary with formatted success probability data
        """
        # Get confidence display text from config
        confidence_display = self.config.CONFIDENCE_DISPLAY.get(confidence, confidence.capitalize())
        
        # Format probability as percentage
        if probability is not None:
            probability_display = f"{probability:.0%}"
        else:
            probability_display = "N/A"
            
        return {
            "value": probability_display,
            "confidence": confidence_display,
            "confidence_level": confidence
        }
    
    def _format_recommendations(self, recommendations: List[Recommendation]) -> Dict[str, Any]:
        """Format recommendations for display.
        
        Args:
            recommendations: List of Recommendation objects
            
        Returns:
            Dictionary with formatted recommendations grouped by type
        """
        # Initialize with keys from OptimizerConfig.RECOMMENDATION_TYPES
        config = OptimizerConfig()
        grouped = {rec_type: [] for rec_type in config.RECOMMENDATION_TYPES.keys()}
        
        # Add backward compatibility for 'remove' type which might be used in UI
        if 'remove' not in grouped:
            grouped['remove'] = []
            
        # Track network-specific recommendations separately
        network_specific = []
        
        for rec in recommendations:
            # Skip invalid recommendations
            if not hasattr(rec, 'recommendation_type') and not hasattr(rec, 'rec_type'):
                continue
                
            # Get recommendation type, defaulting to rec_type if recommendation_type doesn't exist
            rec_type = getattr(rec, 'recommendation_type', getattr(rec, 'rec_type', None))
            
            # Format the recommendation with all data needed for UI display
            formatted_rec = {
                # Display values
                "title": f"{rec.criteria_type.replace('_', ' ').title()}: {rec.suggested_name}",
                "description": rec.explanation,
                "importance": rec.confidence,
                "category": rec_type,
                "impact": rec.impact_score,
                "criteria_type": rec.criteria_type,
                "suggested_name": rec.suggested_name,
                "current_value": rec.current_value,
                "suggested_value": rec.suggested_value,
                
                # Raw data for sorting and filtering
                "_impact_raw": rec.impact_score,
                "_confidence_level": self._get_confidence_level(rec.confidence),
                "_rec_type": rec_type
            }
            
            # Add to appropriate group
            if rec_type == 'network_specific':
                network_specific.append(formatted_rec)
            elif rec_type in grouped:
                grouped[rec_type].append(formatted_rec)
            else:
                # Create group if it doesn't exist
                if rec_type not in grouped:
                    grouped[rec_type] = []
                grouped[rec_type].append(formatted_rec)
        
        # Sort each group by impact score (descending)
        for rec_type in grouped:
            grouped[rec_type].sort(key=lambda x: abs(x["_impact_raw"]), reverse=True)
            
        # Sort network-specific recommendations by impact score
        network_specific.sort(key=lambda x: abs(x["_impact_raw"]), reverse=True)
        
        return {
            "grouped": grouped,
            "network_specific": network_specific,
            "all": [rec for group in grouped.values() for rec in group] + network_specific
        }
    
    def _format_network_matches(self, network_matches: List[NetworkMatch]) -> List[Dict[str, Any]]:
        """Format network matches for display.
        
        Args:
            network_matches: List of NetworkMatch objects
            
        Returns:
            List of formatted network match dictionaries ready for direct display in UI
        """
        # Always add some basic debug output
        st.write(f"OptimizerView: Formatting {len(network_matches) if network_matches else 0} network matches")
        
        # Add detailed debug output
        if st.session_state.get('debug_mode', False):
            st.write("### Debug: Network Match Formatting")
            st.write(f"Debug: Formatting {len(network_matches) if network_matches else 0} network matches")
            if network_matches and len(network_matches) > 0:
                st.write(f"Debug: First network match type: {type(network_matches[0]).__name__}")
                st.write(f"Debug: First network match dir: {dir(network_matches[0])}")
                st.write(f"Debug: First network match network_id: {getattr(network_matches[0], 'network_id', 'Not found')}")
                st.write(f"Debug: First network match network_name: {getattr(network_matches[0], 'network_name', 'Not found')}")
            st.write("---")
            
        formatted = []
        
        # Check if network_matches is None or empty
        if not network_matches:
            if st.session_state.get('debug_mode', False):
                st.write("Debug: No network matches to format")
            return []
        
        for match in network_matches:
            # Get network name from field manager - simple, direct approach
            network_id = match.network_id
            
            # Ensure network_id is an integer for proper lookup
            if isinstance(network_id, str) and network_id.isdigit():
                network_id = int(network_id)
            elif isinstance(network_id, float):
                network_id = int(network_id)
                
            # Get name directly from field manager - no fallbacks
            network_name = self.field_manager.get_name('network', network_id)
            
            if st.session_state.get('debug_mode', False):
                st.write(f"Debug: Formatting network {network_id} -> {network_name}")
            
            # Format compatibility score and success probability as percentages with proper rounding
            if match.compatibility_score is not None:
                # Ensure we're working with a proper decimal value
                compatibility_value = float(match.compatibility_score)
                compatibility_display = f"{compatibility_value*100:.1f}%"
                # Store raw value for sorting
                compatibility_raw = compatibility_value
            else:
                compatibility_display = "N/A"
                compatibility_raw = 0
                
            if match.success_probability is not None:
                # Ensure we're working with a proper decimal value
                success_value = float(match.success_probability)
                success_display = f"{success_value*100:.1f}%"
                # Store raw value for sorting
                success_raw = success_value
            else:
                success_display = "N/A"
                success_raw = 0
                
            # Get confidence display text from config
            confidence = match.confidence or "unknown"
            confidence_display = self.config.CONFIDENCE_DISPLAY.get(confidence, confidence.capitalize())
            
            # Format the network match with all data needed for UI display
            formatted.append({
                # Display values
                'Network': network_name,
                'Compatibility': compatibility_display,
                'Success Probability': success_display,
                'Sample Size': match.sample_size,
                'Confidence': confidence_display,
                
                # Raw values for sorting (not displayed)
                '_compatibility_raw': compatibility_raw,
                '_success_prob_raw': success_raw
            })
            
        # Sort the formatted data by compatibility (descending), then by success probability
        formatted.sort(key=lambda x: (x['_compatibility_raw'] or 0, x['_success_prob_raw'] or 0), reverse=True)
        
        if st.session_state.get('debug_mode', False):
            st.write(f"Debug: Returning {len(formatted)} formatted network matches")
            
        return formatted
    
    def _format_component_scores(self, component_scores: Dict[str, ComponentScore]) -> Dict[str, Dict[str, Any]]:
        """Format component scores for display.
        
        Args:
            component_scores: Dictionary mapping component names to ComponentScore objects
            
        Returns:
            Dictionary with formatted component score data
        """
        formatted = {}
        
        for component, score in component_scores.items():
            # Skip None scores
            if score is None:
                continue
                
            # Format the score
            formatted[component] = {
                "score": score.score,
                "confidence": score.confidence,
                "sample_size": score.sample_size
            }
            
        return formatted
        
    def _get_confidence_level(self, confidence: str) -> int:
        """Convert confidence string to numeric level for sorting.
        
        Args:
            confidence: Confidence string (none, low, medium, high)
            
        Returns:
            Numeric confidence level (0-3)
        """
        confidence_levels = {
            'none': 0,
            'low': 1,
            'medium': 2,
            'high': 3
        }
        return confidence_levels.get(confidence.lower(), 0)
    
    def _format_success_factors(self, success_factors: List) -> List[Dict[str, Any]]:
        """Format success factors for display.
        
        Args:
            success_factors: List of SuccessFactor objects
            
        Returns:
            List of formatted success factor dictionaries ready for direct display in UI
        """
        if not success_factors:
            return []
            
        formatted = []
        
        for factor in success_factors:
            # Get proper display name for criteria type using field_manager
            criteria_type_display = factor.criteria_type.replace("_", " ").title()
            
            # Format the success factor with all data needed for UI display
            formatted.append({
                # Display values
                "Type": criteria_type_display,
                "Name": factor.criteria_name,
                "Impact": factor.impact_score,
                "ImpactDisplay": f"{factor.impact_score:.2f}",
                "Confidence": factor.confidence.capitalize(),
                "Sample": factor.sample_size,
                "SampleDisplay": f"Sample: {factor.sample_size}",
                
                # Raw data for charts and sorting
                "_impact_raw": factor.impact_score,
                "_confidence_level": self._get_confidence_level(factor.confidence),
                "_matching_titles": factor.matching_titles if hasattr(factor, 'matching_titles') else []
            })
        
        # Sort by absolute impact (descending)
        formatted.sort(key=lambda x: abs(x["_impact_raw"]), reverse=True)
        
        return formatted
    
    def _format_match_quality(self, match_quality: Any) -> Dict[str, Any]:
        """Format match quality information for display.
        
        Args:
            match_quality: Match quality object with match level, count, and confidence information
            
        Returns:
            Dictionary with formatted match quality data
        """
        try:
            # Extract required fields from match_quality object
            match_level = getattr(match_quality, 'match_level', 0)
            match_count = getattr(match_quality, 'match_count', 0)
            match_counts_by_level = getattr(match_quality, 'match_counts_by_level', {})
            confidence_score = getattr(match_quality, 'confidence_score', 0.0)
            
            # Get match level name from config
            match_level_name = self.config.MATCH_LEVELS.get(match_level, {}).get('name', f"Level {match_level}")
            
            return {
                "match_level": match_level,
                "match_level_name": match_level_name,
                "match_count": match_count,
                "match_counts_by_level": match_counts_by_level,
                "confidence_score": confidence_score
            }
        except Exception as e:
            st.write(f"Debug: Error formatting match quality: {str(e)}")
            return {
                "match_level": 0,
                "match_level_name": "Unknown",
                "match_count": 0,
                "match_counts_by_level": {},
                "confidence_score": 0.0
            }
        
    def format_criteria_display(self, criteria: Dict[str, Any]) -> Dict[str, str]:
        """Format criteria for display.
        
        Args:
            criteria: Dictionary of criteria
            
        Returns:
            Dictionary mapping criteria types to display names
        """
        formatted = {}
        
        for criteria_type, criteria_value in criteria.items():
            # Skip None values
            if criteria_value is None:
                continue
                
            # Get display name
            display_name = self._get_criteria_name(criteria_type, criteria_value)
            formatted[criteria_type] = display_name
            
        return formatted
    
    def _get_criteria_name(self, criteria_type: str, criteria_value: Any) -> str:
        """Get display name for a criteria value.
        
        Args:
            criteria_type: Type of criteria
            criteria_value: Value of criteria
            
        Returns:
            Display name for the criteria value
        """
        try:
            # Handle None values
            if criteria_value is None:
                return "None"
                
            # Handle list values
            if isinstance(criteria_value, list):
                names = [self._get_criteria_name(criteria_type, v) for v in criteria_value]
                return ", ".join(names)
                
            # Get options from field manager
            options = self.field_manager.get_options(criteria_type)
            
            # Find the option with this ID
            name = str(criteria_value)  # Default if not found
            for option in options:
                if option.id == criteria_value:
                    name = option.name
                    break
                    
            return name
        except Exception as e:
            st.write(f"Debug: Error getting criteria name for {criteria_type}: {str(e)}")
            return str(criteria_value)  # Fallback to string representation
            
    def invalidate_cache(self):
        """Invalidate the optimizer cache.
        
        This method invalidates both the components cache and the data cache,
        forcing a refresh on the next request.
        
        Returns:
            True if cache was invalidated successfully, False otherwise
        """
        if not self.initialized or not self.optimizer:
            st.error("Optimizer not initialized. Cannot invalidate cache.")
            return False
            
        try:
            self.optimizer.invalidate_cache(components=True, data=True)
            return True
        except Exception as e:
            st.error(f"Error invalidating cache: {str(e)}")
            return False
