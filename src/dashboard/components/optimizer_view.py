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
                    # Format the summary for UI display - this method already sets summary.formatted_data
                    summary = self.format_optimization_summary(summary)
                    
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
            # Only show stack trace in debug mode
            if st.session_state.get('debug_mode', False):
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
            # Create a new formatted_data attribute as a dictionary
            # Important: Use a different attribute name to avoid recursive reference
            setattr(summary, '_formatted_data_dict', {})
            
            # Format network matches if available
            if hasattr(summary, 'top_networks') and summary.top_networks:
                summary._formatted_data_dict['networks'] = self._format_network_matches(summary.top_networks)
                
            # Format component scores if available
            if hasattr(summary, 'component_scores') and summary.component_scores:
                summary._formatted_data_dict['component_scores'] = self._format_component_scores(summary.component_scores)
                
            # Format match quality if available
            if hasattr(summary, 'match_quality') and summary.match_quality:
                summary._formatted_data_dict['match_quality'] = self._format_match_quality(summary.match_quality)
                
            # Format success factors if available
            if hasattr(summary, 'success_factors') and summary.success_factors:
                summary._formatted_data_dict['success_factors'] = self._format_success_factors(summary.success_factors)
                
            # Format recommendations if available
            if hasattr(summary, 'recommendations') and summary.recommendations:
                summary._formatted_data_dict['recommendations'] = self._format_recommendations(summary.recommendations)
                
            # Format matching shows if available
            if hasattr(summary, 'matching_shows') and not summary.matching_shows.empty:
                if OptimizerConfig.DEBUG_MODE:
                    # Debug removed for clarity
                    if 'match_level' in summary.matching_shows.columns:
                        # Debug removed for clarity
                        pass
                    else:
                        OptimizerConfig.debug("WARNING - match_level column is missing from matching_shows DataFrame in OptimizerView!", category='recommendation', force=True)
                
                # Format the matching shows for UI display
                formatted_shows = self._format_matching_shows(summary.matching_shows)
                
                # Store the formatted shows in the summary
                summary._formatted_data_dict['matching_shows'] = formatted_shows
                
                # Replace the original DataFrame with the formatted one
                summary.matching_shows = formatted_shows
            
            # Format success probability if available AND we have valid component scores
            has_valid_component_scores = (
                hasattr(summary, 'component_scores') and 
                summary.component_scores and 
                'success' in summary.component_scores and 
                summary.component_scores['success'] is not None and
                summary.component_scores['success'].score is not None
            )
            
            if has_valid_component_scores and hasattr(summary, 'overall_success_probability'):
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
                
                summary._formatted_data_dict['success_probability'] = {
                    "value": probability,
                    "display": display,
                    "confidence": confidence,
                    "subtitle": subtitle
                }
            
            # Set the formatted_data property to the dictionary
            summary.formatted_data = summary._formatted_data_dict
            
            return summary
            
        except Exception as e:
            st.error(f"Error formatting optimization summary: {str(e)}")
            # Ensure formatted_data is a dictionary even after an error
            summary.formatted_data = {}
            return summary
    
    def _format_success_probability(self, probability: Optional[float], confidence: str) -> Dict[str, Any]:
        """
        Format success probability for display.
        
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
        """
        Format recommendations for display.
        
        Args:
            recommendations: List of Recommendation objects
            
        Returns:
            Dictionary with formatted recommendations grouped by type
        """
        # Initialize with standard recommendation types
        config = OptimizerConfig()
        grouped = {
            'add': [],
            'change': [],
            'remove': [],
            'network_specific': []
        }
        
        # Add any additional types from config
        for rec_type in config.RECOMMENDATION_TYPES.keys():
            if rec_type not in grouped:
                grouped[rec_type] = []
            
        # Track network-specific recommendations separately
        network_specific = []
        
        # Debug log the recommendations count
        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"Formatting {len(recommendations)} recommendations", category='recommendation', force=True)
            if recommendations:
                # Count recommendations by type
                type_counts = {}
                for rec in recommendations:
                    rec_type = getattr(rec, 'recommendation_type', 'unknown')
                    if rec_type not in type_counts:
                        type_counts[rec_type] = 0
                    type_counts[rec_type] += 1
                
                OptimizerConfig.debug(f"Recommendation types: {type_counts}", category='recommendation', force=True)
                
                # Special debug for 'remove' recommendations
                remove_recs = [rec for rec in recommendations if getattr(rec, 'recommendation_type', '') == 'remove']
                if remove_recs:
                    OptimizerConfig.debug(f"Found {len(remove_recs)} 'remove' recommendations before formatting", category='recommendation', force=True)
                    # Inspect the first few remove recommendations in detail
                    for i, rec in enumerate(remove_recs[:3]):
                        OptimizerConfig.debug(f"Remove recommendation {i+1} details:", category='recommendation', force=True)
                        OptimizerConfig.debug(f"  - Type: {getattr(rec, 'recommendation_type', 'unknown')}", category='recommendation', force=True)
                        OptimizerConfig.debug(f"  - Criteria Type: {getattr(rec, 'criteria_type', 'unknown')}", category='recommendation', force=True)
                        OptimizerConfig.debug(f"  - Suggested Name: {getattr(rec, 'suggested_name', 'unknown')}", category='recommendation', force=True)
                        OptimizerConfig.debug(f"  - Impact: {getattr(rec, 'impact_score', 0)}", category='recommendation', force=True)
                    for rec in remove_recs:
                        OptimizerConfig.debug(f"Remove rec: {getattr(rec, 'criteria_type', 'unknown')}/{getattr(rec, 'suggested_name', 'unknown')}", category='recommendation', force=True)
                else:
                    OptimizerConfig.debug("No 'remove' recommendations found in original recommendations", category='recommendation', force=True)
                
                # Show the first few recommendations in detail
                for i, rec in enumerate(recommendations[:3]):
                    OptimizerConfig.debug(f"Recommendation {i+1} details:", category='recommendation', force=True)
                    OptimizerConfig.debug(f"  - Type: {getattr(rec, 'recommendation_type', 'unknown')}", category='recommendation')
                    OptimizerConfig.debug(f"  - Criteria Type: {getattr(rec, 'criteria_type', 'unknown')}", category='recommendation')
                    OptimizerConfig.debug(f"  - Suggested Name: {getattr(rec, 'suggested_name', 'unknown')}", category='recommendation')
                    OptimizerConfig.debug(f"  - Impact: {getattr(rec, 'impact_score', 0)}", category='recommendation')
                    OptimizerConfig.debug(f"  - Explanation: {getattr(rec, 'explanation', '')}", category='recommendation')
                    # Debug all attributes
                    # Debug removed for clarity
                
        # If no recommendations, add debug output
        if not recommendations:
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug("No recommendations to format", category='recommendation', force=True)
            # Return empty structure to avoid errors
            return {
                "grouped": grouped,
                "network_specific": [],
                "all": []
            }
            
        formatted_recommendations = []
        
        for rec in recommendations:
            # Skip invalid recommendations
            if not hasattr(rec, 'recommendation_type') and not hasattr(rec, 'rec_type'):
                continue
                
            # Get recommendation type, prioritizing recommendation_type over rec_type
            rec_type = getattr(rec, 'recommendation_type', getattr(rec, 'rec_type', None))
            
            # Ensure we have a valid recommendation type
            if not rec_type:
                # Default to 'add' if no type is specified
                rec_type = 'add'
                
            # We no longer need special handling for 'Remove' recommendations
            # The recommendation engine now properly assigns recommendation types
                
            # Ensure the recommendation type is one of our standard types
            if rec_type not in ['add', 'change', 'remove'] and not rec_type.startswith('network_'):
                # Default to 'add' for unknown types
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Unknown recommendation type '{rec_type}' defaulting to 'add'", category='recommendation')
                rec_type = 'add'
                
            # Debug output for recommendation type
            if OptimizerConfig.DEBUG_MODE:
                # Debug removed for clarity
                pass
                
            # Format impact percentage for display
            impact_percent = abs(rec.impact_score * 100) if hasattr(rec, 'impact_score') and rec.impact_score is not None else 0
            impact_direction = "Increase" if getattr(rec, 'impact_score', 0) > 0 else "Decrease"
            
            # Debug the recommendation type
            if OptimizerConfig.DEBUG_MODE:
                # Debug removed for clarity
                pass
                
            # Special debug for 'remove' recommendations
            if rec_type == 'remove':
                OptimizerConfig.debug(f"REMOVE RECOMMENDATION FOUND IN OPTIMIZER_VIEW: {getattr(rec, 'criteria_type', 'unknown')}/{getattr(rec, 'suggested_name', 'unknown')}", category='recommendation', force=True)
            
            # Debug the recommendation attributes
            if OptimizerConfig.DEBUG_MODE:
                # Debug removed for clarity
                # Debug removed for clarity
                pass
            
            # Create recommendation title without impact percentage
            if rec_type and rec_type.startswith('network_'):
                # For network recommendations
                clean_rec_type = rec_type.replace('network_', '')
                # Extract network name from suggested_name if available
                network_name = ""
                if hasattr(rec, 'suggested_name') and rec.suggested_name:
                    if ':' in rec.suggested_name:
                        network_name = rec.suggested_name.split(':', 1)[0].strip()
                    else:
                        network_name = rec.suggested_name
                    
                    title = f"{network_name} - {clean_rec_type.capitalize()} {getattr(rec, 'criteria_type', '')}"
                else:
                    title = f"{clean_rec_type.capitalize()} {getattr(rec, 'criteria_type', '')} for network"
            else:
                # Format the title to include only the criteria type and suggested name
                criteria_type = getattr(rec, 'criteria_type', '').replace('_', ' ').title()
                suggested_name = getattr(rec, 'suggested_name', '')
                
                # Create a clean title without the impact information
                title = f"{criteria_type}: {suggested_name}"
                
            # Use the explanation directly from the recommendation object
            # The impact information should already be included when the recommendation was generated
            explanation = getattr(rec, 'explanation', '')
            
            # Make sure the explanation ends with a period for consistency
            if explanation and not explanation.strip().endswith('.'):
                explanation = f"{explanation}."
            
            # Create formatted recommendation dictionary with safe access to attributes
            formatted_rec = {
                # Display values
                "title": title,
                "description": explanation,
                "importance": getattr(rec, 'confidence', 'medium'),
                "category": rec_type,  # This is the key field for grouping
                "impact": getattr(rec, 'impact_score', 0),
                "criteria_type": getattr(rec, 'criteria_type', 'unknown'),
                "suggested_name": getattr(rec, 'suggested_name', ''),
                "current_value": getattr(rec, 'current_value', None),
                "suggested_value": getattr(rec, 'suggested_value', None),
                
                # Raw data for sorting and filtering
                "_impact_raw": getattr(rec, 'impact_score', 0),
                "_confidence_level": self._get_confidence_level(getattr(rec, 'confidence', 'medium')),
                "_rec_type": rec_type
            }
            
            # Debug output for 'remove' recommendations
            if rec_type == 'remove' and OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"Created formatted 'remove' recommendation: {title}", category='recommendation', force=True)
                OptimizerConfig.debug(f"  - Category set to: {formatted_rec['category']}", category='recommendation', force=True)
            
            # Add to formatted recommendations list
            formatted_recommendations.append(formatted_rec)
            
            # Add to appropriate group based on recommendation type
            if rec_type.startswith('network_'):
                # Handle network-specific recommendations
                network_specific.append(formatted_rec)
                
                # Ensure network-specific recommendations are properly grouped
                if 'network_specific' not in grouped:
                    grouped['network_specific'] = []
                grouped['network_specific'].append(formatted_rec)
                
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Added network recommendation to 'network_specific' group: {formatted_rec['title']}", category='recommendation', force=True)
                
            # Ensure the group exists
            if rec_type not in grouped:
                grouped[rec_type] = []
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Created missing group '{rec_type}' in grouped dictionary", category='recommendation', force=True)
                
            # Add the recommendation to its group
            grouped[rec_type].append(formatted_rec)
            
            # Debug output
            if OptimizerConfig.DEBUG_MODE:
                # Debug removed for clarity
                pass
                
            # Special debug for 'remove' recommendations
            if rec_type == 'remove' and OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"VERIFY: Added 'remove' recommendation to grouped dictionary", category='recommendation', force=True)
                OptimizerConfig.debug(f"  - Group 'remove' now has {len(grouped['remove'])} recommendations", category='recommendation', force=True)
        
        # Sort each group by impact score (descending)
        for rec_type in grouped:
            grouped[rec_type].sort(key=lambda x: abs(x["_impact_raw"]), reverse=True)
        
        # Debug log the formatted recommendations structure
        if OptimizerConfig.DEBUG_MODE:
            non_empty_groups = [k for k, v in grouped.items() if v]
            OptimizerConfig.debug(f"Formatted recommendations structure", category='recommendation')
            OptimizerConfig.debug(f"Non-empty groups: {non_empty_groups}", category='recommendation')
            OptimizerConfig.debug(f"Network-specific recommendations: {len(network_specific)}", category='recommendation')
            
            # Log counts for each recommendation type
            for rec_type in config.RECOMMENDATION_TYPES.keys():
                count = len(grouped.get(rec_type, []))
                OptimizerConfig.debug(f"Group '{rec_type}' has {count} recommendations", category='recommendation')
                if count > 0:
                    OptimizerConfig.debug(f"First item in group '{rec_type}': {grouped[rec_type][0]['title']}", category='recommendation')
                
        # Sort recommendations within each group by impact (absolute value)
        for rec_type, recs in grouped.items():
            if recs:
                recs.sort(key=lambda x: abs(x.get('_impact_raw', 0)), reverse=True)
        
        # Final debug output before returning
        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"Final grouped recommendations count: {sum(len(recs) for recs in grouped.values())}", category='recommendation', force=True)
            OptimizerConfig.debug(f"Final network_specific recommendations count: {len(network_specific)}", category='recommendation', force=True)
            OptimizerConfig.debug(f"Final all recommendations count: {len([rec for group in grouped.values() for rec in group] + network_specific)}", category='recommendation', force=True)
            OptimizerConfig.debug(f"Groups with recommendations: {[k for k, v in grouped.items() if v]}", category='recommendation', force=True)
            
            # Specifically check for remove recommendations
            remove_recs = grouped.get('remove', [])
            OptimizerConfig.debug(f"FINAL CHECK: Found {len(remove_recs)} 'remove' recommendations in grouped dictionary", category='recommendation', force=True)
            if remove_recs:
                for i, rec in enumerate(remove_recs[:3]):
                    OptimizerConfig.debug(f"  Remove rec {i+1}: {rec.get('title', 'No title')} - Category: {rec.get('category', 'unknown')}", category='recommendation', force=True)
            else:
                OptimizerConfig.debug("WARNING: No 'remove' recommendations found in final grouped dictionary!", category='recommendation', force=True)
            
            # Detailed debug for each group
            for group_name, group_recs in grouped.items():
                OptimizerConfig.debug(f"Group '{group_name}' has {len(group_recs)} recommendations", category='recommendation', force=True)
                if group_recs:
                    for i, rec in enumerate(group_recs[:3]):
                        pass
                        
            # Specifically check for change recommendations
            change_recs = [rec for rec in formatted_recommendations if rec.get('category') == 'change']
            OptimizerConfig.debug(f"Found {len(change_recs)} formatted 'change' recommendations", category='recommendation', force=True)
            if change_recs:
                for i, rec in enumerate(change_recs[:3]):
                    OptimizerConfig.debug(f"  Change rec {i+1}: {rec.get('title', 'No title')} - Impact: {rec.get('impact', 0)}", category='recommendation')
                    
            # Specifically check for remove recommendations in formatted recommendations
            remove_recs_formatted = [rec for rec in formatted_recommendations if rec.get('category') == 'remove']
            OptimizerConfig.debug(f"Found {len(remove_recs_formatted)} 'remove' recommendations in formatted_recommendations", category='recommendation', force=True)
            if remove_recs_formatted:
                for i, rec in enumerate(remove_recs_formatted[:3]):
                    OptimizerConfig.debug(f"  Remove rec in formatted {i+1}: {rec.get('title', 'No title')} - Category: {rec.get('category', 'unknown')}", category='recommendation', force=True)
        
        # Make sure 'remove' recommendations are included in the grouped dictionary
        # This is a critical check to ensure we don't lose any remove recommendations
        remove_recs_formatted = [rec for rec in formatted_recommendations if rec.get('category') == 'remove']
        if remove_recs_formatted and 'remove' not in grouped:
            grouped['remove'] = []
            
        # Add any remove recommendations that might have been missed
        for rec in remove_recs_formatted:
            if rec not in grouped.get('remove', []):
                if 'remove' not in grouped:
                    grouped['remove'] = []
                grouped['remove'].append(rec)
                if OptimizerConfig.DEBUG_MODE:
                    OptimizerConfig.debug(f"Added missing remove recommendation to grouped dictionary: {rec.get('title', 'unknown')}", category='recommendation', force=True)
        
        # Return the formatted recommendations
        # Ensure network-specific recommendations aren't duplicated in the 'all' list
        # Since they're already included in grouped['network_specific']
        if OptimizerConfig.DEBUG_MODE:
            OptimizerConfig.debug(f"Preparing final recommendations dictionary", category='recommendation', force=True)
            OptimizerConfig.debug(f"Network-specific count: {len(network_specific)}", category='recommendation', force=True)
            OptimizerConfig.debug(f"Grouped keys: {list(grouped.keys())}", category='recommendation', force=True)
            
        return {
            "grouped": grouped,
            "network_specific": network_specific,
            "all": [rec for group in grouped.values() for rec in group]
        }
    
    def _format_network_matches(self, network_matches: List[NetworkMatch]) -> List[Dict[str, Any]]:
        """Format network matches for display.
        
        Args:
            network_matches: List of NetworkMatch objects
            
        Returns:
            List of formatted network match dictionaries ready for direct display in UI
        """
        # Always add some basic debug output when in debug mode
        OptimizerConfig.debug(f"OptimizerView: Formatting {len(network_matches) if network_matches else 0} network matches", category='network')
        
        # Add detailed debug output
        if st.session_state.get('debug_mode', False):
            OptimizerConfig.debug("=== Network Match Formatting ===", category='network')
            OptimizerConfig.debug(f"Formatting {len(network_matches) if network_matches else 0} network matches", category='network')
            if network_matches and len(network_matches) > 0:
                OptimizerConfig.debug(f"First network match type: {type(network_matches[0]).__name__}", category='network')
                OptimizerConfig.debug(f"First network match dir: {dir(network_matches[0])}", category='network')
                OptimizerConfig.debug(f"First network match network_id: {getattr(network_matches[0], 'network_id', 'Not found')}", category='network')
                OptimizerConfig.debug(f"First network match network_name: {getattr(network_matches[0], 'network_name', 'Not found')}", category='network')
            OptimizerConfig.debug("----------------------------------------", category='network')
            
        formatted = []
        
        # Check if network_matches is None or empty
        if not network_matches:
            if st.session_state.get('debug_mode', False):
                OptimizerConfig.debug("No network matches to format", category='network')
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
                OptimizerConfig.debug(f"Formatting network {network_id} -> {network_name}", category='network')
            
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
                # Display the raw value directly as it's already in percentage form
                success_display = f"{success_value:.1f}%"
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
            OptimizerConfig.debug(f"Returning {len(formatted)} formatted network matches", category='network')
            
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
            # Skip None scores or scores with None values
            if score is None or score.score is None:
                continue
                
            # Format the score
            formatted[component] = {
                "score": score.score,
                "confidence": score.confidence,
                "sample_size": score.sample_size
            }
            
        return formatted
        
    def _get_match_level_description(self, level: int) -> str:
        """Get a human-readable description for a match level.
        
        Args:
            level: Match level (1-4)
            
        Returns:
            Human-readable description of the match level
        """
        # Use the config's match level descriptions if available
        if hasattr(self.config, 'MATCH_LEVELS') and level in self.config.MATCH_LEVELS:
            return self.config.MATCH_LEVELS[level].get('name', f"Level {level}")
        
        # If not in config, use a generic description
        return f"Match Level {level}"
        
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
        # Initialize empty groups for common recommendation types
        grouped = {
            'add': [],
            'change': [],
            'remove': []
        }
        return confidence_levels.get(confidence.lower(), 0)
        
    def _format_success_factors(self, success_factors: List[SuccessFactor]) -> List[Dict[str, Any]]:
        """Format success factors for display.
        
        Args:
            success_factors: List of success factors
            
        Returns:
            List of formatted success factors
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
            OptimizerConfig.debug(f"Error formatting match quality: {str(e)}", category='recommendation', force=True)
            return {
                "match_level": 0,
                "match_level_name": "Unknown",
                "match_count": 0,
                "match_counts_by_level": {},
                "confidence_score": 0.0
            }
    
    def _format_matching_shows(self, matching_shows: pd.DataFrame) -> pd.DataFrame:
        """Format matching shows for display in the UI.
        
        This method ensures that the DataFrame is properly formatted for UI consumption.
        The match_level column should already exist from the matching process.
        
        Args:
            matching_shows: DataFrame of matching shows
            
        Returns:
            Formatted DataFrame of matching shows
        """
        try:
            # Create a copy to avoid modifying the original
            formatted_shows = matching_shows.copy()
            
            # Ensure match_level is an integer type for proper sorting
            if 'match_level' in formatted_shows.columns:
                formatted_shows['match_level'] = formatted_shows['match_level'].astype(int)
            
            # Only generate match_level_desc if it doesn't exist but match_level does
            if 'match_level_desc' not in formatted_shows.columns and 'match_level' in formatted_shows.columns:
                # Generate descriptions based on match_level
                formatted_shows['match_level_desc'] = formatted_shows['match_level'].apply(
                    lambda x: self._get_match_level_description(x)
                )
            
            # Debug output to help diagnose match_level issues
            if OptimizerConfig.DEBUG_MODE:
                OptimizerConfig.debug(f"After formatting, matching_shows columns: {formatted_shows.columns.tolist()}", category='recommendation')
                if 'match_level' in formatted_shows.columns:
                    OptimizerConfig.debug(f"After formatting, match_level values: {formatted_shows['match_level'].value_counts().to_dict()}", category='recommendation')
                else:
                    OptimizerConfig.debug("WARNING - match_level column is missing from matching shows!", category='recommendation', force=True)
            
            return formatted_shows
            
        except Exception as e:
            if OptimizerConfig.DEBUG_MODE:
                st.error(f"Error formatting matching shows: {str(e)}")
            return matching_shows
        
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
            OptimizerConfig.debug(f"Error getting criteria name for {criteria_type}: {str(e)}", category='recommendation', force=True)
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
