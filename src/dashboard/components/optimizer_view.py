"""Show Optimizer View Component.

This module contains the main view component for the Show Optimizer feature.
It handles the UI presentation of the optimizer results and user interactions.
"""

import streamlit as st
import pandas as pd
import altair as alt
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from data_processing.show_optimizer.show_optimizer import ShowOptimizer
from data_processing.show_optimizer.suggestion_analyzer import OptimizationSummary, Recommendation, SuggestionAnalyzer
from data_processing.show_optimizer.criteria_analyzer import CriteriaAnalyzer
from data_processing.show_optimizer.criteria_scorer import CriteriaScorer
from data_processing.analyze_shows import ShowsAnalyzer
from data_processing.success_analysis import SuccessAnalyzer

# Import helper functions
from dashboard.components.optimizer_helpers import (
    get_id_for_name, get_ids_for_names,
    render_success_metrics, render_metric_card, render_info_card,
    render_success_factors, render_network_compatibility, group_recommendations,
    render_recommendation_group
)

# No logger needed for deployed app

class OptimizerView:
    """Main view component for the Show Optimizer."""
    
    def __init__(self):
        """Initialize the optimizer view."""
        self.optimizer = ShowOptimizer()
        self.initialized = False
        
        # Initialize field options in session state if not already present
        if 'optimizer_field_options' not in st.session_state:
            st.session_state.optimizer_field_options = {}
            
        if 'optimizer_display_options' not in st.session_state:
            st.session_state.optimizer_display_options = {}
            
    def initialize(self) -> bool:
        """Initialize the optimizer components.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if not self.initialized:
            try:
                with st.spinner("Initializing Show Optimizer..."):
                    # Bypass the cached initialize method and directly initialize components
                    # This matches how Comp Builder initializes and avoids st.cache_data issues
                    try:
                        # Initialize components directly (similar to ShowOptimizer.initialize but without cache)
                        self.optimizer.criteria_scorer = CriteriaScorer(self.optimizer.shows_analyzer, self.optimizer.success_analyzer)
                        self.optimizer.field_manager = self.optimizer.criteria_scorer.field_manager
                        self.optimizer.criteria_analyzer = CriteriaAnalyzer(self.optimizer.shows_analyzer, self.optimizer.success_analyzer)
                        self.optimizer.suggestion_analyzer = SuggestionAnalyzer(self.optimizer.shows_analyzer, self.optimizer.success_analyzer)
                        self.optimizer.initialized = True
                        self.optimizer.last_update = datetime.now()
                    except Exception as e:
                        st.error(f"Failed to initialize optimizer components: {str(e)}")
                        return False
                    
                    # Verify field_manager is available
                    if not hasattr(self.optimizer, 'field_manager') or self.optimizer.field_manager is None:
                        st.error("Failed to initialize Show Optimizer. Field manager is not available.")
                        st.write("⚠️ Please try refreshing the page or contact support if the problem persists.")
                        return False
                    
                    # Cache field options in session state (similar to Comp Builder)
                    field_names = ["genre", "character_types", "source_type", "thematic_elements", "plot_elements", 
                                  "tone", "time_setting", "location_setting", "network", "studios", "team_members", "order_type"]
                    
                    # Get all field options and store in session state
                    for field_name in field_names:
                        try:
                            options = self.optimizer.get_field_options(field_name)
                            if options:
                                st.session_state.optimizer_field_options[field_name] = options
                                # Also create display options (id, name) tuples for dropdowns
                                st.session_state.optimizer_display_options[field_name] = [
                                    (option.id, option.name) for option in options
                                ]
                        except Exception:
                            # Skip warning for deployed app
                            pass
                    
                    self.initialized = True
            except Exception as e:
                # Display user-friendly error
                st.error(f"Failed to initialize Show Optimizer: {str(e)}")
                st.write("⚠️ The application requires database access to function properly.")
                st.write("Please try again later or contact support if the problem persists.")
                return False
                
        return self.initialized
    
    def render(self):
        """Render the optimizer view."""
        # Initialize optimizer
        if not self.initialize():
            return
            
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Concept Builder", "Network Analysis", "Recommendations"])
        
        with tab1:
            self._render_concept_builder()
            
        with tab2:
            self._render_network_analysis()
            
        with tab3:
            self._render_recommendations()
    
    def _render_concept_builder(self):
        """Render the concept builder section."""
        # Get criteria from session state or initialize
        if "optimizer_criteria" not in st.session_state:
            st.session_state.optimizer_criteria = {}
            
        criteria = st.session_state.optimizer_criteria
        
        # Get display options from session state
        display_options = st.session_state.get('optimizer_display_options', {})
        
        # Check if field options are available in session state
        if not display_options:
            st.error("Unable to load field options from the database.")
            st.info("This may be due to a temporary connection issue or database maintenance.")
            
            if st.button("Retry Initialization", type="primary"):
                self.initialized = False
                self.initialize()
                
            st.write("If the problem persists, please try again later or contact support.")
            return
        
        st.header("Build Your Show Concept")
        
        # Create a 1:2 column layout (similar to comp builder)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Content criteria section
            with st.expander("Content Criteria (82 pts)", expanded=True):
                st.markdown("### Content")
                
                # Genre selection
                genre_name = st.selectbox("Genre", 
                    options=[name for _, name in display_options.get('genre', []) if name and name.strip()],
                    key="optimizer_genre", index=None, placeholder="Select genre...")
                criteria["genre"] = get_id_for_name(genre_name, display_options.get('genre', [])) if genre_name else None
                
                # Subgenre selection (if available)
                if 'subgenres' in display_options:
                    subgenre_names = st.multiselect("Subgenres",
                        options=[name for _, name in display_options['subgenres'] if name and name.strip()],
                        key="optimizer_subgenres", placeholder="Select subgenres...")
                    criteria["subgenres"] = get_ids_for_names(subgenre_names, display_options['subgenres'])
                
                # Source type selection
                source_name = st.selectbox("Source Type",
                    options=[name for _, name in display_options.get('source_type', []) if name and name.strip()],
                    key="optimizer_source_type", index=None, placeholder="Select source type...")
                criteria["source_type"] = get_id_for_name(source_name, display_options.get('source_type', [])) if source_name else None
                
                # Character types selection
                char_names = st.multiselect("Character Types",
                    options=[name for _, name in display_options.get('character_types', []) if name and name.strip()],
                    key="optimizer_character_types", placeholder="Select character types...")
                criteria["character_types"] = get_ids_for_names(char_names, display_options.get('character_types', []))
                
                # Plot elements selection
                plot_names = st.multiselect("Plot Elements",
                    options=[name for _, name in display_options.get('plot_elements', []) if name and name.strip()],
                    key="optimizer_plot_elements", placeholder="Select plot elements...")
                criteria["plot_elements"] = get_ids_for_names(plot_names, display_options.get('plot_elements', []))
                
                # Theme selection
                theme_names = st.multiselect("Theme Elements",
                    options=[name for _, name in display_options.get('thematic_elements', []) if name and name.strip()],
                    key="optimizer_theme", placeholder="Select theme elements...")
                criteria["thematic_elements"] = get_ids_for_names(theme_names, display_options.get('thematic_elements', []))
                
                # Tone selection
                tone_name = st.selectbox("Tone",
                    options=[name for _, name in display_options.get('tone', []) if name and name.strip()],
                    key="optimizer_tone", index=None, placeholder="Select tone...")
                criteria["tone"] = get_id_for_name(tone_name, display_options.get('tone', [])) if tone_name else None
                
                # Time setting selection
                time_name = st.selectbox("Time Setting",
                    options=[name for _, name in display_options.get('time_setting', []) if name and name.strip()],
                    key="optimizer_time_setting", index=None, placeholder="Select time setting...")
                criteria["time_setting"] = get_id_for_name(time_name, display_options.get('time_setting', [])) if time_name else None
                
                # Location setting selection
                loc_name = st.selectbox("Location",
                    options=[name for _, name in display_options.get('location_setting', []) if name and name.strip()],
                    key="optimizer_location_setting", index=None, placeholder="Select location...")
                criteria["location_setting"] = get_id_for_name(loc_name, display_options.get('location_setting', [])) if loc_name else None
            
            # Production criteria section
            with st.expander("Production Criteria (13 pts)", expanded=True):
                st.markdown("### Production")
                
                # Network selection
                network_name = st.selectbox("Network",
                    options=[name for _, name in display_options.get('network', []) if name and name.strip()],
                    key="optimizer_network", index=None, placeholder="Select network...")
                criteria["network"] = get_id_for_name(network_name, display_options.get('network', [])) if network_name else None
                
                # Studios selection
                studio_names = st.multiselect("Studios",
                    options=[name for _, name in display_options.get('studios', []) if name and name.strip()],
                    key="optimizer_studios", placeholder="Select studios...")
                criteria["studios"] = get_ids_for_names(studio_names, display_options.get('studios', []))
                
                # Team members selection (special handling for team members)
                if 'team_members' in display_options:
                    team_names = st.multiselect("Team Members", 
                        options=[name for _, name in display_options['team_members'] if name and name.strip()],
                        key="optimizer_team_members", placeholder="Select team members...")
                    
                    # Get IDs for team member names
                    team_member_ids = get_ids_for_names(team_names, display_options['team_members'], 'team_members')
                    
                    # Store IDs in both fields for compatibility
                    criteria["team_members"] = team_member_ids
                    criteria["team_member_ids"] = team_member_ids
                    # Also store the names for display
                    criteria["team_member_names"] = team_names
            
            # Format criteria section
            with st.expander("Format Criteria (5 pts)", expanded=True):
                st.markdown("### Format")
                
                # Episode Count
                eps = st.number_input("Episode Count", min_value=1, max_value=100, value=criteria.get("episode_count"),
                    help="Episode count for the show")
                if eps is not None and eps > 0:
                    criteria["episode_count"] = eps
                else:
                    if "episode_count" in criteria:
                        del criteria["episode_count"]
                
                # Order Type
                if 'order_type' in display_options:
                    order_name = st.selectbox("Order Type",
                        options=[name for _, name in display_options['order_type'] if name and name.strip()],
                        key="optimizer_order_type", index=None, placeholder="Select order type...")
                    criteria["order_type"] = get_id_for_name(order_name, display_options['order_type']) if order_name else None
            
            # Run analysis automatically when criteria changes
            if criteria:
                self._run_analysis()
                
        with col2:
            # Results section
            st.subheader("Analysis Results")
            
            # If we have criteria, show results
            if criteria:
                # Get results from session state if available
                if "optimizer_summary" in st.session_state:
                    self._render_results()
                else:
                    st.info("Select or adjust criteria on the left to analyze your show concept.")
            else:
                st.info("Select criteria on the left to analyze your show concept.")
                
    def _run_analysis(self):
        """Run the analysis with the current criteria."""
        # Get criteria from session state
        criteria = st.session_state.optimizer_criteria
        
        # Check if optimizer is initialized
        if not self.initialized:
            st.error("Show Optimizer is not initialized. Please refresh the page and try again.")
            return
        
        # Check if field_manager is available
        if not hasattr(self.optimizer, 'field_manager') or self.optimizer.field_manager is None:
            try:
                # Try to reinitialize
                with st.spinner("Attempting to reinitialize Show Optimizer..."):
                    self.initialized = self.optimizer.initialize(force_refresh=True)
                if not self.initialized or not hasattr(self.optimizer, 'field_manager') or self.optimizer.field_manager is None:
                    st.error("Could not initialize field manager. Please try again later.")
                    return
            except Exception as e:
                st.error(f"Error connecting to database: {str(e)}")
                return
        
        # Validate and analyze criteria
        if not criteria:
            st.info("Select criteria to analyze your concept.")
            return
            
        try:
            # Simple validation
            if not any([criteria.get("genre"), criteria.get("character_types")]):
                st.warning("Please select at least one genre or character type.")
                return
                
            # Run the analysis
            with st.spinner("Analyzing concept..."):
                summary = self.optimizer.analyze_concept(criteria)
            
            # Store results in session state
            if summary:
                st.session_state.optimizer_summary = summary
                st.session_state.optimizer_results = True
            else:
                st.info("No recommendations found for the selected criteria.")
                st.session_state.optimizer_results = False
                if "optimizer_summary" in st.session_state:
                    del st.session_state.optimizer_summary
                    
        except Exception as e:
            st.error(f"Error analyzing concept: {str(e)}")
            st.session_state.optimizer_results = False
            if "optimizer_summary" in st.session_state:
                del st.session_state.optimizer_summary
            
    def _render_results(self):
        """Render the analysis results."""
        if "optimizer_summary" not in st.session_state:
            return
            
        summary = st.session_state.optimizer_summary
        
        # Display success metrics
        if hasattr(summary, 'success_metrics') and summary.success_metrics:
            st.markdown("### Success Metrics")
            render_success_metrics(summary)
            
        # Display recommendations
        if hasattr(summary, 'recommendations') and summary.recommendations:
            st.markdown("### Recommendations")
            for i, rec in enumerate(summary.recommendations):
                with st.expander(f"Recommendation {i+1}: {rec.title}", expanded=i==0):
                    st.markdown(f"**Score:** {rec.score:.2f}")
                    st.markdown(f"**Description:** {rec.description}")
                    
                    # Display match details if available
                    if hasattr(rec, 'match_details') and rec.match_details:
                        st.markdown("**Match Details:**")
                        for category, details in rec.match_details.items():
                            st.markdown(f"*{category}:* {details}")
        else:
            st.info("No recommendations available. Try adjusting your criteria.")
            
        # Add button to reset criteria
        if st.button("Reset Criteria", key="reset_criteria_button"):
            st.session_state.optimizer_criteria = {}
            if "optimizer_summary" in st.session_state:
                del st.session_state.optimizer_summary
            st.experimental_rerun()
    

    
    def _render_concept_analysis(self, summary: OptimizationSummary):
        """Render the concept analysis section.
        
        Args:
            summary: Optimization summary
        """
        st.subheader("Concept Analysis")
        
        # Display success metrics using helper function
        render_success_metrics(summary)
        
        # Display success factors using helper function
        st.subheader("Success Factors")
        render_success_factors(summary.success_factors)
    
    def _render_network_analysis(self):
        """Render the network analysis section."""
        st.header("Network Analysis")
        
        # Check if we have criteria and summary
        if "optimizer_criteria" not in st.session_state or "optimizer_summary" not in st.session_state:
            st.info("Please build and analyze your concept first.")
            return
            
        criteria = st.session_state.optimizer_criteria
        summary = st.session_state.optimizer_summary
        
        # Display network tiers
        st.subheader("Network Compatibility")
        
        # Get network tiers
        network_tiers = self.optimizer.get_network_tiers(criteria)
        
        if network_tiers:
            # Create tabs for each tier
            tier_tabs = st.tabs(list(network_tiers.keys()))
            
            for i, (tier_name, networks) in enumerate(network_tiers.items()):
                with tier_tabs[i]:
                    # Use helper function to render network compatibility table
                    render_network_compatibility(networks)
        else:
            st.info("No network compatibility data available.")
        
        # Display network-specific analysis for top network
        if summary.top_networks:
            st.subheader(f"Network-Specific Analysis: {summary.top_networks[0].network_name}")
            
            # Get network-specific recommendations
            network_recs = [rec for rec in summary.recommendations 
                          if "works particularly well for" in rec.explanation 
                          or "performs" in rec.explanation and "worse than average for" in rec.explanation]
            
            if network_recs:
                for rec in network_recs[:3]:  # Show top 3
                    render_info_card(
                        f"{rec.recommendation_type.capitalize()}: {rec.criteria_type.replace('_', ' ').title()}",
                        rec.explanation
                    )
            else:
                st.info("No network-specific recommendations available.")
    
    def _render_recommendations(self):
        """Render the recommendations section."""
        st.header("Optimization Recommendations")
        
        # Check if we have criteria and summary
        if "optimizer_criteria" not in st.session_state or "optimizer_summary" not in st.session_state:
            st.info("Please build and analyze your concept first.")
            return
            
        criteria = st.session_state.optimizer_criteria
        summary = st.session_state.optimizer_summary
        
        # Display recommendations
        if summary.recommendations:
            # Group recommendations by type
            grouped_recs = group_recommendations(summary.recommendations)
            
            # Render each recommendation group
            for rec_type, recs in grouped_recs.items():
                render_recommendation_group(rec_type, recs, self._apply_recommendation)
        else:
            st.info("No recommendations available.")
    
    def _apply_recommendation(self, rec: Recommendation):
        """Apply a recommendation to the criteria.
        
        Args:
            rec: Recommendation to apply
        """
        if "optimizer_criteria" not in st.session_state:
            return
            
        criteria = st.session_state.optimizer_criteria
        
        if rec.recommendation_type == "add":
            # Add the suggested value
            if rec.criteria_type in criteria:
                if isinstance(criteria[rec.criteria_type], list):
                    if rec.suggested_value not in criteria[rec.criteria_type]:
                        criteria[rec.criteria_type].append(rec.suggested_value)
                else:
                    criteria[rec.criteria_type] = rec.suggested_value
            else:
                criteria[rec.criteria_type] = rec.suggested_value
                
        elif rec.recommendation_type == "replace":
            # Replace the current value with the suggested value
            if rec.criteria_type in criteria:
                if isinstance(criteria[rec.criteria_type], list):
                    if rec.current_value in criteria[rec.criteria_type]:
                        criteria[rec.criteria_type].remove(rec.current_value)
                    criteria[rec.criteria_type].append(rec.suggested_value)
                else:
                    criteria[rec.criteria_type] = rec.suggested_value
                    
        elif rec.recommendation_type == "remove":
            # Remove the current value
            if rec.criteria_type in criteria:
                if isinstance(criteria[rec.criteria_type], list):
                    if rec.current_value in criteria[rec.criteria_type]:
                        criteria[rec.criteria_type].remove(rec.current_value)
                        if not criteria[rec.criteria_type]:
                            del criteria[rec.criteria_type]
                else:
                    del criteria[rec.criteria_type]
        
        # Update session state
        st.session_state.optimizer_criteria = criteria
        
        # Re-analyze concept
        with st.spinner("Updating analysis..."):
            summary = self.optimizer.analyze_concept(criteria)
            
            if summary:
                st.session_state.optimizer_summary = summary
