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
from data_processing.show_optimizer.models import OptimizationSummary, Recommendation
from data_processing.show_optimizer.criteria_analyzer import CriteriaAnalyzer
from data_processing.show_optimizer.criteria_scorer import CriteriaScorer

# Import helper functions
from dashboard.components.optimizer_helpers import (
    get_id_for_name, get_ids_for_names, 
    render_select_field, render_multiselect_field,
    render_metric_card, render_info_card,
    render_success_metrics, render_success_factors, render_network_compatibility,
    group_recommendations
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
                    # Try to initialize with force_refresh=True to ensure we get fresh data
                    self.initialized = self.optimizer.initialize(force_refresh=True)
                    
                    # Check if initialization was successful and field_manager is available
                    if not self.initialized or not hasattr(self.optimizer, 'field_manager') or self.optimizer.field_manager is None:
                        st.error("Failed to initialize Show Optimizer. This may be due to database connection issues.")
                        st.write("⚠️ The application requires database access to function properly.")
                        st.write("Please try again later or contact support if the problem persists.")
                        return False
                    
                    # Cache field options in session state (similar to Comp Builder)
                    field_names = ["genre", "character_types", "source_type", "theme", "plot_elements", 
                                  "tone", "time_setting", "location_setting", "network", "studios", "team_members"]
                    
                    # Get all field options and store in session state
                    for field_name in field_names:
                        try:
                            # Add extra safety check for the optimizer and field_manager
                            if hasattr(self.optimizer, 'field_manager') and self.optimizer.field_manager is not None:
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
            except Exception as e:
                # Display user-friendly error
                st.error(f"An error occurred during initialization: {str(e)}")
                st.write("⚠️ The Show Optimizer requires database access to function properly.")
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
        st.header("Build Your Show Concept")
        
        # Get criteria from session state or initialize
        if "optimizer_criteria" not in st.session_state:
            st.session_state.optimizer_criteria = {}
            
        criteria = st.session_state.optimizer_criteria
        
        # Check if field options are available in session state
        if not st.session_state.optimizer_display_options:
            st.error("Unable to load field options from the database.")
            st.info("This may be due to a temporary connection issue or database maintenance.")
            
            if st.button("Retry Initialization", type="primary"):
                self.initialized = False
                self.initialize()
                
            st.write("If the problem persists, please try again later or contact support.")
            return
        
        # Create form for concept builder
        with st.form("concept_builder_form"):
            st.subheader("Build Your Show Concept")
            
            # Create a 1:2 column layout (similar to comp builder)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Content criteria section
                with st.expander("Content Criteria", expanded=True):
                    st.markdown("### Content")
                    
                    # Genre selection
                    render_select_field('genre', 'Genre', display_options, criteria)
                    
                    # Subgenre selection (if available)
                    if 'subgenres' in display_options:
                        render_multiselect_field('subgenres', 'Subgenres', display_options, criteria)
                    
                    # Source type selection
                    render_select_field('source_type', 'Source Type', display_options, criteria)
                    
                    # Character types selection
                    render_multiselect_field('character_types', 'Character Types', display_options, criteria)
                    
                    # Plot elements selection
                    render_multiselect_field('plot_elements', 'Plot Elements', display_options, criteria)
                    
                    # Theme selection
                    render_multiselect_field('theme', 'Theme Elements', display_options, criteria)
                    
                    # Tone selection
                    render_select_field('tone', 'Tone', display_options, criteria)
                    
                    # Time setting selection
                    render_select_field('time_setting', 'Time Setting', display_options, criteria)
                    
                    # Location setting selection
                    render_select_field('location_setting', 'Location', display_options, criteria)
                
                # Production criteria section
                with st.expander("Production Criteria", expanded=True):
                    st.markdown("### Production")
                    
                    # Network selection
                    render_select_field('network', 'Network', display_options, criteria)
                    
                    # Studios selection
                    render_multiselect_field('studios', 'Studios', display_options, criteria)
                    
                    # Team members selection - using basic approach for deployment compatibility
                    if 'team_members' in display_options:
                        team_names = [name for _, name in display_options['team_members'] if name and name.strip()]
                        
                        selected_teams = st.multiselect(
                            "Team Members", 
                            options=team_names,
                            default=criteria.get("team_member_names", []),
                            placeholder="Select team members..."
                        )
                        
                        # Update criteria with selected team members - preserving special field naming
                        if selected_teams:
                            # Use basic ID lookup for deployment compatibility
                            team_member_ids = []
                            for name in selected_teams:
                                id = next((id for id, opt_name in display_options['team_members'] 
                                         if opt_name == name), None)
                                if id is not None:
                                    team_member_ids.append(id)
                            
                            # Store IDs in both fields for compatibility
                            criteria["team_members"] = team_member_ids
                            criteria["team_member_ids"] = team_member_ids  # For compatibility
                            criteria["team_member_names"] = selected_teams  # Store names for display
                        else:
                            if "team_members" in criteria:
                                del criteria["team_members"]
                            if "team_member_ids" in criteria:
                                del criteria["team_member_ids"]
                            if "team_member_names" in criteria:
                                del criteria["team_member_names"]
                
                # Format criteria section
                with st.expander("Format Criteria", expanded=True):
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
                        render_select_field('order_type', 'Order Type', display_options, criteria)
            
            # Add a clear indication that the form requires submission
            st.write("")
            st.write("👇 Click the button below to analyze your concept")
            
            # Make the submit button more prominent
            submitted = st.form_submit_button("ANALYZE CONCEPT", type="primary")
            
            if submitted:
                # Save criteria to session state
                st.session_state.optimizer_criteria = criteria
                
                # Check if optimizer is initialized
                if not self.initialized:
                    st.error("Show Optimizer is not initialized. Please refresh the page and try again.")
                    return
                
                # Check if field_manager is available
                if not hasattr(self.optimizer, 'field_manager') or self.optimizer.field_manager is None:
                    try:
                        # Try to reinitialize
                        st.warning("Attempting to reinitialize Show Optimizer...")
                        self.initialized = self.optimizer.initialize(force_refresh=True)
                        if not self.initialized or not hasattr(self.optimizer, 'field_manager') or self.optimizer.field_manager is None:
                            st.error("Could not initialize field manager. Please refresh the page and try again.")
                            st.write("⚠️ The application requires database access to function properly.")
                            return
                    except Exception as e:
                        st.error(f"Error reinitializing: {str(e)}")
                        st.write("⚠️ Database connection may be unavailable.")
                        # Skip error logging for deployed app
                        return
                
                # Validate and analyze criteria
                if criteria:
                    try:
                        # First, validate criteria using cached field options
                        validation_errors = {}
                        
                        # Check for required fields
                        if not criteria.get("genre") and not criteria.get("character_types"):
                            validation_errors["criteria"] = "Please select at least one genre or character type"
                        
                        # If there are validation errors from the UI, show them
                        if validation_errors:
                            st.session_state.validation_errors = validation_errors
                            st.experimental_rerun()
                            return
                            
                        # Now use the backend validation
                        normalized_criteria, backend_validation_errors = self.optimizer.validate_criteria(criteria)
                        
                        # If there are validation errors from the backend, store them and rerun
                        if backend_validation_errors:
                            st.session_state.validation_errors = backend_validation_errors
                            st.experimental_rerun()
                            return
                        
                        # Analyze concept if validation passed
                        with st.spinner("Analyzing concept..."):
                            try:
                                summary = self.optimizer.analyze_concept(normalized_criteria)
                                
                                if summary:
                                    st.session_state.optimizer_summary = summary
                                    st.experimental_rerun()
                                else:
                                    st.error("Failed to analyze concept.")
                                    st.write("⚠️ The analyzer couldn't process your criteria. This could be due to:")
                                    st.write("- Missing database connection")
                                    st.write("- Insufficient data for the selected criteria")
                                    st.write("- Internal processing error")
                                    st.write("Try selecting different criteria or check if all components are properly initialized.")
                            except Exception as e:
                                st.error(f"Error analyzing concept: {str(e)}")
                                st.write("⚠️ An unexpected error occurred while analyzing your concept.")
                                st.write("Try selecting different criteria or fewer options.")
                                # Log the full error for developers
                                # Skip error logging for deployed app
                    except Exception as e:
                        st.error(f"Error validating criteria: {str(e)}")
                        st.write("⚠️ There was a problem with your selected criteria.")
                        st.write("This may be due to database connection issues or invalid selections.")
                        # Log the full error for developers
                        # Skip error logging for deployed app
        
        # Display concept analysis if available
        if "optimizer_summary" in st.session_state:
            self._render_concept_analysis(st.session_state.optimizer_summary)
    
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
