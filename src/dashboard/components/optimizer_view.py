"""Show Optimizer View Component.

This module contains the main view component for the Show Optimizer feature.
It handles the UI presentation of the optimizer results and user interactions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, List, Any, Tuple, Optional
import logging

from ...data_processing.show_optimizer.show_optimizer import ShowOptimizer
from ...data_processing.show_optimizer.criteria_analyzer import SuccessFactor
from ...data_processing.show_optimizer.suggestion_analyzer import Recommendation, OptimizationSummary
from ...data_processing.show_optimizer.criteria_scorer import NetworkMatch, ComponentScore
from ..utils.style_config import render_metric_card, render_info_card, render_warning

logger = logging.getLogger(__name__)


class OptimizerView:
    """Main view component for the Show Optimizer."""
    
    def __init__(self):
        """Initialize the optimizer view."""
        self.optimizer = ShowOptimizer()
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize the optimizer components.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if not self.initialized:
            with st.spinner("Initializing Show Optimizer..."):
                self.initialized = self.optimizer.initialize()
                
                if not self.initialized:
                    st.error("Failed to initialize Show Optimizer. Please try again later.")
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
        
        # Create a form for criteria selection
        with st.form("concept_builder_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Genre selection
                genre_options = self.optimizer.get_field_options("genre")
                if genre_options:
                    genre_names = [option.name for option in genre_options]
                    genre_ids = [option.id for option in genre_options]
                    
                    # Get current selection
                    current_genre_ids = criteria.get("genre", [])
                    if not isinstance(current_genre_ids, list):
                        current_genre_ids = [current_genre_ids]
                        
                    # Convert IDs to indices
                    current_indices = []
                    for genre_id in current_genre_ids:
                        if genre_id in genre_ids:
                            current_indices.append(genre_ids.index(genre_id))
                    
                    selected_genres = st.multiselect(
                        "Genres", 
                        options=genre_names,
                        default=[genre_names[i] for i in current_indices] if current_indices else None
                    )
                    
                    # Update criteria with selected genres
                    if selected_genres:
                        selected_ids = [genre_ids[genre_names.index(name)] for name in selected_genres]
                        criteria["genre"] = selected_ids
                    else:
                        if "genre" in criteria:
                            del criteria["genre"]
                
                # Character types selection
                character_options = self.optimizer.get_field_options("character_types")
                if character_options:
                    character_names = [option.name for option in character_options]
                    character_ids = [option.id for option in character_options]
                    
                    # Get current selection
                    current_char_ids = criteria.get("character_types", [])
                    if not isinstance(current_char_ids, list):
                        current_char_ids = [current_char_ids]
                        
                    # Convert IDs to indices
                    current_indices = []
                    for char_id in current_char_ids:
                        if char_id in character_ids:
                            current_indices.append(character_ids.index(char_id))
                    
                    selected_characters = st.multiselect(
                        "Character Types", 
                        options=character_names,
                        default=[character_names[i] for i in current_indices] if current_indices else None
                    )
                    
                    # Update criteria with selected character types
                    if selected_characters:
                        selected_ids = [character_ids[character_names.index(name)] for name in selected_characters]
                        criteria["character_types"] = selected_ids
                    else:
                        if "character_types" in criteria:
                            del criteria["character_types"]
            
            with col2:
                # Source type selection
                source_options = self.optimizer.get_field_options("source_type")
                if source_options:
                    source_names = [option.name for option in source_options]
                    source_ids = [option.id for option in source_options]
                    
                    # Get current selection
                    current_source_id = criteria.get("source_type")
                    current_index = source_ids.index(current_source_id) if current_source_id in source_ids else 0
                    
                    selected_source = st.selectbox(
                        "Source Type", 
                        options=source_names,
                        index=current_index
                    )
                    
                    # Update criteria with selected source type
                    if selected_source:
                        selected_id = source_ids[source_names.index(selected_source)]
                        criteria["source_type"] = selected_id
                
                # Theme selection
                theme_options = self.optimizer.get_field_options("theme")
                if theme_options:
                    theme_names = [option.name for option in theme_options]
                    theme_ids = [option.id for option in theme_options]
                    
                    # Get current selection
                    current_theme_ids = criteria.get("theme", [])
                    if not isinstance(current_theme_ids, list):
                        current_theme_ids = [current_theme_ids]
                        
                    # Convert IDs to indices
                    current_indices = []
                    for theme_id in current_theme_ids:
                        if theme_id in theme_ids:
                            current_indices.append(theme_ids.index(theme_id))
                    
                    selected_themes = st.multiselect(
                        "Themes", 
                        options=theme_names,
                        default=[theme_names[i] for i in current_indices] if current_indices else None
                    )
                    
                    # Update criteria with selected themes
                    if selected_themes:
                        selected_ids = [theme_ids[theme_names.index(name)] for name in selected_themes]
                        criteria["theme"] = selected_ids
                    else:
                        if "theme" in criteria:
                            del criteria["theme"]
            
            # Submit button
            submitted = st.form_submit_button("Analyze Concept")
            
            if submitted:
                # Save criteria to session state
                st.session_state.optimizer_criteria = criteria
                
                # Analyze concept
                with st.spinner("Analyzing concept..."):
                    summary = self.optimizer.analyze_concept(criteria)
                    
                    if summary:
                        st.session_state.optimizer_summary = summary
                        st.experimental_rerun()
                    else:
                        st.error("Failed to analyze concept. Please try again.")
        
        # Display concept analysis if available
        if "optimizer_summary" in st.session_state:
            self._render_concept_analysis(st.session_state.optimizer_summary)
    
    def _render_concept_analysis(self, summary: OptimizationSummary):
        """Render the concept analysis section.
        
        Args:
            summary: Optimization summary
        """
        st.subheader("Concept Analysis")
        
        # Display overall success probability
        col1, col2, col3 = st.columns(3)
        
        with col1:
            render_metric_card(
                "Success Probability", 
                f"{summary.overall_success_probability:.0%}", 
                f"Confidence: {summary.confidence.capitalize()}"
            )
        
        # Display component scores
        with col2:
            audience_score = summary.component_scores.get("audience", None)
            if audience_score:
                render_metric_card(
                    "Audience Appeal", 
                    f"{audience_score.score:.0%}", 
                    f"Confidence: {audience_score.confidence.capitalize()}"
                )
            
        with col3:
            critic_score = summary.component_scores.get("critics", None)
            if critic_score:
                render_metric_card(
                    "Critical Reception", 
                    f"{critic_score.score:.0%}", 
                    f"Confidence: {critic_score.confidence.capitalize()}"
                )
        
        # Display success factors
        st.subheader("Success Factors")
        
        if summary.success_factors:
            # Create a dataframe for the factors
            factor_data = []
            for factor in summary.success_factors:
                factor_data.append({
                    "Type": factor.criteria_type.replace("_", " ").title(),
                    "Name": factor.criteria_name,
                    "Impact": factor.impact_score,
                    "Confidence": factor.confidence.capitalize()
                })
                
            factor_df = pd.DataFrame(factor_data)
            
            # Create a bar chart
            chart = alt.Chart(factor_df).mark_bar().encode(
                x=alt.X('Impact:Q', title='Impact on Success'),
                y=alt.Y('Name:N', title=None, sort='-x'),
                color=alt.Color('Impact:Q', scale=alt.Scale(
                    domain=[-0.5, 0, 0.5],
                    range=['#f77', '#ddd', '#7d7']
                )),
                tooltip=['Type', 'Name', 'Impact', 'Confidence']
            ).properties(
                height=30 * len(factor_data)
            )
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No significant success factors identified.")
    
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
                    if networks:
                        # Create a dataframe for the networks
                        network_data = []
                        for network in networks:
                            network_data.append({
                                "Network": network.network_name,
                                "Success Probability": network.success_probability,
                                "Compatibility": network.compatibility_score,
                                "Sample Size": network.sample_size,
                                "Confidence": network.confidence.capitalize()
                            })
                            
                        network_df = pd.DataFrame(network_data)
                        
                        # Display as a table
                        st.dataframe(
                            network_df,
                            column_config={
                                "Success Probability": st.column_config.ProgressColumn(
                                    "Success Probability",
                                    format="%.0f%%",
                                    min_value=0,
                                    max_value=1
                                ),
                                "Compatibility": st.column_config.ProgressColumn(
                                    "Compatibility",
                                    format="%.0f%%",
                                    min_value=0,
                                    max_value=1
                                )
                            },
                            hide_index=True
                        )
                    else:
                        st.info(f"No networks in the {tier_name} tier.")
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
            add_recs = [rec for rec in summary.recommendations if rec.recommendation_type == "add"]
            replace_recs = [rec for rec in summary.recommendations if rec.recommendation_type == "replace"]
            remove_recs = [rec for rec in summary.recommendations if rec.recommendation_type == "remove"]
            consider_recs = [rec for rec in summary.recommendations if rec.recommendation_type == "consider"]
            
            # Display add recommendations
            if add_recs:
                st.subheader("Consider Adding")
                for rec in add_recs[:3]:  # Show top 3
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.button(
                            f"Add {rec.suggested_name}",
                            key=f"add_{rec.criteria_type}_{rec.suggested_value}",
                            on_click=self._apply_recommendation,
                            args=(rec,)
                        )
                    with col2:
                        render_info_card(
                            f"{rec.criteria_type.replace('_', ' ').title()}: {rec.suggested_name}",
                            rec.explanation
                        )
            
            # Display replace recommendations
            if replace_recs:
                st.subheader("Consider Replacing")
                for rec in replace_recs[:3]:  # Show top 3
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.button(
                            f"Replace with {rec.suggested_name}",
                            key=f"replace_{rec.criteria_type}_{rec.suggested_value}",
                            on_click=self._apply_recommendation,
                            args=(rec,)
                        )
                    with col2:
                        render_info_card(
                            f"{rec.criteria_type.replace('_', ' ').title()}: {rec.current_value} → {rec.suggested_name}",
                            rec.explanation
                        )
            
            # Display remove recommendations
            if remove_recs:
                st.subheader("Consider Removing")
                for rec in remove_recs[:3]:  # Show top 3
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.button(
                            f"Remove {rec.suggested_name}",
                            key=f"remove_{rec.criteria_type}_{rec.current_value}",
                            on_click=self._apply_recommendation,
                            args=(rec,)
                        )
                    with col2:
                        render_warning(
                            f"{rec.criteria_type.replace('_', ' ').title()}: {rec.suggested_name}",
                            rec.explanation
                        )
            
            # Display consider recommendations
            if consider_recs:
                st.subheader("Additional Insights")
                for rec in consider_recs[:3]:  # Show top 3
                    render_info_card(
                        f"{rec.criteria_type.replace('_', ' ').title()}: {rec.suggested_name}",
                        rec.explanation
                    )
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
