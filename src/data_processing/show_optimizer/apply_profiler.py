"""
Apply profiling to Show Optimizer key methods

This module applies the profiler to key methods in the Show Optimizer
to identify performance bottlenecks and who is calling expensive operations.
"""

import streamlit as st
from functools import wraps
from .optimizer_profiler import profile

def apply_profiling():
    """Apply profiling to key methods in the Show Optimizer."""
    from .network_analyzer import NetworkAnalyzer
    from .recommendation_engine import RecommendationEngine
    from .criteria_scorer import CriteriaScorer
    from .score_calculators import NetworkScoreCalculator
    from .optimizer_matcher import Matcher
    
    # Profile NetworkAnalyzer methods
    NetworkAnalyzer.rank_networks_by_compatibility = profile()(NetworkAnalyzer.rank_networks_by_compatibility)
    NetworkAnalyzer.get_network_tiers = profile()(NetworkAnalyzer.get_network_tiers)
    
    # Profile NetworkScoreCalculator methods
    NetworkScoreCalculator.calculate_network_scores = profile()(NetworkScoreCalculator.calculate_network_scores)
    
    # Profile RecommendationEngine methods
    RecommendationEngine.calculate_overall_success_rate = profile()(RecommendationEngine.calculate_overall_success_rate)
    RecommendationEngine.identify_success_factors = profile()(RecommendationEngine.identify_success_factors)
    RecommendationEngine.generate_recommendations = profile()(RecommendationEngine.generate_recommendations)
    
    # Profile CriteriaScorer methods
    CriteriaScorer.calculate_component_scores = profile()(CriteriaScorer.calculate_component_scores)
    CriteriaScorer.calculate_criteria_impact = profile()(CriteriaScorer.calculate_criteria_impact)
    CriteriaScorer._calculate_success_rate = profile()(CriteriaScorer._calculate_success_rate)
    # _get_matching_shows method has been removed as part of architecture refactoring
    
    # Profile Matcher methods
    Matcher.find_matches_with_fallback = profile()(Matcher.find_matches_with_fallback)
    # find_network_matches method has been removed - functionality now handled by NetworkAnalyzer.rank_networks_by_compatibility
    
    st.write("Profiling has been applied to key Show Optimizer methods.")

def add_profiling_to_show_optimizer_page():
    """Add profiling display to the Show Optimizer page."""
    from .optimizer_profiler import display_profiling_results
    
    # Create an expander for profiling results
    with st.expander("Performance Profiling", expanded=False):
        display_profiling_results()
