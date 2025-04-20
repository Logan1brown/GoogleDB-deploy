"""Genre Analysis Component.

This module provides visualization components for analyzing genre distribution
and patterns across networks using Streamlit's layout system.

Usage:
    from src.dashboard.components.genre_view import render_genre_analysis
    
    render_genre_analysis(analysis_results)
"""

import logging
from typing import Dict, List

import streamlit as st
import pandas as pd
import plotly.express as px

from src.dashboard.utils.style_config import COLORS, CHART_DEFAULTS, DIMENSIONS

logger = logging.getLogger(__name__)

def render_genre_analysis(analysis_results: Dict) -> None:
    """Render the genre analysis component.
    
    Args:
        analysis_results: Results from genre_analyzer.analyze_genre_patterns()
    """
    # Title and Description
    st.title("Genre Analysis")
    st.markdown("Analysis of genre distribution and network specialization patterns.")
    
    # Genre Distribution Stats in a row
    genre_stats = analysis_results['genre_stats']
    genre_dist = pd.Series(genre_stats['genre_distribution']).sort_values(ascending=False)
    
    # Top 5 genres as metrics
    metrics = st.columns(5, gap="small")
    for i, (genre, count) in enumerate(list(genre_dist.items())[:5]):
        share = (count / genre_stats['total_shows']) * 100
        with metrics[i]:
            st.metric(
                genre,
                f"{count}",
                None,  # Remove trend indicator entirely
                help=f"{share:.1f}% of all shows"  # Show percentage as tooltip
            )
    
    # Split layout into two columns
    col1, col2 = st.columns([0.7, 0.3])
    
    # Heatmap in the wider left column
    with col1:
        st.markdown("### Network Genre Distribution")
        _render_network_heatmap(analysis_results)
    
    # Insights in the narrower right column
    with col2:
        st.markdown("### Key Insights")
        insights_container = st.container(height=600)  # Taller to match heatmap
        with insights_container:
            _render_network_insights(analysis_results)

def _render_network_heatmap(analysis_results: Dict) -> None:
    """Render network specialization heatmap.
    
    Args:
        analysis_results: Results from genre_analyzer
    """
    network_patterns = analysis_results['network_patterns']
    diversity_metrics = analysis_results['diversity_metrics']
    
    # Get all unique genres
    all_genres = set()
    for data in network_patterns.values():
        all_genres.update(data['genre_shares'].keys())
    all_genres = sorted(list(all_genres))
    
    # Prepare data for heatmap
    z_data = []
    y_labels = []
    hover_text = []
    
    # Sort genres by overall frequency across all networks
    genre_totals = {}
    for genre in all_genres:
        total = sum(network_patterns[net]['genre_shares'].get(genre, 0) for net in network_patterns)
        genre_totals[genre] = total
    
    sorted_genres = sorted(all_genres, key=lambda x: genre_totals[x], reverse=True)
    
    # Sort networks by their primary genre focus
    def get_primary_genre(network):
        shares = network_patterns[network]['genre_shares']
        return max(shares.items(), key=lambda x: x[1])[1] if shares else 0
    
    sorted_networks = sorted(
        network_patterns.keys(),
        key=get_primary_genre,
        reverse=True
    )
    
    for network in sorted_networks:
        data = network_patterns[network]
        shares = data['genre_shares']
        
        # Create row of genre shares using sorted genres
        row = [shares.get(genre, 0) for genre in sorted_genres]
        hover = [f"<b>{genre}</b>: {shares.get(genre, 0):.0f}%" for genre in sorted_genres]
        
        z_data.append(row)
        y_labels.append(network)
        hover_text.append(hover)
    
    # Create DataFrame for heatmap
    df = pd.DataFrame(z_data, columns=sorted_genres, index=y_labels)
    
    # Calculate dimensions based on number of cells
    n_rows = len(network_patterns)
    n_cols = len(sorted_genres)
    cell_size = 40  # Base size for each cell
    plot_width = min(1200, n_cols * cell_size + 300)
    plot_height = min(800, n_rows * cell_size + 150)
    
    # Create heatmap using plotly express
    fig = px.imshow(
        df.round(0),  # Round values for display
        color_continuous_scale=CHART_DEFAULTS['colorscales']['primary'],
        labels=dict(color='% of Network Shows'),
        aspect='auto'  # Let Plotly handle the aspect ratio
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>%{z:.0f}%<extra></extra>"
    )
    
    # Update layout with style config
    fig.update_layout(
        width=plot_width,
        height=plot_height,
        margin=CHART_DEFAULTS['margin']['plot'],
        font=dict(size=12),  # Base font size
        xaxis=dict(
            title=None,
            tickangle=45,
            fixedrange=True,  # Prevent zoom
            side='top'  # Move labels to top
        ),
        yaxis=dict(
            title=None,
            fixedrange=True,  # Prevent zoom
            autorange='reversed'  # Show items from top to bottom
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text='% of Network Shows',
                side='right'
            ),
            thicknessmode='pixels',
            thickness=20,  # Standard width
            lenmode='fraction',
            len=0.75,  # Match plot height more closely
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02  # Slightly offset from plot
        )
    )
    
    # Let Streamlit handle the container width
    st.plotly_chart(fig, use_container_width=True)

def _render_network_insights(analysis_results: Dict) -> None:
    """Render network specialization insights.
    
    Args:
        analysis_results: Results from genre_analyzer
    """
    network_patterns = analysis_results['network_patterns']
    diversity_metrics = analysis_results['diversity_metrics']
    
    # Network Diversity
    st.markdown("<h4 style='color: #1f77b4; margin-bottom: 0.3em;'>Network Diversity</h4>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.95em; font-weight: 600; margin-bottom: 0.3em;'>Most Diverse Networks</div>", unsafe_allow_html=True)
    for network, count in diversity_metrics['most_diverse'].items():
        st.markdown(f"• {network} ({count} genres with >10% share)")
    
    # Network Specialization
    st.markdown("<h4 style='color: #1f77b4; margin: 0.8em 0 0.3em;'>Network Specialization</h4>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.95em; font-weight: 600; margin-bottom: 0.3em;'>Genre Focus (>40% share)</div>", unsafe_allow_html=True)
    for network, data in sorted(network_patterns.items(), key=lambda x: x[1]['primary'][1], reverse=True):
        primary_genre, primary_share = data['primary']
        if primary_share >= 40:
            st.markdown(f"• {network}: {primary_share:.1f}% {primary_genre}")
    
    # Unique Patterns
    if diversity_metrics['unique_patterns']:
        st.markdown("<h4 style='color: #1f77b4; margin: 0.8em 0 0.3em;'>Unique Genre Patterns</h4>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.95em; font-weight: 600; margin-bottom: 0.3em;'>Networks with Distinctive Focus (15% above market)</div>", unsafe_allow_html=True)
        for pattern in diversity_metrics['unique_patterns'][:5]:  # Top 5 most distinctive
            st.markdown(
                f"• {pattern['network']}: {pattern['share']:.1f}% {pattern['genre']} "
                f"(+{pattern['difference']:.1f}% vs. {pattern['market_share']:.1f}% market avg)"
            )
