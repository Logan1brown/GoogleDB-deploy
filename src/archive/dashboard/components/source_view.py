"""
Source Type Analysis Component.

This module provides visualization components for analyzing source type distribution
and patterns across networks using Streamlit's layout system.
"""

import logging
from typing import Dict

import streamlit as st
import plotly.express as px
import pandas as pd

from src.dashboard.utils.style_config import COLORS, CHART_DEFAULTS

logger = logging.getLogger(__name__)

def render_source_analysis(analysis_results: Dict) -> None:
    """Render the source analysis component.
    
    Args:
        analysis_results: Results from source_analyzer.analyze_source_patterns()
    """
    # Title and Description
    st.title("Source Analysis")
    st.markdown("Analysis of source type distribution and network specialization patterns.")
    
    # Source Type Stats in a row
    source_counts = pd.Series(analysis_results['source_counts']).sort_values(ascending=False)
    total_shows = analysis_results['total_shows']
    
    # Show top source types as metrics
    metrics = st.columns(4, gap="small")
    for i, (source, count) in enumerate(list(source_counts.items())[:4]):
        share = (count / total_shows) * 100
        with metrics[i]:
            st.metric(
                source,
                f"{count:,}",
                None,
                help=f"{share:.1f}% of all shows"
            )
    
    # Split layout into two columns
    col1, col2 = st.columns([0.7, 0.3])
    
    # Heatmap in the wider left column
    with col1:
        st.markdown("### Network Source Distribution")
        _render_network_heatmap(analysis_results)
    
    # Insights in the narrower right column
    with col2:
        st.markdown("### Key Insights")
        insights_container = st.container(height=600)  # Taller to match heatmap
        with insights_container:
            _render_source_insights(analysis_results)

def _render_source_insights(analysis_results: Dict) -> None:
    """Render source type insights and key findings.
    
    Args:
        analysis_results: Results from source_analyzer
    """
    # Get key metrics
    original = analysis_results['original_content']
    preferences = analysis_results['source_preferences']
    diversity = analysis_results['diversity_scores']
    
    # Source Diversity
    st.markdown("<h4 style='color: #1f77b4; margin-bottom: 0.3em;'>Source Diversity</h4>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.95em; font-weight: 600; margin-bottom: 0.3em;'>Most Diverse Networks</div>", unsafe_allow_html=True)
    for network, score in sorted(diversity.items(), key=lambda x: x[1], reverse=True)[:3]:
        st.markdown(f"• {network} ({score:.2f} diversity score)")
    
    # Network Specialization
    st.markdown("<h4 style='color: #1f77b4; margin: 0.8em 0 0.3em;'>Network Specialization</h4>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.95em; font-weight: 600; margin-bottom: 0.3em;'>Source Focus (>30% share)</div>", unsafe_allow_html=True)
    for source_type, networks in preferences.items():
        if networks:  # Only show if there are specialists
            top_net = networks[0]  # Most specialized network
            st.markdown(
                f"• {top_net['network']}: {top_net['percentage']:.0f}% {source_type}"
            )
    
    # Original Content Leaders
    st.markdown("<h4 style='color: #1f77b4; margin: 0.8em 0 0.3em;'>Original Content</h4>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.95em; font-weight: 600; margin-bottom: 0.3em;'>Networks with Strong Original Focus</div>", unsafe_allow_html=True)
    for source_type, networks in preferences.items():
        if source_type == 'Original' and networks:
            for net in networks[:3]:  # Show top 3 networks
                st.markdown(
                    f"• {net['network']}: {net['percentage']:.0f}% "
                    f"({net['count']} shows)"
                )

def _render_source_distribution(analysis_results: Dict) -> None:
    """Render source type distribution chart.
    
    Args:
        analysis_results: Results from source_analyzer
    """
    st.markdown("### Source Type Distribution")
    
    # Convert source counts to DataFrame
    source_counts = pd.Series(analysis_results['source_counts']).reset_index()
    source_counts.columns = ['Source Type', 'Count']
    
    # Calculate percentages
    total = source_counts['Count'].sum()
    source_counts['Percentage'] = (source_counts['Count'] / total * 100).round(1)
    
    # Sort by count descending
    source_counts = source_counts.sort_values('Count', ascending=False)
    
    # Create bar chart
    fig = px.bar(
        source_counts,
        x='Source Type',
        y='Count',
        color='Source Type',
        color_discrete_sequence=px.colors.qualitative.Set3,
        title=None
    )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(size=12),
        modebar_add=["downloadimage"],
        xaxis=dict(
            title=None,
            tickangle=0
        ),
        yaxis=dict(
            title="Number of Shows",
            tickformat=","
        ),
        height=300
    )
    
    # Add hover template
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Shows: %{y:,}<br>"
            "Share: %{customdata:.1f}%"
            "<extra></extra>"
        ),
        customdata=source_counts['Percentage']
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _render_network_heatmap(analysis_results: Dict) -> None:
    """Render network source type heatmap.
    
    Args:
        analysis_results: Results from source_analyzer
    """
    network_patterns = analysis_results['network_source_pct']
    
    # Convert percentages to DataFrame
    df = pd.DataFrame(network_patterns)
    
    # Get all unique source types
    all_sources = df.columns
    
    # Sort source types by overall frequency
    source_totals = df.sum()
    sorted_sources = source_totals.sort_values(ascending=False).index
    
    # Sort networks by their value in the highest source type
    network_primary = df[sorted_sources[0]]
    sorted_networks = network_primary.sort_values(ascending=False).index
    
    # Reindex DataFrame with sorted indices
    df = df.loc[sorted_networks, sorted_sources]
    
    # Create hover text
    hover_text = []
    for _, row in df.iterrows():
        hover = [f"<b>{col}</b>: {val:.0f}%" for col, val in row.items()]
        hover_text.append(hover)
    
    # Calculate dimensions based on number of cells
    n_rows = len(df)
    n_cols = len(df.columns)
    cell_size = 40  # Base size for each cell
    plot_width = min(1200, n_cols * cell_size + 300)
    plot_height = min(800, n_rows * cell_size + 150)
    
    # Create heatmap
    fig = px.imshow(
        df,
        color_continuous_scale=CHART_DEFAULTS['colorscales']['primary'],
        labels=dict(color='% of Network Shows'),
        aspect='auto'
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>%{z:.0f}%<extra></extra>"
    )
    
    # Update layout
    fig.update_layout(
        width=plot_width,
        height=plot_height,
        margin=CHART_DEFAULTS['margin']['plot'],
        font=dict(size=12),
        xaxis=dict(
            title=None,
            tickangle=45,
            fixedrange=True,
            side='top'
        ),
        yaxis=dict(
            title=None,
            fixedrange=True,
            autorange='reversed'
        ),
        coloraxis_colorbar=dict(
            title=dict(
                text='% of Network Shows',
                side='right'
            ),
            thicknessmode='pixels',
            thickness=20,
            lenmode='fraction',
            len=0.75,
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
