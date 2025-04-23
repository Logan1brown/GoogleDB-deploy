"""Market Snapshot Component.

A Streamlit component that provides a market snapshot view with:
- Title section
- Executive summary
- KPI metrics (4 indicators)
- Category selectors (Shows, Creators, Genres, Networks)
- Market distribution visualization
- Performance metrics (4 indicators)

Uses secure Supabase views for data access.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import logging
import time
from src.data_processing.market_analysis.market_analyzer import MarketAnalyzer
from src.dashboard.utils.style_config import COLORS

logger = logging.getLogger(__name__)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_insights(_market_analyzer):
    """Get market insights with caching.
    
    Args:
        market_analyzer: MarketAnalyzer instance with processed data
        
    Returns:
        Tuple of (filtered_df, network_df, insights dict)
    """
    # Initialize filtered DataFrame
    filtered_df = _market_analyzer.titles_df.copy()
    
    # Get network data directly from network_df
    network_df = _market_analyzer.network_df.copy()
    
    
    total_creatives = _market_analyzer.get_unique_creatives() if hasattr(_market_analyzer, 'get_unique_creatives') else 0
    if hasattr(_market_analyzer, 'titles_df') and 'title' in _market_analyzer.titles_df.columns:
        total_titles = _market_analyzer.titles_df['title'].nunique()
    else:
        total_titles = 0
    if network_df is not None:
        total_networks = len(network_df)
    else:
        total_networks = 0
        
    return filtered_df, network_df, {
        'total_titles': total_titles,
        'total_networks': total_networks,
        'total_creatives': total_creatives
    }

def render_market_snapshot(market_analyzer):
    """Render the market snapshot component.
    
    Args:
        market_analyzer: MarketAnalyzer instance with processed data
    """
    try:
        with st.spinner('Loading market insights...'):
            filtered_df, network_df, initial_insights = get_market_insights(market_analyzer)
        
        # Use the cached insights
    except Exception as e:
        logger.error("Error generating market insights:")
        logger.error(f"Error generating market insights: {str(e)}")
        st.error(f"Error generating market insights: {str(e)}")
        return
    
    # Add custom CSS for selectbox
    st.markdown("""
    <style>
    div[data-baseweb="select"] > div {
        background-color: white;
        border-radius: 4px;
        border-color: rgb(49, 51, 63);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Get market insights with caching
    try:
        with st.spinner('Loading market analysis...'):
            insights = market_analyzer.generate_market_insights(filtered_df)
            
        if insights is None:
            st.error("generate_market_insights returned None!")

    except Exception as e:
        logger.error(f"Error generating market insights: {str(e)}")
        st.error(f"Error generating market insights: {str(e)}")
        return
    # Display key dataset metrics and filters
    try:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Process initial insights
            st.metric("Total Titles", f"{initial_insights['total_titles']:,}")
            titles = sorted(market_analyzer.titles_df['title'].unique())
            selected_titles = st.multiselect(
                "Filter Titles", 
                titles,
                max_selections=5,
                help="Select up to 5 titles to filter the data",
                key="market_filter_titles"
            )
    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        st.error(f"Error displaying metrics: {str(e)}")
        return
    with col2:
        # Get unique creatives using the filtered method
        unique_creatives = market_analyzer.get_unique_creatives()
        
        # Get the list of creatives for filtering - use the same filtered team_df
        if not market_analyzer.team_df.empty and 'name' in market_analyzer.team_df.columns:
            creatives = sorted(market_analyzer.team_df['name'].dropna().unique())
        else:
            creatives = []
            
        # Display metric for unique creatives
        st.metric("Unique Creatives", str(unique_creatives))
        selected_creatives = st.multiselect(
            "Filter Creatives", 
            creatives,
            max_selections=5,
            help="Select up to 5 creatives to filter the data",
            key="market_filter_creatives"
        )
    with col3:

        st.metric("Networks", f"{initial_insights['total_networks']:,}")
        networks = sorted(market_analyzer.titles_df['network_name'].unique())
        selected_networks = st.multiselect(
            "Filter Networks", 
            networks,
            help="Select networks to filter the data",
            key="market_filter_networks"
        )
    with col4:
        # Get success metrics directly
        success_metrics = market_analyzer.success_analyzer.analyze_market(filtered_df)
        if success_metrics and 'titles' in success_metrics:
            total_score = sum(data['score'] for data in success_metrics['titles'].values())
            num_titles = len(success_metrics['titles'])
            avg_success = total_score / num_titles if num_titles > 0 else 0
        else:
            avg_success = 0
        st.metric(
            "Success Score", 
            f"{avg_success:.0f}/100",
            help="Average title success score (0-100) based on:\n" +
                 "- Number of Seasons (40pts for S2, +20pts each for S3/S4/S5+)\n" +
                 "- Show Status (bonus for planned ending, penalty for cancellation)\n" +
                 "- Episode Volume (penalty for <11 eps)\n\n" +
                 "Note: Limited series typically score low since the metric focuses on multi-season success."
        )
        success_filter = st.selectbox(
            "Success Tier", 
            ["All", "High (>80)", "Medium (50-80)", "Low (<50)"],
            help="Filter by success score range",
            key="market_filter_success",
            label_visibility="visible"
        )
        
    # Track which filter is active and disable others
    active_filter = None
    if selected_titles:
        active_filter = "titles"
    elif selected_networks:
        active_filter = "networks"
    elif selected_creatives:
        active_filter = "creatives"
        
    # Only show warning for multi-select filters that need to be cleared
    if active_filter:
        st.info(f"⚠️ {active_filter.title()} filter is active. Clear it to use other filters.")

    # Network Distribution Section
    st.markdown('<p class="section-header">Network Distribution</p>', unsafe_allow_html=True)
    
    # Filter data based on success level
    # Include all needed columns including studio_names for vertical integration
    needed_cols = ['title', 'network_name', 'tmdb_id', 'tmdb_seasons', 'tmdb_total_episodes', 'tmdb_status', 'tmdb_avg_eps', 'studio_names', 'status_name']
    filtered_df = market_analyzer.titles_df[needed_cols].copy()
    
    # Get success metrics for all shows
    success_metrics = market_analyzer.success_analyzer.analyze_market(filtered_df)
    if not success_metrics or 'titles' not in success_metrics:
        return
        
    # Filter data based on success level
    if success_filter != "All":
        # Filter based on success tier
        if success_filter == "High (>80)":
            filtered_df = filtered_df[filtered_df['tmdb_id'].astype(str).isin(
                [id for id, data in success_metrics['titles'].items() if data['score'] > 80]
            )]
        elif success_filter == "Medium (50-80)":
            filtered_df = filtered_df[filtered_df['tmdb_id'].astype(str).isin(
                [id for id, data in success_metrics['titles'].items() if 50 <= data['score'] <= 80]
            )]
        elif success_filter == "Low (<50)":
            filtered_df = filtered_df[filtered_df['tmdb_id'].astype(str).isin(
                [id for id, data in success_metrics['titles'].items() if data['score'] < 50]
            )]
    
    # First apply creative filters if selected
    if selected_creatives:
        # Get titles where selected creatives work
        creative_titles = market_analyzer.team_df[
            market_analyzer.team_df['name'].isin(selected_creatives)
        ]['title'].unique()
        
        # Filter to only titles with selected creatives
        filtered_df = filtered_df[filtered_df['title'].isin(creative_titles)]
        
        if len(filtered_df) == 0:
            st.info("No titles found for selected creatives.")
            return
    
    # Then get networks with more than 1 show from the current set
    # Save unfiltered DataFrame for metrics
    metrics_df = filtered_df.copy()
    
    # Do NOT filter DataFrame to only include multi-show networks for the chart or filters
    # The multi-show filter will only be used in the top network metric in insights
    # Then apply other filters
    if selected_titles:
        filtered_df = filtered_df[filtered_df['title'].isin(selected_titles)]
    
    if selected_networks:
        filtered_df = filtered_df[filtered_df['network_name'].isin(selected_networks)]
    
    # Calculate insights for metrics using unfiltered data
    insights = market_analyzer.generate_market_insights(metrics_df)
    
    # Already have success metrics from earlier
    
    # Get success scores by network first
    network_scores = {}
    for title_id, title_data in success_metrics['titles'].items():
        # Convert title_id to int since tmdb_id is numeric
        try:
            title_id = int(float(title_id))
        except (ValueError, TypeError):
            logger.warning(f"Invalid tmdb_id: {title_id}")
            continue
        title = filtered_df[filtered_df['tmdb_id'] == title_id].iloc[0] if len(filtered_df[filtered_df['tmdb_id'] == title_id]) > 0 else None
        if title is not None:
            network = title['network_name']
            if network not in network_scores:
                network_scores[network] = []
            network_scores[network].append(title_data['score'])
    
    # If filtering by success tier, only include networks that have scores
    if success_filter != "All":
        networks_with_scores = set(network_scores.keys())
        # Process networks with scores
        filtered_df = filtered_df[filtered_df['network_name'].isin(networks_with_scores)]
    
    # Get network distribution
    titles_by_group = filtered_df.groupby('network_name').size().reset_index()
    titles_by_group.columns = ['network_name', 'count']
    x_title = "Network"
    group_col = 'network_name'
    
    # Sort by count
    titles_by_group = titles_by_group.sort_values('count', ascending=False)
    
    # Calculate average scores and create hover text
    avg_scores = []
    hover_text = []
    for _, row in titles_by_group.iterrows():
        network = row['network_name']
        count = row['count']
        text = f'{network}<br>Titles: {count}'
        
        if network in network_scores:
            avg = sum(network_scores[network]) / len(network_scores[network])
            avg_scores.append(avg)
            text += f'<br>Avg Success Score: {avg:.1f}'
        else:
            avg_scores.append(0)  # No score data
        hover_text.append(text)
    
    # Create color array using Viridis colorscale
    colors = []
    for score in avg_scores:
        if score == 0:
            # Use grey color for networks without score data
            colors.append(COLORS['success']['none'])
        else:
            # Use predefined success colors based on score ranges
            if score > 80:
                colors.append(COLORS['success']['high'])
            elif score >= 50:
                colors.append(COLORS['success']['medium'])
            else:
                colors.append(COLORS['success']['low'])
    
    # Create chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=titles_by_group[group_col],
        y=titles_by_group['count'],
        name=f"Titles per {x_title}",
        marker_color=colors,
        hovertext=hover_text,
        hoverinfo='text'
    ))
    
    # Update layout
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title="Number of Titles",
        font_family="Source Sans Pro",
        showlegend=False,
        margin=dict(t=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    

    
    # Key Metrics Section
    st.markdown('<p class="section-header">Key Metrics</p>', unsafe_allow_html=True)
    
    # Add metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Top Network",
            f"{insights['top_success_network']}",
            help=f"Network with highest combined success score ({insights['top_success_score']:.0f}/100) and show volume. Networks must have at least 3 titles to qualify."
        )
    with col2:
        st.metric(
            "Network Concentration", 
            f"{insights['network_concentration']:.1f}%",
            help=(
                "Herfindahl-Hirschman Index (HHI): A standard measure of market concentration. "
                "0% = highly competitive, 100% = monopoly. "
                "Typical interpretation: <15% = competitive, 15–25% = moderately concentrated, >25% = highly concentrated. "
                "A higher value means a few networks dominate; a lower value means the market is more competitive."
            )
        )
    with col3:
        st.metric(
            "Vertical Integration", 
            f"{insights['vertical_integration']:.0f}%",
            help="Percentage of titles from vertically integrated studios"
        )
        

