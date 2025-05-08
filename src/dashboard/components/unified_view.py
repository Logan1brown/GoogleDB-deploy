"""
Unified Dashboard View Component.

A Streamlit component that provides a unified view with:
- Acquisition View (Networks, Creators, Pairings, Insights)
- Packaging View
- Development View

=== COLUMN NAME STANDARDS ===
1. Base Tables:
   - tmdb_id: Show ID from TMDB
   - network_id: Foreign key to network_list
   - studios: Array of foreign keys to studio_list
   - status_id: Foreign key to status_types

2. Views/APIs:
   - network_name: Network name (from network_id)
   - studio_names: Studio names (from studios array)
   - status_name: Status name (from status_id)
   - title: Show name (no transformation)
"""

import streamlit as st
import pandas as pd
import logging
from typing import Optional
from src.data_processing.unified.unified_analyzer import UnifiedAnalyzer
from src.data_processing.success_analysis.success_analyzer import SuccessAnalyzer
from src.dashboard.utils.style_config import COLORS

logger = logging.getLogger(__name__)



def render_acquisition_view(unified_analyzer: UnifiedAnalyzer, source_type: str, genre: str):
    """Render the acquisition view with its tabs.
    
    Args:
        unified_analyzer: UnifiedAnalyzer instance
        source_type: Selected source type filter
        genre: Selected genre filter
    """
    # Get filter options and validate filters
    filter_options = unified_analyzer.get_filter_options()
    valid_source_type = source_type if source_type != "All" and source_type in filter_options['source_types'] else None
    valid_genre = genre if genre != "All" and genre in filter_options['genres'] else None
    
    filtered_df = unified_analyzer.get_filtered_data(valid_source_type, valid_genre)
    
    tabs = st.tabs(["Networks", "Market Insights"])
    
    with tabs[0]:  # Networks
        st.markdown("### Network Performance")
        network_metrics = unified_analyzer.get_network_metrics(valid_source_type, valid_genre)
        
        for network, metrics in network_metrics.items():
            with st.expander(f"{network} ({metrics['title_count']} titles)"):
                # Only count shows with reliable status in the average
                shows_with_data = [t for t in metrics['titles'] if t['has_reliable_status']]
                if not shows_with_data:
                    score_display = "N/A"
                else:
                    score_display = f"{metrics['success_score']:.0f} pts"
                st.markdown(f"**Network Average Success: {score_display}**")
                st.markdown("**Titles:**")
                for title_info in metrics['titles']:
                    score_display = "N/A" if not title_info['has_reliable_status'] else f"{title_info['success_score']:.0f} pts"
                    st.markdown(f"- {title_info['title']} ({score_display})")
    
    with tabs[1]:  # Market Insights
        # Get market metrics
        filter_options = unified_analyzer.get_filter_options()
        snapshot = unified_analyzer.get_market_snapshot(
            source_type if source_type != "All" and source_type in filter_options['source_types'] else None,
            genre if genre != "All" and genre in filter_options['genres'] else None
        )
        patterns = unified_analyzer.get_success_patterns(
            source_type if source_type != "All" and source_type in filter_options['source_types'] else None,
            genre if genre != "All" and genre in filter_options['genres'] else None
        )
        
        # Market Overview
        st.markdown("### Market Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Titles", snapshot['total_titles'])
        with col2:
            st.metric("Average Success Score", f"{snapshot['avg_success']:.0f} pts")
        
        # Source Type Distribution
        st.markdown("### Source Type Distribution")
        source_data = pd.DataFrame({
            'Source Type': list(snapshot['source_distribution'].keys()),
            'Count': list(snapshot['source_distribution'].values())
        })
        st.bar_chart(source_data.set_index('Source Type'))
        
        # Success Patterns
        st.markdown("### Top Performing Combinations")
        for combo in patterns['top_combinations']:
            with st.expander(f"{combo['genre_name']} + {combo['source_type_name']} ({combo['title_count']} titles | {combo['success_score']:.0f} pts)"): 
                for title in combo['titles']:
                    st.markdown(f"- {title}")
        
        # Network Preferences
        st.markdown("### Network Investment")
        # Sort networks by title count
        sorted_networks = sorted(snapshot['network_preferences'].items(),
                               key=lambda x: x[1]['total_titles'],
                               reverse=True)
        for network, stats in sorted_networks:
            with st.expander(f"{network} ({stats['total_titles']} titles)"):
                st.metric("Success Score", f"{stats['avg_success']:.0f} pts")

def render_packaging_view(unified_analyzer: UnifiedAnalyzer, source_type: str, genre: str):
    """Render the packaging view with creators, pairings and suggestions.
    
    Args:
        unified_analyzer: UnifiedAnalyzer instance
        source_type: Selected source type filter
        genre: Selected genre filter
    """
    tabs = st.tabs(["Creators", "Package Suggestions"])
    
    # Get filter options and validate filters
    filter_options = unified_analyzer.get_filter_options()
    valid_source_type = source_type if source_type != "All" and source_type in filter_options['source_types'] else None
    valid_genre = genre if genre != "All" and genre in filter_options['genres'] else None
    
    with tabs[0]:  # Creators
        st.markdown("### Top Creators")
        creator_metrics = unified_analyzer.get_creator_metrics(
            valid_source_type,
            valid_genre
        )
        
        for creator, metrics in creator_metrics['creators'].items():
            with st.expander(f"{creator} ({metrics['title_count']} titles | {metrics['success_score']:.0f} pts)"): 
                st.markdown("**Titles:**")
                for title in metrics['titles']:
                    st.markdown(f"- {title}")
                
                st.markdown("\n**Network Success:**")
                # Sort networks by success score
                sorted_networks = sorted(metrics['network_success'].items(),
                                       key=lambda x: (x[1]['success_score'], x[1]['title_count']),
                                       reverse=True)
                for network, stats in sorted_networks:
                    st.markdown(f"- **{network}** ({stats['title_count']} titles, {stats['success_score']:.0f} pts)")
                    for title in stats['titles']:
                        st.markdown(f"  - {title}")
    
    with tabs[1]:  # Package Suggestions
        st.markdown("### Package Suggestions")
        st.markdown(f"Showing creators with proven success across multiple networks")
        
        suggestions = unified_analyzer.get_package_suggestions(
            valid_source_type,
            valid_genre
        )
        
        if not suggestions:
            st.info("No suggestions found with the current filters. Try broadening your search.")
        
        for suggestion in suggestions:
            with st.expander(f"{suggestion['creator']} ({suggestion['network_count']} networks, {suggestion['total_titles']} titles)"):
                # Show network breadth
                st.markdown("**Network Breakdown:**")
                for network in suggestion['networks']:
                    st.markdown(f"- **{network['name']}** ({network['title_count']} titles)")
                    for title in network['titles']:
                        # Show different statuses:
                        # - Development score (85.0) -> In Development
                        # - 0 points but ended/canceled -> Limited Run (didn't meet scoring thresholds)
                        # - Other scores -> Show the score
                        if title['success_score'] == 85.0:
                            score_display = " (In Development)"
                        elif title['success_score'] == 0:
                            score_display = " (Limited Run)"
                        else:
                            score_display = f" ({title['success_score']:.0f} pts)"
                        st.markdown(f"  - {title['title']}{score_display}")

def render_development_view(unified_analyzer: UnifiedAnalyzer, source_type: str, genre: str):
    """Render the development view focusing on format decisions.
    
    Args:
        unified_analyzer: UnifiedAnalyzer instance
        source_type: Selected source type filter
        genre: Selected genre filter
    """
    st.markdown("### Format Strategy")
    
    # Get filter options and validate filters
    filter_options = unified_analyzer.get_filter_options()
    valid_source_type = source_type if source_type != "All" and source_type in filter_options['source_types'] else None
    valid_genre = genre if genre != "All" and genre in filter_options['genres'] else None
    
    # Get format insights
    insights = unified_analyzer.get_format_insights(valid_source_type, valid_genre)
    
    if insights['total_titles_analyzed'] == 0:
        st.warning("Not enough data for format analysis with current filters")
        return
        
    # Show key metrics in columns
    col1, col2, col3 = st.columns(3)
    
    # Show metrics for most common format and averages
    with col1:
        if insights['episode_insights'].get('most_common'):
            st.metric(
                "Most Common Format",
                f"{insights['episode_insights']['most_common']:.0f} Episodes",
                ""
            )
        else:
            st.metric("Most Common Format", "No data", "")
    
    with col2:
        if insights['episode_insights'].get('avg_episodes'):
            st.metric(
                "Average Episodes",
                f"{insights['episode_insights']['avg_episodes']:.1f}",
                ""
            )
        else:
            st.metric("Average Episodes", "No data", "")
    
    with col3:
        st.metric(
            "Titles Analyzed",
            insights['total_titles_analyzed'],
            ""
        )
    
    # Show episode count distribution
    st.subheader("Episodes per Season Distribution")
    if insights.get('episode_insights', {}).get('distribution', {}).get('episodes', []):
        episodes_df = pd.DataFrame({
            'Episodes': insights['episode_insights']['distribution']['episodes'],
            'Number of Titles': insights['episode_insights']['distribution']['title_counts']
        })
        # Create a custom chart with whole number y-axis
        chart = st.bar_chart(
            episodes_df.set_index('Episodes'),
            height=400,  # Taller chart for better visibility
            use_container_width=True
        )
        # Force y-axis to use whole numbers
        max_titles = episodes_df['Number of Titles'].max()
        chart.y_range = (0, max_titles + 1)
        if insights['episode_insights'].get('most_common'):
            st.caption(f"Most common: {insights['episode_insights']['most_common']:.0f} episodes")
        
        # Add episode count selector
        episode_count = st.number_input("View titles with episode count:", min_value=1, max_value=100, value=13)
        titles = unified_analyzer.get_titles_by_episode_count(episode_count, source_type, genre)
        if not titles.empty:
            st.dataframe(
                titles.reset_index(drop=True),  # Reset and drop the index
                use_container_width=True,
                column_config={
                    "title": "Title",
                    "network_name": "Network",
                    "genre_name": "Genre",
                    "source_type_name": "Source Type",
                    "order_type": "Order Type",
                    "episode_count": "Episodes"
                }
            )
        else:
            st.write(f"No titles found with {episode_count} episodes per season")
    else:
        st.write("No episode data available for the selected filters.")
    
    # Add some spacing
    st.write("")
    
    # Show series type breakdown
    st.subheader("Series Type Breakdown")
    series_insights = insights.get('series_insights', {})
    if series_insights and series_insights.get('counts'):
        # Get available types and their display names
        type_display = {
            'Limited': 'Limited Series',
            'Ongoing': 'Ongoing Series',
            'Miniseries': 'Miniseries',
            'Anthology': 'Anthology Series'
        }
        
        # Create data table with only types that have data
        available_types = list(series_insights['counts'].keys())
        series_data = pd.DataFrame({
            'Type': [type_display.get(t, t) for t in available_types],
            'Titles': [series_insights['counts'][t] for t in available_types],
            'Percentage': [series_insights['percentages'][t] for t in available_types]
        })
        
        # Display as a table with both counts and percentages
        st.dataframe(
            series_data.assign(
                Percentage=lambda x: x['Percentage'].map('{:.1f}%'.format)
            ).set_index('Type'),
            hide_index=False,
            use_container_width=True  # Use full width
        )
    else:
        st.write("No series type data available for the selected filters.")
    
    # Add some spacing
    st.write("")
    
    # Show network preferences in a sortable table
    st.subheader("Network Preferences")
    network_insights = insights.get('network_insights', {})
    if network_insights:
        # Create data table
        network_data = pd.DataFrame([
            {
                'Network': network_name,
                'Avg Episodes': f"{round(data['avg_episodes'])}",
                'Preferred Format': data['most_successful_format']['preferred_type']
            }
            for network_name, data in network_insights.items()
        ])
        
        # Display as a table
        st.dataframe(
            network_data,
            column_config={
                'Network': st.column_config.TextColumn(
                    'Network',
                    help='Network name'
                ),
                'Avg Episodes': st.column_config.TextColumn(
                    'Avg Episodes',
                    help='Average episodes per title'
                ),
                'Preferred Format': st.column_config.TextColumn(
                    'Preferred Format',
                    help='Most successful series type for this network'
                )
            }
        )

def render_unified_dashboard(success_analyzer: Optional['SuccessAnalyzer'] = None):
    """Main entry point for the unified dashboard view.
    
    Args:
        success_analyzer: Optional SuccessAnalyzer instance for success metrics
    """
    try:
        # Initialize analyzer with normalized data
        unified_analyzer = UnifiedAnalyzer(success_analyzer)
        
        # Add overall description
        st.write("Analyze the TV market across three key stages: network acquisition patterns, creative packaging trends, and development format decisions.")
        
        # Create analysis type selector at the top
        analysis_type = st.radio(
            "",
            ["Acquisition", "Packaging", "Development"],
            horizontal=True,
            key="unified_analysis_type"
        )
        
        if analysis_type == "Acquisition":
            # Create two columns - narrow one for filters, wide one for content
            filter_col, content_col = st.columns([1, 3])
            
            with filter_col:  # Input Panel
                # Get filter options from analyzer
                filter_options = unified_analyzer.get_filter_options()
                
                source_type = st.selectbox(
                    "Source Type",
                    options=["All"] + filter_options['source_types'],
                    key="unified_source_type"
                )
                
                genre = st.selectbox(
                    "Genre",
                    options=["All"] + filter_options['genres'],
                    key="unified_genre"
                )
            
            with content_col:  # Results Panel
                st.write("Analyze network performance and market trends, with detailed success metrics by network and insights into winning genre-source combinations.")
                st.write("---")
                render_acquisition_view(unified_analyzer, source_type, genre)
                
        elif analysis_type == "Packaging":
            # Create two columns - narrow one for filters, wide one for content
            filter_col, content_col = st.columns([1, 3])
            
            with filter_col:  # Input Panel
                # Get filter options from analyzer
                filter_options = unified_analyzer.get_filter_options()
                
                source_type = st.selectbox(
                    "Source Type",
                    options=["All"] + filter_options['source_types'],
                    key="unified_source_type_pkg"
                )
                
                genre = st.selectbox(
                    "Genre",
                    options=["All"] + filter_options['genres'],
                    key="unified_genre_pkg"
                )
            
            with content_col:  # Results Panel
                st.write("Explore creator track records and successful creative pairings, filtered by genre and source type to inform packaging decisions.")
                st.write("---")
                render_packaging_view(unified_analyzer, source_type, genre)
                
        else:  # Development
            # Create two columns - narrow one for filters, wide one for content
            filter_col, content_col = st.columns([1, 3])
            
            with filter_col:  # Input Panel
                # Get filter options from analyzer
                filter_options = unified_analyzer.get_filter_options()
                
                source_type = st.selectbox(
                    "Source Type",
                    options=["All"] + filter_options['source_types'],
                    key="unified_source_type_dev"
                )
                
                genre = st.selectbox(
                    "Genre",
                    options=["All"] + filter_options['genres'],
                    key="unified_genre_dev"
                )
            
            with content_col:  # Results Panel
                st.write("Evaluate format decisions and success patterns across episodes, seasons, and order types to guide development strategy.")
                st.write("---")
                render_development_view(unified_analyzer, source_type, genre)
            
    except Exception as e:
        import traceback
        st.error("Error in unified dashboard:")
        st.error(str(e))
        st.error("Full traceback:")
        st.code(traceback.format_exc())
