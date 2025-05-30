"""Show Detail page for the TV Series Database Dashboard."""

import streamlit as st
import pandas as pd
import sys
import os

from src.dashboard.components.match_breakdown import render_match_breakdown

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.shared.auth import auth_required
from src.dashboard.utils.timing import time_page
from src.dashboard.utils.style_config import COLORS, FONTS, CHART_DEFAULTS

@auth_required()
def show():
    """Main page content."""
    # Page title using style from style_config
    st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em; color: {COLORS["accent"]}; margin-bottom: 1em;">Show Detail</p>', unsafe_allow_html=True)
    
    # Add description
    st.write("View show details and success metrics, find similar titles ranked by match score, and analyze which networks are most active in this content space.")
    
    # Initialize analyzers
    from src.data_processing.show_detail.show_detail_analyzer import ShowDetailAnalyzer
    from src.data_processing.success_analysis.success_analyzer import SuccessAnalyzer
    from src.data_processing.analyze_shows import ShowsAnalyzer
    
    # Create analyzers in correct order
    shows_analyzer = ShowsAnalyzer()
    success_analyzer = SuccessAnalyzer(shows_analyzer=shows_analyzer)
    show_analyzer = ShowDetailAnalyzer(success_analyzer=success_analyzer)
    
    # Get success metrics first since ShowDetailAnalyzer needs them
    success_df = success_analyzer.fetch_success_data()
    
    # Get all shows for selection - this will automatically merge with success data
    shows_df = show_analyzer.fetch_show_data()
    
    # Show selector
    selected_show = st.selectbox(
        "Select a show",
        options=shows_df['title'].sort_values(),
        index=None,
        placeholder="Choose a show..."
    )
    
    if not selected_show:
        st.info("Please select a show to view details")
        return
        
    # Get show details
    show_data = shows_df[shows_df['title'] == selected_show].iloc[0]
    print(f"Show data columns: {show_data.index.tolist()}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Show Information", "Comp List", "Network Analysis"])
    
    with tab1:
        # Show information
        st.write(f"**Title:** {show_data['title']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Network:** {show_data['network_name']}")
            st.write(f"**Genre:** {show_data['genre_name']}")
            subgenres = [s for s in show_data.get('subgenre_names', []) if s is not None]
            st.write(f"**Subgenres:** {', '.join(subgenres) if subgenres else 'None'}")
            st.write(f"**Source Type:** {show_data['source_type_name']}")
        
        with col2:
            st.write(f"**Order Type:** {show_data['order_type_name']}")
            st.write(f"**Status:** {show_data['tmdb_status']}")
            st.write(f"**Seasons:** {show_data['tmdb_seasons']}")
            st.write(f"**Episodes:** {show_data['tmdb_total_episodes']}")
            st.write(f"**Announcement Date:** {show_data['date'] if pd.notna(show_data['date']) else 'None'}")

        
        st.write("")
        st.write("**Description**")
        st.write(show_data.get('description') or 'None')
        
        # Studios section
        if show_data.get('studios'):
            st.write("")
            st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; font-weight: 600; color: {COLORS["text"]["primary"]}; margin: 20px 0;">Studios</p>', unsafe_allow_html=True)
            for studio in show_data['studios']:
                st.write(f"- {studio}")
                
        # Team members section
        if show_data.get('team_members'):
            st.write("")
            st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; font-weight: 600; color: {COLORS["text"]["primary"]}; margin: 20px 0;">Team Members</p>', unsafe_allow_html=True)
            
            # Group members by name
            members_by_name = {}
            for member in show_data['team_members']:
                name = member.get('name')
                role = member.get('role')
                if name not in members_by_name:
                    members_by_name[name] = []
                members_by_name[name].append(role)
                
            # Display each person with their roles
            for name, roles in sorted(members_by_name.items()):
                st.write(f"- **{name}** ({', '.join(roles)})")
        
        st.write("")
        # Success metrics
        st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["title"]}px; font-weight: 600; color: {COLORS["text"]["primary"]}; margin: 20px 0;">Success Metrics</p>', unsafe_allow_html=True)
        # api_show_details uses 'id', api_success_metrics uses 'show_id' - they're the same value
        success_metrics = show_analyzer.get_success_metrics(show_data['id'])
        if success_metrics is None:
            st.write("No success metrics available")
            return
        st.metric("Success Score", f"{success_metrics['score']:.1f}/100")
        
        # Score breakdown
        st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; font-weight: 600; color: {COLORS["text"]["primary"]}; margin: 20px 0;">Score Breakdown</p>', unsafe_allow_html=True)
        
        # Show scoring factors from breakdown
        breakdown = success_metrics['breakdown']
        
        # Get formatted breakdown items from SuccessAnalyzer
        breakdown_items = success_analyzer.format_breakdown_for_display(breakdown)
        for item in breakdown_items:
            st.write(item)    
    with tab2:
        # Get similar shows
        similar_content = show_analyzer.find_similar_shows(show_data['id'])
        if similar_content:
            # Create dataframe
            comp_df = pd.DataFrame([
                {
                    "Show": show.title,
                    "Network": show.network_name,
                    "Match Score": show.match_score['total'],
                    "Success Score": show.success_score if show.success_score is not None else 'N/A'
                }
                for show in similar_content[:25]  # Limit to top 25 shows
            ])
            
            # Sort by match score then success score
            comp_df = comp_df.sort_values(['Match Score', 'Success Score'], ascending=[False, False])
            
            # Ensure correct column order
            comp_df = comp_df.reindex(columns=['Show', 'Network', 'Match Score', 'Success Score'])
            
            # Display table with custom formatting
            st.dataframe(
                comp_df,
                column_config={
                    "Show": st.column_config.TextColumn(
                        "Show",
                        help="Title of the similar show"
                    ),
                    "Network": st.column_config.TextColumn(
                        "Network",
                        help="Network or platform"
                    ),
                    "Match Score": st.column_config.ProgressColumn(
                        "Match",
                        help="Similarity score based on genre, team, source, and timing",
                        min_value=0,
                        max_value=100,
                        format="%.0f%%"
                    ),
                    "Success Score": st.column_config.NumberColumn(
                        "Success",
                        help="Overall success score based on renewals and episodes (N/A means not enough data)",
                        min_value=0,
                        max_value=100,
                        format="%.0f"
                    )
                },
                hide_index=True
            )
            
            # Show scoring explanation
            st.write("")
            with st.expander("ℹ️ Match Score Calculation"):
                st.markdown("""
        Shows are matched based on these factors (100 points total):

        Content Match (55 points):
        - Genre (40 points): 27 for primary genre match, +8 for first subgenre, +5 for second subgenre
        - Source (15 points): Same source type

        Production (30 points):
        - Network (7 points): Same network
        - Studio (3 points): Same production studio
        - Team (20 points): 6.7 points per shared team member (max 20)

        Format Match (15 points):
        - Episodes (8 points): Based on episodes per season difference
            * ≤2 eps: 8 points
            * ≤4 eps: 5 points
            * ≤6 eps: 2 points
        - Order Type (4 points): Same release pattern
        - Timing (3 points): Based on announcement date proximity
        """)
                
            # Show top 10 matches with detailed breakdown
            st.write("")
            # Section header
            st.markdown(
                f'<p style="font-family: {FONTS["primary"]["family"]}; '
                f'font-size: {FONTS["primary"]["sizes"]["header"]}px; '
                f'font-weight: 600; color: {COLORS["text"]["primary"]}; '
                f'margin: 20px 0;">Top Matches Breakdown</p>',
                unsafe_allow_html=True
            )
            
            # Sort by match score and take top 10
            top_matches = sorted(similar_content[:10], key=lambda x: x.match_score['total'], reverse=True)
            
            # Render each match breakdown
            for i, show in enumerate(top_matches):
                render_match_breakdown(show, expanded=(i == 0))  # Auto-expand first match
    with tab3:
        # Network pattern analysis
        network_patterns = show_analyzer.analyze_network_patterns(show_analyzer.find_similar_shows(show_data['id']))
        if network_patterns:
            # Sort networks by number of shows (descending)
            sorted_networks = sorted(network_patterns.similar_show_counts.keys(),
                                    key=lambda x: network_patterns.similar_show_counts[x],
                                    reverse=True)
            for i, network in enumerate(sorted_networks):
                score = network_patterns.success_scores[network]
                rate = network_patterns.success_rates[network]
                count = network_patterns.similar_show_counts[network]
                if i > 0:
                    st.write("")
                st.markdown(f"**{network}**")
                st.markdown("""
                <div style="margin-top: 5px;">
                • {count} similar shows<br>
                • Average Score: {score:.0f} pts<br>
                • Success Rate: {rate:.0%} of shows score 70+ pts
                </div>
                """.format(count=count, score=score, rate=rate), unsafe_allow_html=True)
        else:
            st.info("No network patterns found")

if __name__ == "__main__":
    show()
