"""Show Detail page for the TV Series Database Dashboard."""

import streamlit as st
import pandas as pd
import sys
import os

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
    
    # Initialize analyzers
    from src.data_processing.show_detail.show_detail_analyzer import ShowDetailAnalyzer
    show_analyzer = ShowDetailAnalyzer()
    
    # Get all shows for selection
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
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Show Information", "Comp List", "Network Analysis"])
    
    with tab1:
        # Show information
        st.write(f"**Title:** {show_data['title']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Network:** {show_data['network_name']}")
            st.write(f"**Genre:** {show_data['genre_name']}")
            st.write(f"**Subgenres:** {', '.join(show_data['subgenres']) if show_data.get('subgenres') else 'None'}")
            st.write(f"**Source Type:** {show_data['source_name']}")
        
        with col2:
            st.write(f"**Order Type:** {show_data['order_name']}")
            st.write(f"**Status:** {show_data['tmdb_status']}")
            st.write(f"**Seasons:** {show_data['tmdb_seasons']}")
            st.write(f"**Episodes:** {show_data['tmdb_total_episodes']}")
            st.write(f"**Announcement Date:** {show_data['announced_date'] if pd.notna(show_data['announced_date']) else 'None'}")
        
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
        success_score = show_data['success_score']
        st.metric("Success Score", f"{success_score:.1f}/100")
        
        # Score breakdown
        st.markdown(f'<p style="font-family: {FONTS["primary"]["family"]}; font-size: {FONTS["primary"]["sizes"]["header"]}px; font-weight: 600; color: {COLORS["text"]["primary"]}; margin: 20px 0;">Score Breakdown</p>', unsafe_allow_html=True)
        
        # Only show scoring factors that contribute points
        if pd.notna(show_data['tmdb_seasons']):
            seasons = int(show_data['tmdb_seasons'])
            if seasons >= 2:
                st.write("**Renewed for Season 2** _(+40 points)_")
                extra_seasons = seasons - 2
                if extra_seasons > 0:
                    bonus = min(extra_seasons * 20, 40)
                    st.write(f"**Additional seasons bonus** _(+{bonus} points)_")
        
        if pd.notna(show_data['tmdb_avg_eps']):
            avg_eps = float(show_data['tmdb_avg_eps'])
            if avg_eps >= 10:
                st.write("**High episode volume** _(+40 points)_")
            elif avg_eps >= 8:
                st.write("**Standard episode volume** _(+20 points)_")
        
        # Status modifier (only show if it affects score)
        status = show_data['tmdb_status']
        if status == 'Returning Series':
            st.write("**Active show bonus:** _Score multiplied by 1.2_")
        elif status == 'Canceled':
            st.write("**Canceled show penalty:** _Score multiplied by 0.8_")    
    with tab2:
        # Get similar shows
        similar_content = show_analyzer.find_similar_shows(show_data['show_id'])
        if similar_content:
            # Create dataframe
            comp_df = pd.DataFrame([
                {
                    "Show": show.title,
                    "Network": show.network_name,
                    "Match Score": show.match_score['total'],
                    "Success Score": show.success_score or 0
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
                    "Success Score": st.column_config.ProgressColumn(
                        "Success",
                        help="Overall success score based on renewals and episodes",
                        min_value=0,
                        max_value=100,
                        format="%.0f%%"
                    )
                },
                hide_index=True
            )
            
            # Show scoring explanation
            st.write("")
            with st.expander("ℹ️ Match Score Calculation"):
                st.write("""
                Shows are matched based on these factors:
                - Genre (40 points): 30 for first genre match (main or sub), +5 for second match, +5 for third match
                - Creative Team (30 points): 10 points per shared team member
                - Source Material (20 points): Same source type
                - Release Window (10 points): -2 points per year apart
                """)
        else:
            st.info("No similar shows found")
    
    with tab3:
        # Network pattern analysis
        network_patterns = show_analyzer.analyze_network_patterns(similar_content)
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
