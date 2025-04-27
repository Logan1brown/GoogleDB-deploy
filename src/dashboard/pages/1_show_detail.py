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
from src.dashboard.utils.style_config import COLORS, FONTS

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
    
    # Basic show information
    st.header("Show Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Network:**", show_data['network_name'])
        st.write("**Status:**", show_data['tmdb_status'])
        st.write("**Genre:**", show_data['genre_name'])
        
    with col2:
        st.write("**Source:**", show_data['source_name'])
        st.write("**Episodes:**", show_data['episode_count'])
        st.write("**Announced:**", show_data['announced_date'])
        
    # Success metrics
    st.header("Success Metrics")
    success_score = show_data['success_score']
    st.metric("Success Score", f"{success_score:.1f}/100")
    
    # Score breakdown
    st.subheader("Score Breakdown")
    col1, col2 = st.columns(2)
    
    with col1:
        if pd.notna(show_data['tmdb_seasons']):
            seasons = int(show_data['tmdb_seasons'])
            st.write(f"**Seasons:** {seasons}")
            if seasons >= 2:
                st.success("✓ Renewed for Season 2 (+40 points)")
                extra_seasons = seasons - 2
                if extra_seasons > 0:
                    bonus = min(extra_seasons * 20, 40)
                    st.success(f"✓ Additional seasons bonus (+{bonus} points)")
                    
    with col2:
        if pd.notna(show_data['tmdb_avg_eps']):
            avg_eps = float(show_data['tmdb_avg_eps'])
            st.write(f"**Average Episodes/Season:** {avg_eps:.1f}")
            if avg_eps >= 10:
                st.success("✓ High episode volume (+40 points)")
            elif avg_eps >= 8:
                st.success("✓ Standard episode volume (+20 points)")
                
    # Status modifier
    status = show_data['tmdb_status']
    if status == 'Returning Series':
        st.info("📈 Active show bonus: Score multiplied by 1.2")
    elif status == 'Ended':
        st.info("📊 Completed show: No modifier")
    elif status == 'Canceled':
        st.warning("📉 Canceled show penalty: Score multiplied by 0.8")
    
    # Similar shows
    st.header("Similar Shows")
    
    # Content-based similar shows
    st.subheader("Based on Content")
    show_id = show_data['show_id']
    similar_content = show_analyzer.find_similar_shows(show_id)
    if similar_content:
        similar_content_df = pd.DataFrame([
            {
                'Show': show.title,
                'Network': show.network_name,
                'Success Score': show.success_score or 0,
                'Similarity': show.match_score['total'] / 100
            }
            for show in similar_content
        ])
        st.dataframe(
            similar_content_df,
            column_config={
                "Show": st.column_config.TextColumn("Show"),
                "Network": st.column_config.TextColumn("Network"),
                "Success Score": st.column_config.ProgressColumn(
                    "Success",
                    min_value=0,
                    max_value=100,
                    format="%.0f%%"
                ),
                "Similarity": st.column_config.ProgressColumn(
                    "Similarity",
                    min_value=0,
                    max_value=1,
                    format="%.0f%%"
                )
            },
            hide_index=True
        )
    else:
        st.info("No similar shows found based on content")
        
    # Network pattern analysis
    st.subheader("Network Pattern Analysis")
    network_patterns = show_analyzer.analyze_network_patterns(similar_content)
    if network_patterns:
        # Show network stats
        st.write("**Network Success Rates:**")
        for network, rate in network_patterns.success_rates.items():
            st.write(f"- {network}: {rate:.0%} success rate")
            
        # Show network counts
        st.write("\n**Network Distribution:**")
        for network, count in network_patterns.similar_show_counts.items():
            st.write(f"- {network}: {count} similar shows")
    else:
        st.info("No network patterns found")

if __name__ == "__main__":
    show()
