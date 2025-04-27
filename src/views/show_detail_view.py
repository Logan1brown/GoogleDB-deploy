"""Show Detail View.

This view displays detailed information about a TV show, including:
1. Basic show information (title, network, status)
2. Success metrics and performance analysis
3. Similar shows based on content and network patterns
"""

import streamlit as st
import pandas as pd

from src.data_processing.show_detail.show_detail_analyzer import ShowDetailAnalyzer
from src.data_processing.success_analysis.success_analyzer import SuccessAnalyzer

def render_show_detail_view():
    """Render the show detail view."""
    st.title("Show Details")
    
    # Initialize analyzers
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
    similar_content = show_analyzer.find_similar_shows(selected_show)
    if not similar_content.empty:
        similar_content_df = pd.DataFrame({
            'Show': similar_content.index,
            'Similarity Score': similar_content.values
        })
        st.dataframe(
            similar_content_df,
            column_config={
                "Show": st.column_config.TextColumn("Show"),
                "Similarity Score": st.column_config.ProgressColumn(
                    "Similarity",
                    min_value=0,
                    max_value=100,
                    format="%.0f%%"
                )
            },
            hide_index=True
        )
    else:
        st.info("No similar shows found based on content")
        
    # Network pattern analysis
    st.subheader("Network Pattern Analysis")
    network_patterns = show_analyzer.analyze_network_patterns(selected_show)
    if network_patterns:
        for pattern in network_patterns:
            st.write(f"**{pattern['pattern']}**")
            st.write(f"Confidence: {pattern['confidence']:.0%}")
            if pattern.get('examples'):
                st.write("Examples:", ", ".join(pattern['examples'][:3]))
    else:
        st.info("No network patterns found")
