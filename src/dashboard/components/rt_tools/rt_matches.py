"""Main RT matching interface component."""

import os
import json
import streamlit as st
from typing import Dict, Any, List, Optional
from urllib.parse import unquote, quote

class RTMatches:
    """Component for RT show matching and score collection."""
    
    def __init__(self, unmatched_shows: List[Dict[str, Any]], on_scores_collected: Optional[callable] = None):
        """Initialize the RT matches component.
        
        Args:
            unmatched_shows: List of shows without RT metrics
            on_scores_collected: Callback when scores are collected
        """
        self.shows = unmatched_shows
        self.on_scores_collected = on_scores_collected
        
        # Load bookmarklet code
        bookmarklet_path = os.path.join(os.path.dirname(__file__), 'static', 'rt_bookmarklet.js')
        with open(bookmarklet_path) as f:
            self.bookmarklet_code = f.read()
        
    def render(self):
        """Render the RT matching interface."""
        st.markdown("## RT Metrics Collection")
        
        # Instructions
        with st.expander("Instructions"):
            st.markdown("""
            1. Drag the RT Helper to your bookmarks
            2. Use the batch search to open RT pages
            3. On each RT page:
               - Click the RT Helper bookmark
               - Verify the scores
               - The page will auto-close
            4. Review and save the collected scores
            """)
            
            # Bookmarklet
            st.markdown(f'<a href="javascript:{quote(self.bookmarklet_code)}">RT Helper</a>', unsafe_allow_html=True)
        
        # Batch controls
        st.markdown("### Unmatched Shows")
        if st.button("Search All in Batch", key="batch_search"):
            st.write("Debug - Starting batch search...")
            for show in self.shows:
                query = f"site:rottentomatoes.com tv {show['title']}"
                st.write(f"Debug - Opening search for: {show['title']}")
                import webbrowser
                webbrowser.open_new_tab(f"https://www.google.com/search?q={quote(query)}")
                st.write(f"Debug - Search opened for: {show['title']}")
        
        # Show table
        for show in self.shows:
            col1, col2 = st.columns([2, 8])
            with col1:
                if st.button(f"Search RT", key=f"search_{show['id']}"):
                    query = f"site:rottentomatoes.com tv {show['title']}"
                    st.write(f"Debug - Opening search for: {show['title']}")
                    import webbrowser
                    webbrowser.open_new_tab(f"https://www.google.com/search?q={quote(query)}")
                    st.write(f"Debug - Search opened for: {show['title']}")
            with col2:
                st.write(show['title'])
        
        # Handle incoming data
        params = st.query_params
        if "data" in params:
            try:
                data = json.loads(unquote(params["data"][0]))
                if data.get('title') and data.get('scores'):
                    st.markdown("### Review Scores")
                    st.json(data)
                    
                    # Save button
                    if st.button("Save Scores", key="save_scores"):
                        if self.on_scores_collected:
                            self.on_scores_collected(data)
                        st.success(f"Saved scores for {data['title']}")
            except Exception as e:
                st.error(f"Failed to parse data: {e}")
