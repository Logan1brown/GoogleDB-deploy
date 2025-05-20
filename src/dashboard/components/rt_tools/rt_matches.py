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
        
        # Batch search
        st.markdown("### Unmatched Shows")
        
        # Create batch search HTML
        urls = [f"https://www.google.com/search?q={quote(f'site:rottentomatoes.com tv {show["title"]}')}"
               for show in self.shows]
        urls_json = json.dumps(urls)
        
        html = f"""
        <div style="margin-bottom: 1rem;">
            <button id="batchButton" onclick="prepareSearch()" style="padding: 0.5rem 1rem; border-radius: 0.3rem; border: none; background-color: #ff4b4b; color: white; cursor: pointer;">
                🔍 Open All Searches ({len(self.shows)})
            </button>
            <div id="searchStatus" style="margin-top: 0.5rem; font-size: 0.9em;"></div>
        </div>
        <script>
            const urls = {urls_json};
            let currentIndex = 0;
            
            function updateStatus() {{
                const status = document.getElementById('searchStatus');
                if (currentIndex < urls.length) {{
                    status.innerHTML = `Click anywhere to open next search (${currentIndex + 1}/${urls.length})`;
                }} else {{
                    status.innerHTML = 'All searches opened!';
                }}
            }}
            
            function prepareSearch() {{
                const btn = document.getElementById('batchButton');
                btn.style.display = 'none';
                updateStatus();
                
                document.body.onclick = () => {{
                    if (currentIndex < urls.length) {{
                        window.open(urls[currentIndex], '_blank');
                        currentIndex++;
                        updateStatus();
                    }}
                }};
            }}
        </script>
        """
        
        st.components.v1.html(html, height=50)
        st.markdown("---")
        
        # Debug
        st.write("Debug - Shows:", self.shows)
        
        # Show table
        if self.shows:
            for show in self.shows:
                col1, col2 = st.columns([1, 9])
                with col1:
                    query = f"site:rottentomatoes.com tv {show['title']}"
                    search_url = f"https://www.google.com/search?q={quote(query)}"
                    st.markdown(f"[🔍]({search_url})")
                with col2:
                    st.markdown(f"**{show['title']}**")
        else:
            st.info("No unmatched shows found")
        
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
