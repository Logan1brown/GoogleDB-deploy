"""Main RT matching interface component."""

import os
import json
import streamlit as st
from typing import Dict, Any, List, Optional
from urllib.parse import unquote, quote
import pandas as pd

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
        st.write("Debug - RTMatches render start")
        st.write("Debug - Shows:", self.shows)
        
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
        
        # Show table
        if self.shows:
            # Create table data
            data = []
            for show in self.shows:
                query = f"site:rottentomatoes.com tv {show['title']}"
                search_url = f"https://www.google.com/search?q={quote(query)}"
                data.append({
                    "Search": f"[🔍]({search_url})",
                    "Title": show['title']
                })
            
            # Display as table
            st.table(pd.DataFrame(data))
            
            # Add batch search section
            st.markdown("---")
            st.markdown("##### Batch Search")
            
            # Create batch search extension code
            urls = []
            for show in self.shows:
                query = f"site:rottentomatoes.com tv {show['title']}"
                url = f"https://www.google.com/search?q={quote(query)}"
                urls.append(url)
            
            extension_code = f"""
            javascript:(function() {{
                const urls = {json.dumps(urls)};
                let i = 0;
                function openNext() {{
                    if (i < urls.length) {{
                        window.open(urls[i], '_blank');
                        i++;
                        setTimeout(openNext, 500);
                    }}
                }}
                openNext();
            }})();""".replace('\n', '')
            
            # Show instructions
            st.markdown("To search all shows at once:")
            st.markdown("1. Drag this link to your bookmarks bar: ")
            st.markdown(f'<a href="{extension_code}">🔍 RT Batch Search</a>', unsafe_allow_html=True)
            st.markdown("2. Click the bookmark when you want to search all shows")
            st.caption("Note: You'll need to allow popups once when using this the first time")
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
