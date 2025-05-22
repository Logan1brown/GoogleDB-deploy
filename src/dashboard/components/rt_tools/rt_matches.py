"""Main RT matching interface component."""

import os
import json
import streamlit as st
from typing import Dict, Any, List, Optional
from urllib.parse import quote, unquote
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
        self.pending_scores = {}
        
        # Load bookmarklet code
        bookmarklet_path = os.path.join(os.path.dirname(__file__), 'static', 'rt_bookmarklet.js')
        st.write("Debug - Loading bookmarklet from:", bookmarklet_path)
        try:
            with open(bookmarklet_path) as f:
                self.bookmarklet_code = f.read()
            st.write("Debug - Bookmarklet loaded, first 100 chars:", self.bookmarklet_code[:100])
        except Exception as e:
            st.error(f"Error loading bookmarklet: {e}")
            
        # Initialize session state for scores
        if 'rt_scores' not in st.session_state:
            st.session_state.rt_scores = {}
        
    def handle_score_message(self, data: Dict[str, Any]):
        """Handle score data from bookmarklet"""
        if not data.get('title') or data.get('tomatometer') is None or data.get('audience') is None:
            st.error("Invalid score data received")
            return
            
        # Find matching show
        show = next((s for s in self.shows if s['title'].lower() in data['title'].lower()), None)
        if not show:
            st.error(f"Could not match title: {data['title']}")
            return
            
        # Save scores
        st.session_state.rt_scores[show['id']] = {
            'tomatometer': data['tomatometer'],
            'popcornmeter': data['audience'],
            'title': show['title']
        }
        st.success(f"Saved scores for {show['title']}")
        
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
            
            # Show bookmarklet and message handler
            st.markdown("##### 1. Install Score Collector")
            st.markdown("Drag this link to your bookmarks bar:")
            st.markdown(f'<a href="javascript:{quote(self.bookmarklet_code)}">🎭 RT Score Collector</a>', unsafe_allow_html=True)
            
            # Add message handler with debug
            message_handler = """
            <script>
            console.log('RT Score handler installed');
            window.addEventListener('message', function(event) {
                console.log('Message received:', event.data);
                if (event.data && event.data.type === 'rt_scores') {
                    console.log('Processing RT scores:', event.data.data);
                    // Send scores to Streamlit
                    const scores = event.data.data;
                    const args = encodeURIComponent(JSON.stringify(scores));
                    const url = `?rt_scores=${args}`;
                    console.log('Redirecting to:', url);
                    window.location.search = url;
                }
            });
            </script>
            """
            st.components.v1.html(message_handler, height=0)
            
            # Handle incoming scores
            params = st.query_params
            if 'rt_scores' in params:
                try:
                    scores = json.loads(unquote(params['rt_scores']))
                    self.handle_score_message(scores)
                    # Clear params
                    st.query_params.clear()
                except Exception as e:
                    st.error(f"Error processing scores: {e}")
            
            # Show search batches
            st.markdown("##### 2. Search Shows")
            all_urls = []
            for show in self.shows:
                query = f"site:rottentomatoes.com tv {show['title']}"
                url = f"https://www.google.com/search?q={quote(query)}"
                all_urls.append(url)
            
            batches = [all_urls[i:i+2] for i in range(0, len(all_urls), 2)]
            st.write(f"Shows will open in {len(batches)} batches of 2 to avoid popup blocking")
            
            for i, batch_urls in enumerate(batches):
                if st.button(f"Open Batch {i+1}: {', '.join(s['title'] for s in self.shows[i*2:(i+1)*2])}"):
                    js = ""
                    for url in batch_urls:
                        js += f"window.open('{url}', '_blank');"
                    st.components.v1.html(f"<script>{js}</script>", height=0)
                    st.success(f"Opening batch {i+1} in new tabs...")
                    
            # Show collected scores
            if st.session_state.rt_scores:
                st.markdown("##### Collected Scores")
                scores_df = pd.DataFrame([
                    {
                        'Title': data['title'],
                        'Tomatometer': f"{data['tomatometer']}%",
                        'Audience': f"{data['popcornmeter']}%"
                    }
                    for data in st.session_state.rt_scores.values()
                ])
                st.table(scores_df)
                
                if st.button("Save All Scores"):
                    try:
                        # Get client from session
                        if not st.session_state.get('supabase'):
                            st.error("No database connection available")
                            return
                            
                        client = st.session_state.supabase
                        for show_id, data in st.session_state.rt_scores.items():
                            response = client.table('rt_success_metrics').insert({
                                'show_id': show_id,
                                'tomatometer': data['tomatometer'],
                                'popcornmeter': data['popcornmeter']
                            }).execute()
                            
                            if not response.data:
                                raise Exception(f"Failed to save scores for {data['title']}")
                                
                        st.success("All scores saved successfully!")
                        st.session_state.rt_scores = {}
                        
                        # Refresh page to show updated data
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error saving scores: {e}")
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
