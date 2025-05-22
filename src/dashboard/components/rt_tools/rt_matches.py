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
        
        # Inline bookmarklet code for reliability
        self.bookmarklet_code = """
(function(){
    // Extract data from RT page
    var title = document.querySelector('h1')?.textContent?.trim();
    
    // Get main show scores - try different selectors
    var tomatometer = null;
    var audience = null;
    
    // Try modern selectors first
    var tomatoEl = document.querySelector('[data-qa="tomatometer"]') || document.querySelector('.tomatometer-score');
    var audienceEl = document.querySelector('[data-qa="audience-score"]') || document.querySelector('.audience-score');
    
    if (tomatoEl) {
        tomatometer = parseInt(tomatoEl.textContent.trim());
    }
    if (audienceEl) {
        audience = parseInt(audienceEl.textContent.trim());
    }
    
    // Fallback to score containers
    if (!tomatometer || !audience) {
        var scores = Array.from(document.querySelectorAll('.score-container .percentage')).map(e => parseInt(e.textContent.trim()));
        if (scores.length >= 2) {
            tomatometer = tomatometer || scores[0];
            audience = audience || scores[1];
        }
    }
    
    if (!title || (!tomatometer && !audience)) {
        alert('Could not find show scores. Make sure you are on a show\'s main page.');
        return;
    }
    
    // Show overlay
    var d = document.createElement('div');
    d.style.cssText = 'position:fixed;top:0;left:0;background:white;padding:20px;z-index:9999;border:2px solid black';
    d.innerHTML = `
        <div style="font-family:sans-serif">
            <h3>${title}</h3>
            <p>Tomatometer: ${tomatometer}%</p>
            <p>Audience: ${audience}%</p>
        </div>
    `;
    document.body.appendChild(d);
    
    // Store data in localStorage of opener window
    var data = {
        title: title,
        tomatometer: tomatometer,
        audience: audience
    };
    
    window.opener.localStorage.setItem('rt_scores', JSON.stringify(data));
    
    // Auto close after showing scores
    setTimeout(function() {
        window.close();
    }, 1500);
})();"""
            
        # Initialize session state for scores
        if 'rt_scores' not in st.session_state:
            st.session_state.rt_scores = {}
        
    def handle_score_message(self, data: Dict[str, Any]):
        """Handle score data from bookmarklet"""
        st.write("Debug - Processing score data:", data)
        
        if not data.get('title') or data.get('tomatometer') is None or data.get('audience') is None:
            st.error("Invalid score data received")
            st.write("Debug - Missing required fields")
            return
            
        # Find matching show
        show = next((s for s in self.shows if s['title'].lower() in data['title'].lower()), None)
        if not show:
            st.error(f"Could not match title: {data['title']}")
            st.write("Debug - No matching show found")
            return
            
        st.write("Debug - Found matching show:", show)
            
        # Save scores
        st.session_state.rt_scores[show['id']] = {
            'tomatometer': data['tomatometer'],
            'popcornmeter': data['audience'],
            'title': show['title']
        }
        st.write("Debug - Saved scores to session state")
        st.success(f"Saved scores for {show['title']}")
        
    def render(self):
        """Render the RT matching interface."""
        st.write("Debug - RTMatches render start")
        
        # Handle score messages from component
        score_data = st.session_state.get("score_handler")
        if score_data:
            st.write("Debug - Received score data:", score_data)
            self.handle_score_message(score_data)
            del st.session_state["score_handler"]
        
        # Show debug info
        st.write("Debug - Shows:", json.dumps(self.shows, indent=2))
        
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
            
            # Add score checker
            score_checker = """
            <script>
            console.log('RT Score checker installed');
            
            // Function to check for scores
            function checkForScores() {
                const scores = localStorage.getItem('rt_scores');
                if (scores) {
                    console.log('Found scores:', scores);
                    try {
                        const data = JSON.parse(scores);
                        // Send to Streamlit
                        window.parent.Streamlit.setComponentValue(data);
                        localStorage.removeItem('rt_scores');
                        console.log('Sent scores to Streamlit');
                    } catch (e) {
                        console.error('Error parsing scores:', e);
                    }
                }
            }
            
            // Check periodically
            setInterval(checkForScores, 1000);
            </script>
            """
            st.components.v1.html(score_checker, height=0, key="score_checker")
            
            # Handle incoming scores from component
            score_data = st.session_state.get("score_checker")
            if score_data:
                st.write("Debug - Score data received:", score_data)
                self.handle_score_message(score_data)
                del st.session_state["score_checker"]
            
            # Show search batches
            st.markdown("##### 2. Search Shows")
            
            # Create batches of shows
            show_batches = [self.shows[i:i+2] for i in range(0, len(self.shows), 2)]
            st.write(f"Shows will open in {len(show_batches)} batches of 2 to avoid popup blocking")
            
            # Create buttons for each batch
            for i, batch in enumerate(show_batches):
                # Create URLs for this batch
                urls = []
                titles = []
                for show in batch:
                    titles.append(show['title'])
                    query = f"site:rottentomatoes.com tv {show['title']}"
                    url = f"https://www.google.com/search?q={quote(query)}"
                    urls.append(url)
                
                # Create batch button
                if st.button(f"Open Batch {i+1}: {', '.join(titles)}"):
                    js = ""
                    for url in urls:
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
