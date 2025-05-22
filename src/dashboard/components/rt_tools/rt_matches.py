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
(function(){var t=document.querySelector('h1');if(!t)return alert('Could not find show title');t=t.textContent.trim();var m=document.querySelector('[data-qa="tomatometer"]')||document.querySelector('.tomatometer-score'),a=document.querySelector('[data-qa="audience-score"]')||document.querySelector('.audience-score');alert('Selectors found: '+(m?'tomato ':'no tomato ')+(a?'audience':'no audience'));var o=null,e=null;if(m){var mt=m.textContent.trim();alert('Tomato text: '+mt);mt=mt.match(/\d+/);o=mt?parseInt(mt[0]):null;alert('Tomato score: '+o);}if(a){var at=a.textContent.trim();alert('Audience text: '+at);at=at.match(/\d+/);e=at?parseInt(at[0]):null;alert('Audience score: '+e);}if(!o||!e){alert('Trying fallback selectors');var s=document.querySelectorAll('.score-container .percentage');alert('Found '+s.length+' fallback scores');Array.from(s).forEach(function(s){var txt=s.textContent.trim();alert('Score text: '+txt);var n=txt.match(/\d+/);if(n){var v=parseInt(n[0]);if(!o){o=v;alert('Set tomato to '+v);}else if(!e){e=v;alert('Set audience to '+v);}}});}if(!o&&!e)return alert('Could not find show scores');var d=document.createElement('div');d.style.cssText='position:fixed;top:0;left:0;background:white;padding:20px;z-index:9999;border:2px solid black';d.innerHTML='<h3>'+t+'</h3><p>Tomatometer: '+(o||'N/A')+'%</p><p>Audience: '+(e||'N/A')+'%</p>';document.body.appendChild(d);window.opener.localStorage.setItem('rt_scores',JSON.stringify({title:t,tomatometer:o,audience:e}));setTimeout(function(){window.close()},1500);})()"""
            
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
            st.markdown("##### 1. Unmatched Shows")
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
            
            st.markdown("---")
            
            # Show bookmarklet and message handler
            st.markdown("##### 2. Install Score Collector")
            st.markdown("Drag this link to your bookmarks bar:")
            # Format bookmarklet code into a proper bookmarklet URL
            formatted_code = ''.join(line.strip() for line in self.bookmarklet_code.split('\n'))
            # Double-check the code is properly formatted
            if not formatted_code.startswith('(') or not formatted_code.endswith('()'):
                st.error('Bookmarklet code is malformed')
                return
            bookmarklet_url = f'javascript:{quote(formatted_code)}'
            st.markdown(f'<a href="{bookmarklet_url}">🎭 RT Score Collector</a>', unsafe_allow_html=True)
            
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
                        window.Streamlit.setComponentValue(data);
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
            # Create a component to receive scores
            score_component = st.components.v1.html(score_checker, height=0)
            
            # Handle incoming scores from component
            if 'score_receiver' in st.session_state:
                score_data = st.session_state.score_receiver
                if score_data:
                    st.write("Debug - Score data received:", score_data)
                    self.handle_score_message(score_data)
                    st.session_state.score_receiver = None
            
            # Show search batches
            st.markdown("##### 3. Batch Search")
            
            if not self.shows:
                st.info("No shows to search for")
                return
                
            # Create batches of shows
            show_batches = [self.shows[i:i+2] for i in range(0, len(self.shows), 2)]
            st.write(f"Shows will open in {len(show_batches)} batches of 2 to avoid popup blocking")
            
            # Create buttons for each batch
            for i, batch in enumerate(show_batches):
                # Create URLs for this batch
                urls = []
                titles = []
                for show in batch:
                    if isinstance(show, dict) and 'title' in show:
                        titles.append(show['title'])
                        query = f"site:rottentomatoes.com tv {show['title']}"
                        url = f"https://www.google.com/search?q={quote(query)}"
                        urls.append(url)
                
                if not urls:
                    continue
                    
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
