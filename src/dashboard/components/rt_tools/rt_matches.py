"""Main RT matching interface component."""

import os
import json
import streamlit as st
from typing import Dict, Any, List, Optional
from urllib.parse import quote, unquote
import pandas as pd
from datetime import datetime
from pathlib import Path
import requests

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
(function(){var t=document.querySelector('h1');if(!t)return alert('Could not find show title');t=t.textContent.trim();var m=document.querySelector('rt-text[slot="criticsScore"]'),a=document.querySelector('rt-text[slot="audienceScore"]');var o=null,e=null;if(m){var mt=m.textContent.trim();mt=mt.match(/\d+/);o=mt?parseInt(mt[0]):null;}if(a){var at=a.textContent.trim();at=at.match(/\d+/);e=at?parseInt(at[0]):null;}if(!o||!e){var s=document.querySelectorAll('rt-text[context="label"]');Array.from(s).forEach(function(s){var txt=s.textContent.trim();var n=txt.match(/\d+/);if(n){var v=parseInt(n[0]);if(!o)o=v;else if(!e)e=v;}});}if(!o&&!e)return alert('Could not find show scores');var data={title:t,tomatometer:o,audience:e};var d=document.createElement('div');d.style.cssText='position:fixed;top:0;left:0;background:white;padding:20px;z-index:9999;border:2px solid black';d.innerHTML='<h3>'+t+'</h3><p>Tomatometer: '+o+'%</p><p>Audience: '+e+'%</p><p>Saving scores...</p>';document.body.appendChild(d);fetch('http://localhost:3000/submit-scores',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}).then(function(r){return r.json()}).then(function(j){if(j.status==='success'){d.innerHTML+='<p style="color:green">✓ Scores saved!</p>';setTimeout(function(){window.close()},500);}else{throw new Error(j.message||'Unknown error');}}).catch(function(e){d.innerHTML+='<p style="color:red">Error: '+e.message+'</p>';});})()"""
            
        # Initialize session state for scores and last check time
        if 'rt_scores' not in st.session_state:
            st.session_state.rt_scores = {}
        if 'last_score_check' not in st.session_state:
            st.session_state.last_score_check = datetime.now()
            
        # Set up scores directory path
        self.scores_dir = Path(__file__).parent.parent.parent.parent / 'data_processing' / 'external' / 'rt' / 'scores'
        
    def check_new_scores(self):
        """Check for new score files and process them."""
        try:
            # Ensure scores directory exists
            if not self.scores_dir.exists():
                st.warning("Score directory not found. Creating...")
                self.scores_dir.mkdir(parents=True, exist_ok=True)
                return
                
            # Get all json files modified after last check
            new_files = []
            for file in self.scores_dir.glob('*.json'):
                try:
                    if file.stat().st_mtime > st.session_state.last_score_check.timestamp():
                        new_files.append(file)
                except Exception as e:
                    st.error(f"Error checking file {file.name}: {e}")
                    continue
            
            if not new_files:
                return
                    
            # Process new files
            processed = 0
            errors = 0
            for file in new_files:
                try:
                    with open(file) as f:
                        data = json.load(f)
                    self.handle_score_message(data)
                    # Archive or delete file after processing
                    file.unlink()
                    processed += 1
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON in {file.name}: {e}")
                    errors += 1
                except Exception as e:
                    st.error(f"Error processing {file.name}: {e}")
                    errors += 1
                    
            # Show summary
            if processed > 0:
                st.success(f"✓ Processed {processed} new score files")
            if errors > 0:
                st.error(f"✗ Failed to process {errors} files")
                    
            # Update last check time
            st.session_state.last_score_check = datetime.now()
        except Exception as e:
            st.error(f"Error checking for new scores: {e}")
    
    def handle_score_message(self, data: Dict[str, Any]):
        """Handle score data from proxy server."""
        # Validate required fields
        if not data.get('title') or data.get('tomatometer') is None or data.get('audience') is None:
            raise ValueError("Missing required fields in score data")
            
        # Find matching show using fuzzy match
        show = None
        best_match_ratio = 0
        data_title = data['title'].lower()
        
        for s in self.shows:
            # Check for exact substring match first
            if s['title'].lower() in data_title or data_title in s['title'].lower():
                show = s
                break
            
            # If no exact match, try fuzzy matching
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, s['title'].lower(), data_title).ratio()
            if ratio > 0.8 and ratio > best_match_ratio:  # 80% similarity threshold
                show = s
                best_match_ratio = ratio
        
        if not show:
            raise ValueError(f"Could not match title: {data['title']}")
            
        # Save scores with timestamp
        st.session_state.rt_scores[show['id']] = {
            'tomatometer': data['tomatometer'],
            'popcornmeter': data['audience'],
            'title': show['title'],
            'captured_at': data.get('captured_at', datetime.now().isoformat())
        }
        
        st.success(f"✓ Saved scores for {show['title']}")
        return True  # Indicate successful processing
        
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
            1. First, make sure the RT Score Proxy is running (check status below)
            2. Drag the RT Helper to your bookmarks
            3. Use the batch search to open RT pages
            4. On each RT page:
               - Click the RT Helper bookmark
               - Verify the scores in the popup
               - Scores will be automatically saved
               - Page will auto-close when done
            5. Review and save all collected scores
            
            Note: If the proxy server isn't running, start it with:
            ```bash
            python3 src/data_processing/external/rt/proxy_server.py
            ```
            """)
            
            # Show proxy status
            st.markdown("### Proxy Server Status")
            try:
                proxy_status = requests.get('http://localhost:3000/health', timeout=1).json()
                st.success("✓ RT Score Proxy is running")
            except:
                st.error("✗ RT Score Proxy is not running")
                st.markdown("Please start the proxy server first using the command above.")
                return
            
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
            
                # Check for new scores from proxy
            st.markdown("##### 3. Score Collection Status")
            
            # Add status indicator for proxy server
            try:
                import requests
                proxy_status = requests.get('http://localhost:3000/health').json()
                st.success("✓ Proxy server is running")
            except:
                st.error("✗ Proxy server is not running. Please start it first.")
                st.markdown("Run this command in a terminal:")
                st.code("python src/data_processing/external/rt/proxy_server.py")
                return
                
            # Check for new scores
            self.check_new_scores()
            
            # Show search batches
            st.markdown("##### 4. Batch Search")
            
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
