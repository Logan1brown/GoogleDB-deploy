"""Local Streamlit app for collecting Rotten Tomatoes data."""
import streamlit as st
import time
import sys
import os
import re
from urllib.parse import quote
import os
import sys
import streamlit as st
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from data_processing.external.rt.sync_collector import RTCollector
from supabase import create_client

# Initialize session state
if 'collector' not in st.session_state:
    st.session_state.collector = None

if 'current_show' not in st.session_state:
    st.session_state.current_show = None

def get_client():
    """Get Supabase client with service role."""
    return create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_SERVICE_KEY")
    )

def search_for_matches(collector: RTCollector, show: Dict) -> List[Dict]:
    """Search for a show and get potential matches."""
    if not collector:
        st.error("No collector initialized")
        return []
        
    st.session_state.current_show = show  # Store entire show object
    
    # Initialize browser only when needed
    try:
        with st.spinner("Searching..."):
            collector.ensure_browser()
            
            # Load search page
            from urllib.parse import quote_plus
            search_url = 'https://www.rottentomatoes.com/search?search=' + quote_plus(show['title'])
            collector.page.goto(search_url)
            
            # Wait for TV shows to load
            collector.page.wait_for_selector('search-page-result[type="tvSeries"]', timeout=5000)
            
            # Get TV shows section
            tv_section = collector.page.query_selector('search-page-result[type="tvSeries"]')
            if not tv_section:
                return []
                
            # Get all TV show rows
            results = tv_section.query_selector_all('search-page-media-row')
            if not results:
                return []
            
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        # Reset browser on any error
        st.session_state.collector = RTCollector()
        return []
    
    # Get potential matches
    matches = []
    for result in results:
        # Get title and URL from the result
        title_elem = result.query_selector('a[data-qa="info-name"]')
        if title_elem:
            title = title_elem.text_content().strip()
            url = title_elem.get_attribute('href')
        else:
            title = None
            url = None
            
        if title and url:
            # Extract year from URL if it ends with _YYYY
            url_parts = url.split('_')
            if len(url_parts) > 1 and url_parts[-1].isdigit() and len(url_parts[-1]) == 4:
                title = f"{title} ({url_parts[-1]})"
                
            matches.append({
                'title': title,
                'url': url,
                'element': result
            })
    return matches

def get_unmatched_shows() -> List[Dict]:
    """Get shows that need RT scores."""
    supabase = get_client()
    
    # Get IDs of shows that have been matched
    matched = supabase.table('rt_match_status')\
        .select('show_id')\
        .neq('status', 'error')\
        .execute()
    matched_ids = [row['show_id'] for row in matched.data]
    
    # Get shows that haven't been matched
    response = supabase.table('shows')\
        .select('id, title')\
        .not_.in_('id', matched_ids)\
        .order('title')\
        .execute()
    return response.data

# Configure Streamlit page
st.set_page_config(page_title="RT Data Collector", layout="wide")

def main():
    """Main function."""
    st.title(" Rotten Tomatoes Data Collector")
    
    # Initialize browser if needed
    try:
        if 'collector' not in st.session_state:
            with st.spinner('Initializing browser...'):
                try:
                    # Set a timeout for browser initialization
                    with st.empty():
                        st.session_state.collector = RTCollector()
                        st.session_state.collector.page.set_default_timeout(5000)
                        # Test browser by loading a simple page
                        st.session_state.collector.page.goto('about:blank', timeout=5000)
                except Exception as e:
                    st.error("Browser initialization timed out. Please refresh the page.")
                    return
    except Exception as e:
        st.error(f"Error initializing browser: {str(e)}")
        return
    
    # Initialize session state variables
    if 'current_show' not in st.session_state:
        st.session_state.current_show = None
    
    if 'current_matches' not in st.session_state:
        st.session_state.current_matches = []
        st.session_state.selected_match = None
        st.session_state.scores = None

    # Get shows that need scores
    shows = get_unmatched_shows()
    st.info(f"📺 Found {len(shows)} shows that need RT scores")

    # Show selector
    selected_show = st.selectbox(
        "Select a show to process",
        options=shows,
        format_func=lambda x: x['title']
    )

    if selected_show:
        st.write("---")
        st.subheader(f"Processing: {selected_show['title']}")

        def ensure_browser():
            """Ensure browser is initialized and working"""
            try:
                if not st.session_state.collector or not st.session_state.collector.page:
                    raise Exception("Browser not initialized")
                # Test if browser is responsive
                st.session_state.collector.page.evaluate('1')
            except:
                if st.session_state.collector:
                    try:
                        st.session_state.collector.__exit__(None, None, None)
                    except:
                        pass
                st.session_state.collector = RTCollector()
                st.session_state.collector.__enter__()

        # Initialize browser
        with st.spinner("Checking browser..."):
            ensure_browser()

        col1, col2 = st.columns(2)

        with col1:
            search_col1, search_col2 = st.columns([1, 1])
            with search_col1:
                if st.button("🔍 Search for matches"):
                    try:
                        ensure_browser()
                        matches = search_for_matches(st.session_state.collector, selected_show)
                        # Clean up browser after search
                        st.session_state.collector.cleanup()
                        st.session_state.current_matches = matches
                        st.session_state.selected_match = None
                        st.session_state.scores = None

                    except Exception as e:
                        st.error(f"Error searching: {str(e)}")
                        st.session_state.collector = None
            
            with search_col2:
                if st.button("⛔️ Not Found"):
                    with st.spinner("Marking as not found..."):
                        # Update rt_match_status to not_found
                        supabase = get_client()
                        supabase.table('rt_match_status').upsert({
                            'show_id': selected_show['id'],
                            'status': 'not_found',
                            'attempts': 1,
                            'error_details': None,
                            'manual_url': None
                        }).execute()
                        st.success("✅ Show marked as not found")
                        time.sleep(2)  # Show success message
                        st.session_state.current_matches = None  # Clear any existing matches
                        st.rerun()

            if st.session_state.current_matches:
                st.markdown("### Available Matches:")
                
                # Match selection
                for i, match in enumerate(st.session_state.current_matches):
                    with st.container():
                        st.markdown(f"**Match #{i+1}:** {match['title']}")
                        if st.button("Select this match", key=f"match_{i}"):
                            try:
                                with st.spinner("Loading scores..."):
                                    st.session_state.collector.page.goto(match['url'])
                                    # Wait for media scorecard to load
                                    scorecard = st.session_state.collector.page.wait_for_selector('media-scorecard', timeout=8000)
                                    if not scorecard:
                                        st.error("Could not find scores section on the page")
                                        return

                                    # Look for score text first
                                    critic_score = st.session_state.collector.page.query_selector('rt-text[slot="criticsScore"]')
                                    audience_score = st.session_state.collector.page.query_selector('rt-text[slot="audienceScore"]')
                                    
                                    # Get text content if available
                                    critic_text = critic_score.text_content().strip() if critic_score else None
                                    audience_text = audience_score.text_content().strip() if audience_score else None
                                    
                                    # If no text content, check for empty score icons
                                    if not critic_text:
                                        critic_empty = st.session_state.collector.page.query_selector('score-icon-critics[sentiment="empty"]')
                                        if critic_empty:
                                            critic_text = "--"
                                            
                                    if not audience_text:
                                        audience_empty = st.session_state.collector.page.query_selector('score-icon-audience[sentiment="empty"]')
                                        if audience_empty:
                                            audience_text = "--"
                                            
                                    # If we still don't have any scores, error out
                                    if not critic_text and not audience_text:
                                        st.error("Could not find score elements on the page")
                                        return

                                    # Extract numbers, handle empty, '-', '--', or 'N/A' values
                                    def parse_score(text):
                                        if not text or text.strip() in ['-', '--', 'N/A']:
                                            return None
                                        # Remove any non-numeric characters and try to parse
                                        clean_text = re.sub(r'[^0-9]', '', text)
                                        return int(clean_text) if clean_text else None

                                    tomatometer = parse_score(critic_text)
                                    popcornmeter = parse_score(audience_text)

                                    # Show the scores prominently (even if they're null)
                                    st.markdown("### 🎯 Scores Found!")
                                    score_col1, score_col2 = st.columns(2)
                                    with score_col1:
                                        st.metric("Tomatometer", "No Score Yet" if tomatometer is None else f"{tomatometer}%")
                                    with score_col2:
                                        st.metric("Popcornmeter", "No Score Yet" if popcornmeter is None else f"{popcornmeter}%")

                                with st.spinner("Saving to database..."):
                                    # Save scores
                                    supabase = get_client()
                                    # Update rt_success_metrics
                                    supabase.table('rt_success_metrics').upsert({
                                        'show_id': selected_show['id'],
                                        'tomatometer': tomatometer,
                                        'popcornmeter': popcornmeter,
                                        'is_matched': True
                                    }).execute()

                                    # Update rt_match_status
                                    supabase.table('rt_match_status').upsert({
                                        'show_id': selected_show['id'],
                                        'status': 'matched',
                                        'attempts': 1,
                                        'error_details': None,
                                        'manual_url': match['url']
                                    }).execute()

                                    # Show success and wait
                                    st.success("✅ Successfully saved scores to database!")
                                    time.sleep(3)  # Give time to see the results
                                    st.session_state.current_matches = None  # Clear matches to move to next show
                                    st.rerun()
                                    break
                            except Exception as e:
                                st.error(f"Error loading page or saving scores: {str(e)}")
                                return

        with col2:
            if st.session_state.selected_match:
                st.write("Selected match:")
                st.markdown(f"**{st.session_state.selected_match['title']}**")

# ... (rest of the code remains the same)
            if st.session_state.scores:
                st.write("Scores found:")
                st.write(f"- Tomatometer: {st.session_state.scores['tomatometer']}%")
                st.write(f"- Audience Score: {st.session_state.scores['audience_score']}%")

                if st.button("💾 Save scores"):
                    with st.spinner("Saving..."):
                        try:
                            # Save scores
                            supabase = get_client()
                            
                            # Update success metrics
                            supabase.table('rt_success_metrics')\
                                .upsert({
                                    'show_id': selected_show['id'],
                                    'tomatometer': st.session_state.scores['tomatometer'],
                                    'popcornmeter': st.session_state.scores['audience_score'],
                                    'is_matched': True
                                }).execute()

                            # Update match status
                            supabase.table('rt_match_status').upsert({
                                'show_id': selected_show['id'],
                                'status': 'matched',
                                'manual_url': st.session_state.selected_match['url'],
                                'last_attempt': 'now()'
                            }).execute()

                            st.success("Scores saved!")
                            time.sleep(1)
                            st.rerun()  # Refresh to show updated list

                        except Exception as e:
                            st.error(f"Error saving scores: {e}")

    # Cleanup when the app is closed
    if st.session_state.collector:
        import atexit
        atexit.register(lambda: st.session_state.collector.__exit__(None, None, None))

if __name__ == '__main__':
    main()
