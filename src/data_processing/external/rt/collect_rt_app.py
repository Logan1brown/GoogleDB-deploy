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
    st.info(f"Starting search for {show['title']}...")
    
    # Initialize browser only when needed
    try:
        with st.spinner("Initializing browser..."):
            collector.ensure_browser()
            st.success("Browser initialized")
    except Exception as e:
        st.error(f"Error initializing browser: {str(e)}")
        return []
    
    # Search for show
    try:
        from urllib.parse import quote_plus
        search_url = 'https://www.rottentomatoes.com/search?search=' + quote_plus(show['title'])
        st.info(f"Searching: {search_url}")
        
        with st.spinner("Loading search page..."):
            collector.page.goto(search_url)
            st.success("Page loaded")
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []
        
        # Wait for either TV section or no results message with a shorter timeout
        with st.spinner('Searching...'):
            try:
                collector.page.wait_for_selector('search-page-result[type="tvSeries"], .noresults', timeout=5000)
            except Exception as e:
                st.error("Search timed out. Resetting browser...")
                st.session_state.collector = RTCollector()  # Reset browser
                return []
    
        # Get all search results
        st.write("Debug: Looking for TV shows...")
        
        # Find the TV section with a final timeout
        try:
            tv_section = collector.page.query_selector('search-page-result[type="tvSeries"]')
        except Exception as e:
            st.error("Error finding TV section. Please try again.")
            return []
            
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        # Reset browser on any error
        st.session_state.collector = RTCollector()
        return []
    if not tv_section:
        st.warning("No TV section found")
        return []
        
    # Get all TV shows
    results = tv_section.query_selector_all('search-page-media-row')
    st.write(f"Debug: Found {len(results)} TV shows")
    
    # Get potential matches
    matches = []
    for result in results:
        title_elem = result.query_selector('a[data-qa="info-name"]')
        if not title_elem:
            continue
            
        title = title_elem.text_content().strip()
        url = title_elem.get_attribute('href')
        if title and url:
            # Extract year from URL if it ends with _YYYY
            url_parts = url.split('_')
            if len(url_parts) > 1 and url_parts[-1].isdigit() and len(url_parts[-1]) == 4:
                title = f"{title} ({url_parts[-1]})"
            
            st.write(f"Debug: Found TV show - Title: {title}, URL: {url}")
            matches.append({
                'title': title,
                'url': url,
                'element': title_elem
            })
    
    if not matches:
        st.warning("No TV shows found in search results")
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

    # Instructions
    st.markdown("""
    ### Instructions
    1. Select a show from the dropdown below
    2. Click 'Search for matches' to find it on Rotten Tomatoes
    3. Select the correct match from the results
    4. Click 'Get scores' to fetch the ratings
    5. Save the scores if they look correct
    """)
    
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
            if st.button("🔍 Search for matches"):
                with st.spinner("Searching..."):
                    try:
                        ensure_browser()
                        matches = search_for_matches(st.session_state.collector, selected_show)
                        st.session_state.current_matches = matches
                        st.session_state.selected_match = None
                        st.session_state.scores = None

                    except Exception as e:
                        st.error(f"Error searching: {str(e)}")
                        st.session_state.collector = None

            if st.session_state.current_matches:
                st.success(f"🌐 Found {len(st.session_state.current_matches)} potential matches")
                st.markdown("### Available Matches:")
                
                # Match selection
                for i, match in enumerate(st.session_state.current_matches):
                    with st.container():
                        st.markdown(f"**Match #{i+1}:** {match['title']}")
                        if st.button("Select this match", key=f"match_{i}"):
                            with st.spinner("Getting scores..."):
                                # Navigate to the URL
                                st.session_state.collector.page.goto(match['url'])
                                time.sleep(5)  # Wait for score board to load
                                
                                # Wait and retry if scores not loaded
                                retries = 3
                                while retries > 0:
                                    # Get critic and audience scores
                                    critic_score = st.session_state.collector.page.query_selector('rt-text[slot="criticsScore"]')
                                    audience_score = st.session_state.collector.page.query_selector('rt-text[slot="audienceScore"]')
                                    
                                    if critic_score and audience_score:
                                        try:
                                            # Get and clean scores
                                            critic_text = critic_score.text_content().strip()
                                            audience_text = audience_score.text_content().strip()
                                            
                                            # Extract numbers, handle empty, '-', or 'N/A' values
                                            def parse_score(text):
                                                if not text or text.strip() in ['-', 'N/A']:
                                                    return None
                                                return int(re.sub(r'[^0-9]', '', text))
                                                
                                            tomatometer = parse_score(critic_text)
                                            popcornmeter = parse_score(audience_text)
                                            
                                            st.success(f"Found scores! Tomatometer: {tomatometer if tomatometer else 'N/A'}%, Popcornmeter: {popcornmeter if popcornmeter else 'N/A'}%")
                                            
                                            print("DEBUG - Selected show:", selected_show)
                                            print("DEBUG - Current show in state:", st.session_state.current_show)
                                            
                                            # Save scores
                                            supabase = get_client()
                                            
                                            # Update rt_success_metrics
                                            supabase.table('rt_success_metrics').insert({
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
                                            
                                            st.success("✅ Scores saved successfully!")
                                            with st.spinner('Moving to next show in 3 seconds...'):
                                                time.sleep(3)  # Show success message for 3 seconds
                                            st.session_state.current_matches = None  # Clear matches to move to next show
                                            st.rerun()
                                            break
                                        except Exception as e:
                                            st.error(f"Error saving scores: {e}")
                                    
                                    time.sleep(2)
                                    retries -= 1
                                    
                                if retries == 0:
                                    st.error("Could not find scores after retries")
                                    supabase = get_client()
                                    supabase.table('rt_match_status')\
                                        .upsert({
                                            'show_id': selected_show['id'],
                                            'status': 'matched',
                                            'attempts': 1,
                                            'error_details': None,
                                            'manual_url': match['url']
                                        })\
                                        .execute()

        with col2:
            if st.session_state.selected_match:
                st.write("Selected match:")
                st.markdown(f"**{st.session_state.selected_match['title']}**")

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
                                })\
                                .execute()

                            # Update match status
                            supabase.table('rt_match_status')\
                                .upsert({
                                    'show_id': selected_show['id'],
                                    'status': 'matched',
                                    'manual_url': st.session_state.selected_match['url'],
                                    'last_attempt': 'now()'
                                })\
                                .execute()

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
