"""Local Streamlit app for collecting Rotten Tomatoes data."""
import os
import sys
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import streamlit as st
from dotenv import load_dotenv

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

# Load environment variables
load_dotenv()

from data_processing.external.rt.sync_collector import RTCollector
from supabase import create_client


def get_client():
    """Get Supabase client with service role."""
    return create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_SERVICE_KEY")
    )


def parse_score(text: str) -> Optional[int]:
    """Parse score from text, returning None if no valid score found."""
    try:
        # Extract just the number
        score = ''.join(c for c in text if c.isdigit())
        return int(score) if score else None
    except ValueError:
        return None


def search_for_matches(collector: RTCollector, show: Dict) -> List[Dict]:
    """Search for a show and get potential matches."""
    if not collector:
        st.error("No collector initialized")
        return []
        
    st.info(f"🔍 Searching for: {show['title']}")
    
    # Initialize browser if needed
    try:
        with st.spinner("Initializing browser..."):
            collector.ensure_browser()
    except Exception as e:
        st.error(f"Browser error: {str(e)}")
        return []
    
    # Search RT
    try:
        search_url = f'https://www.rottentomatoes.com/search?search={quote_plus(show["title"])}'
        with st.spinner("Searching Rotten Tomatoes..."):
            collector.page.goto(search_url)
            
            # Wait for results
            selector = 'search-page-result[type="tvSeries"], .noresults'
            collector.page.wait_for_selector(selector)
            
            # Check for no results
            if collector.page.query_selector('.noresults'):
                st.warning("No matches found")
                return []
                
            # Get TV show results
            results = collector.page.query_selector_all('search-page-result[type="tvSeries"]')
            matches = []
            
            for result in results:
                title = result.query_selector('.p--small').inner_text()
                url = result.get_attribute('url')
                matches.append({
                    'title': title,
                    'url': url
                })
                
            return matches
            
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []


def get_scores(collector: RTCollector, url: str) -> Dict[str, Optional[int]]:
    """Get Tomatometer and Audience scores for a show."""
    try:
        with st.spinner("Getting scores..."):
            collector.page.goto(url)
            
            # Wait for scores
            collector.page.wait_for_selector('score-board')
            
            # Get scores
            score_board = collector.page.query_selector('score-board')
            if not score_board:
                st.error("Could not find score board")
                return {'tomatometer': None, 'audience': None}
                
            critic_text = score_board.get_attribute('tomatometerscore')
            audience_text = score_board.get_attribute('audiencescore')
            
            return {
                'tomatometer': parse_score(critic_text) if critic_text else None,
                'audience': parse_score(audience_text) if audience_text else None
            }
            
    except Exception as e:
        st.error(f"Failed to get scores: {str(e)}")
        return {'tomatometer': None, 'audience': None}


def main():
    """Main function."""
    st.set_page_config(page_title="RT Collector", layout="wide")
    st.title("🍅 Rotten Tomatoes Score Collector")
    
    # Initialize collector if needed
    if 'collector' not in st.session_state:
        st.session_state.collector = RTCollector()
    
    # Get shows without RT scores
    try:
        supabase = get_client()
        response = supabase.table('shows').select('*').eq('rotten_tomatoes_score', None).limit(10).execute()
        shows = response.data
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        return
        
    if not shows:
        st.info("✅ No shows found without RT scores")
        return
        
    # Display shows
    st.write(f"Found {len(shows)} shows without RT scores")
    
    for show in shows:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(show['title'])
                if show.get('year'):
                    st.caption(f"Year: {show['year']}")
            
            with col2:
                if st.button("🔍 Search", key=f"search_{show['id']}"):
                    matches = search_for_matches(st.session_state.collector, show)
                    if matches:
                        st.write(f"Found {len(matches)} matches:")
                        for i, match in enumerate(matches):
                            if st.button(f"Get scores for: {match['title']}", key=f"scores_{show['id']}_{i}"):
                                scores = get_scores(st.session_state.collector, match['url'])
                                if scores['tomatometer'] or scores['audience']:
                                    st.success(
                                        f"Found scores! "
                                        f"Tomatometer: {scores['tomatometer'] if scores['tomatometer'] else 'N/A'}%, "
                                        f"Audience: {scores['audience'] if scores['audience'] else 'N/A'}%"
                                    )
                                    
                                    # Update database
                                    try:
                                        supabase.table('shows').update({
                                            'rotten_tomatoes_score': scores['tomatometer'],
                                            'rotten_tomatoes_audience_score': scores['audience']
                                        }).eq('id', show['id']).execute()
                                        st.success("Database updated successfully")
                                    except Exception as e:
                                        st.error(f"Failed to update database: {str(e)}")
                                else:
                                    st.error("Could not find scores")
                    else:
                        st.warning("No matches found")


if __name__ == "__main__":
    main()
