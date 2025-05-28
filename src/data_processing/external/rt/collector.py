"""Rotten Tomatoes Data Collector

This module handles automated collection of show data from Rotten Tomatoes using Playwright.
It includes:
1. Show search functionality
2. Score extraction
3. Rate limiting and error handling
4. Status tracking in database
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import quote

from playwright.async_api import async_playwright, Browser, Page, TimeoutError
from ...services.supabase import get_supabase_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RTCollector:
    """Handles automated collection of show data from Rotten Tomatoes."""
    
    def __init__(self):
        """Initialize the collector with default configuration."""
        self.supabase = get_supabase_client()
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
        # Rate limiting settings
        self.requests_per_minute = 10
        self.last_request_time = None
        
    async def __aenter__(self):
        """Set up Playwright browser when used as context manager."""
        await self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up browser resources."""
        self.page.close()
        self.context.close()
        self.browser.close()

    def collect_show_data(self, show_id: int) -> Dict:
        """Collect RT data for a show.

        Args:
            show_id: ID of the show

        Returns:
            Dict with success flag and either scores or error message
        """
        # Get show data
        show_response = self.supabase.table('shows')\
            .select('title')\
            .eq('id', show_id)\
            .execute()

        if not show_response.data:
            error = f"Show {show_id} not found"
            self.update_status(show_id, 'error', error)
            return {'success': False, 'error': error}

        title = show_response.data[0]['title']

        # Check if we already have scores
        metrics_response = self.supabase.table('rt_success_metrics')\
            .select('*')\
            .eq('show_id', show_id)\
            .execute()

        if metrics_response.data:
            return {
                'success': True,
                'cached': True,
                'scores': {
                    'tomatometer': metrics_response.data[0]['tomatometer'],
                    'popcornmeter': metrics_response.data[0]['popcornmeter']
                }
            }

        # Check last status
        last_status = self.get_last_status(show_id)
        if last_status and last_status['status'] == 'error' and last_status['attempts'] >= 3:
            return {
                'success': False,
                'error': f"Already failed {last_status['attempts']} times..."
            }

        # Try to find RT page
        url = self.find_rt_page(title)
        if not url:
            error = f"Could not find RT page for {title}"
            self.update_status(show_id, 'not_found', error)
            return {'success': False, 'error': error}

        # Get scores
        scores = self.get_rt_scores(url)
        if not scores:
            error = f"Could not extract scores from {url}"
            self.update_status(show_id, 'error', error)
            return {'success': False, 'error': error}

        # Save scores
        data = {
            'show_id': show_id,
            'rt_id': url.split('/')[-1],
            'tomatometer': scores['tomatometer'],
            'popcornmeter': scores['popcornmeter'],
            'is_matched': True
        }
        self.supabase.table('rt_success_metrics').insert(data).execute()

        # Update status
        self.update_status(show_id, 'success')

        return {
            'success': True,
            'cached': False,
            'scores': scores
        }

    def find_rt_page(self, title: str) -> Optional[str]:
        """Find the RT page for a show.

        Args:
            title: Show title to search for

        Returns:
            RT page URL or None if not found
        """
        # Navigate to RT search
        search_url = f"https://www.rottentomatoes.com/search?search={quote(title)}"
        self.page.goto(search_url)

        # Wait for search results
        self.page.wait_for_selector('.search-page-media-row')

        # Get TV show results
        results = self.page.query_selector_all('.search-page-media-row')

        for result in results:
            # Check if it's a TV show
            type_text = result.query_selector('.media-type')
            if not type_text:
                continue

            type_str = type_text.text_content()
            if 'TV Series' not in type_str:
                continue

            # Get title and URL
            title_link = result.query_selector('a[slot="title"]')
            if not title_link:
                continue

            result_title = title_link.text_content()
            result_url = title_link.get_attribute('href')

            # Check if titles match
            if self.titles_match(title, result_title):
                return f"https://www.rottentomatoes.com{result_url}"

        return None

    def get_rt_scores(self, url: str) -> Optional[Dict]:
        """Get RT scores from a show page.

        Args:
            url: RT page URL

        Returns:
            Dict with tomatometer and popcornmeter scores or None if not found
        """
        self.page.goto(url)

        # Wait for score elements
        self.page.wait_for_selector('score-board')

        # Get scores
        score_board = self.page.query_selector('score-board')
        if not score_board:
            return None

        tomatometer = score_board.get_attribute('tomatometerscore')
        popcornmeter = score_board.get_attribute('audiencescore')

        if not tomatometer or not popcornmeter:
            return None

        try:
            return {
                'tomatometer': int(tomatometer),
                'popcornmeter': int(popcornmeter)
            }
        except ValueError:
            return None

    def update_status(self, show_id: int, status: str, error: Optional[str] = None):
        """Update the status of a show in the database.

        Args:
            show_id: ID of the show
            status: Status to set (not_found, error, success)
            error: Optional error message
        """
        # Get existing status
        last_status = self.get_last_status(show_id)
        
        # Get current timestamp
        now = datetime.now()
        
        # Create new status entry
        data = {
            'show_id': show_id,
            'status': status,
            'error': error,
            'timestamp': now.isoformat(),
            'attempts': last_status['attempts'] + 1 if last_status else 1
        }
        
        # Insert into database
        self.supabase.table('rt_match_status').insert(data).execute()

    def get_last_status(self, show_id: int) -> Optional[Dict]:
        """Get the last status for a show.

        Args:
            show_id: ID of the show

        Returns:
            Last status or None if not found
        """
        # Get last status
        last_status_response = self.supabase.table('rt_match_status')\
            .select('*')\
            .eq('show_id', show_id)\
            .order('timestamp', desc=True)\
            .limit(1)\
            .execute()

        if last_status_response.data:
            return last_status_response.data[0]
        else:
            return None

    def titles_match(self, title1: str, title2: str) -> bool:
        """Check if two titles match.

        Args:
            title1: First title
            title2: Second title

        Returns:
            True if titles match, False otherwise
        """
        # Remove common words
        common_words = ['the', 'a', 'an']
        title1_words = [word for word in title1.lower().split() if word not in common_words]
        title2_words = [word for word in title2.lower().split() if word not in common_words]

        # Check if titles match
        return title1_words == title2_words

    def update_status(self, show_id: int, status: str, error: Optional[str] = None):
        try:
            status_data = {
                'show_id': show_id,
                'status': status,
                'last_attempt': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Add error details if present
            if error:
                status_data['error_details'] = error
            
            # Upsert status
            self.supabase.table('rt_match_status')\
                .upsert(status_data, on_conflict='show_id')\
                .execute()
                
        except Exception as e:
            logger.error(f"Error updating status for show {show_id}: {str(e)}")
