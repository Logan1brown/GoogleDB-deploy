"""Rotten Tomatoes Data Collector (Sync Version)

This module handles automated collection of show data from Rotten Tomatoes using sync Playwright.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, Optional
from urllib.parse import quote

from playwright.sync_api import sync_playwright, Browser, Page, TimeoutError

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from src.dashboard.services.supabase import get_supabase_client
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTCollector:
    """Handles automated collection of show data from Rotten Tomatoes."""
    
    def __init__(self):
        """Initialize the collector with default configuration."""
        self.supabase = get_supabase_client()
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        
    def __enter__(self):
        """Set up Playwright browser when used as context manager."""
        playwright = sync_playwright().start()
        self.browser = playwright.chromium.launch(headless=True)
        self.page = self.browser.new_page()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up browser resources."""
        if self.browser:
            self.browser.close()

    def collect_show_data(self, show_id: int) -> Dict:
        """Collect RT data for a show.

        Args:
            show_id: ID of the show to collect data for

        Returns:
            Dict with success status and scores or error message
        """
        try:
            # Update status to pending
            self.update_status(show_id, 'pending')
            
            # Get show title
            response = self.supabase.table('shows').select('title').eq('id', show_id).execute()
            if not response.data:
                raise ValueError(f"Show {show_id} not found")
            
            title = response.data[0]['title']
            logger.info(f"Getting scores for {title}")
            
            # Find RT page
            url = self.find_rt_page(title)
            if not url:
                self.update_status(show_id, 'not_found')
                return {'success': False, 'error': 'Show not found on RT'}
            
            # Get scores
            scores = self.get_rt_scores(url)
            if not scores:
                self.update_status(show_id, 'error', 'Could not get scores')
                return {'success': False, 'error': 'Could not get scores'}
            
            # Save scores
            data = {
                'show_id': show_id,
                'tomatometer': int(scores['critics'].rstrip('%')),
                'popcornmeter': int(scores['audience'].rstrip('%')),
                'is_matched': True,
                'last_update': datetime.now().isoformat()
            }
            
            self.update_status(show_id, 'pending')
            self.supabase.table('rt_success_metrics').upsert(data).execute()
            self.update_status(show_id, 'matched')
            
            return {
                'success': True,
                'scores': scores
            }
            
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            self.update_status(show_id, 'error', str(e))
            return {'success': False, 'error': str(e)}

    def clean_title(self, title: str) -> str:
        """Clean a title for comparison by removing year ranges and common suffixes."""
        # Remove year ranges like (2019 - 2023)
        import re
        title = re.sub(r'\s*\([0-9]{4}\s*-\s*[0-9]{4}\)\s*$', '', title)
        title = re.sub(r'\s*\([0-9]{4}\)\s*$', '', title)
        
        # Remove common suffixes
        suffixes = [': The Series', ': The Show']
        for suffix in suffixes:
            if title.endswith(suffix):
                title = title[:-len(suffix)]
        
        return title.strip().lower()
    
    def titles_match(self, title1: str, title2: str) -> bool:
        """Check if two titles match after cleaning."""
        clean1 = self.clean_title(title1)
        clean2 = self.clean_title(title2)
        
        # Try exact match first
        if clean1 == clean2:
            return True
            
        # Try substring match
        if clean1 in clean2 or clean2 in clean1:
            return True
            
        return False

    def find_rt_page(self, title: str) -> Optional[str]:
        """Find the RT page for a show."""
        try:
            # Go to search with TV filter
            url = f'https://www.rottentomatoes.com/search?search={quote(title)}&type=tv'
            logger.info(f"Searching for: {title}")
            self.page.goto(url)
            
            # Wait for search results to load
            import time
            time.sleep(2)  # Give it 2 seconds to load
            logger.info("Page loaded, looking for results...")
            
            # Look for show link (exclude season links)
            logger.info("Searching for TV links...")
            links = self.page.query_selector_all('a[href*="/tv/"]')
            logger.info(f"Found {len(links)} TV links")
            for link in links:
                href = link.get_attribute('href')
                text = link.text_content()
                
                # Skip season links (ending in /s01, /s02, etc)
                if href and '/s0' in href:
                    logger.info(f"Skipping season link: {href}")
                    continue
                    
                logger.info(f"Found: {text.strip()} -> {href}")
                
                # Debug logging
                clean_title = self.clean_title(title)
                clean_text = self.clean_title(text)
                logger.info(f"Comparing: '{clean_title}' with '{clean_text}'")
                
                # Use flexible title matching
                if clean_title in clean_text or clean_text in clean_title:
                    logger.info(f"Clicking main show link: {href}")
                    link.click()
                    # Wait for both score elements to appear
                    try:
                        self.page.wait_for_selector('rt-text[slot="criticsScore"]', timeout=5000)
                        self.page.wait_for_selector('rt-text[slot="audienceScore"]', timeout=5000)
                        return href
                    except TimeoutError:
                        logger.error("Timeout waiting for score elements")
                        return None
            
            logger.error(f"Could not find {title}")
            return None
        except Exception as e:
            logger.error(f"Error finding RT page: {e}")
            return None

    def get_rt_scores(self, url: str) -> Optional[Dict[str, str]]:
        """Get RT scores from a show page."""
        try:
            scores = {}
            
            # Get critics score
            critics = self.page.query_selector('rt-text[slot="criticsScore"]')
            if critics:
                scores['critics'] = critics.text_content()
            
            # Get audience score
            audience = self.page.query_selector('rt-text[slot="audienceScore"]')
            if audience:
                scores['audience'] = audience.text_content()
            
            logger.info(f"Scores for show:")
            logger.info(f"Critics: {scores.get('critics', 'Not found')}")
            logger.info(f"Audience: {scores.get('audience', 'Not found')}")
            return scores if scores else None
        except Exception as e:
            logger.error(f"Error getting RT scores: {e}")
            return None

    def update_status(self, show_id: int, status: str, error: Optional[str] = None):
        """Update the status of a show in the database."""
        # Map our status to DB enum
        status_map = {
            'not_found': 'not_found',
            'error': 'error',
            'pending': 'pending',
            'matched': 'matched'
        }
        
        status_data = {
            'show_id': show_id,
            'status': status_map[status],
            'error': error,
            'last_attempt': datetime.now().isoformat()
        }
        
        try:
            self.supabase.table('rt_match_status').upsert(status_data, on_conflict='show_id').execute()
        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def get_last_status(self, show_id: int) -> Optional[Dict]:
        """Get the last status for a show."""
        try:
            response = self.supabase.table('rt_match_status')\
                .select('*')\
                .eq('show_id', show_id)\
                .order('last_attempt', desc=True)\
                .limit(1)\
                .execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting last status: {e}")
            return None
