"""Rotten Tomatoes Data Collector

This module handles automated collection of show data from Rotten Tomatoes using Playwright.
It includes:
1. Show search functionality
2. Score extraction
3. Database updates
"""

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, Optional
from urllib.parse import quote

from playwright.async_api import async_playwright, Browser, Page, TimeoutError

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
        
        # Rate limiting settings
        self.requests_per_minute = 10
        self.last_request_time = None
        
    async def __aenter__(self):
        """Set up Playwright browser when used as context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up browser resources."""
        pass
        if self.browser:
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()

    async def collect_show_data(self, show_id: int) -> Dict:
        """Collect RT data for a show.

        Args:
            show_id: ID of the show to collect data for

        Returns:
            Dict with success status and scores or error message
        """
        logger.info(f"Starting collection for show {show_id}")
        
        # Get show title
        try:
            response = await self.supabase.table('shows').select('title').eq('id', show_id).execute()
            if not response.data:
                error = f"Show {show_id} not found"
                await self.update_status(show_id, 'not_found', error)
                return {'success': False, 'error': error}

            title = response.data[0]['title']
            logger.info(f"Found show title: {title}")

            # Find RT page
            url = await self.find_rt_page(title)
            if not url:
                error = f"Could not find RT page for {title}"
                await self.update_status(show_id, 'not_found', error)
                return {'success': False, 'error': error}

            # Get scores
            scores = await self.get_rt_scores(url)
            if not scores:
                error = f"Could not get scores for {title}"
                await self.update_status(show_id, 'error', error)
                return {'success': False, 'error': error}

            # Save scores
            data = {
                'show_id': show_id,
                'rt_id': str(uuid.uuid5(uuid.NAMESPACE_URL, url)),  # Generate UUID from URL
                'tomatometer': int(scores['critics'].strip('%')),  # Convert to int
                'popcornmeter': int(scores['audience'].strip('%')),  # Convert to int
                'is_matched': True,
                'last_updated': datetime.now().isoformat()
            }
            await self.update_status(show_id, 'pending')
            await self.supabase.table('rt_success_metrics').upsert(data).execute()
            await self.update_status(show_id, 'matched')
        except Exception as e:
            logger.error(f"Error saving scores: {e}")
            await self.update_status(show_id, 'error', str(e))
            return {'success': False, 'error': str(e)}

        return {
            'success': True,
            'scores': scores
        }

    async def find_rt_page(self, title: str) -> Optional[str]:
        """Find the RT page for a show.

        Args:
            title: Show title to search for

        Returns:
            RT page URL or None if not found
        """
        try:
            # Go to search with TV filter
            url = f'https://www.rottentomatoes.com/search?search={quote(title)}&type=tv'
            logger.info(f"Searching for: {title}")
            await self.page.goto(url)
            
            # Look for show link (exclude season links)
            links = await self.page.query_selector_all('a[href*="/tv/"]')
            for link in links:
                href = await link.get_attribute('href')
                text = await link.text_content()
                
                # Skip season links (ending in /s01, /s02, etc)
                if href and '/s0' in href:
                    continue
                    
                logger.info(f"Found: {text.strip()} -> {href}")
                
                # Click first matching link that's not a season
                if title.lower() in text.lower():
                    logger.info(f"Clicking main show link: {href}")
                    await link.click()
                    # Wait for both score elements to appear
                    try:
                        await self.page.wait_for_selector('rt-text[slot="criticsScore"]', timeout=5000)
                        await self.page.wait_for_selector('rt-text[slot="audienceScore"]', timeout=5000)
                        return href
                    except TimeoutError:
                        logger.error("Timeout waiting for score elements")
                        return None
            
            logger.error(f"Could not find {title}")
            return None
        except Exception as e:
            logger.error(f"Error finding RT page: {e}")
            return None

    async def get_rt_scores(self, url: str) -> Optional[Dict[str, str]]:
        """Get RT scores from a show page.

        Args:
            url: RT page URL

        Returns:
            Dict with critics and audience scores or None if not found
        """
        try:
            scores = {}
            
            # Get critics score
            critics = await self.page.query_selector('rt-text[slot="criticsScore"]')
            if critics:
                scores['critics'] = await critics.text_content()
            
            # Get audience score
            audience = await self.page.query_selector('rt-text[slot="audienceScore"]')
            if audience:
                scores['audience'] = await audience.text_content()
            
            logger.info(f"Scores for show:")
            logger.info(f"Critics: {scores.get('critics', 'Not found')}")
            logger.info(f"Audience: {scores.get('audience', 'Not found')}")
            return scores if scores else None
        except Exception as e:
            logger.error(f"Error getting RT scores: {e}")
            return None

    async def update_status(self, show_id: int, status: str, error: Optional[str] = None):
        """Update the status of a show in the database.

        Args:
            show_id: ID of the show
            status: Status to set (not_found, error, success)
            error: Optional error message
        """
        # Map our status to DB enum
        status_map = {
            'not_found': 'not_found',
            'error': 'error',
            'success': 'matched',
            'pending': 'pending',
            'matched': 'matched'
        }
        db_status = status_map.get(status, 'error')
        
        # Get existing status
        last_status = await self.get_last_status(show_id)
        
        # Create new status entry
        data = {
            'show_id': show_id,
            'status': db_status,
            'error_details': error,
            'attempts': last_status['attempts'] + 1 if last_status else 1
        }
        
        # Insert or update
        self.supabase.table('rt_match_status').upsert(data).execute()

    async def get_last_status(self, show_id: int) -> Optional[Dict]:
        """Get the last status for a show.

        Args:
            show_id: ID of the show

        Returns:
            Last status or None if not found
        """
        try:
            # Get last status
            last_status_response = await self.supabase.table('rt_match_status')\
                .select('*')\
                .eq('show_id', show_id)\
                .execute()

            if last_status_response.data:
                return last_status_response.data[0]
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting last status: {e}")
            return None

    async def titles_match(self, title1: str, title2: str) -> bool:
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

    async def update_status(self, show_id: int, status: str, error: Optional[str] = None):
        try:
            # Map status to database status
            status_map = {
                'not_found': 'not_found',
                'error': 'error',
                'success': 'matched',
                'pending': 'pending',
                'matched': 'matched'
            }
            db_status = status_map.get(status, 'error')
            
            # Get existing status for attempt count
            last_status = await self.get_last_status(show_id)
            
            now = datetime.now()
            status_data = {
                'show_id': show_id,
                'status': db_status,
                'last_attempt': now.isoformat(),
                'updated_at': now.isoformat(),
                'attempts': last_status['attempts'] + 1 if last_status else 1
            }
            
            # Add error details if present
            if error:
                status_data['error'] = error

            self.supabase.table('rt_match_status').upsert(status_data, on_conflict='show_id').execute()
        except Exception as e:
            logger.error(f"Error updating status: {e}")

async def main():
    """Run collector from command line."""
    # Load environment variables from source directory
    source_dir = os.path.abspath(os.path.join(project_root, '..', 'GoogleDB'))
    env_path = os.path.join(source_dir, '.env')
    load_dotenv(env_path)

    if len(sys.argv) != 2:
        logger.error("Usage: python collector.py <show_id>")
        sys.exit(1)

    try:
        show_id = int(sys.argv[1])
    except ValueError:
        logger.error("Error: show_id must be an integer")
        sys.exit(1)

    collector = RTCollector()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page(viewport={'width': 1280, 'height': 800})
        collector.page = page
        
        try:
            result = await collector.collect_show_data(show_id)
            if result['success']:
                logger.info(f"Success! Scores: {result['scores']}")
                sys.exit(0)
            else:
                logger.error(f"Error: {result['error']}")
                sys.exit(1)
        finally:
            await browser.close()

if __name__ == '__main__':
    print("Starting collector...")
    asyncio.run(main())
