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
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up browser when context manager exits."""
        await self.cleanup()
        
    async def setup(self):
        """Initialize Playwright browser and page."""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)
        self.page = await self.browser.new_page()
        
    async def cleanup(self):
        """Clean up Playwright resources."""
        if self.page:
            await self.page.close()
        if self.browser:
            await self.browser.close()
            
    async def collect_show_data(self, show_id: int, manual_url: Optional[str] = None) -> Dict:
        """Collect show data from Rotten Tomatoes.
        
        Args:
            show_id: ID of the show in our database
            manual_url: Optional manual URL override for the RT page
            
        Returns:
            Dictionary with collected data and status
        """
        try:
            # Check if show already has RT data
            metrics = await self.supabase.table('rt_success_metrics')\
                .select('tomatometer,popcornmeter,updated_at')\
                .eq('show_id', show_id)\
                .eq('is_matched', True)\
                .execute()
                
            if metrics.data:
                logger.info(f"Show {show_id} already has RT data from {metrics.data[0]['updated_at']}")
                return {
                    'success': True,
                    'scores': {
                        'tomatometer': metrics.data[0]['tomatometer'],
                        'popcornmeter': metrics.data[0]['popcornmeter']
                    },
                    'cached': True
                }
            
            # Check if we've already tried and failed recently
            status = await self.supabase.table('rt_match_status')\
                .select('status,attempts,last_attempt,error_details')\
                .eq('show_id', show_id)\
                .execute()
                
            if status.data:
                last_status = status.data[0]
                if last_status['status'] == 'error' and last_status['attempts'] >= 3:
                    return {
                        'success': False,
                        'error': f"Already failed {last_status['attempts']} times. Last error: {last_status['error_details']}",
                        'last_attempt': last_status['last_attempt']
                    }
            
            # Get show title from database
            result = await self.supabase.table('shows').select('title').eq('id', show_id).execute()
            if not result.data:
                error = f"Show {show_id} not found in database"
                await self.update_status(show_id, 'error', error)
                return {'success': False, 'error': error}
                
            title = result.data[0]['title']
            
            # Use manual URL or search for show
            url = manual_url
            if not url:
                url = await self.search_show(title)
                if not url:
                    error = f"Could not find RT page for {title}"
                    await self.update_status(show_id, 'not_found', error)
                    return {'success': False, 'error': error}
            
            # Extract scores
            scores = await self.extract_scores(url)
            if not scores:
                error = f"Could not extract scores from {url}"
                await self.update_status(show_id, 'error', error)
                return {'success': False, 'error': error}
                
            # Update rt_success_metrics
            metrics_data = {
                'show_id': show_id,
                'tomatometer': scores.get('tomatometer'),
                'popcornmeter': scores.get('popcornmeter'),
                'is_matched': True,
                'updated_at': datetime.now().isoformat()
            }
            
            # Upsert metrics
            await self.supabase.table('rt_success_metrics')\
                .upsert(metrics_data, on_conflict='show_id')\
                .execute()
                
            # Update match status
            await self.update_status(show_id, 'matched')
            
            return {
                'success': True,
                'url': url,
                'scores': scores
            }
            
        except Exception as e:
            error = str(e)
            self.update_status(show_id, 'error', error)
            return {'success': False, 'error': error}
        
    async def search_show(self, title: str) -> Optional[str]:
        """Search for a show on RT and return its URL if found.
        
        Args:
            title: Show title to search for
            
        Returns:
            RT URL if found, None if not found
        """
        # Encode title for URL
        url = f'https://www.rottentomatoes.com/search?search={quote(title)}&type=tv'
        logger.info(f"Searching for: {title}")
        
        try:
            await self.page.goto(url)
            
            # Look for show links (exclude season links)
            links = await self.page.query_selector_all('a[href*="/tv/"]')
            for link in links:
                href = await link.get_attribute('href')
                text = await link.text_content()
                
                # Skip season links (ending in /s01, /s02, etc)
                if href and '/s0' in href:
                    continue
                    
                logger.info(f"Found: {text.strip()} -> {href}")
                
                # Return first matching link that's not a season
                if title.lower() in text.lower():
                    logger.info(f"Found main show link: {href}")
                    return href
                    
            logger.warning(f"Could not find {title}")
            return None
            
        except Exception as e:
            logger.error(f"Error searching for {title}: {str(e)}")
            return None
        
    async def extract_scores(self, url: str) -> Dict[str, int]:
        """Extract Tomatometer and audience scores from an RT page.
        
        Args:
            url: RT page URL
            
        Returns:
            Dictionary with tomatometer and popcornmeter scores
        """
        scores = {}
        
        try:
            # Navigate to page and wait for score elements
            await self.page.goto(url)
            
            try:
                # Wait for both score elements with timeout
                await self.page.wait_for_selector('rt-text[slot="criticsScore"]', timeout=5000)
                await self.page.wait_for_selector('rt-text[slot="audienceScore"]', timeout=5000)
                
                # Get critics score
                critics = await self.page.query_selector('rt-text[slot="criticsScore"]')
                if critics:
                    text = await critics.text_content()
                    # Convert '94%' to 94
                    scores['tomatometer'] = int(text.strip('%'))
                
                # Get audience score
                audience = await self.page.query_selector('rt-text[slot="audienceScore"]')
                if audience:
                    text = await audience.text_content()
                    scores['popcornmeter'] = int(text.strip('%'))
                    
                return scores
                
            except TimeoutError:
                logger.error(f"Timeout waiting for score elements on {url}")
                return scores
                
        except Exception as e:
            logger.error(f"Error extracting scores from {url}: {str(e)}")
            return scores
        
    async def update_status(self, show_id: int, status: str, error: Optional[str] = None):
        """Update the status of a show in rt_match_status.
        
        Args:
            show_id: ID of the show
            status: New status (pending, matched, not_found, error)
            error: Optional error message
        """
        try:
            # Get current status if it exists
            result = await self.supabase.table('rt_match_status')\
                .select('attempts')\
                .eq('show_id', show_id)\
                .execute()
            
            # Calculate attempts
            attempts = 1
            if result.data:
                attempts = result.data[0]['attempts'] + 1
            
            # Prepare status data
            status_data = {
                'show_id': show_id,
                'status': status,
                'attempts': attempts,
                'last_attempt': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Add error details if present
            if error:
                status_data['error_details'] = error
            
            # Upsert status
            await self.supabase.table('rt_match_status')\
                .upsert(status_data, on_conflict='show_id')\
                .execute()
                
        except Exception as e:
            logger.error(f"Error updating status for show {show_id}: {str(e)}")
