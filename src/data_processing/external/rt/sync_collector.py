"""Rotten Tomatoes Data Collector (Sync Version)

This module handles automated collection of show data from Rotten Tomatoes using sync Playwright.
"""

import logging
import os
import sys
import subprocess
from urllib.parse import quote
from datetime import datetime
from typing import Dict, Optional
from playwright.sync_api import sync_playwright, TimeoutError

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from src.dashboard.services.supabase import get_supabase_client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RTCollector:
    """Handles automated collection of show data from Rotten Tomatoes."""
    
    def __init__(self):
        """Initialize the collector."""
        self.supabase = get_supabase_client()
        self.playwright = None
        self.browser = None
        self.page = None
        
    def install_browsers(self):
        """Install Playwright browsers if not already installed."""
        try:
            logger.info("Installing browsers...")
            # First check if playwright is in PATH
            which_result = subprocess.run(['which', 'playwright'], 
                                        capture_output=True, text=True)
            logger.info(f"Which playwright: {which_result.stdout}")
            if which_result.returncode != 0:
                logger.error("playwright not found in PATH")
                # Try installing browsers using python -m
                logger.info("Trying python -m playwright install...")
                result = subprocess.run(['python', '-m', 'playwright', 'install', 'chromium'],
                                      capture_output=True, text=True)
            else:
                # Use playwright directly
                logger.info("Using playwright from PATH...")
                result = subprocess.run(['playwright', 'install', 'chromium'],
                                      capture_output=True, text=True)
            
            logger.info(f"Install stdout: {result.stdout}")
            logger.info(f"Install stderr: {result.stderr}")
            if result.returncode != 0:
                logger.error(f"Error installing browsers: {result.stderr}")
                raise Exception(f"Failed to install browsers: {result.stderr}")
            logger.info("Browsers installed successfully")
        except Exception as e:
            logger.error(f"Error running playwright install: {e}")
            raise

    def __enter__(self):
        """Set up Playwright browser when used as context manager."""
        try:
            # First make sure browsers are installed
            try:
                self.install_browsers()
            except Exception as e:
                logger.error(f"Error installing browsers: {e}")
                # Continue anyway - browsers might already be installed
            
            logger.info("Starting playwright...")
            try:
                self.playwright = sync_playwright().start()
            except Exception as e:
                logger.error(f"Error starting Playwright: {e}")
                raise
                
            logger.info("Launching browser...")
            try:
                self.browser = self.playwright.chromium.launch(headless=True)
            except Exception as e:
                logger.error(f"Error launching browser: {e}")
                if self.playwright:
                    self.playwright.stop()
                raise
                
            logger.info("Creating page...")
            try:
                self.page = self.browser.new_page(viewport={'width': 1280, 'height': 800})
            except Exception as e:
                logger.error(f"Error creating page: {e}")
                if self.browser:
                    self.browser.close()
                if self.playwright:
                    self.playwright.stop()
                raise
                
            logger.info("Setup complete")
            return self
        except Exception as e:
            logger.error(f"Error in __enter__: {e}")
            # Make sure to clean up if we fail
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            raise
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up browser resources."""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
            
    def search_show(self, title: str) -> bool:
        """Search for a show and get its scores."""
        # Go to search with TV filter
        url = f'https://www.rottentomatoes.com/search?search={quote(title)}&type=tv'
        logger.info(f"Searching for: {title}")
        self.page.goto(url)
        
        # Look for show link (exclude season links)
        links = self.page.query_selector_all('a[href*="/tv/"]')
        for link in links:
            href = link.get_attribute('href')
            text = link.text_content()
            
            # Skip season links (ending in /s01, /s02, etc)
            if href and '/s0' in href:
                continue
                
            logger.info(f"Found: {text.strip()} -> {href}")
            
            # Click first matching link that's not a season
            if title.lower() in text.lower():
                logger.info(f"Clicking main show link: {href}")
                link.click()
                # Wait for both score elements to appear
                try:
                    self.page.wait_for_selector('rt-text[slot="criticsScore"]', timeout=5000)
                    self.page.wait_for_selector('rt-text[slot="audienceScore"]', timeout=5000)
                    return True
                except TimeoutError:
                    logger.error("Timeout waiting for score elements")
                    return False
        
        logger.error(f"Could not find {title}")
        return False
            
    def get_scores(self) -> Optional[Dict[str, str]]:
        """Get critic and audience scores."""
        scores = {}
        
        # Get critics score
        critics = self.page.query_selector('rt-text[slot="criticsScore"]')
        if critics:
            scores['critics'] = critics.text_content()
        
        # Get audience score
        audience = self.page.query_selector('rt-text[slot="audienceScore"]')
        if audience:
            scores['audience'] = audience.text_content()
            
        return scores if scores else None
            
    def collect_show_data(self, show_id: int) -> Dict:
        """Collect RT data for a show."""
        logger.info(f"Starting collection for show {show_id}")
        logger.info(f"Browser: {self.browser}, Page: {self.page}")
        try:
            if not self.page:
                error = "Browser not initialized"
                logger.error(error)
                self.update_status(show_id, 'error', error)
                return {'success': False, 'error': error}
                
            # Get show title
            response = self.supabase.table('shows')\
                .select('title')\
                .eq('id', show_id)\
                .execute()
                
            if not response.data:
                error = f"Show with id {show_id} not found"
                self.update_status(show_id, 'error', error)
                return {'success': False, 'error': error}
                
            title = response.data[0]['title']
            logger.info(f"Collecting RT data for: {title}")
            
            # Search and get scores
            if not self.search_show(title):
                self.update_status(show_id, 'not_found')
                return {'success': False}
                
            scores = self.get_scores()
            if not scores:
                self.update_status(show_id, 'error', 'Could not find scores')
                return {'success': False, 'error': 'Could not find scores'}
                
            # Save scores
            metrics_data = {
                'show_id': show_id,
                'tomatometer': scores['critics'].rstrip('%'),
                'popcornmeter': scores['audience'].rstrip('%'),
                'last_updated': datetime.now().isoformat()
            }
            
            self.supabase.table('rt_success_metrics').upsert(metrics_data, on_conflict='show_id').execute()
            self.update_status(show_id, 'matched')
            
            return {'success': True, 'scores': scores}
            
        except Exception as e:
            logger.error(f"Error collecting RT data: {e}")
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
