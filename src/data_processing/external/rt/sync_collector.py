"""Rotten Tomatoes Data Collector (Local Version)

This module handles automated collection of show data from Rotten Tomatoes using sync Playwright.
Designed to run locally where system dependencies can be properly installed.
"""

import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from urllib.parse import quote
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from playwright.sync_api import sync_playwright
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from supabase import create_client

def get_client():
    """Get Supabase client with service role."""
    return create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_SERVICE_KEY")
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.insert(0, project_root)

class RTCollector:
    """Handles automated collection of show data from Rotten Tomatoes."""

    def __init__(self):
        """Initialize the collector."""
        self.supabase = get_client()
        self.playwright = None
        self.browser = None
        self.page = None
        
    def cleanup(self):
        """Clean up browser resources and Node processes."""
        try:
            # Clean up Playwright resources
            if self.page:
                try:
                    self.page.close()
                except:
                    pass
                self.page = None
            if self.browser:
                try:
                    self.browser.close()
                except:
                    pass
                self.browser = None
            if self.playwright:
                try:
                    self.playwright.stop()
                except:
                    pass
                self.playwright = None

            # Clean up any lingering Node processes
            try:
                if sys.platform == 'darwin':  # macOS
                    subprocess.run(['pkill', '-f', 'node.*playwright'], capture_output=True)
                elif sys.platform == 'linux':
                    subprocess.run(['pkill', '-f', 'node.*playwright'], capture_output=True)
                elif sys.platform == 'win32':
                    subprocess.run(['taskkill', '/F', '/IM', 'node.exe'], capture_output=True)
            except:
                pass  # Ignore errors from process cleanup

        except:
            # If any cleanup fails, just null out the references
            self.page = None
            self.browser = None
            self.playwright = None

    def ensure_browser(self):
        """Initialize browser if not already initialized."""
        if not self.page:
            try:
                # Clean up any existing resources first
                self.cleanup()
                
                self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.launch(
                    headless=True,  # Run headless for better memory usage
                    timeout=30000  # 30 second timeout
                )
                self.page = self.browser.new_page()
                self.page.set_default_timeout(30000)  # 30 second timeout
            except Exception as e:
                self.cleanup()  # Clean up on error
                print(f"Error initializing browser: {str(e)}")
                raise

    def install_browsers(self):
        """Install Playwright browsers if not already installed."""
        try:
            # Try installing browsers using python -m playwright
            logger.info("Installing Playwright browsers...")
            result = subprocess.run(
                ['python', '-m', 'playwright', 'install', 'chromium'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Browsers installed successfully")
        except Exception as e:
            logger.error(f"Error installing browsers: {e}")
            raise

    def __enter__(self):
        """Set up Playwright browser when used as context manager."""
        try:
            logger.info("Starting browser setup...")
            logger.info("Starting playwright...")
            try:
                self.playwright = sync_playwright().start()
                logger.info("Playwright started successfully")
            except Exception as e:
                logger.error(f"Error starting Playwright: {e}")
                logger.info("Trying to import directly...")
                from playwright.sync_api import sync_playwright
                self.playwright = sync_playwright().start()
                logger.info("Direct import worked")

            logger.info("Launching browser...")
            try:
                self.browser = self.playwright.chromium.launch(headless=True)
            except Exception as e:
                logger.error(f"Error launching browser: {e}")
                if "missing dependencies" in str(e).lower():
                    logger.error("\nThis is likely because the server is missing required system packages.")
                    logger.error("Please contact the system administrator to install the required packages.")
                raise

            self.page = self.browser.new_page(viewport={'width': 1280, 'height': 800})
            return self

        except Exception as e:
            logger.error(f"Error in enter: {e}")
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
        # Clean the title
        cleaned_title = self.clean_title(title)
        
        try:
            # Go to search page with TV filter
            search_url = f'https://www.rottentomatoes.com/search?search={quote(title)}&type=tv'
            logger.info(f"Searching: {search_url}")
            self.page.goto(search_url)
            time.sleep(3)  # Wait longer for page load
            
            # Get all search results
            results = self.page.query_selector_all('search-page-result[type="tvSeries"]')
            if not results:
                logger.info(f"No results found for '{title}'")
                return False
                
            # Get potential matches
            matches = []
            for result in results:
                result_title = result.get_attribute('name')
                cleaned_result = self.clean_title(result_title)
                if self.titles_match(cleaned_title, cleaned_result):
                    matches.append((result, result_title))
                elif cleaned_title in cleaned_result or cleaned_result in cleaned_title:
                    matches.append((result, result_title))
            
            if not matches:
                logger.info(f"No matches found for '{title}' among {len(results)} results")
                return False
                
            # If single exact match, use it
            if len(matches) == 1 and self.titles_match(cleaned_title, self.clean_title(matches[0][1])):
                logger.info(f"Found exact match: '{matches[0][1]}'")
                matches[0][0].click()
                time.sleep(3)  # Wait longer for page load
                return True
                
            # Multiple matches or fuzzy match - need manual confirmation
            logger.info(f"Found {len(matches)} potential matches for '{title}':")
            for i, (_, result_title) in enumerate(matches):
                logger.info(f"{i+1}. {result_title}")
            logger.info("Please review matches and update manually if needed")
            
            # For now, don't auto-select any match
            return False
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return False

    def get_scores(self) -> Dict:
        """Get critic and audience scores."""
        try:
            # Wait longer for score board to load
            time.sleep(5)
            
            # Wait and retry if scores not loaded
            retries = 3
            while retries > 0:
                # Get critic and audience scores
                critic_score = self.page.query_selector('rt-text[slot="criticsScore"]')
                audience_score = self.page.query_selector('rt-text[slot="audienceScore"]')
                
                if critic_score and audience_score:
                    try:
                        tomatometer = int(re.sub(r'[^0-9]', '', critic_score.text_content()))
                        audience = int(re.sub(r'[^0-9]', '', audience_score.text_content()))
                        
                        logger.info(f"Found scores: Tomatometer={tomatometer}, Popcornmeter={audience}")
                        return {
                            'tomatometer': tomatometer,
                            'popcornmeter': audience
                        }
                    except Exception as e:
                        logger.error(f"Error parsing scores: {e}")
                
                time.sleep(2)
                retries -= 1
            
            logger.error("Could not find scores after retries")
            return None
            
        except Exception as e:
            logger.error(f"Error getting scores: {e}")
            return None

    def collect_show_data(self, show_id: int) -> Dict:
        """Collect RT data for a show.

        Args:
            show_id: ID of the show to collect data for

        Returns:
            Dictionary with success status and scores if successful
        """
        # Get show title
        response = self.supabase.table('shows')\
            .select('title')\
            .eq('id', show_id)\
            .execute()

        if not response.data:
            error = f"No show found with ID {show_id}"
            logger.error(error)
            self.update_status(show_id, 'error', error)
            return {'success': False, 'error': error}

        title = response.data[0]['title']
        logger.info(f"Processing show: {title} (ID: {show_id})")

        # Search for show on RT
        if not self.search_show(title):
            self.update_status(show_id, 'not_found')
            return {'success': False, 'error': 'Show not found on Rotten Tomatoes'}

        # Get scores
        scores = self.get_scores()
        if not scores:
            error = 'Could not get scores'
            self.update_status(show_id, 'error', error)
            return {'success': False, 'error': error}

        # Update database
        try:
            self.supabase.table('rt_success_metrics')\
                .upsert({
                    'show_id': show_id,
                    'tomatometer': scores['tomatometer'],
                    'audience_score': scores['audience_score'],
                    'updated_at': datetime.utcnow().isoformat()
                })\
                .execute()
            self.update_status(show_id, 'success')
            return {'success': True, 'scores': scores}
        except Exception as e:
            error = f"Error updating scores in database: {e}"
            self.update_status(show_id, 'error', error)
            return {'success': False, 'error': error}

    def clean_title(self, title: str) -> str:
        """Clean a title for comparison by removing year ranges and common suffixes."""
        import re

        # Remove year ranges like (2019 - 2023)
        title = re.sub(r'\s*\([0-9]{4}\s*-\s*[0-9]{4}\)\s*$', '', title)
        title = re.sub(r'\s*\([0-9]{4}\)\s*$', '', title)

        # Remove common suffixes
        suffixes = [': The Series', ': The Show']
        for suffix in suffixes:
            if title.endswith(suffix):
                title = title[:-len(suffix)]

        return title.strip().lower()
    
    def titles_match(self, title1: str, title2: str) -> bool:
        """Compare two titles, ignoring case and punctuation."""
        # Normalize titles for comparison
        t1 = title1.lower().replace(':', '').replace('-', '').replace('&', 'and')
        t2 = title2.lower().replace(':', '').replace('-', '').replace('&', 'and')

        # Remove articles from start
        articles = ['the ', 'a ', 'an ']
        for article in articles:
            if t1.startswith(article):
                t1 = t1[len(article):]
            if t2.startswith(article):
                t2 = t2[len(article):]

        return t1 == t2

    def update_status(self, show_id: int, status: str, error: Optional[str] = None):
        """Update the status of a show in the database."""
        # Map our status to DB enum
        status_map = {
            'not_found': 'not_found',
            'success': 'success',
            'error': 'error'
        }

        # Create metrics record
        data = {
            'show_id': show_id,
            'status': status_map[status],
            'error': error,
            'created_at': datetime.utcnow().isoformat()
        }

        try:
            self.supabase.table('rt_success_metrics')\
                .insert(data)\
                .execute()
        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def get_last_status(self, show_id: int) -> Optional[Dict]:
        """Get the last status for a show."""
        try:
            response = self.supabase.table('rt_success_metrics')\
                .select('*')\
                .eq('show_id', show_id)\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting last status: {e}")
            return None


def main():
    """Main entry point for the script."""
    import argparse
    parser = argparse.ArgumentParser(description='Collect Rotten Tomatoes data for shows')
    parser.add_argument('--show-id', type=int, help='ID of show to collect data for')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of shows to process in a batch')
    args = parser.parse_args()

    with RTCollector() as collector:
        if args.show_id:
            # Collect data for specific show
            result = collector.collect_show_data(args.show_id)
            if result['success']:
                logger.info(f"Successfully collected data: {result['scores']}")
            else:
                logger.error(f"Failed to collect data: {result['error']}")
        else:
            # Get all shows that need RT scores
            response = collector.supabase.table('shows')\
                .select('id, title')\
                .not_.in_('id', 
                    collector.supabase.table('rt_success_metrics')\
                        .select('show_id')\
                        .execute()\
                        .data
                )\
                .order('title')\
                .execute()

            if not response.data:
                logger.info("No shows need RT scores")
                return

            total_shows = len(response.data)
            logger.info(f"Found {total_shows} shows that need RT scores")
            
            # Process shows in batches
            for i in range(0, total_shows, args.batch_size):
                batch = response.data[i:i + args.batch_size]
                logger.info(f"\nProcessing batch {i//args.batch_size + 1} of {(total_shows-1)//args.batch_size + 1}")
                
                for show in batch:
                    logger.info(f"\nProcessing show: {show['title']} (ID: {show['id']})")
                    result = collector.collect_show_data(show['id'])
                    
                    if result['success']:
                        logger.info(f"Successfully collected data: {result['scores']}")
                    else:
                        logger.error(f"Failed to collect data: {result['error']}")
                    
                    # Wait between shows to avoid rate limiting
                    time.sleep(2)
                
                if i + args.batch_size < total_shows:
                    input("\nPress Enter to continue with next batch...")


if __name__ == '__main__':
    main()
