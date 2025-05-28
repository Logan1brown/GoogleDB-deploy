"""Basic test to verify Playwright can access Rotten Tomatoes."""

import asyncio
import logging
from urllib.parse import quote
from playwright.async_api import async_playwright, TimeoutError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def search_show(page, show_name: str):
    """Search for a show and get its scores."""
    # Go to search with TV filter
    url = f'https://www.rottentomatoes.com/search?search={quote(show_name)}&type=tv'
    logger.info(f"Searching for: {show_name}")
    await page.goto(url)
    
    # Look for show link (exclude season links)
    links = await page.query_selector_all('a[href*="/tv/"]')
    for link in links:
        href = await link.get_attribute('href')
        text = await link.text_content()
        
        # Skip season links (ending in /s01, /s02, etc)
        if href and '/s0' in href:
            continue
            
        logger.info(f"Found: {text.strip()} -> {href}")
        
        # Click first matching link that's not a season
        if show_name.lower() in text.lower():
            logger.info(f"Clicking main show link: {href}")
            await link.click()
            # Wait for both score elements to appear
            try:
                await page.wait_for_selector('rt-text[slot="criticsScore"]', timeout=5000)
                await page.wait_for_selector('rt-text[slot="audienceScore"]', timeout=5000)
                return True
            except TimeoutError:
                logger.error("Timeout waiting for score elements")
                return False
    
    logger.error(f"Could not find {show_name}")
    return False

async def get_scores(page):
    """Get critic and audience scores."""
    scores = {}
    
    # Get critics score
    critics = await page.query_selector('rt-text[slot="criticsScore"]')
    if critics:
        scores['critics'] = await critics.text_content()
    
    # Get audience score
    audience = await page.query_selector('rt-text[slot="audienceScore"]')
    if audience:
        scores['audience'] = await audience.text_content()
    
    return scores

async def main():
    """Test RT scraping."""
    show_name = "The Last of Us"
    
    async with async_playwright() as p:
        # Launch browser with ad blocking
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page(viewport={'width': 1280, 'height': 800})
        
        try:
            # Search and click show
            if await search_show(page, show_name):
                # Get scores
                scores = await get_scores(page)
                logger.info(f"Scores for {show_name}:")
                logger.info(f"Critics: {scores.get('critics', 'Not found')}")
                logger.info(f"Audience: {scores.get('audience', 'Not found')}")
            
            # Wait briefly to see results
            await asyncio.sleep(3)
            
        finally:
            await browser.close()

if __name__ == '__main__':
    asyncio.run(main())
