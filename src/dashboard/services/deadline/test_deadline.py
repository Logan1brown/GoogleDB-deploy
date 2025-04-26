"""Test the Deadline client."""

import logging
from deadline_client import DeadlineClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_search():
    """Test searching Deadline."""
    client = DeadlineClient()
    articles = client.search_straight_to_series()
    
    logger.info(f"Found {len(articles)} articles")
    
    # Print articles
    for article in articles:
        logger.info("---")
        logger.info(f"Title: {article['title']}")
        logger.info(f"Date: {article['published_date']}")

if __name__ == "__main__":
    test_search()
