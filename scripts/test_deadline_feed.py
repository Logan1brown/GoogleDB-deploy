"""Test script for fetching straight-to-series announcements from Deadline."""

import feedparser
from datetime import datetime

def check_straight_to_series(text: str) -> bool:
    """Check if text contains straight-to-series variations."""
    variations = [
        "straight-to-series",
        "straight to series", 
        "straight to series order",
        "straight-to-series order"
    ]
    return any(var.lower() in text.lower() for var in variations)

def main():
    """Main function to test Deadline RSS feed."""
    print("Fetching Deadline RSS feed...")
    feed = feedparser.parse('https://deadline.com/feed/')
    
    print(f"\nFound {len(feed.entries)} total entries")
    print("\nSearching for straight-to-series announcements...")
    
    for entry in feed.entries:
        # Search title and description
        if (check_straight_to_series(entry.title) or 
            check_straight_to_series(entry.description)):
            
            print("\n---")
            print(f"Title: {entry.title}")
            print(f"Date: {entry.published}")
            print(f"Link: {entry.link}")
            print("\nDescription:")
            print(entry.description[:200] + "..." if len(entry.description) > 200 else entry.description)
            
            # Print categories if available
            if hasattr(entry, 'tags'):
                print("\nCategories:", ", ".join(tag.term for tag in entry.tags))

if __name__ == "__main__":
    main()
