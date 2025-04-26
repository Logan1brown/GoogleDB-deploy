"""Script for finding straight-to-series announcements from Deadline."""

from datetime import datetime
import json
import requests
from typing import List, Optional

class DeadlineArticle:
    """Represents a straight-to-series announcement from Deadline."""
    def __init__(self, title: str, url: str, date: str):
        self.title = title
        self.url = url
        self.date = date
    
    def __str__(self) -> str:
        return f"{self.date}: {self.title}\n{self.url}\n"

def get_straight_to_series_articles(page: int = 1, per_page: int = 100) -> List[DeadlineArticle]:
    """Get articles with 'straight to series' in the title."""
    url = f"https://deadline.com/wp-json/wp/v2/search"
    params = {
        "page": page,
        "per_page": per_page,
        "search": "straight to series"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        articles = []
        posts = response.json()
        print(f"Got {len(posts)} posts")
        
        for post in posts:
            title = post['title']
            # Convert HTML entities
            import html
            title = html.unescape(title).lower()
            print(f"Checking title: {title}")
            
            # Only include articles with 'straight to series' or 'straight-to-series' in title
            if 'straight to series' in title or 'straight-to-series' in title:
                print(f"Found match: {title}")
                article = DeadlineArticle(
                    title=post['title'],
                    url=post['url'],
                    date=post.get('date', '')  # Search endpoint doesn't include date
                )
                articles.append(article)
        
        return articles
        
    except Exception as e:
        print(f"Error fetching page {page}: {str(e)}")
        return []

def main():
    """Find all straight-to-series announcements."""
    all_articles = []
    page = 1
    per_page = 100
    
    while True:
        print(f"Fetching page {page}...")
        articles = get_straight_to_series_articles(page, per_page)
        
        if not articles:
            break
            
        all_articles.extend(articles)
        page += 1
    
    print(f"\nFound {len(all_articles)} straight-to-series announcements:")
    for i, article in enumerate(all_articles, 1):
        print(f"\n{i}. {article}")

if __name__ == "__main__":
    main()
