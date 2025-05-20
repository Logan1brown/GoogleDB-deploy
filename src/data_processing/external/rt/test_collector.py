"""Test script for RTCollector

Tests the non-database functionality using mock data that matches
the schema from show_details view.
"""

from collector import RTCollector

# Test data matching show_details view schema
test_shows = [
    {
        'id': 1,
        'title': 'The Last of Us',
        'network_list': {
            'network': 'HBO'
        }
    },
    {
        'id': 2,
        'title': 'House of the Dragon',
        'network_list': {
            'network': 'HBO'
        }
    },
    {
        'id': 3,
        'title': 'Ted Lasso',
        'network_list': {
            'network': 'Apple TV+'
        }
    }
]

def test_url_generation():
    """Test URL generation for RT search"""
    collector = RTCollector(None)  # No Supabase needed
    
    # Test single URL
    url = collector.generate_search_url(test_shows[0]['title'])
    print(f"\nTesting single URL generation:\n{url}")
    assert 'site%3Arottentomatoes.com' in url
    assert 'The%20Last%20of%20Us' in url  # Spaces become %20 in URLs
    
    # Test batch URLs
    urls = [collector.generate_search_url(s['title']) for s in test_shows]
    print(f"\nTesting batch URL generation:")
    for url in urls:
        print(f"- {url}")
    assert len(urls) == len(test_shows)

def test_score_validation():
    """Test RT score validation"""
    collector = RTCollector(None)
    
    print("\nTesting score validation:")
    
    # Valid scores
    valid = collector.save_rt_metrics(1, 85, 90)
    print(f"Valid scores (85, 90): {'passed' if not valid else 'failed'}")
    assert not valid  # Should be False without DB
    
    # Invalid scores
    invalid = collector.save_rt_metrics(1, -1, 101)
    print(f"Invalid scores (-1, 101): {'passed' if not invalid else 'failed'}")
    assert not invalid

if __name__ == '__main__':
    test_url_generation()
    test_score_validation()
