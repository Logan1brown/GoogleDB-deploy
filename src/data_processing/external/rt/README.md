# Rotten Tomatoes Data Collector

This module provides automated collection of TV show data from Rotten Tomatoes using Playwright.

## Features

- Automated show searching
- Score extraction (Tomatometer and Audience Score)
- Rate limiting to avoid IP blocks
- Status tracking in database
- Error handling and recovery
- Manual URL override support

## Setup

1. Install dependencies:
```bash
pip install playwright pytest pytest-asyncio
playwright install chromium
```

2. Run tests:
```bash
pytest test_collector.py -v
```

## Usage

```python
async with RTCollector() as collector:
    # Search and collect data for a show
    result = await collector.collect_show_data(show_id=123)
    
    # Or use a manual URL override
    result = await collector.collect_show_data(
        show_id=123,
        manual_url="https://www.rottentomatoes.com/tv/show_name"
    )
```

## Database Integration

The collector uses two tables:

1. `rt_success_metrics`:
   - Stores the actual RT scores
   - Updated when collection succeeds

2. `rt_match_status`:
   - Tracks collection status
   - Records errors and attempts
   - Supports manual URL overrides

## Rate Limiting

The collector implements rate limiting to avoid being blocked:
- Maximum 10 requests per minute
- Automatic retry on errors
- Exponential backoff for failed attempts
