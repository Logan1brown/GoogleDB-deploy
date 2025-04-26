"""Service for managing announcements."""

from datetime import datetime
from typing import List, Optional

from .supabase import get_supabase_client

def add_announcement(url: str, title: str, published_date: datetime) -> dict:
    """Add a new announcement.
    
    Args:
        url: URL of the announcement
        title: Title of the announcement
        published_date: When the announcement was published
        
    Returns:
        The created announcement record
    """
    client = get_supabase_client()
    result = client.table('announcements').insert({
        'url': url,
        'title': title,
        'published_date': published_date.isoformat(),
        'reviewed': False
    }).execute()
    
    return result.data[0]

def get_announcements(reviewed: Optional[bool] = None) -> List[dict]:
    """Get announcements, optionally filtered by review status.
    
    Args:
        reviewed: If provided, filter by reviewed status
        
    Returns:
        List of announcement records
    """
    client = get_supabase_client()
    query = client.table('announcements').select('*')
    
    if reviewed is not None:
        query = query.eq('reviewed', reviewed)
        
    result = query.order('published_date', desc=True).execute()
    return result.data

def mark_reviewed(announcement_id: str) -> dict:
    """Mark an announcement as reviewed.
    
    Args:
        announcement_id: ID of the announcement to mark
        
    Returns:
        The updated announcement record
    """
    client = get_supabase_client()
    result = client.table('announcements').update({
        'reviewed': True,
        'reviewed_at': datetime.now().isoformat()
    }).eq('id', announcement_id).execute()
    
    return result.data[0]
