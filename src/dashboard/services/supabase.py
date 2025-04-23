"""Supabase client utilities."""

import os
from supabase import create_client

def get_supabase_client():
    """Get Supabase client with service role for admin operations."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise ValueError("Missing Supabase credentials")
    return create_client(url, key)
