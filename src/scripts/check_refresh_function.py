"""Check if refresh_materialized_views function exists in database."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.supabase_client import get_client

def check_refresh_function():
    client = get_client(use_service_key=True)
    if not client:
        print("Failed to connect to Supabase")
        return
        
    # Query to check if function exists
    result = client.rpc('refresh_materialized_views').execute()
    print("Function exists and returned:", result)

if __name__ == "__main__":
    check_refresh_function()
