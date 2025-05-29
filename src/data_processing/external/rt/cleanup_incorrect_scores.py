"""Clean up incorrectly marked scores between Avalon and Bosch: Legacy."""
import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

def get_client():
    """Get Supabase client with service role."""
    return create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_SERVICE_KEY")
    )

def cleanup_incorrect_scores():
    """Remove incorrect scores between Avalon and Bosch: Legacy."""
    supabase = get_client()
    
    # Get affected show IDs from rt_success_metrics
    response = supabase.table('rt_success_metrics')\
        .select('show_id')\
        .gte('created_at', '2025-05-29 18:12:51.423593+00')\
        .lte('created_at', '2025-05-29 18:24:03.956192+00')\
        .execute()
    
    affected_ids = [row['show_id'] for row in response.data]
    print(f"Found {len(affected_ids)} shows to clean up")
    
    if not affected_ids:
        print("No shows found in the time range")
        return
        
    # Delete from rt_success_metrics
    response = supabase.table('rt_success_metrics')\
        .delete()\
        .gte('created_at', '2025-05-29 18:12:51.423593+00')\
        .lte('created_at', '2025-05-29 18:24:03.956192+00')\
        .execute()
    print(f"Deleted entries from rt_success_metrics")
    
    # Delete from rt_match_status
    response = supabase.table('rt_match_status')\
        .delete()\
        .in_('show_id', affected_ids)\
        .execute()
    print(f"Deleted entries from rt_match_status")
    
    print("\nShows that will need to be re-processed:")
    # Get show titles for reference
    response = supabase.table('shows')\
        .select('id,title')\
        .in_('id', affected_ids)\
        .execute()
    
    for show in response.data:
        print(f"- {show['title']} (ID: {show['id']})")

if __name__ == '__main__':
    cleanup_incorrect_scores()
