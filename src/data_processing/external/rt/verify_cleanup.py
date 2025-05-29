"""Verify cleanup of incorrect scores."""
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

def verify_cleanup():
    """Verify that affected shows were properly cleaned up."""
    supabase = get_client()
    
    # List of affected show IDs
    affected_ids = [3746, 3749, 3752, 3753, 3755, 3756, 3757, 3758, 3760, 3761, 3762, 11671, 11686, 11687]
    
    print("Checking rt_success_metrics...")
    response = supabase.table('rt_success_metrics')\
        .select('show_id,created_at')\
        .in_('show_id', affected_ids)\
        .gte('created_at', '2025-05-29 18:12:51.423593+00')\
        .lte('created_at', '2025-05-29 18:24:03.956192+00')\
        .execute()
    
    if response.data:
        print("❌ Found entries in rt_success_metrics that should have been deleted:")
        for entry in response.data:
            print(f"  - Show ID: {entry['show_id']}, Created: {entry['created_at']}")
    else:
        print("✅ No entries found in rt_success_metrics")
        
    print("\nChecking rt_match_status...")
    response = supabase.table('rt_match_status')\
        .select('show_id,status')\
        .in_('show_id', affected_ids)\
        .execute()
        
    if response.data:
        print("❌ Found entries in rt_match_status that should have been deleted:")
        for entry in response.data:
            print(f"  - Show ID: {entry['show_id']}, Status: {entry['status']}")
    else:
        print("✅ No entries found in rt_match_status")
        
    # Verify shows appear in unmatched list
    print("\nVerifying shows appear as unmatched...")
    response = supabase.table('shows')\
        .select('id,title')\
        .in_('id', affected_ids)\
        .execute()
        
    show_titles = {show['id']: show['title'] for show in response.data}
    
    # Get all matched show IDs
    matched = supabase.table('rt_match_status')\
        .select('show_id')\
        .execute()
    matched_ids = [row['show_id'] for row in matched.data]
    
    # Check each affected show
    for show_id in affected_ids:
        if show_id in matched_ids:
            print(f"❌ {show_titles[show_id]} (ID: {show_id}) is still marked as matched")
        else:
            print(f"✅ {show_titles[show_id]} (ID: {show_id}) correctly appears as unmatched")

if __name__ == '__main__':
    verify_cleanup()
