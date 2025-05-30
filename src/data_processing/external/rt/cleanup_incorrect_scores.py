"""Clean up incorrectly marked RT scores for Rebel."""
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
    """Remove incorrect RT scores for Rebel."""
    supabase = get_client()
    
    # Get show ID for Rebel
    response = supabase.table('shows')\
        .select('id')\
        .eq('title', 'Rebel')\
        .execute()
    
    if not response.data:
        print("Show 'Rebel' not found")
        return
        
    show_id = response.data[0]['id']
    print(f"Found show ID {show_id} for 'Rebel'")
        
    # Delete from rt_success_metrics
    response = supabase.table('rt_success_metrics')\
        .delete()\
        .eq('show_id', show_id)\
        .execute()
    print(f"Deleted entries from rt_success_metrics")
    
    # Delete from rt_match_status
    response = supabase.table('rt_match_status')\
        .delete()\
        .eq('show_id', show_id)\
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
