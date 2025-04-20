"""One-time script to clean up test users"""
import os
from supabase import create_client

def get_admin_client():
    """Get Supabase client with service role for admin operations."""
    return create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_SERVICE_KEY")
    )

def main():
    client = get_admin_client()
    
    # List of emails to remove
    emails_to_remove = [
        'test@googledb-test.com',
        'fictivebooks@gmail.com'
    ]
    
    # Get all users
    users = client.auth.admin.list_users()
    
    # Show all users
    print("\nCurrent users:")
    for user in users:
        print(f"- {user.email}")
    print()
    
    # Delete specified users
    for user in users:
        if user.email in emails_to_remove:
            print(f"Processing user: {user.email}")
            try:
                # First remove from user_roles
                print(f"  Removing from user_roles...")
                client.table('user_roles').delete().eq('id', user.id).execute()
                
                # Then delete from auth
                print(f"  Removing from auth...")
                client.auth.admin.delete_user(user.id)
                
                print(f"✓ Successfully deleted {user.email}")
            except Exception as e:
                print(f"✗ Error deleting {user.email}: {str(e)}")

if __name__ == "__main__":
    main()
