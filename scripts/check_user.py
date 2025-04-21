import os
from supabase import create_client, Client

# Get Supabase credentials from environment
url = os.environ.get("SUPABASE_URL")
service_key = os.environ.get("SUPABASE_SERVICE_KEY")

# Initialize Supabase client with service key
supabase: Client = create_client(url, service_key)

def check_user(email: str):
    """Check user status in both auth and user_roles table."""
    try:
        # Check auth system
        print("Checking auth system...")
        users = supabase.auth.admin.list_users()
        auth_user = next((u for u in users if u.email == email), None)
        if auth_user:
            print(f"Found in auth system: {auth_user.id}")
            
            # Check if soft deleted
            print(f"Deleted: {auth_user.deleted}")
            print(f"Banned: {auth_user.banned}")
        else:
            print("Not found in auth system")
        
        # Check user_roles table
        print("\nChecking user_roles table...")
        if auth_user:
            result = supabase.table('user_roles').select('*').eq('id', auth_user.id).execute()
            if result.data:
                print(f"Found in user_roles: {result.data}")
            else:
                print("Not found in user_roles table")
    except Exception as e:
        print(f"Error checking user: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python check_user.py <email>")
        sys.exit(1)
    
    email = sys.argv[1]
    check_user(email)
