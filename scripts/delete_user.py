import os
from supabase import create_client, Client

# Get Supabase credentials from environment
url = os.environ.get("SUPABASE_URL")
service_key = os.environ.get("SUPABASE_SERVICE_KEY")

# Initialize Supabase client with service key
supabase: Client = create_client(url, service_key)

def delete_user(email: str):
    """Completely delete a user from both auth and user_roles."""
    try:
        # Find user in auth system
        users = supabase.auth.admin.list_users()
        auth_user = next((u for u in users if u.email == email), None)
        
        if auth_user:
            user_id = auth_user.id
            print(f"Found user {email} with ID {user_id}")
            
            # Delete from auth system
            supabase.auth.admin.delete_user(user_id)
            print("Deleted from auth system")
            
            # Delete from user_roles (if exists)
            supabase.table('user_roles').delete().eq('id', user_id).execute()
            print("Deleted from user_roles table")
            
            return True
        else:
            print(f"User {email} not found")
            return False
            
    except Exception as e:
        print(f"Error deleting user: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python delete_user.py <email>")
        sys.exit(1)
    
    email = sys.argv[1]
    delete_user(email)
