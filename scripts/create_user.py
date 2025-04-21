import os
from supabase import create_client, Client

# Get Supabase credentials from environment
url = os.environ.get("SUPABASE_URL")
service_key = os.environ.get("SUPABASE_SERVICE_KEY")

# Initialize Supabase client with service key
supabase: Client = create_client(url, service_key)

def create_user_with_password(email: str, password: str, role: str = "viewer"):
    """Create a new user with a specified password and role."""
    try:
        # Create user with password
        response = supabase.auth.admin.create_user({
            'email': email,
            'password': password,
            'email_confirm': True  # Skip email verification
        })
        user = response.user
        
        # Set role in user_roles table
        supabase.table('user_roles').insert({
            'id': user.id,
            'role': role,
            'created_by': user.id,  # Use the new user's ID
            'updated_by': user.id
        }).execute()
        
        print(f"Successfully created user {email} with role {role}")
        return True
    except Exception as e:
        print(f"Error creating user: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python create_user.py <email> <password> [role]")
        sys.exit(1)
    
    email = sys.argv[1]
    password = sys.argv[2]
    role = sys.argv[3] if len(sys.argv) > 3 else "viewer"
    
    create_user_with_password(email, password, role)
