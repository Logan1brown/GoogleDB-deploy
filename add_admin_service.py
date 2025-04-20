from supabase import create_client

# Initialize Supabase client with service key
url = "https://hlwnwcxylueaoemdqiwo.supabase.co"
service_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imhsd253Y3h5bHVlYW9lbWRxaXdvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDEyOTk2OCwiZXhwIjoyMDU5NzA1OTY4fQ.1RG8_Ey-qVLzEGAGkYFbJnlG8eDDpWmNxHJYBEBpbZE"

client = create_client(url, service_key)

# Add admin role
user_id = "5808eecf-b492-4546-9179-7b8660fc644c"
data = {
    'id': user_id,
    'role': 'admin'
}

response = client.table('user_roles').insert(data).execute()
print(f"Insert response: {response.data}")

# Verify it worked
check = client.table('user_roles').select('*').eq('id', user_id).execute()
print(f"Verification: {check.data}")
