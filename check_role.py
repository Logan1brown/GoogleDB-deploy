from supabase import create_client

# Initialize Supabase client
url = "https://hlwnwcxylueaoemdqiwo.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imhsd253Y3h5bHVlYW9lbWRxaXdvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDQxMjk5NjgsImV4cCI6MjA1OTcwNTk2OH0.CV-osPzTaWMDMmfz9vLQPaPY-0pi5iEtZTnFJAcDRC8"

client = create_client(url, key)

# Check user_roles table
user_id = "5808eecf-b492-4546-9179-7b8660fc644c"
response = client.table('user_roles').select('*').eq('id', user_id).execute()
print(f"User roles: {response.data}")
