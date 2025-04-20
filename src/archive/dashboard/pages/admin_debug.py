"""Debug admin client methods"""
import streamlit as st
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
    st.write("Admin client methods:")
    st.write([method for method in dir(client.auth.admin) if not method.startswith('_')])

if __name__ == "__main__":
    main()
