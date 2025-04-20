"""Debug auth client methods"""
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
    st.write("Auth methods:")
    st.write([method for method in dir(client.auth) if not method.startswith('_')])
    st.write("\nAuth client type:")
    st.write(type(client.auth))

if __name__ == "__main__":
    main()
