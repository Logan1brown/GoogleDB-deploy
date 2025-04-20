"""Admin Dashboard Page

This page provides access to admin-only functions.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add src to path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.dashboard.auth.auth_required import auth_required
from src.dashboard.auth.auth_state import init_auth_state
from src.shared.auth import get_user_role

# Initialize auth state
init_auth_state()

@auth_required(['admin'])
def main():
    """Main admin page."""
    # Import admin dashboard
    from src.dashboard.pages.admin_dashboard import admin_main
    
    # Run admin dashboard
    admin_main()

if __name__ == "__main__":
    main()
