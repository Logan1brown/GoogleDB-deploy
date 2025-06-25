"""Debug helper functions for the Show Optimizer."""

import streamlit as st
import traceback
from typing import Any, Optional

def safe_debug(message: str, data: Optional[Any] = None) -> None:
    """
    Safely write debug messages without relying on local st imports.
    
    Args:
        message: The debug message to display
        data: Optional data to display alongside the message
    """
    try:
        if data is not None:
            st.write(f"DEBUG: {message}", data)
        else:
            st.write(f"DEBUG: {message}")
    except Exception as e:
        # If we can't use st.write, at least print to console
        print(f"DEBUG ERROR: {message} - {str(e)}")
        print(traceback.format_exc())
