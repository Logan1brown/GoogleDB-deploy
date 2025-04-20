"""List Item Component

Provides a consistent list item component with optional remove button.
"""
import streamlit as st
from typing import Any, Callable, Optional, Tuple

from .remove_button import render_remove_button

def render_list_item(
    content: str,
    key: str,
    on_remove: Optional[Callable] = None,
    remove_args: Tuple = None,
    suffix: str = "",
    help_text: str = None,
    cols: Tuple[int, int] = (10, 1)
) -> bool:
    """Render a list item with optional remove button.
    
    Args:
        content: The text content to display
        key: Unique key for the item
        on_remove: Optional callback when remove is clicked
        remove_args: Optional arguments for remove callback
        suffix: Optional suffix to append to content (e.g. "(New)")
        help_text: Optional hover text for remove button
        cols: Column ratio for content vs remove button
    
    Returns:
        bool: True if remove was clicked
    """
    with st.container():
        cols = st.columns(cols)
        cols[0].write(f"- {content}{suffix}")
        if on_remove:
            return render_remove_button(
                key=key,
                help_text=help_text or f"Remove {content}",
                on_click=on_remove,
                args=remove_args
            )
