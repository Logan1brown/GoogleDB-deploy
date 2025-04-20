"""Remove Button Component

Provides a consistent remove button component used across the dashboard.
"""
import streamlit as st

def render_remove_button(
    key: str,
    help_text: str = None,
    on_click = None,
    args: tuple = None,
    kwargs: dict = None
) -> bool:
    """Render a remove button with consistent styling.
    
    Args:
        key: Unique key for the button
        help_text: Optional hover text
        on_click: Optional callback function
        args: Optional positional arguments for callback
        kwargs: Optional keyword arguments for callback
    
    Returns:
        bool: True if button was clicked
    """
    args = args or ()
    kwargs = kwargs or {}
    
    clicked = st.button(
        "✕",
        key=f"remove_{key}",
        help=help_text,
        use_container_width=False
    )
    
    if clicked and on_click:
        on_click(*args, **kwargs)
    
    return clicked
