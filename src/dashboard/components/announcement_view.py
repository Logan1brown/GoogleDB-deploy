"""Component for displaying announcement information."""

import streamlit as st
from typing import Dict, Any, List
from ..utils.style_config import COLORS, FONTS

def render_announcement_card(announcement: Dict[str, Any], on_review: callable = None) -> None:
    """Render an announcement card with consistent styling and review button.
    
    Args:
        announcement: Dictionary containing announcement data (title, url, published_date)
        on_review: Callback when review button is clicked
    """
    title_color = COLORS['text']['primary']
    date_color = COLORS['text']['secondary']
    
    # Create columns for content and button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Add title with link
        st.markdown(
            f"<div style='font-family: {FONTS['primary']['family']};'>"
            f"<h4 style='margin: 0; color: {title_color}; font-size: {FONTS['primary']['sizes']['header']}px;'>"
            f"<a href='{announcement['url']}' target='_blank' "
            f"style='color: {title_color}; text-decoration: none; transition: all 0.2s ease;' "
            f"onmouseover=\"this.style.textDecoration='underline'; this.style.opacity='0.8';\" "
            f"onmouseout=\"this.style.textDecoration='none'; this.style.opacity='1.0';\">"
            f"{announcement['title']}</a>"
            f"</h4>"
            f"<p style='margin: 4px 0 0 0; color: {date_color}; font-size: {FONTS['primary']['sizes']['small']}px;'>"
            f"Published: {announcement['published_date']}"
            f"</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    # Add review button if not reviewed
    with col2:
        if not announcement['reviewed'] and on_review:
            if st.button("Mark Reviewed", key=f"review_{announcement['id']}", type="primary", use_container_width=True):
                on_review(announcement)

def render_announcements_list(announcements: List[Dict[str, Any]], on_review: callable = None) -> None:
    """Render a list of announcements.
    
    Args:
        announcements: List of announcement dictionaries
        on_review: Callback when review button is clicked
    """
    # Add container for consistent card styling
    st.markdown(
        "<style>"
        "div[data-testid='stHorizontalBlock'] { background: white; padding: 0.75em; border-radius: 4px; margin-bottom: 0.5em; }"
        "</style>",
        unsafe_allow_html=True
    )
    
    for announcement in announcements:
        render_announcement_card(announcement, on_review)
        st.write("")  # Add spacing between announcements
