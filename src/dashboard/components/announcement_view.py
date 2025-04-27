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
    
    # Add consistent card styling
    st.markdown(
        f"<div style='padding: 1em; border-radius: 4px; background: white; margin-bottom: 1em;'>"
        f"<div style='display: flex; justify-content: space-between; align-items: flex-start;'>"
        f"<div style='flex: 4; font-family: {FONTS['primary']['family']};'>"
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
        f"</div>"
        f"<div style='flex: 1; text-align: right;'>",
        unsafe_allow_html=True
    )
    
    # Add review button if not reviewed
    if not announcement['reviewed'] and on_review:
        if st.button("Mark Reviewed", key=f"review_{announcement['id']}", type="primary"):
            on_review(announcement)
    
    # Close the flex container and card
    st.markdown("</div></div>", unsafe_allow_html=True)

def render_announcements_list(announcements: List[Dict[str, Any]], on_review: callable = None) -> None:
    """Render a list of announcements.
    
    Args:
        announcements: List of announcement dictionaries
        on_review: Callback when review button is clicked
    """
    for ann in announcements:
        render_announcement_card(ann, on_review)
        st.write("")  # Add spacing between announcements
