"""Style configuration for the dashboard."""

COLORS = {
    'text': {
        'primary': 'rgb(49, 51, 63)',
        'secondary': 'rgb(120, 120, 120)'
    },
    'background': '#FFFFFF',
    'accent': 'rgb(55, 83, 109)',
    'success': {
        'high': 'rgb(72, 17, 121)',    # Dark purple (Viridis)
        'medium': 'rgb(26, 152, 80)',   # Green (Viridis)
        'low': 'rgb(253, 231, 37)',     # Yellow (Viridis)
        'none': 'rgb(220, 220, 220)'    # Grey
    }
}

FONTS = {
    'primary': {
        'family': 'Source Sans Pro',
        'sizes': {
            'title': 20,
            'header': 16,
            'body': 14,
            'small': 12
        }
    }
}

CHART_DEFAULTS = {
    'margin': {
        'plot': dict(t=30, b=20),
        'section': dict(t=20, b=20)
    },
    'colorscales': {
        'primary': 'Viridis',
        'secondary': 'Plasma'
    }
}

DIMENSIONS = {
    'dashboard': {
        'width': None,  # Use container width
        'height': 400,
        'margin': CHART_DEFAULTS['margin']['plot']
    },
    'standalone': {
        'width': 800,
        'height': 500,
        'margin': CHART_DEFAULTS['margin']['section']
    }
}

# UI Utility Functions
import streamlit as st

def render_metric_card(title, value, subtitle=None):
    """Render a metric card with a title, value, and optional subtitle.
    
    Args:
        title: Title of the metric
        value: Value to display
        subtitle: Optional subtitle
    """
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; border: 1px solid #eee; background-color: white;">
            <h4 style="margin: 0; color: {COLORS['text']['secondary']}; font-size: 14px;">{title}</h4>
            <div style="font-size: 28px; font-weight: bold; color: {COLORS['text']['primary']};">{value}</div>
            {f'<div style="font-size: 12px; color: {COLORS["text"]["secondary"]};">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_info_card(title, content, color=None):
    """Render an info card with a title and content.
    
    Args:
        title: Title of the card
        content: Content to display
        color: Optional custom color for the title (defaults to accent color)
    """
    title_color = color if color else COLORS['accent']
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; border: 1px solid #eee; background-color: white; margin-bottom: 10px;">
            <h4 style="margin: 0; color: {title_color}; font-size: 16px;">{title}</h4>
            <div style="font-size: 14px; color: {COLORS['text']['primary']}; margin-top: 5px;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_warning(title, content):
    """Render a warning card with a title and content.
    
    Args:
        title: Title of the card
        content: Content to display
    """
    st.markdown(
        f"""
        <div style="padding: 10px; border-radius: 5px; border: 1px solid #f77; background-color: #fff8f8; margin-bottom: 10px;">
            <h4 style="margin: 0; color: #d33; font-size: 16px;">{title}</h4>
            <div style="font-size: 14px; color: {COLORS['text']['primary']}; margin-top: 5px;">{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
