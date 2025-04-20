"""Base template with common styles from style_config.

IMPORTANT: Templates are for STYLING only (colors, fonts, markers, etc).
For layout structure (rows, columns, subplots), use the grid layout functions in grids/.

This module provides the base template that all other templates build upon.
It reads from style_config.py to ensure consistent branding across all charts.

Architecture:
1. Templates (defaults/): Visual styling via go.layout.Template
   - Base template (this file)
   - Chart-specific templates (bar.py, scatter.py, etc)
   
2. Grids (grids/): Layout structure via make_subplots()
   - Side-by-side (dual.py)
   - Vertical stack (stacked.py)
   - Chart with table (with_table.py)

Usage:
    from src.dashboard.templates.base import create_base_template
    
    fig = go.Figure(template=create_base_template())
"""

import plotly.graph_objects as go
from src.dashboard.utils.style_config import COLORS, FONTS, CHART_DEFAULTS, DIMENSIONS

def create_base_template():
    """Create base template with common styles.
    
    Returns:
        go.layout.Template: Base template with:
        - Font family and sizes
        - Color scheme
        - Margins and spacing
        - Legend position
        - Hover defaults
        - Download button
    """
    template = go.layout.Template()
    
    # Basic layout
    template.layout = dict(
        # Font settings
        font=dict(
            family=FONTS['primary']['family'],
            size=FONTS['primary']['sizes']['body'],
            color=COLORS['text']['primary']
        ),
        
        # Colors
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['background'],
        
        # Margins (use dashboard defaults)
        margin=DIMENSIONS['dashboard']['margin'],
        height=DIMENSIONS['dashboard']['height'],
        
        # Legend
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        
        # Hover
        hovermode='closest',
        hoverlabel=dict(
            font=dict(
                family=FONTS['primary']['family'],
                size=FONTS['primary']['sizes']['small']
            ),
            bgcolor='white',
            bordercolor=COLORS['accent']
        ),
        
        # Modebar (download button)
        modebar=dict(
            bgcolor='rgba(255, 255, 255, 0.8)',
            activecolor=COLORS['accent'],
            color=COLORS['text']['secondary']
        ),
        
        # Axis defaults
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False,
            showline=True,
            linecolor='rgba(0,0,0,0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False,
            showline=True,
            linecolor='rgba(0,0,0,0.2)'
        )
    )
    
    return template
