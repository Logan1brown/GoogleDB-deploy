"""Grid layout for chart with two data tables.

This module provides a function to create a figure that combines:
- Chart on the left
- Two stacked data tables on the right
"""

from plotly.subplots import make_subplots


def create_chart_dual_table_grid(
    title=None,
    chart_title=None,
    top_table_title=None,
    bottom_table_title=None,
    chart_width=0.6  # Proportion of width for chart vs tables
):
    """Create a figure with chart and two tables.

    Args:
        title: Optional title for entire figure
        chart_title: Optional title for chart section
        top_table_title: Optional title for top table
        bottom_table_title: Optional title for bottom table
        chart_width: Width proportion for chart section (default 0.6)

    Returns:
        go.Figure: Figure with grid layout:
        - Chart on left side
        - Two tables stacked on right side
        - Configurable proportions
    """
    # Build subplot titles
    subplot_titles = []
    if chart_title:
        subplot_titles.append(chart_title)
    if top_table_title:
        subplot_titles.append(top_table_title)
    if bottom_table_title:
        subplot_titles.append(bottom_table_title)
    
    # Create subplot structure with proper specs
    specs = [
        # First row: Chart (left) and Top Table (right)
        [{'type': 'xy', 'rowspan': 2}, {'type': 'table'}],
        # Second row: (Chart continued), Bottom Table
        [None, {'type': 'table'}]
    ]
    
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[chart_width, 1 - chart_width],
        row_heights=[0.5, 0.5],  # Equal height tables
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
        subplot_titles=subplot_titles if subplot_titles else None,
        specs=specs
    )
    
    # Apply layout settings
    fig.update_layout(
        title=title,
        showlegend=True,
        margin=dict(l=50, r=50, t=100, b=50),
        height=800
    )
    
    # Configure table sections
    fig.update_xaxes(showgrid=False, showticklabels=False, col=2)
    fig.update_yaxes(showgrid=False, showticklabels=False, col=2)
    
    return fig
