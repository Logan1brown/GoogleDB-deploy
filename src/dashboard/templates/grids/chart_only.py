"""Grid layout for a single chart with title.

This module provides a function to create a simple figure with:
- Single chart taking full width/height
- Optional title
- Configurable margins and spacing
"""

from plotly.subplots import make_subplots


def create_chart_grid(
    title=None,
    chart_title=None,
    margin=None
):
    """Create a figure with a single chart.

    Args:
        title: Optional title for entire figure
        chart_title: Optional title for chart section
        margin: Optional dict with margin settings (l, r, t, b)

    Returns:
        go.Figure: Figure with single chart layout:
        - Full width/height chart
        - Optional titles
        - Configurable margins
    """
    # Create subplot structure
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=[chart_title] if chart_title else None
    )

    # Apply layout settings
    fig.update_layout(
        title=title,
        showlegend=True,
        margin=margin if margin else dict(l=50, r=50, t=80, b=50)
    )

    return fig
