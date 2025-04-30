"""Grid layout for form with results.

This module provides a function to create a figure that combines:
- Form panel on left (criteria, filters, etc)
- Results section on right (table + details)
"""

from plotly.subplots import make_subplots


def create_form_results_grid(
    title=None,
    form_title=None,
    results_title=None,
    form_width=0.3  # Proportion of width for form
):
    """Create a figure with form and results sections.

    Args:
        title: Optional title for entire figure
        form_title: Optional title for form section
        results_title: Optional title for results section
        form_width: Width proportion for form section (default 0.3)

    Returns:
        go.Figure: Figure with grid layout:
        - Form panel on left (criteria, filters)
        - Results section on right (table + details)
        - Configurable proportions
    """
    # Calculate column widths
    col_widths = [form_width, 1 - form_width]
    
    # Build subplot titles
    subplot_titles = []
    if form_title:
        subplot_titles.append(form_title)
    if results_title:
        subplot_titles.append(results_title)
    
    # Create subplot structure
    specs = [
        [{'type': 'domain'}, {'type': 'table'}]  # Form | Results
    ]
    
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=col_widths,
        horizontal_spacing=0.05,
        subplot_titles=subplot_titles if subplot_titles else None,
        specs=specs
    )
    
    # Apply layout settings
    fig.update_layout(
        title=title,
        showlegend=False,
        margin=dict(l=20, r=20, t=100, b=20),
        height=800  # Fixed height to prevent squishing
    )
    
    return fig
