"""
Plotly templates for the dashboard.

This package provides three types of templates:
1. Base template (base.py) - Common styles from style_config
2. Chart defaults (defaults/) - Type-specific styling
3. Grid layouts (grids/) - Common arrangements

Usage:
    from src.dashboard.templates.defaults import create_bar_defaults
    from src.dashboard.templates.grids import create_with_table_grid
    
    # Create figure with bar defaults
    fig = go.Figure(template=create_bar_defaults())
    
    # Add data (styling handled by template)
    fig.add_bar(x=genres, y=counts)
    
    # Apply grid layout if needed
    fig.update_layout(template=create_with_table_grid())
"""
