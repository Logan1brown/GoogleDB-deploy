"""Grid layout templates.

This package provides common grid arrangements:
- dual.py: Side-by-side charts
- stacked.py: Vertical stacking
- with_table.py: Chart + data table
"""

from .dual import create_dual_grid
from .chart_insights_table import create_with_table_grid
from .stacked import create_stacked_grid

__all__ = ['create_dual_grid', 'create_with_table_grid', 'create_stacked_grid']
