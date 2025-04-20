"""Chart-specific default styles.

This package provides default styling for different chart types:
- bar.py: Bar chart defaults
- heatmap.py: Heatmap defaults
- scatter.py: Scatter plot defaults
- table.py: Table defaults
"""

from .bar import create_bar_defaults
from .heatmap import create_heatmap_defaults
from .scatter import create_scatter_defaults
from .table import create_table_defaults
from .sankey import create_sankey_defaults

__all__ = [
    'create_bar_defaults',
    'create_heatmap_defaults',
    'create_scatter_defaults',
    'create_table_defaults',
    'create_sankey_defaults'
]
