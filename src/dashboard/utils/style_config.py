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
    },
    'scrollable_table': {
        'max_height': 400,
        'padding': 10,
        'margin_bottom': 20
    }
}

CSS_CLASSES = {
    'unmatched_shows': f"""
        .unmatched-shows {{
            max-height: {DIMENSIONS['scrollable_table']['max_height']}px;
            overflow-y: auto;
            border: 1px solid {COLORS['text']['secondary']};
            padding: {DIMENSIONS['scrollable_table']['padding']}px;
            margin-bottom: {DIMENSIONS['scrollable_table']['margin_bottom']}px;
        }}
    """
}
