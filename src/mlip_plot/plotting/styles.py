"""
Plot styling constants and utilities.
"""

# Colors for different elements
ELEMENT_COLORS = {
    'H': '#6181ad',
    'O': '#da4a64',
    'Cl': '#226556',
    'Na': '#70608d',
    'C': '#8B4513',
    'N': '#4169E1',
    'S': '#FFD700',
    'F': '#00CED1',
    'K': '#9932CC',
    'Ca': '#32CD32',
    'Mg': '#FF8C00',
    'Fe': '#CD853F',
    'Cu': '#B87333',
    'Pt': '#E5E4E2',
    'Au': '#FFD700',
    'Ag': '#C0C0C0',
}

# Colors for multiple trajectories
TRAJECTORY_COLORS = [
    '#da4a64',  # Red/Pink
    '#6181ad',  # Blue
    '#226556',  # Dark green/teal
    '#70608d',  # Purple
    '#E67E22',  # Orange
    '#16A085',  # Turquoise
    '#8E44AD',  # Violet
    '#C0392B',  # Dark red
]

# Line styles for multiple trajectories
LINE_STYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']

# Linestyle name mapping
LINESTYLE_MAP = {
    'solid': '-',
    'dashed': '--',
    'dotted': ':',
    'dashdot': '-.',
    '-': '-',
    '--': '--',
    ':': ':',
    '-.': '-.',
}


def get_element_color(element: str, default: str = '#000000') -> str:
    """Get color for an element."""
    return ELEMENT_COLORS.get(element, default)


def get_trajectory_color(index: int) -> str:
    """Get color for trajectory by index."""
    return TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)]


def get_line_style(index: int) -> str:
    """Get line style by index."""
    return LINE_STYLES[index % len(LINE_STYLES)]


def parse_linestyle(style: str) -> str:
    """Convert linestyle name to matplotlib code."""
    return LINESTYLE_MAP.get(style, style)
