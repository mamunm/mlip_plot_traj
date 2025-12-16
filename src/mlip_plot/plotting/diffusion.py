"""
Diffusion coefficient plotting functions.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from .styles import TRAJECTORY_COLORS


def _log(logger: Optional[Any], method: str, message: str, verbose: bool = True):
    """Helper to log or print based on logger availability."""
    if not verbose:
        return
    if logger is not None:
        getattr(logger, method)(message)
    else:
        print(message)


# Colors for different MSD types
MSD_COLORS = {
    'planar': '#da4a64',       # Red/Pink - x-y plane
    'perpendicular': '#6181ad', # Blue - z direction
    'total': '#226556',         # Dark green/teal - 3D total
}

MSD_LABELS = {
    'planar': r'MSD$_{xy}$ (planar)',
    'perpendicular': r'MSD$_z$ (perpendicular)',
    'total': r'MSD$_{3D}$ (total)',
}


def plot_msd(
    msd_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    diffusion_results: Optional[Dict[str, Dict]] = None,
    output_file: str = 'msd.png',
    msd_types: Optional[List[str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    linewidth: float = 2.5,
    show_fit: bool = True,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Plot MSD vs time with optional linear fits.

    Parameters
    ----------
    msd_data : dict
        Dictionary with keys 'planar', 'perpendicular', 'total'
        Each value is (time_array, msd_array)
    diffusion_results : dict, optional
        Dictionary with diffusion fit results for each type
    output_file : str
        Output filename
    msd_types : list, optional
        Which MSD types to plot (default: all)
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    linewidth : float
        Line width
    show_fit : bool
        Show linear fit lines
    show : bool
        Show plot interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_file : str
        Path to saved figure
    """
    if msd_types is None:
        msd_types = ['planar', 'perpendicular', 'total']

    fig, ax = plt.subplots(figsize=(10, 6))

    for msd_type in msd_types:
        if msd_type not in msd_data:
            continue

        time, msd = msd_data[msd_type]
        color = MSD_COLORS.get(msd_type, '#000000')
        label = MSD_LABELS.get(msd_type, msd_type)

        # Plot MSD data
        ax.plot(time, msd, '-', color=color, linewidth=linewidth, label=label)

        # Plot fit if available
        if show_fit and diffusion_results and msd_type in diffusion_results:
            fit = diffusion_results[msd_type]
            fit_start = fit['fit_start_ps']
            fit_end = fit['fit_end_ps']
            slope = fit['slope']
            intercept = fit['intercept']
            D = fit['D_A2ps']

            # Generate fit line
            mask = (time >= fit_start) & (time <= fit_end)
            time_fit = time[mask]
            msd_fit = slope * time_fit + intercept

            # Plot fit line
            dim_map = {'planar': 2, 'perpendicular': 1, 'total': 3}
            dim = dim_map.get(msd_type, 2)
            ax.plot(
                time_fit, msd_fit, '--', color=color,
                linewidth=linewidth * 0.7, alpha=0.8,
                label=f'Fit: D = {D:.4f} A$^2$/ps'
            )

    ax.set_xlabel('Time (ps)', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'MSD ($\AA^2$)', fontsize=14, fontweight='bold')
    ax.set_title('Mean Square Displacement', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()

    plt.close()
    return output_file


def plot_msd_subplots(
    msd_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    diffusion_results: Optional[Dict[str, Dict]] = None,
    output_file: str = 'msd_subplots.png',
    xlim: Optional[Tuple[float, float]] = None,
    linewidth: float = 2.5,
    show_fit: bool = True,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Plot MSD in separate subplots for each component.

    Parameters
    ----------
    msd_data : dict
        Dictionary with keys 'planar', 'perpendicular', 'total'
    diffusion_results : dict, optional
        Dictionary with diffusion fit results
    output_file : str
        Output filename
    xlim : tuple, optional
        X-axis limits
    linewidth : float
        Line width
    show_fit : bool
        Show linear fit lines
    show : bool
        Show plot interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_file : str
        Path to saved figure
    """
    msd_types = ['planar', 'perpendicular', 'total']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = {
        'planar': 'Planar (x-y)',
        'perpendicular': 'Perpendicular (z)',
        'total': 'Total (3D)'
    }

    ylabels = {
        'planar': r'MSD$_{xy}$ ($\AA^2$)',
        'perpendicular': r'MSD$_z$ ($\AA^2$)',
        'total': r'MSD$_{3D}$ ($\AA^2$)'
    }

    for idx, msd_type in enumerate(msd_types):
        ax = axes[idx]

        if msd_type not in msd_data:
            continue

        time, msd = msd_data[msd_type]
        color = MSD_COLORS.get(msd_type, '#000000')

        # Plot MSD data
        ax.plot(time, msd, '-', color=color, linewidth=linewidth, label='MSD')

        # Plot fit if available
        if show_fit and diffusion_results and msd_type in diffusion_results:
            fit = diffusion_results[msd_type]
            fit_start = fit['fit_start_ps']
            fit_end = fit['fit_end_ps']
            slope = fit['slope']
            intercept = fit['intercept']
            D = fit['D_A2ps']
            D_cm2s = fit['D_cm2s']
            r_squared = fit['r_squared']

            # Generate fit line
            mask = (time >= fit_start) & (time <= fit_end)
            time_fit = time[mask]
            msd_fit = slope * time_fit + intercept

            ax.plot(
                time_fit, msd_fit, '--', color='black',
                linewidth=linewidth * 0.7, alpha=0.8,
                label=f'Fit (R$^2$={r_squared:.3f})'
            )

            # Add text annotation
            ax.text(
                0.05, 0.95,
                f'D = {D:.4f} A$^2$/ps\n   = {D_cm2s:.2e} cm$^2$/s',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

        ax.set_xlabel('Time (ps)', fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabels[msd_type], fontsize=12, fontweight='bold')
        ax.set_title(titles[msd_type], fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)

        if xlim is not None:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()

    plt.close()
    return output_file


def plot_msd_comparison(
    all_msd_data: List[Tuple[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]],
    msd_type: str = 'planar',
    output_file: str = 'msd_comparison.png',
    colors: Optional[List[str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    linewidth: float = 2.5,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Compare MSD from multiple trajectories/systems.

    Parameters
    ----------
    all_msd_data : list
        List of (label, msd_data) tuples
    msd_type : str
        Which MSD type to compare ('planar', 'perpendicular', 'total')
    output_file : str
        Output filename
    colors : list, optional
        Colors for each trajectory
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    linewidth : float
        Line width
    show : bool
        Show plot interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_file : str
        Path to saved figure
    """
    if colors is None:
        colors = TRAJECTORY_COLORS

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (label, msd_data) in enumerate(all_msd_data):
        if msd_type not in msd_data:
            continue

        time, msd = msd_data[msd_type]
        color = colors[idx % len(colors)]

        ax.plot(time, msd, '-', color=color, linewidth=linewidth, label=label)

    titles = {
        'planar': 'Planar (x-y) MSD Comparison',
        'perpendicular': 'Perpendicular (z) MSD Comparison',
        'total': 'Total (3D) MSD Comparison'
    }

    ylabels = {
        'planar': r'MSD$_{xy}$ ($\AA^2$)',
        'perpendicular': r'MSD$_z$ ($\AA^2$)',
        'total': r'MSD$_{3D}$ ($\AA^2$)'
    }

    ax.set_xlabel('Time (ps)', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabels.get(msd_type, r'MSD ($\AA^2$)'), fontsize=14, fontweight='bold')
    ax.set_title(titles.get(msd_type, 'MSD Comparison'), fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()

    plt.close()
    return output_file


REGION_COLORS = {
    'interface_a': '#da4a64',   # Red/Pink
    'interface_b': '#6181ad',   # Blue
    'bulk': '#226556',          # Dark green/teal
}

REGION_LABELS = {
    'interface_a': 'Interface A (Lower)',
    'interface_b': 'Interface B (Upper)',
    'bulk': 'Bulk (Central)',
}


def plot_msd_regions(
    msd_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    diffusion_results: Dict[str, Dict[str, Dict]],
    regions: Dict[str, Tuple[float, float]],
    output_file: str = 'msd_regions.png',
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    linewidth: float = 2.0,
    show_fit: bool = True,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Plot MSD for each region (interface_a, interface_b, bulk).

    Creates a 3x3 grid: rows = regions, columns = MSD types (planar, perp, total).

    Parameters
    ----------
    msd_data : dict
        Nested dict: {region: {'planar': (time, msd), ...}}
    diffusion_results : dict
        Nested dict: {region: {'planar': {D_A2ps, ...}, ...}}
    regions : dict
        Dict mapping region names to (z_min, z_max) tuples
    output_file : str
        Output filename
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    linewidth : float
        Line width
    show_fit : bool
        Show linear fit lines
    show : bool
        Show plot interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_file : str
        Path to saved figure
    """
    region_order = ['interface_a', 'interface_b', 'bulk']
    msd_types = ['planar', 'perpendicular', 'total']

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))

    type_titles = {
        'planar': 'Planar (x-y)',
        'perpendicular': 'Perpendicular (z)',
        'total': 'Total (3D)'
    }

    for row_idx, region_name in enumerate(region_order):
        if region_name not in msd_data:
            continue

        region_msd = msd_data[region_name]
        region_diff = diffusion_results.get(region_name, {})
        z_range = regions.get(region_name, (0, 0))
        color = REGION_COLORS.get(region_name, '#000000')

        for col_idx, msd_type in enumerate(msd_types):
            ax = axes[row_idx, col_idx]

            if msd_type not in region_msd:
                continue

            time, msd = region_msd[msd_type]

            # Plot MSD data
            ax.plot(time, msd, '-', color=color, linewidth=linewidth, label='MSD')

            # Plot fit if available
            if show_fit and msd_type in region_diff:
                fit = region_diff[msd_type]
                fit_start = fit['fit_start_ps']
                fit_end = fit['fit_end_ps']
                slope = fit['slope']
                intercept = fit['intercept']
                D = fit['D_A2ps']
                r_squared = fit['r_squared']

                # Check for block statistics
                if 'D_mean' in fit:
                    D_mean = fit['D_mean']
                    D_std = fit['D_std']
                    d_text = f'D = {D_mean:.3f} ± {D_std:.3f}'
                else:
                    d_text = f'D = {D:.4f}'

                # Generate fit line
                mask = (time >= fit_start) & (time <= fit_end)
                time_fit = time[mask]
                msd_fit = slope * time_fit + intercept

                ax.plot(
                    time_fit, msd_fit, '--', color='black',
                    linewidth=linewidth * 0.7, alpha=0.8,
                    label=f'Fit (R²={r_squared:.3f})'
                )

                # Add text annotation
                ax.text(
                    0.05, 0.95,
                    f'{d_text} A²/ps',
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

            # Labels
            if row_idx == 2:  # Bottom row
                ax.set_xlabel('Time (ps)', fontsize=11, fontweight='bold')
            if col_idx == 0:  # Left column
                ax.set_ylabel(f'{REGION_LABELS[region_name]}\n({z_range[0]:.1f}-{z_range[1]:.1f} Å)\nMSD (Ų)',
                              fontsize=10, fontweight='bold')
            if row_idx == 0:  # Top row
                ax.set_title(type_titles[msd_type], fontsize=12, fontweight='bold')

            ax.legend(fontsize=8, loc='lower right')
            ax.grid(True, alpha=0.3)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()

    plt.close()
    return output_file


def plot_diffusion_regions_summary(
    diffusion_results: Dict[str, Dict[str, Dict]],
    regions: Dict[str, Tuple[float, float]],
    output_file: str = 'diffusion_regions_summary.png',
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Create a grouped bar chart comparing diffusion coefficients across regions.

    Parameters
    ----------
    diffusion_results : dict
        Nested dict: {region: {'planar': {D_A2ps, D_mean, D_std, ...}, ...}}
    regions : dict
        Dict mapping region names to (z_min, z_max) tuples
    output_file : str
        Output filename
    show : bool
        Show plot interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_file : str
        Path to saved figure
    """
    region_order = ['interface_a', 'interface_b', 'bulk']
    msd_types = ['planar', 'total']  # Skip perpendicular (usually ~0)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(region_order))
    width = 0.35
    offsets = [-width/2, width/2]

    type_labels = {'planar': 'Planar (x-y)', 'total': 'Total (3D)'}
    type_colors = {'planar': '#da4a64', 'total': '#226556'}

    for i, msd_type in enumerate(msd_types):
        D_values = []
        D_errors = []

        for region_name in region_order:
            if region_name in diffusion_results and msd_type in diffusion_results[region_name]:
                d = diffusion_results[region_name][msd_type]
                if 'D_mean' in d:
                    D_values.append(d['D_mean'])
                    D_errors.append(d['D_std'])
                else:
                    D_values.append(d['D_A2ps'])
                    D_errors.append(0)
            else:
                D_values.append(0)
                D_errors.append(0)

        bars = ax.bar(x + offsets[i], D_values, width, yerr=D_errors,
                      label=type_labels[msd_type], color=type_colors[msd_type],
                      edgecolor='black', linewidth=1, capsize=4)

        # Add value labels
        for bar, val, err in zip(bars, D_values, D_errors):
            height = bar.get_height()
            if err > 0:
                text = f'{val:.3f}±{err:.3f}'
            else:
                text = f'{val:.4f}'
            ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
                    text, ha='center', va='bottom', fontsize=9, rotation=0)

    # X-axis labels with z-ranges
    x_labels = []
    for region_name in region_order:
        z_range = regions.get(region_name, (0, 0))
        x_labels.append(f'{REGION_LABELS[region_name]}\n({z_range[0]:.1f}-{z_range[1]:.1f} Å)')

    ax.set_xlabel('Region', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'D ($\AA^2$/ps)', fontsize=12, fontweight='bold')
    ax.set_title('Diffusion Coefficients by Region', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Set y-limit
    all_vals = [diffusion_results.get(r, {}).get(t, {}).get('D_mean', diffusion_results.get(r, {}).get(t, {}).get('D_A2ps', 0))
                for r in region_order for t in msd_types]
    all_errs = [diffusion_results.get(r, {}).get(t, {}).get('D_std', 0) for r in region_order for t in msd_types]
    max_val = max(v + e for v, e in zip(all_vals, all_errs)) if all_vals else 1
    ax.set_ylim(0, max_val * 1.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()

    plt.close()
    return output_file


def plot_diffusion_summary(
    diffusion_results: Dict[str, Dict],
    output_file: str = 'diffusion_summary.png',
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Create a bar chart comparing diffusion coefficients.

    Parameters
    ----------
    diffusion_results : dict
        Dictionary with diffusion results for 'planar', 'perpendicular', 'total'
    output_file : str
        Output filename
    show : bool
        Show plot interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_file : str
        Path to saved figure
    """
    types = ['planar', 'perpendicular', 'total']
    labels = ['Planar\n(x-y)', 'Perpendicular\n(z)', 'Total\n(3D)']
    colors = [MSD_COLORS[t] for t in types]

    D_values = []
    for t in types:
        if t in diffusion_results:
            D_values.append(diffusion_results[t]['D_A2ps'])
        else:
            D_values.append(0)

    fig, ax = plt.subplots(figsize=(8, 6))

    x_pos = np.arange(len(types))
    bars = ax.bar(x_pos, D_values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, D_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_xlabel('Diffusion Type', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'D ($\AA^2$/ps)', fontsize=14, fontweight='bold')
    ax.set_title('Diffusion Coefficient Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(0, max(D_values) * 1.2 if D_values else 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()

    plt.close()
    return output_file
