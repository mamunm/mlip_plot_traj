"""
Hydrogen bond plotting functions.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def _log(logger: Optional[Any], method: str, message: str, verbose: bool = True):
    """Helper to log or print based on logger availability."""
    if not verbose:
        return
    if logger is not None:
        getattr(logger, method)(message)
    else:
        print(message)


# Colors for regions (matching diffusion module)
REGION_COLORS = {
    'global': '#226556',       # Dark green/teal
    'interface_a': '#da4a64',  # Red/Pink
    'interface_b': '#6181ad',  # Blue
    'bulk': '#70608d',         # Purple
}

REGION_LABELS = {
    'global': 'Global',
    'interface_a': 'Interface A (Lower)',
    'interface_b': 'Interface B (Upper)',
    'bulk': 'Bulk (Central)',
}


def plot_hbond_time_series(
    results: Dict[str, Any],
    output_file: str = 'hbond_timeseries.png',
    metric: str = 'nhbond',
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_mean: bool = True,
    linewidth: float = 0.5,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Plot H-bond count or <nhbond> over time with mean line.

    Creates a single plot for global analysis, or subplots for region analysis.

    Parameters
    ----------
    results : dict
        Results from compute_hbond_analysis()
    output_file : str
        Output filename
    metric : str
        'nhbond' for normalized metric, 'n_hbonds' for raw count
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    show_mean : bool
        Show horizontal mean line
    linewidth : float
        Line width for time series
    show : bool
        Show plot interactively
    verbose : bool
        Print progress
    logger : optional
        Logger instance

    Returns
    -------
    output_file : str
        Path to saved figure
    """
    ts = results['time_series']
    time = ts['time']
    summary = results['summary']
    regions = results['regions']
    use_region_analysis = regions is not None

    if use_region_analysis:
        # Create 2x2 subplot for regions + global
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        region_order = ['global', 'interface_a', 'bulk', 'interface_b']

        for idx, region_name in enumerate(region_order):
            ax = axes[idx]

            if region_name not in ts:
                continue

            data = ts[region_name][metric]
            color = REGION_COLORS.get(region_name, '#000000')

            # Plot time series
            ax.plot(time, data, '-', color=color, linewidth=linewidth, alpha=0.7)

            # Plot mean line
            if show_mean and region_name in summary:
                if metric == 'nhbond':
                    mean_val = summary[region_name]['mean_nhbond']
                    std_val = summary[region_name]['std_nhbond']
                else:
                    mean_val = summary[region_name]['mean_n_hbonds']
                    std_val = summary[region_name]['std_n_hbonds']

                ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.5,
                           label=f'Mean: {mean_val:.3f} +/- {std_val:.3f}')

            # Labels
            if region_name == 'global':
                title = REGION_LABELS[region_name]
            else:
                z_range = regions.get(region_name, (0, 0))
                title = f'{REGION_LABELS[region_name]}\n({z_range[0]:.1f}-{z_range[1]:.1f} A)'

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (ps)', fontsize=11)

            if metric == 'nhbond':
                ax.set_ylabel(r'$\langle n_{hbond} \rangle$', fontsize=11)
            else:
                ax.set_ylabel('N H-bonds', fontsize=11)

            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

        plt.tight_layout()
    else:
        # Single plot for global
        fig, ax = plt.subplots(figsize=(10, 6))

        data = ts['global'][metric]
        color = REGION_COLORS['global']

        ax.plot(time, data, '-', color=color, linewidth=linewidth, alpha=0.7)

        if show_mean and 'global' in summary:
            if metric == 'nhbond':
                mean_val = summary['global']['mean_nhbond']
                std_val = summary['global']['std_nhbond']
            else:
                mean_val = summary['global']['mean_n_hbonds']
                std_val = summary['global']['std_n_hbonds']

            ax.axhline(mean_val, color='red', linestyle='--', linewidth=1.5,
                       label=f'Mean: {mean_val:.3f} +/- {std_val:.3f}')

        ax.set_xlabel('Time (ps)', fontsize=14, fontweight='bold')

        if metric == 'nhbond':
            ax.set_ylabel(r'$\langle n_{hbond} \rangle$', fontsize=14, fontweight='bold')
            ax.set_title(r'$\langle n_{hbond} \rangle$ = 2 $\times$ N$_{hbond}$ / N$_O$ Over Time',
                         fontsize=16, fontweight='bold')
        else:
            ax.set_ylabel('Number of H-bonds', fontsize=14, fontweight='bold')
            ax.set_title('Hydrogen Bonds Over Time', fontsize=16, fontweight='bold')

        ax.legend(fontsize=11, loc='upper right')
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


def plot_nhbond_distribution(
    results: Dict[str, Any],
    output_file: str = 'hbond_distribution.png',
    bins: int = 30,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Plot histogram of <nhbond> distribution.

    Creates subplots for each region if region analysis is enabled.

    Parameters
    ----------
    results : dict
        Results from compute_hbond_analysis()
    output_file : str
        Output filename
    bins : int
        Number of histogram bins
    show : bool
        Show plot interactively
    verbose : bool
        Print progress
    logger : optional
        Logger instance

    Returns
    -------
    output_file : str
        Path to saved figure
    """
    ts = results['time_series']
    summary = results['summary']
    regions = results['regions']
    use_region_analysis = regions is not None

    if use_region_analysis:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        region_order = ['global', 'interface_a', 'bulk', 'interface_b']

        for idx, region_name in enumerate(region_order):
            ax = axes[idx]

            if region_name not in ts:
                continue

            data = ts[region_name]['nhbond']
            color = REGION_COLORS.get(region_name, '#000000')

            ax.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.7)

            if region_name in summary:
                mean_val = summary[region_name]['mean_nhbond']
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_val:.3f}')

            if region_name == 'global':
                title = REGION_LABELS[region_name]
            else:
                z_range = regions.get(region_name, (0, 0))
                title = f'{REGION_LABELS[region_name]}\n({z_range[0]:.1f}-{z_range[1]:.1f} A)'

            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(r'$\langle n_{hbond} \rangle$', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

        data = ts['global']['nhbond']
        color = REGION_COLORS['global']

        ax.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.7)

        if 'global' in summary:
            mean_val = summary['global']['mean_nhbond']
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {mean_val:.3f}')

        ax.set_xlabel(r'$\langle n_{hbond} \rangle$', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title(r'Distribution of $\langle n_{hbond} \rangle$', fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()

    plt.close()
    return output_file


def plot_hbond_geometry(
    results: Dict[str, Any],
    output_file: str = 'hbond_geometry.png',
    bins: int = 50,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Plot D-A distance and D-H-A angle distributions.

    Creates 2x1 subplot:
    - Top: D-A distance histogram
    - Bottom: D-H-A angle histogram

    Parameters
    ----------
    results : dict
        Results from compute_hbond_analysis()
    output_file : str
        Output filename
    bins : int
        Number of histogram bins
    show : bool
        Show plot interactively
    verbose : bool
        Print progress
    logger : optional
        Logger instance

    Returns
    -------
    output_file : str
        Path to saved figure
    """
    geometry = results['geometry']
    distances = geometry['distances']
    angles = geometry['angles']

    if len(distances) == 0:
        if verbose:
            _log(logger, 'warning', "No H-bonds detected, skipping geometry plot")
        return output_file

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Distance histogram
    ax1 = axes[0]
    ax1.hist(distances, bins=bins, color='#226556', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(distances):.3f} A')
    ax1.set_xlabel('D-A Distance (A)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Donor-Acceptor Distance Distribution', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # Angle histogram
    ax2 = axes[1]
    ax2.hist(angles, bins=bins, color='#da4a64', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(angles), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(angles):.1f} deg')
    ax2.set_xlabel('D-H-A Angle (deg)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('D-H-A Angle Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()

    plt.close()
    return output_file


def plot_hbond_regions_summary(
    results: Dict[str, Any],
    output_file: str = 'hbond_regions_summary.png',
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Create bar chart comparing <nhbond> across regions with error bars.

    Parameters
    ----------
    results : dict
        Results from compute_hbond_analysis()
    output_file : str
        Output filename
    show : bool
        Show plot interactively
    verbose : bool
        Print progress
    logger : optional
        Logger instance

    Returns
    -------
    output_file : str
        Path to saved figure
    """
    summary = results['summary']
    regions = results['regions']

    if regions is None:
        if verbose:
            _log(logger, 'warning', "No region analysis, skipping region summary plot")
        return output_file

    fig, ax = plt.subplots(figsize=(10, 6))

    region_order = ['interface_a', 'bulk', 'interface_b']
    x = np.arange(len(region_order))
    width = 0.6

    nhbond_values = []
    nhbond_errors = []
    colors = []

    for region_name in region_order:
        if region_name in summary:
            s = summary[region_name]
            nhbond_values.append(s['mean_nhbond'])
            nhbond_errors.append(s['std_nhbond'])
            colors.append(REGION_COLORS.get(region_name, '#000000'))
        else:
            nhbond_values.append(0)
            nhbond_errors.append(0)
            colors.append('#000000')

    bars = ax.bar(x, nhbond_values, width, yerr=nhbond_errors,
                  color=colors, edgecolor='black', linewidth=1, capsize=6)

    # Add value labels on bars
    for bar, val, err in zip(bars, nhbond_values, nhbond_errors):
        height = bar.get_height()
        text = f'{val:.3f}+/-{err:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.02,
                text, ha='center', va='bottom', fontsize=10, fontweight='bold')

    # X-axis labels with z-ranges
    x_labels = []
    for region_name in region_order:
        z_range = regions.get(region_name, (0, 0))
        x_labels.append(f'{REGION_LABELS[region_name]}\n({z_range[0]:.1f}-{z_range[1]:.1f} A)')

    ax.set_xlabel('Region', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$\langle n_{hbond} \rangle$', fontsize=12, fontweight='bold')
    ax.set_title('Average H-bonds per Water Molecule by Region', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Set y-limit
    max_val = max(v + e for v, e in zip(nhbond_values, nhbond_errors)) if nhbond_values else 1
    ax.set_ylim(0, max_val * 1.25)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()

    plt.close()
    return output_file
