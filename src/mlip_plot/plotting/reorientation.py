"""
Plotting functions for water reorientation analysis.

Generates individual P(cos θ) and P(cos φ) distribution plots per region.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Region colors
REGION_COLORS = {
    'interface_a': '#da4a64',  # Red
    'interface_b': '#6181ad',  # Blue
    'bulk': '#226556',         # Green
    'global': '#70608d',       # Purple
}

REGION_LABELS = {
    'interface_a': 'Interface A (Lower)',
    'interface_b': 'Interface B (Upper)',
    'bulk': 'Bulk',
    'global': 'Global',
}


def _log(logger: Optional[Any], method: str, message: str, verbose: bool = True):
    """Helper to log or print based on logger availability."""
    if not verbose:
        return
    if logger is not None:
        getattr(logger, method)(message)
    else:
        print(message)


def plot_orientation_single_region(
    region_data: Dict,
    region_name: str,
    angle_type: str,
    output_prefix: str,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Plot P(cos) distribution for a single region.

    Parameters
    ----------
    region_data : dict
        Distribution data with 'bins', 'P', 'err' keys
    region_name : str
        Region name: 'interface_a', 'bulk', 'interface_b', or 'global'
    angle_type : str
        'theta' or 'phi' for axis labels
    output_prefix : str
        Output file prefix (without extension)
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    show : bool
        Show plot interactively
    verbose : bool
        Print progress
    logger : optional
        Logger instance

    Returns
    -------
    output_file : str
        Path to saved plot file
    """
    import matplotlib.pyplot as plt

    bins = region_data['bins']
    P = region_data['P']
    err = region_data['err']

    color = REGION_COLORS.get(region_name, '#333333')
    label = REGION_LABELS.get(region_name, region_name)

    # Add z-range to label if available
    if 'z_range' in region_data:
        z_min, z_max = region_data['z_range']
        label = f"{label} ({z_min:.1f}-{z_max:.1f} Å)"

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(bins, P, color=color, linewidth=1.5, label=label)

    # Error band
    if np.any(err > 0):
        ax.fill_between(bins, P - err, P + err, color=color, alpha=0.3)

    # Isotropic reference line at P = 0.5
    ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Isotropic')

    # Labels
    if angle_type == 'theta':
        ax.set_xlabel(r'cos($\theta$)', fontsize=14)
        ax.set_ylabel(r'P(cos $\theta$)', fontsize=14)
        title = f'O-H Bond Orientation - {REGION_LABELS.get(region_name, region_name)}'
    else:
        ax.set_xlabel(r'cos($\phi$)', fontsize=14)
        ax.set_ylabel(r'P(cos $\phi$)', fontsize=14)
        title = f'Dipole Orientation - {REGION_LABELS.get(region_name, region_name)}'

    ax.set_title(title, fontsize=16)

    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(-1, 1)

    if ylim:
        ax.set_ylim(ylim)

    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    output_file = f"{output_prefix}_{region_name}_{angle_type}.png"
    fig.savefig(output_file, dpi=150)

    if show:
        plt.show()

    plt.close(fig)

    if verbose:
        _log(logger, 'success', f"Saved: [bold]{output_file}[/bold]")

    return output_file


def plot_all_regions(
    theta_data: Dict[str, Dict],
    phi_data: Dict[str, Dict],
    output_prefix: str,
    regions_order: List[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> List[str]:
    """
    Plot individual P(cos) distributions for each region.

    Creates 6 plots (3 regions × 2 angle types) when region analysis is enabled,
    or 2 plots (global only) when disabled.

    Parameters
    ----------
    theta_data : dict
        Theta distribution data per region
    phi_data : dict
        Phi distribution data per region
    output_prefix : str
        Output file prefix
    regions_order : list, optional
        Order of regions to plot. Default: interface_a, bulk, interface_b
    xlim : tuple, optional
        X-axis limits (min, max)
    ylim : tuple, optional
        Y-axis limits (min, max)
    show : bool
        Show plots interactively
    verbose : bool
        Print progress
    logger : optional
        Logger instance

    Returns
    -------
    output_files : list of str
        Paths to saved plot files
    """
    if regions_order is None:
        regions_order = ['interface_a', 'bulk', 'interface_b']

    output_files = []

    for region_name in regions_order:
        # Theta plot for this region
        if region_name in theta_data:
            output_file = plot_orientation_single_region(
                theta_data[region_name],
                region_name,
                'theta',
                output_prefix,
                xlim=xlim,
                ylim=ylim,
                show=show,
                verbose=verbose,
                logger=logger
            )
            output_files.append(output_file)

        # Phi plot for this region
        if region_name in phi_data:
            output_file = plot_orientation_single_region(
                phi_data[region_name],
                region_name,
                'phi',
                output_prefix,
                xlim=xlim,
                ylim=ylim,
                show=show,
                verbose=verbose,
                logger=logger
            )
            output_files.append(output_file)

    return output_files


def plot_combined_regions(
    theta_data: Dict[str, Dict],
    phi_data: Dict[str, Dict],
    output_prefix: str,
    regions_order: List[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> List[str]:
    """
    Plot all regions on combined plots (one for theta, one for phi).

    Parameters
    ----------
    theta_data : dict
        Theta distribution data per region
    phi_data : dict
        Phi distribution data per region
    output_prefix : str
        Output file prefix
    regions_order : list, optional
        Order of regions to plot. Default: interface_a, bulk, interface_b
    xlim : tuple, optional
        X-axis limits (min, max)
    ylim : tuple, optional
        Y-axis limits (min, max)
    show : bool
        Show plots interactively
    verbose : bool
        Print progress
    logger : optional
        Logger instance

    Returns
    -------
    output_files : list of str
        Paths to saved plot files
    """
    import matplotlib.pyplot as plt

    if regions_order is None:
        regions_order = ['interface_a', 'bulk', 'interface_b']

    output_files = []

    # Plot theta (all regions combined)
    fig, ax = plt.subplots(figsize=(8, 6))

    for region_name in regions_order:
        if region_name not in theta_data:
            continue

        region_data = theta_data[region_name]
        bins = region_data['bins']
        P = region_data['P']
        err = region_data['err']

        color = REGION_COLORS.get(region_name, '#333333')
        label = REGION_LABELS.get(region_name, region_name)

        if 'z_range' in region_data:
            z_min, z_max = region_data['z_range']
            label = f"{label} ({z_min:.1f}-{z_max:.1f} Å)"

        ax.plot(bins, P, color=color, linewidth=1.5, label=label)

        if np.any(err > 0):
            ax.fill_between(bins, P - err, P + err, color=color, alpha=0.3)

    ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Isotropic')
    ax.set_xlabel(r'cos($\theta$)', fontsize=14)
    ax.set_ylabel(r'P(cos $\theta$)', fontsize=14)
    ax.set_title('O-H Bond Orientation Distribution', fontsize=16)

    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(-1, 1)

    if ylim:
        ax.set_ylim(ylim)

    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_file = f"{output_prefix}_theta.png"
    fig.savefig(output_file, dpi=150)

    if show:
        plt.show()

    plt.close(fig)

    if verbose:
        _log(logger, 'success', f"Saved: [bold]{output_file}[/bold]")

    output_files.append(output_file)

    # Plot phi (all regions combined)
    fig, ax = plt.subplots(figsize=(8, 6))

    for region_name in regions_order:
        if region_name not in phi_data:
            continue

        region_data = phi_data[region_name]
        bins = region_data['bins']
        P = region_data['P']
        err = region_data['err']

        color = REGION_COLORS.get(region_name, '#333333')
        label = REGION_LABELS.get(region_name, region_name)

        if 'z_range' in region_data:
            z_min, z_max = region_data['z_range']
            label = f"{label} ({z_min:.1f}-{z_max:.1f} Å)"

        ax.plot(bins, P, color=color, linewidth=1.5, label=label)

        if np.any(err > 0):
            ax.fill_between(bins, P - err, P + err, color=color, alpha=0.3)

    ax.axhline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Isotropic')
    ax.set_xlabel(r'cos($\phi$)', fontsize=14)
    ax.set_ylabel(r'P(cos $\phi$)', fontsize=14)
    ax.set_title('Dipole Orientation Distribution', fontsize=16)

    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(-1, 1)

    if ylim:
        ax.set_ylim(ylim)

    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_file = f"{output_prefix}_phi.png"
    fig.savefig(output_file, dpi=150)

    if show:
        plt.show()

    plt.close(fig)

    if verbose:
        _log(logger, 'success', f"Saved: [bold]{output_file}[/bold]")

    output_files.append(output_file)

    return output_files
