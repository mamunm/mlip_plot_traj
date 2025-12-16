"""
Density profile plotting functions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

from .styles import (
    ELEMENT_COLORS, TRAJECTORY_COLORS, LINE_STYLES,
    get_element_color, get_trajectory_color, get_line_style
)


def _log(logger: Optional[Any], method: str, message: str, verbose: bool = True):
    """Helper to log or print based on logger availability."""
    if not verbose:
        return
    if logger is not None:
        getattr(logger, method)(message)
    else:
        print(message)


def plot_density_profiles(
    profiles: Union[Dict, List[Tuple[str, Dict]]],
    output_prefix: str = 'density',
    combined: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    linewidth: float = 2.5,
    show_errorbar: bool = True,
    fill_alpha: float = 0.3,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> List[str]:
    """
    Create density profile plots.

    Parameters
    ----------
    profiles : dict or list
        If dict: {element: (z, density, std_error)} for single trajectory
        If list: [(label, profiles_dict), ...] for multiple trajectories
    output_prefix : str
        Prefix for output filenames
    combined : bool
        If True, plot all elements in one figure
    xlim : tuple, optional
        X-axis limits (min, max)
    ylim : tuple, optional
        Y-axis limits (min, max)
    labels : list, optional
        Labels for trajectories
    colors : list, optional
        Custom colors for trajectories
    linestyles : list, optional
        Line styles for trajectories
    linewidth : float
        Line width
    show_errorbar : bool
        Show error bands using fill_between (default: True)
    fill_alpha : float
        Transparency of error band fill (default: 0.3)
    show : bool
        Show plot interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_files : list
        List of saved file paths
    """
    if verbose:
        _log(logger, 'info', "Creating density plots...")

    output_files = []

    # Normalize input to list format
    if isinstance(profiles, dict):
        all_profiles = [('Trajectory', profiles)]
    else:
        all_profiles = list(profiles)

    # Apply custom labels if provided
    if labels is not None:
        all_profiles = [(labels[i], prof) for i, (_, prof) in enumerate(all_profiles)]

    multiple_trajectories = len(all_profiles) > 1

    # Setup colors and linestyles
    if colors is None:
        colors = TRAJECTORY_COLORS if multiple_trajectories else None
    if linestyles is None:
        linestyles = LINE_STYLES

    # Get all unique elements
    all_elements = set()
    for _, prof in all_profiles:
        all_elements.update(prof.keys())
    all_elements = sorted(all_elements)

    # Check if we have error bars
    first_prof = all_profiles[0][1]
    first_elem = list(first_prof.keys())[0]
    has_errors = first_prof[first_elem][2] is not None

    if combined:
        # Single plot with all elements
        fig, ax = plt.subplots(figsize=(10, 6))

        for element in all_elements:
            for traj_idx, (label, prof) in enumerate(all_profiles):
                if element not in prof:
                    continue

                z, density, std_error = prof[element]

                if multiple_trajectories:
                    color = colors[traj_idx % len(colors)]
                    linestyle = linestyles[traj_idx % len(linestyles)]
                    plot_label = f"{element} ({label})"
                else:
                    color = get_element_color(element)
                    linestyle = '-'
                    plot_label = element

                # Plot the line
                ax.plot(
                    z, density, linewidth=linewidth,
                    label=plot_label, color=color, linestyle=linestyle
                )

                # Add error band if available and requested
                if show_errorbar and has_errors and std_error is not None:
                    ax.fill_between(
                        z, density - std_error, density + std_error,
                        color=color, alpha=fill_alpha, linewidth=0
                    )

        ax.set_xlabel('z (Å)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Density (g/cm³)', fontsize=14, fontweight='bold')
        ax.set_title('Density Profiles', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10, ncol=2 if multiple_trajectories else 1)
        ax.grid(True, alpha=0.3)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        output_file = f"{output_prefix}_combined.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        output_files.append(output_file)
        if verbose:
            _log(logger, 'success', f"Saved: {output_file}")

        if show:
            plt.show()
        plt.close()

    else:
        # Individual plots for each element
        for element in all_elements:
            fig, ax = plt.subplots(figsize=(8, 6))

            for traj_idx, (label, prof) in enumerate(all_profiles):
                if element not in prof:
                    continue

                z, density, std_error = prof[element]

                if multiple_trajectories:
                    color = colors[traj_idx % len(colors)]
                    linestyle = linestyles[traj_idx % len(linestyles)]
                    plot_label = label
                else:
                    color = get_element_color(element)
                    linestyle = '-'
                    plot_label = None

                # Plot the line
                ax.plot(
                    z, density, linewidth=linewidth,
                    color=color, linestyle=linestyle, label=plot_label
                )

                # Add error band if available and requested
                if show_errorbar and has_errors and std_error is not None:
                    ax.fill_between(
                        z, density - std_error, density + std_error,
                        color=color, alpha=fill_alpha, linewidth=0
                    )

            ax.set_xlabel('z (Å)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Density (g/cm³)', fontsize=14, fontweight='bold')
            ax.set_title(f'{element} Density Profile', fontsize=16, fontweight='bold')

            if multiple_trajectories:
                ax.legend(fontsize=12)

            ax.grid(True, alpha=0.3)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            else:
                max_density = max(
                    np.max(prof[element][1])
                    for _, prof in all_profiles if element in prof
                )
                ax.set_ylim(0, max_density * 1.1)

            output_file = f"{output_prefix}_{element}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            output_files.append(output_file)
            if verbose:
                _log(logger, 'success', f"Saved: {output_file}")

            if show:
                plt.show()
            plt.close()

    return output_files


def plot_first_peaks(
    peak_data: Dict[str, Dict[str, Tuple[float, float, Optional[float]]]],
    output_prefix: str = 'density',
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    linewidth: float = 2.5,
    combined: bool = False,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> List[str]:
    """
    Plot first peak values as line graphs with error bars.

    Parameters
    ----------
    peak_data : dict
        Dictionary mapping element to {label: (peak_z, peak_density, peak_stderr)}
    output_prefix : str
        Output filename prefix
    colors : list, optional
        Colors for trajectories
    linestyles : list, optional
        Line styles for trajectories
    linewidth : float
        Line width
    combined : bool
        If True, plot all elements in one figure
    show : bool
        Show plot interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_files : list
        List of saved file paths
    """
    if verbose:
        _log(logger, 'info', "Creating first peak plots...")

    output_files = []

    if not peak_data:
        if verbose:
            _log(logger, 'warning', "No peaks detected!")
        return output_files

    elements = sorted(peak_data.keys())

    # Get all trajectory labels
    all_labels = []
    for element_peaks in peak_data.values():
        all_labels.extend(element_peaks.keys())
    labels = sorted(set(all_labels), key=lambda x: all_labels.index(x))

    if colors is None:
        colors = TRAJECTORY_COLORS
    if linestyles is None:
        linestyles = LINE_STYLES

    # Check if we have error bars
    has_errors = any(
        stderr is not None
        for element_peaks in peak_data.values()
        for _, _, stderr in element_peaks.values()
    )

    multiple_trajectories = len(labels) > 1

    if combined:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(labels))

        for elem_idx, element in enumerate(elements):
            element_peaks = peak_data[element]
            peak_densities = []
            peak_stderrs = [] if has_errors else None

            for label in labels:
                if label in element_peaks:
                    _, peak_density, peak_stderr = element_peaks[label]
                    peak_densities.append(peak_density)
                    if has_errors:
                        peak_stderrs.append(peak_stderr if peak_stderr else 0)
                else:
                    peak_densities.append(np.nan)
                    if has_errors:
                        peak_stderrs.append(0)

            color = get_element_color(element)

            if has_errors:
                ax.errorbar(
                    x_pos, peak_densities, yerr=peak_stderrs,
                    linewidth=linewidth, marker='o', markersize=8,
                    color=color, label=element,
                    elinewidth=linewidth * 0.5, capsize=5,
                    capthick=linewidth * 0.5, alpha=0.9
                )
            else:
                ax.plot(
                    x_pos, peak_densities, linewidth=linewidth,
                    marker='o', markersize=8, color=color, label=element
                )

        ax.set_xlabel('System', fontsize=14, fontweight='bold')
        ax.set_ylabel('First Peak Density (g/cm$^3$)', fontsize=14, fontweight='bold')
        ax.set_title('First Peak Comparison - All Elements', fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        output_file = f"{output_prefix}_first_peaks_combined.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        output_files.append(output_file)
        if verbose:
            _log(logger, 'success', f"Saved: {output_file}")

        if show:
            plt.show()
        plt.close()

    else:
        for element in elements:
            fig, ax = plt.subplots(figsize=(8, 6))

            element_peaks = peak_data[element]
            x_pos = np.arange(len(labels))
            peak_densities = []
            peak_stderrs = [] if has_errors else None

            for label in labels:
                if label in element_peaks:
                    _, peak_density, peak_stderr = element_peaks[label]
                    peak_densities.append(peak_density)
                    if has_errors:
                        peak_stderrs.append(peak_stderr if peak_stderr else 0)
                else:
                    peak_densities.append(np.nan)
                    if has_errors:
                        peak_stderrs.append(0)

            if multiple_trajectories:
                for i, (x, y) in enumerate(zip(x_pos, peak_densities)):
                    color = colors[i % len(colors)]
                    if has_errors:
                        ax.errorbar(
                            [x], [y], yerr=[peak_stderrs[i]],
                            linewidth=linewidth, marker='o', markersize=10,
                            color=color, linestyle='none',
                            elinewidth=linewidth * 0.5, capsize=5,
                            capthick=linewidth * 0.5, alpha=0.9, label=labels[i]
                        )
                    else:
                        ax.plot(
                            [x], [y], linewidth=linewidth, marker='o',
                            markersize=10, color=color, label=labels[i]
                        )

                ax.plot(
                    x_pos, peak_densities, linewidth=linewidth * 0.7,
                    color='gray', alpha=0.5, linestyle='--', zorder=0
                )
            else:
                color = get_element_color(element)
                if has_errors:
                    ax.errorbar(
                        x_pos, peak_densities, yerr=peak_stderrs,
                        linewidth=linewidth, marker='o', markersize=10,
                        color=color, elinewidth=linewidth * 0.5,
                        capsize=5, capthick=linewidth * 0.5, alpha=0.9
                    )
                else:
                    ax.plot(
                        x_pos, peak_densities, linewidth=linewidth,
                        marker='o', markersize=10, color=color
                    )

            ax.set_xlabel('System', fontsize=14, fontweight='bold')
            ax.set_ylabel('First Peak Density (g/cm$^3$)', fontsize=14, fontweight='bold')
            ax.set_title(f'{element} First Peak', fontsize=16, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right')

            if multiple_trajectories:
                ax.legend(fontsize=11, loc='best')

            ax.grid(True, alpha=0.3)

            valid_densities = [d for d in peak_densities if not np.isnan(d)]
            if valid_densities:
                ax.set_ylim(0, max(valid_densities) * 1.15)

            plt.tight_layout()
            output_file = f"{output_prefix}_first_peak_{element}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            output_files.append(output_file)
            if verbose:
                _log(logger, 'success', f"Saved: {output_file}")

            if show:
                plt.show()
            plt.close()

    return output_files


def plot_flypet_diagnostics(
    diagnostics: Dict[str, Dict],
    output_prefix: str = 'density',
    show: bool = False
) -> List[str]:
    """
    Plot Flyvbjerg-Petersen blocking diagnostics for density analysis.

    Creates a plot showing standard error vs blocking level for each element.

    Parameters
    ----------
    diagnostics : dict
        Dictionary mapping element name to FlyPet diagnostics dict
        containing 'block_sizes' and 'variances' arrays
    output_prefix : str
        Prefix for output filename
    show : bool
        Show plot interactively

    Returns
    -------
    output_files : list
        List of saved file paths
    """
    output_files = []

    if not diagnostics:
        return output_files

    elements = sorted(diagnostics.keys())

    fig, ax = plt.subplots(figsize=(8, 6))

    for elem in elements:
        diag = diagnostics[elem]
        if 'block_sizes' not in diag or 'variances' not in diag:
            continue

        block_sizes = diag['block_sizes']
        variances = diag['variances']

        # Convert variance to standard error
        std_errors = np.sqrt(variances)

        color = get_element_color(elem)
        ax.plot(
            block_sizes, std_errors,
            'o-', color=color, label=elem,
            linewidth=2, markersize=6
        )

    ax.set_xlabel('Block Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Standard Error (g/cm³)', fontsize=14, fontweight='bold')
    ax.set_title('Flyvbjerg-Petersen Blocking Analysis', fontsize=16, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    output_file = f"{output_prefix}_flypet_blocking.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    output_files.append(output_file)

    if show:
        plt.show()
    plt.close()

    return output_files
