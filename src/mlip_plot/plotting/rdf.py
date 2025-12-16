"""
RDF plotting functions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt

from .styles import ELEMENT_COLORS, TRAJECTORY_COLORS, get_element_color


def _log(logger: Optional[Any], method: str, message: str, verbose: bool = True):
    """Helper to log or print based on logger availability."""
    if not verbose:
        return
    if logger is not None:
        getattr(logger, method)(message)
    else:
        print(message)


def plot_rdf(
    rdf_data: Union[Dict, List[Tuple[str, Dict]]],
    output_prefix: str = 'rdf',
    pairs: Optional[List[Tuple[str, str]]] = None,
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
    Create RDF plots.

    Parameters
    ----------
    rdf_data : dict or list
        If dict: {(type1, type2): (r, g_r, std_error)} for single trajectory
        If list: [(label, rdf_dict), ...] for multiple trajectories
    output_prefix : str
        Prefix for output filenames
    pairs : list of tuple, optional
        Specific pairs to plot (e.g., [('O', 'O'), ('O', 'H')])
    combined : bool
        If True, plot all pairs in one figure
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    labels : list, optional
        Labels for trajectories
    colors : list, optional
        Custom colors
    linestyles : list, optional
        Line styles
    linewidth : float
        Line width
    show_errorbar : bool
        Show error bands
    fill_alpha : float
        Error band transparency
    show : bool
        Show plot interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_files : list
        Saved file paths
    """
    if verbose:
        _log(logger, 'info', "Creating RDF plots...")

    output_files = []

    # Normalize input
    if isinstance(rdf_data, dict):
        all_rdf_data = [('Trajectory', rdf_data)]
    else:
        all_rdf_data = list(rdf_data)

    if labels is not None:
        all_rdf_data = [(labels[i], rdf) for i, (_, rdf) in enumerate(all_rdf_data)]

    multiple_trajectories = len(all_rdf_data) > 1

    # Setup colors and linestyles
    if colors is None:
        colors = TRAJECTORY_COLORS
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':']

    # Get all pairs
    all_pairs = set()
    for _, rdf in all_rdf_data:
        all_pairs.update(rdf.keys())

    if pairs is not None:
        all_pairs = [p for p in pairs if p in all_pairs]
    else:
        all_pairs = sorted(all_pairs)

    # Check for errors
    first_rdf = all_rdf_data[0][1]
    first_pair = list(first_rdf.keys())[0]
    has_errors = first_rdf[first_pair][2] is not None

    # Pair-specific colors
    pair_colors = {
        ('O', 'O'): '#1f77b4',  # Blue
        ('O', 'H'): '#d62728',  # Red
        ('H', 'H'): '#2ca02c',  # Green
        ('Na', 'O'): '#9467bd',  # Purple
        ('Cl', 'O'): '#ff7f0e',  # Orange
    }

    if combined:
        fig, ax = plt.subplots(figsize=(10, 6))

        for pair in all_pairs:
            pair_label = f"{pair[0]}-{pair[1]}"

            for traj_idx, (label, rdf) in enumerate(all_rdf_data):
                if pair not in rdf:
                    continue

                r, g_r, std_error = rdf[pair]

                if multiple_trajectories:
                    color = colors[traj_idx % len(colors)]
                    linestyle = linestyles[traj_idx % len(linestyles)]
                    plot_label = f"{pair_label} ({label})"
                else:
                    color = pair_colors.get(pair, colors[list(all_pairs).index(pair) % len(colors)])
                    linestyle = '-'
                    plot_label = pair_label

                ax.plot(r, g_r, linewidth=linewidth, label=plot_label,
                        color=color, linestyle=linestyle)

                if show_errorbar and has_errors and std_error is not None:
                    ax.fill_between(r, g_r - std_error, g_r + std_error,
                                    color=color, alpha=fill_alpha, linewidth=0)

        ax.set_xlabel('r (Å)', fontsize=14, fontweight='bold')
        ax.set_ylabel('g(r)', fontsize=14, fontweight='bold')
        ax.set_title('Radial Distribution Functions', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

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
        # Individual plots for each pair
        for pair in all_pairs:
            fig, ax = plt.subplots(figsize=(8, 6))
            pair_label = f"{pair[0]}-{pair[1]}"

            for traj_idx, (label, rdf) in enumerate(all_rdf_data):
                if pair not in rdf:
                    continue

                r, g_r, std_error = rdf[pair]

                if multiple_trajectories:
                    color = colors[traj_idx % len(colors)]
                    linestyle = linestyles[traj_idx % len(linestyles)]
                    plot_label = label
                else:
                    color = pair_colors.get(pair, '#1f77b4')
                    linestyle = '-'
                    plot_label = None

                ax.plot(r, g_r, linewidth=linewidth, color=color,
                        linestyle=linestyle, label=plot_label)

                if show_errorbar and has_errors and std_error is not None:
                    ax.fill_between(r, g_r - std_error, g_r + std_error,
                                    color=color, alpha=fill_alpha, linewidth=0)

            ax.set_xlabel('r (Å)', fontsize=14, fontweight='bold')
            ax.set_ylabel('g(r)', fontsize=14, fontweight='bold')
            ax.set_title(f'{pair_label} Radial Distribution Function', fontsize=16, fontweight='bold')

            if multiple_trajectories:
                ax.legend(fontsize=12)

            ax.grid(True, alpha=0.3)
            ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            else:
                max_g = max(np.max(rdf[pair][1]) for _, rdf in all_rdf_data if pair in rdf)
                ax.set_ylim(0, max_g * 1.1)

            output_file = f"{output_prefix}_{pair[0]}-{pair[1]}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            output_files.append(output_file)
            if verbose:
                _log(logger, 'success', f"Saved: {output_file}")

            if show:
                plt.show()
            plt.close()

    return output_files


def plot_rdf_subplots(
    rdf_data: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
    output_file: str,
    pairs: Optional[List[Tuple[str, str]]] = None,
    xlims: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
    ylims: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
    linewidth: float = 2.5,
    show_errorbar: bool = True,
    fill_alpha: float = 0.3,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Create a single figure with subplots for multiple RDF pairs.

    Parameters
    ----------
    rdf_data : dict
        {(type1, type2): (r, g_r, std_error)}
    output_file : str
        Output filename
    pairs : list, optional
        Pairs to plot (default: all)
    xlims : dict, optional
        X-axis limits per pair
    ylims : dict, optional
        Y-axis limits per pair
    linewidth : float
        Line width
    show_errorbar : bool
        Show error bands
    fill_alpha : float
        Error band transparency
    show : bool
        Show interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_file : str
        Saved file path
    """
    if pairs is None:
        pairs = sorted(rdf_data.keys())

    n_pairs = len(pairs)
    if n_pairs == 0:
        return None

    # Create subplot layout
    if n_pairs <= 3:
        fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
        if n_pairs == 1:
            axes = [axes]
    else:
        ncols = 3
        nrows = (n_pairs + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
        axes = axes.flatten()

    pair_colors = {
        ('O', 'O'): '#1f77b4',
        ('O', 'H'): '#d62728',
        ('H', 'H'): '#2ca02c',
    }

    for idx, pair in enumerate(pairs):
        ax = axes[idx]

        if pair not in rdf_data:
            ax.set_visible(False)
            continue

        r, g_r, std_error = rdf_data[pair]
        pair_label = f"{pair[0]}-{pair[1]}"
        color = pair_colors.get(pair, '#1f77b4')

        ax.plot(r, g_r, linewidth=linewidth, color=color, label=pair_label)

        if show_errorbar and std_error is not None:
            ax.fill_between(r, g_r - std_error, g_r + std_error,
                            color=color, alpha=fill_alpha, linewidth=0)

        ax.set_xlabel('r (Å)', fontsize=12)
        ax.set_ylabel('g(r)', fontsize=12)
        ax.set_title(f'{pair_label} RDF', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

        if xlims and pair in xlims:
            ax.set_xlim(xlims[pair])
        if ylims and pair in ylims:
            ax.set_ylim(ylims[pair])

    # Hide unused subplots
    for idx in range(len(pairs), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()
    plt.close()

    return output_file
