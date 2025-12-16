"""
Hydrogen bond plotting functions.
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


def plot_hbond_profile(
    bin_centers: np.ndarray,
    hb_per_bin: np.ndarray,
    std_error: Optional[np.ndarray] = None,
    output_file: str = 'hbond_profile.png',
    axis_label: str = 'z',
    ylabel: str = 'H-bonds per donor',
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    linewidth: float = 2.5,
    color: str = '#1f77b4',
    show_errorbar: bool = True,
    fill_alpha: float = 0.3,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Plot H-bond spatial profile.

    Parameters
    ----------
    bin_centers : ndarray
        Bin center positions
    hb_per_bin : ndarray
        H-bonds per bin (or per donor per bin)
    std_error : ndarray, optional
        Standard error for error bands
    output_file : str
        Output filename
    axis_label : str
        Axis label ('x', 'y', or 'z')
    ylabel : str
        Y-axis label
    title : str, optional
        Plot title
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    linewidth : float
        Line width
    color : str
        Line color
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
    output_file : str
        Saved file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(bin_centers, hb_per_bin, linewidth=linewidth, color=color, label='H-bonds')

    if show_errorbar and std_error is not None:
        ax.fill_between(bin_centers, hb_per_bin - std_error, hb_per_bin + std_error,
                        color=color, alpha=fill_alpha, linewidth=0)

    ax.set_xlabel(f'{axis_label} (Å)', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    else:
        ax.set_title(f'Hydrogen Bonds vs {axis_label}', fontsize=16, fontweight='bold')

    ax.grid(True, alpha=0.3)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()
    plt.close()

    return output_file


def plot_hbond_timeseries(
    timesteps: np.ndarray,
    hbonds: np.ndarray,
    output_file: str = 'hbond_timeseries.png',
    ylabel: str = 'H-bonds per molecule',
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    linewidth: float = 1.5,
    color: str = '#1f77b4',
    show_mean: bool = True,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Plot H-bond time series.

    Parameters
    ----------
    timesteps : ndarray
        Timestep values
    hbonds : ndarray
        H-bonds per frame
    output_file : str
        Output filename
    ylabel : str
        Y-axis label
    title : str, optional
        Plot title
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    linewidth : float
        Line width
    color : str
        Line color
    show_mean : bool
        Show mean line
    show : bool
        Show plot interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_file : str
        Saved file path
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(timesteps, hbonds, linewidth=linewidth, color=color, alpha=0.7)

    if show_mean:
        mean_val = np.mean(hbonds)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_val:.2f}')
        ax.legend(fontsize=12)

    ax.set_xlabel('Timestep', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    else:
        ax.set_title('Hydrogen Bonds vs Time', fontsize=16, fontweight='bold')

    ax.grid(True, alpha=0.3)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()
    plt.close()

    return output_file


def plot_hbond_histogram(
    hbonds_per_frame: np.ndarray,
    output_file: str = 'hbond_histogram.png',
    n_bins: int = 30,
    xlabel: str = 'Number of H-bonds',
    title: Optional[str] = None,
    color: str = '#1f77b4',
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Plot histogram of H-bond counts.

    Parameters
    ----------
    hbonds_per_frame : ndarray
        H-bonds per frame
    output_file : str
        Output filename
    n_bins : int
        Number of histogram bins
    xlabel : str
        X-axis label
    title : str, optional
        Plot title
    color : str
        Bar color
    show : bool
        Show plot interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_file : str
        Saved file path
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(hbonds_per_frame, bins=n_bins, color=color, edgecolor='black', alpha=0.7)

    mean_val = np.mean(hbonds_per_frame)
    std_val = np.std(hbonds_per_frame, ddof=1)
    ax.axvline(x=mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.2f} ± {std_val:.2f}')

    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel('Count', fontsize=14, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    else:
        ax.set_title('H-bond Distribution', fontsize=16, fontweight='bold')

    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()
    plt.close()

    return output_file


def plot_hbond_lifetime(
    time: np.ndarray,
    phi_t: np.ndarray,
    tau: float,
    output_file: str = 'hbond_lifetime.png',
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    linewidth: float = 2.5,
    color: str = '#1f77b4',
    show_fit: bool = True,
    show: bool = False,
    verbose: bool = True,
    use_log_scale: Optional[bool] = None,
    logger: Optional[Any] = None
) -> str:
    """
    Plot H-bond autocorrelation function and fitted decay.

    Parameters
    ----------
    time : ndarray
        Time values (ps)
    phi_t : ndarray
        Autocorrelation function Φ(t)
    tau : float
        Fitted lifetime (ps)
    output_file : str
        Output filename
    title : str, optional
        Plot title
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    linewidth : float
        Line width
    color : str
        Line color
    show_fit : bool
        Show fitted exponential
    show : bool
        Show interactively
    verbose : bool
        Print progress
    use_log_scale : bool, optional
        Use log scale for y-axis. If None, auto-detect based on decay rate.

    Returns
    -------
    output_file : str
        Saved file path
    """
    # Detect if we need log scale (fast decay: phi drops below 0.1 at t=1)
    if use_log_scale is None:
        use_log_scale = len(phi_t) > 1 and phi_t[1] < 0.1

    # Determine time units based on tau value (use fs for small lifetimes)
    if not np.isnan(tau) and tau < 0.01:  # tau < 10 fs, use fs units
        time_plot = time * 1000  # Convert to fs
        time_unit = 'fs'
        tau_plot = tau * 1000
    else:
        time_plot = time
        time_unit = 'ps'
        tau_plot = tau if not np.isnan(tau) else np.nan

    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out zero/negative values for log scale
    if use_log_scale:
        valid = phi_t > 0
        ax.semilogy(time_plot[valid], phi_t[valid], 'o', markersize=4, color=color, alpha=0.6, label='Data')
    else:
        ax.plot(time_plot, phi_t, 'o', markersize=4, color=color, alpha=0.6, label='Data')

    if show_fit and not np.isnan(tau):
        # Generate fit curve with more points for smooth line
        t_fit = np.linspace(0, time[-1], 200)
        fit_curve = np.exp(-t_fit / tau)
        t_fit_plot = t_fit * 1000 if time_unit == 'fs' else t_fit

        if use_log_scale:
            valid_fit = fit_curve > 0
            ax.semilogy(t_fit_plot[valid_fit], fit_curve[valid_fit], '-', linewidth=linewidth, color='red',
                        label=f'Fit: τ = {tau_plot:.2f} {time_unit}')
        else:
            ax.plot(t_fit_plot, fit_curve, '-', linewidth=linewidth, color='red',
                    label=f'Fit: τ = {tau_plot:.2f} {time_unit}')

    ax.set_xlabel(f'Time ({time_unit})', fontsize=14, fontweight='bold')
    ax.set_ylabel('Φ(t)', fontsize=14, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    else:
        ax.set_title('H-bond Autocorrelation Function', fontsize=16, fontweight='bold')

    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which='both')

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    elif not use_log_scale:
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()
    plt.close()

    return output_file


def plot_hbond_lifetime_profile(
    bin_centers: np.ndarray,
    tau_per_bin: np.ndarray,
    output_file: str = 'hbond_lifetime_profile.png',
    axis_label: str = 'z',
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    linewidth: float = 2.5,
    color: str = '#1f77b4',
    marker: str = 'o',
    markersize: float = 8,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Plot H-bond lifetime vs spatial position.

    Parameters
    ----------
    bin_centers : ndarray
        Bin center positions
    tau_per_bin : ndarray
        Lifetime per bin (ps)
    output_file : str
        Output filename
    axis_label : str
        Axis label
    title : str, optional
        Plot title
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    linewidth : float
        Line width
    color : str
        Line color
    marker : str
        Marker style
    markersize : float
        Marker size
    show : bool
        Show interactively
    verbose : bool
        Print progress

    Returns
    -------
    output_file : str
        Saved file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out NaN values for plotting
    valid = ~np.isnan(tau_per_bin)

    # Determine units based on data range
    valid_taus = tau_per_bin[valid]
    if len(valid_taus) > 0 and np.max(valid_taus) < 0.01:  # < 10 fs
        tau_plot = tau_per_bin * 1000  # Convert to fs
        tau_unit = 'fs'
    else:
        tau_plot = tau_per_bin
        tau_unit = 'ps'

    ax.plot(bin_centers[valid], tau_plot[valid], marker=marker, linestyle='-',
            linewidth=linewidth, markersize=markersize, color=color)

    ax.set_xlabel(f'{axis_label} (Å)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'τ ({tau_unit})', fontsize=14, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    else:
        ax.set_title(f'H-bond Lifetime vs {axis_label}', fontsize=16, fontweight='bold')

    ax.grid(True, alpha=0.3)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()
    plt.close()

    return output_file


def plot_hbond_comparison(
    data_list: List[Tuple[str, np.ndarray, np.ndarray, Optional[np.ndarray]]],
    output_file: str = 'hbond_comparison.png',
    axis_label: str = 'z',
    ylabel: str = 'H-bonds per donor',
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    linewidth: float = 2.5,
    colors: Optional[List[str]] = None,
    show_errorbar: bool = True,
    fill_alpha: float = 0.2,
    show: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> str:
    """
    Compare H-bond profiles from multiple trajectories.

    Parameters
    ----------
    data_list : list of tuple
        List of (label, bin_centers, hb_per_bin, std_error) tuples
    output_file : str
        Output filename
    axis_label : str
        Axis label
    ylabel : str
        Y-axis label
    title : str, optional
        Plot title
    xlim : tuple, optional
        X-axis limits
    ylim : tuple, optional
        Y-axis limits
    linewidth : float
        Line width
    colors : list, optional
        Custom colors
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
    fig, ax = plt.subplots(figsize=(10, 6))

    if colors is None:
        colors = TRAJECTORY_COLORS

    linestyles = ['-', '--', '-.', ':']

    for idx, (label, bin_centers, hb_per_bin, std_error) in enumerate(data_list):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]

        ax.plot(bin_centers, hb_per_bin, linewidth=linewidth, color=color,
                linestyle=linestyle, label=label)

        if show_errorbar and std_error is not None:
            ax.fill_between(bin_centers, hb_per_bin - std_error, hb_per_bin + std_error,
                            color=color, alpha=fill_alpha, linewidth=0)

    ax.set_xlabel(f'{axis_label} (Å)', fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    else:
        ax.set_title(f'Hydrogen Bonds vs {axis_label}', fontsize=16, fontweight='bold')

    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    if verbose:
        _log(logger, 'success', f"Saved: {output_file}")

    if show:
        plt.show()
    plt.close()

    return output_file
