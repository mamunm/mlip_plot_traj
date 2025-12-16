"""Hydrogen bond analysis command."""

import click
import csv
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from .. import logger
from .common import get_output_prefix, create_spinner_context, create_progress_context, console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn


@click.command()
@click.argument('trajectory', type=click.Path(exists=True))
@click.option('--donor', '-d', required=True, type=str,
              help='Donor element (e.g., O for water)')
@click.option('--hydrogen', '-h', required=True, type=str,
              help='Hydrogen element (e.g., H)')
@click.option('--acceptor', '-a', required=True, type=str,
              help='Acceptor element (e.g., O for water)')
@click.option('--d-a-cutoff', default=3.5, type=float,
              help='Donor-Acceptor distance cutoff in Angstroms (default: 3.5)')
@click.option('--angle-cutoff', default=120.0, type=float,
              help='D-H...A angle cutoff in degrees (default: 120)')
@click.option('--d-h-cutoff', default=1.3, type=float,
              help='Donor-Hydrogen bond distance in Angstroms (default: 1.3)')
@click.option('--skip-frames', default=0, type=int,
              help='Number of initial frames to skip (equilibration)')
@click.option('--skip', default=1, type=int,
              help='Process every N-th frame (default: 1)')
@click.option('--profile', is_flag=True,
              help='Compute spatial profile of H-bonds')
@click.option('--axis', default='z', type=click.Choice(['x', 'y', 'z']),
              help='Axis for spatial profile (default: z)')
@click.option('--n-bins', default=50, type=int,
              help='Number of spatial bins for profile (default: 50)')
@click.option('--bin-range', nargs=2, type=float, default=None,
              help='Range for spatial bins (min max)')
@click.option('--n-blocks', default=None, type=int,
              help='Number of blocks for error estimation')
@click.option('--normalize/--no-normalize', default=True,
              help='Normalize H-bonds by number of donors (default: on)')
@click.option('--timeseries', is_flag=True,
              help='Plot H-bonds as time series')
@click.option('--histogram', is_flag=True,
              help='Plot histogram of H-bond counts')
@click.option('--lifetime', is_flag=True,
              help='Compute H-bond lifetime via autocorrelation')
@click.option('--lifetime-profile', is_flag=True,
              help='Compute H-bond lifetime as function of position')
@click.option('--lifetime-bins', default=10, type=int,
              help='Number of bins for lifetime profile (default: 10)')
@click.option('--frame-interval', default=0.001, type=float,
              help='Time between frames in ps (default: 0.001 = 1 fs)')
@click.option('--output', '-o', default=None, type=str,
              help='Output prefix (default: trajectory filename)')
@click.option('--save-folder', default='analysis', type=str,
              help='Folder to save plots and data (default: analysis)')
@click.option('--xlim', nargs=2, type=float, default=None,
              help='X-axis limits (min max)')
@click.option('--ylim', nargs=2, type=float, default=None,
              help='Y-axis limits (min max)')
@click.option('--linewidth', default=2.5, type=float,
              help='Line width (default: 2.5)')
@click.option('--errorbar/--no-errorbar', default=True,
              help='Show error bands (default: on)')
@click.option('--fill-alpha', default=0.3, type=float,
              help='Error band transparency (default: 0.3)')
@click.option('--export-csv', is_flag=True,
              help='Export H-bond data to CSV')
@click.option('--no-plot', is_flag=True,
              help='Compute only, no plotting')
@click.option('--show', is_flag=True,
              help='Show plots interactively')
def hbond(
    trajectory: str,
    donor: str,
    hydrogen: str,
    acceptor: str,
    d_a_cutoff: float,
    angle_cutoff: float,
    d_h_cutoff: float,
    skip_frames: int,
    skip: int,
    profile: bool,
    axis: str,
    n_bins: int,
    bin_range: Optional[Tuple[float, float]],
    n_blocks: Optional[int],
    normalize: bool,
    timeseries: bool,
    histogram: bool,
    lifetime: bool,
    lifetime_profile: bool,
    lifetime_bins: int,
    frame_interval: float,
    output: Optional[str],
    save_folder: str,
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
    linewidth: float,
    errorbar: bool,
    fill_alpha: float,
    export_csv: bool,
    no_plot: bool,
    show: bool
):
    """
    Detect and analyze hydrogen bonds in trajectory.

    H-bond criteria:
    - D-A distance < d_a_cutoff (default: 3.5 Å)
    - D-H...A angle > angle_cutoff (default: 120°)

    \b
    Examples:
      mlip-plot hbond trajectory.lammpstrj -d O -h H -a O
      mlip-plot hbond trajectory.lammpstrj -d O -h H -a O --profile --n-blocks 5
      mlip-plot hbond trajectory.lammpstrj -d O -h H -a O --lifetime --frame-interval 0.01
      mlip-plot hbond trajectory.lammpstrj -d O -h H -a O --lifetime-profile --lifetime-bins 5

    \b
    References:
      - Distance cutoff: DOI 10.1039/B808326F, DOI 10.1039/C3CP52271G
      - Angle cutoff: DOI 10.1039/B902330E
    """
    from ..io.lammps import read_lammpstrj
    from ..analysis.hbond import (
        compute_hbonds, compute_hbonds_profile, compute_hbonds_timeseries,
        compute_hbond_lifetime, compute_hbond_lifetime_profile
    )
    from ..plotting.hbond import (
        plot_hbond_profile, plot_hbond_timeseries, plot_hbond_histogram,
        plot_hbond_lifetime, plot_hbond_lifetime_profile
    )

    # Set output prefix
    output_prefix = get_output_prefix(save_folder, output, trajectory)

    # Print header
    logger.header("MLIP Plot", "Hydrogen Bond Analysis")

    # Configuration table
    config = {
        "Trajectory": str(trajectory),
        "Output folder": f"[cyan]{save_folder}/[/cyan]",
        "H-bond triplet": f"{donor}-{hydrogen}...{acceptor}",
        "D-A cutoff": f"{d_a_cutoff:.2f} Å",
        "Angle cutoff": f"{angle_cutoff:.1f}°",
        "D-H cutoff": f"{d_h_cutoff:.2f} Å",
        "Skip frames": str(skip_frames),
        "Frame skip": f"every {skip}" if skip > 1 else "all frames",
        "Block averaging": f"[green]Yes ({n_blocks} blocks)[/green]" if n_blocks else "[dim]No[/dim]",
    }
    logger.print_table(logger.config_table(config))

    # Read trajectory
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Reading trajectory...", total=None)
        frames = read_lammpstrj(trajectory, skip_frames=skip_frames, verbose=False)
        progress.remove_task(task)

    if not frames:
        logger.error("No frames available for analysis")
        raise SystemExit(1)

    logger.success(f"Loaded trajectory: {len(frames)} frames")

    # Show available elements
    if frames[0]['elements'] is not None:
        available_elements = sorted(set(frames[0]['elements']))
        logger.detail(f"Elements found: {', '.join(available_elements)}")

    output_files = []

    # Compute basic H-bond statistics
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Detecting H-bonds...", total=None)

        hbonds_per_frame, mean_hbonds, std_hbonds = compute_hbonds(
            frames, donor, hydrogen, acceptor,
            d_a_cutoff=d_a_cutoff, angle_cutoff=angle_cutoff, d_h_cutoff=d_h_cutoff,
            skip=skip, verbose=False
        )

        progress.remove_task(task)

    # H-bond summary
    summary_table = logger.results_table("H-bond Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")
    summary_table.add_row("Total H-bonds", f"{int(np.sum(hbonds_per_frame))}")
    summary_table.add_row("Mean per frame", f"{mean_hbonds:.2f} ± {std_hbonds:.2f}")
    summary_table.add_row("Min per frame", f"{int(np.min(hbonds_per_frame))}")
    summary_table.add_row("Max per frame", f"{int(np.max(hbonds_per_frame))}")
    logger.print_table(summary_table)

    # If no specific plot type requested, default to profile + lifetime + lifetime-profile
    if not profile and not timeseries and not histogram and not lifetime and not lifetime_profile:
        profile = True
        lifetime = True
        lifetime_profile = True

    # Compute spatial profile if requested
    if profile:
        logger.section("Computing spatial profile")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Computing H-bond profile along {axis}...", total=None)

            bin_centers, hb_per_bin, std_error = compute_hbonds_profile(
                frames, donor, hydrogen, acceptor,
                axis=axis, n_bins=n_bins, bin_range=bin_range,
                d_a_cutoff=d_a_cutoff, angle_cutoff=angle_cutoff, d_h_cutoff=d_h_cutoff,
                n_blocks=n_blocks, skip=skip, normalize_by_donors=normalize, verbose=False
            )

            progress.remove_task(task)

        valid_bins = hb_per_bin > 0
        if np.any(valid_bins):
            logger.success(f"Profile computed: mean H-bonds/donor = {np.mean(hb_per_bin[valid_bins]):.3f}")

        if not no_plot:
            ylabel = "H-bonds per donor" if normalize else "H-bonds per bin"
            output_file = f"{output_prefix}_hbond_profile.png"
            plot_hbond_profile(
                bin_centers, hb_per_bin, std_error,
                output_file=output_file,
                axis_label=axis,
                ylabel=ylabel,
                xlim=xlim, ylim=ylim,
                linewidth=linewidth,
                show_errorbar=errorbar,
                fill_alpha=fill_alpha,
                show=show,
                verbose=False
            )
            logger.success(f"Saved: [bold]{output_file}[/bold]")
            output_files.append(output_file)

        if export_csv:
            csv_file = f"{output_prefix}_hbond_profile.csv"
            _export_hbond_profile_csv(bin_centers, hb_per_bin, std_error, axis, csv_file)
            logger.success(f"Saved: [bold]{csv_file}[/bold]")

    # Compute time series if requested
    if timeseries:
        logger.section("Computing time series")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Computing H-bond time series...", total=None)

            timesteps, hbonds_ts = compute_hbonds_timeseries(
                frames, donor, hydrogen, acceptor,
                d_a_cutoff=d_a_cutoff, angle_cutoff=angle_cutoff, d_h_cutoff=d_h_cutoff,
                skip=skip, normalize_by_molecules=normalize, verbose=False
            )

            progress.remove_task(task)

        logger.success(f"Time series computed: {len(timesteps)} points")

        if not no_plot:
            ylabel = "H-bonds per molecule" if normalize else "H-bonds"
            output_file = f"{output_prefix}_hbond_timeseries.png"
            plot_hbond_timeseries(
                timesteps, hbonds_ts,
                output_file=output_file,
                ylabel=ylabel,
                xlim=xlim, ylim=ylim,
                linewidth=1.5,
                show=show,
                verbose=False
            )
            logger.success(f"Saved: [bold]{output_file}[/bold]")
            output_files.append(output_file)

        if export_csv:
            csv_file = f"{output_prefix}_hbond_timeseries.csv"
            _export_hbond_timeseries_csv(timesteps, hbonds_ts, normalize, csv_file)
            logger.success(f"Saved: [bold]{csv_file}[/bold]")

    # Create histogram if requested
    if histogram:
        logger.section("Creating histogram")

        if not no_plot:
            output_file = f"{output_prefix}_hbond_histogram.png"
            plot_hbond_histogram(
                hbonds_per_frame,
                output_file=output_file,
                show=show,
                verbose=False
            )
            logger.success(f"Saved: [bold]{output_file}[/bold]")
            output_files.append(output_file)

    # Compute lifetime if requested
    if lifetime:
        logger.section("Computing H-bond lifetime")

        # Calculate max_lag for progress tracking
        frames_to_use = frames[::skip]
        max_lag = int(len(frames_to_use) * 0.5)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            detect_task = progress.add_task("[cyan]Detecting H-bond pairs...", total=len(frames_to_use))
            autocorr_task = progress.add_task("[cyan]Computing autocorrelation...", total=max_lag, visible=False)

            def on_detect_progress(n):
                progress.update(detect_task, completed=n)
                if n >= len(frames_to_use):
                    progress.update(detect_task, visible=False)
                    progress.update(autocorr_task, visible=True)

            def on_autocorr_progress(n):
                progress.update(autocorr_task, completed=n)

            time_arr, phi_t, tau = compute_hbond_lifetime(
                frames, donor, hydrogen, acceptor,
                d_a_cutoff=d_a_cutoff, angle_cutoff=angle_cutoff, d_h_cutoff=d_h_cutoff,
                frame_interval=frame_interval, skip=skip, verbose=False,
                progress_callback=on_detect_progress,
                autocorr_callback=on_autocorr_progress
            )

            progress.update(autocorr_task, completed=max_lag)

        if not np.isnan(tau):
            if tau < 0.01:
                logger.success(f"H-bond lifetime τ = {tau*1000:.2f} fs ({tau:.4f} ps)")
            else:
                logger.success(f"H-bond lifetime τ = {tau:.2f} ps")
        else:
            logger.warning("Could not fit lifetime")

        if not no_plot:
            output_file = f"{output_prefix}_hbond_lifetime.png"
            plot_hbond_lifetime(
                time_arr, phi_t, tau,
                output_file=output_file,
                show=show,
                verbose=False
            )
            logger.success(f"Saved: [bold]{output_file}[/bold]")
            output_files.append(output_file)

        if export_csv:
            csv_file = f"{output_prefix}_hbond_autocorr.csv"
            _export_hbond_autocorr_csv(time_arr, phi_t, tau, csv_file)
            logger.success(f"Saved: [bold]{csv_file}[/bold]")

    # Compute lifetime profile if requested
    if lifetime_profile:
        logger.section("Computing H-bond lifetime profile")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"[cyan]Computing lifetime vs {axis}...", total=None)

            bin_centers_lt, tau_per_bin, autocorr_data = compute_hbond_lifetime_profile(
                frames, donor, hydrogen, acceptor,
                axis=axis, n_bins=lifetime_bins, bin_range=bin_range,
                d_a_cutoff=d_a_cutoff, angle_cutoff=angle_cutoff, d_h_cutoff=d_h_cutoff,
                frame_interval=frame_interval, skip=skip, verbose=False
            )

            progress.remove_task(task)

        # Show results table
        valid_taus = tau_per_bin[~np.isnan(tau_per_bin)]
        use_fs = len(valid_taus) > 0 and np.max(valid_taus) < 0.01

        lt_table = logger.results_table("H-bond Lifetime Profile")
        lt_table.add_column(f"{axis} (Å)", style="cyan", justify="right")
        lt_table.add_column("τ (fs)" if use_fs else "τ (ps)", justify="right")

        for i in range(len(bin_centers_lt)):
            if np.isnan(tau_per_bin[i]):
                tau_str = "N/A"
            elif use_fs:
                tau_str = f"{tau_per_bin[i]*1000:.2f}"
            else:
                tau_str = f"{tau_per_bin[i]:.2f}"
            lt_table.add_row(f"{bin_centers_lt[i]:.1f}", tau_str)

        logger.print_table(lt_table)

        if not no_plot:
            output_file = f"{output_prefix}_hbond_lifetime_profile.png"
            plot_hbond_lifetime_profile(
                bin_centers_lt, tau_per_bin,
                output_file=output_file,
                axis_label=axis,
                xlim=xlim, ylim=ylim,
                linewidth=linewidth,
                show=show,
                verbose=False
            )
            logger.success(f"Saved: [bold]{output_file}[/bold]")
            output_files.append(output_file)

        if export_csv:
            csv_file = f"{output_prefix}_hbond_lifetime_profile.csv"
            _export_hbond_lifetime_csv(bin_centers_lt, tau_per_bin, axis, csv_file)
            logger.success(f"Saved: [bold]{csv_file}[/bold]")

    logger.complete()


def _export_hbond_profile_csv(bin_centers: np.ndarray, hb_per_bin: np.ndarray,
                               std_error: Optional[np.ndarray], axis: str, output_file: str):
    """Export H-bond profile to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        if std_error is not None:
            writer.writerow([f'{axis}_angstrom', 'hbonds_per_donor', 'stderr'])
            for i in range(len(bin_centers)):
                writer.writerow([f'{bin_centers[i]:.4f}', f'{hb_per_bin[i]:.6f}', f'{std_error[i]:.6f}'])
        else:
            writer.writerow([f'{axis}_angstrom', 'hbonds_per_donor'])
            for i in range(len(bin_centers)):
                writer.writerow([f'{bin_centers[i]:.4f}', f'{hb_per_bin[i]:.6f}'])


def _export_hbond_timeseries_csv(timesteps: np.ndarray, hbonds: np.ndarray,
                                  normalized: bool, output_file: str):
    """Export H-bond time series to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        col_name = 'hbonds_per_molecule' if normalized else 'hbonds'
        writer.writerow(['timestep', col_name])
        for i in range(len(timesteps)):
            writer.writerow([f'{timesteps[i]}', f'{hbonds[i]:.6f}'])


def _export_hbond_autocorr_csv(time: np.ndarray, phi_t: np.ndarray,
                                tau: float, output_file: str):
    """Export H-bond autocorrelation to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['# Fitted lifetime tau (ps):', f'{tau:.4f}' if not np.isnan(tau) else 'N/A'])
        writer.writerow(['time_ps', 'phi_t'])
        for i in range(len(time)):
            writer.writerow([f'{time[i]:.4f}', f'{phi_t[i]:.6f}'])


def _export_hbond_lifetime_csv(bin_centers: np.ndarray, tau_per_bin: np.ndarray,
                                axis: str, output_file: str):
    """Export H-bond lifetime profile to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'{axis}_angstrom', 'tau_ps'])
        for i in range(len(bin_centers)):
            tau_str = f'{tau_per_bin[i]:.4f}' if not np.isnan(tau_per_bin[i]) else 'NaN'
            writer.writerow([f'{bin_centers[i]:.4f}', tau_str])
