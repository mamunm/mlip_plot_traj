"""Density profile command."""

import click
import csv
from pathlib import Path
from typing import Optional, Tuple, Union

from .. import logger
from .common import (
    parse_n_blocks, format_n_blocks_config,
    parse_skip_frames, format_skip_frames_config,
    console
)
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


@click.command()
@click.argument('trajectories', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--skip-frames', default=None, type=str, callback=parse_skip_frames,
              help='Frames to skip: integer or fraction (e.g., 0.1 for 10%%). Default: 0.1')
@click.option('--n-blocks', default=5, type=str, callback=parse_n_blocks,
              help='Error estimation: integer for fixed blocks, or "autocorr", "FlyPet", "convergence". Default: 5')
@click.option('--bin-size', default=0.5, type=float,
              help='Bin size in Angstroms (default: 0.5)')
@click.option('--axis', default='z', type=click.Choice(['x', 'y', 'z']),
              help='Axis for density profile (default: z)')
@click.option('--exclude', '-e', multiple=True, default=['Cu', 'Pt'],
              help='Elements to exclude (default: Cu, Pt)')
@click.option('--output', '-o', default=None, type=str,
              help='Output prefix (default: trajectory filename)')
@click.option('--label', '-l', multiple=True, default=None,
              help='Labels for each trajectory')
@click.option('--xlim', nargs=2, type=float, default=None,
              help='X-axis limits (min max)')
@click.option('--ylim', nargs=2, type=float, default=None,
              help='Y-axis limits (min max)')
@click.option('--show', is_flag=True,
              help='Show plots interactively')
def density(
    trajectories: Tuple[str, ...],
    skip_frames: Optional[Union[int, float]],
    n_blocks: Optional[Union[int, str]],
    bin_size: float,
    axis: str,
    exclude: Tuple[str, ...],
    output: Optional[str],
    label: Tuple[str, ...],
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
    show: bool
):
    """
    Calculate and plot density profiles from LAMMPS trajectory files.

    \b
    Examples:
      mlip-plot density trajectory.lammpstrj
      mlip-plot density traj1.lammpstrj traj2.lammpstrj -l "System A" -l "System B"
      mlip-plot density trajectory.lammpstrj --n-blocks autocorr
      mlip-plot density trajectory.lammpstrj --skip-frames 0.2 --bin-size 0.25
    """
    from ..io.lammps import read_lammpstrj
    from ..analysis.density import calculate_density_profile
    from ..plotting.density import plot_density_profiles, plot_flypet_diagnostics

    # Validate inputs
    labels = list(label) if label else None
    if labels and len(labels) != len(trajectories):
        logger.error(f"Number of labels ({len(labels)}) must match trajectories ({len(trajectories)})")
        raise SystemExit(1)

    # Fixed save folder for density analysis
    save_folder = 'analysis/density'
    save_path = Path(save_folder)
    # Clear existing files before new run
    if save_path.exists():
        for f in save_path.iterdir():
            if f.is_file():
                f.unlink()
    save_path.mkdir(parents=True, exist_ok=True)

    # Set output prefix
    if output is None:
        if len(trajectories) == 1:
            output_prefix = save_path / Path(trajectories[0]).stem
        else:
            output_prefix = save_path / "combined"
    else:
        output_prefix = save_path / output
    output_prefix = str(output_prefix)

    # Generate labels if not provided
    if labels is None:
        labels = [Path(f).stem for f in trajectories]

    # Convert axis to index
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map[axis]

    exclude_set = set(exclude)

    # Print header
    logger.header("MLIP Plot", "Density Profile Analysis")

    # Configuration table
    config = {
        "Trajectories": str(len(trajectories)),
        "Output folder": f"[cyan]{save_folder}/[/cyan]",
        "Skip frames": format_skip_frames_config(skip_frames),
        "Error estimation": format_n_blocks_config(n_blocks),
        "Bin size": f"{bin_size} Å",
        "Axis": axis,
        "Excluded elements": ", ".join(exclude) if exclude else "[dim]None[/dim]",
    }
    logger.print_table(logger.config_table(config))

    # Process all trajectories
    all_profiles = []
    all_diagnostics = []
    frame_counts = []

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, style="dim", complete_style="cyan", finished_style="green"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:

        for idx, (traj_file, traj_label) in enumerate(zip(trajectories, labels)):
            # Read trajectory
            task = progress.add_task(f"Reading {traj_label}...", total=None)

            # Handle skip_frames: int or float (fraction)
            if isinstance(skip_frames, float):
                # Fraction: first read to get total frame count, then compute skip
                all_frames = read_lammpstrj(traj_file, skip_frames=0, verbose=False)
                if not all_frames:
                    progress.remove_task(task)
                    progress.stop()
                    logger.error(f"No frames available for analysis in {traj_file}")
                    raise SystemExit(1)
                actual_skip = int(len(all_frames) * skip_frames)
                frames = all_frames[actual_skip:]
            else:
                # Integer: direct skip
                actual_skip = skip_frames if skip_frames else 0
                frames = read_lammpstrj(traj_file, skip_frames=actual_skip, verbose=False)

            progress.remove_task(task)

            if not frames:
                progress.stop()
                logger.error(f"No frames available for analysis in {traj_file}")
                raise SystemExit(1)

            frame_counts.append((traj_label, len(frames)))

            # Calculate density profiles
            if isinstance(n_blocks, str):
                task = progress.add_task(f"Computing density for {traj_label} ({n_blocks})...", total=None)
            else:
                task = progress.add_task(f"Computing density for {traj_label}...", total=None)

            profiles, diagnostics = calculate_density_profile(
                frames,
                bin_size=bin_size,
                axis=axis_idx,
                exclude_elements=exclude_set,
                n_blocks=n_blocks
            )
            progress.remove_task(task)

            if not profiles:
                progress.stop()
                logger.error(f"No density profiles calculated for {traj_file}")
                raise SystemExit(1)

            all_profiles.append((traj_label, profiles))
            all_diagnostics.append((traj_label, diagnostics))

    # Show results
    for traj_label, count in frame_counts:
        logger.success(f"Loaded [bold]{traj_label}[/bold]: {count} frames")

    for traj_label, profiles in all_profiles:
        density_table = logger.results_table(f"Density Summary: {traj_label}")
        density_table.add_column("Element", style="cyan")
        density_table.add_column("Max Density", justify="right")
        density_table.add_column("Unit", style="dim")

        for elem in sorted(profiles.keys()):
            z, dens, stderr = profiles[elem]
            max_dens = dens.max()
            density_table.add_row(elem, f"{max_dens:.4f}", "g/cm³")

        logger.print_table(density_table)

    # Create plots
    logger.section("Creating plots")
    output_files = plot_density_profiles(
        all_profiles,
        output_prefix=output_prefix,
        xlim=xlim,
        ylim=ylim,
        labels=labels,
        show_errorbar=True,
        fill_alpha=0.3,
        show=show,
        verbose=False
    )
    logger.success(f"Saved {len(output_files)} plot(s) to [bold]{save_folder}/[/bold]")

    # Create FlyPet diagnostic plots if using FlyPet method
    if n_blocks == 'FlyPet':
        for traj_label, diagnostics in all_diagnostics:
            if not diagnostics:
                continue
            if len(all_diagnostics) == 1:
                diag_prefix = output_prefix
            else:
                diag_prefix = f"{output_prefix}_{traj_label.replace(' ', '_')}"
            diag_files = plot_flypet_diagnostics(diagnostics, output_prefix=diag_prefix, show=show)
            if diag_files:
                logger.success(f"Saved FlyPet diagnostic plot: [bold]{diag_files[0]}[/bold]")

    # Export CSV (always)
    logger.section("Exporting CSV")
    if len(all_profiles) == 1:
        csv_file = f"{output_prefix}_density.csv"
        _export_density_csv(all_profiles[0][1], csv_file)
        logger.success(f"Saved: [bold]{csv_file}[/bold]")
    else:
        for traj_label, profiles in all_profiles:
            csv_file = f"{output_prefix}_{traj_label.replace(' ', '_')}_density.csv"
            _export_density_csv(profiles, csv_file)
            logger.success(f"Saved: [bold]{csv_file}[/bold]")

    logger.complete()


def _export_density_csv(profiles: dict, output_file: str):
    """Export density profiles to CSV."""
    z_coords = next(iter(profiles.values()))[0]
    has_errors = next(iter(profiles.values()))[2] is not None
    elements = sorted(profiles.keys())

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['z_angstrom']
        for e in elements:
            header.append(f'{e}_density_g_cm3')
            if has_errors:
                header.append(f'{e}_stderr_g_cm3')
        writer.writerow(header)

        # Data
        for i, z in enumerate(z_coords):
            row = [f'{z:.4f}']
            for element in elements:
                density = profiles[element][1][i]
                row.append(f'{density:.6f}')
                if has_errors:
                    std_err = profiles[element][2][i]
                    row.append(f'{std_err:.6f}')
            writer.writerow(row)
