"""Hydrogen bond analysis command."""

import click
import csv
from pathlib import Path
from typing import Optional, Tuple, Union

from .. import logger
from .common import (
    parse_skip_frames, format_skip_frames_config,
    console
)
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


@click.command()
@click.argument('trajectory', type=click.Path(exists=True))
@click.option('--skip-frames', default='0.1', type=str, callback=parse_skip_frames,
              help='Frames to skip: integer or fraction (e.g., 0.1 for 10%%). Default: 0.1')
@click.option('--dt', default=1.0, type=float,
              help='Time between frames in picoseconds (default: 1.0)')
@click.option('--d-a-cutoff', default=3.5, type=float,
              help='Donor-Acceptor distance cutoff in Angstroms (default: 3.5)')
@click.option('--angle-cutoff', default=120.0, type=float,
              help='D-H-A angle cutoff in degrees (default: 120)')
@click.option('--z-interface', default=None, type=float,
              help='Interface thickness from each surface (Angstroms). Requires --d-bulk.')
@click.option('--d-bulk', default=None, type=float,
              help='Half-width of bulk region around midpoint (Angstroms). Requires --z-interface.')
@click.option('--n-blocks', default=5, type=int,
              help='Number of blocks for error estimation (default: 5)')
@click.option('--output', '-o', default=None, type=str,
              help='Output prefix (default: trajectory filename)')
@click.option('--xlim', nargs=2, type=float, default=None,
              help='X-axis limits for time series plots (min max)')
@click.option('--ylim', nargs=2, type=float, default=None,
              help='Y-axis limits (min max)')
@click.option('--show', is_flag=True,
              help='Show plots interactively')
@click.option('--verbose', '-v', is_flag=True,
              help='Print detailed progress information')
def hbond(
    trajectory: str,
    skip_frames: Optional[Union[int, float]],
    dt: float,
    d_a_cutoff: float,
    angle_cutoff: float,
    z_interface: Optional[float],
    d_bulk: Optional[float],
    n_blocks: int,
    output: Optional[str],
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
    show: bool,
    verbose: bool
):
    """
    Analyze hydrogen bonds in water from trajectory.

    Detects water-water H-bonds (O-H...O) using geometric criteria:
    - Donor-Acceptor distance < d_a_cutoff (default: 3.5 A)
    - D-H-A angle > angle_cutoff (default: 120 degrees)

    \b
    Key metric: <nhbond> = (N_hbonds * 2) / N_oxygen
    This gives the average number of H-bonds per water molecule.
    Typical value for bulk water: ~3.5

    \b
    Region analysis (--z-interface and --d-bulk):
    Computes separate statistics for three regions:
    - Interface A: from metal surface to z_interface
    - Interface B: z_length - z_interface to z_length
    - Bulk: midpoint +/- d_bulk

    H-bonds are attributed to a region if EITHER donor OR acceptor is in that region.

    \b
    Examples:
      mlip-plot hbond trajectory.lammpstrj --dt 2.0
      mlip-plot hbond trajectory.lammpstrj --d-a-cutoff 3.2 --angle-cutoff 140
      mlip-plot hbond trajectory.lammpstrj --z-interface 20 --d-bulk 10
    """
    from ..io.lammps import read_lammpstrj
    from ..analysis.hbond import compute_hbond_analysis
    from ..plotting.hbond import (
        plot_hbond_time_series,
        plot_nhbond_distribution,
        plot_hbond_geometry,
        plot_hbond_regions_summary
    )

    # Fixed save folder for hbond analysis
    save_folder = 'analysis/hbond'
    save_path = Path(save_folder)
    # Clear existing files before new run
    if save_path.exists():
        for f in save_path.iterdir():
            if f.is_file():
                f.unlink()
    save_path.mkdir(parents=True, exist_ok=True)

    # Set output prefix
    if output is None:
        output_prefix = save_path / Path(trajectory).stem
    else:
        output_prefix = save_path / output
    output_prefix = str(output_prefix)

    # Validate region parameters
    if (z_interface is None) != (d_bulk is None):
        logger.error("--z-interface and --d-bulk must both be provided, or neither")
        raise SystemExit(1)
    use_region_analysis = z_interface is not None

    # Print header
    logger.header("MLIP Plot", "Hydrogen Bond Analysis")

    # Configuration table
    config = {
        "Trajectory": str(trajectory),
        "Output folder": f"[cyan]{save_folder}/[/cyan]",
        "Skip frames": format_skip_frames_config(skip_frames),
        "Time step": f"{dt} ps",
        "D-A cutoff": f"{d_a_cutoff} A",
        "D-H-A angle cutoff": f"{angle_cutoff} deg",
    }
    if use_region_analysis:
        config["Region analysis"] = "[bold green]Enabled[/bold green]"
        config["Interface thickness"] = f"{z_interface} A"
        config["Bulk half-width"] = f"{d_bulk} A"
    config["Block averaging"] = f"{n_blocks} blocks" if n_blocks > 1 else "Disabled"

    logger.print_table(logger.config_table(config))

    # Read trajectory
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, style="dim", complete_style="cyan", finished_style="green"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Reading trajectory...", total=None)

        # Handle skip_frames: int or float (fraction)
        if isinstance(skip_frames, float):
            all_frames = read_lammpstrj(trajectory, skip_frames=0, verbose=False)
            if not all_frames:
                progress.remove_task(task)
                progress.stop()
                logger.error("No frames available for analysis")
                raise SystemExit(1)
            actual_skip = int(len(all_frames) * skip_frames)
            frames = all_frames[actual_skip:]
        else:
            actual_skip = skip_frames if skip_frames else 0
            frames = read_lammpstrj(trajectory, skip_frames=actual_skip, verbose=False)

        progress.remove_task(task)

    if not frames:
        logger.error("No frames available for analysis")
        raise SystemExit(1)

    logger.success(f"Loaded trajectory: {len(frames)} frames")

    # Validate elements
    if frames[0]['elements'] is None:
        logger.error("Trajectory must contain element information")
        raise SystemExit(1)

    available_elements = sorted(set(frames[0]['elements']))
    logger.detail(f"Elements found: {', '.join(available_elements)}")

    # Check for water (O and H atoms)
    if 'O' not in available_elements or 'H' not in available_elements:
        logger.error("H-bond analysis requires O and H atoms in trajectory")
        raise SystemExit(1)

    # Compute H-bond analysis
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, style="dim", complete_style="cyan", finished_style="green"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Computing hydrogen bonds...", total=None)

        results = compute_hbond_analysis(
            frames, dt,
            d_a_cutoff=d_a_cutoff,
            d_h_a_angle_cutoff=angle_cutoff,
            z_interface=z_interface, d_bulk=d_bulk,
            n_blocks=n_blocks, verbose=verbose
        )

        progress.remove_task(task)

    logger.success(f"H-bond analysis complete for {results['n_water']} water molecules")

    # Results tables
    region_labels = {
        'interface_a': 'Interface A',
        'interface_b': 'Interface B',
        'bulk': 'Bulk',
        'global': 'Global'
    }

    summary = results['summary']
    regions = results['regions']

    if use_region_analysis:
        # Region-based results
        for region_name in ['interface_a', 'bulk', 'interface_b', 'global']:
            if region_name in summary:
                s = summary[region_name]

                if region_name == 'global':
                    title = f"H-bonds: {region_labels[region_name]}"
                    z_range_str = "all"
                else:
                    z_range = regions[region_name]
                    title = f"H-bonds: {region_labels[region_name]}"
                    z_range_str = f"{z_range[0]:.1f}-{z_range[1]:.1f} A"

                results_tbl = logger.results_table(title)
                results_tbl.add_column("Metric", style="cyan")
                results_tbl.add_column("Value", justify="right")

                results_tbl.add_row("Z range", z_range_str)
                results_tbl.add_row("Avg water molecules", f"{s['n_water']}")
                results_tbl.add_row("Mean N_hbonds", f"{s['mean_n_hbonds']:.1f} +/- {s['std_n_hbonds']:.1f}")
                results_tbl.add_row("<nhbond>", f"{s['mean_nhbond']:.4f} +/- {s['std_nhbond']:.4f}")

                logger.print_table(results_tbl)
    else:
        # Global-only results
        s = summary['global']
        results_tbl = logger.results_table("Hydrogen Bond Statistics")
        results_tbl.add_column("Metric", style="cyan")
        results_tbl.add_column("Value", justify="right")

        results_tbl.add_row("Water molecules", f"{s['n_water']}")
        results_tbl.add_row("Mean N_hbonds", f"{s['mean_n_hbonds']:.1f} +/- {s['std_n_hbonds']:.1f}")
        results_tbl.add_row("<nhbond>", f"{s['mean_nhbond']:.4f} +/- {s['std_nhbond']:.4f}")

        logger.print_table(results_tbl)

    # Geometry statistics
    geometry = results['geometry']
    if len(geometry['distances']) > 0:
        geo_tbl = logger.results_table("H-bond Geometry")
        geo_tbl.add_column("Parameter", style="cyan")
        geo_tbl.add_column("Mean", justify="right")
        geo_tbl.add_column("Std", justify="right")

        geo_tbl.add_row(
            "D-A distance (A)",
            f"{np.mean(geometry['distances']):.3f}",
            f"{np.std(geometry['distances']):.3f}"
        )
        geo_tbl.add_row(
            "D-H-A angle (deg)",
            f"{np.mean(geometry['angles']):.1f}",
            f"{np.std(geometry['angles']):.1f}"
        )
        logger.print_table(geo_tbl)

    # Create plots
    logger.section("Creating plots")

    # 1. Time series plot (nhbond)
    output_file = f"{output_prefix}_nhbond_timeseries.png"
    plot_hbond_time_series(
        results, output_file=output_file,
        metric='nhbond',
        xlim=xlim, ylim=ylim,
        show=show, verbose=False
    )
    logger.success(f"Saved: [bold]{output_file}[/bold]")

    # 2. Time series plot (n_hbonds)
    output_file = f"{output_prefix}_nhbonds_count_timeseries.png"
    plot_hbond_time_series(
        results, output_file=output_file,
        metric='n_hbonds',
        xlim=xlim, ylim=None,
        show=show, verbose=False
    )
    logger.success(f"Saved: [bold]{output_file}[/bold]")

    # 3. Distribution plot
    output_file = f"{output_prefix}_distribution.png"
    plot_nhbond_distribution(
        results, output_file=output_file,
        show=show, verbose=False
    )
    logger.success(f"Saved: [bold]{output_file}[/bold]")

    # 4. Geometry plot
    if len(geometry['distances']) > 0:
        output_file = f"{output_prefix}_geometry.png"
        plot_hbond_geometry(
            results, output_file=output_file,
            show=show, verbose=False
        )
        logger.success(f"Saved: [bold]{output_file}[/bold]")

    # 5. Region summary (if region analysis)
    if use_region_analysis:
        output_file = f"{output_prefix}_regions_summary.png"
        plot_hbond_regions_summary(
            results, output_file=output_file,
            show=show, verbose=False
        )
        logger.success(f"Saved: [bold]{output_file}[/bold]")

    # Export CSV
    logger.section("Exporting CSV")

    csv_file = f"{output_prefix}_timeseries.csv"
    _export_timeseries_csv(results, csv_file, use_region_analysis)
    logger.success(f"Saved: [bold]{csv_file}[/bold]")

    csv_file = f"{output_prefix}_summary.csv"
    _export_summary_csv(results, csv_file, d_a_cutoff, angle_cutoff, n_blocks)
    logger.success(f"Saved: [bold]{csv_file}[/bold]")

    logger.complete()


# Need to import numpy for geometry stats
import numpy as np


def _export_timeseries_csv(results: dict, output_file: str, use_region_analysis: bool):
    """Export time series data to CSV."""
    ts = results['time_series']
    time = ts['time']

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        if use_region_analysis:
            # Header with all region columns
            header = ['time_ps']
            for region in ['global', 'interface_a', 'bulk', 'interface_b']:
                if region in ts:
                    header.extend([
                        f'{region}_n_hbonds',
                        f'{region}_nhbond',
                        f'{region}_n_water'
                    ])
            writer.writerow(header)

            # Data rows
            for i in range(len(time)):
                row = [f'{time[i]:.4f}']
                for region in ['global', 'interface_a', 'bulk', 'interface_b']:
                    if region in ts:
                        row.extend([
                            f'{ts[region]["n_hbonds"][i]}',
                            f'{ts[region]["nhbond"][i]:.6f}',
                            f'{ts[region]["n_water"][i]}'
                        ])
                writer.writerow(row)
        else:
            # Global only
            header = ['time_ps', 'n_hbonds', 'nhbond', 'n_water']
            writer.writerow(header)

            for i in range(len(time)):
                row = [
                    f'{time[i]:.4f}',
                    f'{ts["global"]["n_hbonds"][i]}',
                    f'{ts["global"]["nhbond"][i]:.6f}',
                    f'{ts["global"]["n_water"][i]}'
                ]
                writer.writerow(row)


def _export_summary_csv(results: dict, output_file: str,
                        d_a_cutoff: float, angle_cutoff: float, n_blocks: int):
    """Export summary statistics to CSV."""
    summary = results['summary']
    regions = results['regions']

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header comments
        f.write(f'# Hydrogen Bond Analysis\n')
        f.write(f'# D-A cutoff: {d_a_cutoff} A\n')
        f.write(f'# D-H-A angle cutoff: {angle_cutoff} deg\n')
        f.write(f'# Block averaging: {n_blocks} blocks\n')
        f.write(f'# Total frames: {results["n_frames"]}\n')
        f.write('#\n')

        # Header row
        header = ['region', 'z_min', 'z_max', 'n_water', 'mean_n_hbonds', 'std_n_hbonds', 'mean_nhbond', 'std_nhbond']
        writer.writerow(header)

        # Data rows
        region_order = ['global', 'interface_a', 'bulk', 'interface_b']
        for region_name in region_order:
            if region_name in summary:
                s = summary[region_name]

                if region_name == 'global' or regions is None:
                    z_min, z_max = 'N/A', 'N/A'
                else:
                    z_range = regions.get(region_name, (0, 0))
                    z_min, z_max = f'{z_range[0]:.2f}', f'{z_range[1]:.2f}'

                row = [
                    region_name,
                    z_min,
                    z_max,
                    s['n_water'],
                    f'{s["mean_n_hbonds"]:.4f}',
                    f'{s["std_n_hbonds"]:.4f}',
                    f'{s["mean_nhbond"]:.6f}',
                    f'{s["std_nhbond"]:.6f}'
                ]
                writer.writerow(row)
