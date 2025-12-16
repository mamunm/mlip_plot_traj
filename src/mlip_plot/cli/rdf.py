"""RDF (Radial Distribution Function) command."""

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
@click.argument('trajectory', type=click.Path(exists=True))
@click.option('--skip-frames', default=None, type=str, callback=parse_skip_frames,
              help='Frames to skip: integer or fraction (e.g., 0.1 for 10%%). Default: 0.1')
@click.option('--n-blocks', default=5, type=str, callback=parse_n_blocks,
              help='Error estimation: integer for fixed blocks, or "autocorr", "FlyPet", "convergence". Default: 5')
@click.option('--rmin', default=0.05, type=float,
              help='Minimum distance in Angstroms (default: 0.05)')
@click.option('--rmax', default=6.0, type=float,
              help='Maximum distance in Angstroms (default: 6.0)')
@click.option('--n-bins', default=100, type=int,
              help='Number of bins (default: 100)')
@click.option('--exclude', '-e', multiple=True, default=['Cu', 'Pt'],
              help='Elements to exclude (default: Cu, Pt)')
@click.option('--output', '-o', default=None, type=str,
              help='Output prefix (default: trajectory filename)')
@click.option('--xlim', nargs=2, type=float, default=None,
              help='X-axis limits (min max)')
@click.option('--ylim', nargs=2, type=float, default=None,
              help='Y-axis limits (min max)')
@click.option('--show', is_flag=True,
              help='Show plots interactively')
def rdf(
    trajectory: str,
    skip_frames: Optional[Union[int, float]],
    n_blocks: Union[int, str],
    rmin: float,
    rmax: float,
    n_bins: int,
    exclude: Tuple[str, ...],
    output: Optional[str],
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
    show: bool
):
    """
    Calculate and plot radial distribution functions (RDF).

    \b
    Examples:
      mlip-plot rdf trajectory.lammpstrj
      mlip-plot rdf trajectory.lammpstrj --n-blocks autocorr
      mlip-plot rdf trajectory.lammpstrj --skip-frames 0.2 --rmax 8.0
    """
    from ..io.lammps import read_lammpstrj
    from ..analysis.rdf import compute_rdf
    from ..plotting.rdf import plot_rdf

    # Fixed save folder for RDF analysis
    save_folder = 'analysis/rdf'
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

    exclude_set = set(exclude)

    # Print header
    logger.header("MLIP Plot", "RDF Analysis")

    # Configuration table
    config = {
        "Trajectory": str(trajectory),
        "Output folder": f"[cyan]{save_folder}/[/cyan]",
        "Skip frames": format_skip_frames_config(skip_frames),
        "Error estimation": format_n_blocks_config(n_blocks),
        "Distance range": f"{rmin:.2f} - {rmax:.2f} Å",
        "Bins": str(n_bins),
        "Excluded elements": ", ".join(exclude) if exclude else "[dim]None[/dim]",
    }
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
            # Fraction: first read to get total frame count, then compute skip
            all_frames = read_lammpstrj(trajectory, skip_frames=0, verbose=False)
            if not all_frames:
                progress.remove_task(task)
                progress.stop()
                logger.error("No frames available for analysis")
                raise SystemExit(1)
            actual_skip = int(len(all_frames) * skip_frames)
            frames = all_frames[actual_skip:]
        else:
            # Integer: direct skip
            actual_skip = skip_frames if skip_frames else 0
            frames = read_lammpstrj(trajectory, skip_frames=actual_skip, verbose=False)

        progress.remove_task(task)

    if not frames:
        logger.error("No frames available for analysis")
        raise SystemExit(1)

    logger.success(f"Loaded trajectory: {len(frames)} frames")

    # Detect available elements and generate pairs
    if frames[0]['elements'] is None:
        logger.error("Trajectory must contain element information")
        raise SystemExit(1)

    all_elements = sorted(set(frames[0]['elements']))
    # Filter out excluded elements
    available_elements = [e for e in all_elements if e not in exclude_set]
    logger.detail(f"Elements for RDF: {', '.join(available_elements)}")

    # Generate all unique pairs from available elements
    pair_list = []
    for i, e1 in enumerate(available_elements):
        for e2 in available_elements[i:]:
            pair_list.append((e1, e2))

    logger.detail(f"Pairs to compute: {', '.join([f'{p[0]}-{p[1]}' for p in pair_list])}")

    # Compute RDF for each pair
    rdf_data = {}
    all_diagnostics = {}

    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, style="dim", complete_style="cyan", finished_style="green"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        for type1, type2 in pair_list:
            pair_label = f"{type1}-{type2}"

            if isinstance(n_blocks, str):
                task = progress.add_task(f"Computing {pair_label} ({n_blocks})...", total=None)
            else:
                task = progress.add_task(f"Computing {pair_label}...", total=None)

            r, g_r, std_error, diagnostics = compute_rdf(
                frames, type1, type2,
                rmin=rmin, rmax=rmax, n_bins=n_bins,
                n_blocks=n_blocks
            )

            progress.remove_task(task)

            rdf_data[(type1, type2)] = (r, g_r, std_error)
            if diagnostics:
                all_diagnostics[pair_label] = diagnostics

    # Show results
    rdf_table = logger.results_table("RDF Summary")
    rdf_table.add_column("Pair", style="cyan")
    rdf_table.add_column("First Peak r", justify="right")
    rdf_table.add_column("First Peak g(r)", justify="right")

    for (type1, type2), (r, g_r, _) in rdf_data.items():
        pair_label = f"{type1}-{type2}"
        max_idx = g_r.argmax()
        rdf_table.add_row(pair_label, f"{r[max_idx]:.2f} Å", f"{g_r[max_idx]:.3f}")

    logger.print_table(rdf_table)

    # Create plots
    logger.section("Creating plots")
    output_files = plot_rdf(
        rdf_data,
        output_prefix=output_prefix,
        xlim=xlim,
        ylim=ylim,
        show_errorbar=True,
        fill_alpha=0.3,
        show=show,
        verbose=False
    )
    logger.success(f"Saved {len(output_files)} plot(s) to [bold]{save_folder}/[/bold]")

    # Export CSV (always)
    logger.section("Exporting CSV")
    csv_file = f"{output_prefix}_rdf.csv"
    _export_rdf_csv(rdf_data, csv_file)
    logger.success(f"Saved: [bold]{csv_file}[/bold]")

    logger.complete()


def _export_rdf_csv(rdf_data: dict, output_file: str):
    """Export RDF data to CSV."""
    # Get r values from first pair
    first_pair = list(rdf_data.keys())[0]
    r = rdf_data[first_pair][0]
    has_errors = rdf_data[first_pair][2] is not None

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['r_angstrom']
        for pair in rdf_data.keys():
            pair_label = f"{pair[0]}-{pair[1]}"
            header.append(f'g_{pair_label}')
            if has_errors:
                header.append(f'stderr_{pair_label}')
        writer.writerow(header)

        # Data
        for i in range(len(r)):
            row = [f'{r[i]:.4f}']
            for pair in rdf_data.keys():
                _, g_r, std_error = rdf_data[pair]
                row.append(f'{g_r[i]:.6f}')
                if has_errors and std_error is not None:
                    row.append(f'{std_error[i]:.6f}')
            writer.writerow(row)
