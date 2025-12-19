"""Diffusion coefficient analysis command."""

import click
import csv
from pathlib import Path
from typing import Optional, Tuple, Set, Union

from .. import logger
from .common import (
    parse_skip_frames, format_skip_frames_config,
    console
)
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn


@click.command()
@click.argument('trajectory', type=click.Path(exists=True))
@click.option('--skip-frames', default=None, type=str, callback=parse_skip_frames,
              help='Frames to skip: integer or fraction (e.g., 0.1 for 10%%). Default: 0.1')
@click.option('--dt', default=1.0, type=float,
              help='Time between frames in picoseconds (default: 1.0)')
@click.option('--elements', '-e', multiple=True, default=None,
              help='Elements to analyze (e.g., -e O -e H). Default: water COM')
@click.option('--z-interface', default=None, type=float,
              help='Interface thickness from each surface (Angstroms). Requires --d-bulk.')
@click.option('--d-bulk', default=None, type=float,
              help='Half-width of bulk region around midpoint (Angstroms). Requires --z-interface.')
@click.option('--fit-start-frac', default=0.2, type=float,
              help='Start fraction for fitting (default: 0.2)')
@click.option('--fit-end-frac', default=0.9, type=float,
              help='End fraction for fitting (default: 0.9)')
@click.option('--n-blocks', default=5, type=int,
              help='Number of blocks for error estimation (default: 5)')
@click.option('--output', '-o', default=None, type=str,
              help='Output prefix (default: trajectory filename)')
@click.option('--xlim', nargs=2, type=float, default=None,
              help='X-axis limits (min max)')
@click.option('--ylim', nargs=2, type=float, default=None,
              help='Y-axis limits (min max)')
@click.option('--show', is_flag=True,
              help='Show plots interactively')
@click.option('--verbose', '-v', is_flag=True,
              help='Print detailed progress information')
def diffusion(
    trajectory: str,
    skip_frames: Optional[Union[int, float]],
    dt: float,
    elements: Tuple[str, ...],
    z_interface: Optional[float],
    d_bulk: Optional[float],
    fit_start_frac: float,
    fit_end_frac: float,
    n_blocks: int,
    output: Optional[str],
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
    show: bool,
    verbose: bool
):
    """
    Calculate diffusion coefficients from Mean Square Displacement.

    Computes planar (x-y), perpendicular (z), and total (3D) diffusion.

    \b
    By default, computes diffusion for water molecule center of mass.
    Use -e/--elements to analyze specific atoms instead.

    \b
    Region analysis (--z-interface and --d-bulk):
    Computes separate diffusion coefficients for three regions:
    - Interface A: 0 to z_interface
    - Interface B: z_length - z_interface to z_length
    - Bulk: midpoint +/- d_bulk

    \b
    Examples:
      mlip-plot diffusion trajectory.lammpstrj --dt 2.0
      mlip-plot diffusion trajectory.lammpstrj --z-interface 20 --d-bulk 10
      mlip-plot diffusion trajectory.lammpstrj -e O
      mlip-plot diffusion trajectory.lammpstrj -e Na -e Cl
    """
    from ..io.lammps import read_lammpstrj
    from ..analysis.diffusion import compute_diffusion
    from ..plotting.diffusion import (
        plot_msd, plot_diffusion_summary,
        plot_msd_regions, plot_diffusion_regions_summary
    )

    # Fixed save folder for diffusion analysis
    save_folder = 'analysis/diffusion'
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

    # Determine analysis mode: water COM if no elements specified
    elements_set = set(elements) if elements else None
    use_water_com = elements_set is None

    # Validate region parameters
    if (z_interface is None) != (d_bulk is None):
        logger.error("--z-interface and --d-bulk must both be provided, or neither")
        raise SystemExit(1)
    use_region_analysis = z_interface is not None

    # Print header
    logger.header("MLIP Plot", "Diffusion Coefficient Analysis")

    # Configuration table
    config = {
        "Trajectory": str(trajectory),
        "Output folder": f"[cyan]{save_folder}/[/cyan]",
        "Skip frames": format_skip_frames_config(skip_frames),
        "Time step": f"{dt} ps",
        "Analysis mode": "[bold green]Water COM[/bold green]" if use_water_com else f"Atoms: {', '.join(sorted(elements))}",
    }
    if use_region_analysis:
        config["Region analysis"] = "[bold green]Enabled[/bold green]"
        config["Interface thickness"] = f"{z_interface} A"
        config["Bulk half-width"] = f"{d_bulk} A"
    config["Fit range"] = f"{fit_start_frac*100:.0f}% - {fit_end_frac*100:.0f}%"
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

    if use_water_com:
        if 'O' not in available_elements or 'H' not in available_elements:
            logger.error("Water COM mode requires O and H atoms in trajectory")
            logger.detail("Use -e <element> to analyze specific atoms instead")
            raise SystemExit(1)
    else:
        missing = elements_set - set(available_elements)
        if missing:
            logger.error(f"Elements not found in trajectory: {missing}")
            raise SystemExit(1)

    # Compute diffusion
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, style="dim", complete_style="cyan", finished_style="green"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Computing MSD and diffusion...", total=None)

        results = compute_diffusion(
            frames, dt,
            elements=elements_set,
            use_water_com=use_water_com,
            z_interface=z_interface, d_bulk=d_bulk,
            fit_start_frac=fit_start_frac, fit_end_frac=fit_end_frac,
            unwrap_z=False, n_blocks=n_blocks, verbose=verbose
        )

        progress.remove_task(task)

    msd_data = results['msd']
    diffusion_results = results['diffusion']
    regions = results.get('regions')

    particle_type = "water molecules" if use_water_com else "atoms"
    logger.success(f"MSD computed for {results['n_particles']} {particle_type}")

    # Results table(s)
    use_blocks = n_blocks > 1

    if use_region_analysis:
        # Region-based results: create a table for each region
        region_labels = {
            'interface_a': 'Interface A (Lower)',
            'interface_b': 'Interface B (Upper)',
            'bulk': 'Bulk (Central)'
        }
        type_labels = {
            'planar': 'Planar (x-y)',
            'perpendicular': 'Perpendicular (z)',
            'total': 'Total (3D)'
        }

        for region_name in ['interface_a', 'bulk', 'interface_b']:
            if region_name in diffusion_results:
                region_diff = diffusion_results[region_name]
                z_range = regions[region_name]

                results_tbl = logger.results_table(
                    f"Diffusion: {region_labels[region_name]} ({z_range[0]:.1f}-{z_range[1]:.1f} A)"
                )
                results_tbl.add_column("Type", style="cyan")
                if use_blocks:
                    results_tbl.add_column("D (A^2/ps)", justify="right")
                    results_tbl.add_column("D_mean ± std", justify="right")
                else:
                    results_tbl.add_column("D (A^2/ps)", justify="right")
                    results_tbl.add_column("D (cm^2/s)", justify="right")
                results_tbl.add_column("R^2", justify="right")

                for diff_type in ['planar', 'perpendicular', 'total']:
                    if diff_type in region_diff:
                        d = region_diff[diff_type]
                        if use_blocks:
                            results_tbl.add_row(
                                type_labels[diff_type],
                                f"{d['D_A2ps']:.4f}",
                                f"{d['D_mean']:.4f} ± {d['D_std']:.4f}",
                                f"{d['r_squared']:.4f}"
                            )
                        else:
                            results_tbl.add_row(
                                type_labels[diff_type],
                                f"{d['D_A2ps']:.4f}",
                                f"{d['D_cm2s']:.2e}",
                                f"{d['r_squared']:.4f}"
                            )

                logger.print_table(results_tbl)
    else:
        # Standard results table
        results_tbl = logger.results_table("Diffusion Coefficients")
        results_tbl.add_column("Type", style="cyan")
        if use_blocks:
            results_tbl.add_column("D (A^2/ps)", justify="right")
            results_tbl.add_column("D_mean ± std", justify="right")
        else:
            results_tbl.add_column("D (A^2/ps)", justify="right")
            results_tbl.add_column("D (cm^2/s)", justify="right")
        results_tbl.add_column("R^2", justify="right")

        type_labels = {
            'planar': 'Planar (x-y)',
            'perpendicular': 'Perpendicular (z)',
            'total': 'Total (3D)'
        }

        for diff_type in ['planar', 'perpendicular', 'total']:
            if diff_type in diffusion_results:
                d = diffusion_results[diff_type]
                if use_blocks:
                    results_tbl.add_row(
                        type_labels[diff_type],
                        f"{d['D_A2ps']:.4f}",
                        f"{d['D_mean']:.4f} ± {d['D_std']:.4f}",
                        f"{d['r_squared']:.4f}"
                    )
                else:
                    results_tbl.add_row(
                        type_labels[diff_type],
                        f"{d['D_A2ps']:.4f}",
                        f"{d['D_cm2s']:.2e}",
                        f"{d['r_squared']:.4f}"
                    )

        logger.print_table(results_tbl)

    # Create plots
    logger.section("Creating plots")

    if use_region_analysis:
        # Region-based plots
        output_file = f"{output_prefix}_msd_regions.png"
        plot_msd_regions(
            msd_data, diffusion_results, regions,
            output_file=output_file,
            xlim=xlim, ylim=ylim,
            show=show, verbose=False
        )
        logger.success(f"Saved: [bold]{output_file}[/bold]")

        output_file = f"{output_prefix}_diffusion_regions_summary.png"
        plot_diffusion_regions_summary(
            diffusion_results, regions,
            output_file=output_file,
            show=show, verbose=False
        )
        logger.success(f"Saved: [bold]{output_file}[/bold]")
    else:
        # Standard plots
        output_file = f"{output_prefix}_msd.png"
        plot_msd(
            msd_data, diffusion_results,
            output_file=output_file,
            xlim=xlim, ylim=ylim,
            show=show, verbose=False
        )
        logger.success(f"Saved: [bold]{output_file}[/bold]")

        output_file = f"{output_prefix}_diffusion_summary.png"
        plot_diffusion_summary(
            diffusion_results,
            output_file=output_file,
            show=show, verbose=False
        )
        logger.success(f"Saved: [bold]{output_file}[/bold]")

    # Export CSV (always)
    logger.section("Exporting CSV")

    csv_file = f"{output_prefix}_msd.csv"
    _export_msd_csv(msd_data, csv_file, use_region_analysis)
    logger.success(f"Saved: [bold]{csv_file}[/bold]")

    csv_file = f"{output_prefix}_diffusion.csv"
    _export_diffusion_csv(diffusion_results, use_water_com, elements_set,
                          z_interface, d_bulk, regions, n_blocks, csv_file)
    logger.success(f"Saved: [bold]{csv_file}[/bold]")

    logger.complete()


def _export_msd_csv(msd_data: dict, output_file: str, use_region_analysis: bool = False):
    """Export MSD data to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        if use_region_analysis:
            # Region-based format
            # Get time from first region
            first_region = list(msd_data.keys())[0]
            time = msd_data[first_region]['planar'][0]

            # Header with all region columns
            header = ['time_ps']
            for region in ['interface_a', 'interface_b', 'bulk']:
                if region in msd_data:
                    header.extend([
                        f'{region}_planar_A2',
                        f'{region}_perp_A2',
                        f'{region}_total_A2'
                    ])
            writer.writerow(header)

            # Data rows
            for i in range(len(time)):
                row = [f'{time[i]:.4f}']
                for region in ['interface_a', 'interface_b', 'bulk']:
                    if region in msd_data:
                        row.extend([
                            f'{msd_data[region]["planar"][1][i]:.6f}',
                            f'{msd_data[region]["perpendicular"][1][i]:.6f}',
                            f'{msd_data[region]["total"][1][i]:.6f}'
                        ])
                writer.writerow(row)
        else:
            # Standard format
            time = msd_data['planar'][0]
            writer.writerow(['time_ps', 'msd_planar_A2', 'msd_perpendicular_A2', 'msd_total_A2'])

            for i in range(len(time)):
                writer.writerow([
                    f'{time[i]:.4f}',
                    f'{msd_data["planar"][1][i]:.6f}',
                    f'{msd_data["perpendicular"][1][i]:.6f}',
                    f'{msd_data["total"][1][i]:.6f}'
                ])


def _export_diffusion_csv(diffusion_results: dict, use_water_com: bool,
                          elements: Optional[Set[str]], z_interface: Optional[float],
                          d_bulk: Optional[float], regions: Optional[dict],
                          n_blocks: int, output_file: str):
    """Export diffusion coefficients to CSV."""
    use_region_analysis = regions is not None
    use_blocks = n_blocks > 1

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['# Diffusion Coefficient Analysis'])
        if use_water_com:
            writer.writerow(['# Mode: Water center of mass'])
        else:
            writer.writerow([f'# Mode: Individual atoms ({", ".join(sorted(elements))})'])
        if use_region_analysis:
            writer.writerow([f'# Region analysis: z_interface={z_interface} A, d_bulk={d_bulk} A'])
        if use_blocks:
            writer.writerow([f'# Block averaging: {n_blocks} blocks'])
        writer.writerow([])

        if use_region_analysis:
            # Region-based format
            header = ['region', 'type', 'D_A2_ps', 'D_cm2_s']
            if use_blocks:
                header.extend(['D_mean_A2_ps', 'D_std_A2_ps'])
            header.extend(['r_squared', 'fit_start_ps', 'fit_end_ps', 'n_points'])
            writer.writerow(header)

            for region_name in ['interface_a', 'interface_b', 'bulk']:
                if region_name in diffusion_results:
                    region_diff = diffusion_results[region_name]
                    for diff_type in ['planar', 'perpendicular', 'total']:
                        if diff_type in region_diff:
                            d = region_diff[diff_type]
                            row = [
                                region_name,
                                diff_type,
                                f'{d["D_A2ps"]:.6f}',
                                f'{d["D_cm2s"]:.6e}',
                            ]
                            if use_blocks:
                                row.extend([
                                    f'{d["D_mean"]:.6f}',
                                    f'{d["D_std"]:.6f}',
                                ])
                            row.extend([
                                f'{d["r_squared"]:.6f}',
                                f'{d["fit_start_ps"]:.4f}',
                                f'{d["fit_end_ps"]:.4f}',
                                d['n_points']
                            ])
                            writer.writerow(row)
        else:
            # Standard format
            header = ['type', 'D_A2_ps', 'D_cm2_s']
            if use_blocks:
                header.extend(['D_mean_A2_ps', 'D_std_A2_ps'])
            header.extend(['r_squared', 'fit_start_ps', 'fit_end_ps', 'n_points'])
            writer.writerow(header)

            for diff_type in ['planar', 'perpendicular', 'total']:
                if diff_type in diffusion_results:
                    d = diffusion_results[diff_type]
                    row = [
                        diff_type,
                        f'{d["D_A2ps"]:.6f}',
                        f'{d["D_cm2s"]:.6e}',
                    ]
                    if use_blocks:
                        row.extend([
                            f'{d["D_mean"]:.6f}',
                            f'{d["D_std"]:.6f}',
                        ])
                    row.extend([
                        f'{d["r_squared"]:.6f}',
                        f'{d["fit_start_ps"]:.4f}',
                        f'{d["fit_end_ps"]:.4f}',
                        d['n_points']
                    ])
                    writer.writerow(row)
