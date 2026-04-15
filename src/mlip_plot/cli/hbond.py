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
@click.option('--skip-frames', default=None, type=str, callback=parse_skip_frames,
              help='Frames to skip: integer or fraction (e.g., 0.1 for 10%%). Default: 0.1')
@click.option('--da-cutoff', default=3.5, type=float,
              help='Donor-Acceptor distance cutoff in Angstroms (default: 3.5)')
@click.option('--angle-cutoff', default=150.0, type=float,
              help='D-H-A angle cutoff in degrees (default: 150)')
@click.option('--o-h-cutoff', default=1.2, type=float,
              help='O-H bond cutoff for water identification (default: 1.2)')
@click.option('--n-blocks', default=5, type=int,
              help='Number of blocks for error estimation (default: 5)')
# Region options (metal-water interface)
@click.option('--z-surface', default=None, type=float,
              help='Surface-layer thickness from each metal surface (Angstroms). Requires --d-bulk. '
                   'Manual mode. Pair with --z-subsurface for surface/subsurface/bulk split.')
@click.option('--z-subsurface', default=None, type=float,
              help='Subsurface-layer thickness (Angstroms). Optional. When provided, regions become '
                   'surface / subsurface / bulk; when omitted, legacy interface_a / interface_b / bulk.')
@click.option('--z-interface', default=None, type=float,
              help='[DEPRECATED] Alias for --z-surface.')
@click.option('--d-bulk', default=None, type=float,
              help='Half-width of bulk region around midpoint (Angstroms).')
@click.option('--no-auto-layers', is_flag=True,
              help='Disable automatic layer detection (only compute bulk when no manual region given).')
@click.option('--allow-fraction', is_flag=True,
              help='Allow waters with ANY atom in region (default: require ALL atoms in region)')
@click.option('--da-inside', is_flag=True,
              help='Require BOTH donor AND acceptor in region (default: EITHER in region)')
@click.option('--n-bins', default=50, type=int,
              help='Number of bins for z profile (default: 50)')
@click.option('--dt', default=1.0, type=float,
              help='Time between frames in ps (default: 1.0)')
# Output options
@click.option('--save-hbonds', is_flag=True,
              help='Save H-bonds and waters as numpy arrays')
@click.option('--output', '-o', default=None, type=str,
              help='Output prefix (default: trajectory filename)')
@click.option('--show', is_flag=True,
              help='Show plots interactively')
@click.option('--verbose', '-v', is_flag=True,
              help='Print detailed progress')
def hbond(
    trajectory: str,
    skip_frames: Optional[Union[int, float]],
    da_cutoff: float,
    angle_cutoff: float,
    o_h_cutoff: float,
    n_blocks: int,
    z_surface: Optional[float],
    z_subsurface: Optional[float],
    z_interface: Optional[float],
    d_bulk: Optional[float],
    no_auto_layers: bool,
    allow_fraction: bool,
    da_inside: bool,
    n_bins: int,
    dt: float,
    save_hbonds: bool,
    output: Optional[str],
    show: bool,
    verbose: bool
):
    """
    Analyze hydrogen bonds in MD trajectories.

    Computes H-bond statistics using geometric criteria:
    - Donor-Acceptor distance < da_cutoff (default: 3.5 A)
    - D-H-A angle > angle_cutoff (default: 150 deg)

    \b
    Reports H-bonds per water molecule (2 * n_hbonds / n_waters)
    with block averaging for error estimation.

    \b
    Region analysis:
    1. Auto-detect (default): surface / subsurface / bulk from the O density
       profile (requires scipy). Disable with --no-auto-layers.
    2. Manual --z-surface + --d-bulk (legacy naming):
       interface_a / interface_b / bulk.
    3. Manual --z-surface + --z-subsurface + --d-bulk:
       surface / subsurface / bulk.

    \b
    Examples:
      mlip-plot hbond trajectory.lammpstrj
      mlip-plot hbond trajectory.lammpstrj --n-blocks 10 -v
      mlip-plot hbond trajectory.lammpstrj --z-surface 5 --d-bulk 10
      mlip-plot hbond trajectory.lammpstrj --z-surface 3 --z-subsurface 3 --d-bulk 10
      mlip-plot hbond trajectory.lammpstrj --save-hbonds -o output
    """
    from ..io.lammps import read_lammpstrj
    from ..utils.water import extract_water_molecules
    from ..utils.hbond import identify_hbonds
    from ..analysis.hbond import (
        analyze_hbonds,
        compute_hbond_z_profile,
        compute_hbond_time_profile,
        save_hbonds_numpy,
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

    # Resolve deprecated alias
    if z_interface is not None:
        if z_surface is None:
            z_surface = z_interface
            logger.warning("[deprecated] --z-interface is an alias for --z-surface; please migrate.")
        else:
            logger.error("Specify either --z-surface or --z-interface, not both")
            raise SystemExit(1)

    # Validate manual-region coupling
    if (z_surface is None) != (d_bulk is None):
        logger.error("--z-surface (or --z-interface) and --d-bulk must both be provided, or neither")
        raise SystemExit(1)
    if z_subsurface is not None and z_surface is None:
        logger.error("--z-subsurface requires --z-surface and --d-bulk to be set")
        raise SystemExit(1)

    manual_regions = z_surface is not None
    use_auto_layers = not manual_regions and not no_auto_layers
    use_region_analysis = manual_regions or use_auto_layers

    # Print header
    logger.header("MLIP Plot", "Hydrogen Bond Analysis")

    # Configuration table
    config = {
        "Trajectory": str(trajectory),
        "Output folder": f"[cyan]{save_folder}/[/cyan]",
        "Skip frames": format_skip_frames_config(skip_frames),
        "D-A cutoff": f"{da_cutoff} A",
        "Angle cutoff": f"{angle_cutoff} deg",
        "O-H cutoff": f"{o_h_cutoff} A",
        "Block averaging": f"{n_blocks} blocks",
    }
    if manual_regions:
        config["Region analysis"] = "[bold green]Manual[/bold green]"
        config["Surface thickness"] = f"{z_surface} A"
        if z_subsurface is not None:
            config["Subsurface thickness"] = f"{z_subsurface} A"
        config["Bulk half-width"] = f"{d_bulk} A"
        config["Allow fraction"] = "[green]Yes[/green]" if allow_fraction else "[dim]No[/dim]"
        config["D-A inside"] = "[green]Yes[/green]" if da_inside else "[dim]No[/dim]"
    elif use_auto_layers:
        config["Region analysis"] = "[bold cyan]Auto-detect[/bold cyan]"
        config["Allow fraction"] = "[green]Yes[/green]" if allow_fraction else "[dim]No[/dim]"
        config["D-A inside"] = "[green]Yes[/green]" if da_inside else "[dim]No[/dim]"
    else:
        config["Region analysis"] = "[dim]Disabled (bulk only)[/dim]"
    config["Z profile bins"] = f"{n_bins}"
    config["Time step"] = f"{dt} ps"
    if save_hbonds:
        config["Save numpy"] = "[green]Yes[/green]"

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

    if 'O' not in available_elements or 'H' not in available_elements:
        logger.error("H-bond analysis requires O and H atoms in trajectory")
        raise SystemExit(1)

    # Extract water molecules
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, style="dim", complete_style="cyan", finished_style="green"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Extracting water molecules...", total=None)
        waters = extract_water_molecules(frames, o_h_cutoff=o_h_cutoff)
        progress.remove_task(task)

    n_waters = len(waters[0]) if waters else 0
    logger.success(f"Found {n_waters} water molecules")

    # Get box lengths for PBC
    box_lengths = [
        (f['box']['xhi'] - f['box']['xlo'],
         f['box']['yhi'] - f['box']['ylo'],
         f['box']['zhi'] - f['box']['zlo'])
        for f in frames
    ]

    # Identify hydrogen bonds
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, style="dim", complete_style="cyan", finished_style="green"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Identifying hydrogen bonds...", total=None)
        hbonds = identify_hbonds(
            waters, box_lengths,
            da_cutoff=da_cutoff, angle_cutoff=angle_cutoff
        )
        progress.remove_task(task)

    total_hbonds = sum(len(frame_hbonds) for frame_hbonds in hbonds)
    logger.success(f"Identified {total_hbonds} H-bonds across {len(frames)} frames")

    # Analyze hydrogen bonds
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, style="dim", complete_style="cyan", finished_style="green"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Analyzing H-bond statistics...", total=None)

        results = analyze_hbonds(
            frames, hbonds, waters,
            n_blocks=n_blocks,
            z_interface=z_surface,
            d_bulk=d_bulk,
            z_subsurface=z_subsurface,
            auto_layers=use_auto_layers,
            allow_fraction=allow_fraction,
            da_inside=da_inside,
            verbose=verbose
        )

        progress.remove_task(task)

    # Display results
    bulk_stats = results['bulk']

    if use_region_analysis and 'regions' in results:
        from ..analysis.regions import get_display_label, ordered_region_names
        # Region-based results
        region_stats = results['regions']
        region_defs = results['region_definitions']

        for region_name in ordered_region_names(region_defs):
            if region_name in region_stats:
                stats = region_stats[region_name]
                z_range = region_defs[region_name]
                label = get_display_label(region_name)

                title = f"H-bonds: {label} ({z_range[0]:.1f}-{z_range[1]:.1f} A)"
                results_tbl = logger.results_table(title)
                results_tbl.add_column("Metric", style="cyan")
                results_tbl.add_column("Value", justify="right")

                results_tbl.add_row("H-bonds/water", f"{stats['mean']:.3f} +/- {stats['std']:.3f}")
                results_tbl.add_row("Avg waters in region", f"{stats['avg_waters']:.1f}")
                results_tbl.add_row("Total H-bonds", f"{stats['total_hbonds']}")

                logger.print_table(results_tbl)

        # Global (bulk) statistics
        title = "H-bonds: Global (All Waters)"
        results_tbl = logger.results_table(title)
        results_tbl.add_column("Metric", style="cyan")
        results_tbl.add_column("Value", justify="right")

        results_tbl.add_row("H-bonds/water", f"{bulk_stats['mean']:.3f} +/- {bulk_stats['std']:.3f}")
        results_tbl.add_row("Water molecules", f"{bulk_stats['n_waters']}")
        results_tbl.add_row("Total H-bonds", f"{bulk_stats['total_hbonds']}")
        results_tbl.add_row("Blocks", f"{bulk_stats['n_blocks']}")

        logger.print_table(results_tbl)

    else:
        # Bulk-only results
        results_tbl = logger.results_table("Hydrogen Bond Statistics")
        results_tbl.add_column("Metric", style="cyan")
        results_tbl.add_column("Value", justify="right")

        results_tbl.add_row("H-bonds/water", f"{bulk_stats['mean']:.3f} +/- {bulk_stats['std']:.3f}")
        results_tbl.add_row("Water molecules", f"{bulk_stats['n_waters']}")
        results_tbl.add_row("Frames analyzed", f"{bulk_stats['n_frames']}")
        results_tbl.add_row("Total H-bonds", f"{bulk_stats['total_hbonds']}")
        results_tbl.add_row("Blocks", f"{bulk_stats['n_blocks']}")

        logger.print_table(results_tbl)

    # Profiles (always generated)
    logger.section("Computing profiles")

    # Z profile
    box = frames[0]['box']
    z_min = box['zlo']
    z_max = box['zhi']

    bin_centers, hbonds_per_water_z, water_counts, hbond_counts = compute_hbond_z_profile(
        hbonds, waters, z_min, z_max, n_bins
    )

    # Save z profile data
    csv_file = f"{output_prefix}_z_profile.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['z_A', 'hbonds_per_water', 'water_count', 'hbond_count'])
        for i in range(len(bin_centers)):
            writer.writerow([f'{bin_centers[i]:.4f}', f'{hbonds_per_water_z[i]:.6f}',
                           f'{water_counts[i]:.0f}', f'{hbond_counts[i]:.0f}'])
    logger.success(f"Saved: [bold]{csv_file}[/bold]")

    # Plot z profile
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(bin_centers, hbonds_per_water_z, 'b-', linewidth=1.5)
        ax.set_xlabel('z (A)')
        ax.set_ylabel('H-bonds per water')
        ax.set_title('H-bonds per Water along z')
        ax.grid(True, alpha=0.3)

        plot_file = f"{output_prefix}_z_profile.png"
        fig.tight_layout()
        fig.savefig(plot_file, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        logger.success(f"Saved: [bold]{plot_file}[/bold]")
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")

    # Time profile
    time_arr, hbonds_per_water = compute_hbond_time_profile(hbonds, n_waters, dt)

    # Save time profile data
    csv_file = f"{output_prefix}_time_profile.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_ps', 'hbonds_per_water'])
        for i in range(len(time_arr)):
            writer.writerow([f'{time_arr[i]:.4f}', f'{hbonds_per_water[i]:.6f}'])
    logger.success(f"Saved: [bold]{csv_file}[/bold]")

    # Plot time profile
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_arr, hbonds_per_water, 'b-', linewidth=0.5, alpha=0.7)
        ax.axhline(bulk_stats['mean'], color='r', linestyle='--', label=f"Mean: {bulk_stats['mean']:.2f}")
        ax.set_xlabel('Time (ps)')
        ax.set_ylabel('H-bonds per water')
        ax.set_title('H-bonds per Water vs Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_file = f"{output_prefix}_time_profile.png"
        fig.tight_layout()
        fig.savefig(plot_file, dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        logger.success(f"Saved: [bold]{plot_file}[/bold]")
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")

    # Save numpy arrays
    if save_hbonds:
        logger.section("Saving numpy arrays")
        save_hbonds_numpy(hbonds, waters, output_prefix)
        logger.success(f"Saved: [bold]{output_prefix}_hbonds.npz[/bold]")
        logger.success(f"Saved: [bold]{output_prefix}_waters.npz[/bold]")

    # Export statistics CSV
    logger.section("Exporting CSV")

    csv_file = f"{output_prefix}_stats.csv"
    has_regions = 'regions' in results
    _export_stats_csv(results, has_regions, n_blocks, da_cutoff, angle_cutoff, csv_file)
    logger.success(f"Saved: [bold]{csv_file}[/bold]")

    logger.complete()


def _export_stats_csv(results: dict, use_region_analysis: bool, n_blocks: int,
                      da_cutoff: float, angle_cutoff: float, output_file: str):
    """Export H-bond statistics to CSV."""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['# Hydrogen Bond Analysis'])
        writer.writerow([f'# D-A cutoff: {da_cutoff} A'])
        writer.writerow([f'# Angle cutoff: {angle_cutoff} deg'])
        writer.writerow([f'# Block averaging: {n_blocks} blocks'])
        writer.writerow([])

        if use_region_analysis:
            # Region-based format
            writer.writerow(['region', 'z_min_A', 'z_max_A', 'hbonds_per_water_mean',
                           'hbonds_per_water_std', 'avg_waters', 'total_hbonds'])

            region_stats = results['regions']
            region_defs = results['region_definitions']
            from ..analysis.regions import ordered_region_names

            for region_name in ordered_region_names(region_defs):
                if region_name in region_stats:
                    stats = region_stats[region_name]
                    z_range = region_defs[region_name]
                    writer.writerow([
                        region_name,
                        f'{z_range[0]:.4f}',
                        f'{z_range[1]:.4f}',
                        f'{stats["mean"]:.6f}',
                        f'{stats["std"]:.6f}',
                        f'{stats["avg_waters"]:.2f}',
                        stats['total_hbonds']
                    ])

            # Add global
            bulk_stats = results['bulk']
            box_zlo = results.get('z_lo', 0.0)
            box_zhi = results.get('z_hi', 0.0)
            writer.writerow([
                'global',
                f'{box_zlo:.4f}',
                f'{box_zhi:.4f}',
                f'{bulk_stats["mean"]:.6f}',
                f'{bulk_stats["std"]:.6f}',
                f'{bulk_stats["n_waters"]:.2f}',
                bulk_stats['total_hbonds']
            ])
        else:
            # Bulk-only format
            writer.writerow(['metric', 'value'])
            bulk_stats = results['bulk']
            writer.writerow(['hbonds_per_water_mean', f'{bulk_stats["mean"]:.6f}'])
            writer.writerow(['hbonds_per_water_std', f'{bulk_stats["std"]:.6f}'])
            writer.writerow(['n_waters', bulk_stats['n_waters']])
            writer.writerow(['n_frames', bulk_stats['n_frames']])
            writer.writerow(['total_hbonds', bulk_stats['total_hbonds']])
            writer.writerow(['n_blocks', bulk_stats['n_blocks']])
