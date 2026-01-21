"""Water reorientation analysis command."""

import click
import csv
from pathlib import Path
from typing import Optional, Union
import numpy as np

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
@click.option('--o-h-cutoff', default=1.2, type=float,
              help='O-H bond cutoff for water identification (default: 1.2)')
@click.option('--no-z-pbc', is_flag=True,
              help='Disable periodic boundary conditions in z direction (for slab geometries)')
@click.option('--n-bins', default=50, type=int,
              help='Number of bins for histogram (default: 50)')
@click.option('--n-blocks', default=5, type=int,
              help='Number of blocks for error estimation (default: 5)')
# Region options (metal-water interface)
@click.option('--z-interface', default=None, type=float,
              help='Interface thickness from each surface (Angstroms). Requires --d-bulk. For manual region mode.')
@click.option('--d-bulk', default=None, type=float,
              help='Half-width of bulk region around midpoint (Angstroms). Requires --z-interface. For manual region mode.')
@click.option('--no-auto-layers', is_flag=True,
              help='Disable automatic layer detection (compute global only when no region params given)')
# Output options
@click.option('--output', '-o', default=None, type=str,
              help='Output prefix (default: trajectory filename)')
@click.option('--show', is_flag=True,
              help='Show plots interactively')
# Plot options
@click.option('--x-lim', nargs=2, type=float, default=None,
              help='X-axis limits for plots (e.g., --x-lim -1 1)')
@click.option('--y-lim', nargs=2, type=float, default=None,
              help='Y-axis limits for plots (e.g., --y-lim 0 1.5)')
@click.option('--combined-plot', is_flag=True,
              help='Plot all regions on a single plot (one for theta, one for phi)')
@click.option('--save-data', is_flag=True,
              help='Save raw orientation data (cos_theta_1, cos_theta_2, cos_phi, O_z) per water per frame')
@click.option('--verbose', '-v', is_flag=True,
              help='Print detailed progress')
def reorientation(
    trajectory: str,
    skip_frames: Optional[Union[int, float]],
    o_h_cutoff: float,
    no_z_pbc: bool,
    n_bins: int,
    n_blocks: int,
    z_interface: Optional[float],
    d_bulk: Optional[float],
    no_auto_layers: bool,
    output: Optional[str],
    show: bool,
    x_lim: Optional[tuple],
    y_lim: Optional[tuple],
    combined_plot: bool,
    save_data: bool,
    verbose: bool
):
    """
    Analyze water molecule orientation in MD trajectories.

    Computes probability distributions for water orientation angles:
    - P(cos θ): O-H bond orientation relative to surface normal
    - P(cos φ): Dipole orientation relative to surface normal

    \b
    Interpretation:
    - cos = +1: vector pointing away from surface (up)
    - cos = -1: vector pointing toward surface (down)
    - P = 0.5: isotropic (random) orientation

    \b
    Region analysis modes:
    1. Auto-detect (default): Automatically detects surface, subsurface, and
       bulk layers from oxygen density profile when no region params given.
    2. Manual (--z-interface and --d-bulk): User-specified regions.
    3. Global only (--no-auto-layers): Only compute global distributions.

    \b
    Auto-detected regions (default):
    - Surface: From metal surface to first O density minimum
    - Subsurface: From first to second O density minimum
    - Bulk: From second minimum to middle of box

    \b
    Manual regions (--z-interface and --d-bulk):
    - Interface A: From lower metal surface to z_interface
    - Interface B: From upper boundary - z_interface to upper boundary
    - Bulk: midpoint +/- d_bulk

    \b
    Examples:
      mlip-plot reorientation trajectory.lammpstrj
      mlip-plot reorientation trajectory.lammpstrj --n-bins 100 -v
      mlip-plot reorientation trajectory.lammpstrj --z-interface 5.0 --d-bulk 10.0
      mlip-plot reorientation trajectory.lammpstrj --no-auto-layers
    """
    from ..io.lammps import read_lammpstrj
    from ..utils.water import extract_water_molecules
    from ..analysis.reorientation import analyze_reorientation
    from ..plotting.reorientation import plot_all_regions, plot_combined_regions
    from mlip_plot._core import compute_orientations

    # Fixed save folder for reorientation analysis
    save_folder = 'analysis/reorientation'
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

    # Determine region analysis mode
    manual_regions = z_interface is not None
    auto_layers_enabled = not manual_regions and not no_auto_layers

    # Print header
    logger.header("MLIP Plot", "Water Reorientation Analysis")

    # Configuration table
    config = {
        "Trajectory": str(trajectory),
        "Output folder": f"[cyan]{save_folder}/[/cyan]",
        "Skip frames": format_skip_frames_config(skip_frames),
        "O-H cutoff": f"{o_h_cutoff} Å",
        "Histogram bins": f"{n_bins}",
        "Block averaging": f"{n_blocks} blocks",
    }
    if manual_regions:
        config["Region analysis"] = "[bold green]Manual[/bold green]"
        config["Interface thickness"] = f"{z_interface} Å"
        config["Bulk half-width"] = f"{d_bulk} Å"
    elif auto_layers_enabled:
        config["Region analysis"] = "[bold cyan]Auto-detect[/bold cyan]"
    else:
        config["Region analysis"] = "[dim]Disabled (global only)[/dim]"

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
        logger.error("Reorientation analysis requires O and H atoms in trajectory")
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

    # Compute orientations using C++ core (per-frame to handle different atom ordering)
    # IMPORTANT: Atoms may be in different order in different frames of LAMMPS dump,
    # so we must use per-frame water indices, not indices from frame 0.
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, style="dim", complete_style="cyan", finished_style="green"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Computing water orientations...", total=None)

        orientations = []
        surface_normal = (0.0, 0.0, 1.0)

        for frame_idx, frame in enumerate(frames):
            # Get water indices for THIS frame
            frame_waters = waters[frame_idx]
            water_indices = np.array([
                [w.O_index, w.H1_index, w.H2_index] for w in frame_waters
            ], dtype=np.int32)

            # Get box lengths and positions for this frame
            box_lengths = (
                frame['box']['xhi'] - frame['box']['xlo'],
                frame['box']['yhi'] - frame['box']['ylo'],
                frame['box']['zhi'] - frame['box']['zlo']
            )

            # Compute orientations for single frame
            frame_orientations = compute_orientations(
                [frame['positions']],
                water_indices,
                [box_lengths],
                surface_normal,
                no_z_pbc
            )

            # frame_orientations is a list with 1 element (single frame)
            orientations.append(frame_orientations[0])

        progress.remove_task(task)

    logger.success(f"Computed orientations for {len(orientations)} frames")

    # Save raw orientation data if requested
    if save_data:
        _export_raw_orientation_data(orientations, waters, output_prefix, logger)

    # Analyze orientations
    with Progress(
        SpinnerColumn("dots"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, style="dim", complete_style="cyan", finished_style="green"),
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Analyzing orientation distributions...", total=None)

        results = analyze_reorientation(
            frames, orientations,
            n_bins=n_bins,
            n_blocks=n_blocks,
            z_interface=z_interface,
            d_bulk=d_bulk,
            auto_layers=auto_layers_enabled,
            verbose=verbose,
            logger=logger
        )

        progress.remove_task(task)

    # Display results
    theta_data = results['theta']
    phi_data = results['phi']
    region_mode = results.get('region_mode', 'global')

    # Region labels for display
    region_labels = {
        # Manual regions
        'interface_a': 'Interface A (Lower)',
        'interface_b': 'Interface B (Upper)',
        # Auto-detected regions
        'surface': 'Surface',
        'subsurface': 'Subsurface',
        'bulk': 'Bulk',
        'global': 'Global',
    }

    # Determine which regions to display based on mode
    if region_mode == 'manual':
        regions_order = ['interface_a', 'bulk', 'interface_b']
    elif region_mode == 'auto':
        regions_order = ['surface', 'subsurface', 'bulk']
    else:
        regions_order = []

    # Display region-based results
    for region_name in regions_order:
        if region_name in theta_data:
            theta_stats = theta_data[region_name]
            phi_stats = phi_data[region_name]

            z_range = theta_stats.get('z_range', (0, 0))
            label = region_labels.get(region_name, region_name.title())

            title = f"Orientation: {label} ({z_range[0]:.1f}-{z_range[1]:.1f} Å)"
            results_tbl = logger.results_table(title)
            results_tbl.add_column("Angle", style="cyan")
            results_tbl.add_column("⟨cos⟩", justify="right")
            results_tbl.add_column("σ(cos)", justify="right")
            results_tbl.add_column("⟨P⟩", justify="right")
            results_tbl.add_column("σ(P)", justify="right")
            results_tbl.add_column("Samples", justify="right")

            results_tbl.add_row(
                "θ (O-H)",
                f"{theta_stats['mean_cos']:.4f}",
                f"{theta_stats['std_cos']:.4f}",
                f"{theta_stats['mean_P']:.4f}",
                f"{theta_stats['std_P']:.4f}",
                f"{theta_stats['n_samples']}"
            )
            results_tbl.add_row(
                "φ (dipole)",
                f"{phi_stats['mean_cos']:.4f}",
                f"{phi_stats['std_cos']:.4f}",
                f"{phi_stats['mean_P']:.4f}",
                f"{phi_stats['std_P']:.4f}",
                f"{phi_stats['n_samples']}"
            )

            logger.print_table(results_tbl)

    # Global statistics
    theta_global = theta_data['global']
    phi_global = phi_data['global']

    title = "Orientation: Global (All Waters)"
    results_tbl = logger.results_table(title)
    results_tbl.add_column("Angle", style="cyan")
    results_tbl.add_column("⟨cos⟩", justify="right")
    results_tbl.add_column("σ(cos)", justify="right")
    results_tbl.add_column("⟨P⟩", justify="right")
    results_tbl.add_column("σ(P)", justify="right")
    results_tbl.add_column("Samples", justify="right")

    results_tbl.add_row(
        "θ (O-H)",
        f"{theta_global['mean_cos']:.4f}",
        f"{theta_global['std_cos']:.4f}",
        f"{theta_global['mean_P']:.4f}",
        f"{theta_global['std_P']:.4f}",
        f"{theta_global['n_samples']}"
    )
    results_tbl.add_row(
        "φ (dipole)",
        f"{phi_global['mean_cos']:.4f}",
        f"{phi_global['std_cos']:.4f}",
        f"{phi_global['mean_P']:.4f}",
        f"{phi_global['std_P']:.4f}",
        f"{phi_global['n_samples']}"
    )

    logger.print_table(results_tbl)

    # Generate plots
    logger.section("Generating plots")

    # Determine which regions to plot
    if region_mode == 'manual':
        regions_to_plot = ['interface_a', 'bulk', 'interface_b']
    elif region_mode == 'auto':
        regions_to_plot = ['surface', 'subsurface', 'bulk']
    else:
        regions_to_plot = ['global']

    # Plot layer detection if auto-layers was used
    if region_mode == 'auto' and 'layer_diagnostics' in results:
        from ..plotting.reorientation import plot_layer_detection
        plot_layer_detection(
            results['layer_diagnostics'],
            output_prefix,
            show=show,
            logger=logger
        )

    if combined_plot:
        # Plot all regions on single plots (one for theta, one for phi)
        plot_combined_regions(
            theta_data, phi_data, output_prefix,
            regions_order=regions_to_plot,
            xlim=x_lim, ylim=y_lim,
            show=show, verbose=True, logger=logger
        )
    else:
        # Generate individual plots for each region (6 plots for region analysis, 2 for global)
        plot_all_regions(
            theta_data, phi_data, output_prefix,
            regions_order=regions_to_plot,
            xlim=x_lim, ylim=y_lim,
            show=show, verbose=True, logger=logger
        )

    # Export CSV files
    logger.section("Exporting CSV")

    _export_distribution_csv(theta_data, output_prefix, 'theta', regions_to_plot, logger)
    _export_distribution_csv(phi_data, output_prefix, 'phi', regions_to_plot, logger)
    _export_stats_csv(results, output_prefix, region_mode, logger)

    logger.complete()


def _export_distribution_csv(
    data: dict,
    output_prefix: str,
    angle_type: str,
    regions: list,
    logger
):
    """Export orientation distribution to CSV."""
    csv_file = f"{output_prefix}_{angle_type}.csv"

    # Build header
    header = [f'cos_{angle_type}']
    for region in regions:
        if region in data:
            header.append(f'P_{region}')
            header.append(f'err_{region}')

    # Get bin centers from first available region
    bins = None
    for region in regions:
        if region in data:
            bins = data[region]['bins']
            break

    if bins is None:
        return

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Metadata comments
        if angle_type == 'theta':
            writer.writerow(['# Water O-H Bond Orientation Distribution'])
            writer.writerow(['# cos(theta) = rOH dot surface_normal'])
        else:
            writer.writerow(['# Water Dipole Orientation Distribution'])
            writer.writerow(['# cos(phi) = dipole dot surface_normal'])
        writer.writerow(['# cos = +1: pointing away from surface'])
        writer.writerow(['# cos = -1: pointing toward surface'])
        writer.writerow([])

        writer.writerow(header)

        for i, cos_val in enumerate(bins):
            row = [f'{cos_val:.6f}']
            for region in regions:
                if region in data:
                    row.append(f'{data[region]["P"][i]:.6f}')
                    row.append(f'{data[region]["err"][i]:.6f}')
            writer.writerow(row)

    logger.success(f"Saved: [bold]{csv_file}[/bold]")


def _export_stats_csv(results: dict, output_prefix: str, region_mode: str, logger):
    """Export orientation statistics to CSV."""
    csv_file = f"{output_prefix}_stats.csv"

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['# Water Orientation Statistics'])
        writer.writerow([f'# region_mode: {region_mode}'])
        writer.writerow([f'# n_frames: {results["n_frames"]}'])
        writer.writerow([f'# n_waters: {results["n_waters"]}'])
        writer.writerow([f'# n_bins: {results["n_bins"]}'])
        writer.writerow([f'# n_blocks: {results["n_blocks"]}'])
        writer.writerow([])

        writer.writerow(['angle_type', 'region', 'z_min', 'z_max', 'mean_cos', 'std_cos', 'mean_P', 'std_P', 'n_samples', 'avg_waters'])

        for angle_type in ['theta', 'phi']:
            data = results[angle_type]
            for region_name, stats in data.items():
                z_range = stats.get('z_range', ('', ''))
                avg_waters = stats.get('avg_waters', results['n_waters'])

                writer.writerow([
                    angle_type,
                    region_name,
                    f'{z_range[0]:.4f}' if z_range[0] != '' else '',
                    f'{z_range[1]:.4f}' if z_range[1] != '' else '',
                    f'{stats["mean_cos"]:.6f}',
                    f'{stats["std_cos"]:.6f}',
                    f'{stats["mean_P"]:.6f}',
                    f'{stats["std_P"]:.6f}',
                    stats['n_samples'],
                    f'{avg_waters:.2f}'
                ])

    logger.success(f"Saved: [bold]{csv_file}[/bold]")


def _export_raw_orientation_data(orientations: list, waters: list, output_prefix: str, logger):
    """Export raw orientation data (per water, per frame) as numpy arrays.

    Saves:
    - frame: frame index for each entry
    - water_coords: oxygen coordinates (x, y, z) for each water
    - cos_theta_1: O->H1 dot surface_normal
    - cos_theta_2: O->H2 dot surface_normal
    - cos_phi: dipole dot surface_normal

    Note: Uses waters list (per-frame water identification) to get correct
    oxygen coordinates since atoms may be in different order across frames.
    """
    npy_file = f"{output_prefix}_raw_data.npz"

    # Collect all data into lists
    all_frames = []
    all_water_coords = []
    all_cos_theta_1 = []
    all_cos_theta_2 = []
    all_cos_phi = []

    for frame_idx, frame_data in enumerate(orientations):
        n_waters = frame_data['n_waters']
        frame_waters = waters[frame_idx]

        for w in range(n_waters):
            # Get oxygen coordinates from per-frame water identification
            O_coords = frame_waters[w].O_position  # [x, y, z]

            all_frames.append(frame_idx)
            all_water_coords.append(O_coords)
            all_cos_theta_1.append(frame_data['cos_theta_1'][w])
            all_cos_theta_2.append(frame_data['cos_theta_2'][w])
            all_cos_phi.append(frame_data['cos_phi'][w])

    # Save as compressed numpy archive
    np.savez(
        npy_file,
        frame=np.array(all_frames, dtype=np.int32),
        water_coords=np.array(all_water_coords, dtype=np.float64),
        cos_theta_1=np.array(all_cos_theta_1, dtype=np.float64),
        cos_theta_2=np.array(all_cos_theta_2, dtype=np.float64),
        cos_phi=np.array(all_cos_phi, dtype=np.float64)
    )

    logger.success(f"Saved: [bold]{npy_file}[/bold]")
