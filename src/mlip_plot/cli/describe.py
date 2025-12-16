"""Describe command for trajectory information."""

import click
from pathlib import Path
from typing import Optional
from collections import Counter

from .. import logger


@click.command()
@click.argument('trajectory', type=click.Path(exists=True))
@click.option('--max-frames', default=None, type=int,
              help='Maximum frames to scan (default: all)')
def describe(trajectory: str, max_frames: Optional[int]):
    """
    Show basic information about a LAMMPS trajectory file.

    Displays number of frames, atoms, elements, box dimensions, and timestep range.

    \b
    Examples:
      mlip-plot describe trajectory.lammpstrj
      mlip-plot describe trajectory.lammpstrj --max-frames 100
    """
    from ..io.lammps import read_lammpstrj

    # Print header
    logger.header("MLIP Plot", "Trajectory Description")

    # Read trajectory
    logger.info(f"Reading: [bold]{trajectory}[/bold]")
    frames = read_lammpstrj(trajectory, max_frames=max_frames, verbose=False)

    if not frames:
        logger.error("No frames found in trajectory")
        raise SystemExit(1)

    # Basic info
    first_frame = frames[0]
    last_frame = frames[-1]

    info_table = logger.results_table("Trajectory Info")
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", justify="right")

    info_table.add_row("File", str(Path(trajectory).name))
    info_table.add_row("Total frames", str(len(frames)))
    info_table.add_row("Atoms per frame", str(first_frame['natoms']))
    info_table.add_row("Timestep range", f"{first_frame['timestep']} - {last_frame['timestep']}")

    # Timestep interval (if more than one frame)
    if len(frames) > 1:
        dt = frames[1]['timestep'] - frames[0]['timestep']
        info_table.add_row("Timestep interval", str(dt))

    logger.print_table(info_table)

    # Box dimensions
    box = first_frame['box']
    lx = box['xhi'] - box['xlo']
    ly = box['yhi'] - box['ylo']
    lz = box['zhi'] - box['zlo']
    volume = lx * ly * lz

    box_table = logger.results_table("Box Dimensions (first frame)")
    box_table.add_column("Axis", style="cyan")
    box_table.add_column("Min (Å)", justify="right")
    box_table.add_column("Max (Å)", justify="right")
    box_table.add_column("Length (Å)", justify="right")

    box_table.add_row("X", f"{box['xlo']:.2f}", f"{box['xhi']:.2f}", f"{lx:.2f}")
    box_table.add_row("Y", f"{box['ylo']:.2f}", f"{box['yhi']:.2f}", f"{ly:.2f}")
    box_table.add_row("Z", f"{box['zlo']:.2f}", f"{box['zhi']:.2f}", f"{lz:.2f}")

    logger.print_table(box_table)
    logger.detail(f"Volume: {volume:.2f} Å³")

    # Element composition
    if first_frame['elements'] is not None:
        elem_counts = Counter(first_frame['elements'])

        elem_table = logger.results_table("Element Composition")
        elem_table.add_column("Element", style="cyan")
        elem_table.add_column("Count", justify="right")
        elem_table.add_column("Fraction", justify="right")

        total_atoms = first_frame['natoms']
        for elem in sorted(elem_counts.keys()):
            count = elem_counts[elem]
            fraction = count / total_atoms * 100
            elem_table.add_row(elem, str(count), f"{fraction:.1f}%")

        logger.print_table(elem_table)
    else:
        logger.warning("No element information in trajectory")

        # Show type distribution instead
        type_counts = Counter(first_frame['types'])
        type_table = logger.results_table("Atom Types")
        type_table.add_column("Type", style="cyan")
        type_table.add_column("Count", justify="right")

        for typ in sorted(type_counts.keys()):
            type_table.add_row(str(typ), str(type_counts[typ]))

        logger.print_table(type_table)

    logger.complete()
