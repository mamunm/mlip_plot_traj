"""Shared utilities for CLI commands."""

import click
from pathlib import Path
from typing import Optional, Union

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# Valid principled error estimation methods
VALID_ERROR_METHODS = {'autocorr', 'FlyPet', 'convergence'}

# Valid equilibration detection methods
VALID_EQUILIBRATION_METHODS = {'mser', 'geweke', 'chodera'}


def parse_n_blocks(ctx, param, value: Optional[str]) -> Optional[Union[int, str]]:
    """Parse n_blocks value: integer or method name."""
    if value is None:
        return None

    # Try to parse as integer
    try:
        return int(value)
    except ValueError:
        pass

    # Check for valid method name
    if value in VALID_ERROR_METHODS:
        return value

    raise click.BadParameter(
        f"Must be an integer or one of: {', '.join(sorted(VALID_ERROR_METHODS))}"
    )


def parse_skip_frames(ctx, param, value: Optional[str]) -> Optional[Union[int, float]]:
    """Parse skip_frames value: integer (frame count) or float (fraction like 0.1 for 10%)."""
    if value is None:
        return 0.1  # Default: skip 10% of frames

    # Try to parse as integer first
    try:
        return int(value)
    except ValueError:
        pass

    # Try to parse as float (fraction)
    try:
        frac = float(value)
        if 0.0 <= frac < 1.0:
            return frac
        else:
            raise click.BadParameter(
                f"Fraction must be between 0.0 and 1.0, got {frac}"
            )
    except ValueError:
        pass

    raise click.BadParameter(
        "Must be an integer (frame count) or float between 0.0-1.0 (fraction)"
    )


def format_n_blocks_config(n_blocks: Optional[Union[int, str]]) -> str:
    """Format n_blocks value for configuration display."""
    if n_blocks is None:
        return "[dim]No[/dim]"
    elif isinstance(n_blocks, int):
        return f"[green]Yes ({n_blocks} blocks)[/green]"
    else:
        method_names = {
            'autocorr': 'Autocorrelation',
            'FlyPet': 'Flyvbjerg-Petersen',
            'convergence': 'Convergence'
        }
        method_label = method_names.get(n_blocks, n_blocks)
        return f"[green]{method_label}[/green]"


def format_skip_frames_config(skip_frames: Optional[Union[int, float]], n_frames: Optional[int] = None) -> str:
    """Format skip_frames value for configuration display."""
    if skip_frames is None or skip_frames == 0:
        return "[dim]0[/dim]"
    elif isinstance(skip_frames, float):
        pct = int(skip_frames * 100)
        if n_frames is not None:
            actual = int(n_frames * skip_frames)
            return f"{pct}% ({actual} frames)"
        return f"{pct}%"
    else:
        return str(skip_frames)


def get_output_prefix(save_folder: str, output: Optional[str], trajectories, default_name: str = "combined") -> str:
    """Get the output prefix path for saving files."""
    save_path = Path(save_folder)
    save_path.mkdir(parents=True, exist_ok=True)

    if output is None:
        if isinstance(trajectories, (list, tuple)) and len(trajectories) == 1:
            output_prefix = save_path / Path(trajectories[0]).stem
        elif isinstance(trajectories, str):
            output_prefix = save_path / Path(trajectories).stem
        else:
            output_prefix = save_path / default_name
    else:
        output_prefix = save_path / output

    return str(output_prefix)


def create_progress_context():
    """Create a rich Progress context for tracking operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )


def create_spinner_context():
    """Create a simple spinner Progress context."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    )
