"""
Unified logger with elapsed time display.

Format: [local_time](+elapsed_seconds) message
"""

import time
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Global console instance
console = Console()

# Global start time
_start_time: Optional[float] = None


def reset_timer():
    """Reset the timer to current time."""
    global _start_time
    _start_time = time.time()


def _get_local_time() -> str:
    """Get current local time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


def _get_total_elapsed() -> str:
    """Get total elapsed time in seconds."""
    if _start_time is None:
        reset_timer()

    total = time.time() - _start_time
    return f"+{total:.2f}s"


def _format_prefix() -> str:
    """Format the time prefix: [local_time](+elapsed)."""
    return f"[dim][{_get_local_time()}][/dim]([cyan]{_get_total_elapsed()}[/cyan])"


def log(message: str, style: str = ""):
    """
    Log a message with elapsed time.

    Parameters
    ----------
    message : str
        Message to log
    style : str, optional
        Rich style to apply
    """
    prefix = _format_prefix()
    if style:
        console.print(f"{prefix} [{style}]{message}[/{style}]")
    else:
        console.print(f"{prefix} {message}")


def info(message: str):
    """Log an info message."""
    prefix = _format_prefix()
    console.print(f"{prefix} {message}")


def success(message: str):
    """Log a success message with green checkmark."""
    prefix = _format_prefix()
    console.print(f"{prefix} [green]\u2713[/green] {message}")


def warning(message: str):
    """Log a warning message."""
    prefix = _format_prefix()
    console.print(f"{prefix} [yellow]\u26a0[/yellow] {message}")


def error(message: str):
    """Log an error message."""
    prefix = _format_prefix()
    console.print(f"{prefix} [red]\u2717 Error:[/red] {message}")


def header(title: str, subtitle: str = ""):
    """Print a header panel."""
    reset_timer()
    if subtitle:
        console.print(Panel.fit(
            f"[bold cyan]{title}[/bold cyan] - {subtitle}",
            border_style="cyan"
        ))
    else:
        console.print(Panel.fit(
            f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan"
        ))


def complete():
    """Print completion message with total time."""
    total = time.time() - _start_time if _start_time else 0
    console.print(Panel.fit(
        f"[bold green]Analysis complete![/bold green]\nTotal time: {total:.2f}s",
        border_style="green"
    ))


def config_table(settings: dict, title: str = "Configuration") -> Table:
    """
    Create a configuration table.

    Parameters
    ----------
    settings : dict
        Dictionary of setting name -> value
    title : str
        Table title

    Returns
    -------
    table : Table
        Rich Table object
    """
    table = Table(show_header=False, box=None, padding=(0, 2), title=title if title else None)
    table.add_column("Setting", style="dim")
    table.add_column("Value", style="green")

    for key, value in settings.items():
        table.add_row(key, str(value))

    return table


def print_table(table: Table):
    """Print a table."""
    console.print(table)
    console.print()


def results_table(title: str = "Results") -> Table:
    """
    Create a results table.

    Parameters
    ----------
    title : str
        Table title

    Returns
    -------
    table : Table
        Rich Table object
    """
    table = Table(title=title, box=None)
    return table


def section(title: str):
    """Print a section header."""
    prefix = _format_prefix()
    console.print(f"\n{prefix} [bold]{title}[/bold]")


def detail(message: str):
    """Print a detail/indented message (no timing)."""
    console.print(f"  {message}")


# For backwards compatibility with direct console usage
def print(*args, **kwargs):
    """Direct print passthrough."""
    console.print(*args, **kwargs)
