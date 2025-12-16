"""Main CLI entry point for mlip-plot."""

import click

from .describe import describe
from .density import density
from .rdf import rdf
from .hbond import hbond
from .diffusion import diffusion


@click.group()
@click.version_option()
def main():
    """
    MLIP Plot - Fast MD trajectory analysis with C++ backend.

    \b
    Available commands:
      describe   Show basic information about a trajectory file
      density    Calculate and plot density profiles along an axis
      rdf        Calculate and plot radial distribution functions
      hbond      Detect and analyze hydrogen bonds
      diffusion  Calculate diffusion coefficients from MSD

    \b
    Examples:
      mlip-plot describe trajectory.lammpstrj
      mlip-plot density trajectory.lammpstrj
      mlip-plot rdf trajectory.lammpstrj -p O-O -p O-H
      mlip-plot hbond trajectory.lammpstrj -d O -h H -a O
      mlip-plot diffusion trajectory.lammpstrj --dt 2.0

    \b
    Use 'mlip-plot <command> --help' for more information on a command.
    """
    pass


# Register all commands
main.add_command(describe)
main.add_command(density)
main.add_command(rdf)
main.add_command(hbond)
main.add_command(diffusion)


if __name__ == '__main__':
    main()
