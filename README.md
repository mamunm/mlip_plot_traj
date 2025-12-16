# MLIP Plot

Fast MD trajectory analysis with C++ backend and matplotlib plotting.

## Installation

```bash
uv pip install -e .
```

## Usage

```bash
# Density profile
mlip-plot density trajectory.lammpstrj

# Radial distribution function
mlip-plot rdf trajectory.lammpstrj

# Diffusion coefficient
mlip-plot diffusion trajectory.lammpstrj --dt 2.0

# Region-based diffusion with block averaging
mlip-plot diffusion trajectory.lammpstrj --z-interface 20 --d-bulk 10 --n-blocks 5
```

## Features

- Density profiles along any axis
- Radial distribution functions (RDF)
- Diffusion coefficients (planar, perpendicular, total)
- Region-based analysis (interface/bulk)
- Error estimation via block averaging
- CSV export
