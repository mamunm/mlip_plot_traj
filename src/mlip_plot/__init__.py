"""
MLIP Plot - Fast MD trajectory analysis with C++ backend
"""

__version__ = "0.1.0"

from .io.lammps import read_lammpstrj

# Analysis functions
from .analysis import (
    # Density
    calculate_density_profile,
    # RDF
    compute_rdf,
    apply_smoothing,
    # Diffusion
    calculate_msd,
    compute_diffusion,
    fit_diffusion_coefficient,
    find_water_molecules,
    compute_water_com_positions,
    # Statistics - Error estimation
    ErrorEstimationResult,
    compute_autocorrelation,
    compute_integrated_autocorrelation_time,
    compute_statistical_inefficiency,
    estimate_error_autocorr,
    flyvbjerg_petersen_blocking,
    estimate_error_flyvbjerg_petersen,
    compute_standard_error_vs_block_size,
    detect_plateau,
    estimate_error_convergence,
    estimate_error,
    determine_optimal_block_count,
    check_trajectory_length,
    # Statistics - Equilibration detection
    EquilibrationResult,
    detect_equilibration_mser,
    detect_equilibration_geweke,
    detect_equilibration_chodera,
    detect_equilibration,
    check_equilibration_trajectory_length,
)

__all__ = [
    # IO
    "read_lammpstrj",
    # Density
    "calculate_density_profile",
    # RDF
    "compute_rdf",
    "apply_smoothing",
    # Diffusion
    "calculate_msd",
    "compute_diffusion",
    "fit_diffusion_coefficient",
    "find_water_molecules",
    "compute_water_com_positions",
    # Statistics - Error estimation
    "ErrorEstimationResult",
    "compute_autocorrelation",
    "compute_integrated_autocorrelation_time",
    "compute_statistical_inefficiency",
    "estimate_error_autocorr",
    "flyvbjerg_petersen_blocking",
    "estimate_error_flyvbjerg_petersen",
    "compute_standard_error_vs_block_size",
    "detect_plateau",
    "estimate_error_convergence",
    "estimate_error",
    "determine_optimal_block_count",
    "check_trajectory_length",
    # Statistics - Equilibration detection
    "EquilibrationResult",
    "detect_equilibration_mser",
    "detect_equilibration_geweke",
    "detect_equilibration_chodera",
    "detect_equilibration",
    "check_equilibration_trajectory_length",
]
