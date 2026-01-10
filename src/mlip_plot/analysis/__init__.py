"""Analysis modules for trajectory data."""

from .density import calculate_density_profile
from .rdf import compute_rdf, apply_smoothing
from .diffusion import (
    calculate_msd,
    compute_diffusion,
    fit_diffusion_coefficient,
    find_water_molecules,
    compute_water_com_positions,
)
from .hbond import (
    detect_hbonds_frame,
    compute_hbond_analysis,
)
from .statistics import (
    # Error estimation
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
    # Equilibration detection
    EquilibrationResult,
    detect_equilibration_mser,
    detect_equilibration_geweke,
    detect_equilibration_chodera,
    detect_equilibration,
    check_equilibration_trajectory_length,
)

__all__ = [
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
    # H-bond
    "detect_hbonds_frame",
    "compute_hbond_analysis",
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
