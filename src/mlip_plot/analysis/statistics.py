"""
Statistical utilities for time series analysis.

Provides:
1. Error estimation methods:
   - Autocorrelation method (integrated autocorrelation time)
   - Flyvbjerg-Petersen blocking transformation
   - Convergence analysis with plateau detection

2. Equilibration detection methods:
   - MSER (Marginal Standard Error Rule)
   - Geweke convergence diagnostic
   - Chodera automated equilibration detection

These methods are generic and can be applied to any 1D time series,
making them reusable across density, RDF, H-bond, and other analyses.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import warnings


@dataclass
class ErrorEstimationResult:
    """Result of error estimation analysis."""
    mean: float
    std_error: float
    method: str
    optimal_block_size: int
    n_effective_blocks: int
    diagnostics: Dict[str, Any]


# =============================================================================
# Autocorrelation Functions
# =============================================================================

def compute_autocorrelation(
    data: np.ndarray,
    max_lag: Optional[int] = None,
    method: str = 'fft'
) -> np.ndarray:
    """
    Compute normalized autocorrelation function C(t).

    C(t) = <(A_i - Ā)(A_{i+t} - Ā)> / <(A_i - Ā)²>

    Parameters
    ----------
    data : ndarray
        1D time series data
    max_lag : int, optional
        Maximum lag to compute (default: len(data) // 2)
    method : str
        'fft' for O(N log N) or 'direct' for O(N²)

    Returns
    -------
    acf : ndarray
        Normalized autocorrelation C(t) where C(0) = 1
    """
    n = len(data)
    if max_lag is None:
        max_lag = n // 2

    max_lag = min(max_lag, n - 1)

    # Center the data
    centered = data - np.mean(data)
    variance = np.var(data, ddof=0)

    if variance < 1e-15:
        # Constant data, no correlation structure
        return np.ones(max_lag + 1)

    if method == 'fft':
        return _compute_autocorrelation_fft(centered, variance, max_lag)
    else:
        return _compute_autocorrelation_direct(centered, variance, max_lag)


def _compute_autocorrelation_fft(
    centered: np.ndarray,
    variance: float,
    max_lag: int
) -> np.ndarray:
    """FFT-based autocorrelation computation O(N log N)."""
    n = len(centered)

    # Pad to next power of 2 for efficiency
    n_padded = 2 ** int(np.ceil(np.log2(2 * n - 1)))

    # FFT-based correlation
    fft_data = np.fft.rfft(centered, n=n_padded)
    acf_full = np.fft.irfft(fft_data * np.conj(fft_data), n=n_padded)

    # Normalize by number of overlapping points and variance
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        n_overlap = n - lag
        if n_overlap > 0:
            acf[lag] = acf_full[lag] / (n_overlap * variance)

    return acf


def _compute_autocorrelation_direct(
    centered: np.ndarray,
    variance: float,
    max_lag: int
) -> np.ndarray:
    """Direct autocorrelation computation O(N²)."""
    n = len(centered)
    acf = np.zeros(max_lag + 1)

    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            n_overlap = n - lag
            if n_overlap > 0:
                acf[lag] = np.sum(centered[:-lag] * centered[lag:]) / (n_overlap * variance)

    return acf


def compute_integrated_autocorrelation_time(
    acf: np.ndarray,
    cutoff_threshold: float = 0.05,
    auto_truncate: bool = True
) -> Tuple[float, int]:
    """
    Compute integrated autocorrelation time τ_int.

    τ_int = 1/2 + Σ_{t=1}^{M} C(t)

    where M is determined by truncation when C(t) < cutoff_threshold
    or C(t) crosses zero.

    Parameters
    ----------
    acf : ndarray
        Autocorrelation function from compute_autocorrelation()
    cutoff_threshold : float
        Truncate sum when |C(t)| < threshold (default: 0.05)
    auto_truncate : bool
        If True, truncate at first zero crossing or noise level

    Returns
    -------
    tau_int : float
        Integrated autocorrelation time
    cutoff_lag : int
        Lag at which summation was truncated
    """
    n = len(acf)
    tau_int = 0.5  # Start with 1/2 term

    cutoff_lag = n - 1
    for t in range(1, n):
        # Check truncation conditions
        if auto_truncate:
            # Stop at first zero crossing
            if acf[t] <= 0:
                cutoff_lag = t - 1
                break
            # Stop when ACF drops below threshold
            if acf[t] < cutoff_threshold:
                cutoff_lag = t
                break

        tau_int += acf[t]

    return tau_int, cutoff_lag


def compute_statistical_inefficiency(tau_int: float) -> float:
    """
    Compute statistical inefficiency g = 1 + 2 * τ_int.

    The effective sample size is N_eff = N / g.
    Block size should be >= g for uncorrelated blocks.

    Parameters
    ----------
    tau_int : float
        Integrated autocorrelation time

    Returns
    -------
    g : float
        Statistical inefficiency factor
    """
    return 1.0 + 2.0 * tau_int


def estimate_error_autocorr(
    time_series: np.ndarray,
    min_blocks: int = 5,
    safety_factor: float = 3.0,
    max_lag_fraction: float = 0.5
) -> ErrorEstimationResult:
    """
    Estimate error using autocorrelation method.

    1. Compute autocorrelation C(t)
    2. Calculate τ_int via integration
    3. Compute g = 1 + 2 * τ_int
    4. Block size b = ceil(safety_factor * g)
    5. Number of blocks n_b = floor(N / b)
    6. Standard error = σ / √n_b

    Parameters
    ----------
    time_series : ndarray
        1D time series of observable values
    min_blocks : int
        Minimum number of blocks required (default: 5)
    safety_factor : float
        Multiply g by this factor for block size (default: 2.0)
    max_lag_fraction : float
        Maximum lag as fraction of series length (default: 0.5)

    Returns
    -------
    result : ErrorEstimationResult
        Contains mean, std_error, and diagnostics including:
        - 'tau_int': integrated autocorrelation time
        - 'g': statistical inefficiency
        - 'acf': autocorrelation function array
    """
    n = len(time_series)
    mean = np.mean(time_series)
    std = np.std(time_series, ddof=1)

    if std < 1e-15:
        # Constant data
        return ErrorEstimationResult(
            mean=mean,
            std_error=0.0,
            method='autocorr',
            optimal_block_size=1,
            n_effective_blocks=n,
            diagnostics={'tau_int': 0.0, 'g': 1.0, 'acf': np.array([1.0])}
        )

    # Compute autocorrelation
    max_lag = int(n * max_lag_fraction)
    acf = compute_autocorrelation(time_series, max_lag=max_lag, method='fft')

    # Compute integrated autocorrelation time
    tau_int, cutoff_lag = compute_integrated_autocorrelation_time(acf)

    # Statistical inefficiency
    g = compute_statistical_inefficiency(tau_int)

    # Determine block size
    block_size = int(np.ceil(safety_factor * g))
    block_size = max(block_size, 1)

    # Ensure minimum number of blocks
    n_blocks = n // block_size
    if n_blocks < min_blocks:
        # Adjust block size to get at least min_blocks
        block_size = n // min_blocks
        block_size = max(block_size, 1)
        n_blocks = n // block_size

    if n_blocks < 2:
        warnings.warn(
            f"Trajectory too short for reliable autocorrelation analysis. "
            f"Only {n_blocks} block(s) with block_size={block_size}."
        )
        n_blocks = max(n_blocks, 1)

    # Compute standard error using block averaging
    std_error = _compute_block_std_error(time_series, n_blocks)

    return ErrorEstimationResult(
        mean=mean,
        std_error=std_error,
        method='autocorr',
        optimal_block_size=block_size,
        n_effective_blocks=n_blocks,
        diagnostics={
            'tau_int': tau_int,
            'g': g,
            'cutoff_lag': cutoff_lag,
            'acf': acf[:cutoff_lag + 1]
        }
    )


# =============================================================================
# Flyvbjerg-Petersen Blocking
# =============================================================================

def flyvbjerg_petersen_blocking(
    time_series: np.ndarray,
    max_levels: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Flyvbjerg-Petersen blocking transformation.

    At each level:
    1. Compute variance of mean: var_mean = var(A) / N
    2. Transform: A'_i = (A_{2i} + A_{2i+1}) / 2
    3. Repeat with N' = N/2

    The true variance of the mean is where this plateaus.

    Parameters
    ----------
    time_series : ndarray
        1D time series
    max_levels : int, optional
        Maximum blocking levels (default: log2(N) - 4, ensuring at least 16 points)

    Returns
    -------
    block_sizes : ndarray
        Array of block sizes [1, 2, 4, 8, ...]
    variances : ndarray
        Variance of mean at each level
    n_blocks : ndarray
        Number of blocks at each level
    """
    n = len(time_series)

    if max_levels is None:
        # Ensure at least 16 data points at highest level
        max_levels = max(1, int(np.log2(n)) - 4)

    block_sizes = []
    variances = []
    n_blocks_list = []

    data = time_series.copy()
    block_size = 1

    for level in range(max_levels + 1):
        n_current = len(data)
        if n_current < 2:
            break

        # Variance of the mean at this level
        var_mean = np.var(data, ddof=1) / n_current

        block_sizes.append(block_size)
        variances.append(var_mean)
        n_blocks_list.append(n_current)

        # Block transform: average consecutive pairs
        n_pairs = n_current // 2
        if n_pairs < 1:
            break

        data = 0.5 * (data[::2][:n_pairs] + data[1::2][:n_pairs])
        block_size *= 2

    return np.array(block_sizes), np.array(variances), np.array(n_blocks_list)


def estimate_error_flyvbjerg_petersen(
    time_series: np.ndarray,
    min_blocks_plateau: int = 8,
    plateau_rel_threshold: float = 0.1
) -> ErrorEstimationResult:
    """
    Estimate error using Flyvbjerg-Petersen blocking method.

    Parameters
    ----------
    time_series : ndarray
        1D time series
    min_blocks_plateau : int
        Minimum blocks required to identify plateau (default: 8)
    plateau_rel_threshold : float
        Relative change threshold for plateau detection (default: 0.1)

    Returns
    -------
    result : ErrorEstimationResult
        Contains mean, std_error, and diagnostics including:
        - 'block_sizes': array of block sizes tested
        - 'variances': variance at each level
        - 'plateau_level': level where plateau detected
    """
    n = len(time_series)
    mean = np.mean(time_series)

    if np.std(time_series) < 1e-15:
        return ErrorEstimationResult(
            mean=mean,
            std_error=0.0,
            method='FlyPet',
            optimal_block_size=1,
            n_effective_blocks=n,
            diagnostics={'block_sizes': np.array([1]), 'variances': np.array([0.0]), 'plateau_level': 0}
        )

    # Perform blocking transformation
    block_sizes, variances, n_blocks = flyvbjerg_petersen_blocking(time_series)

    # Find plateau (where variance stops increasing significantly)
    plateau_level = _detect_flypet_plateau(variances, n_blocks, min_blocks_plateau, plateau_rel_threshold)

    # Use variance at plateau level
    std_error = np.sqrt(variances[plateau_level])
    optimal_block_size = int(block_sizes[plateau_level])
    n_effective = int(n_blocks[plateau_level])

    return ErrorEstimationResult(
        mean=mean,
        std_error=std_error,
        method='FlyPet',
        optimal_block_size=optimal_block_size,
        n_effective_blocks=n_effective,
        diagnostics={
            'block_sizes': block_sizes,
            'variances': variances,
            'n_blocks': n_blocks,
            'plateau_level': plateau_level
        }
    )


def _detect_flypet_plateau(
    variances: np.ndarray,
    n_blocks: np.ndarray,
    min_blocks: int,
    rel_threshold: float
) -> int:
    """Detect plateau in Flyvbjerg-Petersen blocking curve."""
    n_levels = len(variances)

    if n_levels < 2:
        return 0

    # Find highest level with sufficient blocks
    valid_levels = np.where(n_blocks >= min_blocks)[0]
    if len(valid_levels) == 0:
        valid_levels = np.arange(n_levels)

    max_valid = valid_levels[-1]

    # Look for plateau: where relative change becomes small
    for i in range(1, max_valid + 1):
        if variances[i - 1] > 0:
            rel_change = (variances[i] - variances[i - 1]) / variances[i - 1]
            if rel_change < rel_threshold:
                return i

    # If no clear plateau, use highest valid level
    return max_valid


# =============================================================================
# Convergence Analysis
# =============================================================================

def compute_standard_error_vs_block_size(
    time_series: np.ndarray,
    block_sizes: Optional[np.ndarray] = None,
    min_blocks: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute standard error as function of block size.

    SE(b) = sqrt(1 / (n_b * (n_b - 1)) * sum((Ā_j - Ā)²))

    where n_b = floor(N / b) and Ā_j is the mean of block j.

    Parameters
    ----------
    time_series : ndarray
        1D time series
    block_sizes : ndarray, optional
        Block sizes to test (default: logarithmically spaced)
    min_blocks : int
        Minimum number of blocks for each size (default: 5)

    Returns
    -------
    block_sizes : ndarray
        Block sizes tested
    standard_errors : ndarray
        SE at each block size
    """
    n = len(time_series)
    overall_mean = np.mean(time_series)

    if block_sizes is None:
        # Generate logarithmically spaced block sizes
        max_block = n // min_blocks
        if max_block < 1:
            max_block = 1
        block_sizes = np.unique(np.logspace(0, np.log10(max_block), 30).astype(int))
        block_sizes = block_sizes[block_sizes >= 1]

    valid_sizes = []
    standard_errors = []

    for b in block_sizes:
        n_blocks = n // b
        if n_blocks < min_blocks:
            continue

        # Compute block means
        block_means = np.zeros(n_blocks)
        for j in range(n_blocks):
            start = j * b
            end = start + b
            block_means[j] = np.mean(time_series[start:end])

        # Standard error of the mean
        if n_blocks > 1:
            se = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
        else:
            se = 0.0

        valid_sizes.append(b)
        standard_errors.append(se)

    return np.array(valid_sizes), np.array(standard_errors)


def detect_plateau(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 3,
    rel_threshold: float = 0.05
) -> Tuple[int, float]:
    """
    Detect plateau in y(x) curve.

    Uses derivative-based detection: plateau where relative change
    is below threshold for `window` consecutive points.

    Parameters
    ----------
    x : ndarray
        X values (e.g., block sizes)
    y : ndarray
        Y values (e.g., standard errors)
    window : int
        Number of consecutive points for plateau (default: 3)
    rel_threshold : float
        Relative change threshold (default: 0.05)

    Returns
    -------
    plateau_idx : int
        Index where plateau begins
    plateau_value : float
        Average y-value in plateau region
    """
    n = len(y)
    if n < window + 1:
        return n - 1, y[-1] if n > 0 else 0.0

    # Compute relative changes
    rel_changes = np.zeros(n - 1)
    for i in range(n - 1):
        if y[i] > 0:
            rel_changes[i] = abs(y[i + 1] - y[i]) / y[i]
        else:
            rel_changes[i] = 0.0

    # Find first window where all changes are below threshold
    for start in range(n - window):
        if np.all(rel_changes[start:start + window - 1] < rel_threshold):
            plateau_value = np.mean(y[start:start + window])
            return start, plateau_value

    # No clear plateau found, return last point
    return n - 1, y[-1]


def estimate_error_convergence(
    time_series: np.ndarray,
    block_sizes: Optional[np.ndarray] = None,
    min_blocks: int = 5,
    plateau_window: int = 3,
    plateau_threshold: float = 0.05
) -> ErrorEstimationResult:
    """
    Estimate error using convergence/plateau method.

    Parameters
    ----------
    time_series : ndarray
        1D time series
    block_sizes : ndarray, optional
        Block sizes to test
    min_blocks : int
        Minimum blocks per block size (default: 5)
    plateau_window : int
        Window size for plateau detection (default: 3)
    plateau_threshold : float
        Relative threshold for plateau (default: 0.05)

    Returns
    -------
    result : ErrorEstimationResult
        Contains mean, std_error, and diagnostics including:
        - 'block_sizes': array tested
        - 'se_curve': SE at each block size
        - 'plateau_idx': where plateau detected
    """
    n = len(time_series)
    mean = np.mean(time_series)

    if np.std(time_series) < 1e-15:
        return ErrorEstimationResult(
            mean=mean,
            std_error=0.0,
            method='convergence',
            optimal_block_size=1,
            n_effective_blocks=n,
            diagnostics={'block_sizes': np.array([1]), 'se_curve': np.array([0.0]), 'plateau_idx': 0}
        )

    # Compute SE vs block size
    sizes, se_curve = compute_standard_error_vs_block_size(
        time_series, block_sizes=block_sizes, min_blocks=min_blocks
    )

    if len(sizes) == 0:
        # Fallback: not enough data
        std_error = np.std(time_series, ddof=1) / np.sqrt(n)
        return ErrorEstimationResult(
            mean=mean,
            std_error=std_error,
            method='convergence',
            optimal_block_size=1,
            n_effective_blocks=n,
            diagnostics={'block_sizes': np.array([1]), 'se_curve': np.array([std_error]), 'plateau_idx': 0}
        )

    # Detect plateau
    plateau_idx, plateau_value = detect_plateau(
        sizes, se_curve, window=plateau_window, rel_threshold=plateau_threshold
    )

    optimal_block_size = int(sizes[plateau_idx])
    n_effective = n // optimal_block_size

    return ErrorEstimationResult(
        mean=mean,
        std_error=plateau_value,
        method='convergence',
        optimal_block_size=optimal_block_size,
        n_effective_blocks=n_effective,
        diagnostics={
            'block_sizes': sizes,
            'se_curve': se_curve,
            'plateau_idx': plateau_idx
        }
    )


# =============================================================================
# Unified Entry Point
# =============================================================================

def estimate_error(
    time_series: np.ndarray,
    method: str,
    **kwargs
) -> ErrorEstimationResult:
    """
    Unified interface for error estimation.

    Parameters
    ----------
    time_series : ndarray
        1D time series of observable
    method : str
        'autocorr', 'FlyPet', or 'convergence'
    **kwargs
        Method-specific parameters

    Returns
    -------
    result : ErrorEstimationResult

    Raises
    ------
    ValueError
        If method is unknown
    """
    if method == 'autocorr':
        return estimate_error_autocorr(time_series, **kwargs)
    elif method == 'FlyPet':
        return estimate_error_flyvbjerg_petersen(time_series, **kwargs)
    elif method == 'convergence':
        return estimate_error_convergence(time_series, **kwargs)
    else:
        raise ValueError(f"Unknown error estimation method: {method}. "
                         f"Valid options: 'autocorr', 'FlyPet', 'convergence'")


def determine_optimal_block_count(
    time_series: np.ndarray,
    method: str,
    **kwargs
) -> int:
    """
    Determine optimal number of blocks for a time series.

    This is a convenience function that returns just the optimal
    block count, useful for the hybrid approach where we determine
    block size from per-frame data, then use aggregated computation.

    Parameters
    ----------
    time_series : ndarray
        1D time series of observable
    method : str
        'autocorr', 'FlyPet', or 'convergence'
    **kwargs
        Method-specific parameters

    Returns
    -------
    n_blocks : int
        Optimal number of blocks
    """
    result = estimate_error(time_series, method, **kwargs)
    return result.n_effective_blocks


# =============================================================================
# Helper Functions
# =============================================================================

def _compute_block_std_error(time_series: np.ndarray, n_blocks: int) -> float:
    """Compute standard error using block averaging."""
    n = len(time_series)
    block_size = n // n_blocks

    if block_size < 1 or n_blocks < 2:
        return np.std(time_series, ddof=1) / np.sqrt(n)

    block_means = np.zeros(n_blocks)
    for j in range(n_blocks):
        start = j * block_size
        end = start + block_size if j < n_blocks - 1 else n
        block_means[j] = np.mean(time_series[start:end])

    return np.std(block_means, ddof=1) / np.sqrt(n_blocks)


def check_trajectory_length(
    n_frames: int,
    method: str,
    warn: bool = True
) -> bool:
    """
    Check if trajectory is long enough for reliable error estimation.

    Parameters
    ----------
    n_frames : int
        Number of frames in trajectory
    method : str
        Error estimation method
    warn : bool
        If True, emit warning for short trajectories

    Returns
    -------
    sufficient : bool
        True if trajectory is long enough
    """
    min_frames = {
        'autocorr': 100,
        'FlyPet': 64,
        'convergence': 50
    }

    required = min_frames.get(method, 50)
    sufficient = n_frames >= required

    if not sufficient and warn:
        warnings.warn(
            f"Trajectory ({n_frames} frames) may be too short for reliable "
            f"'{method}' error estimation (recommended: >= {required}). "
            "Consider using fixed-block averaging or longer trajectory."
        )

    return sufficient


# =============================================================================
# Equilibration Detection
# =============================================================================

@dataclass
class EquilibrationResult:
    """Result of equilibration detection analysis."""
    t0: int  # Index where equilibration ends (first production frame)
    method: str
    n_frames_discarded: int
    n_frames_production: int
    fraction_discarded: float
    diagnostics: Dict[str, Any]


def detect_equilibration_mser(
    time_series: np.ndarray,
    batch_size: int = 1,
    min_frames_production: int = 10
) -> EquilibrationResult:
    """
    Detect equilibration using Marginal Standard Error Rule (MSER).

    MSER finds the truncation point that minimizes the standard error
    of the remaining data. The MSER statistic at truncation point d is:

        MSER(d) = Var(X[d:]) / (n - d)

    The optimal truncation point minimizes MSER(d).

    References
    ----------
    - White, K.P. Jr. (1997) Simulation 69:323-334
    - Sala et al. (2024) J. Chem. Theory Comput. 20:8559-8568

    Parameters
    ----------
    time_series : ndarray
        1D time series of observable values
    batch_size : int
        Batch data into groups of this size before analysis (default: 1)
    min_frames_production : int
        Minimum frames required in production region (default: 10)

    Returns
    -------
    result : EquilibrationResult
        Contains t0 (equilibration end index) and diagnostics
    """
    n = len(time_series)

    # Apply batching if requested
    if batch_size > 1:
        n_batches = n // batch_size
        if n_batches < min_frames_production + 1:
            batch_size = 1
            n_batches = n
        else:
            # Compute batch means
            data = np.array([
                np.mean(time_series[i * batch_size:(i + 1) * batch_size])
                for i in range(n_batches)
            ])
    else:
        data = time_series
        n_batches = n

    # Compute MSER statistic for each potential truncation point
    max_truncation = n_batches - min_frames_production
    if max_truncation < 1:
        # Not enough data, no equilibration detected
        return EquilibrationResult(
            t0=0,
            method='mser',
            n_frames_discarded=0,
            n_frames_production=n,
            fraction_discarded=0.0,
            diagnostics={'mser_values': np.array([]), 'batch_size': batch_size}
        )

    mser_values = np.zeros(max_truncation)

    for d in range(max_truncation):
        remaining = data[d:]
        n_remaining = len(remaining)
        if n_remaining > 1:
            variance = np.var(remaining, ddof=1)
            mser_values[d] = variance / n_remaining
        else:
            mser_values[d] = np.inf

    # Find minimum MSER
    optimal_d = np.argmin(mser_values)

    # Convert back to original frame index
    t0 = optimal_d * batch_size
    n_discarded = t0
    n_production = n - t0

    return EquilibrationResult(
        t0=t0,
        method='mser',
        n_frames_discarded=n_discarded,
        n_frames_production=n_production,
        fraction_discarded=n_discarded / n,
        diagnostics={
            'mser_values': mser_values,
            'optimal_batch_idx': optimal_d,
            'batch_size': batch_size
        }
    )


def detect_equilibration_geweke(
    time_series: np.ndarray,
    frac_start: float = 0.1,
    frac_end: float = 0.5,
    z_threshold: float = 2.0,
    n_intervals: int = 20
) -> EquilibrationResult:
    """
    Detect equilibration using Geweke convergence diagnostic.

    Compares means of initial and final portions of the time series.
    The z-score is computed as:

        z = (mean_start - mean_end) / sqrt(var_start + var_end)

    Equilibration is detected when |z| < z_threshold.

    References
    ----------
    - Geweke, J. (1992) Bayesian Statistics 4, Oxford University Press

    Parameters
    ----------
    time_series : ndarray
        1D time series of observable values
    frac_start : float
        Fraction of data for initial segment (default: 0.1)
    frac_end : float
        Fraction of data for final segment (default: 0.5)
    z_threshold : float
        Z-score threshold for convergence (default: 2.0)
    n_intervals : int
        Number of starting points to test (default: 20)

    Returns
    -------
    result : EquilibrationResult
        Contains t0 (equilibration end index) and diagnostics
    """
    n = len(time_series)

    # Compute spectral density estimate for variance
    def spectral_variance(data):
        """Estimate variance using spectral density at frequency 0."""
        if len(data) < 2:
            return np.var(data) if len(data) > 0 else 0.0

        # Simple spectral estimate using autocorrelation
        centered = data - np.mean(data)
        n_data = len(centered)

        # Use first few lags for spectral estimate
        max_lag = min(n_data // 4, 50)
        acf = compute_autocorrelation(data, max_lag=max_lag, method='fft')

        # Spectral density at zero frequency
        tau_int = 0.5 + np.sum(acf[1:])
        tau_int = max(tau_int, 0.5)  # Ensure positive
        g = 1 + 2 * tau_int

        return np.var(data, ddof=1) * g / n_data

    # Test different starting points
    n_end = int(n * frac_end)
    end_segment = time_series[-n_end:]
    mean_end = np.mean(end_segment)
    var_end = spectral_variance(end_segment)

    # Test starting points
    test_points = np.linspace(0, n - n_end - int(n * frac_start), n_intervals, dtype=int)
    z_scores = np.zeros(len(test_points))

    for i, t in enumerate(test_points):
        n_start = int(n * frac_start)
        start_segment = time_series[t:t + n_start]

        if len(start_segment) < 2:
            z_scores[i] = np.inf
            continue

        mean_start = np.mean(start_segment)
        var_start = spectral_variance(start_segment)

        # Compute z-score
        denom = np.sqrt(var_start + var_end)
        if denom > 1e-15:
            z_scores[i] = abs(mean_start - mean_end) / denom
        else:
            z_scores[i] = 0.0

    # Find first point where z-score is below threshold
    converged_mask = z_scores < z_threshold
    if np.any(converged_mask):
        first_converged_idx = np.argmax(converged_mask)
        t0 = test_points[first_converged_idx]
    else:
        # No convergence detected, use point with lowest z-score
        t0 = test_points[np.argmin(z_scores)]

    n_discarded = t0
    n_production = n - t0

    return EquilibrationResult(
        t0=t0,
        method='geweke',
        n_frames_discarded=n_discarded,
        n_frames_production=n_production,
        fraction_discarded=n_discarded / n,
        diagnostics={
            'test_points': test_points,
            'z_scores': z_scores,
            'z_threshold': z_threshold,
            'converged': np.any(converged_mask)
        }
    )


def detect_equilibration_chodera(
    time_series: np.ndarray,
    nskip: int = 1,
    min_frames_production: int = 10
) -> EquilibrationResult:
    """
    Detect equilibration using Chodera's method.

    Maximizes the number of effectively uncorrelated samples in the
    production region. For each potential truncation point t0:

        N_eff(t0) = (N - t0) / g(t0)

    where g(t0) is the statistical inefficiency of data[t0:].
    The optimal t0 maximizes N_eff.

    References
    ----------
    - Chodera, J.D. (2016) J. Chem. Theory Comput. 12:1799-1805

    Parameters
    ----------
    time_series : ndarray
        1D time series of observable values
    nskip : int
        Skip every nskip frames when scanning (default: 1)
    min_frames_production : int
        Minimum frames required in production region (default: 10)

    Returns
    -------
    result : EquilibrationResult
        Contains t0 (equilibration end index) and diagnostics
    """
    n = len(time_series)

    if n < min_frames_production + 1:
        return EquilibrationResult(
            t0=0,
            method='chodera',
            n_frames_discarded=0,
            n_frames_production=n,
            fraction_discarded=0.0,
            diagnostics={'n_eff_values': np.array([n]), 'g_values': np.array([1.0])}
        )

    # Test truncation points
    max_t0 = n - min_frames_production
    test_points = np.arange(0, max_t0, nskip)

    n_eff_values = np.zeros(len(test_points))
    g_values = np.zeros(len(test_points))

    for i, t0 in enumerate(test_points):
        remaining = time_series[t0:]
        n_remaining = len(remaining)

        if n_remaining < 2:
            n_eff_values[i] = 0
            g_values[i] = np.inf
            continue

        # Compute statistical inefficiency
        try:
            acf = compute_autocorrelation(remaining, max_lag=n_remaining // 2, method='fft')
            tau_int, _ = compute_integrated_autocorrelation_time(acf)
            g = compute_statistical_inefficiency(tau_int)
        except Exception:
            g = 1.0

        g = max(g, 1.0)  # Ensure g >= 1
        g_values[i] = g
        n_eff_values[i] = n_remaining / g

    # Find t0 that maximizes N_eff
    optimal_idx = np.argmax(n_eff_values)
    t0 = test_points[optimal_idx]

    n_discarded = t0
    n_production = n - t0

    return EquilibrationResult(
        t0=t0,
        method='chodera',
        n_frames_discarded=n_discarded,
        n_frames_production=n_production,
        fraction_discarded=n_discarded / n,
        diagnostics={
            'test_points': test_points,
            'n_eff_values': n_eff_values,
            'g_values': g_values,
            'max_n_eff': n_eff_values[optimal_idx]
        }
    )


def detect_equilibration(
    time_series: np.ndarray,
    method: str,
    **kwargs
) -> EquilibrationResult:
    """
    Unified interface for equilibration detection.

    Parameters
    ----------
    time_series : ndarray
        1D time series of observable
    method : str
        'mser', 'geweke', or 'chodera'
    **kwargs
        Method-specific parameters

    Returns
    -------
    result : EquilibrationResult

    Raises
    ------
    ValueError
        If method is unknown
    """
    if method == 'mser':
        return detect_equilibration_mser(time_series, **kwargs)
    elif method == 'geweke':
        return detect_equilibration_geweke(time_series, **kwargs)
    elif method == 'chodera':
        return detect_equilibration_chodera(time_series, **kwargs)
    else:
        raise ValueError(
            f"Unknown equilibration detection method: {method}. "
            f"Valid options: 'mser', 'geweke', 'chodera'"
        )


def check_equilibration_trajectory_length(
    n_frames: int,
    method: str,
    warn: bool = True
) -> bool:
    """
    Check if trajectory is long enough for reliable equilibration detection.

    Parameters
    ----------
    n_frames : int
        Number of frames in trajectory
    method : str
        Equilibration detection method
    warn : bool
        If True, emit warning for short trajectories

    Returns
    -------
    sufficient : bool
        True if trajectory is long enough
    """
    min_frames = {
        'mser': 50,
        'geweke': 100,
        'chodera': 50
    }

    required = min_frames.get(method, 50)
    sufficient = n_frames >= required

    if not sufficient and warn:
        warnings.warn(
            f"Trajectory ({n_frames} frames) may be too short for reliable "
            f"'{method}' equilibration detection (recommended: >= {required}). "
            "Consider using a fixed skip-frames value."
        )

    return sufficient
