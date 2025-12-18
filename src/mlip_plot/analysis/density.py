"""
Density profile analysis using C++ backend.
"""

from typing import Any, Dict, List, Optional, Tuple, Set, Union
import numpy as np

# Try to import C++ core, fall back to pure Python if not available
try:
    from mlip_plot._core import accumulate_density_frames, compute_density_histogram_per_frame
    HAS_CPP_CORE = True
except ImportError:
    HAS_CPP_CORE = False

from .statistics import (
    estimate_error, determine_optimal_block_count, check_trajectory_length
)


def _log(logger: Optional[Any], method: str, message: str):
    """Helper to log or print based on logger availability."""
    if logger is not None:
        getattr(logger, method)(message)
    else:
        print(message)

# Atomic masses (g/mol)
ATOMIC_MASSES = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'Na': 22.990, 'Cl': 35.45, 'Cu': 63.546, 'Pt': 195.084,
    'S': 32.065, 'F': 18.998, 'K': 39.098, 'Ca': 40.078,
    'Mg': 24.305, 'Fe': 55.845, 'Zn': 65.38, 'Ag': 107.868,
    'Au': 196.967, 'Si': 28.086, 'P': 30.974, 'Li': 6.941,
}

AVOGADRO = 6.02214076e23


def calculate_density_profile(
    frames: List[Dict],
    bin_size: float = 0.5,
    axis: int = 2,
    exclude_elements: Optional[Set[str]] = None,
    n_blocks: Union[int, str] = 5,
    logger: Optional[Any] = None
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]], Dict[str, Any]]:
    """
    Calculate mass density profile along specified axis.

    Uses C++ backend for fast histogram computation.

    Parameters
    ----------
    frames : list
        List of trajectory frames from read_lammpstrj
    bin_size : float
        Bin size in Angstroms
    axis : int
        Axis for density profile: 0=x, 1=y, 2=z (default: 2)
    exclude_elements : set, optional
        Elements to exclude (e.g., substrate atoms like {'Cu', 'Pt'})
    n_blocks : int or str
        Error estimation method:
        - int: Fixed number of blocks (default: 5)
        - 'autocorr': Determine block size from autocorrelation time
        - 'FlyPet': Flyvbjerg-Petersen blocking analysis
        - 'convergence': Block size convergence analysis
    logger : optional
        Logger instance for output messages

    Returns
    -------
    profiles : dict
        Dictionary mapping element name to tuple of:
        (z_centers, density, std_error)
    diagnostics : dict
        Dictionary with error estimation diagnostics per element (empty if using fixed blocks)
    """
    if not frames:
        raise ValueError("No frames provided")

    if exclude_elements is None:
        exclude_elements = set()

    # Get box dimensions from first frame
    box = frames[0]['box']
    axis_names = ['x', 'y', 'z']
    axis_name = axis_names[axis]

    lo_key = f'{axis_name}lo'
    hi_key = f'{axis_name}hi'
    axis_min = box[lo_key]
    axis_max = box[hi_key]

    # Calculate perpendicular area
    perp_axes = [a for a in [0, 1, 2] if a != axis]
    perp_names = [axis_names[a] for a in perp_axes]
    area = 1.0
    for pn in perp_names:
        area *= (box[f'{pn}hi'] - box[f'{pn}lo'])

    # Setup bins
    n_bins = int(np.ceil((axis_max - axis_min) / bin_size))
    z_edges = np.linspace(axis_min, axis_max, n_bins + 1)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    actual_bin_size = (axis_max - axis_min) / n_bins
    bin_volume = area * actual_bin_size  # Angstrom^3

    # Build element-to-type mapping
    if frames[0]['elements'] is not None:
        elem_to_type = {}
        type_to_elem = {}
        for elem, typ in zip(frames[0]['elements'], frames[0]['types']):
            if elem not in elem_to_type:
                type_idx = typ - 1  # 0-indexed
                elem_to_type[elem] = type_idx
                type_to_elem[type_idx] = elem
    else:
        # No element info, use type numbers
        unique_types = np.unique(frames[0]['types'])
        elem_to_type = {str(t): t - 1 for t in unique_types}
        type_to_elem = {t - 1: str(t) for t in unique_types}

    # Filter out excluded elements
    active_elements = {e for e in elem_to_type if e not in exclude_elements}
    active_type_indices = {elem_to_type[e] for e in active_elements}
    n_types = max(elem_to_type.values()) + 1

    n_frames = len(frames)
    diagnostics = {}

    # Determine error estimation approach
    if isinstance(n_blocks, int):
        # Fixed number of blocks
        _log(logger, 'detail', f"Block analysis: {n_blocks} blocks")
        profiles = _compute_density_fixed_blocks(
            frames, axis, n_bins, n_types, axis_min, axis_max,
            active_type_indices, elem_to_type, active_elements,
            z_centers, bin_volume, n_blocks, logger
        )
    elif isinstance(n_blocks, str):
        # Principled error estimation methods
        valid_methods = {'autocorr', 'FlyPet', 'convergence'}
        if n_blocks not in valid_methods:
            raise ValueError(f"Unknown error estimation method: {n_blocks}. "
                             f"Valid options: {', '.join(sorted(valid_methods))}")

        # Check trajectory length
        check_trajectory_length(n_frames, n_blocks, warn=(logger is not None))

        method_names = {
            'autocorr': 'Autocorrelation time',
            'FlyPet': 'Flyvbjerg-Petersen blocking',
            'convergence': 'Convergence analysis'
        }
        _log(logger, 'detail', f"Error estimation: {method_names[n_blocks]}")

        profiles, diagnostics = _compute_density_principled_error(
            frames, axis, n_bins, n_types, axis_min, axis_max,
            active_type_indices, elem_to_type, active_elements,
            z_centers, bin_volume, n_blocks, logger
        )
    else:
        raise ValueError(f"n_blocks must be int or str, got {type(n_blocks)}")

    return profiles, diagnostics


def _compute_density_fixed_blocks(
    frames: List[Dict],
    axis: int,
    n_bins: int,
    n_types: int,
    axis_min: float,
    axis_max: float,
    active_type_indices: Set[int],
    elem_to_type: Dict[str, int],
    active_elements: Set[str],
    z_centers: np.ndarray,
    bin_volume: float,
    n_blocks: int,
    logger: Optional[Any]
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """Compute density profiles with fixed block averaging."""
    n_frames = len(frames)
    frames_per_block = n_frames // n_blocks

    _log(logger, 'detail', f"Frames per block: {frames_per_block}")

    block_densities = {elem: [] for elem in active_elements}

    for block_idx in range(n_blocks):
        start_idx = block_idx * frames_per_block
        end_idx = (block_idx + 1) * frames_per_block if block_idx < n_blocks - 1 else n_frames
        block_frames = frames[start_idx:end_idx]

        # Compute histogram for this block
        histogram = _compute_histogram_batch(
            block_frames, axis, n_bins, n_types, axis_min, axis_max,
            active_type_indices, elem_to_type
        )

        # Convert to density for each element
        for elem in active_elements:
            type_idx = elem_to_type[elem]
            counts = histogram[type_idx]
            avg_counts = counts / len(block_frames)
            mass = ATOMIC_MASSES.get(elem, 1.0)
            density = (avg_counts * mass / AVOGADRO) / (bin_volume * 1e-24)
            block_densities[elem].append(density)

    # Calculate mean and standard error
    profiles = {}
    for elem in active_elements:
        bd = np.array(block_densities[elem])
        mean_density = np.mean(bd, axis=0)
        std_error = np.std(bd, axis=0, ddof=1) / np.sqrt(n_blocks)
        profiles[elem] = (z_centers, mean_density, std_error)

        _log(logger, 'success', f"{elem}: max density = {np.max(mean_density):.4f} g/cm^3")

    return profiles


def _compute_density_principled_error(
    frames: List[Dict],
    axis: int,
    n_bins: int,
    n_types: int,
    axis_min: float,
    axis_max: float,
    active_type_indices: Set[int],
    elem_to_type: Dict[str, int],
    active_elements: Set[str],
    z_centers: np.ndarray,
    bin_volume: float,
    method: str,
    logger: Optional[Any]
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]], Dict[str, Any]]:
    """
    Compute density profiles with principled error estimation.

    Uses per-frame histograms to determine optimal block size via
    autocorrelation, Flyvbjerg-Petersen, or convergence analysis,
    then computes errors using block averaging with the optimal block count.
    """
    n_frames = len(frames)

    # Step 1: Get per-frame density time series
    density_timeseries = _compute_density_timeseries(
        frames, axis, n_bins, n_types, axis_min, axis_max,
        active_type_indices, elem_to_type, active_elements, bin_volume
    )

    profiles = {}
    diagnostics = {}

    for elem in active_elements:
        elem_ts = density_timeseries[elem]  # shape: (n_frames, n_bins)
        mean_density = np.zeros(n_bins)
        std_error = np.zeros(n_bins)

        # For each bin, apply error estimation
        # Use representative bin (with highest signal) to determine block size
        representative_bin = np.argmax(np.mean(elem_ts, axis=0))
        representative_ts = elem_ts[:, representative_bin]

        # Determine optimal block count from representative bin
        if np.std(representative_ts) > 1e-15:
            result = estimate_error(representative_ts, method=method)
            optimal_n_blocks = result.n_effective_blocks

            _log(logger, 'detail',
                 f"{elem}: optimal block size = {result.optimal_block_size} "
                 f"({optimal_n_blocks} blocks)")

            # Store diagnostics
            diagnostics[elem] = result.diagnostics
        else:
            optimal_n_blocks = 5  # fallback

        # Apply block averaging with optimal block count
        for bin_idx in range(n_bins):
            bin_ts = elem_ts[:, bin_idx]
            mean_density[bin_idx] = np.mean(bin_ts)

            if np.std(bin_ts) < 1e-15:
                std_error[bin_idx] = 0.0
            else:
                # Use block averaging with determined block count
                block_size = n_frames // optimal_n_blocks
                if block_size < 1:
                    block_size = 1
                    optimal_n_blocks = n_frames

                block_means = []
                for b in range(optimal_n_blocks):
                    start = b * block_size
                    end = start + block_size if b < optimal_n_blocks - 1 else n_frames
                    block_means.append(np.mean(bin_ts[start:end]))

                if len(block_means) > 1:
                    std_error[bin_idx] = np.std(block_means, ddof=1) / np.sqrt(len(block_means))
                else:
                    std_error[bin_idx] = 0.0

        profiles[elem] = (z_centers, mean_density, std_error)

        _log(logger, 'success', f"{elem}: max density = {np.max(mean_density):.4f} g/cm^3")

    return profiles, diagnostics


def _compute_density_timeseries(
    frames: List[Dict],
    axis: int,
    n_bins: int,
    n_types: int,
    axis_min: float,
    axis_max: float,
    active_type_indices: Set[int],
    elem_to_type: Dict[str, int],
    active_elements: Set[str],
    bin_volume: float
) -> Dict[str, np.ndarray]:
    """
    Compute per-frame density values for each bin.

    Returns
    -------
    timeseries : dict
        Element -> ndarray of shape (n_frames, n_bins)
    """
    n_frames = len(frames)
    box = frames[0]['box']
    box_lo = (box['xlo'], box['ylo'], box['zlo'])
    box_hi = (box['xhi'], box['yhi'], box['zhi'])

    if HAS_CPP_CORE:
        # Use C++ backend for per-frame histograms
        positions_list = []
        types_list = []

        for frame in frames:
            types = frame['types'].copy() - 1  # Convert to 0-indexed
            mask = np.isin(types, list(active_type_indices))
            types[~mask] = -1  # Mark excluded types

            positions_list.append(frame['positions'].astype(np.float64))
            types_list.append(types.astype(np.int32))

        # Call C++ backend - returns (n_frames, n_types, n_bins)
        per_frame_hist = compute_density_histogram_per_frame(
            positions_list, types_list, box_lo, box_hi, axis, n_bins, n_types
        )
    else:
        # Pure Python fallback
        per_frame_hist = np.zeros((n_frames, n_types, n_bins), dtype=np.float64)
        bin_size = (axis_max - axis_min) / n_bins

        for f_idx, frame in enumerate(frames):
            positions = frame['positions']
            types = frame['types'] - 1  # 0-indexed

            for i in range(len(positions)):
                type_idx = types[i]
                if type_idx not in active_type_indices:
                    continue

                coord = positions[i, axis]
                bin_idx = int((coord - axis_min) / bin_size)
                bin_idx = max(0, min(bin_idx, n_bins - 1))
                per_frame_hist[f_idx, type_idx, bin_idx] += 1

    # Convert counts to density for each element
    timeseries = {}
    for elem in active_elements:
        type_idx = elem_to_type[elem]
        counts = per_frame_hist[:, type_idx, :]  # (n_frames, n_bins)
        mass = ATOMIC_MASSES.get(elem, 1.0)
        # density in g/cm^3
        density = (counts * mass / AVOGADRO) / (bin_volume * 1e-24)
        timeseries[elem] = density

    return timeseries


def _compute_histogram_batch(
    frames: List[Dict],
    axis: int,
    n_bins: int,
    n_types: int,
    axis_min: float,
    axis_max: float,
    active_type_indices: Set[int],
    elem_to_type: Dict[str, int]
) -> np.ndarray:
    """
    Compute histogram for a batch of frames using C++ backend.
    """
    box = frames[0]['box']
    box_lo = (box['xlo'], box['ylo'], box['zlo'])
    box_hi = (box['xhi'], box['yhi'], box['zhi'])

    if HAS_CPP_CORE:
        # Prepare data for C++ function
        positions_list = []
        types_list = []

        for frame in frames:
            # Filter to active types only by setting excluded types to -1
            types = frame['types'].copy() - 1  # Convert to 0-indexed
            mask = np.isin(types, list(active_type_indices))
            types[~mask] = -1  # Mark excluded types

            positions_list.append(frame['positions'].astype(np.float64))
            types_list.append(types.astype(np.int32))

        # Call C++ backend
        histogram = accumulate_density_frames(
            positions_list, types_list, box_lo, box_hi, axis, n_bins, n_types
        )
        return histogram

    else:
        # Pure Python fallback
        histogram = np.zeros((n_types, n_bins), dtype=np.float64)
        bin_size = (axis_max - axis_min) / n_bins

        for frame in frames:
            positions = frame['positions']
            types = frame['types'] - 1  # 0-indexed

            for i in range(len(positions)):
                type_idx = types[i]
                if type_idx not in active_type_indices:
                    continue

                coord = positions[i, axis]
                bin_idx = int((coord - axis_min) / bin_size)
                bin_idx = max(0, min(bin_idx, n_bins - 1))
                histogram[type_idx, bin_idx] += 1

        return histogram


def find_first_peak(
    z: np.ndarray,
    density: np.ndarray,
    min_height: Optional[float] = None,
    prominence: Optional[float] = None
) -> Tuple[Optional[float], Optional[float]]:
    """
    Find the first (highest) peak in density profile.

    Parameters
    ----------
    z : ndarray
        Z-coordinates
    density : ndarray
        Density values
    min_height : float, optional
        Minimum peak height (default: 10% of max)
    prominence : float, optional
        Required prominence (default: 5% of max)

    Returns
    -------
    peak_z : float or None
        Z-coordinate of first peak
    peak_density : float or None
        Density value at first peak
    """
    from scipy.signal import find_peaks

    if min_height is None:
        min_height = 0.1 * np.max(density)
    if prominence is None:
        prominence = 0.05 * np.max(density)

    peaks, properties = find_peaks(density, height=min_height, prominence=prominence)

    if len(peaks) == 0:
        return None, None

    # Get the highest peak
    peak_idx = peaks[np.argmax(properties['peak_heights'])]
    return z[peak_idx], density[peak_idx]


def extract_peaks_with_errors(
    profiles: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]
) -> Dict[str, Tuple[float, float, Optional[float]]]:
    """
    Extract first peak for each element with error estimation.

    Parameters
    ----------
    profiles : dict
        Output from calculate_density_profile

    Returns
    -------
    peaks : dict
        Dictionary mapping element to (peak_z, peak_density, peak_stderr)
    """
    peaks = {}

    for element, (z, density, std_error) in profiles.items():
        peak_z, peak_density = find_first_peak(z, density)

        if peak_z is not None:
            peak_stderr = None
            if std_error is not None:
                peak_idx = np.argmin(np.abs(z - peak_z))
                peak_stderr = std_error[peak_idx]

            peaks[element] = (peak_z, peak_density, peak_stderr)

    return peaks
