"""
Radial Distribution Function (RDF) analysis using C++ backend.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

from .statistics import estimate_error, check_trajectory_length

# Try to import C++ core
try:
    from mlip_plot._core import accumulate_rdf_frames
    HAS_CPP_CORE = True
except ImportError:
    HAS_CPP_CORE = False


def compute_rdf(
    frames: List[Dict],
    type1: str,
    type2: str,
    rmin: float = 0.05,
    rmax: float = 6.0,
    n_bins: int = 100,
    n_blocks: Union[int, str] = 5,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Compute radial distribution function g(r) for a pair of atom types.

    Parameters
    ----------
    frames : list
        List of trajectory frames from read_lammpstrj
    type1 : str
        First element type (e.g., 'O')
    type2 : str
        Second element type (e.g., 'O' for O-O RDF)
    rmin : float
        Minimum distance in Angstroms
    rmax : float
        Maximum distance in Angstroms
    n_bins : int
        Number of histogram bins
    n_blocks : int or str
        Error estimation method:
        - int: Fixed number of blocks (default: 5)
        - 'autocorr': Determine block size from autocorrelation time
        - 'FlyPet': Flyvbjerg-Petersen blocking analysis
        - 'convergence': Block size convergence analysis

    Returns
    -------
    r : ndarray
        Bin centers (distances)
    g_r : ndarray
        RDF values
    std_error : ndarray or None
        Standard error
    diagnostics : dict
        Error estimation diagnostics (empty if using fixed blocks)
    """
    if not frames:
        raise ValueError("No frames provided")

    # Build element-to-type mapping
    if frames[0]['elements'] is not None:
        elem_to_type = {}
        for elem, typ in zip(frames[0]['elements'], frames[0]['types']):
            if elem not in elem_to_type:
                elem_to_type[elem] = typ - 1  # 0-indexed
    else:
        raise ValueError("Frames must contain element information for RDF calculation")

    if type1 not in elem_to_type:
        raise ValueError(f"Element '{type1}' not found in trajectory")
    if type2 not in elem_to_type:
        raise ValueError(f"Element '{type2}' not found in trajectory")

    type1_idx = elem_to_type[type1]
    type2_idx = elem_to_type[type2]

    # Get cell vectors from first frame
    box = frames[0]['box']
    cell_vectors = np.array([
        [box['xhi'] - box['xlo'], 0.0, 0.0],
        [0.0, box['yhi'] - box['ylo'], 0.0],
        [0.0, 0.0, box['zhi'] - box['zlo']]
    ], dtype=np.float64)

    # Bin setup
    bin_edges = np.linspace(rmin, rmax, n_bins + 1)
    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    n_frames = len(frames)
    is_self_pair = (type1 == type2)
    diagnostics = {}

    if isinstance(n_blocks, int):
        # Fixed number of blocks
        g_r, std_error = _compute_rdf_fixed_blocks(
            frames, cell_vectors, type1_idx, type2_idx,
            rmin, rmax, n_bins, is_self_pair, n_blocks
        )
    elif isinstance(n_blocks, str):
        # Principled error estimation methods
        valid_methods = {'autocorr', 'FlyPet', 'convergence'}
        if n_blocks not in valid_methods:
            raise ValueError(f"Unknown error estimation method: {n_blocks}. "
                             f"Valid options: {', '.join(sorted(valid_methods))}")

        # Check trajectory length
        check_trajectory_length(n_frames, n_blocks, warn=True)

        g_r, std_error, diagnostics = _compute_rdf_principled_error(
            frames, cell_vectors, type1_idx, type2_idx,
            rmin, rmax, n_bins, is_self_pair, n_blocks
        )
    else:
        raise ValueError(f"n_blocks must be int or str, got {type(n_blocks)}")

    return r, g_r, std_error, diagnostics


def _compute_rdf_fixed_blocks(
    frames: List[Dict],
    cell_vectors: np.ndarray,
    type1_idx: int,
    type2_idx: int,
    rmin: float,
    rmax: float,
    n_bins: int,
    is_self_pair: bool,
    n_blocks: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute RDF with fixed block averaging."""
    n_frames = len(frames)
    frames_per_block = n_frames // n_blocks

    block_rdfs = []

    for block_idx in range(n_blocks):
        start_idx = block_idx * frames_per_block
        end_idx = (block_idx + 1) * frames_per_block if block_idx < n_blocks - 1 else n_frames
        block_frames = frames[start_idx:end_idx]

        g_r_block = _compute_rdf_block(
            block_frames, cell_vectors, type1_idx, type2_idx,
            rmin, rmax, n_bins, is_self_pair
        )
        block_rdfs.append(g_r_block)

    block_rdfs = np.array(block_rdfs)
    g_r = np.mean(block_rdfs, axis=0)
    std_error = np.std(block_rdfs, axis=0, ddof=1) / np.sqrt(n_blocks)

    return g_r, std_error


def _compute_rdf_principled_error(
    frames: List[Dict],
    cell_vectors: np.ndarray,
    type1_idx: int,
    type2_idx: int,
    rmin: float,
    rmax: float,
    n_bins: int,
    is_self_pair: bool,
    method: str
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Compute RDF with principled error estimation."""
    n_frames = len(frames)

    # Compute per-frame RDF time series
    rdf_timeseries = np.zeros((n_frames, n_bins))
    for i, frame in enumerate(frames):
        rdf_timeseries[i] = _compute_rdf_single_frame(
            frame, cell_vectors, type1_idx, type2_idx,
            rmin, rmax, n_bins, is_self_pair
        )

    # Use representative bin (highest signal) to determine block size
    representative_bin = np.argmax(np.mean(rdf_timeseries, axis=0))
    representative_ts = rdf_timeseries[:, representative_bin]

    diagnostics = {}

    if np.std(representative_ts) > 1e-15:
        result = estimate_error(representative_ts, method=method)
        optimal_n_blocks = result.n_effective_blocks
        diagnostics = result.diagnostics
    else:
        optimal_n_blocks = 5  # fallback

    # Apply block averaging with optimal block count
    g_r = np.mean(rdf_timeseries, axis=0)
    std_error = np.zeros(n_bins)

    block_size = n_frames // optimal_n_blocks
    if block_size < 1:
        block_size = 1
        optimal_n_blocks = n_frames

    for bin_idx in range(n_bins):
        bin_ts = rdf_timeseries[:, bin_idx]

        if np.std(bin_ts) < 1e-15:
            std_error[bin_idx] = 0.0
        else:
            block_means = []
            for b in range(optimal_n_blocks):
                start = b * block_size
                end = start + block_size if b < optimal_n_blocks - 1 else n_frames
                block_means.append(np.mean(bin_ts[start:end]))

            if len(block_means) > 1:
                std_error[bin_idx] = np.std(block_means, ddof=1) / np.sqrt(len(block_means))
            else:
                std_error[bin_idx] = 0.0

    return g_r, std_error, diagnostics


def _compute_rdf_single_frame(
    frame: Dict,
    cell_vectors: np.ndarray,
    type1_idx: int,
    type2_idx: int,
    rmin: float,
    rmax: float,
    n_bins: int,
    is_self_pair: bool
) -> np.ndarray:
    """Compute RDF for a single frame."""
    dr = (rmax - rmin) / n_bins
    bin_edges = np.linspace(rmin, rmax, n_bins + 1)
    histogram = np.zeros(n_bins)
    volume = np.abs(np.linalg.det(cell_vectors))

    positions = frame['positions']
    types = frame['types'] - 1  # 0-indexed

    # Get indices for each type
    idx1 = np.where(types == type1_idx)[0]
    idx2 = np.where(types == type2_idx)[0]

    n1, n2 = len(idx1), len(idx2)

    # Precompute inverse cell matrix for PBC
    inv_cell = np.linalg.inv(cell_vectors)

    if is_self_pair:
        total_pairs = n1 * (n1 - 1) / 2.0

        for i in range(len(idx1)):
            for j in range(i + 1, len(idx1)):
                ai, aj = idx1[i], idx1[j]
                dr_vec = positions[ai] - positions[aj]

                # Apply PBC
                frac = inv_cell @ dr_vec
                frac -= np.round(frac)
                dr_vec = cell_vectors.T @ frac

                r_ij = np.linalg.norm(dr_vec)

                if rmin <= r_ij < rmax:
                    bin_idx = int((r_ij - rmin) / dr)
                    if 0 <= bin_idx < n_bins:
                        histogram[bin_idx] += 1.0
    else:
        total_pairs = n1 * n2

        for ai in idx1:
            for aj in idx2:
                dr_vec = positions[ai] - positions[aj]

                # Apply PBC
                frac = inv_cell @ dr_vec
                frac -= np.round(frac)
                dr_vec = cell_vectors.T @ frac

                r_ij = np.linalg.norm(dr_vec)

                if rmin <= r_ij < rmax:
                    bin_idx = int((r_ij - rmin) / dr)
                    if 0 <= bin_idx < n_bins:
                        histogram[bin_idx] += 1.0

    # Normalize to get g(r)
    g_r = np.zeros(n_bins)

    for i in range(n_bins):
        r_left = bin_edges[i]
        if r_left < 1e-9:
            continue

        shell_volume = 4.0 * np.pi * r_left * r_left * dr
        pair_density = total_pairs / volume if total_pairs > 0 else 0
        norm_factor = pair_density * shell_volume

        if norm_factor > 1e-9:
            g_r[i] = histogram[i] / norm_factor

    return g_r


def _compute_rdf_block(
    frames: List[Dict],
    cell_vectors: np.ndarray,
    type1_idx: int,
    type2_idx: int,
    rmin: float,
    rmax: float,
    n_bins: int,
    is_self_pair: bool
) -> np.ndarray:
    """Compute RDF for a block of frames."""
    dr = (rmax - rmin) / n_bins
    bin_edges = np.linspace(rmin, rmax, n_bins + 1)
    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    volume = np.abs(np.linalg.det(cell_vectors))

    if HAS_CPP_CORE:
        # Prepare data for C++ function
        positions_list = []
        types_list = []

        for frame in frames:
            positions_list.append(frame['positions'].astype(np.float64))
            types_list.append((frame['types'] - 1).astype(np.int32))  # 0-indexed

        # Call C++ backend
        histogram, total_pairs, total_volume, n_frames = accumulate_rdf_frames(
            positions_list, types_list, cell_vectors,
            rmin, rmax, n_bins, type1_idx, type2_idx
        )

        # Normalize to get g(r)
        g_r = np.zeros(n_bins)
        avg_pairs = total_pairs / n_frames if n_frames > 0 else 0
        avg_volume = total_volume / n_frames if n_frames > 0 else volume

        for i in range(n_bins):
            r_left = bin_edges[i]
            if r_left < 1e-9:
                continue

            shell_volume = 4.0 * np.pi * r_left * r_left * dr
            pair_density = avg_pairs / avg_volume
            norm_factor = pair_density * shell_volume

            if norm_factor > 1e-9:
                g_r[i] = histogram[i] / (n_frames * norm_factor)

    else:
        # Pure Python fallback
        g_r = _compute_rdf_python(
            frames, cell_vectors, type1_idx, type2_idx,
            rmin, rmax, n_bins, is_self_pair
        )

    return g_r


def _compute_rdf_python(
    frames: List[Dict],
    cell_vectors: np.ndarray,
    type1_idx: int,
    type2_idx: int,
    rmin: float,
    rmax: float,
    n_bins: int,
    is_self_pair: bool
) -> np.ndarray:
    """Pure Python RDF computation (slow fallback)."""
    dr = (rmax - rmin) / n_bins
    bin_edges = np.linspace(rmin, rmax, n_bins + 1)
    histogram = np.zeros(n_bins)
    volume = np.abs(np.linalg.det(cell_vectors))

    total_pairs = 0.0
    n_frames = len(frames)

    # Precompute inverse cell matrix for PBC
    inv_cell = np.linalg.inv(cell_vectors)

    for frame in frames:
        positions = frame['positions']
        types = frame['types'] - 1  # 0-indexed

        # Get indices for each type
        idx1 = np.where(types == type1_idx)[0]
        idx2 = np.where(types == type2_idx)[0]

        n1, n2 = len(idx1), len(idx2)

        if is_self_pair:
            total_pairs += n1 * (n1 - 1) / 2.0

            for i in range(len(idx1)):
                for j in range(i + 1, len(idx1)):
                    ai, aj = idx1[i], idx1[j]
                    dr_vec = positions[ai] - positions[aj]

                    # Apply PBC
                    frac = inv_cell @ dr_vec
                    frac -= np.round(frac)
                    dr_vec = cell_vectors.T @ frac

                    r_ij = np.linalg.norm(dr_vec)

                    if rmin <= r_ij < rmax:
                        bin_idx = int((r_ij - rmin) / dr)
                        if 0 <= bin_idx < n_bins:
                            histogram[bin_idx] += 1.0
        else:
            total_pairs += n1 * n2

            for ai in idx1:
                for aj in idx2:
                    dr_vec = positions[ai] - positions[aj]

                    # Apply PBC
                    frac = inv_cell @ dr_vec
                    frac -= np.round(frac)
                    dr_vec = cell_vectors.T @ frac

                    r_ij = np.linalg.norm(dr_vec)

                    if rmin <= r_ij < rmax:
                        bin_idx = int((r_ij - rmin) / dr)
                        if 0 <= bin_idx < n_bins:
                            histogram[bin_idx] += 1.0

    # Normalize
    g_r = np.zeros(n_bins)
    avg_pairs = total_pairs / n_frames if n_frames > 0 else 0

    for i in range(n_bins):
        r_left = bin_edges[i]
        if r_left < 1e-9:
            continue

        shell_volume = 4.0 * np.pi * r_left * r_left * dr
        pair_density = avg_pairs / volume
        norm_factor = pair_density * shell_volume

        if norm_factor > 1e-9:
            g_r[i] = histogram[i] / (n_frames * norm_factor)

    return g_r


def apply_smoothing(
    data: np.ndarray,
    method: str = 'none',
    sigma: float = 1.5,
    window_size: int = 9,
    polyorder: int = 3,
    spline_smoothness: float = 0.005
) -> np.ndarray:
    """
    Apply smoothing to RDF data.

    Parameters
    ----------
    data : ndarray
        Data to smooth
    method : str
        'none', 'gaussian', 'savgol', 'moving_avg', 'spline'
    sigma : float
        Sigma for Gaussian smoothing
    window_size : int
        Window size for Savitzky-Golay or moving average
    polyorder : int
        Polynomial order for Savitzky-Golay
    spline_smoothness : float
        Smoothing factor for spline

    Returns
    -------
    smoothed : ndarray
        Smoothed data
    """
    if method == 'none' or method is None:
        return data.copy()

    smoothed = data.copy()

    if method == 'gaussian':
        smoothed = gaussian_filter1d(data, sigma=sigma, mode='reflect')

    elif method == 'savgol':
        if window_size % 2 == 0:
            window_size += 1
        if window_size <= polyorder:
            window_size = polyorder + 2
            if window_size % 2 == 0:
                window_size += 1
        if len(data) > window_size:
            smoothed = savgol_filter(data, window_size, polyorder, mode='interp')

    elif method == 'moving_avg':
        if window_size > 1:
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(data, kernel, mode='same')

    elif method == 'spline':
        x = np.arange(len(data))
        weights = np.ones_like(data)
        weights[data < 0.1] = 0.5
        spline = UnivariateSpline(x, data, w=weights, s=spline_smoothness * len(data))
        smoothed = spline(x)

    # Ensure non-negative
    smoothed = np.maximum(smoothed, 0)

    return smoothed


def compute_coordination_number(
    r: np.ndarray,
    g_r: np.ndarray,
    rho: float,
    r_cutoff: float
) -> float:
    """
    Compute coordination number by integrating RDF.

    Parameters
    ----------
    r : ndarray
        Distances
    g_r : ndarray
        RDF values
    rho : float
        Number density (atoms per Å³)
    r_cutoff : float
        Integration cutoff (first minimum of g(r))

    Returns
    -------
    n_coord : float
        Coordination number
    """
    mask = r <= r_cutoff
    dr = r[1] - r[0] if len(r) > 1 else 1.0

    # N = 4π ρ ∫ r² g(r) dr
    integrand = 4 * np.pi * rho * r[mask]**2 * g_r[mask]
    n_coord = np.trapz(integrand, r[mask])

    return n_coord
