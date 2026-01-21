"""
Water molecule reorientation analysis for MD trajectories.

Computes orientation distributions P(cos θ) and P(cos φ) for water molecules
relative to the surface normal, with support for region-based analysis
(interface_a, bulk, interface_b).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .diffusion import find_metal_surfaces, SUPPORTED_METALS

try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _log(logger: Optional[Any], method: str, message: str, verbose: bool = True):
    """Helper to log or print based on logger availability."""
    if not verbose:
        return
    if logger is not None:
        getattr(logger, method)(message)
    else:
        print(message)


def define_z_regions_reorientation(
    z_lo: float,
    z_hi: float,
    z_interface: float,
    d_bulk: float,
    lower_metal_surface_z: Optional[float] = None,
    upper_metal_surface_z: Optional[float] = None
) -> Dict[str, Tuple[float, float]]:
    """
    Define three z-regions for region-based orientation analysis.

    Parameters
    ----------
    z_lo : float
        Lower z-boundary of simulation box
    z_hi : float
        Upper z-boundary of simulation box
    z_interface : float
        Thickness of interfacial region from each surface
    d_bulk : float
        Half-width of bulk region around midpoint
    lower_metal_surface_z : float, optional
        Top of LOWER metal surface (where water starts).
        If None, uses z_lo.
    upper_metal_surface_z : float, optional
        Bottom of UPPER metal surface (where water ends).
        If None, uses z_hi.

    Returns
    -------
    regions : dict
        Dictionary with keys 'interface_a', 'interface_b', 'bulk',
        each mapping to (z_min, z_max) tuple
    """
    if lower_metal_surface_z is not None:
        interface_a_start = lower_metal_surface_z
    else:
        interface_a_start = z_lo

    if upper_metal_surface_z is not None:
        interface_b_end = upper_metal_surface_z
    else:
        interface_b_end = z_hi

    midpoint = (interface_a_start + interface_b_end) / 2.0

    return {
        'interface_a': (interface_a_start, interface_a_start + z_interface),
        'interface_b': (interface_b_end - z_interface, interface_b_end),
        'bulk': (midpoint - d_bulk, midpoint + d_bulk),
    }


def detect_layer_boundaries(
    frames: List[Dict],
    z_surface: float,
    z_max: float,
    n_bins: int = 300,
    sigma: float = 2.0,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Automatically detect water layer boundaries from oxygen density profile.

    Follows the reference algorithm:
    1. Calculate oxygen density profile from metal surface to z_max
    2. Smooth with gaussian filter
    3. Find first peak (surface layer maximum) within 6A of surface
    4. Find first minimum after peak (surface layer end) within 4A of peak
    5. Find second minimum (subsurface layer end) within 6A of first minimum

    Parameters
    ----------
    frames : list of dict
        Trajectory frames from read_lammpstrj
    z_surface : float
        Z-coordinate of metal surface (top of lower slab)
    z_max : float
        Maximum z-coordinate to analyze (middle of box)
    n_bins : int
        Number of bins for density histogram (default: 300)
    sigma : float
        Gaussian smoothing sigma (default: 2.0)
    verbose : bool
        Print detected boundaries
    logger : optional
        Logger instance for output

    Returns
    -------
    boundaries : dict
        Dictionary with keys:
        - 'z_surface': metal surface z-coordinate
        - 'z_surface_peak': z of first density peak
        - 'z_surface_min': z of first minimum (surface/subsurface boundary)
        - 'z_subsurface_min': z of second minimum (subsurface/bulk boundary)
        - 'bin_centers': array of bin centers
        - 'density': raw density histogram
        - 'density_smooth': smoothed density profile

    Raises
    ------
    ValueError
        If scipy is not available or required peaks/minima cannot be detected
    """
    if not SCIPY_AVAILABLE:
        raise ValueError("scipy is required for auto-layer detection (gaussian_filter1d)")

    # 1. Build histogram of oxygen z-coordinates
    # Extend slightly below surface and up to z_max (or 20A from surface, whichever is smaller)
    z_upper = min(z_surface + 20, z_max)
    bins = np.linspace(z_surface - 1, z_upper, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0]
    hist = np.zeros(len(bins) - 1)

    # Get elements array
    elements = frames[0].get('elements')
    if elements is None:
        raise ValueError("Frames must contain element information for auto-layer detection")

    # Find O atoms and accumulate histogram
    for frame in frames:
        positions = frame['positions']
        frame_elements = frame['elements']

        # Find oxygen atoms
        for i, elem in enumerate(frame_elements):
            if elem == 'O':
                z = positions[i, 2]
                idx = np.searchsorted(bins, z) - 1
                if 0 <= idx < len(hist):
                    hist[idx] += 1

    # Normalize by number of frames and bin width
    hist = hist / len(frames) / bin_width

    # 2. Smooth with gaussian filter
    hist_smooth = gaussian_filter1d(hist, sigma=sigma)

    # 3. Find first peak (surface layer maximum) within 6A of surface
    search_region = (bin_centers >= z_surface) & (bin_centers < z_surface + 6)
    if not np.any(search_region):
        raise ValueError(f"No density data found near surface (z={z_surface:.2f})")

    search_idx = np.where(search_region)[0]
    peak_local_idx = np.argmax(hist_smooth[search_region])
    z_surface_peak = bin_centers[search_idx[peak_local_idx]]

    if verbose:
        _log(logger, 'detail', f"  Surface layer peak: z = {z_surface_peak:.2f} A")

    # 4. Find first minimum after peak within 4A of peak
    search_region = (bin_centers > z_surface_peak) & (bin_centers < z_surface_peak + 4)
    if not np.any(search_region):
        raise ValueError(f"Cannot find minimum after surface peak at z={z_surface_peak:.2f}")

    search_idx = np.where(search_region)[0]
    min1_local_idx = np.argmin(hist_smooth[search_region])
    z_surface_min = bin_centers[search_idx[min1_local_idx]]

    if verbose:
        _log(logger, 'detail', f"  Surface/subsurface boundary: z = {z_surface_min:.2f} A")

    # 5. Find second minimum within 6A after first minimum
    # Skip first 1A to avoid local noise
    search_region = (bin_centers > z_surface_min + 1.0) & (bin_centers < z_surface_min + 6)
    if not np.any(search_region):
        # If no second minimum found, use midpoint between z_surface_min and z_max
        z_subsurface_min = (z_surface_min + z_max) / 2
        if verbose:
            _log(logger, 'warning', f"  No second minimum found, using midpoint: z = {z_subsurface_min:.2f} A")
    else:
        search_idx = np.where(search_region)[0]
        min2_local_idx = np.argmin(hist_smooth[search_region])
        z_subsurface_min = bin_centers[search_idx[min2_local_idx]]
        if verbose:
            _log(logger, 'detail', f"  Subsurface/bulk boundary: z = {z_subsurface_min:.2f} A")

    return {
        'z_surface': z_surface,
        'z_surface_peak': z_surface_peak,
        'z_surface_min': z_surface_min,
        'z_subsurface_min': z_subsurface_min,
        'bin_centers': bin_centers,
        'density': hist,
        'density_smooth': hist_smooth,
    }


def define_auto_regions(
    frames: List[Dict],
    verbose: bool = True,
    logger: Optional[Any] = None
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
    """
    Automatically define water regions based on oxygen density profile.

    Detects surface, subsurface, and bulk layers from the lower metal surface
    toward the middle of the simulation box.

    Parameters
    ----------
    frames : list of dict
        Trajectory frames
    verbose : bool
        Print progress
    logger : optional
        Logger instance

    Returns
    -------
    regions : dict
        Dictionary mapping region names to (z_min, z_max) tuples:
        - 'surface': surface layer at lower interface
        - 'subsurface': subsurface layer at lower interface
        - 'bulk': bulk water region (from subsurface end to box midpoint)
    diagnostics : dict
        Contains density profiles and detected boundaries for plotting
    """
    # Get box dimensions
    box = frames[0]['box']
    z_lo = box['zlo']
    z_hi = box['zhi']
    z_mid = (z_lo + z_hi) / 2.0

    # Find metal surfaces
    try:
        lower_surface_z, upper_surface_z, found_metals = find_metal_surfaces(frames)
        if verbose:
            _log(logger, 'info', f"Metal surfaces detected: {found_metals}")
            _log(logger, 'detail', f"  Lower surface top: {lower_surface_z:.2f} A")
            if upper_surface_z is not None:
                _log(logger, 'detail', f"  Upper surface bottom: {upper_surface_z:.2f} A")
                # For sandwich systems, use the region between metal surfaces
                z_mid = (lower_surface_z + upper_surface_z) / 2.0
    except ValueError:
        if verbose:
            _log(logger, 'warning', "No metal surfaces detected, using box z_lo as surface")
        lower_surface_z = z_lo
        upper_surface_z = None

    # Detect layer boundaries from O density profile
    if verbose:
        _log(logger, 'info', "Detecting layer boundaries from O density profile...")

    boundaries = detect_layer_boundaries(
        frames,
        z_surface=lower_surface_z,
        z_max=z_mid,
        verbose=verbose,
        logger=logger
    )

    # Define regions
    regions = {
        'surface': (lower_surface_z, boundaries['z_surface_min']),
        'subsurface': (boundaries['z_surface_min'], boundaries['z_subsurface_min']),
        'bulk': (boundaries['z_subsurface_min'], z_mid),
    }

    if verbose:
        _log(logger, 'info', "Auto-detected regions:")
        for region_name, (rz_min, rz_max) in regions.items():
            _log(logger, 'detail', f"  {region_name}: {rz_min:.2f} to {rz_max:.2f} A")

    # Store diagnostics for plotting
    diagnostics = {
        'boundaries': boundaries,
        'z_mid': z_mid,
        'lower_surface_z': lower_surface_z,
        'upper_surface_z': upper_surface_z,
    }

    return regions, diagnostics


def filter_orientations_by_region(
    orientations: List[Dict],
    z_min: float,
    z_max: float
) -> List[Dict]:
    """
    Filter orientations to include only waters with O_z in [z_min, z_max].

    Parameters
    ----------
    orientations : list of dict
        Orientation data per frame from compute_orientations()
    z_min : float
        Minimum z coordinate
    z_max : float
        Maximum z coordinate

    Returns
    -------
    filtered : list of dict
        Filtered orientation data per frame
    """
    result = []
    for frame in orientations:
        mask = (frame['O_z'] >= z_min) & (frame['O_z'] <= z_max)
        result.append({
            'cos_theta_1': frame['cos_theta_1'][mask],
            'cos_theta_2': frame['cos_theta_2'][mask],
            'cos_phi': frame['cos_phi'][mask],
            'O_z': frame['O_z'][mask],
            'n_waters': int(np.sum(mask)),
        })
    return result


def extract_cos_theta(orientations: List[Dict]) -> List[np.ndarray]:
    """
    Combine cos_theta_1 and cos_theta_2 into single array per frame.

    Parameters
    ----------
    orientations : list of dict
        Orientation data per frame

    Returns
    -------
    cos_values : list of ndarray
        Combined cos(theta) values per frame (2 values per water)
    """
    result = []
    for frame in orientations:
        combined = np.concatenate([frame['cos_theta_1'], frame['cos_theta_2']])
        result.append(combined)
    return result


def extract_cos_phi(orientations: List[Dict]) -> List[np.ndarray]:
    """
    Extract cos_phi as array per frame.

    Parameters
    ----------
    orientations : list of dict
        Orientation data per frame

    Returns
    -------
    cos_values : list of ndarray
        cos(phi) values per frame (1 value per water)
    """
    return [frame['cos_phi'] for frame in orientations]


def compute_orientation_distribution(
    cos_per_frame: List[np.ndarray],
    n_bins: int = 50,
    n_blocks: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute P(cos) histogram with block averaging for error estimation.

    Parameters
    ----------
    cos_per_frame : list of ndarray
        cos values per frame
    n_bins : int
        Number of bins for histogram (default: 50)
    n_blocks : int
        Number of blocks for error estimation (default: 5)

    Returns
    -------
    bin_centers : ndarray
        Center of each bin
    P_mean : ndarray
        Mean probability density
    P_std : ndarray
        Standard error from block averaging
    """
    bins = np.linspace(-1, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    n_frames = len(cos_per_frame)

    if n_frames == 0 or all(len(f) == 0 for f in cos_per_frame):
        return bin_centers, np.zeros(n_bins), np.zeros(n_bins)

    if n_blocks <= 1 or n_frames < n_blocks:
        # No block averaging, just compute single histogram
        all_values = np.concatenate([f for f in cos_per_frame if len(f) > 0])
        if len(all_values) == 0:
            return bin_centers, np.zeros(n_bins), np.zeros(n_bins)
        hist, _ = np.histogram(all_values, bins=bins, density=True)
        return bin_centers, hist, np.zeros(n_bins)

    frames_per_block = n_frames // n_blocks

    block_histograms = []
    for b in range(n_blocks):
        start = b * frames_per_block
        end = start + frames_per_block if b < n_blocks - 1 else n_frames

        block_frames = cos_per_frame[start:end]
        block_values = np.concatenate([f for f in block_frames if len(f) > 0])

        if len(block_values) > 0:
            hist, _ = np.histogram(block_values, bins=bins, density=True)
            block_histograms.append(hist)

    if len(block_histograms) == 0:
        return bin_centers, np.zeros(n_bins), np.zeros(n_bins)

    block_histograms = np.array(block_histograms)
    P_mean = np.mean(block_histograms, axis=0)
    P_std = np.std(block_histograms, axis=0, ddof=1) / np.sqrt(len(block_histograms))

    return bin_centers, P_mean, P_std


def compute_orientation_statistics(
    orientations: List[Dict],
    angle_type: str = 'theta',
    n_bins: int = 50,
    n_blocks: int = 5
) -> Dict:
    """
    Compute orientation distribution statistics.

    Parameters
    ----------
    orientations : list of dict
        Orientation data per frame
    angle_type : str
        'theta' for O-H bond orientation, 'phi' for dipole orientation
    n_bins : int
        Number of bins for histogram
    n_blocks : int
        Number of blocks for error estimation

    Returns
    -------
    results : dict
        Dictionary containing:
        - bins: bin centers
        - P: probability density
        - err: standard error
        - n_samples: total number of samples
        - mean_cos: mean cos value
        - std_cos: standard deviation of cos values
    """
    if angle_type == 'theta':
        cos_values = extract_cos_theta(orientations)
    else:
        cos_values = extract_cos_phi(orientations)

    bin_centers, P_mean, P_std = compute_orientation_distribution(
        cos_values, n_bins, n_blocks
    )

    # Compute cos statistics
    all_values = np.concatenate([f for f in cos_values if len(f) > 0])
    n_samples = len(all_values)

    if n_samples > 0:
        mean_cos = float(np.mean(all_values))
        std_cos = float(np.std(all_values))
    else:
        mean_cos = 0.0
        std_cos = 0.0

    # Compute P(cos) statistics
    if len(P_mean) > 0 and np.any(P_mean > 0):
        mean_P = float(np.mean(P_mean))
        std_P = float(np.std(P_mean))
    else:
        mean_P = 0.0
        std_P = 0.0

    return {
        'bins': bin_centers,
        'P': P_mean,
        'err': P_std,
        'n_samples': n_samples,
        # cos statistics
        'mean_cos': mean_cos,
        'std_cos': std_cos,
        # P(cos) statistics
        'mean_P': mean_P,
        'std_P': std_P,
        'n_frames': len(orientations),
        'n_blocks': n_blocks,
    }


def analyze_reorientation(
    frames: List[Dict],
    orientations: List[Dict],
    n_bins: int = 50,
    n_blocks: int = 5,
    z_interface: Optional[float] = None,
    d_bulk: Optional[float] = None,
    auto_layers: bool = True,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> Dict:
    """
    Comprehensive water reorientation analysis.

    Parameters
    ----------
    frames : list of dict
        Trajectory frames from read_lammpstrj
    orientations : list of dict
        Orientation data per frame from compute_orientations()
    n_bins : int
        Number of bins for histogram (default: 50)
    n_blocks : int
        Number of blocks for error estimation (default: 5)
    z_interface : float, optional
        Interface thickness from each surface (Angstroms). For manual region mode.
    d_bulk : float, optional
        Half-width of bulk region around midpoint (Angstroms). For manual region mode.
    auto_layers : bool
        If True and z_interface/d_bulk not specified, automatically detect
        layers (surface, subsurface, bulk) from oxygen density profile (default: True)
    verbose : bool
        Print progress (default: True)
    logger : optional
        Logger instance for output

    Returns
    -------
    results : dict
        Dictionary containing:
        - theta: dict with 'global' and optionally region distributions
        - phi: dict with 'global' and optionally region distributions
        - regions: region definitions (if region analysis)
        - region_mode: 'manual', 'auto', or 'global'
        - layer_diagnostics: density profile data (if auto_layers used)
        - n_frames: number of frames
        - n_waters: number of water molecules
    """
    n_frames = len(orientations)
    n_waters = orientations[0]['n_waters'] if orientations else 0

    # Validate region parameters
    if (z_interface is None) != (d_bulk is None):
        raise ValueError("z_interface and d_bulk must both be provided, or neither")

    # Determine region analysis mode
    manual_regions = z_interface is not None
    use_auto_layers = not manual_regions and auto_layers

    if verbose:
        _log(logger, 'info', f"Analyzing {n_frames} frames with {n_waters} water molecules")

    # Compute global (all waters) distributions
    if verbose:
        _log(logger, 'info', "Computing global orientation distributions...")

    theta_global = compute_orientation_statistics(orientations, 'theta', n_bins, n_blocks)
    phi_global = compute_orientation_statistics(orientations, 'phi', n_bins, n_blocks)

    results = {
        'theta': {'global': theta_global},
        'phi': {'global': phi_global},
        'n_frames': n_frames,
        'n_waters': n_waters,
        'n_bins': n_bins,
        'n_blocks': n_blocks,
        'region_mode': 'global',
    }

    # Manual region-based analysis
    if manual_regions:
        if verbose:
            _log(logger, 'info', "Performing manual region-based analysis...")

        results['region_mode'] = 'manual'

        # Get box dimensions
        box = frames[0]['box']
        z_lo = box['zlo']
        z_hi = box['zhi']

        # Find metal surfaces (auto-detects Cu or Pt)
        try:
            lower_surface_z, upper_surface_z, found_metals = find_metal_surfaces(frames)
            if verbose:
                _log(logger, 'info', f"Metal surfaces detected: {found_metals}")
                _log(logger, 'detail', f"  Lower surface top: {lower_surface_z:.2f} A")
                if upper_surface_z is not None:
                    _log(logger, 'detail', f"  Upper surface bottom: {upper_surface_z:.2f} A")
        except ValueError:
            if verbose:
                _log(logger, 'info', "No metal surfaces detected, using box boundaries")
            lower_surface_z = None
            upper_surface_z = None

        # Define regions
        regions = define_z_regions_reorientation(
            z_lo, z_hi, z_interface, d_bulk,
            lower_metal_surface_z=lower_surface_z,
            upper_metal_surface_z=upper_surface_z
        )

        if verbose:
            for region_name, (rz_min, rz_max) in regions.items():
                _log(logger, 'detail', f"  {region_name}: {rz_min:.2f} to {rz_max:.2f} A")

        results['regions'] = regions

        # Compute distributions for each region
        for region_name, (z_min, z_max) in regions.items():
            if verbose:
                _log(logger, 'info', f"Processing region: {region_name}")

            filtered = filter_orientations_by_region(orientations, z_min, z_max)

            theta_region = compute_orientation_statistics(filtered, 'theta', n_bins, n_blocks)
            phi_region = compute_orientation_statistics(filtered, 'phi', n_bins, n_blocks)

            # Add region z-range info
            theta_region['z_range'] = (z_min, z_max)
            phi_region['z_range'] = (z_min, z_max)

            # Count average waters in region
            avg_waters = np.mean([f['n_waters'] for f in filtered])
            theta_region['avg_waters'] = avg_waters
            phi_region['avg_waters'] = avg_waters

            results['theta'][region_name] = theta_region
            results['phi'][region_name] = phi_region

    # Auto-layer detection
    elif use_auto_layers:
        if verbose:
            _log(logger, 'info', "Performing auto-layer detection from O density profile...")

        results['region_mode'] = 'auto'

        try:
            regions, diagnostics = define_auto_regions(frames, verbose=verbose, logger=logger)
            results['regions'] = regions
            results['layer_diagnostics'] = diagnostics

            # Compute distributions for each auto-detected region
            for region_name, (z_min, z_max) in regions.items():
                if verbose:
                    _log(logger, 'info', f"Processing region: {region_name}")

                filtered = filter_orientations_by_region(orientations, z_min, z_max)

                theta_region = compute_orientation_statistics(filtered, 'theta', n_bins, n_blocks)
                phi_region = compute_orientation_statistics(filtered, 'phi', n_bins, n_blocks)

                # Add region z-range info
                theta_region['z_range'] = (z_min, z_max)
                phi_region['z_range'] = (z_min, z_max)

                # Count average waters in region
                avg_waters = np.mean([f['n_waters'] for f in filtered])
                theta_region['avg_waters'] = avg_waters
                phi_region['avg_waters'] = avg_waters

                results['theta'][region_name] = theta_region
                results['phi'][region_name] = phi_region

        except ValueError as e:
            if verbose:
                _log(logger, 'warning', f"Auto-layer detection failed: {e}")
                _log(logger, 'warning', "Falling back to global analysis only")
            results['region_mode'] = 'global'

    return results
