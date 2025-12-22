"""
Diffusion coefficient analysis from Mean Square Displacement (MSD).

Computes planar (x-y), perpendicular (z), and total (x-y-z) diffusion coefficients.
"""

from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
from scipy import stats

# Try to import C++ core, fall back to pure Python if not available
try:
    from mlip_plot._core import compute_msd, compute_msd_regions
    HAS_CPP_CORE = True
except ImportError:
    HAS_CPP_CORE = False


def _log(logger: Optional[Any], method: str, message: str, verbose: bool = True):
    """Helper to log or print based on logger availability."""
    if not verbose:
        return
    if logger is not None:
        getattr(logger, method)(message)
    else:
        print(message)

# Atomic masses for COM calculation (g/mol)
ATOMIC_MASSES = {
    'H': 1.008, 'O': 15.999, 'C': 12.011, 'N': 14.007,
    'Na': 22.990, 'Cl': 35.45, 'K': 39.098, 'Li': 6.941,
    'Ca': 40.078, 'Mg': 24.305, 'S': 32.065, 'F': 18.998,
}


def find_water_molecules(
    frames: List[Dict],
    o_h_cutoff: float = 1.2
) -> List[Tuple[int, int, int]]:
    """
    Find water molecules by detecting O-H bonds.

    Parameters
    ----------
    frames : list
        List of trajectory frames
    o_h_cutoff : float
        O-H bond distance cutoff in Angstroms (default: 1.2)

    Returns
    -------
    water_molecules : list of tuple
        List of (O_idx, H1_idx, H2_idx) tuples (indices into atom arrays)
    """
    if frames[0]['elements'] is None:
        raise ValueError("Frames must contain element information")

    elements = frames[0]['elements']
    positions = frames[0]['positions']

    # Find O and H atoms
    o_indices = np.where(elements == 'O')[0]
    h_indices = np.where(elements == 'H')[0]

    if len(o_indices) == 0 or len(h_indices) == 0:
        raise ValueError("No O or H atoms found in trajectory")

    water_molecules = []
    used_h = set()

    for o_idx in o_indices:
        o_pos = positions[o_idx]

        # Find H atoms bonded to this O
        bonded_h = []
        for h_idx in h_indices:
            if h_idx in used_h:
                continue
            h_pos = positions[h_idx]
            dist = np.linalg.norm(h_pos - o_pos)
            if dist < o_h_cutoff:
                bonded_h.append((h_idx, dist))

        # Sort by distance and take closest 2
        bonded_h.sort(key=lambda x: x[1])

        if len(bonded_h) >= 2:
            h1_idx = bonded_h[0][0]
            h2_idx = bonded_h[1][0]
            water_molecules.append((o_idx, h1_idx, h2_idx))
            used_h.add(h1_idx)
            used_h.add(h2_idx)

    return water_molecules


def compute_water_com_positions(
    frames: List[Dict],
    water_molecules: List[Tuple[int, int, int]]
) -> Tuple[List[np.ndarray], int]:
    """
    Compute center of mass positions for water molecules.

    Parameters
    ----------
    frames : list
        List of trajectory frames
    water_molecules : list
        List of (O_idx, H1_idx, H2_idx) tuples

    Returns
    -------
    positions_list : list of ndarray
        List of COM position arrays, each shape (n_molecules, 3)
    n_molecules : int
        Number of water molecules
    """
    m_O = ATOMIC_MASSES['O']
    m_H = ATOMIC_MASSES['H']
    total_mass = m_O + 2 * m_H

    n_molecules = len(water_molecules)
    if n_molecules == 0:
        raise ValueError("No water molecules found")

    positions_list = []

    for frame in frames:
        pos = frame['positions']
        com_positions = np.zeros((n_molecules, 3), dtype=np.float64)

        for i, (o_idx, h1_idx, h2_idx) in enumerate(water_molecules):
            o_pos = pos[o_idx]
            h1_pos = pos[h1_idx]
            h2_pos = pos[h2_idx]

            # Center of mass
            com = (m_O * o_pos + m_H * h1_pos + m_H * h2_pos) / total_mass
            com_positions[i] = com

        positions_list.append(com_positions)

    return positions_list, n_molecules


def filter_atoms_by_element(
    frames: List[Dict],
    elements: Set[str]
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Filter atoms by element and return positions per frame.

    Parameters
    ----------
    frames : list
        List of trajectory frames from read_lammpstrj
    elements : set
        Elements to include (e.g., {'O', 'H'} for water)

    Returns
    -------
    positions_list : list of ndarray
        List of position arrays, each shape (n_atoms, 3)
    atom_ids : ndarray
        Atom IDs of shape (n_atoms,)
    """
    if not frames:
        raise ValueError("No frames provided")

    if frames[0]['elements'] is None:
        raise ValueError("Frames must contain element information")

    # Get mask for atoms of specified elements from first frame
    first_elements = frames[0]['elements']
    mask = np.isin(first_elements, list(elements))
    atom_ids = frames[0]['atom_ids'][mask]

    positions_list = []
    for frame in frames:
        frame_mask = np.isin(frame['elements'], list(elements))
        positions_list.append(frame['positions'][frame_mask].astype(np.float64))

    return positions_list, atom_ids


def define_z_regions(
    z_lo: float,
    z_hi: float,
    z_interface: float,
    d_bulk: float
) -> Dict[str, Tuple[float, float]]:
    """
    Define three z-regions for region-based diffusion analysis.

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

    Returns
    -------
    regions : dict
        Dictionary with keys 'interface_a', 'interface_b', 'bulk',
        each mapping to (z_min, z_max) tuple
    """
    z_length = z_hi - z_lo
    midpoint = z_lo + z_length / 2.0

    return {
        'interface_a': (z_lo, z_lo + z_interface),
        'interface_b': (z_hi - z_interface, z_hi),
        'bulk': (midpoint - d_bulk, midpoint + d_bulk),
    }


def get_atoms_in_region(
    positions: np.ndarray,
    z_min: float,
    z_max: float
) -> np.ndarray:
    """
    Return boolean mask of atoms within z_min <= z <= z_max.

    Parameters
    ----------
    positions : ndarray
        Position array of shape (n_atoms, 3)
    z_min : float
        Minimum z-coordinate
    z_max : float
        Maximum z-coordinate

    Returns
    -------
    mask : ndarray
        Boolean mask of shape (n_atoms,)
    """
    z_coords = positions[:, 2]
    return (z_coords >= z_min) & (z_coords <= z_max)


def _unwrap_coordinates_python(
    positions_list: List[np.ndarray],
    box_lengths: Tuple[float, float, float],
    unwrap_xy: bool = True,
    unwrap_z: bool = False
) -> List[np.ndarray]:
    """
    Pure Python implementation of coordinate unwrapping.

    Parameters
    ----------
    positions_list : list of ndarray
        List of position arrays, each shape (n_atoms, 3)
    box_lengths : tuple
        Box dimensions (Lx, Ly, Lz)
    unwrap_xy : bool
        Unwrap x and y coordinates
    unwrap_z : bool
        Unwrap z coordinate

    Returns
    -------
    unwrapped : list of ndarray
        List of unwrapped position arrays
    """
    n_frames = len(positions_list)
    Lx, Ly, Lz = box_lengths

    unwrapped = [positions_list[0].copy()]

    for i in range(1, n_frames):
        prev_pos = unwrapped[-1]
        curr_pos = positions_list[i]

        new_pos = prev_pos.copy()

        # X coordinate
        if unwrap_xy:
            dx = curr_pos[:, 0] - prev_pos[:, 0]
            dx = dx - np.round(dx / Lx) * Lx
            new_pos[:, 0] = prev_pos[:, 0] + dx

        # Y coordinate
        if unwrap_xy:
            dy = curr_pos[:, 1] - prev_pos[:, 1]
            dy = dy - np.round(dy / Ly) * Ly
            new_pos[:, 1] = prev_pos[:, 1] + dy

        # Z coordinate
        if unwrap_z:
            dz = curr_pos[:, 2] - prev_pos[:, 2]
            dz = dz - np.round(dz / Lz) * Lz
            new_pos[:, 2] = prev_pos[:, 2] + dz
        else:
            new_pos[:, 2] = curr_pos[:, 2]

        unwrapped.append(new_pos)

    return unwrapped


def _compute_msd_python(
    positions_list: List[np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Pure Python implementation of MSD calculation.

    Parameters
    ----------
    positions_list : list of ndarray
        List of unwrapped position arrays, each shape (n_atoms, 3)

    Returns
    -------
    msd_data : dict
        Dictionary with 'planar', 'perpendicular', 'total' MSD arrays
    """
    n_frames = len(positions_list)
    max_lag = n_frames - 1

    msd_planar = np.zeros(max_lag)
    msd_perp = np.zeros(max_lag)
    msd_total = np.zeros(max_lag)
    counts = np.zeros(max_lag)

    # Average over different time origins
    for t0 in range(n_frames):
        for dt in range(1, n_frames - t0):
            r0 = positions_list[t0]
            rt = positions_list[t0 + dt]

            dr = rt - r0

            # MSD components
            msd_xy = np.mean(dr[:, 0]**2 + dr[:, 1]**2)
            msd_z = np.mean(dr[:, 2]**2)
            msd_3d = np.mean(np.sum(dr**2, axis=1))

            msd_planar[dt - 1] += msd_xy
            msd_perp[dt - 1] += msd_z
            msd_total[dt - 1] += msd_3d
            counts[dt - 1] += 1

    # Average over time origins
    msd_planar /= counts
    msd_perp /= counts
    msd_total /= counts

    return {
        'planar': msd_planar,
        'perpendicular': msd_perp,
        'total': msd_total
    }


def _compute_msd_region_python(
    positions_list: List[np.ndarray],
    regions: Dict[str, Tuple[float, float]]
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute MSD for each defined region with endpoint checking.

    For each (t0, t0+dt) pair, only include atoms that are inside
    the region at BOTH t0 AND t0+dt.

    Parameters
    ----------
    positions_list : list of ndarray
        List of unwrapped position arrays, each shape (n_atoms, 3)
    regions : dict
        Dictionary mapping region names to (z_min, z_max) tuples

    Returns
    -------
    results : dict
        Nested dict: {region_name: {'planar', 'perpendicular', 'total'}}
    """
    n_frames = len(positions_list)
    max_lag = n_frames - 1

    # Initialize result structure
    results = {}
    for region_name in regions:
        results[region_name] = {
            'planar': np.zeros(max_lag),
            'perpendicular': np.zeros(max_lag),
            'total': np.zeros(max_lag),
            'counts': np.zeros(max_lag),
        }

    for region_name, (z_min, z_max) in regions.items():
        for t0 in range(n_frames):
            pos_t0 = positions_list[t0]
            mask_t0 = get_atoms_in_region(pos_t0, z_min, z_max)

            for dt in range(1, n_frames - t0):
                pos_tdt = positions_list[t0 + dt]
                mask_tdt = get_atoms_in_region(pos_tdt, z_min, z_max)

                # Only include atoms in region at BOTH times
                combined_mask = mask_t0 & mask_tdt
                n_atoms_in_region = np.sum(combined_mask)

                if n_atoms_in_region > 0:
                    dr = pos_tdt[combined_mask] - pos_t0[combined_mask]

                    msd_xy = np.mean(dr[:, 0]**2 + dr[:, 1]**2)
                    msd_z = np.mean(dr[:, 2]**2)
                    msd_3d = np.mean(np.sum(dr**2, axis=1))

                    results[region_name]['planar'][dt - 1] += msd_xy
                    results[region_name]['perpendicular'][dt - 1] += msd_z
                    results[region_name]['total'][dt - 1] += msd_3d
                    results[region_name]['counts'][dt - 1] += 1

    # Average over time origins
    for region_name in regions:
        counts = results[region_name]['counts']
        for key in ['planar', 'perpendicular', 'total']:
            mask = counts > 0
            results[region_name][key][mask] /= counts[mask]
        del results[region_name]['counts']

    return results


def calculate_msd(
    frames: List[Dict],
    dt_ps: float,
    elements: Optional[Set[str]] = None,
    use_water_com: bool = False,
    z_interface: Optional[float] = None,
    d_bulk: Optional[float] = None,
    unwrap_z: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> Tuple[Dict, int, Optional[Dict[str, Tuple[float, float]]]]:
    """
    Calculate planar, perpendicular, and total MSD.

    Uses C++ backend when available for performance.
    Supports region-based analysis with z_interface and d_bulk parameters.

    Parameters
    ----------
    frames : list
        List of trajectory frames from read_lammpstrj
    dt_ps : float
        Time between frames in picoseconds
    elements : set, optional
        Elements to analyze (e.g., {'O', 'H'}). Required if use_water_com=False.
    use_water_com : bool
        If True, compute MSD for water molecule center of mass (default: False)
    z_interface : float, optional
        Thickness of interfacial region from each surface (Angstroms).
        Must be provided together with d_bulk for region analysis.
    d_bulk : float, optional
        Half-width of bulk region around midpoint (Angstroms).
        Must be provided together with z_interface for region analysis.
    unwrap_z : bool
        Whether to unwrap z coordinates (default: False)
    verbose : bool
        Print progress

    Returns
    -------
    msd_data : dict
        If region analysis: {region: {'planar': (time, msd), ...}}
        Otherwise: {'planar': (time, msd), ...}
    n_particles : int
        Number of particles (atoms or molecules) analyzed
    regions : dict or None
        If region analysis: {region_name: (z_min, z_max)}
        Otherwise: None
    """
    # Validate z-region parameters
    if (z_interface is None) != (d_bulk is None):
        raise ValueError("z_interface and d_bulk must both be provided, or neither")
    use_region_analysis = z_interface is not None

    # Get box dimensions
    box = frames[0]['box']
    z_lo = box['zlo']
    z_hi = box['zhi']
    Lx = box['xhi'] - box['xlo']
    Ly = box['yhi'] - box['ylo']
    Lz = z_hi - z_lo
    box_lengths = (Lx, Ly, Lz)

    # Define regions if region analysis is enabled
    regions = None
    if use_region_analysis:
        regions = define_z_regions(z_lo, z_hi, z_interface, d_bulk)
        if verbose:
            _log(logger, 'info', "Region-based analysis enabled")
            for region_name, (rz_min, rz_max) in regions.items():
                _log(logger, 'detail', f"  {region_name}: {rz_min:.2f} to {rz_max:.2f} A")

    # Get positions based on mode
    if use_water_com:
        if verbose:
            _log(logger, 'info', "Calculating MSD for water center of mass")

        # Find water molecules
        if verbose:
            _log(logger, 'info', "Detecting water molecules...")
        water_molecules = find_water_molecules(frames)
        if verbose:
            _log(logger, 'success', f"Found {len(water_molecules)} water molecules")

        # Compute COM positions (no z-filtering at this stage)
        positions_list, n_particles = compute_water_com_positions(
            frames, water_molecules
        )

    else:
        if elements is None:
            raise ValueError("elements must be specified when use_water_com=False")

        if verbose:
            _log(logger, 'info', f"Calculating MSD for elements: {sorted(elements)}")

        # Filter atoms by element only
        positions_list, atom_ids = filter_atoms_by_element(frames, elements)
        n_particles = len(atom_ids)
        if verbose:
            _log(logger, 'detail', f"Total atoms: {n_particles}")

    n_frames = len(frames)
    if verbose:
        _log(logger, 'detail', f"Frames: {n_frames}")

    # Create time array
    max_lag = n_frames - 1
    time = np.arange(1, max_lag + 1) * dt_ps

    # Compute MSD
    if use_region_analysis:
        # Region-based analysis
        if HAS_CPP_CORE:
            if verbose:
                _log(logger, 'info', "Using C++ backend for region analysis...")
            # Convert regions dict to format expected by C++: {name: (z_min, z_max)}
            msd_result = compute_msd_regions(
                positions_list, box_lengths, regions,
                unwrap_xy=True, unwrap_z=unwrap_z
            )
            # Also compute global MSD
            if verbose:
                _log(logger, 'info', "Computing global MSD...")
            global_msd_result = compute_msd(
                positions_list, box_lengths,
                unwrap_xy=True, unwrap_z=unwrap_z
            )
        else:
            if verbose:
                _log(logger, 'info', "Using Python backend for region analysis...")
                _log(logger, 'info', "Unwrapping coordinates...")

            unwrapped = _unwrap_coordinates_python(
                positions_list, box_lengths,
                unwrap_xy=True, unwrap_z=unwrap_z
            )

            if verbose:
                _log(logger, 'info', "Computing region-based MSD...")
            msd_result = _compute_msd_region_python(unwrapped, regions)

            if verbose:
                _log(logger, 'info', "Computing global MSD...")
            global_msd_result = _compute_msd_python(unwrapped)

        # Format output with time arrays
        msd_data = {}
        for region_name in regions:
            msd_data[region_name] = {
                'planar': (time, msd_result[region_name]['planar']),
                'perpendicular': (time, msd_result[region_name]['perpendicular']),
                'total': (time, msd_result[region_name]['total'])
            }
        # Add global MSD
        msd_data['global'] = {
            'planar': (time, global_msd_result['planar']),
            'perpendicular': (time, global_msd_result['perpendicular']),
            'total': (time, global_msd_result['total'])
        }

    else:
        # Standard analysis (use C++ if available)
        if HAS_CPP_CORE:
            if verbose:
                _log(logger, 'info', "Using C++ backend...")
            msd_result = compute_msd(
                positions_list, box_lengths,
                unwrap_xy=True, unwrap_z=unwrap_z
            )
        else:
            if verbose:
                _log(logger, 'info', "Using Python backend (C++ not available)...")
                _log(logger, 'info', "Unwrapping coordinates...")

            unwrapped = _unwrap_coordinates_python(
                positions_list, box_lengths,
                unwrap_xy=True, unwrap_z=unwrap_z
            )

            if verbose:
                _log(logger, 'info', "Computing MSD...")
            msd_result = _compute_msd_python(unwrapped)

        msd_data = {
            'planar': (time, msd_result['planar']),
            'perpendicular': (time, msd_result['perpendicular']),
            'total': (time, msd_result['total'])
        }

    return msd_data, n_particles, regions


def fit_diffusion_coefficient(
    time: np.ndarray,
    msd: np.ndarray,
    fit_start_frac: float = 0.2,
    fit_end_frac: float = 0.9,
    fit_start_ps: Optional[float] = None,
    fit_end_ps: Optional[float] = None,
    dimensionality: int = 2,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> Dict[str, float]:
    """
    Calculate diffusion coefficient from MSD slope.

    D = slope / (2 * n) where n is dimensionality:
    - n=2 for planar diffusion (x-y)
    - n=1 for perpendicular diffusion (z)
    - n=3 for total 3D diffusion

    Parameters
    ----------
    time : ndarray
        Time array in picoseconds
    msd : ndarray
        MSD values in Angstrom^2
    fit_start_frac : float
        Starting fraction of time range for fitting (default: 0.2)
    fit_end_frac : float
        Ending fraction of time range for fitting (default: 0.9)
    fit_start_ps : float, optional
        Explicit start time in ps (overrides fit_start_frac)
    fit_end_ps : float, optional
        Explicit end time in ps (overrides fit_end_frac)
    dimensionality : int
        Spatial dimensionality (1, 2, or 3)
    verbose : bool
        Print results

    Returns
    -------
    results : dict
        Dictionary containing:
        - D_A2ps: diffusion coefficient in Angstrom^2/ps
        - D_cm2s: diffusion coefficient in cm^2/s
        - slope: fitted slope
        - intercept: fitted intercept
        - r_squared: R^2 value
        - fit_start_ps: actual fit start time
        - fit_end_ps: actual fit end time
    """
    # Determine fitting range
    max_time = time.max()

    if fit_start_ps is None:
        fit_start_ps = max_time * fit_start_frac
    if fit_end_ps is None:
        fit_end_ps = max_time * fit_end_frac

    # Select fitting range
    mask = (time >= fit_start_ps) & (time <= fit_end_ps)
    time_fit = time[mask]
    msd_fit = msd[mask]

    if len(time_fit) < 3:
        raise ValueError(
            f"Insufficient data points ({len(time_fit)}) in fitting range. "
            f"Available time: {time.min():.2f} to {max_time:.2f} ps. "
            f"Requested: {fit_start_ps:.2f} to {fit_end_ps:.2f} ps."
        )

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(time_fit, msd_fit)

    # Diffusion coefficient: D = slope / (2 * n)
    D_A2ps = slope / (2.0 * dimensionality)

    # Convert to cm^2/s (1 Angstrom^2/ps = 1e-4 cm^2/s)
    D_cm2s = D_A2ps * 1e-4

    results = {
        'D_A2ps': D_A2ps,
        'D_cm2s': D_cm2s,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'fit_start_ps': fit_start_ps,
        'fit_end_ps': fit_end_ps,
        'n_points': len(time_fit)
    }

    if verbose:
        dim_labels = {1: 'perpendicular (z)', 2: 'planar (x-y)', 3: 'total (3D)'}
        _log(logger, 'info', f"Diffusion Coefficient ({dim_labels.get(dimensionality, f'{dimensionality}D')}):")
        _log(logger, 'detail', f"Fit range: {fit_start_ps:.2f} to {fit_end_ps:.2f} ps ({len(time_fit)} points)")
        _log(logger, 'success', f"D = {D_A2ps:.4f} A^2/ps = {D_cm2s:.4e} cm^2/s")
        _log(logger, 'detail', f"R^2 = {results['r_squared']:.4f}")

    return results


def compute_diffusion(
    frames: List[Dict],
    dt_ps: float,
    elements: Optional[Set[str]] = None,
    use_water_com: bool = True,
    z_interface: Optional[float] = None,
    d_bulk: Optional[float] = None,
    fit_start_frac: float = 0.2,
    fit_end_frac: float = 0.9,
    fit_start_ps: Optional[float] = None,
    fit_end_ps: Optional[float] = None,
    unwrap_z: bool = False,
    n_blocks: int = 1,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> Dict:
    """
    Compute planar, perpendicular, and total diffusion coefficients.

    Parameters
    ----------
    frames : list
        List of trajectory frames from read_lammpstrj
    dt_ps : float
        Time between frames in picoseconds
    elements : set, optional
        Elements to analyze (e.g., {'O', 'H'}). Required if use_water_com=False.
    use_water_com : bool
        If True, compute diffusion for water molecule center of mass (default: True)
    z_interface : float, optional
        Thickness of interfacial region from each surface (Angstroms).
        Must be provided together with d_bulk for region analysis.
    d_bulk : float, optional
        Half-width of bulk region around midpoint (Angstroms).
        Must be provided together with z_interface for region analysis.
    fit_start_frac : float
        Starting fraction of time range for fitting (default: 0.2)
    fit_end_frac : float
        Ending fraction of time range for fitting (default: 0.9)
    fit_start_ps : float, optional
        Explicit start time for fitting in ps
    fit_end_ps : float, optional
        Explicit end time for fitting in ps
    unwrap_z : bool
        Whether to unwrap z coordinates (default: False)
    n_blocks : int
        Number of trajectory blocks for error estimation (default: 1, no blocking).
        If > 1, trajectory is split into n_blocks chunks and diffusion is computed
        for each block independently. Results include mean ± std.
    verbose : bool
        Print progress

    Returns
    -------
    results : dict
        Dictionary containing:
        - msd: dict with MSD data (region-based or standard)
        - diffusion: dict with diffusion coefficient results
        - n_particles: number of particles (atoms or molecules) analyzed
        - n_frames: number of frames analyzed
        - mode: 'water_com' or 'atoms'
        - regions: dict of region z-ranges (if region analysis) or None
        - n_blocks: number of blocks used
        - block_results: list of per-block diffusion results (if n_blocks > 1)
    """
    n_frames = len(frames)

    # Validate n_blocks
    if n_blocks < 1:
        raise ValueError("n_blocks must be >= 1")
    if n_blocks > 1 and n_frames < n_blocks * 10:
        raise ValueError(f"Not enough frames ({n_frames}) for {n_blocks} blocks. "
                         f"Need at least {n_blocks * 10} frames.")

    # Calculate MSD on full trajectory (for plotting)
    msd_data, n_particles, regions = calculate_msd(
        frames, dt_ps,
        elements=elements,
        use_water_com=use_water_com,
        z_interface=z_interface, d_bulk=d_bulk,
        unwrap_z=unwrap_z, verbose=verbose, logger=logger
    )

    use_region_analysis = regions is not None

    # Helper function to fit diffusion for a single block
    def _fit_block_diffusion(block_frames: List[Dict]) -> Dict:
        """Fit diffusion coefficients for a single block of frames."""
        block_msd, _, _ = calculate_msd(
            block_frames, dt_ps,
            elements=elements,
            use_water_com=use_water_com,
            z_interface=z_interface, d_bulk=d_bulk,
            unwrap_z=unwrap_z, verbose=False
        )

        block_diff = {}
        if use_region_analysis:
            # Fit for each region
            for region_name in regions:
                block_diff[region_name] = {}
                for diff_type, dim in [('planar', 2), ('perpendicular', 1), ('total', 3)]:
                    time, msd = block_msd[region_name][diff_type]
                    block_diff[region_name][diff_type] = fit_diffusion_coefficient(
                        time, msd,
                        fit_start_frac=fit_start_frac, fit_end_frac=fit_end_frac,
                        fit_start_ps=fit_start_ps, fit_end_ps=fit_end_ps,
                        dimensionality=dim, verbose=False
                    )
            # Also fit global
            block_diff['global'] = {}
            for diff_type, dim in [('planar', 2), ('perpendicular', 1), ('total', 3)]:
                time, msd = block_msd['global'][diff_type]
                block_diff['global'][diff_type] = fit_diffusion_coefficient(
                    time, msd,
                    fit_start_frac=fit_start_frac, fit_end_frac=fit_end_frac,
                    fit_start_ps=fit_start_ps, fit_end_ps=fit_end_ps,
                    dimensionality=dim, verbose=False
                )
        else:
            for diff_type, dim in [('planar', 2), ('perpendicular', 1), ('total', 3)]:
                time, msd = block_msd[diff_type]
                block_diff[diff_type] = fit_diffusion_coefficient(
                    time, msd,
                    fit_start_frac=fit_start_frac, fit_end_frac=fit_end_frac,
                    fit_start_ps=fit_start_ps, fit_end_ps=fit_end_ps,
                    dimensionality=dim, verbose=False
                )
        return block_diff

    # Compute block-wise diffusion if n_blocks > 1
    block_results = []
    if n_blocks > 1:
        block_size = n_frames // n_blocks
        if verbose:
            _log(logger, 'info', f"Block averaging: {n_blocks} blocks of {block_size} frames each")

        for i in range(n_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size if i < n_blocks - 1 else n_frames
            block_frames = frames[start_idx:end_idx]

            if verbose:
                _log(logger, 'detail', f"Processing block {i+1}/{n_blocks} (frames {start_idx}-{end_idx-1})")

            block_diff = _fit_block_diffusion(block_frames)
            block_results.append(block_diff)

    # Fit diffusion coefficients on full trajectory
    diffusion_results = {}

    if use_region_analysis:
        # Region-based analysis: fit for each region
        region_labels = {
            'interface_a': 'Interface A (Lower)',
            'interface_b': 'Interface B (Upper)',
            'bulk': 'Bulk (Central)'
        }

        for region_name in regions:
            diffusion_results[region_name] = {}
            region_msd = msd_data[region_name]

            if verbose:
                _log(logger, 'section', f"Region: {region_labels.get(region_name, region_name)}")

            for diff_type, dim, label in [
                ('planar', 2, 'Planar (x-y)'),
                ('perpendicular', 1, 'Perpendicular (z)'),
                ('total', 3, 'Total (3D)')
            ]:
                time, msd = region_msd[diff_type]
                if verbose:
                    _log(logger, 'info', f"{label} Diffusion")
                result = fit_diffusion_coefficient(
                    time, msd,
                    fit_start_frac=fit_start_frac, fit_end_frac=fit_end_frac,
                    fit_start_ps=fit_start_ps, fit_end_ps=fit_end_ps,
                    dimensionality=dim, verbose=verbose, logger=logger
                )

                # Add block statistics if available
                if n_blocks > 1:
                    D_values = np.array([br[region_name][diff_type]['D_A2ps'] for br in block_results])
                    result['D_mean'] = np.mean(D_values)
                    result['D_std'] = np.std(D_values, ddof=1)  # Sample std
                    result['D_blocks'] = D_values.tolist()

                diffusion_results[region_name][diff_type] = result

        # Also fit global (whole system) diffusion
        diffusion_results['global'] = {}
        global_msd = msd_data['global']

        if verbose:
            _log(logger, 'section', "Region: Global (Whole System)")

        for diff_type, dim, label in [
            ('planar', 2, 'Planar (x-y)'),
            ('perpendicular', 1, 'Perpendicular (z)'),
            ('total', 3, 'Total (3D)')
        ]:
            time, msd = global_msd[diff_type]
            if verbose:
                _log(logger, 'info', f"{label} Diffusion")
            result = fit_diffusion_coefficient(
                time, msd,
                fit_start_frac=fit_start_frac, fit_end_frac=fit_end_frac,
                fit_start_ps=fit_start_ps, fit_end_ps=fit_end_ps,
                dimensionality=dim, verbose=verbose, logger=logger
            )

            # Add block statistics if available
            if n_blocks > 1:
                D_values = np.array([br['global'][diff_type]['D_A2ps'] for br in block_results])
                result['D_mean'] = np.mean(D_values)
                result['D_std'] = np.std(D_values, ddof=1)  # Sample std
                result['D_blocks'] = D_values.tolist()

            diffusion_results['global'][diff_type] = result

    else:
        # Standard analysis (no regions)
        for diff_type, dim, label in [
            ('planar', 2, 'Planar (x-y)'),
            ('perpendicular', 1, 'Perpendicular (z)'),
            ('total', 3, 'Total (3D)')
        ]:
            time, msd = msd_data[diff_type]
            if verbose:
                _log(logger, 'section', f"{label} Diffusion")
            result = fit_diffusion_coefficient(
                time, msd,
                fit_start_frac=fit_start_frac, fit_end_frac=fit_end_frac,
                fit_start_ps=fit_start_ps, fit_end_ps=fit_end_ps,
                dimensionality=dim, verbose=verbose, logger=logger
            )

            # Add block statistics if available
            if n_blocks > 1:
                D_values = np.array([br[diff_type]['D_A2ps'] for br in block_results])
                result['D_mean'] = np.mean(D_values)
                result['D_std'] = np.std(D_values, ddof=1)  # Sample std
                result['D_blocks'] = D_values.tolist()

            diffusion_results[diff_type] = result

    return {
        'msd': msd_data,
        'diffusion': diffusion_results,
        'n_particles': n_particles,
        'n_frames': n_frames,
        'mode': 'water_com' if use_water_com else 'atoms',
        'regions': regions,
        'n_blocks': n_blocks,
        'block_results': block_results if n_blocks > 1 else None
    }
