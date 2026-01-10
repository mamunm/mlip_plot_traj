"""
Hydrogen bond analysis for water molecules.

Detects water-water H-bonds (O-H...O) using geometric criteria:
- Donor-Acceptor distance < d_a_cutoff (default: 3.5 A)
- D-H-A angle > angle_cutoff (default: 120 degrees)

Key metric: <nhbond> = (N_hbonds * 2) / N_oxygen
This gives the average number of H-bonds per water molecule.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .diffusion import (
    find_metal_surface_z,
    define_z_regions,
    get_atoms_in_region,
    SUPPORTED_METALS,
)


def _log(logger: Optional[Any], method: str, message: str, verbose: bool = True):
    """Helper to log or print based on logger availability."""
    if not verbose:
        return
    if logger is not None:
        getattr(logger, method)(message)
    else:
        print(message)


def _sort_frame_by_atom_id(frame: Dict) -> Dict:
    """
    Sort a frame's arrays by atom_id to ensure consistent indexing.

    LAMMPS dump files may have atoms in different orders in each frame.
    This function reorders all arrays so that atoms are sorted by atom_id.

    Parameters
    ----------
    frame : dict
        Frame dictionary with 'atom_ids', 'positions', 'elements', etc.

    Returns
    -------
    sorted_frame : dict
        New frame dictionary with arrays sorted by atom_id
    """
    atom_ids = frame['atom_ids']
    sort_idx = np.argsort(atom_ids)

    natoms = frame['natoms']

    sorted_frame = {
        'timestep': frame['timestep'],
        'natoms': natoms,
        'box': frame['box'],
        'atom_ids': atom_ids[sort_idx],
        'positions': frame['positions'][sort_idx],
        'elements': frame['elements'][sort_idx] if frame['elements'] is not None else None,
    }

    # Copy any other keys (like 'types')
    for key in frame:
        if key not in sorted_frame:
            val = frame[key]
            if isinstance(val, np.ndarray) and len(val) == natoms:
                sorted_frame[key] = val[sort_idx]
            else:
                sorted_frame[key] = val

    return sorted_frame


def _apply_minimum_image(diff: np.ndarray, box_lengths: Tuple[float, float, float]) -> np.ndarray:
    """
    Apply minimum image convention for periodic boundary conditions.

    Parameters
    ----------
    diff : ndarray
        Displacement vectors, shape (..., 3)
    box_lengths : tuple
        (Lx, Ly, Lz) box dimensions

    Returns
    -------
    diff_mic : ndarray
        Displacement vectors with minimum image convention applied
    """
    box = np.array(box_lengths)
    # Shift to range [-L/2, L/2]
    diff_mic = diff - box * np.round(diff / box)
    return diff_mic


def find_water_molecules_pbc(
    frame: Dict,
) -> List[Tuple[int, int, int]]:
    """
    Find water molecules by assigning the 2 closest H atoms to each O atom.

    This is more robust than using a fixed O-H cutoff because:
    1. It always finds exactly 2 H per O (assuming stoichiometric water)
    2. It applies minimum image convention for PBC

    Parameters
    ----------
    frame : dict
        Single frame with 'positions', 'elements', 'box'

    Returns
    -------
    water_molecules : list of tuple
        List of (O_idx, H1_idx, H2_idx) tuples sorted by O_idx
    """
    elements = frame['elements']
    positions = frame['positions']
    box = frame['box']

    if elements is None:
        raise ValueError("Frame must contain element information")

    # Get box dimensions for PBC
    box_lengths = (
        box['xhi'] - box['xlo'],
        box['yhi'] - box['ylo'],
        box['zhi'] - box['zlo']
    )

    # Find O and H atom indices
    o_indices = np.where(elements == 'O')[0]
    h_indices = np.where(elements == 'H')[0]

    if len(o_indices) == 0 or len(h_indices) == 0:
        raise ValueError("No O or H atoms found in frame")

    o_positions = positions[o_indices]
    h_positions = positions[h_indices]

    # For each O, find the 2 closest H atoms using PBC
    water_molecules = []
    used_h = set()

    for o_local_idx, o_idx in enumerate(o_indices):
        o_pos = o_positions[o_local_idx]

        # Compute distances to all H atoms with PBC
        diff = h_positions - o_pos  # shape (n_H, 3)
        diff_mic = _apply_minimum_image(diff, box_lengths)
        distances = np.linalg.norm(diff_mic, axis=1)

        # Find 2 closest H atoms that haven't been used
        sorted_h_local = np.argsort(distances)

        bonded_h = []
        for h_local_idx in sorted_h_local:
            h_idx = h_indices[h_local_idx]
            if h_idx not in used_h:
                bonded_h.append(h_idx)
                if len(bonded_h) == 2:
                    break

        if len(bonded_h) == 2:
            water_molecules.append((o_idx, bonded_h[0], bonded_h[1]))
            used_h.add(bonded_h[0])
            used_h.add(bonded_h[1])

    return water_molecules


def detect_hbonds_frame(
    positions: np.ndarray,
    water_molecules: List[Tuple[int, int, int]],
    box_lengths: Tuple[float, float, float],
    d_a_cutoff: float = 3.5,
    d_h_a_angle_cutoff: float = 120.0
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect hydrogen bonds in a single frame with PBC.

    An H-bond is detected when:
    1. Donor O and Acceptor O are within d_a_cutoff distance (with PBC)
    2. The D-H-A angle (angle at H between D and A) >= d_h_a_angle_cutoff

    Parameters
    ----------
    positions : ndarray, shape (n_atoms, 3)
        Atomic positions
    water_molecules : list of tuple
        List of (O_idx, H1_idx, H2_idx) from find_water_molecules_pbc()
    box_lengths : tuple
        (Lx, Ly, Lz) box dimensions for PBC
    d_a_cutoff : float
        Maximum donor-acceptor distance (default: 3.5 A)
    d_h_a_angle_cutoff : float
        Minimum D-H-A angle in degrees (default: 120)

    Returns
    -------
    n_hbonds : int
        Number of H-bonds detected
    donor_indices : ndarray
        Indices of donor O atoms
    h_indices : ndarray
        Indices of H atoms involved in H-bonds
    acceptor_indices : ndarray
        Indices of acceptor O atoms
    distances : ndarray
        D-A distances for each H-bond
    angles : ndarray
        D-H-A angles (in degrees) for each H-bond
    """
    n_water = len(water_molecules)
    if n_water == 0:
        return 0, np.array([], dtype=int), np.array([], dtype=int), \
               np.array([], dtype=int), np.array([]), np.array([])

    # Extract oxygen indices and positions
    o_indices = np.array([mol[0] for mol in water_molecules])
    o_positions = positions[o_indices]

    # Compute all O-O distances with PBC (vectorized)
    diff = o_positions[:, np.newaxis, :] - o_positions[np.newaxis, :, :]
    diff_mic = _apply_minimum_image(diff, box_lengths)
    dist_matrix = np.sqrt(np.sum(diff_mic**2, axis=2))

    # Exclude self-pairs
    np.fill_diagonal(dist_matrix, np.inf)

    # Find candidate pairs where D-A distance < cutoff
    candidate_mask = dist_matrix < d_a_cutoff
    donor_water_idx, acceptor_water_idx = np.where(candidate_mask)

    # Check angles for each candidate pair
    hbonds = []
    for d_w, a_w in zip(donor_water_idx, acceptor_water_idx):
        donor_o_idx = o_indices[d_w]
        acceptor_o_idx = o_indices[a_w]

        donor_pos = positions[donor_o_idx]
        acceptor_pos = positions[acceptor_o_idx]

        # Check both H atoms of donor
        h1_idx, h2_idx = water_molecules[d_w][1], water_molecules[d_w][2]

        for h_idx in [h1_idx, h2_idx]:
            h_pos = positions[h_idx]

            # Compute D-H-A angle (angle at H between D and A) with PBC
            vec_hd = _apply_minimum_image(donor_pos - h_pos, box_lengths)      # H -> D vector
            vec_ha = _apply_minimum_image(acceptor_pos - h_pos, box_lengths)   # H -> A vector

            # Normalize
            norm_hd = np.linalg.norm(vec_hd)
            norm_ha = np.linalg.norm(vec_ha)

            if norm_hd < 1e-10 or norm_ha < 1e-10:
                continue

            vec_hd_norm = vec_hd / norm_hd
            vec_ha_norm = vec_ha / norm_ha

            # Angle at H
            cos_angle = np.clip(np.dot(vec_hd_norm, vec_ha_norm), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))

            if angle >= d_h_a_angle_cutoff:
                d_a_dist = dist_matrix[d_w, a_w]
                hbonds.append((donor_o_idx, h_idx, acceptor_o_idx, d_a_dist, angle))

    # Convert to arrays
    if hbonds:
        hbonds_arr = np.array(hbonds)
        n_hbonds = len(hbonds_arr)
        return (
            n_hbonds,
            hbonds_arr[:, 0].astype(int),  # donor indices
            hbonds_arr[:, 1].astype(int),  # h indices
            hbonds_arr[:, 2].astype(int),  # acceptor indices
            hbonds_arr[:, 3],              # distances
            hbonds_arr[:, 4]               # angles
        )
    else:
        return 0, np.array([], dtype=int), np.array([], dtype=int), \
               np.array([], dtype=int), np.array([]), np.array([])


def detect_hbonds_region_frame(
    positions: np.ndarray,
    water_molecules: List[Tuple[int, int, int]],
    box_lengths: Tuple[float, float, float],
    regions: Dict[str, Tuple[float, float]],
    d_a_cutoff: float = 3.5,
    d_h_a_angle_cutoff: float = 120.0
) -> Dict[str, Tuple[int, int]]:
    """
    Detect H-bonds with region attribution.

    Region Attribution Rule:
    - An H-bond is counted in a region if EITHER the donor OR acceptor
      oxygen is within that region's z-bounds.
    - Same H-bond can be counted in multiple regions if donor and acceptor
      are in different regions.

    Parameters
    ----------
    positions : ndarray, shape (n_atoms, 3)
        Atomic positions
    water_molecules : list of tuple
        List of (O_idx, H1_idx, H2_idx)
    box_lengths : tuple
        (Lx, Ly, Lz) box dimensions for PBC
    regions : dict
        {region_name: (z_min, z_max)} for each region
    d_a_cutoff : float
        Maximum donor-acceptor distance (default: 3.5 A)
    d_h_a_angle_cutoff : float
        Minimum D-H-A angle in degrees (default: 120)

    Returns
    -------
    results : dict
        {region_name: (n_hbonds, n_water_in_region)} for each region
        Also includes 'global' with total counts.
    """
    # First detect all H-bonds globally
    n_hbonds, donor_indices, h_indices, acceptor_indices, distances, angles = \
        detect_hbonds_frame(positions, water_molecules, box_lengths, d_a_cutoff, d_h_a_angle_cutoff)

    # Get oxygen indices and positions
    o_indices = np.array([mol[0] for mol in water_molecules])
    o_positions = positions[o_indices]
    n_water_total = len(water_molecules)

    results = {
        'global': (n_hbonds, n_water_total)
    }

    # For each region, count H-bonds where either donor or acceptor is in region
    for region_name, (z_min, z_max) in regions.items():
        # Find waters in this region (based on O position)
        o_z = o_positions[:, 2]
        in_region_mask = (o_z >= z_min) & (o_z <= z_max)
        n_water_in_region = np.sum(in_region_mask)

        # Map water index to whether it's in region
        # We need to map from atom index to water index
        o_idx_to_water_idx = {mol[0]: i for i, mol in enumerate(water_molecules)}

        if n_hbonds > 0:
            # Count H-bonds where either donor or acceptor is in region
            region_hbond_count = 0
            for d_idx, a_idx in zip(donor_indices, acceptor_indices):
                d_water_idx = o_idx_to_water_idx.get(d_idx)
                a_water_idx = o_idx_to_water_idx.get(a_idx)

                if d_water_idx is not None and a_water_idx is not None:
                    donor_in_region = in_region_mask[d_water_idx]
                    acceptor_in_region = in_region_mask[a_water_idx]
                    # Count if EITHER is in region
                    if donor_in_region or acceptor_in_region:
                        region_hbond_count += 1

            results[region_name] = (region_hbond_count, int(n_water_in_region))
        else:
            results[region_name] = (0, int(n_water_in_region))

    return results


def _compute_block_statistics(
    time_series: np.ndarray,
    n_blocks: int
) -> Tuple[float, float]:
    """
    Compute mean and standard error using block averaging.

    Parameters
    ----------
    time_series : ndarray
        1D array of values over time
    n_blocks : int
        Number of blocks for averaging

    Returns
    -------
    mean : float
        Mean of the time series
    std_error : float
        Standard error from block averaging
    """
    n = len(time_series)
    if n_blocks <= 1 or n < n_blocks:
        return np.mean(time_series), np.std(time_series)

    block_size = n // n_blocks
    block_means = []

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size
        block_means.append(np.mean(time_series[start:end]))

    block_means = np.array(block_means)
    mean = np.mean(block_means)
    std_error = np.std(block_means, ddof=1) / np.sqrt(n_blocks)

    return mean, std_error


def compute_hbond_analysis(
    frames: List[Dict],
    dt_ps: float,
    d_a_cutoff: float = 3.5,
    d_h_a_angle_cutoff: float = 120.0,
    z_interface: Optional[float] = None,
    d_bulk: Optional[float] = None,
    n_blocks: int = 5,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Compute hydrogen bond analysis over trajectory.

    Parameters
    ----------
    frames : list
        Trajectory frames from read_lammpstrj
    dt_ps : float
        Time between frames in picoseconds
    d_a_cutoff : float
        D-A distance cutoff (default: 3.5 A)
    d_h_a_angle_cutoff : float
        D-H-A angle cutoff in degrees (default: 120)
    z_interface : float, optional
        Interface thickness for region analysis
    d_bulk : float, optional
        Bulk region half-width
    n_blocks : int
        Number of blocks for error estimation (default: 5)
    verbose : bool
        Print progress
    logger : optional
        Logger instance

    Returns
    -------
    results : dict
        {
            'time_series': {
                'time': np.ndarray,  # Time in ps
                'global': {
                    'n_hbonds': np.ndarray,  # H-bond count per frame
                    'nhbond': np.ndarray,    # <nhbond> per frame
                    'n_water': np.ndarray,   # Water count per frame
                },
                'interface_a': {...},  # if region analysis
                'interface_b': {...},
                'bulk': {...}
            },
            'summary': {
                'global': {
                    'mean_nhbond': float,
                    'std_nhbond': float,
                    'mean_n_hbonds': float,
                    'std_n_hbonds': float,
                    'n_water': int,
                },
                ... per region
            },
            'geometry': {
                'distances': np.ndarray,  # All D-A distances
                'angles': np.ndarray,     # All D-H-A angles
            },
            'n_water': int,
            'n_frames': int,
            'regions': dict or None,
            'parameters': {
                'd_a_cutoff': float,
                'd_h_a_angle_cutoff': float,
            }
        }
    """
    if not frames:
        raise ValueError("No frames provided")

    if frames[0]['elements'] is None:
        raise ValueError("Frames must contain element information")

    # Sort all frames by atom_id to ensure consistent indexing
    # LAMMPS dump files may have atoms in different orders in each frame
    _log(logger, 'info', "Sorting frames by atom ID...", verbose)
    sorted_frames = [_sort_frame_by_atom_id(f) for f in frames]

    # Find water molecules using PBC-aware detection (from first frame)
    # Uses closest 2 H atoms per O with minimum image convention
    _log(logger, 'info', "Finding water molecules...", verbose)
    water_molecules = find_water_molecules_pbc(sorted_frames[0])
    n_water = len(water_molecules)
    _log(logger, 'success', f"Found {n_water} water molecules", verbose)

    # Determine if region analysis is needed
    use_region_analysis = z_interface is not None and d_bulk is not None
    regions = None

    if use_region_analysis:
        box = sorted_frames[0]['box']
        z_lo, z_hi = box['zlo'], box['zhi']

        # Try to find metal surface
        try:
            metal_surface_z, found_metals = find_metal_surface_z(sorted_frames)
            _log(logger, 'detail',
                 f"Metal surface detected at z = {metal_surface_z:.2f} A ({', '.join(found_metals)})",
                 verbose)
        except ValueError:
            metal_surface_z = None
            _log(logger, 'detail', "No metal surface found, using box boundaries", verbose)

        regions = define_z_regions(z_lo, z_hi, z_interface, d_bulk, metal_surface_z)
        _log(logger, 'detail', f"Regions defined: {list(regions.keys())}", verbose)

    # Initialize time series storage
    n_frames = len(sorted_frames)
    time = np.arange(n_frames) * dt_ps

    if use_region_analysis:
        region_names = ['global', 'interface_a', 'interface_b', 'bulk']
    else:
        region_names = ['global']

    time_series = {
        'time': time,
    }
    for rname in region_names:
        time_series[rname] = {
            'n_hbonds': np.zeros(n_frames, dtype=int),
            'nhbond': np.zeros(n_frames, dtype=float),
            'n_water': np.zeros(n_frames, dtype=int),
        }

    # Collect all distances and angles for geometry histograms
    all_distances = []
    all_angles = []

    # Process each frame
    _log(logger, 'info', f"Analyzing {n_frames} frames...", verbose)

    for frame_idx, frame in enumerate(sorted_frames):
        positions = frame['positions']
        box = frame['box']
        box_lengths = (
            box['xhi'] - box['xlo'],
            box['yhi'] - box['ylo'],
            box['zhi'] - box['zlo']
        )

        if use_region_analysis:
            # Region-aware detection
            region_results = detect_hbonds_region_frame(
                positions, water_molecules, box_lengths, regions,
                d_a_cutoff, d_h_a_angle_cutoff
            )

            for rname in region_names:
                if rname in region_results:
                    n_hb, n_w = region_results[rname]
                    time_series[rname]['n_hbonds'][frame_idx] = n_hb
                    time_series[rname]['n_water'][frame_idx] = n_w
                    # <nhbond> = (n_hbonds * 2) / n_water
                    if n_w > 0:
                        time_series[rname]['nhbond'][frame_idx] = (n_hb * 2) / n_w
                    else:
                        time_series[rname]['nhbond'][frame_idx] = 0.0

            # Also get geometry data from global detection
            n_hbonds, _, _, _, distances, angles = detect_hbonds_frame(
                positions, water_molecules, box_lengths, d_a_cutoff, d_h_a_angle_cutoff
            )
            if len(distances) > 0:
                all_distances.extend(distances.tolist())
                all_angles.extend(angles.tolist())
        else:
            # Global-only detection
            n_hbonds, _, _, _, distances, angles = detect_hbonds_frame(
                positions, water_molecules, box_lengths, d_a_cutoff, d_h_a_angle_cutoff
            )

            time_series['global']['n_hbonds'][frame_idx] = n_hbonds
            time_series['global']['n_water'][frame_idx] = n_water
            # <nhbond> = (n_hbonds * 2) / n_water
            if n_water > 0:
                time_series['global']['nhbond'][frame_idx] = (n_hbonds * 2) / n_water
            else:
                time_series['global']['nhbond'][frame_idx] = 0.0

            if len(distances) > 0:
                all_distances.extend(distances.tolist())
                all_angles.extend(angles.tolist())

    # Compute summary statistics with block averaging
    summary = {}
    for rname in region_names:
        nhbond_series = time_series[rname]['nhbond']
        n_hbonds_series = time_series[rname]['n_hbonds']
        n_water_series = time_series[rname]['n_water']

        mean_nhbond, std_nhbond = _compute_block_statistics(nhbond_series, n_blocks)
        mean_n_hbonds, std_n_hbonds = _compute_block_statistics(n_hbonds_series.astype(float), n_blocks)

        # Average water count in region
        avg_n_water = np.mean(n_water_series)

        summary[rname] = {
            'mean_nhbond': mean_nhbond,
            'std_nhbond': std_nhbond,
            'mean_n_hbonds': mean_n_hbonds,
            'std_n_hbonds': std_n_hbonds,
            'n_water': int(avg_n_water),
        }

    _log(logger, 'success', f"H-bond analysis complete", verbose)

    return {
        'time_series': time_series,
        'summary': summary,
        'geometry': {
            'distances': np.array(all_distances),
            'angles': np.array(all_angles),
        },
        'n_water': n_water,
        'n_frames': n_frames,
        'regions': regions,
        'parameters': {
            'd_a_cutoff': d_a_cutoff,
            'd_h_a_angle_cutoff': d_h_a_angle_cutoff,
        }
    }
