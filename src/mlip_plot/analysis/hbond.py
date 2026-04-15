"""
Hydrogen bond analysis for MD trajectories.

Computes hydrogen bond statistics for bulk and region-based (metal-water interface) analysis.
Supports block averaging for error estimation and profiles along z and time.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..utils.hbond import HydrogenBond
from ..utils.water import WaterMolecule
from .diffusion import find_metal_surfaces, SUPPORTED_METALS
from .regions import define_manual_regions, define_auto_regions


def _log(logger: Optional[Any], method: str, message: str, verbose: bool = True):
    """Helper to log or print based on logger availability."""
    if not verbose:
        return
    if logger is not None:
        getattr(logger, method)(message)
    else:
        print(message)


def define_z_regions_hbond(
    z_lo: float,
    z_hi: float,
    z_interface: float,
    d_bulk: float,
    lower_metal_surface_z: Optional[float] = None,
    upper_metal_surface_z: Optional[float] = None,
    z_subsurface: Optional[float] = None,
) -> Dict[str, Tuple[float, float]]:
    """Thin wrapper around :func:`regions.define_manual_regions`.

    ``z_interface`` plays the role of ``z_surface``. When ``z_subsurface`` is
    omitted the legacy ``interface_a/interface_b/bulk`` scheme is returned.
    """
    return define_manual_regions(
        z_lo=z_lo,
        z_hi=z_hi,
        z_surface=z_interface,
        d_bulk=d_bulk,
        z_subsurface=z_subsurface,
        lower_metal_surface_z=lower_metal_surface_z,
        upper_metal_surface_z=upper_metal_surface_z,
    )


def _get_water_z_positions(water: WaterMolecule) -> Tuple[float, float, float]:
    """Get z coordinates of water (O, H1, H2)."""
    return (
        float(water.O_position[2]),
        float(water.H1_position[2]),
        float(water.H2_position[2])
    )


def _is_water_in_region(
    water: WaterMolecule,
    z_min: float,
    z_max: float,
    allow_fraction: bool = False
) -> bool:
    """
    Check if water is within z region.

    Parameters
    ----------
    water : WaterMolecule
        Water molecule to check
    z_min : float
        Minimum z coordinate of region
    z_max : float
        Maximum z coordinate of region
    allow_fraction : bool
        If True, ANY atom (O, H1, H2) in region counts.
        If False, ALL atoms must be in region.

    Returns
    -------
    bool
        True if water is in region according to the criteria
    """
    z_O, z_H1, z_H2 = _get_water_z_positions(water)

    O_in = z_min <= z_O <= z_max
    H1_in = z_min <= z_H1 <= z_max
    H2_in = z_min <= z_H2 <= z_max

    if allow_fraction:
        # ANY atom in region
        return O_in or H1_in or H2_in
    else:
        # ALL atoms must be in region
        return O_in and H1_in and H2_in


def _block_average(values: np.ndarray, n_blocks: int) -> Tuple[float, float]:
    """
    Compute mean and std from block averaging.

    Parameters
    ----------
    values : ndarray
        Array of values (one per frame)
    n_blocks : int
        Number of blocks

    Returns
    -------
    mean : float
        Mean of block averages
    std : float
        Standard deviation of block averages
    """
    n = len(values)
    if n_blocks <= 1 or n < n_blocks:
        return float(np.mean(values)), 0.0

    block_size = n // n_blocks
    block_means = np.zeros(n_blocks)

    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else n
        block_means[i] = np.mean(values[start:end])

    return float(np.mean(block_means)), float(np.std(block_means, ddof=1))


def compute_hbond_statistics_bulk(
    hbonds: List[List[HydrogenBond]],
    n_waters: int,
    n_blocks: int = 5
) -> Dict:
    """
    Compute bulk hydrogen bond statistics with block averaging.

    The formula for hbonds/water is: 2 * total_hbonds / n_waters
    (factor of 2 because each hbond involves both a donor and acceptor).

    Parameters
    ----------
    hbonds : list of list of HydrogenBond
        H-bonds per frame from identify_hbonds()
    n_waters : int
        Number of water molecules
    n_blocks : int
        Number of blocks for error estimation (default: 5)

    Returns
    -------
    results : dict
        Dictionary containing:
        - hbonds_per_frame: array of hbond counts per frame
        - hbonds_per_water_per_frame: array of hbonds/water per frame
        - mean: mean hbonds/water
        - std: standard deviation from block averaging
        - total_hbonds: total hbonds across all frames
        - n_frames: number of frames
        - n_waters: number of water molecules
    """
    n_frames = len(hbonds)
    hbonds_per_frame = np.array([len(frame_hbonds) for frame_hbonds in hbonds])

    # hbonds/water = 2 * n_hbonds / n_waters
    hbonds_per_water = 2.0 * hbonds_per_frame / n_waters

    mean, std = _block_average(hbonds_per_water, n_blocks)

    return {
        'hbonds_per_frame': hbonds_per_frame,
        'hbonds_per_water_per_frame': hbonds_per_water,
        'mean': mean,
        'std': std,
        'total_hbonds': int(np.sum(hbonds_per_frame)),
        'n_frames': n_frames,
        'n_waters': n_waters,
        'n_blocks': n_blocks,
    }


def compute_hbond_statistics_regions(
    hbonds: List[List[HydrogenBond]],
    waters: List[List[WaterMolecule]],
    regions: Dict[str, Tuple[float, float]],
    n_blocks: int = 5,
    allow_fraction: bool = False,
    da_inside: bool = False
) -> Dict[str, Dict]:
    """
    Compute hydrogen bond statistics per region with block averaging.

    Water in region check:
    - If allow_fraction=True: ANY of O, H1, H2 within z bounds counts
    - If allow_fraction=False: ALL of O, H1, H2 must be within z bounds

    H-bond counting:
    - If da_inside=True: BOTH donor AND acceptor water must be in region
    - If da_inside=False: EITHER donor OR acceptor water in region counts

    Parameters
    ----------
    hbonds : list of list of HydrogenBond
        H-bonds per frame from identify_hbonds()
    waters : list of list of WaterMolecule
        Water molecules per frame from extract_water_molecules()
    regions : dict
        Dictionary mapping region names to (z_min, z_max) tuples
    n_blocks : int
        Number of blocks for error estimation (default: 5)
    allow_fraction : bool
        If True, water with ANY atom in region counts (default: False)
    da_inside : bool
        If True, BOTH donor and acceptor must be in region.
        If False, EITHER donor or acceptor in region counts (default: False)

    Returns
    -------
    results : dict
        Nested dict: {region_name: {mean, std, hbonds_per_frame, ...}}
    """
    n_frames = len(hbonds)
    results = {}

    for region_name, (z_min, z_max) in regions.items():
        hbonds_in_region = np.zeros(n_frames)
        waters_in_region = np.zeros(n_frames)

        for f in range(n_frames):
            frame_waters = waters[f]
            frame_hbonds = hbonds[f]

            # Build set of water indices in this region
            water_in_region_mask = []
            for w_idx, water in enumerate(frame_waters):
                in_region = _is_water_in_region(water, z_min, z_max, allow_fraction)
                water_in_region_mask.append(in_region)

            waters_in_region[f] = sum(water_in_region_mask)

            # Count hbonds in this region
            for hb in frame_hbonds:
                donor_in = water_in_region_mask[hb.donor_water_idx]
                acceptor_in = water_in_region_mask[hb.acceptor_water_idx]

                if da_inside:
                    # BOTH must be in region
                    if donor_in and acceptor_in:
                        hbonds_in_region[f] += 1
                else:
                    # EITHER in region counts
                    if donor_in or acceptor_in:
                        hbonds_in_region[f] += 1

        # Compute hbonds/water per frame
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            hbonds_per_water = np.where(
                waters_in_region > 0,
                2.0 * hbonds_in_region / waters_in_region,
                0.0
            )

        mean, std = _block_average(hbonds_per_water, n_blocks)

        results[region_name] = {
            'hbonds_per_frame': hbonds_in_region,
            'waters_per_frame': waters_in_region,
            'hbonds_per_water_per_frame': hbonds_per_water,
            'mean': mean,
            'std': std,
            'total_hbonds': int(np.sum(hbonds_in_region)),
            'avg_waters': float(np.mean(waters_in_region)),
            'n_frames': n_frames,
            'n_blocks': n_blocks,
            'z_range': (z_min, z_max),
        }

    return results


def compute_hbond_z_profile(
    hbonds: List[List[HydrogenBond]],
    waters: List[List[WaterMolecule]],
    z_min: float,
    z_max: float,
    n_bins: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute average H-bonds per water along z direction.

    For each bin, computes: 2 * n_hbonds / n_waters

    Parameters
    ----------
    hbonds : list of list of HydrogenBond
        H-bonds per frame from identify_hbonds()
    waters : list of list of WaterMolecule
        Water molecules per frame from extract_water_molecules()
    z_min : float
        Minimum z coordinate
    z_max : float
        Maximum z coordinate
    n_bins : int
        Number of bins for histogram (default: 50)

    Returns
    -------
    bin_centers : ndarray
        Center of each bin
    hbonds_per_water : ndarray
        Average H-bonds per water for each bin (2 * n_hbonds / n_waters)
    water_counts : ndarray
        Total water count per bin (summed over all frames)
    hbond_counts : ndarray
        Total H-bond count per bin (summed over all frames)
    """
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Accumulate counts across all frames
    water_counts = np.zeros(n_bins)
    hbond_counts = np.zeros(n_bins)

    n_frames = len(hbonds)

    for f in range(n_frames):
        frame_waters = waters[f]
        frame_hbonds = hbonds[f]

        # Get z positions of all waters (oxygen position)
        water_z = np.array([w.O_position[2] for w in frame_waters])

        # Bin water molecules
        water_bin_idx = np.digitize(water_z, bin_edges) - 1
        water_bin_idx = np.clip(water_bin_idx, 0, n_bins - 1)

        # Count waters per bin
        frame_water_counts, _ = np.histogram(water_z, bins=bin_edges)
        water_counts += frame_water_counts

        # Count H-bonds per bin (based on donor oxygen position)
        for hb in frame_hbonds:
            donor_z = hb.donor_position[2]
            if z_min <= donor_z < z_max:
                bin_idx = int((donor_z - z_min) / (z_max - z_min) * n_bins)
                bin_idx = min(bin_idx, n_bins - 1)
                hbond_counts[bin_idx] += 1

    # Compute H-bonds per water: 2 * n_hbonds / n_waters
    with np.errstate(divide='ignore', invalid='ignore'):
        hbonds_per_water = np.where(
            water_counts > 0,
            2.0 * hbond_counts / water_counts,
            0.0
        )

    return bin_centers, hbonds_per_water, water_counts, hbond_counts


def compute_hbond_time_profile(
    hbonds: List[List[HydrogenBond]],
    n_waters: int,
    dt: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute hydrogen bonds per water vs time.

    Parameters
    ----------
    hbonds : list of list of HydrogenBond
        H-bonds per frame from identify_hbonds()
    n_waters : int
        Number of water molecules
    dt : float
        Time between frames in picoseconds (default: 1.0)

    Returns
    -------
    time : ndarray
        Time array in picoseconds
    hbonds_per_water : ndarray
        H-bonds per water at each time point
    """
    n_frames = len(hbonds)
    time = np.arange(n_frames) * dt
    hbonds_per_frame = np.array([len(frame_hbonds) for frame_hbonds in hbonds])
    hbonds_per_water = 2.0 * hbonds_per_frame / n_waters

    return time, hbonds_per_water


def save_hbonds_numpy(
    hbonds: List[List[HydrogenBond]],
    waters: List[List[WaterMolecule]],
    output_prefix: str
):
    """
    Save hydrogen bonds and water molecules as numpy arrays.

    Parameters
    ----------
    hbonds : list of list of HydrogenBond
        H-bonds per frame from identify_hbonds()
    waters : list of list of WaterMolecule
        Water molecules per frame from extract_water_molecules()
    output_prefix : str
        Output file prefix (without extension)

    Files created:
    - {prefix}_hbonds.npz: H-bond data per frame
        - n_hbonds: array of hbond counts per frame
        - donor_water_idx: list of arrays (index into waters)
        - acceptor_water_idx: list of arrays (index into waters)
        - donor_O_idx: list of arrays (atom index of donor oxygen)
        - acceptor_O_idx: list of arrays (atom index of acceptor oxygen)
        - H_idx: list of arrays (atom index of bridging hydrogen)
        - donor_positions: list of arrays (n_hbonds, 3)
        - acceptor_positions: list of arrays (n_hbonds, 3)
        - H_positions: list of arrays (n_hbonds, 3)
        - DA_distances: list of arrays
        - HA_distances: list of arrays
        - DHA_angles: list of arrays
    - {prefix}_waters.npz: Water molecule data per frame
        - n_waters: number of waters per frame
        - O_positions: list of arrays (n_waters, 3)
        - H1_positions: list of arrays
        - H2_positions: list of arrays
    """
    n_frames = len(hbonds)

    # H-bond data
    n_hbonds = np.array([len(frame_hbonds) for frame_hbonds in hbonds])
    donor_water_idx = []
    acceptor_water_idx = []
    donor_O_idx = []
    acceptor_O_idx = []
    H_idx = []
    donor_positions = []
    acceptor_positions = []
    H_positions = []
    DA_distances = []
    HA_distances = []
    DHA_angles = []

    for frame_hbonds in hbonds:
        if len(frame_hbonds) > 0:
            donor_water_idx.append(np.array([hb.donor_water_idx for hb in frame_hbonds], dtype=np.int32))
            acceptor_water_idx.append(np.array([hb.acceptor_water_idx for hb in frame_hbonds], dtype=np.int32))
            donor_O_idx.append(np.array([hb.donor_O_idx for hb in frame_hbonds], dtype=np.int32))
            acceptor_O_idx.append(np.array([hb.acceptor_O_idx for hb in frame_hbonds], dtype=np.int32))
            H_idx.append(np.array([hb.H_idx for hb in frame_hbonds], dtype=np.int32))
            donor_positions.append(np.array([hb.donor_position for hb in frame_hbonds], dtype=np.float64))
            acceptor_positions.append(np.array([hb.acceptor_position for hb in frame_hbonds], dtype=np.float64))
            H_positions.append(np.array([hb.H_position for hb in frame_hbonds], dtype=np.float64))
            DA_distances.append(np.array([hb.DA_distance for hb in frame_hbonds], dtype=np.float64))
            HA_distances.append(np.array([hb.HA_distance for hb in frame_hbonds], dtype=np.float64))
            DHA_angles.append(np.array([hb.DHA_angle for hb in frame_hbonds], dtype=np.float64))
        else:
            donor_water_idx.append(np.array([], dtype=np.int32))
            acceptor_water_idx.append(np.array([], dtype=np.int32))
            donor_O_idx.append(np.array([], dtype=np.int32))
            acceptor_O_idx.append(np.array([], dtype=np.int32))
            H_idx.append(np.array([], dtype=np.int32))
            donor_positions.append(np.zeros((0, 3), dtype=np.float64))
            acceptor_positions.append(np.zeros((0, 3), dtype=np.float64))
            H_positions.append(np.zeros((0, 3), dtype=np.float64))
            DA_distances.append(np.array([], dtype=np.float64))
            HA_distances.append(np.array([], dtype=np.float64))
            DHA_angles.append(np.array([], dtype=np.float64))

    np.savez(
        f"{output_prefix}_hbonds.npz",
        n_hbonds=n_hbonds,
        donor_water_idx=np.array(donor_water_idx, dtype=object),
        acceptor_water_idx=np.array(acceptor_water_idx, dtype=object),
        donor_O_idx=np.array(donor_O_idx, dtype=object),
        acceptor_O_idx=np.array(acceptor_O_idx, dtype=object),
        H_idx=np.array(H_idx, dtype=object),
        donor_positions=np.array(donor_positions, dtype=object),
        acceptor_positions=np.array(acceptor_positions, dtype=object),
        H_positions=np.array(H_positions, dtype=object),
        DA_distances=np.array(DA_distances, dtype=object),
        HA_distances=np.array(HA_distances, dtype=object),
        DHA_angles=np.array(DHA_angles, dtype=object),
    )

    # Water molecule data
    n_waters_per_frame = np.array([len(frame_waters) for frame_waters in waters])
    O_positions = []
    H1_positions = []
    H2_positions = []

    for frame_waters in waters:
        if len(frame_waters) > 0:
            O_positions.append(np.array([w.O_position for w in frame_waters]))
            H1_positions.append(np.array([w.H1_position for w in frame_waters]))
            H2_positions.append(np.array([w.H2_position for w in frame_waters]))
        else:
            O_positions.append(np.array([]).reshape(0, 3))
            H1_positions.append(np.array([]).reshape(0, 3))
            H2_positions.append(np.array([]).reshape(0, 3))

    np.savez(
        f"{output_prefix}_waters.npz",
        n_waters=n_waters_per_frame,
        O_positions=np.array(O_positions, dtype=object),
        H1_positions=np.array(H1_positions, dtype=object),
        H2_positions=np.array(H2_positions, dtype=object),
    )


def analyze_hbonds(
    frames: List[Dict],
    hbonds: List[List[HydrogenBond]],
    waters: List[List[WaterMolecule]],
    n_blocks: int = 5,
    z_interface: Optional[float] = None,
    d_bulk: Optional[float] = None,
    z_subsurface: Optional[float] = None,
    auto_layers: bool = False,
    allow_fraction: bool = False,
    da_inside: bool = False,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> Dict:
    """
    Comprehensive hydrogen bond analysis.

    Parameters
    ----------
    frames : list of dict
        Trajectory frames from read_lammpstrj
    hbonds : list of list of HydrogenBond
        H-bonds per frame from identify_hbonds()
    waters : list of list of WaterMolecule
        Water molecules per frame from extract_water_molecules()
    n_blocks : int
        Number of blocks for error estimation (default: 5)
    z_interface : float, optional
        Interface thickness from each surface (Angstroms)
    d_bulk : float, optional
        Half-width of bulk region around midpoint (Angstroms)
    allow_fraction : bool
        If True, water with ANY atom in region counts (default: False)
    da_inside : bool
        If True, BOTH donor and acceptor must be in region (default: False)
    verbose : bool
        Print progress (default: True)
    logger : optional
        Logger instance for output

    Returns
    -------
    results : dict
        Dictionary containing:
        - bulk: bulk statistics (always present)
        - regions: region-based statistics (if z_interface/d_bulk provided)
        - region_definitions: region z-ranges (if region analysis)
        - n_frames: number of frames
        - n_waters: number of water molecules
    """
    n_frames = len(hbonds)
    n_waters = len(waters[0]) if waters else 0

    # Validate region parameters
    if (z_interface is None) != (d_bulk is None):
        raise ValueError("z_interface and d_bulk must both be provided, or neither")
    manual_regions = z_interface is not None
    use_auto_layers = auto_layers and not manual_regions
    use_region_analysis = manual_regions or use_auto_layers

    if verbose:
        _log(logger, 'info', f"Analyzing {n_frames} frames with {n_waters} water molecules")

    # Always compute bulk statistics
    if verbose:
        _log(logger, 'info', "Computing bulk H-bond statistics...")

    bulk_stats = compute_hbond_statistics_bulk(hbonds, n_waters, n_blocks)

    results = {
        'bulk': bulk_stats,
        'n_frames': n_frames,
        'n_waters': n_waters,
    }

    # Region-based analysis
    if use_region_analysis:
        if verbose:
            mode_str = "auto-detected" if use_auto_layers else "manual"
            _log(logger, 'info', f"Performing region-based analysis ({mode_str})...")

        regions = None
        diagnostics = None

        if use_auto_layers:
            try:
                regions, diagnostics = define_auto_regions(
                    frames, verbose=verbose, logger=logger
                )
            except ValueError as e:
                if verbose:
                    _log(logger, 'warning', f"Auto-layer detection failed: {e}")
                    _log(logger, 'warning', "Falling back to bulk-only analysis")
                regions = None

        else:
            # Manual regions: determine metal surfaces for anchoring.
            box = frames[0]['box']
            z_lo = box['zlo']
            z_hi = box['zhi']
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

            regions = define_z_regions_hbond(
                z_lo, z_hi, z_interface, d_bulk,
                lower_metal_surface_z=lower_surface_z,
                upper_metal_surface_z=upper_surface_z,
                z_subsurface=z_subsurface,
            )

            if verbose:
                for region_name, (rz_min, rz_max) in regions.items():
                    _log(logger, 'detail', f"  {region_name}: {rz_min:.2f} to {rz_max:.2f} A")

        if regions is not None:
            region_stats = compute_hbond_statistics_regions(
                hbonds, waters, regions, n_blocks, allow_fraction, da_inside
            )
            results['regions'] = region_stats
            results['region_definitions'] = regions
            if diagnostics is not None:
                results['layer_diagnostics'] = diagnostics
                results['region_mode'] = 'auto'
            else:
                results['region_mode'] = 'manual'

    return results
