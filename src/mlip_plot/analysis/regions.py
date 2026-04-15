"""
Shared z-region definitions for MD trajectory analyses.

Unifies region handling across reorientation, H-bond, and diffusion analyses.

Two manual modes:
- Legacy (z_subsurface is None): returns {'interface_a', 'interface_b', 'bulk'}.
  Preserves historical behavior when only z_surface (= z_interface) and d_bulk
  are given.
- New (z_subsurface provided): returns {'surface', 'subsurface', 'bulk'} built
  up from the lower metal surface.

Auto mode: {'surface', 'subsurface', 'bulk'} detected from the oxygen density
profile at the lower metal surface (same behavior as the original reorientation
auto-detection).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# Semantic color palette keyed by region role. Legacy keys map to the same
# colors as the corresponding new keys so plots are visually consistent
# regardless of which manual mode was used.
REGION_COLORS: Dict[str, str] = {
    'surface': '#da4a64',       # Red/Pink
    'subsurface': '#8e5ea2',    # Purple
    'bulk': '#3cb44b',          # Green
    'interface_a': '#da4a64',
    'interface_b': '#6181ad',   # Blue
    'global': '#444444',
}

REGION_LABELS: Dict[str, str] = {
    'surface': 'Surface',
    'subsurface': 'Subsurface',
    'bulk': 'Bulk',
    'interface_a': 'Interface A (Lower)',
    'interface_b': 'Interface B (Upper)',
    'global': 'Global',
}

# Canonical plot ordering. Regions not in this list fall back to z-sorted.
_REGION_ORDER: Tuple[str, ...] = (
    'interface_a', 'surface', 'subsurface', 'bulk', 'interface_b',
)


def _log(logger: Optional[Any], method: str, message: str, verbose: bool = True):
    if not verbose:
        return
    if logger is not None:
        getattr(logger, method)(message)
    else:
        print(message)


def get_display_label(region_name: str) -> str:
    """Return a human-readable label for a region key."""
    return REGION_LABELS.get(region_name, region_name.replace('_', ' ').title())


def get_display_labels(regions: Dict[str, Tuple[float, float]]) -> Dict[str, str]:
    """Return a display-label dict for every key in ``regions``."""
    return {name: get_display_label(name) for name in regions}


def get_region_color(region_name: str) -> str:
    """Return the plot color associated with a region key."""
    return REGION_COLORS.get(region_name, '#888888')


def ordered_region_names(
    regions: Dict[str, Tuple[float, float]]
) -> List[str]:
    """
    Return region names in canonical plotting order.

    Known roles (interface_a, surface, subsurface, bulk, interface_b) are
    ordered via a fixed list. Any extra keys are appended sorted by z_min.
    """
    known = [name for name in _REGION_ORDER if name in regions]
    extras = [name for name in regions if name not in _REGION_ORDER]
    extras.sort(key=lambda n: regions[n][0])
    return known + extras


def define_manual_regions(
    z_lo: float,
    z_hi: float,
    z_surface: float,
    d_bulk: float,
    z_subsurface: Optional[float] = None,
    lower_metal_surface_z: Optional[float] = None,
    upper_metal_surface_z: Optional[float] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Define z-regions from user-supplied thicknesses.

    Parameters
    ----------
    z_lo, z_hi : float
        Simulation box z-boundaries.
    z_surface : float
        Thickness of the surface (or legacy "interface") layer from each metal
        surface, in Angstroms.
    d_bulk : float
        Half-width of the bulk region around the water-region midpoint.
    z_subsurface : float, optional
        If provided, enables the new 3-region scheme. Thickness of the
        subsurface layer that sits between the surface layer and bulk.
        If ``None`` the legacy ``interface_a/interface_b/bulk`` scheme is
        returned with ``z_surface`` playing the role of the old
        ``z_interface`` argument (fully backward compatible).
    lower_metal_surface_z : float, optional
        Top of the lower metal slab. Defaults to ``z_lo``.
    upper_metal_surface_z : float, optional
        Bottom of the upper metal slab. Defaults to ``z_hi``.

    Returns
    -------
    regions : dict
        Mapping of region name to (z_min, z_max).
    """
    lower = lower_metal_surface_z if lower_metal_surface_z is not None else z_lo
    upper = upper_metal_surface_z if upper_metal_surface_z is not None else z_hi
    midpoint = (lower + upper) / 2.0

    if z_subsurface is None:
        return {
            'interface_a': (lower, lower + z_surface),
            'interface_b': (upper - z_surface, upper),
            'bulk': (midpoint - d_bulk, midpoint + d_bulk),
        }

    surface_end = lower + z_surface
    subsurface_end = surface_end + z_subsurface
    return {
        'surface': (lower, surface_end),
        'subsurface': (surface_end, subsurface_end),
        'bulk': (midpoint - d_bulk, midpoint + d_bulk),
    }


def detect_layer_boundaries(
    frames: List[Dict],
    z_surface: float,
    z_max: float,
    n_bins: int = 300,
    sigma: float = 2.0,
    verbose: bool = True,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Detect water layer boundaries from the oxygen density profile.

    Algorithm:
      1. Build O density histogram from ``z_surface`` to ``z_max``.
      2. Smooth with a Gaussian filter.
      3. First peak within 6 A of the surface -> surface-layer maximum.
      4. First minimum within 4 A after the peak -> surface/subsurface.
      5. Second minimum within 6 A after the first -> subsurface/bulk.

    Returns a dict with the detected boundaries plus the density profile.
    """
    if not SCIPY_AVAILABLE:
        raise ValueError("scipy is required for auto-layer detection (gaussian_filter1d)")

    z_upper = min(z_surface + 20, z_max)
    bins = np.linspace(z_surface - 1, z_upper, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0]
    hist = np.zeros(len(bins) - 1)

    elements = frames[0].get('elements')
    if elements is None:
        raise ValueError("Frames must contain element information for auto-layer detection")

    for frame in frames:
        positions = frame['positions']
        frame_elements = frame['elements']
        for i, elem in enumerate(frame_elements):
            if elem == 'O':
                z = positions[i, 2]
                idx = np.searchsorted(bins, z) - 1
                if 0 <= idx < len(hist):
                    hist[idx] += 1

    hist = hist / len(frames) / bin_width
    hist_smooth = gaussian_filter1d(hist, sigma=sigma)

    search_region = (bin_centers >= z_surface) & (bin_centers < z_surface + 6)
    if not np.any(search_region):
        raise ValueError(f"No density data found near surface (z={z_surface:.2f})")
    search_idx = np.where(search_region)[0]
    peak_local_idx = np.argmax(hist_smooth[search_region])
    z_surface_peak = bin_centers[search_idx[peak_local_idx]]
    if verbose:
        _log(logger, 'detail', f"  Surface layer peak: z = {z_surface_peak:.2f} A")

    search_region = (bin_centers > z_surface_peak) & (bin_centers < z_surface_peak + 4)
    if not np.any(search_region):
        raise ValueError(f"Cannot find minimum after surface peak at z={z_surface_peak:.2f}")
    search_idx = np.where(search_region)[0]
    min1_local_idx = np.argmin(hist_smooth[search_region])
    z_surface_min = bin_centers[search_idx[min1_local_idx]]
    if verbose:
        _log(logger, 'detail', f"  Surface/subsurface boundary: z = {z_surface_min:.2f} A")

    search_region = (bin_centers > z_surface_min + 1.0) & (bin_centers < z_surface_min + 6)
    if not np.any(search_region):
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
    logger: Optional[Any] = None,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
    """
    Auto-detect water regions from the oxygen density profile.

    Returns ``{'surface', 'subsurface', 'bulk'}`` (lower surface only) plus a
    diagnostics dict containing the density profile and detected boundaries.
    """
    # Local import to avoid a circular dependency with analysis.diffusion,
    # which imports this module for its thin wrapper.
    from .diffusion import find_metal_surfaces

    box = frames[0]['box']
    z_lo = box['zlo']
    z_hi = box['zhi']
    z_mid = (z_lo + z_hi) / 2.0

    try:
        lower_surface_z, upper_surface_z, found_metals = find_metal_surfaces(frames)
        if verbose:
            _log(logger, 'info', f"Metal surfaces detected: {found_metals}")
            _log(logger, 'detail', f"  Lower surface top: {lower_surface_z:.2f} A")
            if upper_surface_z is not None:
                _log(logger, 'detail', f"  Upper surface bottom: {upper_surface_z:.2f} A")
                z_mid = (lower_surface_z + upper_surface_z) / 2.0
    except ValueError:
        if verbose:
            _log(logger, 'warning', "No metal surfaces detected, using box z_lo as surface")
        lower_surface_z = z_lo
        upper_surface_z = None

    if verbose:
        _log(logger, 'info', "Detecting layer boundaries from O density profile...")

    boundaries = detect_layer_boundaries(
        frames,
        z_surface=lower_surface_z,
        z_max=z_mid,
        verbose=verbose,
        logger=logger,
    )

    regions = {
        'surface': (lower_surface_z, boundaries['z_surface_min']),
        'subsurface': (boundaries['z_surface_min'], boundaries['z_subsurface_min']),
        'bulk': (boundaries['z_subsurface_min'], z_mid),
    }

    if verbose:
        _log(logger, 'info', "Auto-detected regions:")
        for region_name, (rz_min, rz_max) in regions.items():
            _log(logger, 'detail', f"  {region_name}: {rz_min:.2f} to {rz_max:.2f} A")

    diagnostics = {
        'boundaries': boundaries,
        'z_mid': z_mid,
        'lower_surface_z': lower_surface_z,
        'upper_surface_z': upper_surface_z,
    }
    return regions, diagnostics


def define_regions(
    frames: List[Dict],
    auto: bool,
    manual_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    logger: Optional[Any] = None,
) -> Tuple[Optional[Dict[str, Tuple[float, float]]], Optional[Dict[str, Any]]]:
    """
    Top-level dispatcher used by CLIs.

    Parameters
    ----------
    frames : list of dict
        Trajectory frames.
    auto : bool
        If True, run auto-detection (ignores manual_kwargs).
    manual_kwargs : dict, optional
        Keyword arguments forwarded to ``define_manual_regions``. Must contain
        at least ``z_surface`` and ``d_bulk``. May contain ``z_subsurface``,
        ``lower_metal_surface_z``, ``upper_metal_surface_z``.
    verbose, logger
        Diagnostics reporting.

    Returns
    -------
    regions : dict or None
        Mapping of region name to (z_min, z_max). ``None`` if neither auto nor
        manual is requested.
    diagnostics : dict or None
        Auto-detection diagnostics (density profile, boundaries). ``None`` in
        manual mode.
    """
    if auto:
        regions, diagnostics = define_auto_regions(frames, verbose=verbose, logger=logger)
        return regions, diagnostics

    if manual_kwargs is None:
        return None, None

    from .diffusion import find_metal_surfaces

    box = frames[0]['box']
    z_lo = box['zlo']
    z_hi = box['zhi']

    lower = manual_kwargs.get('lower_metal_surface_z')
    upper = manual_kwargs.get('upper_metal_surface_z')
    if lower is None and upper is None:
        try:
            lower, upper, found_metals = find_metal_surfaces(frames)
            if verbose:
                _log(logger, 'info', f"Metal surfaces detected: {found_metals}")
                _log(logger, 'detail', f"  Lower surface top: {lower:.2f} A")
                if upper is not None:
                    _log(logger, 'detail', f"  Upper surface bottom: {upper:.2f} A")
        except ValueError:
            if verbose:
                _log(logger, 'info', "No metal surfaces detected, using box boundaries")
            lower = None
            upper = None

    regions = define_manual_regions(
        z_lo=z_lo,
        z_hi=z_hi,
        z_surface=manual_kwargs['z_surface'],
        d_bulk=manual_kwargs['d_bulk'],
        z_subsurface=manual_kwargs.get('z_subsurface'),
        lower_metal_surface_z=lower,
        upper_metal_surface_z=upper,
    )

    if verbose:
        for region_name, (rz_min, rz_max) in regions.items():
            _log(logger, 'detail', f"  {region_name}: {rz_min:.2f} to {rz_max:.2f} A")

    return regions, None
