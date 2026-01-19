"""
Water molecule identification and property extraction.

This module provides utilities for identifying water molecules in MD trajectories
and extracting their geometric properties (positions, distances, angles).
Supports periodic boundary conditions via minimum image convention.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Import C++ core (required)
try:
    from mlip_plot._core import identify_water_molecules as _cpp_identify_water
    from mlip_plot._core import extract_water_properties as _cpp_extract_properties
except ImportError:
    raise ImportError("C++ core module not available. Please rebuild the package.")


@dataclass(frozen=True, slots=True)
class WaterMolecule:
    """
    Represents a water molecule with its geometric properties.

    Attributes
    ----------
    O_index : int
        Index of the oxygen atom in the frame
    H1_index : int
        Index of the first hydrogen atom
    H2_index : int
        Index of the second hydrogen atom
    O_position : np.ndarray
        Position of oxygen atom [x, y, z]
    H1_position : np.ndarray
        Position of first hydrogen [x, y, z]
    H2_position : np.ndarray
        Position of second hydrogen [x, y, z]
    OH1_distance : float
        O-H1 bond distance in Angstroms
    OH2_distance : float
        O-H2 bond distance in Angstroms
    H1H2_distance : float
        H1-H2 distance in Angstroms
    H1OH2_angle : float
        H1-O-H2 bond angle in degrees
    """
    O_index: int
    H1_index: int
    H2_index: int
    O_position: np.ndarray
    H1_position: np.ndarray
    H2_position: np.ndarray
    OH1_distance: float
    OH2_distance: float
    H1H2_distance: float
    H1OH2_angle: float

    def __repr__(self) -> str:
        return (f"WaterMolecule(O={self.O_index}, H1={self.H1_index}, H2={self.H2_index}, "
                f"angle={self.H1OH2_angle:.1f}deg)")


def _get_box_lengths(frames: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """Extract box lengths from frame data."""
    box = frames[0]['box']
    Lx = box['xhi'] - box['xlo']
    Ly = box['yhi'] - box['ylo']
    Lz = box['zhi'] - box['zlo']
    return (Lx, Ly, Lz)


def identify_water_molecules(
    frames: List[Dict[str, Any]],
    o_h_cutoff: float = 1.2,
) -> Tuple[np.ndarray, int]:
    """
    Identify water molecules from trajectory frames.

    Uses the first frame to identify O-H bonding topology, which is
    then assumed to remain constant throughout the trajectory.
    Supports periodic boundary conditions via minimum image convention.

    Parameters
    ----------
    frames : list of dict
        Trajectory frames from read_lammpstrj
    o_h_cutoff : float, optional
        O-H bond distance cutoff in Angstroms (default: 1.2)

    Returns
    -------
    molecule_indices : ndarray, shape (n_molecules, 3)
        Array of [O_index, H1_index, H2_index] for each water molecule
    n_molecules : int
        Number of water molecules found

    Raises
    ------
    ValueError
        If frames don't contain element information or no water molecules found
    """
    if not frames:
        raise ValueError("No frames provided")

    if frames[0]['elements'] is None:
        raise ValueError("Frames must contain element information")

    # Build element-to-type mapping
    elements = frames[0]['elements']
    types = frames[0]['types']
    positions = frames[0]['positions']

    # Find O and H type indices
    o_type_idx = None
    h_type_idx = None

    for elem, typ in zip(elements, types):
        if elem == 'O' and o_type_idx is None:
            o_type_idx = typ - 1  # 0-indexed
        elif elem == 'H' and h_type_idx is None:
            h_type_idx = typ - 1

        if o_type_idx is not None and h_type_idx is not None:
            break

    if o_type_idx is None:
        raise ValueError("No oxygen atoms found in trajectory")
    if h_type_idx is None:
        raise ValueError("No hydrogen atoms found in trajectory")

    # Convert types to 0-indexed
    element_types = (types - 1).astype(np.int32)

    # Get box dimensions for PBC
    box_lengths = _get_box_lengths(frames)

    # Use C++ backend
    molecule_indices, n_molecules = _cpp_identify_water(
        positions.astype(np.float64),
        element_types,
        o_type_idx,
        h_type_idx,
        box_lengths,
        o_h_cutoff
    )

    if n_molecules == 0:
        raise ValueError("No water molecules found")

    return molecule_indices, n_molecules




def extract_water_molecules(
    frames: List[Dict[str, Any]],
    o_h_cutoff: float = 1.2,
) -> List[List[WaterMolecule]]:
    """
    Extract water molecule properties for all frames.

    Each frame is identified independently to ensure correct O-H assignments
    even when atoms move significantly between frames.

    Supports periodic boundary conditions via minimum image convention
    for distance and angle calculations.

    Parameters
    ----------
    frames : list of dict
        Trajectory frames from read_lammpstrj
    o_h_cutoff : float, optional
        O-H bond cutoff for identification (default: 1.2)

    Returns
    -------
    water_molecules : list of list of WaterMolecule
        Nested list where water_molecules[frame_idx][mol_idx] gives
        the WaterMolecule object for that frame and molecule.

    Note
    ----
    Water molecules are identified independently for each frame, so
    water indices may not be consistent across frames (Water 0 in frame 0
    may be a different physical molecule than Water 0 in frame 1).

    Example
    -------
    >>> frames = read_lammpstrj('trajectory.lammpstrj')
    >>> waters = extract_water_molecules(frames)
    >>> # Get first water molecule in frame 0
    >>> mol = waters[0][0]
    >>> print(f"O-H distance: {mol.OH1_distance:.3f} A")
    >>> print(f"H-O-H angle: {mol.H1OH2_angle:.1f} degrees")
    """
    all_waters = []

    for frame in frames:
        # Identify water molecules for THIS frame
        molecule_indices, n_molecules = identify_water_molecules([frame], o_h_cutoff)

        if n_molecules == 0:
            all_waters.append([])
            continue

        # Get box dimensions for this frame
        box_lengths = _get_box_lengths([frame])

        # Use C++ backend for single frame
        positions_list = [frame['positions'].astype(np.float64)]

        props = _cpp_extract_properties(
            positions_list,
            molecule_indices.astype(np.int32),
            box_lengths
        )

        # Convert to WaterMolecule objects
        frame_waters = []
        for m in range(n_molecules):
            mol = WaterMolecule(
                O_index=int(molecule_indices[m, 0]),
                H1_index=int(molecule_indices[m, 1]),
                H2_index=int(molecule_indices[m, 2]),
                O_position=props['O_positions'][0, m].copy(),
                H1_position=props['H1_positions'][0, m].copy(),
                H2_position=props['H2_positions'][0, m].copy(),
                OH1_distance=float(props['OH1_distances'][0, m]),
                OH2_distance=float(props['OH2_distances'][0, m]),
                H1H2_distance=float(props['H1H2_distances'][0, m]),
                H1OH2_angle=float(props['H1OH2_angles'][0, m]),
            )
            frame_waters.append(mol)
        all_waters.append(frame_waters)

    return all_waters


