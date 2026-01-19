"""
Hydrogen bond identification between water molecules.

This module provides utilities for identifying hydrogen bonds in MD trajectories
using geometric criteria (D-A distance and D-H-A angle cutoffs).
Supports periodic boundary conditions via minimum image convention.
"""

from dataclasses import dataclass
from typing import List, Tuple, Union
import numpy as np

from .water import WaterMolecule

# Import C++ core (required)
try:
    from mlip_plot._core import identify_hbonds as _cpp_identify_hbonds
except ImportError:
    raise ImportError("C++ core module not available. Please rebuild the package.")


@dataclass(frozen=True, slots=True)
class HydrogenBond:
    """
    Represents a hydrogen bond D-H...A between water molecules.

    Attributes
    ----------
    donor_water_idx : int
        Index of the donor water molecule
    acceptor_water_idx : int
        Index of the acceptor water molecule
    donor_O_idx : int
        Atom index of the donor oxygen
    acceptor_O_idx : int
        Atom index of the acceptor oxygen
    H_idx : int
        Atom index of the bridging hydrogen
    donor_position : np.ndarray
        Position of donor oxygen [x, y, z]
    acceptor_position : np.ndarray
        Position of acceptor oxygen [x, y, z]
    H_position : np.ndarray
        Position of bridging hydrogen [x, y, z]
    DA_distance : float
        Donor-Acceptor distance in Angstroms
    HA_distance : float
        Hydrogen-Acceptor distance in Angstroms
    DHA_angle : float
        D-H-A angle in degrees
    """
    donor_water_idx: int
    acceptor_water_idx: int
    donor_O_idx: int
    acceptor_O_idx: int
    H_idx: int
    donor_position: np.ndarray
    acceptor_position: np.ndarray
    H_position: np.ndarray
    DA_distance: float
    HA_distance: float
    DHA_angle: float

    def __repr__(self) -> str:
        return (f"HydrogenBond(donor={self.donor_water_idx}, acceptor={self.acceptor_water_idx}, "
                f"D-A={self.DA_distance:.2f}A, angle={self.DHA_angle:.1f}deg)")


def identify_hbonds(
    waters: List[List[WaterMolecule]],
    box_lengths: Union[np.ndarray, List[Tuple[float, float, float]]],
    da_cutoff: float = 3.5,
    angle_cutoff: float = 150.0,
) -> List[List[HydrogenBond]]:
    """
    Identify hydrogen bonds between water molecules across all frames.

    A hydrogen bond D-H...A exists when:
    - Donor-Acceptor distance < da_cutoff
    - D-H-A angle > angle_cutoff

    Parameters
    ----------
    waters : list of list of WaterMolecule
        Water molecules per frame from extract_water_molecules()
    box_lengths : ndarray or list of tuples
        Box dimensions for PBC. Shape (n_frames, 3) or list of (Lx, Ly, Lz) tuples.
    da_cutoff : float, optional
        Donor-Acceptor distance cutoff in Angstroms (default: 3.5)
    angle_cutoff : float, optional
        Minimum D-H-A angle in degrees (default: 150.0)

    Returns
    -------
    hbonds : list of list of HydrogenBond
        Nested list where hbonds[frame_idx][bond_idx] gives
        the HydrogenBond object for that frame and bond.

    Example
    -------
    >>> frames = read_lammpstrj('trajectory.lammpstrj')
    >>> waters = extract_water_molecules(frames)
    >>> box_lengths = [(f['box']['xhi']-f['box']['xlo'],
    ...                 f['box']['yhi']-f['box']['ylo'],
    ...                 f['box']['zhi']-f['box']['zlo']) for f in frames]
    >>> hbonds = identify_hbonds(waters, box_lengths)
    >>> print(f"H-bonds in frame 0: {len(hbonds[0])}")
    """
    if not waters:
        raise ValueError("waters must not be empty")

    n_frames = len(waters)
    n_waters = len(waters[0])

    # Convert box_lengths to list of tuples if needed
    if isinstance(box_lengths, np.ndarray):
        box_lengths_list = [tuple(box_lengths[f]) for f in range(n_frames)]
    else:
        box_lengths_list = list(box_lengths)

    if len(box_lengths_list) != n_frames:
        raise ValueError(f"box_lengths length ({len(box_lengths_list)}) must match waters length ({n_frames})")

    # Get molecule indices from waters[0]
    molecule_indices = np.array([
        [w.O_index, w.H1_index, w.H2_index]
        for w in waters[0]
    ], dtype=np.int32)

    # Build positions arrays from waters for C++ backend
    # We pack positions as [O0, H1_0, H2_0, O1, H1_1, H2_1, ...] per frame
    # and create new indices [0,1,2], [3,4,5], ...
    packed_indices = np.array([
        [i * 3, i * 3 + 1, i * 3 + 2]
        for i in range(n_waters)
    ], dtype=np.int32)

    positions_list = []
    for f in range(n_frames):
        frame_positions = np.zeros((n_waters * 3, 3), dtype=np.float64)
        for i, w in enumerate(waters[f]):
            frame_positions[i * 3] = w.O_position
            frame_positions[i * 3 + 1] = w.H1_position
            frame_positions[i * 3 + 2] = w.H2_position
        positions_list.append(frame_positions)

    cpp_results = _cpp_identify_hbonds(
        positions_list,
        packed_indices,
        box_lengths_list,
        da_cutoff,
        angle_cutoff
    )

    # Convert to HydrogenBond objects
    all_hbonds = []
    for f in range(n_frames):
        frame_data = cpp_results[f]
        n_hbonds = frame_data['n_hbonds']
        frame_hbonds = []

        for h in range(n_hbonds):
            hb = HydrogenBond(
                donor_water_idx=int(frame_data['donor_water_idx'][h]),
                acceptor_water_idx=int(frame_data['acceptor_water_idx'][h]),
                donor_O_idx=int(molecule_indices[frame_data['donor_water_idx'][h], 0]),
                acceptor_O_idx=int(molecule_indices[frame_data['acceptor_water_idx'][h], 0]),
                H_idx=int(molecule_indices[frame_data['donor_water_idx'][h], 1 if frame_data['H_idx'][h] % 3 == 1 else 2]),
                donor_position=frame_data['donor_positions'][h].copy(),
                acceptor_position=frame_data['acceptor_positions'][h].copy(),
                H_position=frame_data['H_positions'][h].copy(),
                DA_distance=float(frame_data['DA_distances'][h]),
                HA_distance=float(frame_data['HA_distances'][h]),
                DHA_angle=float(frame_data['DHA_angles'][h]),
            )
            frame_hbonds.append(hb)

        all_hbonds.append(frame_hbonds)

    return all_hbonds
