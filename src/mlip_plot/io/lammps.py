"""
LAMMPS trajectory file reader.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


def _log(logger: Optional[Any], method: str, message: str, verbose: bool = True):
    """Helper to log or print based on logger availability."""
    if not verbose:
        return
    if logger is not None:
        getattr(logger, method)(message)
    else:
        print(message)


def read_lammpstrj(
    filename: str,
    skip_frames: int = 0,
    max_frames: Optional[int] = None,
    verbose: bool = True,
    logger: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Read LAMMPS trajectory file (.lammpstrj).

    Parameters
    ----------
    filename : str
        Path to .lammpstrj file
    skip_frames : int
        Number of initial frames to skip (for equilibration)
    max_frames : int, optional
        Maximum number of frames to read after skipping
    verbose : bool
        Print progress information

    Returns
    -------
    frames : list of dict
        List of frame dictionaries containing:
        - timestep: int
        - box: dict with xlo, xhi, ylo, yhi, zlo, zhi
        - natoms: int
        - positions: ndarray of shape (N, 3)
        - types: ndarray of shape (N,) with type indices
        - elements: ndarray of shape (N,) with element strings
        - atom_ids: ndarray of shape (N,)
    """
    if verbose:
        _log(logger, 'info', f"Reading trajectory: {filename}")

    filepath = Path(filename)
    if not filepath.exists():
        raise FileNotFoundError(f"Trajectory file not found: {filename}")

    frames = []
    frame_count = 0
    frames_read = 0

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    n_lines = len(lines)

    while i < n_lines:
        if 'ITEM: TIMESTEP' in lines[i]:
            frame_count += 1

            # Check if we should skip this frame
            if frame_count <= skip_frames:
                # Skip to next frame
                i += 1
                timestep = int(lines[i].strip())
                i += 2  # ITEM: NUMBER OF ATOMS
                natoms = int(lines[i].strip())
                i += 5 + natoms  # Skip box bounds + header + atoms
                continue

            # Check if we've read enough frames
            if max_frames is not None and frames_read >= max_frames:
                break

            # Read timestep
            i += 1
            timestep = int(lines[i].strip())

            # Read number of atoms
            i += 2
            natoms = int(lines[i].strip())

            # Read box bounds
            i += 2
            xlo, xhi = map(float, lines[i].split()[:2])
            ylo, yhi = map(float, lines[i + 1].split()[:2])
            zlo, zhi = map(float, lines[i + 2].split()[:2])

            # Parse header to find column indices
            i += 4
            header_line = lines[i - 1].strip()
            # Expected format: ITEM: ATOMS id type element x y z ...
            columns = header_line.replace('ITEM: ATOMS', '').split()

            # Find indices for required columns
            try:
                id_idx = columns.index('id')
                type_idx = columns.index('type')
                x_idx = columns.index('x')
                y_idx = columns.index('y')
                z_idx = columns.index('z')
            except ValueError:
                # Try alternative column names
                try:
                    id_idx = columns.index('id')
                    type_idx = columns.index('type')
                    x_idx = columns.index('xu') if 'xu' in columns else columns.index('xs')
                    y_idx = columns.index('yu') if 'yu' in columns else columns.index('ys')
                    z_idx = columns.index('zu') if 'zu' in columns else columns.index('zs')
                except ValueError as e:
                    raise ValueError(f"Could not find required columns in header: {columns}") from e

            # Check for element column
            has_element = 'element' in columns
            elem_idx = columns.index('element') if has_element else None

            # Read atom data
            positions = np.zeros((natoms, 3), dtype=np.float64)
            types = np.zeros(natoms, dtype=np.int32)
            atom_ids = np.zeros(natoms, dtype=np.int32)
            elements = []

            for j in range(natoms):
                parts = lines[i].strip().split()
                atom_ids[j] = int(parts[id_idx])
                types[j] = int(parts[type_idx])
                positions[j, 0] = float(parts[x_idx])
                positions[j, 1] = float(parts[y_idx])
                positions[j, 2] = float(parts[z_idx])

                if has_element:
                    elements.append(parts[elem_idx])

                i += 1

            frames.append({
                'timestep': timestep,
                'box': {
                    'xlo': xlo, 'xhi': xhi,
                    'ylo': ylo, 'yhi': yhi,
                    'zlo': zlo, 'zhi': zhi
                },
                'natoms': natoms,
                'positions': positions,
                'types': types,
                'atom_ids': atom_ids,
                'elements': np.array(elements) if elements else None
            })
            frames_read += 1
        else:
            i += 1

    if verbose:
        _log(logger, 'detail', f"Total frames in file: {frame_count}")
        _log(logger, 'detail', f"Frames skipped: {skip_frames}")
        _log(logger, 'success', f"Frames read: {len(frames)}")

    return frames


def get_element_type_mapping(frames: List[Dict]) -> Dict[str, int]:
    """
    Extract element-to-type mapping from frames.

    Parameters
    ----------
    frames : list
        Frames from read_lammpstrj

    Returns
    -------
    mapping : dict
        Dictionary mapping element name to type index (0-indexed)
    """
    if not frames or frames[0]['elements'] is None:
        raise ValueError("Frames do not contain element information")

    # Get unique element-type pairs from first frame
    elements = frames[0]['elements']
    types = frames[0]['types']

    mapping = {}
    for elem, typ in zip(elements, types):
        if elem not in mapping:
            mapping[elem] = typ - 1  # Convert to 0-indexed

    return mapping


def get_type_element_mapping(frames: List[Dict]) -> Dict[int, str]:
    """
    Extract type-to-element mapping from frames.

    Parameters
    ----------
    frames : list
        Frames from read_lammpstrj

    Returns
    -------
    mapping : dict
        Dictionary mapping type index (0-indexed) to element name
    """
    elem_to_type = get_element_type_mapping(frames)
    return {v: k for k, v in elem_to_type.items()}
