#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <string>
#include <cmath>

namespace mlip {

/**
 * Compute density histogram for atoms along a specified axis.
 *
 * @param positions Flat array of positions [x0, y0, z0, x1, y1, z1, ...]
 * @param types     Array of atom type indices (0-indexed)
 * @param n_atoms   Number of atoms
 * @param box_lo    Box lower bounds [xlo, ylo, zlo]
 * @param box_hi    Box upper bounds [xhi, yhi, zhi]
 * @param axis      Axis for density profile: 0=x, 1=y, 2=z
 * @param n_bins    Number of bins
 * @param n_types   Number of unique atom types
 * @return          2D vector [type_idx][bin_idx] of counts
 */
std::vector<std::vector<double>> compute_density_histogram(
    const double* positions,
    const int* types,
    size_t n_atoms,
    const std::array<double, 3>& box_lo,
    const std::array<double, 3>& box_hi,
    int axis,
    int n_bins,
    int n_types
);

/**
 * Accumulate density counts over multiple frames.
 * Used for block averaging - processes a batch of frames at once.
 *
 * @param all_positions  Vector of position arrays (one per frame)
 * @param all_types      Vector of type arrays (one per frame)
 * @param atoms_per_frame Number of atoms in each frame
 * @param box_lo         Box lower bounds
 * @param box_hi         Box upper bounds
 * @param axis           Axis for density profile
 * @param n_bins         Number of bins
 * @param n_types        Number of unique atom types
 * @return               2D vector [type_idx][bin_idx] of accumulated counts
 */
std::vector<std::vector<double>> accumulate_density_frames(
    const std::vector<const double*>& all_positions,
    const std::vector<const int*>& all_types,
    const std::vector<size_t>& atoms_per_frame,
    const std::array<double, 3>& box_lo,
    const std::array<double, 3>& box_hi,
    int axis,
    int n_bins,
    int n_types
);

/**
 * Compute density histogram for each frame separately.
 * Used for principled error estimation (autocorrelation, Flyvbjerg-Petersen).
 *
 * @param all_positions  Vector of position arrays (one per frame)
 * @param all_types      Vector of type arrays (one per frame)
 * @param atoms_per_frame Number of atoms in each frame
 * @param box_lo         Box lower bounds
 * @param box_hi         Box upper bounds
 * @param axis           Axis for density profile
 * @param n_bins         Number of bins
 * @param n_types        Number of unique atom types
 * @return               3D vector [frame_idx][type_idx][bin_idx] of counts
 */
std::vector<std::vector<std::vector<double>>> compute_density_histogram_per_frame(
    const std::vector<const double*>& all_positions,
    const std::vector<const int*>& all_types,
    const std::vector<size_t>& atoms_per_frame,
    const std::array<double, 3>& box_lo,
    const std::array<double, 3>& box_hi,
    int axis,
    int n_bins,
    int n_types
);

/**
 * Compute RDF histogram for a single frame.
 * Returns histograms for all requested pair types.
 *
 * @param positions     Flat array of positions [x0, y0, z0, x1, y1, z1, ...]
 * @param types         Array of atom type indices (0-indexed)
 * @param n_atoms       Number of atoms
 * @param cell_vectors  Cell vectors as 3x3 matrix (row-major: a, b, c)
 * @param rmin          Minimum distance for histogram
 * @param rmax          Maximum distance for histogram
 * @param n_bins        Number of bins
 * @param type1         First atom type for pair (-1 for all types)
 * @param type2         Second atom type for pair (-1 for all types)
 * @return              Histogram counts for the specified pair type
 */
std::vector<double> compute_rdf_histogram(
    const double* positions,
    const int* types,
    size_t n_atoms,
    const std::array<std::array<double, 3>, 3>& cell_vectors,
    double rmin,
    double rmax,
    int n_bins,
    int type1,
    int type2
);

/**
 * Accumulate RDF histogram over multiple frames.
 *
 * @param all_positions   Vector of position arrays (one per frame)
 * @param all_types       Vector of type arrays (one per frame)
 * @param atoms_per_frame Number of atoms in each frame
 * @param cell_vectors    Cell vectors (assumed constant across frames)
 * @param rmin            Minimum distance
 * @param rmax            Maximum distance
 * @param n_bins          Number of bins
 * @param type1           First atom type (-1 for all)
 * @param type2           Second atom type (-1 for all)
 * @return                Accumulated histogram and pair count info
 */
struct RDFResult {
    std::vector<double> histogram;
    double total_pairs;
    double total_volume;
    int n_frames;
};

RDFResult accumulate_rdf_frames(
    const std::vector<const double*>& all_positions,
    const std::vector<const int*>& all_types,
    const std::vector<size_t>& atoms_per_frame,
    const std::array<std::array<double, 3>, 3>& cell_vectors,
    double rmin,
    double rmax,
    int n_bins,
    int type1,
    int type2
);

/**
 * Result structure for MSD calculation.
 */
struct MSDResult {
    std::vector<double> msd_planar;       // MSD in x-y plane
    std::vector<double> msd_perpendicular; // MSD in z direction
    std::vector<double> msd_total;         // Total 3D MSD
};

/**
 * Region definition for region-based MSD calculation.
 */
struct Region {
    std::string name;
    double z_min;
    double z_max;
};

/**
 * Result structure for region-based MSD calculation.
 * Contains MSD results for each region.
 */
struct RegionMSDResult {
    std::unordered_map<std::string, MSDResult> region_results;
};

/**
 * Unwrap trajectory coordinates accounting for periodic boundary conditions.
 *
 * @param all_positions   Vector of position arrays (one per frame), flat [x0,y0,z0,x1,y1,z1,...]
 * @param atoms_per_frame Number of atoms in each frame
 * @param box_lengths     Box dimensions [Lx, Ly, Lz]
 * @param unwrap_xy       Whether to unwrap x and y coordinates
 * @param unwrap_z        Whether to unwrap z coordinate
 * @return                Vector of unwrapped position arrays (one per frame)
 */
std::vector<std::vector<double>> unwrap_trajectory(
    const std::vector<const double*>& all_positions,
    const std::vector<size_t>& atoms_per_frame,
    const std::array<double, 3>& box_lengths,
    bool unwrap_xy,
    bool unwrap_z
);

/**
 * Compute MSD for all three components from unwrapped positions.
 *
 * Calculates planar (x-y), perpendicular (z), and total (3D) MSD
 * by averaging over all time origins.
 *
 * @param positions  Vector of unwrapped position arrays (one per frame)
 * @param n_atoms    Number of atoms
 * @return           MSDResult with all three MSD components
 */
MSDResult compute_msd_all(
    const std::vector<std::vector<double>>& positions,
    size_t n_atoms
);

/**
 * Compute MSD directly from wrapped positions.
 *
 * This function handles unwrapping and MSD calculation in one step.
 *
 * @param all_positions   Vector of position arrays (wrapped)
 * @param atoms_per_frame Number of atoms per frame
 * @param box_lengths     Box dimensions [Lx, Ly, Lz]
 * @param unwrap_xy       Whether to unwrap x and y coordinates
 * @param unwrap_z        Whether to unwrap z coordinate
 * @return                MSDResult with all three MSD components
 */
MSDResult compute_msd_from_positions(
    const std::vector<const double*>& all_positions,
    const std::vector<size_t>& atoms_per_frame,
    const std::array<double, 3>& box_lengths,
    bool unwrap_xy,
    bool unwrap_z
);

/**
 * Compute region-based MSD with endpoint checking.
 *
 * For each (t0, t0+dt) pair, only include atoms that are inside
 * the region at BOTH t0 AND t0+dt.
 *
 * @param positions       Vector of unwrapped position arrays (one per frame)
 * @param n_atoms         Number of atoms
 * @param regions         Vector of Region structs defining z-ranges
 * @return                RegionMSDResult with MSD for each region
 */
RegionMSDResult compute_msd_regions(
    const std::vector<std::vector<double>>& positions,
    size_t n_atoms,
    const std::vector<Region>& regions
);

/**
 * Compute region-based MSD directly from wrapped positions.
 *
 * This function handles unwrapping and region-based MSD calculation.
 *
 * @param all_positions   Vector of position arrays (wrapped)
 * @param atoms_per_frame Number of atoms per frame
 * @param box_lengths     Box dimensions [Lx, Ly, Lz]
 * @param regions         Vector of Region structs defining z-ranges
 * @param unwrap_xy       Whether to unwrap x and y coordinates
 * @param unwrap_z        Whether to unwrap z coordinate
 * @return                RegionMSDResult with MSD for each region
 */
RegionMSDResult compute_msd_regions_from_positions(
    const std::vector<const double*>& all_positions,
    const std::vector<size_t>& atoms_per_frame,
    const std::array<double, 3>& box_lengths,
    const std::vector<Region>& regions,
    bool unwrap_xy,
    bool unwrap_z
);

// ============================================================================
// Water Molecule Identification
// ============================================================================

/**
 * Structure representing a single water molecule with geometric properties.
 */
struct WaterMolecule {
    int O_index;                        // Index of oxygen atom
    int H1_index;                       // Index of first hydrogen atom
    int H2_index;                       // Index of second hydrogen atom
    std::array<double, 3> O_position;   // Position of oxygen [x, y, z]
    std::array<double, 3> H1_position;  // Position of first hydrogen
    std::array<double, 3> H2_position;  // Position of second hydrogen
    double OH1_distance;                // O-H1 bond distance (Angstroms)
    double OH2_distance;                // O-H2 bond distance (Angstroms)
    double H1H2_distance;               // H1-H2 distance (Angstroms)
    double H1OH2_angle;                 // H1-O-H2 bond angle (degrees)
};

/**
 * Result structure for water molecule identification.
 */
struct WaterIdentificationResult {
    std::vector<std::array<int, 3>> molecule_indices;  // [O, H1, H2] for each molecule
    size_t n_molecules;                                 // Number of molecules found
};

/**
 * Identify water molecules from atomic positions and element types.
 *
 * Uses greedy nearest-neighbor assignment: for each oxygen atom,
 * finds the two closest hydrogen atoms within the cutoff distance.
 * Supports periodic boundary conditions via minimum image convention.
 *
 * @param positions     Flat array of positions [x0, y0, z0, x1, y1, z1, ...]
 * @param element_types Array of element type indices (0-indexed)
 * @param n_atoms       Number of atoms
 * @param o_type_idx    Type index for oxygen atoms
 * @param h_type_idx    Type index for hydrogen atoms
 * @param box_lengths   Box dimensions [Lx, Ly, Lz] for PBC (zero = no PBC)
 * @param o_h_cutoff    O-H bond distance cutoff (default: 1.2 Angstroms)
 * @return              WaterIdentificationResult with molecule indices
 */
WaterIdentificationResult identify_water_molecules(
    const double* positions,
    const int* element_types,
    size_t n_atoms,
    int o_type_idx,
    int h_type_idx,
    const std::array<double, 3>& box_lengths,
    double o_h_cutoff = 1.2
);

/**
 * Extract geometric properties for water molecules in a single frame.
 *
 * @param positions        Flat array of positions for this frame
 * @param molecule_indices Vector of [O, H1, H2] index tuples
 * @param n_molecules      Number of water molecules
 * @param box_lengths      Box dimensions [Lx, Ly, Lz] for PBC (zero = no PBC)
 * @return                 Vector of WaterMolecule with computed properties
 */
std::vector<WaterMolecule> extract_water_properties(
    const double* positions,
    const std::vector<std::array<int, 3>>& molecule_indices,
    size_t n_molecules,
    const std::array<double, 3>& box_lengths
);

/**
 * Extract geometric properties for water molecules across multiple frames.
 *
 * Uses OpenMP parallelization when available.
 *
 * @param all_positions    Vector of position arrays (one per frame)
 * @param molecule_indices Vector of [O, H1, H2] index tuples
 * @param n_molecules      Number of water molecules
 * @param n_frames         Number of frames
 * @param box_lengths      Box dimensions [Lx, Ly, Lz] for PBC (zero = no PBC)
 * @return                 Vector of vectors: [frame_idx][molecule_idx]
 */
std::vector<std::vector<WaterMolecule>> extract_water_properties_multiframe(
    const std::vector<const double*>& all_positions,
    const std::vector<std::array<int, 3>>& molecule_indices,
    size_t n_molecules,
    size_t n_frames,
    const std::array<double, 3>& box_lengths
);

// ============================================================================
// Hydrogen Bond Identification
// ============================================================================

/**
 * Structure representing a single hydrogen bond.
 *
 * A hydrogen bond D-H...A consists of:
 * - D (Donor): The oxygen atom covalently bonded to H
 * - H: The hydrogen atom forming the bridge
 * - A (Acceptor): The oxygen atom accepting the hydrogen bond
 */
struct HydrogenBond {
    int donor_water_idx;                    // Index of donor water molecule
    int acceptor_water_idx;                 // Index of acceptor water molecule
    int donor_O_idx;                        // Atom index of donor oxygen
    int acceptor_O_idx;                     // Atom index of acceptor oxygen
    int H_idx;                              // Atom index of bridging hydrogen
    std::array<double, 3> donor_position;   // Position of donor oxygen
    std::array<double, 3> acceptor_position;// Position of acceptor oxygen
    std::array<double, 3> H_position;       // Position of bridging hydrogen
    double DA_distance;                     // Donor-Acceptor distance (Angstroms)
    double HA_distance;                     // Hydrogen-Acceptor distance (Angstroms)
    double DHA_angle;                       // D-H-A angle (degrees)
};

/**
 * Identify hydrogen bonds in a single frame.
 *
 * Searches for H-bonds between water molecules using geometric criteria:
 * - D-A distance < da_cutoff
 * - D-H-A angle > angle_cutoff
 *
 * @param positions      Flat array of positions [x0, y0, z0, x1, y1, z1, ...]
 * @param water_indices  Vector of [O, H1, H2] indices for each water molecule
 * @param n_waters       Number of water molecules
 * @param box_lengths    Box dimensions [Lx, Ly, Lz] for PBC
 * @param da_cutoff      Donor-Acceptor distance cutoff (default: 3.5 Angstroms)
 * @param angle_cutoff   Minimum D-H-A angle (default: 150 degrees)
 * @return               Vector of identified hydrogen bonds
 */
std::vector<HydrogenBond> identify_hbonds_frame(
    const double* positions,
    const std::vector<std::array<int, 3>>& water_indices,
    size_t n_waters,
    const std::array<double, 3>& box_lengths,
    double da_cutoff = 3.5,
    double angle_cutoff = 150.0
);

/**
 * Identify hydrogen bonds across multiple frames.
 *
 * Uses OpenMP parallelization when available.
 *
 * @param all_positions    Vector of position arrays (one per frame)
 * @param water_indices    Vector of [O, H1, H2] indices for each water molecule
 * @param n_waters         Number of water molecules
 * @param n_frames         Number of frames
 * @param all_box_lengths  Box dimensions per frame for PBC
 * @param da_cutoff        Donor-Acceptor distance cutoff (default: 3.5 Angstroms)
 * @param angle_cutoff     Minimum D-H-A angle (default: 150 degrees)
 * @return                 Vector of vectors: [frame_idx][hbond_idx]
 */
std::vector<std::vector<HydrogenBond>> identify_hbonds_multiframe(
    const std::vector<const double*>& all_positions,
    const std::vector<std::array<int, 3>>& water_indices,
    size_t n_waters,
    size_t n_frames,
    const std::vector<std::array<double, 3>>& all_box_lengths,
    double da_cutoff = 3.5,
    double angle_cutoff = 150.0
);

// ============================================================================
// Water Molecule Orientation Analysis
// ============================================================================

/**
 * Structure storing orientation data for a single water molecule.
 *
 * Contains cosine values of angles between molecular vectors and the
 * surface normal, used for analyzing water orientation at interfaces.
 */
struct WaterOrientation {
    int water_idx;           // Index of water molecule
    double cos_theta_1;      // rOH1 · n (O→H1 dotted with surface normal)
    double cos_theta_2;      // rOH2 · n (O→H2 dotted with surface normal)
    double cos_phi;          // (H_mid → O) · n (dipole dotted with surface normal)
    double O_z;              // Oxygen z-coordinate for region assignment
};

/**
 * Compute orientation for all water molecules in a single frame.
 *
 * Calculates cos(theta) for O-H bonds and cos(phi) for dipole orientation
 * relative to the surface normal vector.
 *
 * @param positions      Flat array of positions [x0, y0, z0, x1, y1, z1, ...]
 * @param water_indices  Vector of [O, H1, H2] indices for each water molecule
 * @param n_waters       Number of water molecules
 * @param box_lengths    Box dimensions [Lx, Ly, Lz] for PBC
 * @param surface_normal Surface normal vector (default: +z direction)
 * @return               Vector of WaterOrientation for each molecule
 */
std::vector<WaterOrientation> compute_orientations_frame(
    const double* positions,
    const std::vector<std::array<int, 3>>& water_indices,
    size_t n_waters,
    const std::array<double, 3>& box_lengths,
    const std::array<double, 3>& surface_normal = {0.0, 0.0, 1.0}
);

/**
 * Compute orientations across multiple frames.
 *
 * Uses OpenMP parallelization when available.
 *
 * @param all_positions    Vector of position arrays (one per frame)
 * @param water_indices    Vector of [O, H1, H2] indices for each water molecule
 * @param n_waters         Number of water molecules
 * @param n_frames         Number of frames
 * @param all_box_lengths  Box dimensions per frame for PBC
 * @param surface_normal   Surface normal vector (default: +z direction)
 * @return                 Vector of vectors: [frame_idx][water_idx]
 */
std::vector<std::vector<WaterOrientation>> compute_orientations_multiframe(
    const std::vector<const double*>& all_positions,
    const std::vector<std::array<int, 3>>& water_indices,
    size_t n_waters,
    size_t n_frames,
    const std::vector<std::array<double, 3>>& all_box_lengths,
    const std::array<double, 3>& surface_normal = {0.0, 0.0, 1.0}
);

} // namespace mlip
