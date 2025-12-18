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

} // namespace mlip
