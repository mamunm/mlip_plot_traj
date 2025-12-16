/**
 * Hydrogen bond detection using geometric criteria.
 *
 * H-bond: D-H...A where
 *   - D-A distance < d_a_cutoff (typically 3.5 Å)
 *   - D-H...A angle > angle_cutoff (typically 120°)
 */

#include "../include/mlip_core.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace mlip {

namespace {

// Apply minimum image convention
inline void apply_pbc(double& dx, double& dy, double& dz,
                      const std::array<std::array<double, 3>, 3>& cell) {
    // For orthorhombic cells, simple wrapping
    double Lx = cell[0][0];
    double Ly = cell[1][1];
    double Lz = cell[2][2];

    dx -= Lx * std::round(dx / Lx);
    dy -= Ly * std::round(dy / Ly);
    dz -= Lz * std::round(dz / Lz);
}

// Compute distance with PBC
inline double distance_pbc(const double* pos1, const double* pos2,
                           const std::array<std::array<double, 3>, 3>& cell) {
    double dx = pos1[0] - pos2[0];
    double dy = pos1[1] - pos2[1];
    double dz = pos1[2] - pos2[2];
    apply_pbc(dx, dy, dz, cell);
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// Compute vector from pos1 to pos2 with PBC
inline void vector_pbc(const double* pos1, const double* pos2,
                       const std::array<std::array<double, 3>, 3>& cell,
                       double& dx, double& dy, double& dz) {
    dx = pos2[0] - pos1[0];
    dy = pos2[1] - pos1[1];
    dz = pos2[2] - pos1[2];
    apply_pbc(dx, dy, dz, cell);
}

// Compute angle D-H...A (angle at H) in degrees
inline double compute_angle(const double* pos_d, const double* pos_h, const double* pos_a,
                            const std::array<std::array<double, 3>, 3>& cell) {
    // Vector H->D
    double hd_x, hd_y, hd_z;
    vector_pbc(pos_h, pos_d, cell, hd_x, hd_y, hd_z);

    // Vector H->A
    double ha_x, ha_y, ha_z;
    vector_pbc(pos_h, pos_a, cell, ha_x, ha_y, ha_z);

    // Dot product
    double dot = hd_x * ha_x + hd_y * ha_y + hd_z * ha_z;

    // Magnitudes
    double mag_hd = std::sqrt(hd_x*hd_x + hd_y*hd_y + hd_z*hd_z);
    double mag_ha = std::sqrt(ha_x*ha_x + ha_y*ha_y + ha_z*ha_z);

    if (mag_hd < 1e-10 || mag_ha < 1e-10) {
        return 0.0;
    }

    double cos_angle = dot / (mag_hd * mag_ha);
    // Clamp to [-1, 1] to avoid numerical issues
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));

    // Convert to degrees
    return std::acos(cos_angle) * 180.0 / M_PI;
}

// Find the donor atom bonded to a hydrogen
// Returns -1 if no donor found within d_h_cutoff
inline int find_bonded_donor(size_t h_idx, const double* positions, const int* types,
                             size_t n_atoms, int donor_type, double d_h_cutoff,
                             const std::array<std::array<double, 3>, 3>& cell) {
    const double* h_pos = &positions[h_idx * 3];
    double min_dist = d_h_cutoff;
    int bonded_donor = -1;

    for (size_t i = 0; i < n_atoms; ++i) {
        if (types[i] != donor_type) continue;
        if (i == h_idx) continue;

        const double* d_pos = &positions[i * 3];
        double dist = distance_pbc(h_pos, d_pos, cell);

        if (dist < min_dist) {
            min_dist = dist;
            bonded_donor = static_cast<int>(i);
        }
    }

    return bonded_donor;
}

} // anonymous namespace

HBondResult detect_hbonds_frame(
    const double* positions,
    const int* types,
    size_t n_atoms,
    const std::array<std::array<double, 3>, 3>& cell_vectors,
    int donor_type,
    int hydrogen_type,
    int acceptor_type,
    double d_a_cutoff,
    double angle_cutoff,
    double d_h_cutoff,
    int bin_axis,
    const std::array<double, 3>& box_lo,
    const std::array<double, 3>& box_hi,
    int n_bins
) {
    HBondResult result;
    result.total_hbonds = 0;

    bool do_binning = (bin_axis >= 0 && bin_axis <= 2 && n_bins > 0);
    if (do_binning) {
        result.hbonds_per_bin.resize(n_bins, 0.0);
        result.donors_per_bin.resize(n_bins, 0.0);
    }

    double bin_width = 0.0;
    double axis_lo = 0.0;
    if (do_binning) {
        axis_lo = box_lo[bin_axis];
        double axis_hi = box_hi[bin_axis];
        bin_width = (axis_hi - axis_lo) / n_bins;
    }

    // Collect indices for each type
    std::vector<size_t> hydrogen_indices;
    std::vector<size_t> acceptor_indices;

    for (size_t i = 0; i < n_atoms; ++i) {
        if (types[i] == hydrogen_type) {
            hydrogen_indices.push_back(i);
        }
        if (types[i] == acceptor_type) {
            acceptor_indices.push_back(i);
        }
    }

    // Count donors per bin (for normalization)
    if (do_binning) {
        for (size_t i = 0; i < n_atoms; ++i) {
            if (types[i] == donor_type) {
                double pos_axis = positions[i * 3 + bin_axis];
                int bin_idx = static_cast<int>((pos_axis - axis_lo) / bin_width);
                if (bin_idx >= 0 && bin_idx < n_bins) {
                    result.donors_per_bin[bin_idx] += 1.0;
                }
            }
        }
    }

    // For each hydrogen, find its bonded donor
    for (size_t h_idx : hydrogen_indices) {
        int donor_idx = find_bonded_donor(h_idx, positions, types, n_atoms,
                                          donor_type, d_h_cutoff, cell_vectors);
        if (donor_idx < 0) continue;

        const double* pos_d = &positions[donor_idx * 3];
        const double* pos_h = &positions[h_idx * 3];

        // Find acceptors within cutoff of donor
        for (size_t a_idx : acceptor_indices) {
            // Skip if acceptor is the same as donor
            if (static_cast<size_t>(donor_idx) == a_idx) continue;

            const double* pos_a = &positions[a_idx * 3];

            // Check D-A distance
            double d_a_dist = distance_pbc(pos_d, pos_a, cell_vectors);
            if (d_a_dist > d_a_cutoff) continue;

            // Check D-H...A angle
            double angle = compute_angle(pos_d, pos_h, pos_a, cell_vectors);
            if (angle < angle_cutoff) continue;

            // H-bond found!
            result.total_hbonds++;

            // Bin by donor position
            if (do_binning) {
                double donor_pos_axis = pos_d[bin_axis];
                int bin_idx = static_cast<int>((donor_pos_axis - axis_lo) / bin_width);
                if (bin_idx >= 0 && bin_idx < n_bins) {
                    result.hbonds_per_bin[bin_idx] += 1.0;
                }
            }
        }
    }

    return result;
}

std::vector<HBondResult> accumulate_hbonds_frames(
    const std::vector<const double*>& all_positions,
    const std::vector<const int*>& all_types,
    const std::vector<size_t>& atoms_per_frame,
    const std::array<std::array<double, 3>, 3>& cell_vectors,
    int donor_type,
    int hydrogen_type,
    int acceptor_type,
    double d_a_cutoff,
    double angle_cutoff,
    double d_h_cutoff,
    int bin_axis,
    const std::array<double, 3>& box_lo,
    const std::array<double, 3>& box_hi,
    int n_bins
) {
    std::vector<HBondResult> results;
    results.reserve(all_positions.size());

    for (size_t frame_idx = 0; frame_idx < all_positions.size(); ++frame_idx) {
        HBondResult frame_result = detect_hbonds_frame(
            all_positions[frame_idx],
            all_types[frame_idx],
            atoms_per_frame[frame_idx],
            cell_vectors,
            donor_type,
            hydrogen_type,
            acceptor_type,
            d_a_cutoff,
            angle_cutoff,
            d_h_cutoff,
            bin_axis,
            box_lo,
            box_hi,
            n_bins
        );
        results.push_back(std::move(frame_result));
    }

    return results;
}

std::vector<std::pair<int, int>> detect_hbond_pairs(
    const double* positions,
    const int* types,
    size_t n_atoms,
    const std::array<std::array<double, 3>, 3>& cell_vectors,
    int donor_type,
    int hydrogen_type,
    int acceptor_type,
    double d_a_cutoff,
    double angle_cutoff,
    double d_h_cutoff
) {
    std::vector<std::pair<int, int>> pairs;

    // Collect indices for each type
    std::vector<size_t> hydrogen_indices;
    std::vector<size_t> acceptor_indices;

    for (size_t i = 0; i < n_atoms; ++i) {
        if (types[i] == hydrogen_type) {
            hydrogen_indices.push_back(i);
        }
        if (types[i] == acceptor_type) {
            acceptor_indices.push_back(i);
        }
    }

    // For each hydrogen, find its bonded donor and check acceptors
    for (size_t h_idx : hydrogen_indices) {
        int donor_idx = find_bonded_donor(h_idx, positions, types, n_atoms,
                                          donor_type, d_h_cutoff, cell_vectors);
        if (donor_idx < 0) continue;

        const double* pos_d = &positions[donor_idx * 3];
        const double* pos_h = &positions[h_idx * 3];

        for (size_t a_idx : acceptor_indices) {
            if (static_cast<size_t>(donor_idx) == a_idx) continue;

            const double* pos_a = &positions[a_idx * 3];

            // Check D-A distance
            double d_a_dist = distance_pbc(pos_d, pos_a, cell_vectors);
            if (d_a_dist > d_a_cutoff) continue;

            // Check D-H...A angle
            double angle = compute_angle(pos_d, pos_h, pos_a, cell_vectors);
            if (angle < angle_cutoff) continue;

            // H-bond found
            pairs.emplace_back(donor_idx, static_cast<int>(a_idx));
        }
    }

    return pairs;
}

} // namespace mlip
