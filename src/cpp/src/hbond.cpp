/**
 * Hydrogen bond identification between water molecules.
 *
 * Identifies H-bonds using geometric criteria:
 * - Donor-Acceptor distance < cutoff (default 3.5 Å)
 * - D-H-A angle > cutoff (default 150°)
 *
 * Supports periodic boundary conditions via minimum image convention.
 */

#include "mlip_core.hpp"
#include <cmath>
#include <algorithm>
#include <vector>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace mlip {

namespace {

/**
 * Apply minimum image convention for periodic boundary conditions.
 */
inline void apply_minimum_image(
    double& dx, double& dy, double& dz,
    const std::array<double, 3>& box_lengths
) {
    if (box_lengths[0] > 0) {
        dx = dx - std::round(dx / box_lengths[0]) * box_lengths[0];
    }
    if (box_lengths[1] > 0) {
        dy = dy - std::round(dy / box_lengths[1]) * box_lengths[1];
    }
    if (box_lengths[2] > 0) {
        dz = dz - std::round(dz / box_lengths[2]) * box_lengths[2];
    }
}

/**
 * Calculate distance between two 3D points with PBC support.
 */
inline double distance_pbc(
    const double* pos1,
    const double* pos2,
    const std::array<double, 3>& box_lengths
) {
    double dx = pos1[0] - pos2[0];
    double dy = pos1[1] - pos2[1];
    double dz = pos1[2] - pos2[2];

    apply_minimum_image(dx, dy, dz, box_lengths);

    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Calculate the displacement vector from pos1 to pos2 with PBC.
 */
inline void displacement_pbc(
    const double* pos1,
    const double* pos2,
    const std::array<double, 3>& box_lengths,
    double* disp
) {
    disp[0] = pos2[0] - pos1[0];
    disp[1] = pos2[1] - pos1[1];
    disp[2] = pos2[2] - pos1[2];

    apply_minimum_image(disp[0], disp[1], disp[2], box_lengths);
}

/**
 * Calculate angle at vertex B for triangle A-B-C with PBC support.
 * Returns angle in degrees.
 */
inline double angle_abc_pbc(
    const double* A,
    const double* B,
    const double* C,
    const std::array<double, 3>& box_lengths
) {
    // Vectors BA and BC with PBC
    double ba[3], bc[3];
    displacement_pbc(B, A, box_lengths, ba);
    displacement_pbc(B, C, box_lengths, bc);

    double dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
    double mag_ba = std::sqrt(ba[0] * ba[0] + ba[1] * ba[1] + ba[2] * ba[2]);
    double mag_bc = std::sqrt(bc[0] * bc[0] + bc[1] * bc[1] + bc[2] * bc[2]);

    if (mag_ba < 1e-10 || mag_bc < 1e-10) {
        return 0.0;
    }

    double cos_angle = dot / (mag_ba * mag_bc);
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));

    return std::acos(cos_angle) * 180.0 / M_PI;
}

}  // anonymous namespace

std::vector<HydrogenBond> identify_hbonds_frame(
    const double* positions,
    const std::vector<std::array<int, 3>>& water_indices,
    size_t n_waters,
    const std::array<double, 3>& box_lengths,
    double da_cutoff,
    double angle_cutoff
) {
    std::vector<HydrogenBond> hbonds;

    if (n_waters < 2) {
        return hbonds;
    }

    // Pre-square the cutoff for faster comparison
    const double da_cutoff_sq = da_cutoff * da_cutoff;

    // For each water molecule as potential donor
    for (size_t donor_idx = 0; donor_idx < n_waters; ++donor_idx) {
        const auto& donor = water_indices[donor_idx];
        const int donor_O_idx = donor[0];
        const int donor_H1_idx = donor[1];
        const int donor_H2_idx = donor[2];

        const double* donor_O_pos = positions + donor_O_idx * 3;
        const double* donor_H1_pos = positions + donor_H1_idx * 3;
        const double* donor_H2_pos = positions + donor_H2_idx * 3;

        // For each other water molecule as potential acceptor
        for (size_t acceptor_idx = 0; acceptor_idx < n_waters; ++acceptor_idx) {
            if (acceptor_idx == donor_idx) continue;

            const auto& acceptor = water_indices[acceptor_idx];
            const int acceptor_O_idx = acceptor[0];
            const double* acceptor_O_pos = positions + acceptor_O_idx * 3;

            // Quick distance check (D-A)
            double dx = donor_O_pos[0] - acceptor_O_pos[0];
            double dy = donor_O_pos[1] - acceptor_O_pos[1];
            double dz = donor_O_pos[2] - acceptor_O_pos[2];
            apply_minimum_image(dx, dy, dz, box_lengths);
            double da_dist_sq = dx * dx + dy * dy + dz * dz;

            if (da_dist_sq >= da_cutoff_sq) continue;

            double da_dist = std::sqrt(da_dist_sq);

            // Check H1 for potential H-bond
            // D-H-A angle: donor_O is D, H1 is H, acceptor_O is A
            double angle_h1 = angle_abc_pbc(donor_O_pos, donor_H1_pos, acceptor_O_pos, box_lengths);

            if (angle_h1 > angle_cutoff) {
                HydrogenBond hb;
                hb.donor_water_idx = static_cast<int>(donor_idx);
                hb.acceptor_water_idx = static_cast<int>(acceptor_idx);
                hb.donor_O_idx = donor_O_idx;
                hb.acceptor_O_idx = acceptor_O_idx;
                hb.H_idx = donor_H1_idx;

                for (int j = 0; j < 3; ++j) {
                    hb.donor_position[j] = donor_O_pos[j];
                    hb.acceptor_position[j] = acceptor_O_pos[j];
                    hb.H_position[j] = donor_H1_pos[j];
                }

                hb.DA_distance = da_dist;
                hb.HA_distance = distance_pbc(donor_H1_pos, acceptor_O_pos, box_lengths);
                hb.DHA_angle = angle_h1;

                hbonds.push_back(hb);
            }

            // Check H2 for potential H-bond
            double angle_h2 = angle_abc_pbc(donor_O_pos, donor_H2_pos, acceptor_O_pos, box_lengths);

            if (angle_h2 > angle_cutoff) {
                HydrogenBond hb;
                hb.donor_water_idx = static_cast<int>(donor_idx);
                hb.acceptor_water_idx = static_cast<int>(acceptor_idx);
                hb.donor_O_idx = donor_O_idx;
                hb.acceptor_O_idx = acceptor_O_idx;
                hb.H_idx = donor_H2_idx;

                for (int j = 0; j < 3; ++j) {
                    hb.donor_position[j] = donor_O_pos[j];
                    hb.acceptor_position[j] = acceptor_O_pos[j];
                    hb.H_position[j] = donor_H2_pos[j];
                }

                hb.DA_distance = da_dist;
                hb.HA_distance = distance_pbc(donor_H2_pos, acceptor_O_pos, box_lengths);
                hb.DHA_angle = angle_h2;

                hbonds.push_back(hb);
            }
        }
    }

    return hbonds;
}

std::vector<std::vector<HydrogenBond>> identify_hbonds_multiframe(
    const std::vector<const double*>& all_positions,
    const std::vector<std::array<int, 3>>& water_indices,
    size_t n_waters,
    size_t n_frames,
    const std::vector<std::array<double, 3>>& all_box_lengths,
    double da_cutoff,
    double angle_cutoff
) {
    std::vector<std::vector<HydrogenBond>> all_hbonds(n_frames);

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t f = 0; f < n_frames; ++f) {
        all_hbonds[f] = identify_hbonds_frame(
            all_positions[f],
            water_indices,
            n_waters,
            all_box_lengths[f],
            da_cutoff,
            angle_cutoff
        );
    }

    return all_hbonds;
}

}  // namespace mlip
