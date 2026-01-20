/**
 * Water molecule orientation analysis.
 *
 * Computes orientation angles (theta and phi) for water molecules
 * relative to a surface normal vector.
 *
 * - cos_theta_1: O→H1 vector dotted with surface normal
 * - cos_theta_2: O→H2 vector dotted with surface normal
 * - cos_phi: dipole (H_midpoint→O) dotted with surface normal
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
 *
 * @param no_z_pbc If true, disable PBC in z direction (for slab geometries)
 */
inline void apply_minimum_image(
    double& dx, double& dy, double& dz,
    const std::array<double, 3>& box_lengths,
    bool no_z_pbc = false
) {
    if (box_lengths[0] > 0) {
        dx = dx - std::round(dx / box_lengths[0]) * box_lengths[0];
    }
    if (box_lengths[1] > 0) {
        dy = dy - std::round(dy / box_lengths[1]) * box_lengths[1];
    }
    if (box_lengths[2] > 0 && !no_z_pbc) {
        dz = dz - std::round(dz / box_lengths[2]) * box_lengths[2];
    }
}

/**
 * Calculate the displacement vector from pos1 to pos2 with PBC.
 * Result is stored in disp array: disp = pos2 - pos1 (with PBC correction).
 *
 * @param no_z_pbc If true, disable PBC in z direction (for slab geometries)
 */
inline void displacement_pbc(
    const double* pos1,
    const double* pos2,
    const std::array<double, 3>& box_lengths,
    double* disp,
    bool no_z_pbc = false
) {
    disp[0] = pos2[0] - pos1[0];
    disp[1] = pos2[1] - pos1[1];
    disp[2] = pos2[2] - pos1[2];

    apply_minimum_image(disp[0], disp[1], disp[2], box_lengths, no_z_pbc);
}

/**
 * Compute magnitude of a 3D vector.
 */
inline double magnitude(const double* vec) {
    return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

/**
 * Compute dot product of two 3D vectors.
 */
inline double dot_product(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Compute cosine of angle between vector and surface normal.
 * Returns cos(angle) = (vec · normal) / |vec|
 */
inline double cos_angle_with_normal(
    const double* vec,
    const std::array<double, 3>& surface_normal
) {
    double mag = magnitude(vec);
    if (mag < 1e-10) {
        return 0.0;  // Degenerate case
    }

    double dot = vec[0] * surface_normal[0] +
                 vec[1] * surface_normal[1] +
                 vec[2] * surface_normal[2];

    double cos_val = dot / mag;

    // Clamp to [-1, 1] to handle numerical errors
    return std::max(-1.0, std::min(1.0, cos_val));
}

}  // anonymous namespace

std::vector<WaterOrientation> compute_orientations_frame(
    const double* positions,
    const std::vector<std::array<int, 3>>& water_indices,
    size_t n_waters,
    const std::array<double, 3>& box_lengths,
    const std::array<double, 3>& surface_normal,
    bool no_z_pbc
) {
    std::vector<WaterOrientation> orientations;
    orientations.reserve(n_waters);

    for (size_t water_idx = 0; water_idx < n_waters; ++water_idx) {
        const auto& water = water_indices[water_idx];
        const int O_idx = water[0];
        const int H1_idx = water[1];
        const int H2_idx = water[2];

        const double* O_pos = positions + O_idx * 3;
        const double* H1_pos = positions + H1_idx * 3;
        const double* H2_pos = positions + H2_idx * 3;

        // Compute O→H1 and O→H2 vectors with PBC (optionally disabled in z)
        double oh1[3], oh2[3];
        displacement_pbc(O_pos, H1_pos, box_lengths, oh1, no_z_pbc);  // O → H1
        displacement_pbc(O_pos, H2_pos, box_lengths, oh2, no_z_pbc);  // O → H2

        // H midpoint relative to O
        double h_mid[3] = {
            (oh1[0] + oh2[0]) / 2.0,
            (oh1[1] + oh2[1]) / 2.0,
            (oh1[2] + oh2[2]) / 2.0
        };

        // Dipole: from H_mid toward O (negative of h_mid)
        double dipole[3] = {-h_mid[0], -h_mid[1], -h_mid[2]};

        // Compute orientations
        WaterOrientation orient;
        orient.water_idx = static_cast<int>(water_idx);
        orient.cos_theta_1 = cos_angle_with_normal(oh1, surface_normal);
        orient.cos_theta_2 = cos_angle_with_normal(oh2, surface_normal);
        orient.cos_phi = cos_angle_with_normal(dipole, surface_normal);
        orient.O_z = O_pos[2];

        orientations.push_back(orient);
    }

    return orientations;
}

std::vector<std::vector<WaterOrientation>> compute_orientations_multiframe(
    const std::vector<const double*>& all_positions,
    const std::vector<std::array<int, 3>>& water_indices,
    size_t n_waters,
    size_t n_frames,
    const std::vector<std::array<double, 3>>& all_box_lengths,
    const std::array<double, 3>& surface_normal,
    bool no_z_pbc
) {
    std::vector<std::vector<WaterOrientation>> all_orientations(n_frames);

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t f = 0; f < n_frames; ++f) {
        all_orientations[f] = compute_orientations_frame(
            all_positions[f],
            water_indices,
            n_waters,
            all_box_lengths[f],
            surface_normal,
            no_z_pbc
        );
    }

    return all_orientations;
}

}  // namespace mlip
