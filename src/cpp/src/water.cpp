/**
 * Water molecule identification and property extraction.
 *
 * Provides functions for identifying water molecules in MD trajectories
 * and computing their geometric properties (positions, distances, angles).
 * Supports periodic boundary conditions via minimum image convention.
 */

#include "mlip_core.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_set>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace mlip {

namespace {

/**
 * Apply minimum image convention for periodic boundary conditions.
 * Modifies dx, dy, dz in place to be the shortest distance vector.
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
 * Returns the vector (pos2 - pos1) under minimum image convention.
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
 *
 * Uses the dot product formula:
 *   cos(angle) = (BA . BC) / (|BA| * |BC|)
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
    // Clamp for numerical safety
    cos_angle = std::max(-1.0, std::min(1.0, cos_angle));

    // Convert to degrees
    return std::acos(cos_angle) * 180.0 / M_PI;
}

}  // anonymous namespace

WaterIdentificationResult identify_water_molecules(
    const double* positions,
    const int* element_types,
    size_t n_atoms,
    int o_type_idx,
    int h_type_idx,
    const std::array<double, 3>& box_lengths,
    double o_h_cutoff
) {
    WaterIdentificationResult result;

    // Collect O and H atom indices
    std::vector<int> o_indices;
    std::vector<int> h_indices;

    for (size_t i = 0; i < n_atoms; ++i) {
        if (element_types[i] == o_type_idx) {
            o_indices.push_back(static_cast<int>(i));
        } else if (element_types[i] == h_type_idx) {
            h_indices.push_back(static_cast<int>(i));
        }
    }

    if (o_indices.empty() || h_indices.size() < 2) {
        result.n_molecules = 0;
        return result;
    }

    std::unordered_set<int> used_h;

    // For each O atom, find the two closest H atoms within cutoff
    for (int o_idx : o_indices) {
        const double* o_pos = positions + o_idx * 3;

        // Find H atoms within cutoff
        std::vector<std::pair<int, double>> bonded_h;

        for (int h_idx : h_indices) {
            if (used_h.count(h_idx)) continue;

            const double* h_pos = positions + h_idx * 3;
            double dist = distance_pbc(o_pos, h_pos, box_lengths);

            if (dist < o_h_cutoff) {
                bonded_h.push_back({h_idx, dist});
            }
        }

        // Sort by distance and take closest 2
        if (bonded_h.size() >= 2) {
            std::sort(bonded_h.begin(), bonded_h.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });

            int h1_idx = bonded_h[0].first;
            int h2_idx = bonded_h[1].first;

            result.molecule_indices.push_back({o_idx, h1_idx, h2_idx});
            used_h.insert(h1_idx);
            used_h.insert(h2_idx);
        }
    }

    result.n_molecules = result.molecule_indices.size();
    return result;
}

std::vector<WaterMolecule> extract_water_properties(
    const double* positions,
    const std::vector<std::array<int, 3>>& molecule_indices,
    size_t n_molecules,
    const std::array<double, 3>& box_lengths
) {
    std::vector<WaterMolecule> molecules(n_molecules);

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < n_molecules; ++i) {
        const auto& indices = molecule_indices[i];
        WaterMolecule& mol = molecules[i];

        mol.O_index = indices[0];
        mol.H1_index = indices[1];
        mol.H2_index = indices[2];

        const double* o_pos = positions + indices[0] * 3;
        const double* h1_pos = positions + indices[1] * 3;
        const double* h2_pos = positions + indices[2] * 3;

        // Copy positions (raw positions, not unwrapped)
        for (int j = 0; j < 3; ++j) {
            mol.O_position[j] = o_pos[j];
            mol.H1_position[j] = h1_pos[j];
            mol.H2_position[j] = h2_pos[j];
        }

        // Calculate distances with PBC
        mol.OH1_distance = distance_pbc(o_pos, h1_pos, box_lengths);
        mol.OH2_distance = distance_pbc(o_pos, h2_pos, box_lengths);
        mol.H1H2_distance = distance_pbc(h1_pos, h2_pos, box_lengths);

        // Calculate H1-O-H2 angle (angle at O between the two H atoms) with PBC
        mol.H1OH2_angle = angle_abc_pbc(h1_pos, o_pos, h2_pos, box_lengths);
    }

    return molecules;
}

std::vector<std::vector<WaterMolecule>> extract_water_properties_multiframe(
    const std::vector<const double*>& all_positions,
    const std::vector<std::array<int, 3>>& molecule_indices,
    size_t n_molecules,
    size_t n_frames,
    const std::array<double, 3>& box_lengths
) {
    std::vector<std::vector<WaterMolecule>> all_molecules(n_frames);

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t f = 0; f < n_frames; ++f) {
        all_molecules[f] = extract_water_properties(
            all_positions[f], molecule_indices, n_molecules, box_lengths
        );
    }

    return all_molecules;
}

}  // namespace mlip
