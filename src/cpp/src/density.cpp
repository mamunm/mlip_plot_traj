#include "mlip_core.hpp"
#include <algorithm>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace mlip {

std::vector<std::vector<double>> compute_density_histogram(
    const double* positions,
    const int* types,
    size_t n_atoms,
    const std::array<double, 3>& box_lo,
    const std::array<double, 3>& box_hi,
    int axis,
    int n_bins,
    int n_types
) {
    // Initialize histogram: [n_types][n_bins]
    std::vector<std::vector<double>> histogram(n_types, std::vector<double>(n_bins, 0.0));

    const double axis_min = box_lo[axis];
    const double axis_max = box_hi[axis];
    const double bin_size = (axis_max - axis_min) / n_bins;
    const double inv_bin_size = 1.0 / bin_size;

    // Process each atom
    for (size_t i = 0; i < n_atoms; ++i) {
        const int type_idx = types[i];

        // Bounds check on type
        if (type_idx < 0 || type_idx >= n_types) {
            continue;
        }

        // Get coordinate along the specified axis
        const double coord = positions[i * 3 + axis];

        // Calculate bin index
        int bin_idx = static_cast<int>((coord - axis_min) * inv_bin_size);

        // Clamp to valid range
        if (bin_idx < 0) bin_idx = 0;
        if (bin_idx >= n_bins) bin_idx = n_bins - 1;

        histogram[type_idx][bin_idx] += 1.0;
    }

    return histogram;
}

std::vector<std::vector<double>> accumulate_density_frames(
    const std::vector<const double*>& all_positions,
    const std::vector<const int*>& all_types,
    const std::vector<size_t>& atoms_per_frame,
    const std::array<double, 3>& box_lo,
    const std::array<double, 3>& box_hi,
    int axis,
    int n_bins,
    int n_types
) {
    const size_t n_frames = all_positions.size();

    // Initialize accumulated histogram
    std::vector<std::vector<double>> total_histogram(n_types, std::vector<double>(n_bins, 0.0));

    const double axis_min = box_lo[axis];
    const double axis_max = box_hi[axis];
    const double bin_size = (axis_max - axis_min) / n_bins;
    const double inv_bin_size = 1.0 / bin_size;

#ifdef USE_OPENMP
    // Parallel version with thread-local histograms
    #pragma omp parallel
    {
        std::vector<std::vector<double>> local_histogram(n_types, std::vector<double>(n_bins, 0.0));

        #pragma omp for schedule(dynamic)
        for (size_t f = 0; f < n_frames; ++f) {
            const double* positions = all_positions[f];
            const int* types = all_types[f];
            const size_t n_atoms = atoms_per_frame[f];

            for (size_t i = 0; i < n_atoms; ++i) {
                const int type_idx = types[i];
                if (type_idx < 0 || type_idx >= n_types) continue;

                const double coord = positions[i * 3 + axis];
                int bin_idx = static_cast<int>((coord - axis_min) * inv_bin_size);

                if (bin_idx < 0) bin_idx = 0;
                if (bin_idx >= n_bins) bin_idx = n_bins - 1;

                local_histogram[type_idx][bin_idx] += 1.0;
            }
        }

        // Reduce to total
        #pragma omp critical
        {
            for (int t = 0; t < n_types; ++t) {
                for (int b = 0; b < n_bins; ++b) {
                    total_histogram[t][b] += local_histogram[t][b];
                }
            }
        }
    }
#else
    // Sequential version
    for (size_t f = 0; f < n_frames; ++f) {
        const double* positions = all_positions[f];
        const int* types = all_types[f];
        const size_t n_atoms = atoms_per_frame[f];

        for (size_t i = 0; i < n_atoms; ++i) {
            const int type_idx = types[i];
            if (type_idx < 0 || type_idx >= n_types) continue;

            const double coord = positions[i * 3 + axis];
            int bin_idx = static_cast<int>((coord - axis_min) * inv_bin_size);

            if (bin_idx < 0) bin_idx = 0;
            if (bin_idx >= n_bins) bin_idx = n_bins - 1;

            total_histogram[type_idx][bin_idx] += 1.0;
        }
    }
#endif

    return total_histogram;
}

std::vector<std::vector<std::vector<double>>> compute_density_histogram_per_frame(
    const std::vector<const double*>& all_positions,
    const std::vector<const int*>& all_types,
    const std::vector<size_t>& atoms_per_frame,
    const std::array<double, 3>& box_lo,
    const std::array<double, 3>& box_hi,
    int axis,
    int n_bins,
    int n_types
) {
    const size_t n_frames = all_positions.size();

    // Initialize output: [n_frames][n_types][n_bins]
    std::vector<std::vector<std::vector<double>>> per_frame_histograms(
        n_frames,
        std::vector<std::vector<double>>(n_types, std::vector<double>(n_bins, 0.0))
    );

    const double axis_min = box_lo[axis];
    const double axis_max = box_hi[axis];
    const double bin_size = (axis_max - axis_min) / n_bins;
    const double inv_bin_size = 1.0 / bin_size;

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t f = 0; f < n_frames; ++f) {
        const double* positions = all_positions[f];
        const int* types = all_types[f];
        const size_t n_atoms = atoms_per_frame[f];

        for (size_t i = 0; i < n_atoms; ++i) {
            const int type_idx = types[i];
            if (type_idx < 0 || type_idx >= n_types) continue;

            const double coord = positions[i * 3 + axis];
            int bin_idx = static_cast<int>((coord - axis_min) * inv_bin_size);

            if (bin_idx < 0) bin_idx = 0;
            if (bin_idx >= n_bins) bin_idx = n_bins - 1;

            per_frame_histograms[f][type_idx][bin_idx] += 1.0;
        }
    }

    return per_frame_histograms;
}

} // namespace mlip
