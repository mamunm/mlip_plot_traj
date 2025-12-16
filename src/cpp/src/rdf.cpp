#include "mlip_core.hpp"
#include <cmath>
#include <algorithm>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace mlip {

namespace {

// Apply minimum image convention for periodic boundary conditions
inline void apply_pbc(
    double& dx, double& dy, double& dz,
    const std::array<std::array<double, 3>, 3>& cell
) {
    // Calculate cell matrix determinant
    const auto& a = cell[0];
    const auto& b = cell[1];
    const auto& c = cell[2];

    double det = a[0] * (b[1] * c[2] - b[2] * c[1])
               - a[1] * (b[0] * c[2] - b[2] * c[0])
               + a[2] * (b[0] * c[1] - b[1] * c[0]);

    if (std::abs(det) < 1e-10) return;

    double inv_det = 1.0 / det;

    // Calculate inverse cell matrix
    double inv00 = (b[1] * c[2] - b[2] * c[1]) * inv_det;
    double inv01 = (a[2] * c[1] - a[1] * c[2]) * inv_det;
    double inv02 = (a[1] * b[2] - a[2] * b[1]) * inv_det;
    double inv10 = (b[2] * c[0] - b[0] * c[2]) * inv_det;
    double inv11 = (a[0] * c[2] - a[2] * c[0]) * inv_det;
    double inv12 = (a[2] * b[0] - a[0] * b[2]) * inv_det;
    double inv20 = (b[0] * c[1] - b[1] * c[0]) * inv_det;
    double inv21 = (a[1] * c[0] - a[0] * c[1]) * inv_det;
    double inv22 = (a[0] * b[1] - a[1] * b[0]) * inv_det;

    // Convert to fractional coordinates
    double s = inv00 * dx + inv01 * dy + inv02 * dz;
    double t = inv10 * dx + inv11 * dy + inv12 * dz;
    double u = inv20 * dx + inv21 * dy + inv22 * dz;

    // Apply minimum image convention
    s -= std::round(s);
    t -= std::round(t);
    u -= std::round(u);

    // Convert back to Cartesian
    dx = s * a[0] + t * b[0] + u * c[0];
    dy = s * a[1] + t * b[1] + u * c[1];
    dz = s * a[2] + t * b[2] + u * c[2];
}

// Calculate cell volume
inline double cell_volume(const std::array<std::array<double, 3>, 3>& cell) {
    const auto& a = cell[0];
    const auto& b = cell[1];
    const auto& c = cell[2];

    return std::abs(
        a[0] * (b[1] * c[2] - b[2] * c[1])
      - a[1] * (b[0] * c[2] - b[2] * c[0])
      + a[2] * (b[0] * c[1] - b[1] * c[0])
    );
}

} // anonymous namespace

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
) {
    std::vector<double> histogram(n_bins, 0.0);

    const double dr = (rmax - rmin) / n_bins;
    const double inv_dr = 1.0 / dr;
    const double rmax_sq = rmax * rmax;
    const double rmin_sq = rmin * rmin;

    // Determine if this is a self-pair (e.g., O-O) or cross-pair (e.g., O-H)
    bool self_pair = (type1 == type2);

    for (size_t i = 0; i < n_atoms; ++i) {
        // Check if atom i matches type1
        if (type1 >= 0 && types[i] != type1) continue;

        const double xi = positions[i * 3 + 0];
        const double yi = positions[i * 3 + 1];
        const double zi = positions[i * 3 + 2];

        // For self-pairs, only count j > i to avoid double counting
        size_t j_start = self_pair ? (i + 1) : 0;

        for (size_t j = j_start; j < n_atoms; ++j) {
            if (i == j) continue;

            // Check if atom j matches type2
            if (type2 >= 0 && types[j] != type2) continue;

            double dx = xi - positions[j * 3 + 0];
            double dy = yi - positions[j * 3 + 1];
            double dz = zi - positions[j * 3 + 2];

            // Apply periodic boundary conditions
            apply_pbc(dx, dy, dz, cell_vectors);

            double r_sq = dx * dx + dy * dy + dz * dz;

            if (r_sq >= rmin_sq && r_sq < rmax_sq) {
                double r = std::sqrt(r_sq);
                int bin_idx = static_cast<int>((r - rmin) * inv_dr);

                if (bin_idx >= 0 && bin_idx < n_bins) {
                    histogram[bin_idx] += 1.0;
                }
            }
        }
    }

    return histogram;
}

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
) {
    RDFResult result;
    result.histogram.resize(n_bins, 0.0);
    result.total_pairs = 0.0;
    result.total_volume = 0.0;
    result.n_frames = 0;

    const size_t n_frames = all_positions.size();
    const double vol = cell_volume(cell_vectors);

    // Determine if self-pair
    bool self_pair = (type1 == type2);

#ifdef USE_OPENMP
    // Parallel version
    #pragma omp parallel
    {
        std::vector<double> local_hist(n_bins, 0.0);
        double local_pairs = 0.0;
        int local_frames = 0;

        #pragma omp for schedule(dynamic)
        for (size_t f = 0; f < n_frames; ++f) {
            auto frame_hist = compute_rdf_histogram(
                all_positions[f], all_types[f], atoms_per_frame[f],
                cell_vectors, rmin, rmax, n_bins, type1, type2
            );

            for (int b = 0; b < n_bins; ++b) {
                local_hist[b] += frame_hist[b];
            }

            // Count atoms of each type
            size_t n_type1 = 0, n_type2 = 0;
            for (size_t i = 0; i < atoms_per_frame[f]; ++i) {
                if (type1 < 0 || all_types[f][i] == type1) n_type1++;
                if (type2 < 0 || all_types[f][i] == type2) n_type2++;
            }

            if (self_pair) {
                local_pairs += n_type1 * (n_type1 - 1) / 2.0;
            } else {
                local_pairs += n_type1 * n_type2;
            }
            local_frames++;
        }

        #pragma omp critical
        {
            for (int b = 0; b < n_bins; ++b) {
                result.histogram[b] += local_hist[b];
            }
            result.total_pairs += local_pairs;
            result.n_frames += local_frames;
        }
    }
#else
    // Sequential version
    for (size_t f = 0; f < n_frames; ++f) {
        auto frame_hist = compute_rdf_histogram(
            all_positions[f], all_types[f], atoms_per_frame[f],
            cell_vectors, rmin, rmax, n_bins, type1, type2
        );

        for (int b = 0; b < n_bins; ++b) {
            result.histogram[b] += frame_hist[b];
        }

        // Count atoms of each type
        size_t n_type1 = 0, n_type2 = 0;
        for (size_t i = 0; i < atoms_per_frame[f]; ++i) {
            if (type1 < 0 || all_types[f][i] == type1) n_type1++;
            if (type2 < 0 || all_types[f][i] == type2) n_type2++;
        }

        if (self_pair) {
            result.total_pairs += n_type1 * (n_type1 - 1) / 2.0;
        } else {
            result.total_pairs += n_type1 * n_type2;
        }
        result.n_frames++;
    }
#endif

    result.total_volume = vol * result.n_frames;

    return result;
}

} // namespace mlip
