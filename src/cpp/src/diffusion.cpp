#include "mlip_core.hpp"
#include <cmath>
#include <algorithm>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace mlip {

namespace {

// Unwrap single coordinate across PBC
inline double unwrap_coord(double curr, double prev, double box_length) {
    double dx = curr - prev;
    return prev + dx - std::round(dx / box_length) * box_length;
}

} // anonymous namespace

std::vector<std::vector<double>> unwrap_trajectory(
    const std::vector<const double*>& all_positions,
    const std::vector<size_t>& atoms_per_frame,
    const std::array<double, 3>& box_lengths,
    bool unwrap_xy,
    bool unwrap_z
) {
    const size_t n_frames = all_positions.size();
    if (n_frames == 0) return {};

    const size_t n_atoms = atoms_per_frame[0];

    // Allocate output: n_frames * n_atoms * 3
    std::vector<std::vector<double>> unwrapped(n_frames);
    for (size_t f = 0; f < n_frames; ++f) {
        unwrapped[f].resize(n_atoms * 3);
    }

    // Copy first frame
    for (size_t i = 0; i < n_atoms * 3; ++i) {
        unwrapped[0][i] = all_positions[0][i];
    }

    // Unwrap subsequent frames
    for (size_t f = 1; f < n_frames; ++f) {
        for (size_t i = 0; i < n_atoms; ++i) {
            size_t idx = i * 3;

            // X coordinate
            if (unwrap_xy) {
                unwrapped[f][idx] = unwrap_coord(
                    all_positions[f][idx],
                    unwrapped[f-1][idx],
                    box_lengths[0]
                );
            } else {
                unwrapped[f][idx] = all_positions[f][idx];
            }

            // Y coordinate
            if (unwrap_xy) {
                unwrapped[f][idx + 1] = unwrap_coord(
                    all_positions[f][idx + 1],
                    unwrapped[f-1][idx + 1],
                    box_lengths[1]
                );
            } else {
                unwrapped[f][idx + 1] = all_positions[f][idx + 1];
            }

            // Z coordinate
            if (unwrap_z) {
                unwrapped[f][idx + 2] = unwrap_coord(
                    all_positions[f][idx + 2],
                    unwrapped[f-1][idx + 2],
                    box_lengths[2]
                );
            } else {
                unwrapped[f][idx + 2] = all_positions[f][idx + 2];
            }
        }
    }

    return unwrapped;
}

MSDResult compute_msd_all(
    const std::vector<std::vector<double>>& positions,
    size_t n_atoms
) {
    const size_t n_frames = positions.size();
    const size_t max_lag = n_frames - 1;

    MSDResult result;
    result.msd_planar.resize(max_lag, 0.0);
    result.msd_perpendicular.resize(max_lag, 0.0);
    result.msd_total.resize(max_lag, 0.0);

    std::vector<double> counts(max_lag, 0.0);

#ifdef USE_OPENMP
    // Parallel version using reduction
    std::vector<double> msd_xy_local(max_lag, 0.0);
    std::vector<double> msd_z_local(max_lag, 0.0);
    std::vector<double> msd_total_local(max_lag, 0.0);
    std::vector<double> counts_local(max_lag, 0.0);

    #pragma omp parallel
    {
        std::vector<double> thread_msd_xy(max_lag, 0.0);
        std::vector<double> thread_msd_z(max_lag, 0.0);
        std::vector<double> thread_msd_total(max_lag, 0.0);
        std::vector<double> thread_counts(max_lag, 0.0);

        #pragma omp for schedule(dynamic)
        for (size_t t0 = 0; t0 < n_frames; ++t0) {
            for (size_t dt = 1; dt < n_frames - t0; ++dt) {
                double sum_xy = 0.0;
                double sum_z = 0.0;
                double sum_total = 0.0;

                for (size_t i = 0; i < n_atoms; ++i) {
                    size_t idx = i * 3;

                    double dx = positions[t0 + dt][idx] - positions[t0][idx];
                    double dy = positions[t0 + dt][idx + 1] - positions[t0][idx + 1];
                    double dz = positions[t0 + dt][idx + 2] - positions[t0][idx + 2];

                    sum_xy += dx * dx + dy * dy;
                    sum_z += dz * dz;
                    sum_total += dx * dx + dy * dy + dz * dz;
                }

                thread_msd_xy[dt - 1] += sum_xy / n_atoms;
                thread_msd_z[dt - 1] += sum_z / n_atoms;
                thread_msd_total[dt - 1] += sum_total / n_atoms;
                thread_counts[dt - 1] += 1.0;
            }
        }

        #pragma omp critical
        {
            for (size_t dt = 0; dt < max_lag; ++dt) {
                result.msd_planar[dt] += thread_msd_xy[dt];
                result.msd_perpendicular[dt] += thread_msd_z[dt];
                result.msd_total[dt] += thread_msd_total[dt];
                counts[dt] += thread_counts[dt];
            }
        }
    }
#else
    // Sequential version - average over different time origins
    for (size_t t0 = 0; t0 < n_frames; ++t0) {
        for (size_t dt = 1; dt < n_frames - t0; ++dt) {
            double sum_xy = 0.0;
            double sum_z = 0.0;
            double sum_total = 0.0;

            for (size_t i = 0; i < n_atoms; ++i) {
                size_t idx = i * 3;

                double dx = positions[t0 + dt][idx] - positions[t0][idx];
                double dy = positions[t0 + dt][idx + 1] - positions[t0][idx + 1];
                double dz = positions[t0 + dt][idx + 2] - positions[t0][idx + 2];

                sum_xy += dx * dx + dy * dy;
                sum_z += dz * dz;
                sum_total += dx * dx + dy * dy + dz * dz;
            }

            result.msd_planar[dt - 1] += sum_xy / n_atoms;
            result.msd_perpendicular[dt - 1] += sum_z / n_atoms;
            result.msd_total[dt - 1] += sum_total / n_atoms;
            counts[dt - 1] += 1.0;
        }
    }
#endif

    // Average over time origins
    for (size_t dt = 0; dt < max_lag; ++dt) {
        if (counts[dt] > 0) {
            result.msd_planar[dt] /= counts[dt];
            result.msd_perpendicular[dt] /= counts[dt];
            result.msd_total[dt] /= counts[dt];
        }
    }

    return result;
}

MSDResult compute_msd_from_positions(
    const std::vector<const double*>& all_positions,
    const std::vector<size_t>& atoms_per_frame,
    const std::array<double, 3>& box_lengths,
    bool unwrap_xy,
    bool unwrap_z
) {
    // First unwrap the trajectory
    auto unwrapped = unwrap_trajectory(
        all_positions, atoms_per_frame, box_lengths, unwrap_xy, unwrap_z
    );

    // Then compute MSD
    return compute_msd_all(unwrapped, atoms_per_frame[0]);
}

namespace {

// Check if atom is within z-region
inline bool in_region(double z, double z_min, double z_max) {
    return z >= z_min && z <= z_max;
}

} // anonymous namespace

RegionMSDResult compute_msd_regions(
    const std::vector<std::vector<double>>& positions,
    size_t n_atoms,
    const std::vector<Region>& regions
) {
    const size_t n_frames = positions.size();
    const size_t max_lag = n_frames - 1;

    RegionMSDResult result;

    // Initialize results for each region
    for (const auto& region : regions) {
        MSDResult msd_result;
        msd_result.msd_planar.resize(max_lag, 0.0);
        msd_result.msd_perpendicular.resize(max_lag, 0.0);
        msd_result.msd_total.resize(max_lag, 0.0);
        result.region_results[region.name] = msd_result;
    }

    // Counts for averaging
    std::unordered_map<std::string, std::vector<double>> counts;
    for (const auto& region : regions) {
        counts[region.name].resize(max_lag, 0.0);
    }

    // Process each region
    for (const auto& region : regions) {
        const std::string& region_name = region.name;
        const double z_min = region.z_min;
        const double z_max = region.z_max;

#ifdef USE_OPENMP
        #pragma omp parallel
        {
            std::vector<double> thread_msd_xy(max_lag, 0.0);
            std::vector<double> thread_msd_z(max_lag, 0.0);
            std::vector<double> thread_msd_total(max_lag, 0.0);
            std::vector<double> thread_counts(max_lag, 0.0);

            #pragma omp for schedule(dynamic)
            for (size_t t0 = 0; t0 < n_frames; ++t0) {
                // Pre-compute which atoms are in region at t0
                std::vector<bool> in_region_t0(n_atoms);
                for (size_t i = 0; i < n_atoms; ++i) {
                    double z = positions[t0][i * 3 + 2];
                    in_region_t0[i] = in_region(z, z_min, z_max);
                }

                for (size_t dt = 1; dt < n_frames - t0; ++dt) {
                    double sum_xy = 0.0;
                    double sum_z = 0.0;
                    double sum_total = 0.0;
                    size_t count_in_region = 0;

                    for (size_t i = 0; i < n_atoms; ++i) {
                        // Check if atom is in region at t0+dt
                        double z_tdt = positions[t0 + dt][i * 3 + 2];
                        bool in_region_tdt = in_region(z_tdt, z_min, z_max);

                        // Only include if in region at BOTH times
                        if (in_region_t0[i] && in_region_tdt) {
                            size_t idx = i * 3;

                            double dx = positions[t0 + dt][idx] - positions[t0][idx];
                            double dy = positions[t0 + dt][idx + 1] - positions[t0][idx + 1];
                            double dz = positions[t0 + dt][idx + 2] - positions[t0][idx + 2];

                            sum_xy += dx * dx + dy * dy;
                            sum_z += dz * dz;
                            sum_total += dx * dx + dy * dy + dz * dz;
                            count_in_region++;
                        }
                    }

                    if (count_in_region > 0) {
                        thread_msd_xy[dt - 1] += sum_xy / count_in_region;
                        thread_msd_z[dt - 1] += sum_z / count_in_region;
                        thread_msd_total[dt - 1] += sum_total / count_in_region;
                        thread_counts[dt - 1] += 1.0;
                    }
                }
            }

            #pragma omp critical
            {
                for (size_t dt = 0; dt < max_lag; ++dt) {
                    result.region_results[region_name].msd_planar[dt] += thread_msd_xy[dt];
                    result.region_results[region_name].msd_perpendicular[dt] += thread_msd_z[dt];
                    result.region_results[region_name].msd_total[dt] += thread_msd_total[dt];
                    counts[region_name][dt] += thread_counts[dt];
                }
            }
        }
#else
        // Sequential version
        for (size_t t0 = 0; t0 < n_frames; ++t0) {
            // Pre-compute which atoms are in region at t0
            std::vector<bool> in_region_t0(n_atoms);
            for (size_t i = 0; i < n_atoms; ++i) {
                double z = positions[t0][i * 3 + 2];
                in_region_t0[i] = in_region(z, z_min, z_max);
            }

            for (size_t dt = 1; dt < n_frames - t0; ++dt) {
                double sum_xy = 0.0;
                double sum_z = 0.0;
                double sum_total = 0.0;
                size_t count_in_region = 0;

                for (size_t i = 0; i < n_atoms; ++i) {
                    // Check if atom is in region at t0+dt
                    double z_tdt = positions[t0 + dt][i * 3 + 2];
                    bool in_region_tdt = in_region(z_tdt, z_min, z_max);

                    // Only include if in region at BOTH times
                    if (in_region_t0[i] && in_region_tdt) {
                        size_t idx = i * 3;

                        double dx = positions[t0 + dt][idx] - positions[t0][idx];
                        double dy = positions[t0 + dt][idx + 1] - positions[t0][idx + 1];
                        double dz = positions[t0 + dt][idx + 2] - positions[t0][idx + 2];

                        sum_xy += dx * dx + dy * dy;
                        sum_z += dz * dz;
                        sum_total += dx * dx + dy * dy + dz * dz;
                        count_in_region++;
                    }
                }

                if (count_in_region > 0) {
                    result.region_results[region_name].msd_planar[dt - 1] += sum_xy / count_in_region;
                    result.region_results[region_name].msd_perpendicular[dt - 1] += sum_z / count_in_region;
                    result.region_results[region_name].msd_total[dt - 1] += sum_total / count_in_region;
                    counts[region_name][dt - 1] += 1.0;
                }
            }
        }
#endif

        // Average over time origins for this region
        for (size_t dt = 0; dt < max_lag; ++dt) {
            if (counts[region_name][dt] > 0) {
                result.region_results[region_name].msd_planar[dt] /= counts[region_name][dt];
                result.region_results[region_name].msd_perpendicular[dt] /= counts[region_name][dt];
                result.region_results[region_name].msd_total[dt] /= counts[region_name][dt];
            }
        }
    }

    return result;
}

RegionMSDResult compute_msd_regions_from_positions(
    const std::vector<const double*>& all_positions,
    const std::vector<size_t>& atoms_per_frame,
    const std::array<double, 3>& box_lengths,
    const std::vector<Region>& regions,
    bool unwrap_xy,
    bool unwrap_z
) {
    // First unwrap the trajectory
    auto unwrapped = unwrap_trajectory(
        all_positions, atoms_per_frame, box_lengths, unwrap_xy, unwrap_z
    );

    // Then compute region-based MSD
    return compute_msd_regions(unwrapped, atoms_per_frame[0], regions);
}

} // namespace mlip
