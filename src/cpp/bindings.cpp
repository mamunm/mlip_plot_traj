#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "mlip_core.hpp"

namespace py = pybind11;

/**
 * Python wrapper for single-frame density histogram.
 *
 * @param positions  numpy array of shape (N, 3) with atom positions
 * @param types      numpy array of shape (N,) with type indices (0-indexed)
 * @param box_lo     tuple (xlo, ylo, zlo)
 * @param box_hi     tuple (xhi, yhi, zhi)
 * @param axis       axis for profile: 0=x, 1=y, 2=z
 * @param n_bins     number of bins
 * @param n_types    number of unique atom types
 * @return           numpy array of shape (n_types, n_bins)
 */
py::array_t<double> py_compute_density_histogram(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<int, py::array::c_style | py::array::forcecast> types,
    std::tuple<double, double, double> box_lo,
    std::tuple<double, double, double> box_hi,
    int axis,
    int n_bins,
    int n_types
) {
    // Get buffer info
    auto pos_buf = positions.request();
    auto type_buf = types.request();

    if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
        throw std::runtime_error("positions must have shape (N, 3)");
    }
    if (type_buf.ndim != 1) {
        throw std::runtime_error("types must be 1D array");
    }
    if (pos_buf.shape[0] != type_buf.shape[0]) {
        throw std::runtime_error("positions and types must have same length");
    }

    size_t n_atoms = pos_buf.shape[0];
    const double* pos_ptr = static_cast<const double*>(pos_buf.ptr);
    const int* type_ptr = static_cast<const int*>(type_buf.ptr);

    std::array<double, 3> lo = {std::get<0>(box_lo), std::get<1>(box_lo), std::get<2>(box_lo)};
    std::array<double, 3> hi = {std::get<0>(box_hi), std::get<1>(box_hi), std::get<2>(box_hi)};

    // Compute histogram
    auto histogram = mlip::compute_density_histogram(
        pos_ptr, type_ptr, n_atoms, lo, hi, axis, n_bins, n_types
    );

    // Convert to numpy array
    py::array_t<double> result({static_cast<py::ssize_t>(n_types), static_cast<py::ssize_t>(n_bins)});
    auto result_buf = result.mutable_unchecked<2>();

    for (int t = 0; t < n_types; ++t) {
        for (int b = 0; b < n_bins; ++b) {
            result_buf(t, b) = histogram[t][b];
        }
    }

    return result;
}

/**
 * Python wrapper for multi-frame density accumulation.
 *
 * @param positions_list  list of numpy arrays, each (N_i, 3)
 * @param types_list      list of numpy arrays, each (N_i,)
 * @param box_lo          tuple (xlo, ylo, zlo)
 * @param box_hi          tuple (xhi, yhi, zhi)
 * @param axis            axis for profile
 * @param n_bins          number of bins
 * @param n_types         number of unique atom types
 * @return                numpy array of shape (n_types, n_bins) with accumulated counts
 */
py::array_t<double> py_accumulate_density_frames(
    py::list positions_list,
    py::list types_list,
    std::tuple<double, double, double> box_lo,
    std::tuple<double, double, double> box_hi,
    int axis,
    int n_bins,
    int n_types
) {
    size_t n_frames = positions_list.size();

    if (n_frames != types_list.size()) {
        throw std::runtime_error("positions_list and types_list must have same length");
    }

    std::vector<const double*> all_positions;
    std::vector<const int*> all_types;
    std::vector<size_t> atoms_per_frame;

    // Keep references to prevent garbage collection
    std::vector<py::array_t<double>> pos_arrays;
    std::vector<py::array_t<int>> type_arrays;

    for (size_t f = 0; f < n_frames; ++f) {
        auto pos_arr = positions_list[f].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto type_arr = types_list[f].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();

        pos_arrays.push_back(pos_arr);
        type_arrays.push_back(type_arr);

        auto pos_buf = pos_arr.request();
        auto type_buf = type_arr.request();

        all_positions.push_back(static_cast<const double*>(pos_buf.ptr));
        all_types.push_back(static_cast<const int*>(type_buf.ptr));
        atoms_per_frame.push_back(pos_buf.shape[0]);
    }

    std::array<double, 3> lo = {std::get<0>(box_lo), std::get<1>(box_lo), std::get<2>(box_lo)};
    std::array<double, 3> hi = {std::get<0>(box_hi), std::get<1>(box_hi), std::get<2>(box_hi)};

    // Compute accumulated histogram
    auto histogram = mlip::accumulate_density_frames(
        all_positions, all_types, atoms_per_frame, lo, hi, axis, n_bins, n_types
    );

    // Convert to numpy array
    py::array_t<double> result({static_cast<py::ssize_t>(n_types), static_cast<py::ssize_t>(n_bins)});
    auto result_buf = result.mutable_unchecked<2>();

    for (int t = 0; t < n_types; ++t) {
        for (int b = 0; b < n_bins; ++b) {
            result_buf(t, b) = histogram[t][b];
        }
    }

    return result;
}

/**
 * Python wrapper for per-frame density histogram computation.
 * Used for principled error estimation methods (autocorrelation, Flyvbjerg-Petersen).
 *
 * @param positions_list  list of numpy arrays, each (N_i, 3)
 * @param types_list      list of numpy arrays, each (N_i,)
 * @param box_lo          tuple (xlo, ylo, zlo)
 * @param box_hi          tuple (xhi, yhi, zhi)
 * @param axis            axis for profile
 * @param n_bins          number of bins
 * @param n_types         number of unique atom types
 * @return                numpy array of shape (n_frames, n_types, n_bins) with per-frame counts
 */
py::array_t<double> py_compute_density_histogram_per_frame(
    py::list positions_list,
    py::list types_list,
    std::tuple<double, double, double> box_lo,
    std::tuple<double, double, double> box_hi,
    int axis,
    int n_bins,
    int n_types
) {
    size_t n_frames = positions_list.size();

    if (n_frames != types_list.size()) {
        throw std::runtime_error("positions_list and types_list must have same length");
    }

    std::vector<const double*> all_positions;
    std::vector<const int*> all_types;
    std::vector<size_t> atoms_per_frame;

    // Keep references to prevent garbage collection
    std::vector<py::array_t<double>> pos_arrays;
    std::vector<py::array_t<int>> type_arrays;

    for (size_t f = 0; f < n_frames; ++f) {
        auto pos_arr = positions_list[f].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto type_arr = types_list[f].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();

        pos_arrays.push_back(pos_arr);
        type_arrays.push_back(type_arr);

        auto pos_buf = pos_arr.request();
        auto type_buf = type_arr.request();

        all_positions.push_back(static_cast<const double*>(pos_buf.ptr));
        all_types.push_back(static_cast<const int*>(type_buf.ptr));
        atoms_per_frame.push_back(pos_buf.shape[0]);
    }

    std::array<double, 3> lo = {std::get<0>(box_lo), std::get<1>(box_lo), std::get<2>(box_lo)};
    std::array<double, 3> hi = {std::get<0>(box_hi), std::get<1>(box_hi), std::get<2>(box_hi)};

    // Compute per-frame histograms
    auto histograms = mlip::compute_density_histogram_per_frame(
        all_positions, all_types, atoms_per_frame, lo, hi, axis, n_bins, n_types
    );

    // Convert to numpy array of shape (n_frames, n_types, n_bins)
    py::array_t<double> result({
        static_cast<py::ssize_t>(n_frames),
        static_cast<py::ssize_t>(n_types),
        static_cast<py::ssize_t>(n_bins)
    });
    auto result_buf = result.mutable_unchecked<3>();

    for (size_t f = 0; f < n_frames; ++f) {
        for (int t = 0; t < n_types; ++t) {
            for (int b = 0; b < n_bins; ++b) {
                result_buf(f, t, b) = histograms[f][t][b];
            }
        }
    }

    return result;
}

/**
 * Python wrapper for RDF histogram accumulation.
 *
 * @param positions_list  list of numpy arrays, each (N_i, 3)
 * @param types_list      list of numpy arrays, each (N_i,)
 * @param cell_vectors    3x3 numpy array of cell vectors (a, b, c as rows)
 * @param rmin            minimum distance
 * @param rmax            maximum distance
 * @param n_bins          number of bins
 * @param type1           first atom type (0-indexed, -1 for all)
 * @param type2           second atom type (0-indexed, -1 for all)
 * @return                tuple of (histogram, total_pairs, total_volume, n_frames)
 */
py::tuple py_accumulate_rdf_frames(
    py::list positions_list,
    py::list types_list,
    py::array_t<double, py::array::c_style | py::array::forcecast> cell_vectors,
    double rmin,
    double rmax,
    int n_bins,
    int type1,
    int type2
) {
    size_t n_frames = positions_list.size();

    if (n_frames != types_list.size()) {
        throw std::runtime_error("positions_list and types_list must have same length");
    }

    auto cell_buf = cell_vectors.request();
    if (cell_buf.ndim != 2 || cell_buf.shape[0] != 3 || cell_buf.shape[1] != 3) {
        throw std::runtime_error("cell_vectors must have shape (3, 3)");
    }

    const double* cell_ptr = static_cast<const double*>(cell_buf.ptr);
    std::array<std::array<double, 3>, 3> cell;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cell[i][j] = cell_ptr[i * 3 + j];
        }
    }

    std::vector<const double*> all_positions;
    std::vector<const int*> all_types;
    std::vector<size_t> atoms_per_frame;

    // Keep references to prevent garbage collection
    std::vector<py::array_t<double>> pos_arrays;
    std::vector<py::array_t<int>> type_arrays;

    for (size_t f = 0; f < n_frames; ++f) {
        auto pos_arr = positions_list[f].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto type_arr = types_list[f].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();

        pos_arrays.push_back(pos_arr);
        type_arrays.push_back(type_arr);

        auto pos_buf = pos_arr.request();
        auto type_buf = type_arr.request();

        all_positions.push_back(static_cast<const double*>(pos_buf.ptr));
        all_types.push_back(static_cast<const int*>(type_buf.ptr));
        atoms_per_frame.push_back(pos_buf.shape[0]);
    }

    // Compute accumulated RDF
    auto result = mlip::accumulate_rdf_frames(
        all_positions, all_types, atoms_per_frame,
        cell, rmin, rmax, n_bins, type1, type2
    );

    // Convert histogram to numpy array
    py::array_t<double> hist_array(n_bins);
    auto hist_buf = hist_array.mutable_unchecked<1>();
    for (int b = 0; b < n_bins; ++b) {
        hist_buf(b) = result.histogram[b];
    }

    return py::make_tuple(hist_array, result.total_pairs, result.total_volume, result.n_frames);
}

/**
 * Python wrapper for H-bond detection over multiple frames.
 */
py::list py_accumulate_hbonds_frames(
    py::list positions_list,
    py::list types_list,
    py::array_t<double, py::array::c_style | py::array::forcecast> cell_vectors,
    int donor_type,
    int hydrogen_type,
    int acceptor_type,
    double d_a_cutoff,
    double angle_cutoff,
    double d_h_cutoff,
    int bin_axis,
    std::tuple<double, double, double> box_lo,
    std::tuple<double, double, double> box_hi,
    int n_bins
) {
    size_t n_frames = positions_list.size();

    if (n_frames != types_list.size()) {
        throw std::runtime_error("positions_list and types_list must have same length");
    }

    auto cell_buf = cell_vectors.request();
    if (cell_buf.ndim != 2 || cell_buf.shape[0] != 3 || cell_buf.shape[1] != 3) {
        throw std::runtime_error("cell_vectors must have shape (3, 3)");
    }

    const double* cell_ptr = static_cast<const double*>(cell_buf.ptr);
    std::array<std::array<double, 3>, 3> cell;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cell[i][j] = cell_ptr[i * 3 + j];
        }
    }

    std::array<double, 3> lo = {std::get<0>(box_lo), std::get<1>(box_lo), std::get<2>(box_lo)};
    std::array<double, 3> hi = {std::get<0>(box_hi), std::get<1>(box_hi), std::get<2>(box_hi)};

    std::vector<const double*> all_positions;
    std::vector<const int*> all_types;
    std::vector<size_t> atoms_per_frame;

    // Keep references to prevent garbage collection
    std::vector<py::array_t<double>> pos_arrays;
    std::vector<py::array_t<int>> type_arrays;

    for (size_t f = 0; f < n_frames; ++f) {
        auto pos_arr = positions_list[f].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto type_arr = types_list[f].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();

        pos_arrays.push_back(pos_arr);
        type_arrays.push_back(type_arr);

        auto pos_buf = pos_arr.request();
        auto type_buf = type_arr.request();

        all_positions.push_back(static_cast<const double*>(pos_buf.ptr));
        all_types.push_back(static_cast<const int*>(type_buf.ptr));
        atoms_per_frame.push_back(pos_buf.shape[0]);
    }

    // Compute H-bonds
    auto results = mlip::accumulate_hbonds_frames(
        all_positions, all_types, atoms_per_frame,
        cell, donor_type, hydrogen_type, acceptor_type,
        d_a_cutoff, angle_cutoff, d_h_cutoff,
        bin_axis, lo, hi, n_bins
    );

    // Convert to Python list of dicts
    py::list result_list;
    for (const auto& r : results) {
        py::dict frame_result;
        frame_result["total_hbonds"] = r.total_hbonds;

        py::array_t<double> hb_array(r.hbonds_per_bin.size());
        auto hb_buf = hb_array.mutable_unchecked<1>();
        for (size_t i = 0; i < r.hbonds_per_bin.size(); ++i) {
            hb_buf(i) = r.hbonds_per_bin[i];
        }
        frame_result["hbonds_per_bin"] = hb_array;

        py::array_t<double> d_array(r.donors_per_bin.size());
        auto d_buf = d_array.mutable_unchecked<1>();
        for (size_t i = 0; i < r.donors_per_bin.size(); ++i) {
            d_buf(i) = r.donors_per_bin[i];
        }
        frame_result["donors_per_bin"] = d_array;

        result_list.append(frame_result);
    }

    return result_list;
}

/**
 * Python wrapper for H-bond pair detection (for lifetime analysis).
 * Returns list of (donor_idx, acceptor_idx) tuples per frame.
 */
py::list py_detect_hbond_pairs_frames(
    py::list positions_list,
    py::list types_list,
    py::array_t<double, py::array::c_style | py::array::forcecast> cell_vectors,
    int donor_type,
    int hydrogen_type,
    int acceptor_type,
    double d_a_cutoff,
    double angle_cutoff,
    double d_h_cutoff
) {
    size_t n_frames = positions_list.size();

    if (n_frames != types_list.size()) {
        throw std::runtime_error("positions_list and types_list must have same length");
    }

    auto cell_buf = cell_vectors.request();
    const double* cell_ptr = static_cast<const double*>(cell_buf.ptr);
    std::array<std::array<double, 3>, 3> cell;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cell[i][j] = cell_ptr[i * 3 + j];
        }
    }

    py::list result_list;

    for (size_t f = 0; f < n_frames; ++f) {
        auto pos_arr = positions_list[f].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto type_arr = types_list[f].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();

        auto pos_buf = pos_arr.request();
        auto type_buf = type_arr.request();

        auto pairs = mlip::detect_hbond_pairs(
            static_cast<const double*>(pos_buf.ptr),
            static_cast<const int*>(type_buf.ptr),
            pos_buf.shape[0],
            cell, donor_type, hydrogen_type, acceptor_type,
            d_a_cutoff, angle_cutoff, d_h_cutoff
        );

        // Convert to list of tuples
        py::list frame_pairs;
        for (const auto& p : pairs) {
            frame_pairs.append(py::make_tuple(p.first, p.second));
        }
        result_list.append(frame_pairs);
    }

    return result_list;
}

/**
 * Python wrapper for region-based MSD computation.
 *
 * @param positions_list  list of numpy arrays, each (N, 3)
 * @param box_lengths     tuple (Lx, Ly, Lz)
 * @param regions         dict mapping region name to (z_min, z_max) tuples
 * @param unwrap_xy       whether to unwrap x and y coordinates
 * @param unwrap_z        whether to unwrap z coordinate
 * @return                dict mapping region name to MSD dict
 */
py::dict py_compute_msd_regions(
    py::list positions_list,
    std::tuple<double, double, double> box_lengths,
    py::dict regions_dict,
    bool unwrap_xy,
    bool unwrap_z
) {
    size_t n_frames = positions_list.size();

    if (n_frames < 2) {
        throw std::runtime_error("Need at least 2 frames to compute MSD");
    }

    std::vector<const double*> all_positions;
    std::vector<size_t> atoms_per_frame;

    // Keep references to prevent garbage collection
    std::vector<py::array_t<double>> pos_arrays;

    for (size_t f = 0; f < n_frames; ++f) {
        auto pos_arr = positions_list[f].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        pos_arrays.push_back(pos_arr);

        auto pos_buf = pos_arr.request();

        if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
            throw std::runtime_error("Each position array must have shape (N, 3)");
        }

        all_positions.push_back(static_cast<const double*>(pos_buf.ptr));
        atoms_per_frame.push_back(pos_buf.shape[0]);
    }

    // Check all frames have same number of atoms
    for (size_t f = 1; f < n_frames; ++f) {
        if (atoms_per_frame[f] != atoms_per_frame[0]) {
            throw std::runtime_error("All frames must have the same number of atoms");
        }
    }

    std::array<double, 3> box = {
        std::get<0>(box_lengths),
        std::get<1>(box_lengths),
        std::get<2>(box_lengths)
    };

    // Convert Python dict to C++ regions vector
    std::vector<mlip::Region> regions;
    for (auto item : regions_dict) {
        mlip::Region region;
        region.name = item.first.cast<std::string>();
        auto bounds = item.second.cast<std::tuple<double, double>>();
        region.z_min = std::get<0>(bounds);
        region.z_max = std::get<1>(bounds);
        regions.push_back(region);
    }

    // Compute region-based MSD
    auto result = mlip::compute_msd_regions_from_positions(
        all_positions, atoms_per_frame, box, regions, unwrap_xy, unwrap_z
    );

    size_t n_lags = n_frames - 1;

    // Convert to Python dict of dicts
    py::dict output;
    for (const auto& region : regions) {
        const auto& msd_result = result.region_results.at(region.name);

        py::array_t<double> msd_planar(n_lags);
        py::array_t<double> msd_perp(n_lags);
        py::array_t<double> msd_total(n_lags);

        auto planar_buf = msd_planar.mutable_unchecked<1>();
        auto perp_buf = msd_perp.mutable_unchecked<1>();
        auto total_buf = msd_total.mutable_unchecked<1>();

        for (size_t i = 0; i < n_lags; ++i) {
            planar_buf(i) = msd_result.msd_planar[i];
            perp_buf(i) = msd_result.msd_perpendicular[i];
            total_buf(i) = msd_result.msd_total[i];
        }

        py::dict region_output;
        region_output["planar"] = msd_planar;
        region_output["perpendicular"] = msd_perp;
        region_output["total"] = msd_total;

        output[py::str(region.name)] = region_output;
    }

    return output;
}

/**
 * Python wrapper for MSD computation.
 *
 * @param positions_list  list of numpy arrays, each (N, 3)
 * @param box_lengths     tuple (Lx, Ly, Lz)
 * @param unwrap_xy       whether to unwrap x and y coordinates
 * @param unwrap_z        whether to unwrap z coordinate
 * @return                dict with 'planar', 'perpendicular', 'total' MSD arrays
 */
py::dict py_compute_msd(
    py::list positions_list,
    std::tuple<double, double, double> box_lengths,
    bool unwrap_xy,
    bool unwrap_z
) {
    size_t n_frames = positions_list.size();

    if (n_frames < 2) {
        throw std::runtime_error("Need at least 2 frames to compute MSD");
    }

    std::vector<const double*> all_positions;
    std::vector<size_t> atoms_per_frame;

    // Keep references to prevent garbage collection
    std::vector<py::array_t<double>> pos_arrays;

    for (size_t f = 0; f < n_frames; ++f) {
        auto pos_arr = positions_list[f].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        pos_arrays.push_back(pos_arr);

        auto pos_buf = pos_arr.request();

        if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
            throw std::runtime_error("Each position array must have shape (N, 3)");
        }

        all_positions.push_back(static_cast<const double*>(pos_buf.ptr));
        atoms_per_frame.push_back(pos_buf.shape[0]);
    }

    // Check all frames have same number of atoms
    for (size_t f = 1; f < n_frames; ++f) {
        if (atoms_per_frame[f] != atoms_per_frame[0]) {
            throw std::runtime_error("All frames must have the same number of atoms");
        }
    }

    std::array<double, 3> box = {
        std::get<0>(box_lengths),
        std::get<1>(box_lengths),
        std::get<2>(box_lengths)
    };

    // Compute MSD
    auto result = mlip::compute_msd_from_positions(
        all_positions, atoms_per_frame, box, unwrap_xy, unwrap_z
    );

    size_t n_lags = result.msd_planar.size();

    // Convert to numpy arrays
    py::array_t<double> msd_planar(n_lags);
    py::array_t<double> msd_perp(n_lags);
    py::array_t<double> msd_total(n_lags);

    auto planar_buf = msd_planar.mutable_unchecked<1>();
    auto perp_buf = msd_perp.mutable_unchecked<1>();
    auto total_buf = msd_total.mutable_unchecked<1>();

    for (size_t i = 0; i < n_lags; ++i) {
        planar_buf(i) = result.msd_planar[i];
        perp_buf(i) = result.msd_perpendicular[i];
        total_buf(i) = result.msd_total[i];
    }

    py::dict output;
    output["planar"] = msd_planar;
    output["perpendicular"] = msd_perp;
    output["total"] = msd_total;

    return output;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "MLIP trajectory analysis C++ core module";

    m.def("compute_density_histogram", &py_compute_density_histogram,
          py::arg("positions"),
          py::arg("types"),
          py::arg("box_lo"),
          py::arg("box_hi"),
          py::arg("axis"),
          py::arg("n_bins"),
          py::arg("n_types"),
          R"doc(
          Compute density histogram for a single frame.

          Parameters
          ----------
          positions : ndarray, shape (N, 3)
              Atom positions
          types : ndarray, shape (N,), dtype=int
              Atom type indices (0-indexed)
          box_lo : tuple
              Box lower bounds (xlo, ylo, zlo)
          box_hi : tuple
              Box upper bounds (xhi, yhi, zhi)
          axis : int
              Axis for density profile: 0=x, 1=y, 2=z
          n_bins : int
              Number of histogram bins
          n_types : int
              Number of unique atom types

          Returns
          -------
          ndarray, shape (n_types, n_bins)
              Histogram counts for each type
          )doc"
    );

    m.def("accumulate_density_frames", &py_accumulate_density_frames,
          py::arg("positions_list"),
          py::arg("types_list"),
          py::arg("box_lo"),
          py::arg("box_hi"),
          py::arg("axis"),
          py::arg("n_bins"),
          py::arg("n_types"),
          R"doc(
          Accumulate density histogram over multiple frames.

          Parameters
          ----------
          positions_list : list of ndarray
              List of position arrays, each shape (N_i, 3)
          types_list : list of ndarray
              List of type arrays, each shape (N_i,)
          box_lo : tuple
              Box lower bounds (xlo, ylo, zlo)
          box_hi : tuple
              Box upper bounds (xhi, yhi, zhi)
          axis : int
              Axis for density profile: 0=x, 1=y, 2=z
          n_bins : int
              Number of histogram bins
          n_types : int
              Number of unique atom types

          Returns
          -------
          ndarray, shape (n_types, n_bins)
              Accumulated histogram counts
          )doc"
    );

    m.def("compute_density_histogram_per_frame", &py_compute_density_histogram_per_frame,
          py::arg("positions_list"),
          py::arg("types_list"),
          py::arg("box_lo"),
          py::arg("box_hi"),
          py::arg("axis"),
          py::arg("n_bins"),
          py::arg("n_types"),
          R"doc(
          Compute density histogram for each frame separately.

          Used for principled error estimation methods (autocorrelation,
          Flyvbjerg-Petersen blocking) that require per-frame time series data.

          Parameters
          ----------
          positions_list : list of ndarray
              List of position arrays, each shape (N_i, 3)
          types_list : list of ndarray
              List of type arrays, each shape (N_i,)
          box_lo : tuple
              Box lower bounds (xlo, ylo, zlo)
          box_hi : tuple
              Box upper bounds (xhi, yhi, zhi)
          axis : int
              Axis for density profile: 0=x, 1=y, 2=z
          n_bins : int
              Number of histogram bins
          n_types : int
              Number of unique atom types

          Returns
          -------
          ndarray, shape (n_frames, n_types, n_bins)
              Per-frame histogram counts
          )doc"
    );

    m.def("accumulate_rdf_frames", &py_accumulate_rdf_frames,
          py::arg("positions_list"),
          py::arg("types_list"),
          py::arg("cell_vectors"),
          py::arg("rmin"),
          py::arg("rmax"),
          py::arg("n_bins"),
          py::arg("type1"),
          py::arg("type2"),
          R"doc(
          Accumulate RDF histogram over multiple frames.

          Parameters
          ----------
          positions_list : list of ndarray
              List of position arrays, each shape (N_i, 3)
          types_list : list of ndarray
              List of type arrays, each shape (N_i,)
          cell_vectors : ndarray, shape (3, 3)
              Cell vectors as rows (a, b, c)
          rmin : float
              Minimum distance for RDF
          rmax : float
              Maximum distance for RDF
          n_bins : int
              Number of histogram bins
          type1 : int
              First atom type (0-indexed, -1 for all)
          type2 : int
              Second atom type (0-indexed, -1 for all)

          Returns
          -------
          tuple
              (histogram, total_pairs, total_volume, n_frames)
          )doc"
    );

    m.def("accumulate_hbonds_frames", &py_accumulate_hbonds_frames,
          py::arg("positions_list"),
          py::arg("types_list"),
          py::arg("cell_vectors"),
          py::arg("donor_type"),
          py::arg("hydrogen_type"),
          py::arg("acceptor_type"),
          py::arg("d_a_cutoff"),
          py::arg("angle_cutoff"),
          py::arg("d_h_cutoff"),
          py::arg("bin_axis"),
          py::arg("box_lo"),
          py::arg("box_hi"),
          py::arg("n_bins"),
          R"doc(
          Detect hydrogen bonds over multiple frames.

          H-bond criteria:
          - D-A distance < d_a_cutoff
          - D-H...A angle > angle_cutoff

          Parameters
          ----------
          positions_list : list of ndarray
              List of position arrays, each shape (N_i, 3)
          types_list : list of ndarray
              List of type arrays, each shape (N_i,)
          cell_vectors : ndarray, shape (3, 3)
              Cell vectors as rows (a, b, c)
          donor_type : int
              Atom type index for donors (0-indexed)
          hydrogen_type : int
              Atom type index for hydrogens
          acceptor_type : int
              Atom type index for acceptors
          d_a_cutoff : float
              Donor-Acceptor distance cutoff (Angstroms)
          angle_cutoff : float
              D-H...A angle cutoff (degrees)
          d_h_cutoff : float
              Donor-Hydrogen bond distance (Angstroms)
          bin_axis : int
              Axis for spatial binning (-1 = no binning)
          box_lo : tuple
              Box lower bounds (xlo, ylo, zlo)
          box_hi : tuple
              Box upper bounds (xhi, yhi, zhi)
          n_bins : int
              Number of spatial bins

          Returns
          -------
          list of dict
              Per-frame results with 'total_hbonds', 'hbonds_per_bin', 'donors_per_bin'
          )doc"
    );

    m.def("detect_hbond_pairs_frames", &py_detect_hbond_pairs_frames,
          py::arg("positions_list"),
          py::arg("types_list"),
          py::arg("cell_vectors"),
          py::arg("donor_type"),
          py::arg("hydrogen_type"),
          py::arg("acceptor_type"),
          py::arg("d_a_cutoff"),
          py::arg("angle_cutoff"),
          py::arg("d_h_cutoff"),
          R"doc(
          Detect H-bond pairs in each frame (for lifetime analysis).

          Parameters
          ----------
          positions_list : list of ndarray
              List of position arrays, each shape (N_i, 3)
          types_list : list of ndarray
              List of type arrays, each shape (N_i,)
          cell_vectors : ndarray, shape (3, 3)
              Cell vectors
          donor_type : int
              Donor atom type (0-indexed)
          hydrogen_type : int
              Hydrogen atom type
          acceptor_type : int
              Acceptor atom type
          d_a_cutoff : float
              D-A distance cutoff
          angle_cutoff : float
              Angle cutoff (degrees)
          d_h_cutoff : float
              D-H bond distance

          Returns
          -------
          list of list of tuple
              Per-frame list of (donor_idx, acceptor_idx) pairs
          )doc"
    );

    m.def("compute_msd", &py_compute_msd,
          py::arg("positions_list"),
          py::arg("box_lengths"),
          py::arg("unwrap_xy") = true,
          py::arg("unwrap_z") = false,
          R"doc(
          Compute Mean Square Displacement for planar, perpendicular, and total motion.

          Parameters
          ----------
          positions_list : list of ndarray
              List of position arrays, each shape (N, 3)
          box_lengths : tuple
              Box dimensions (Lx, Ly, Lz)
          unwrap_xy : bool, optional
              Whether to unwrap x and y coordinates (default: True)
          unwrap_z : bool, optional
              Whether to unwrap z coordinate (default: False)

          Returns
          -------
          dict
              Dictionary with keys:
              - 'planar': MSD in x-y plane (ndarray)
              - 'perpendicular': MSD in z direction (ndarray)
              - 'total': Total 3D MSD (ndarray)
          )doc"
    );

    m.def("compute_msd_regions", &py_compute_msd_regions,
          py::arg("positions_list"),
          py::arg("box_lengths"),
          py::arg("regions"),
          py::arg("unwrap_xy") = true,
          py::arg("unwrap_z") = false,
          R"doc(
          Compute region-based Mean Square Displacement with endpoint checking.

          For each (t0, t0+dt) pair, only include atoms that are inside
          the region at BOTH t0 AND t0+dt.

          Parameters
          ----------
          positions_list : list of ndarray
              List of position arrays, each shape (N, 3)
          box_lengths : tuple
              Box dimensions (Lx, Ly, Lz)
          regions : dict
              Dictionary mapping region names to (z_min, z_max) tuples
          unwrap_xy : bool, optional
              Whether to unwrap x and y coordinates (default: True)
          unwrap_z : bool, optional
              Whether to unwrap z coordinate (default: False)

          Returns
          -------
          dict
              Dictionary mapping region names to MSD dicts.
              Each MSD dict has keys:
              - 'planar': MSD in x-y plane (ndarray)
              - 'perpendicular': MSD in z direction (ndarray)
              - 'total': Total 3D MSD (ndarray)
          )doc"
    );
}
