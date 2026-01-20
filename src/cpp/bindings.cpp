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

/**
 * Python wrapper for water molecule identification.
 *
 * @param positions      numpy array of shape (N, 3) with atom positions
 * @param element_types  numpy array of shape (N,) with element type indices (0-indexed)
 * @param o_type_idx     type index for oxygen atoms
 * @param h_type_idx     type index for hydrogen atoms
 * @param box_lengths    tuple (Lx, Ly, Lz) for PBC (0 = no PBC for that dimension)
 * @param o_h_cutoff     O-H bond distance cutoff in Angstroms
 * @return               tuple of (molecule_indices, n_molecules)
 */
py::tuple py_identify_water_molecules(
    py::array_t<double, py::array::c_style | py::array::forcecast> positions,
    py::array_t<int, py::array::c_style | py::array::forcecast> element_types,
    int o_type_idx,
    int h_type_idx,
    std::tuple<double, double, double> box_lengths,
    double o_h_cutoff
) {
    auto pos_buf = positions.request();
    auto type_buf = element_types.request();

    if (pos_buf.ndim != 2 || pos_buf.shape[1] != 3) {
        throw std::runtime_error("positions must have shape (N, 3)");
    }
    if (type_buf.ndim != 1) {
        throw std::runtime_error("element_types must be 1D array");
    }
    if (pos_buf.shape[0] != type_buf.shape[0]) {
        throw std::runtime_error("positions and element_types must have same length");
    }

    size_t n_atoms = pos_buf.shape[0];
    const double* pos_ptr = static_cast<const double*>(pos_buf.ptr);
    const int* type_ptr = static_cast<const int*>(type_buf.ptr);

    std::array<double, 3> box = {
        std::get<0>(box_lengths),
        std::get<1>(box_lengths),
        std::get<2>(box_lengths)
    };

    auto result = mlip::identify_water_molecules(
        pos_ptr, type_ptr, n_atoms, o_type_idx, h_type_idx, box, o_h_cutoff
    );

    // Convert to numpy array of shape (n_molecules, 3)
    py::array_t<int> indices({static_cast<py::ssize_t>(result.n_molecules), static_cast<py::ssize_t>(3)});
    auto idx_buf = indices.mutable_unchecked<2>();

    for (size_t i = 0; i < result.n_molecules; ++i) {
        idx_buf(i, 0) = result.molecule_indices[i][0];
        idx_buf(i, 1) = result.molecule_indices[i][1];
        idx_buf(i, 2) = result.molecule_indices[i][2];
    }

    return py::make_tuple(indices, static_cast<int>(result.n_molecules));
}

/**
 * Python wrapper for extracting water properties from multiple frames.
 *
 * @param positions_list   list of numpy arrays, each (N, 3)
 * @param molecule_indices numpy array of shape (n_molecules, 3) with [O, H1, H2] indices
 * @param box_lengths      tuple (Lx, Ly, Lz) for PBC (0 = no PBC for that dimension)
 * @return                 dict of numpy arrays with water properties
 */
py::dict py_extract_water_properties(
    py::list positions_list,
    py::array_t<int, py::array::c_style | py::array::forcecast> molecule_indices,
    std::tuple<double, double, double> box_lengths
) {
    size_t n_frames = positions_list.size();

    auto idx_buf = molecule_indices.request();
    if (idx_buf.ndim != 2 || idx_buf.shape[1] != 3) {
        throw std::runtime_error("molecule_indices must have shape (n_molecules, 3)");
    }

    size_t n_molecules = idx_buf.shape[0];
    const int* idx_ptr = static_cast<const int*>(idx_buf.ptr);

    // Convert indices to C++ format
    std::vector<std::array<int, 3>> mol_indices(n_molecules);
    for (size_t i = 0; i < n_molecules; ++i) {
        mol_indices[i] = {idx_ptr[i * 3], idx_ptr[i * 3 + 1], idx_ptr[i * 3 + 2]};
    }

    // Collect positions
    std::vector<const double*> all_positions;
    std::vector<py::array_t<double>> pos_arrays;  // Keep references

    for (size_t f = 0; f < n_frames; ++f) {
        auto pos_arr = positions_list[f].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        pos_arrays.push_back(pos_arr);
        all_positions.push_back(static_cast<const double*>(pos_arr.request().ptr));
    }

    std::array<double, 3> box = {
        std::get<0>(box_lengths),
        std::get<1>(box_lengths),
        std::get<2>(box_lengths)
    };

    // Extract properties
    auto all_molecules = mlip::extract_water_properties_multiframe(
        all_positions, mol_indices, n_molecules, n_frames, box
    );

    // Convert to numpy arrays: shape (n_frames, n_molecules, ...)
    py::array_t<double> O_positions({static_cast<py::ssize_t>(n_frames),
                                      static_cast<py::ssize_t>(n_molecules),
                                      static_cast<py::ssize_t>(3)});
    py::array_t<double> H1_positions({static_cast<py::ssize_t>(n_frames),
                                       static_cast<py::ssize_t>(n_molecules),
                                       static_cast<py::ssize_t>(3)});
    py::array_t<double> H2_positions({static_cast<py::ssize_t>(n_frames),
                                       static_cast<py::ssize_t>(n_molecules),
                                       static_cast<py::ssize_t>(3)});
    py::array_t<double> OH1_distances({static_cast<py::ssize_t>(n_frames),
                                        static_cast<py::ssize_t>(n_molecules)});
    py::array_t<double> OH2_distances({static_cast<py::ssize_t>(n_frames),
                                        static_cast<py::ssize_t>(n_molecules)});
    py::array_t<double> H1H2_distances({static_cast<py::ssize_t>(n_frames),
                                         static_cast<py::ssize_t>(n_molecules)});
    py::array_t<double> H1OH2_angles({static_cast<py::ssize_t>(n_frames),
                                       static_cast<py::ssize_t>(n_molecules)});

    auto O_buf = O_positions.mutable_unchecked<3>();
    auto H1_buf = H1_positions.mutable_unchecked<3>();
    auto H2_buf = H2_positions.mutable_unchecked<3>();
    auto OH1_buf = OH1_distances.mutable_unchecked<2>();
    auto OH2_buf = OH2_distances.mutable_unchecked<2>();
    auto H1H2_buf = H1H2_distances.mutable_unchecked<2>();
    auto angle_buf = H1OH2_angles.mutable_unchecked<2>();

    for (size_t f = 0; f < n_frames; ++f) {
        for (size_t m = 0; m < n_molecules; ++m) {
            const auto& mol = all_molecules[f][m];

            for (int j = 0; j < 3; ++j) {
                O_buf(f, m, j) = mol.O_position[j];
                H1_buf(f, m, j) = mol.H1_position[j];
                H2_buf(f, m, j) = mol.H2_position[j];
            }

            OH1_buf(f, m) = mol.OH1_distance;
            OH2_buf(f, m) = mol.OH2_distance;
            H1H2_buf(f, m) = mol.H1H2_distance;
            angle_buf(f, m) = mol.H1OH2_angle;
        }
    }

    py::dict result;
    result["O_positions"] = O_positions;
    result["H1_positions"] = H1_positions;
    result["H2_positions"] = H2_positions;
    result["OH1_distances"] = OH1_distances;
    result["OH2_distances"] = OH2_distances;
    result["H1H2_distances"] = H1H2_distances;
    result["H1OH2_angles"] = H1OH2_angles;

    return result;
}

/**
 * Python wrapper for hydrogen bond identification across multiple frames.
 *
 * @param positions_list    list of numpy arrays, each (N, 3)
 * @param water_indices     numpy array of shape (n_waters, 3) with [O, H1, H2] indices
 * @param box_lengths_list  list of tuples (Lx, Ly, Lz) for each frame
 * @param da_cutoff         Donor-Acceptor distance cutoff
 * @param angle_cutoff      Minimum D-H-A angle
 * @return                  list of dicts with H-bond properties per frame
 */
py::list py_identify_hbonds(
    py::list positions_list,
    py::array_t<int, py::array::c_style | py::array::forcecast> water_indices,
    py::list box_lengths_list,
    double da_cutoff,
    double angle_cutoff
) {
    size_t n_frames = positions_list.size();

    if (n_frames != box_lengths_list.size()) {
        throw std::runtime_error("positions_list and box_lengths_list must have same length");
    }

    auto idx_buf = water_indices.request();
    if (idx_buf.ndim != 2 || idx_buf.shape[1] != 3) {
        throw std::runtime_error("water_indices must have shape (n_waters, 3)");
    }

    size_t n_waters = idx_buf.shape[0];
    const int* idx_ptr = static_cast<const int*>(idx_buf.ptr);

    // Convert water indices to C++ format
    std::vector<std::array<int, 3>> water_idx_vec(n_waters);
    for (size_t i = 0; i < n_waters; ++i) {
        water_idx_vec[i] = {idx_ptr[i * 3], idx_ptr[i * 3 + 1], idx_ptr[i * 3 + 2]};
    }

    // Collect positions and box lengths
    std::vector<const double*> all_positions;
    std::vector<py::array_t<double>> pos_arrays;
    std::vector<std::array<double, 3>> all_box_lengths;

    for (size_t f = 0; f < n_frames; ++f) {
        auto pos_arr = positions_list[f].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        pos_arrays.push_back(pos_arr);
        all_positions.push_back(static_cast<const double*>(pos_arr.request().ptr));

        auto box_tuple = box_lengths_list[f].cast<std::tuple<double, double, double>>();
        all_box_lengths.push_back({
            std::get<0>(box_tuple),
            std::get<1>(box_tuple),
            std::get<2>(box_tuple)
        });
    }

    // Identify H-bonds
    auto all_hbonds = mlip::identify_hbonds_multiframe(
        all_positions, water_idx_vec, n_waters, n_frames,
        all_box_lengths, da_cutoff, angle_cutoff
    );

    // Convert to Python list of dicts
    py::list result;

    for (size_t f = 0; f < n_frames; ++f) {
        const auto& frame_hbonds = all_hbonds[f];
        size_t n_hbonds = frame_hbonds.size();

        // Create arrays for this frame
        py::array_t<int> donor_water_idx(n_hbonds);
        py::array_t<int> acceptor_water_idx(n_hbonds);
        py::array_t<int> donor_O_idx(n_hbonds);
        py::array_t<int> acceptor_O_idx(n_hbonds);
        py::array_t<int> H_idx(n_hbonds);
        py::array_t<double> donor_positions({static_cast<py::ssize_t>(n_hbonds), static_cast<py::ssize_t>(3)});
        py::array_t<double> acceptor_positions({static_cast<py::ssize_t>(n_hbonds), static_cast<py::ssize_t>(3)});
        py::array_t<double> H_positions({static_cast<py::ssize_t>(n_hbonds), static_cast<py::ssize_t>(3)});
        py::array_t<double> DA_distances(n_hbonds);
        py::array_t<double> HA_distances(n_hbonds);
        py::array_t<double> DHA_angles(n_hbonds);

        auto donor_water_buf = donor_water_idx.mutable_unchecked<1>();
        auto acceptor_water_buf = acceptor_water_idx.mutable_unchecked<1>();
        auto donor_O_buf = donor_O_idx.mutable_unchecked<1>();
        auto acceptor_O_buf = acceptor_O_idx.mutable_unchecked<1>();
        auto H_buf = H_idx.mutable_unchecked<1>();
        auto donor_pos_buf = donor_positions.mutable_unchecked<2>();
        auto acceptor_pos_buf = acceptor_positions.mutable_unchecked<2>();
        auto H_pos_buf = H_positions.mutable_unchecked<2>();
        auto DA_buf = DA_distances.mutable_unchecked<1>();
        auto HA_buf = HA_distances.mutable_unchecked<1>();
        auto DHA_buf = DHA_angles.mutable_unchecked<1>();

        for (size_t h = 0; h < n_hbonds; ++h) {
            const auto& hb = frame_hbonds[h];

            donor_water_buf(h) = hb.donor_water_idx;
            acceptor_water_buf(h) = hb.acceptor_water_idx;
            donor_O_buf(h) = hb.donor_O_idx;
            acceptor_O_buf(h) = hb.acceptor_O_idx;
            H_buf(h) = hb.H_idx;

            for (int j = 0; j < 3; ++j) {
                donor_pos_buf(h, j) = hb.donor_position[j];
                acceptor_pos_buf(h, j) = hb.acceptor_position[j];
                H_pos_buf(h, j) = hb.H_position[j];
            }

            DA_buf(h) = hb.DA_distance;
            HA_buf(h) = hb.HA_distance;
            DHA_buf(h) = hb.DHA_angle;
        }

        py::dict frame_result;
        frame_result["n_hbonds"] = static_cast<int>(n_hbonds);
        frame_result["donor_water_idx"] = donor_water_idx;
        frame_result["acceptor_water_idx"] = acceptor_water_idx;
        frame_result["donor_O_idx"] = donor_O_idx;
        frame_result["acceptor_O_idx"] = acceptor_O_idx;
        frame_result["H_idx"] = H_idx;
        frame_result["donor_positions"] = donor_positions;
        frame_result["acceptor_positions"] = acceptor_positions;
        frame_result["H_positions"] = H_positions;
        frame_result["DA_distances"] = DA_distances;
        frame_result["HA_distances"] = HA_distances;
        frame_result["DHA_angles"] = DHA_angles;

        result.append(frame_result);
    }

    return result;
}

/**
 * Python wrapper for water molecule orientation computation across multiple frames.
 *
 * @param positions_list    list of numpy arrays, each (N, 3)
 * @param water_indices     numpy array of shape (n_waters, 3) with [O, H1, H2] indices
 * @param box_lengths_list  list of tuples (Lx, Ly, Lz) for each frame
 * @param surface_normal    tuple (nx, ny, nz) for surface normal vector
 * @return                  list of dicts with orientation properties per frame
 */
py::list py_compute_orientations(
    py::list positions_list,
    py::array_t<int, py::array::c_style | py::array::forcecast> water_indices,
    py::list box_lengths_list,
    std::tuple<double, double, double> surface_normal
) {
    size_t n_frames = positions_list.size();

    if (n_frames != box_lengths_list.size()) {
        throw std::runtime_error("positions_list and box_lengths_list must have same length");
    }

    auto idx_buf = water_indices.request();
    if (idx_buf.ndim != 2 || idx_buf.shape[1] != 3) {
        throw std::runtime_error("water_indices must have shape (n_waters, 3)");
    }

    size_t n_waters = idx_buf.shape[0];
    const int* idx_ptr = static_cast<const int*>(idx_buf.ptr);

    // Convert water indices to C++ format
    std::vector<std::array<int, 3>> water_idx_vec(n_waters);
    for (size_t i = 0; i < n_waters; ++i) {
        water_idx_vec[i] = {idx_ptr[i * 3], idx_ptr[i * 3 + 1], idx_ptr[i * 3 + 2]};
    }

    // Collect positions and box lengths
    std::vector<const double*> all_positions;
    std::vector<py::array_t<double>> pos_arrays;
    std::vector<std::array<double, 3>> all_box_lengths;

    for (size_t f = 0; f < n_frames; ++f) {
        auto pos_arr = positions_list[f].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        pos_arrays.push_back(pos_arr);
        all_positions.push_back(static_cast<const double*>(pos_arr.request().ptr));

        auto box_tuple = box_lengths_list[f].cast<std::tuple<double, double, double>>();
        all_box_lengths.push_back({
            std::get<0>(box_tuple),
            std::get<1>(box_tuple),
            std::get<2>(box_tuple)
        });
    }

    std::array<double, 3> surf_norm = {
        std::get<0>(surface_normal),
        std::get<1>(surface_normal),
        std::get<2>(surface_normal)
    };

    // Compute orientations
    auto all_orientations = mlip::compute_orientations_multiframe(
        all_positions, water_idx_vec, n_waters, n_frames,
        all_box_lengths, surf_norm
    );

    // Convert to Python list of dicts
    py::list result;

    for (size_t f = 0; f < n_frames; ++f) {
        const auto& frame_orientations = all_orientations[f];
        size_t n_orient = frame_orientations.size();

        // Create arrays for this frame
        py::array_t<int> water_idx_arr(n_orient);
        py::array_t<double> cos_theta_1_arr(n_orient);
        py::array_t<double> cos_theta_2_arr(n_orient);
        py::array_t<double> cos_phi_arr(n_orient);
        py::array_t<double> O_z_arr(n_orient);

        auto water_idx_buf = water_idx_arr.mutable_unchecked<1>();
        auto cos_theta_1_buf = cos_theta_1_arr.mutable_unchecked<1>();
        auto cos_theta_2_buf = cos_theta_2_arr.mutable_unchecked<1>();
        auto cos_phi_buf = cos_phi_arr.mutable_unchecked<1>();
        auto O_z_buf = O_z_arr.mutable_unchecked<1>();

        for (size_t w = 0; w < n_orient; ++w) {
            const auto& orient = frame_orientations[w];

            water_idx_buf(w) = orient.water_idx;
            cos_theta_1_buf(w) = orient.cos_theta_1;
            cos_theta_2_buf(w) = orient.cos_theta_2;
            cos_phi_buf(w) = orient.cos_phi;
            O_z_buf(w) = orient.O_z;
        }

        py::dict frame_result;
        frame_result["n_waters"] = static_cast<int>(n_orient);
        frame_result["water_idx"] = water_idx_arr;
        frame_result["cos_theta_1"] = cos_theta_1_arr;
        frame_result["cos_theta_2"] = cos_theta_2_arr;
        frame_result["cos_phi"] = cos_phi_arr;
        frame_result["O_z"] = O_z_arr;

        result.append(frame_result);
    }

    return result;
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

    // Water molecule identification functions
    m.def("identify_water_molecules", &py_identify_water_molecules,
          py::arg("positions"),
          py::arg("element_types"),
          py::arg("o_type_idx"),
          py::arg("h_type_idx"),
          py::arg("box_lengths"),
          py::arg("o_h_cutoff") = 1.2,
          R"doc(
          Identify water molecules from positions and element types.

          Uses greedy nearest-neighbor assignment: for each oxygen atom,
          finds the two closest hydrogen atoms within the cutoff distance.
          Supports periodic boundary conditions via minimum image convention.

          Parameters
          ----------
          positions : ndarray, shape (N, 3)
              Atom positions
          element_types : ndarray, shape (N,), dtype=int
              Element type indices (0-indexed)
          o_type_idx : int
              Type index for oxygen atoms
          h_type_idx : int
              Type index for hydrogen atoms
          box_lengths : tuple
              Box dimensions (Lx, Ly, Lz) for PBC. Use 0 to disable PBC for a dimension.
          o_h_cutoff : float, optional
              O-H bond distance cutoff in Angstroms (default: 1.2)

          Returns
          -------
          tuple
              (molecule_indices, n_molecules)
              molecule_indices: ndarray of shape (n_molecules, 3) with [O, H1, H2] indices
          )doc"
    );

    m.def("extract_water_properties", &py_extract_water_properties,
          py::arg("positions_list"),
          py::arg("molecule_indices"),
          py::arg("box_lengths"),
          R"doc(
          Extract water molecule properties for multiple frames.

          Computes positions, distances, and angles for each water molecule
          in each frame. Uses OpenMP parallelization when available.
          Supports periodic boundary conditions via minimum image convention.

          Parameters
          ----------
          positions_list : list of ndarray
              List of position arrays, each shape (N, 3)
          molecule_indices : ndarray, shape (n_molecules, 3), dtype=int
              Water molecule indices [O, H1, H2] from identify_water_molecules
          box_lengths : tuple
              Box dimensions (Lx, Ly, Lz) for PBC. Use 0 to disable PBC for a dimension.

          Returns
          -------
          dict
              Dictionary with keys:
              - 'O_positions': shape (n_frames, n_molecules, 3)
              - 'H1_positions': shape (n_frames, n_molecules, 3)
              - 'H2_positions': shape (n_frames, n_molecules, 3)
              - 'OH1_distances': shape (n_frames, n_molecules)
              - 'OH2_distances': shape (n_frames, n_molecules)
              - 'H1H2_distances': shape (n_frames, n_molecules)
              - 'H1OH2_angles': shape (n_frames, n_molecules) in degrees
          )doc"
    );

    // Hydrogen bond identification functions
    m.def("identify_hbonds", &py_identify_hbonds,
          py::arg("positions_list"),
          py::arg("water_indices"),
          py::arg("box_lengths_list"),
          py::arg("da_cutoff") = 3.5,
          py::arg("angle_cutoff") = 150.0,
          R"doc(
          Identify hydrogen bonds between water molecules across multiple frames.

          Uses geometric criteria:
          - Donor-Acceptor distance < da_cutoff
          - D-H-A angle > angle_cutoff

          Parameters
          ----------
          positions_list : list of ndarray
              List of position arrays, each shape (N, 3)
          water_indices : ndarray, shape (n_waters, 3), dtype=int
              Water molecule indices [O, H1, H2] from identify_water_molecules
          box_lengths_list : list of tuple
              Box dimensions (Lx, Ly, Lz) for each frame (supports NPT)
          da_cutoff : float, optional
              Donor-Acceptor distance cutoff in Angstroms (default: 3.5)
          angle_cutoff : float, optional
              Minimum D-H-A angle in degrees (default: 150.0)

          Returns
          -------
          list of dict
              List of dictionaries (one per frame), each containing:
              - 'n_hbonds': int, number of H-bonds in frame
              - 'donor_water_idx': ndarray, donor water molecule indices
              - 'acceptor_water_idx': ndarray, acceptor water molecule indices
              - 'donor_O_idx': ndarray, donor oxygen atom indices
              - 'acceptor_O_idx': ndarray, acceptor oxygen atom indices
              - 'H_idx': ndarray, bridging hydrogen atom indices
              - 'donor_positions': ndarray (n_hbonds, 3), donor oxygen positions
              - 'acceptor_positions': ndarray (n_hbonds, 3), acceptor oxygen positions
              - 'H_positions': ndarray (n_hbonds, 3), hydrogen positions
              - 'DA_distances': ndarray, donor-acceptor distances
              - 'HA_distances': ndarray, hydrogen-acceptor distances
              - 'DHA_angles': ndarray, D-H-A angles in degrees
          )doc"
    );

    // Water orientation analysis functions
    m.def("compute_orientations", &py_compute_orientations,
          py::arg("positions_list"),
          py::arg("water_indices"),
          py::arg("box_lengths_list"),
          py::arg("surface_normal") = std::make_tuple(0.0, 0.0, 1.0),
          R"doc(
          Compute water molecule orientation angles relative to a surface normal.

          Calculates cos(theta) for O-H bond orientations and cos(phi) for
          dipole orientation across multiple frames.

          Parameters
          ----------
          positions_list : list of ndarray
              List of position arrays, each shape (N, 3)
          water_indices : ndarray, shape (n_waters, 3), dtype=int
              Water molecule indices [O, H1, H2] from identify_water_molecules
          box_lengths_list : list of tuple
              Box dimensions (Lx, Ly, Lz) for each frame (supports NPT)
          surface_normal : tuple, optional
              Surface normal vector (nx, ny, nz), default: (0, 0, 1) for +z

          Returns
          -------
          list of dict
              List of dictionaries (one per frame), each containing:
              - 'n_waters': int, number of water molecules
              - 'water_idx': ndarray, water molecule indices
              - 'cos_theta_1': ndarray, rOH1 · n (O->H1 dot surface normal)
              - 'cos_theta_2': ndarray, rOH2 · n (O->H2 dot surface normal)
              - 'cos_phi': ndarray, dipole · n (H_mid->O dot surface normal)
              - 'O_z': ndarray, oxygen z-coordinates for region assignment
          )doc"
    );
}
