// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef RAKAU_BENCHMARKS_COMMON_HPP
#define RAKAU_BENCHMARKS_COMMON_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <rakau/detail/simple_timer.hpp>
#include <rakau/tree.hpp>

namespace rakau_benchmark
{

inline thread_local std::mt19937 rng;

// See: http://www.artcompsci.org/kali/vol/plummer/ch03.html
template <typename F>
inline std::vector<F> get_plummer_sphere(std::size_t n, F a, F size, bool parallel)
{
    rakau::simple_timer st("plummer init");
    if (!std::isfinite(a) || a <= F(0)) {
        throw std::invalid_argument("The Plummer 'a' parameter must be finite and positive, but it is "
                                    + std::to_string(a) + " instead");
    }
    if (!std::isfinite(size) || size < F(0)) {
        throw std::invalid_argument("The Plummer 'size' parameter must be finite and non-negative, but it is "
                                    + std::to_string(size) + " instead");
    }
    std::vector<F> retval(n * 4u);
    // Compute the limit for the coordinates based on the input box size. If size is zero, there is not limit.
    const auto size_limit = (size > F(0)) ? (size / F(2) - size / F(100)) : std::numeric_limits<F>::infinity();
    // Small helper to check that the x, y, z coords are within the limit computed above.
    auto check_bounds = [size_limit](F x, F y, F z) {
        return x >= -size_limit && x < size_limit && y >= -size_limit && y < size_limit && z >= -size_limit
               && z < size_limit;
    };
    if (parallel) {
        tbb::parallel_for(tbb::blocked_range<std::size_t>(0u, n), [&retval, a, n, check_bounds](const auto &range) {
            // Uniform [0, 1) dist.
            std::uniform_real_distribution<F> udist(F(0), F(1));
            // Uniform [0.1, 1.9) dist (average mass of 1).
            std::uniform_real_distribution<F> mdist(F(0.1), F(1.9));
            // Init the thread-local rng with the first index in the range.
            rng.seed(range.begin());
            for (auto i = range.begin(); i != range.end();) {
                retval[i] = mdist(rng);
                // Generate a random radius.
                // NOTE: reject r if it gets a non-finite value by chance.
                F r;
                do {
                    r = a / std::sqrt(std::pow(udist(rng), F(-2) / F(3)) - F(1));
                } while (!std::isfinite(r));
                // Generate random u, v for sphere picking.
                const F u = udist(rng), v = udist(rng);
                // Longitude.
                const F lon = std::clamp(F(2) * boost::math::constants::pi<F>() * u, F(0),
                                         F(2) * boost::math::constants::pi<F>());
                // Colatitude.
                const F colat = std::acos(std::clamp(F(2) * v - F(1), F(-1), F(1)));
                // Compute x, y, z and assign them if they are within the bounds.
                const auto x = r * std::cos(lon) * std::sin(colat), y = r * std::sin(lon) * std::sin(colat),
                           z = r * std::cos(colat);
                if (check_bounds(x, y, z)) {
                    retval[n + i] = x;
                    retval[2u * n + i] = y;
                    retval[3u * n + i] = z;
                    ++i;
                }
            }
        });
        return retval;
    } else {
        // Uniform [0, 1) dist.
        std::uniform_real_distribution<F> udist(F(0), F(1));
        // Uniform [0.1, 1.9) dist.
        std::uniform_real_distribution<F> mdist(F(0.1), F(1.9));
        // Average mass of 1.
        std::generate(retval.data(), retval.data() + n, [&mdist]() { return mdist(rng); });
        for (std::size_t i = 0; i < n;) {
            // Generate a random radius.
            F r;
            do {
                r = a / std::sqrt(std::pow(udist(rng), F(-2) / F(3)) - F(1));
            } while (!std::isfinite(r));
            // Generate random u, v for sphere picking.
            const F u = udist(rng), v = udist(rng);
            // Longitude.
            const F lon
                = std::clamp(F(2) * boost::math::constants::pi<F>() * u, F(0), F(2) * boost::math::constants::pi<F>());
            // Colatitude.
            const F colat = std::acos(std::clamp(F(2) * v - F(1), F(-1), F(1)));
            // Compute x, y, z.
            const auto x = r * std::cos(lon) * std::sin(colat), y = r * std::sin(lon) * std::sin(colat),
                       z = r * std::cos(colat);
            if (check_bounds(x, y, z)) {
                *(retval.data() + n + i) = x;
                *(retval.data() + 2 * n + i) = y;
                *(retval.data() + 3 * n + i) = z;
                ++i;
            }
        }
        return retval;
    }
}

template <std::size_t D, typename F>
inline std::vector<F> get_uniform_particles(std::size_t n, F size)
{
    rakau::simple_timer st("uniform init");
    std::vector<F> retval(n * (D + 1u));
    // Mass.
    std::uniform_real_distribution<F> mdist(F(0), F(1));
    std::generate(retval.begin(), retval.begin() + n, [&mdist]() { return mdist(rng); });
    // Positions.
    const auto size_limit = size / F(2) - F(0.01);
    std::uniform_real_distribution<F> rdist(-size_limit, size_limit);
    std::generate(retval.begin() + n, retval.end(), [&rdist]() { return rdist(rng); });
    return retval;
}

inline auto parse_accpot_benchmark_options(int argc, char **argv)
{
    namespace po = boost::program_options;

    unsigned long nparts, idx;
    unsigned max_leaf_n, ncrit, nthreads;
    double bsize, a, mac_value;
    bool parinit = false;
    std::vector<double> split;
    std::string fp_type, mac_type;

    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")(
        "nparts", po::value<unsigned long>(&nparts)->default_value(1'000'000ul), "number of particles")(
        "idx", po::value<unsigned long>(&idx)->default_value(0ul), "index of the particle to test against")(
        "max_leaf_n", po::value<unsigned>(&max_leaf_n)->default_value(rakau::default_max_leaf_n),
        "max number of particles in a leaf node")("ncrit",
                                                  po::value<unsigned>(&ncrit)->default_value(rakau::default_ncrit),
                                                  "maximum number of particles in a critical node")(
        "a", po::value<double>(&a)->default_value(1.), "Plummer core radius")(
        "bsize", po::value<double>(&bsize)->default_value(0.),
        "size of the domain (if 0, it is automatically deduced)")("nthreads",
                                                                  po::value<unsigned>(&nthreads)->default_value(0u),
                                                                  "number of threads to use (0 for auto-detection)")(
        "mac_value", po::value<double>(&mac_value)->default_value(0.75),
        "MAC value")("parinit", "parallel nondeterministic initialisation of the particle distribution")(
        "split", po::value<std::vector<double>>()->multitoken(), "split vector for heterogeneous computations")(
        "fp_type", po::value<std::string>(&fp_type)->default_value("float"),
        "floating-point type to use in the computations")(
        "mac_type", po::value<std::string>(&mac_type)->default_value("bh"), "MAC type");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        std::exit(0);
    }

    if (nparts == 0u) {
        throw std::invalid_argument("The number of particles cannot be zero");
    }

    if (idx >= nparts) {
        throw std::invalid_argument(
            "The index of the particle to test against needs to be less-than the total number of particles ("
            + std::to_string(nparts) + ")");
    }

    if (vm.count("parinit")) {
        parinit = true;
    }

    if (vm.count("split")) {
        split = vm["split"].as<std::vector<double>>();
    }

    if (fp_type != "float" && fp_type != "double") {
        throw std::invalid_argument("Only the 'float' and 'double' floating-point types are supported, but the type '"
                                    + fp_type + "' was specified instead");
    }

    if (mac_type != "bh" && mac_type != "bh_geom") {
        throw std::invalid_argument("'" + mac_type + "' is not a valid MAC type");
    }

    return std::tuple{nparts,
                      idx,
                      max_leaf_n,
                      ncrit,
                      nthreads,
                      bsize,
                      a,
                      mac_value,
                      parinit,
                      std::move(split),
                      std::move(fp_type),
                      std::move(mac_type)};
}

} // namespace rakau_benchmark

#endif
