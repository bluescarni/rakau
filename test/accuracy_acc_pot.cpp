// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <rakau/tree.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <rakau/config.hpp>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau_test;

using fp_types = std::tuple<float, double>;

static std::mt19937 rng(0);

static const std::vector<double> split =
#if defined(RAKAU_WITH_ROCM)
    {0.5, 0.5}
#else
    {}
#endif
;

TEST_CASE("acceleration/potential accuracy ordered")
{
    tuple_for_each(fp_types{}, [](auto x) {
        using fp_type = decltype(x);
        constexpr auto theta = static_cast<fp_type>(.001), bsize = static_cast<fp_type>(1);
        auto sizes = {10u, 100u, 1000u, 2000u};
        auto max_leaf_ns = {1u, 2u, 8u, 16u};
        auto ncrits = {1u, 16u, 128u, 256u};
        std::array<std::vector<fp_type>, 4> accpots;
        fp_type tot_max_x_diff(0), tot_max_y_diff(0), tot_max_z_diff(0), tot_max_pot_diff(0);
        for (auto s : sizes) {
            auto parts = get_uniform_particles<3>(s, bsize, rng);
            for (auto max_leaf_n : max_leaf_ns) {
                for (auto ncrit : ncrits) {
                    std::vector<fp_type> x_diff, y_diff, z_diff, pot_diff;
                    octree<fp_type> t(
                        {parts.begin() + s, parts.begin() + 2u * s, parts.begin() + 3u * s, parts.begin()}, s,
                        kwargs::box_size = bsize, kwargs::max_leaf_n = max_leaf_n, kwargs::ncrit = ncrit);
                    t.accs_pots_o(accpots, theta);
                    // Check that all accelerations/potentials are finite.
                    REQUIRE(std::all_of(accpots[0].begin(), accpots[0].end(), [](auto c) { return std::isfinite(c); }));
                    REQUIRE(std::all_of(accpots[1].begin(), accpots[1].end(), [](auto c) { return std::isfinite(c); }));
                    REQUIRE(std::all_of(accpots[2].begin(), accpots[2].end(), [](auto c) { return std::isfinite(c); }));
                    REQUIRE(std::all_of(accpots[3].begin(), accpots[3].end(), [](auto c) { return std::isfinite(c); }));
                    for (auto i = 0u; i < s; ++i) {
                        auto eacc = t.exact_acc_pot_o(i);
                        x_diff.emplace_back(std::abs((eacc[0] - accpots[0][i]) / eacc[0]));
                        y_diff.emplace_back(std::abs((eacc[1] - accpots[1][i]) / eacc[1]));
                        z_diff.emplace_back(std::abs((eacc[2] - accpots[2][i]) / eacc[2]));
                        pot_diff.emplace_back(std::abs((eacc[3] - accpots[3][i]) / eacc[3]));
                    }
                    std::cout << "Results for size=" << s << ", max_leaf_n=" << max_leaf_n << ", ncrit=" << ncrit
                              << ".\n=========\n";
                    const auto local_max_x_diff = *std::max_element(x_diff.begin(), x_diff.end()),
                               local_max_y_diff = *std::max_element(y_diff.begin(), y_diff.end()),
                               local_max_z_diff = *std::max_element(z_diff.begin(), z_diff.end()),
                               local_max_pot_diff = *std::max_element(pot_diff.begin(), pot_diff.end());
                    std::cout << "max_x_diff=" << local_max_x_diff << '\n';
                    std::cout << "max_y_diff=" << local_max_y_diff << '\n';
                    std::cout << "max_z_diff=" << local_max_z_diff << '\n';
                    std::cout << "max_pot_diff=" << local_max_pot_diff << '\n';
                    std::cout << "average_x_diff="
                              << (std::accumulate(x_diff.begin(), x_diff.end(), fp_type(0)) / fp_type(s)) << '\n';
                    std::cout << "average_y_diff="
                              << (std::accumulate(y_diff.begin(), y_diff.end(), fp_type(0)) / fp_type(s)) << '\n';
                    std::cout << "average_z_diff="
                              << (std::accumulate(z_diff.begin(), z_diff.end(), fp_type(0)) / fp_type(s)) << '\n';
                    std::cout << "average_pot_diff="
                              << (std::accumulate(pot_diff.begin(), pot_diff.end(), fp_type(0)) / fp_type(s)) << '\n';
                    std::cout << "median_x_diff=" << median(x_diff) << '\n';
                    std::cout << "median_y_diff=" << median(y_diff) << '\n';
                    std::cout << "median_z_diff=" << median(z_diff) << '\n';
                    std::cout << "median_pot_diff=" << median(pot_diff) << '\n';
                    std::cout << "=========\n\n";
                    tot_max_x_diff = std::max(local_max_x_diff, tot_max_x_diff);
                    tot_max_y_diff = std::max(local_max_y_diff, tot_max_y_diff);
                    tot_max_z_diff = std::max(local_max_z_diff, tot_max_z_diff);
                    tot_max_pot_diff = std::max(local_max_pot_diff, tot_max_pot_diff);
                }
            }
        }
        std::cout << "\n\n\ntot_max_x_diff=" << tot_max_x_diff << '\n';
        std::cout << "tot_max_y_diff=" << tot_max_y_diff << '\n';
        std::cout << "tot_max_z_diff=" << tot_max_z_diff << '\n';
        std::cout << "tot_max_pot_diff=" << tot_max_pot_diff << "\n\n\n";
        if constexpr (std::is_same_v<fp_type, double> && std::numeric_limits<fp_type>::is_iec559) {
            // These numbers are, of course, totally arbitrary, based
            // on the fact that 'double' is actually double-precision,
            // and derived experimentally.
            REQUIRE(tot_max_x_diff < fp_type(1E-10));
            REQUIRE(tot_max_y_diff < fp_type(1E-10));
            REQUIRE(tot_max_z_diff < fp_type(1E-10));
            REQUIRE(tot_max_pot_diff < fp_type(1E-10));
        }
    });
}

TEST_CASE("acceleration/potential accuracy unordered")
{
    tuple_for_each(fp_types{}, [](auto x) {
        using fp_type = decltype(x);
        constexpr auto theta = static_cast<fp_type>(.001), bsize = static_cast<fp_type>(1);
        auto sizes = {10u, 100u, 1000u, 2000u};
        auto max_leaf_ns = {1u, 2u, 8u, 16u};
        auto ncrits = {1u, 16u, 128u, 256u};
        std::array<std::vector<fp_type>, 4> accpots;
        fp_type tot_max_x_diff(0), tot_max_y_diff(0), tot_max_z_diff(0), tot_max_pot_diff(0);
        for (auto s : sizes) {
            auto parts = get_uniform_particles<3>(s, bsize, rng);
            for (auto max_leaf_n : max_leaf_ns) {
                for (auto ncrit : ncrits) {
                    std::vector<fp_type> x_diff, y_diff, z_diff, pot_diff;
                    octree<fp_type> t(
                        {parts.begin() + s, parts.begin() + 2u * s, parts.begin() + 3u * s, parts.begin()}, s,
                        kwargs::box_size = bsize, kwargs::max_leaf_n = max_leaf_n, kwargs::ncrit = ncrit);
                    t.accs_pots_u(accpots, theta, kwargs::split = split);
                    // Check that all accelerations/potentials are finite.
                    REQUIRE(std::all_of(accpots[0].begin(), accpots[0].end(), [](auto c) { return std::isfinite(c); }));
                    REQUIRE(std::all_of(accpots[1].begin(), accpots[1].end(), [](auto c) { return std::isfinite(c); }));
                    REQUIRE(std::all_of(accpots[2].begin(), accpots[2].end(), [](auto c) { return std::isfinite(c); }));
                    REQUIRE(std::all_of(accpots[3].begin(), accpots[3].end(), [](auto c) { return std::isfinite(c); }));
                    for (auto i = 0u; i < s; ++i) {
                        auto eacc = t.exact_acc_pot_u(i);
                        x_diff.emplace_back(std::abs((eacc[0] - accpots[0][i]) / eacc[0]));
                        y_diff.emplace_back(std::abs((eacc[1] - accpots[1][i]) / eacc[1]));
                        z_diff.emplace_back(std::abs((eacc[2] - accpots[2][i]) / eacc[2]));
                        pot_diff.emplace_back(std::abs((eacc[3] - accpots[3][i]) / eacc[3]));
                    }
                    std::cout << "Results for size=" << s << ", max_leaf_n=" << max_leaf_n << ", ncrit=" << ncrit
                              << ".\n=========\n";
                    const auto local_max_x_diff = *std::max_element(x_diff.begin(), x_diff.end()),
                               local_max_y_diff = *std::max_element(y_diff.begin(), y_diff.end()),
                               local_max_z_diff = *std::max_element(z_diff.begin(), z_diff.end()),
                               local_max_pot_diff = *std::max_element(pot_diff.begin(), pot_diff.end());
                    std::cout << "max_x_diff=" << local_max_x_diff << '\n';
                    std::cout << "max_y_diff=" << local_max_y_diff << '\n';
                    std::cout << "max_z_diff=" << local_max_z_diff << '\n';
                    std::cout << "max_pot_diff=" << local_max_pot_diff << '\n';
                    std::cout << "average_x_diff="
                              << (std::accumulate(x_diff.begin(), x_diff.end(), fp_type(0)) / fp_type(s)) << '\n';
                    std::cout << "average_y_diff="
                              << (std::accumulate(y_diff.begin(), y_diff.end(), fp_type(0)) / fp_type(s)) << '\n';
                    std::cout << "average_z_diff="
                              << (std::accumulate(z_diff.begin(), z_diff.end(), fp_type(0)) / fp_type(s)) << '\n';
                    std::cout << "average_pot_diff="
                              << (std::accumulate(pot_diff.begin(), pot_diff.end(), fp_type(0)) / fp_type(s)) << '\n';
                    std::cout << "median_x_diff=" << median(x_diff) << '\n';
                    std::cout << "median_y_diff=" << median(y_diff) << '\n';
                    std::cout << "median_z_diff=" << median(z_diff) << '\n';
                    std::cout << "median_pot_diff=" << median(pot_diff) << '\n';
                    std::cout << "=========\n\n";
                    tot_max_x_diff = std::max(local_max_x_diff, tot_max_x_diff);
                    tot_max_y_diff = std::max(local_max_y_diff, tot_max_y_diff);
                    tot_max_z_diff = std::max(local_max_z_diff, tot_max_z_diff);
                    tot_max_pot_diff = std::max(local_max_pot_diff, tot_max_pot_diff);
                }
            }
        }
        std::cout << "\n\n\ntot_max_x_diff=" << tot_max_x_diff << '\n';
        std::cout << "tot_max_y_diff=" << tot_max_y_diff << '\n';
        std::cout << "tot_max_z_diff=" << tot_max_z_diff << '\n';
        std::cout << "tot_max_pot_diff=" << tot_max_pot_diff << "\n\n\n";
        if constexpr (std::is_same_v<fp_type, double> && std::numeric_limits<fp_type>::is_iec559) {
            // These numbers are, of course, totally arbitrary, based
            // on the fact that 'double' is actually double-precision,
            // and derived experimentally.
            REQUIRE(tot_max_x_diff < fp_type(1E-10));
            REQUIRE(tot_max_y_diff < fp_type(1E-10));
            REQUIRE(tot_max_z_diff < fp_type(1E-10));
            REQUIRE(tot_max_pot_diff < fp_type(1E-10));
        }
    });
}
