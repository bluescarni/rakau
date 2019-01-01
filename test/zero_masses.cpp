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
#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau_test;

using fp_types = std::tuple<float, double>;
using macs = std::tuple<std::integral_constant<mac, mac::bh>, std::integral_constant<mac, mac::bh_geom>>;

static std::mt19937 rng;

TEST_CASE("zero masses")
{
    tuple_for_each(macs{}, [](auto mac_type) {
        tuple_for_each(fp_types{}, [](auto x) {
            using fp_type = decltype(x);
            using tree_t = octree<fp_type, decltype(mac_type)::value>;

            constexpr auto theta = fp_type(0.75);
            constexpr auto bsize = fp_type(10);
            constexpr std::size_t s = 10000u;

            std::array<std::vector<fp_type>, 3> accs;
            std::vector<fp_type> pots;
            std::array<std::vector<fp_type>, 4> accs_pots;
            auto parts = get_uniform_particles<3>(s, bsize, rng);
            // Zero out the masses.
            std::fill(parts.data(), parts.data() + s, fp_type(0));
            tree_t t({parts.begin() + s, parts.begin() + 2u * s, parts.begin() + 3u * s, parts.begin()}, s,
                     kwargs::box_size = bsize);
            t.accs_u(accs, theta);
            REQUIRE(std::all_of(accs[0].begin(), accs[0].end(), [](auto x) { return std::isfinite(x) && x == 0; }));
            REQUIRE(std::all_of(accs[1].begin(), accs[1].end(), [](auto x) { return std::isfinite(x) && x == 0; }));
            REQUIRE(std::all_of(accs[2].begin(), accs[2].end(), [](auto x) { return std::isfinite(x) && x == 0; }));
            t.accs_o(accs, theta);
            REQUIRE(std::all_of(accs[0].begin(), accs[0].end(), [](auto x) { return std::isfinite(x) && x == 0; }));
            REQUIRE(std::all_of(accs[1].begin(), accs[1].end(), [](auto x) { return std::isfinite(x) && x == 0; }));
            REQUIRE(std::all_of(accs[2].begin(), accs[2].end(), [](auto x) { return std::isfinite(x) && x == 0; }));
            t.pots_u(pots, theta);
            REQUIRE(std::all_of(pots.begin(), pots.end(), [](auto x) { return std::isfinite(x) && x == 0; }));
            t.pots_o(pots, theta);
            REQUIRE(std::all_of(pots.begin(), pots.end(), [](auto x) { return std::isfinite(x) && x == 0; }));
            t.accs_pots_u(accs_pots, theta);
            for (std::size_t j = 0; j < 4; ++j) {
                REQUIRE(std::all_of(accs_pots[j].begin(), accs_pots[j].end(),
                                    [](auto x) { return std::isfinite(x) && x == 0; }));
            }
            t.accs_pots_o(accs_pots, theta);
            for (std::size_t j = 0; j < 4; ++j) {
                REQUIRE(std::all_of(accs_pots[j].begin(), accs_pots[j].end(),
                                    [](auto x) { return std::isfinite(x) && x == 0; }));
            }
        });
    });
}
