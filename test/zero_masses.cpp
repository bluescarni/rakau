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
#include <vector>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau_test;

static std::mt19937 rng;

TEST_CASE("zero masses")
{
    using tree_t = octree<double>;
    constexpr std::size_t s = 10000u;
    std::array<std::vector<double>, 3> accs;
    std::vector<double> pots;
    std::array<std::vector<double>, 4> accs_pots;
    auto parts = get_uniform_particles<3>(s, 10., rng);
    // Zero out the masses.
    std::fill(parts.data(), parts.data() + s, 0.);
    tree_t t({parts.begin() + s, parts.begin() + 2u * s, parts.begin() + 3u * s, parts.begin()}, s,
             kwargs::box_size = 10.);
    t.accs_u(accs, 0.75);
    REQUIRE(std::all_of(accs[0].begin(), accs[0].end(), [](auto x) { return std::isfinite(x) && x == 0.; }));
    REQUIRE(std::all_of(accs[1].begin(), accs[1].end(), [](auto x) { return std::isfinite(x) && x == 0.; }));
    REQUIRE(std::all_of(accs[2].begin(), accs[2].end(), [](auto x) { return std::isfinite(x) && x == 0.; }));
    t.accs_o(accs, 0.75);
    REQUIRE(std::all_of(accs[0].begin(), accs[0].end(), [](auto x) { return std::isfinite(x) && x == 0.; }));
    REQUIRE(std::all_of(accs[1].begin(), accs[1].end(), [](auto x) { return std::isfinite(x) && x == 0.; }));
    REQUIRE(std::all_of(accs[2].begin(), accs[2].end(), [](auto x) { return std::isfinite(x) && x == 0.; }));
    t.pots_u(pots, 0.75);
    REQUIRE(std::all_of(pots.begin(), pots.end(), [](auto x) { return std::isfinite(x) && x == 0.; }));
    t.pots_o(pots, 0.75);
    REQUIRE(std::all_of(pots.begin(), pots.end(), [](auto x) { return std::isfinite(x) && x == 0.; }));
    t.accs_pots_u(accs_pots, 0.75);
    for (std::size_t j = 0; j < 4; ++j) {
        REQUIRE(
            std::all_of(accs_pots[j].begin(), accs_pots[j].end(), [](auto x) { return std::isfinite(x) && x == 0.; }));
    }
    t.accs_pots_o(accs_pots, 0.75);
    for (std::size_t j = 0; j < 4; ++j) {
        REQUIRE(
            std::all_of(accs_pots[j].begin(), accs_pots[j].end(), [](auto x) { return std::isfinite(x) && x == 0.; }));
    }
}
