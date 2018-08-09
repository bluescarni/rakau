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

#include <initializer_list>
// #include <iterator>
// #include <random>
#include <tuple>
// #include <type_traits>
// #include <utility>
#include <vector>

// #include "test_utils.hpp"

using namespace rakau;
// using namespace rakau_test;

using fp_types = std::tuple<float, double>;

// static std::mt19937 rng;

TEST_CASE("softening")
{
    tuple_for_each(fp_types{}, [](auto x) {
        using fp_type = decltype(x);
        // Set up a domain with 2 overlapping particles (indices 1 and 2).
        fp_type x_coords[] = {0, 1, 1, 3}, y_coords[] = {-4, -5, -5, -7}, z_coords[] = {4, 5, 5, 1},
                masses[] = {1, 1, 1, 1};
        octree<fp_type> t(0, {x_coords, y_coords, z_coords, masses}, 4, 1, 1);
    });
}
