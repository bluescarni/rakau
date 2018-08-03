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

#include <cstddef>
#include <initializer_list>
#include <tuple>

using namespace rakau;

using fp_types = std::tuple<float, double>;

TEST_CASE("automatic box size")
{
    tuple_for_each(fp_types{}, [](auto x) {
        using fp_type = decltype(x);
        {
            fp_type x_coords[] = {0, 1, 2, 3}, y_coords[] = {-4, -5, -6, -7}, z_coords[] = {4, 5, 3, 1},
                    masses[] = {1, 1, 1, 1};
            octree<fp_type> t(0, {x_coords, y_coords, z_coords, masses}, 4, 1, 1);
            REQUIRE(t.box_size_deduced());
            REQUIRE(t.get_box_size() == 14 + fp_type(0.7));
            t.update_particles_u([](const auto &its) {
                for (std::size_t i = 0; i < 4u; ++i) {
                    for (std::size_t j = 0; j < 3u; ++j) {
                        its[j][i] *= 2;
                    }
                }
            });
            REQUIRE(t.box_size_deduced());
            REQUIRE(t.get_box_size() == 28 + fp_type(1.4));
        }
    });
}
