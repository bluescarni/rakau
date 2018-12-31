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
#include <type_traits>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau::kwargs;
using namespace rakau_test;

using fp_types = std::tuple<float, double>;
using macs = std::tuple<std::integral_constant<mac, mac::bh>, std::integral_constant<mac, mac::bh_geom>>;

TEST_CASE("automatic box size")
{
    tuple_for_each(macs{}, [](auto mac_type) {
        tuple_for_each(fp_types{}, [](auto x) {
            using fp_type = decltype(x);
            {
                fp_type x_coords[] = {0, 1, 2, 3}, y_coords[] = {-4, -5, -6, -7}, z_coords[] = {4, 5, 3, 1},
                        masses[] = {1, 1, 1, 1};
                octree<fp_type, decltype(mac_type)::value> t({x_coords, y_coords, z_coords, masses}, 4, max_leaf_n = 1,
                                                             ncrit = 1);
                REQUIRE(t.box_size_deduced());
                REQUIRE(t.box_size() == 14 + fp_type(0.7));
                t.update_particles_u([](const auto &its) {
                    for (std::size_t i = 0; i < 4u; ++i) {
                        for (std::size_t j = 0; j < 3u; ++j) {
                            its[j][i] *= 2;
                        }
                    }
                });
                REQUIRE(t.box_size_deduced());
                REQUIRE(t.box_size() == 28 + fp_type(1.4));
                t.update_particles_u([](const auto &its) {
                    for (std::size_t i = 0; i < 4u; ++i) {
                        for (std::size_t j = 0; j < 3u; ++j) {
                            its[j][i] /= 4;
                        }
                    }
                });
                REQUIRE(t.box_size_deduced());
                REQUIRE(t.box_size() == 7 + fp_type(0.35));
                auto its = t.p_its_o();

                REQUIRE(its[0][0] == 0);
                REQUIRE(its[0][1] == fp_type(1) / 2);
                REQUIRE(its[0][2] == fp_type(2) / 2);
                REQUIRE(its[0][3] == fp_type(3) / 2);

                REQUIRE(its[1][0] == fp_type(-4) / 2);
                REQUIRE(its[1][1] == fp_type(-5) / 2);
                REQUIRE(its[1][2] == fp_type(-6) / 2);
                REQUIRE(its[1][3] == fp_type(-7) / 2);

                REQUIRE(its[2][0] == fp_type(4) / 2);
                REQUIRE(its[2][1] == fp_type(5) / 2);
                REQUIRE(its[2][2] == fp_type(3) / 2);
                REQUIRE(its[2][3] == fp_type(1) / 2);
            }
        });
    });
}
