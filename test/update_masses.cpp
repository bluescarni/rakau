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
#include <cmath>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau::kwargs;
using namespace rakau_test;

using fp_types = std::tuple<float, double>;
using macs = std::tuple<std::integral_constant<mac, mac::bh>, std::integral_constant<mac, mac::bh_geom>>;

static std::mt19937 rng;

TEST_CASE("update masses")
{
    tuple_for_each(macs{}, [](auto mac_type) {
        tuple_for_each(fp_types{}, [](auto x) {
            using fp_type = decltype(x);
            constexpr auto bsize = static_cast<fp_type>(1);
            constexpr auto s = 10000u;
            auto parts = get_uniform_particles<3>(s, bsize, rng);
            octree<fp_type, decltype(mac_type)::value> t{x_coords = parts.begin() + s,
                                                         y_coords = parts.begin() + 2u * s,
                                                         z_coords = parts.begin() + 3u * s,
                                                         masses = parts.begin(),
                                                         nparts = s,
                                                         box_size = fp_type(10)};
            const auto t2(t);
            // Run an update which does not update anything.
            t.update_masses_u([](auto) {});
            // Check that the nodes have not changed.
            REQUIRE(t.nodes() == t2.nodes());
            t.update_masses_o([](auto) {});
            REQUIRE(t.nodes() == t2.nodes());
            // Double the mass.
            t.update_masses_u([](auto it) {
                for (auto i = 0u; i < s; ++i) {
                    *(it + i) *= 2;
                }
            });
            REQUIRE(t.nodes() != t2.nodes());
            for (decltype(t.nodes().size()) i = 0; i < t.nodes().size(); ++i) {
                // Total mass in the node must double.
                REQUIRE(t.nodes()[i].props[3] == t2.nodes()[i].props[3] * 2);
                // COM stays the same.
                REQUIRE(std::equal(t.nodes()[i].props, t.nodes()[i].props + 3, t2.nodes()[i].props));
            }
            // Do also the ordered version.
            t = t2;
            t.update_masses_o([](auto it) {
                for (auto i = 0u; i < s; ++i) {
                    *(it + i) *= 2;
                }
            });
            REQUIRE(t.nodes() != t2.nodes());
            for (decltype(t.nodes().size()) i = 0; i < t.nodes().size(); ++i) {
                REQUIRE(t.nodes()[i].props[3] == t2.nodes()[i].props[3] * 2);
                REQUIRE(std::equal(t.nodes()[i].props, t.nodes()[i].props + 3, t2.nodes()[i].props));
            }
            // Set all masses to zero. The COMs will become the geometrical centres.
            t = t2;
            t.update_masses_u([](auto it) {
                for (auto i = 0u; i < s; ++i) {
                    *(it + i) = 0;
                }
            });
            REQUIRE(t.nodes() != t2.nodes());
            for (decltype(t.nodes().size()) i = 0; i < t.nodes().size(); ++i) {
                REQUIRE(t.nodes()[i].props[3] == 0);
                fp_type c_pos[3];
                get_node_centre(c_pos, t.nodes()[i].code, fp_type(10));
                REQUIRE(std::equal(c_pos, c_pos + 3, t.nodes()[i].props));
            }
            t = t2;
            t.update_masses_o([](auto it) {
                for (auto i = 0u; i < s; ++i) {
                    *(it + i) = 0;
                }
            });
            REQUIRE(t.nodes() != t2.nodes());
            for (decltype(t.nodes().size()) i = 0; i < t.nodes().size(); ++i) {
                REQUIRE(t.nodes()[i].props[3] == 0);
                fp_type c_pos[3];
                get_node_centre(c_pos, t.nodes()[i].code, fp_type(10));
                REQUIRE(std::equal(c_pos, c_pos + 3, t.nodes()[i].props));
            }
            // Select a few indices and check them individually.
            const std::vector indices{1u, 100u, 123u, 1045u, 9800u};
            t = t2;
            t.update_masses_u([&indices](auto it) {
                for (auto idx : indices) {
                    *(it + idx) += 1;
                }
            });
            auto its_u = t.p_its_u();
            for (auto idx : indices) {
                REQUIRE(its_u[3][idx] == t2.p_its_u()[3][idx] + 1);
            }
            REQUIRE(!(t.nodes()[0] == t2.nodes()[0]));
            t = t2;
            t.update_masses_o([&indices](auto it) {
                for (auto idx : indices) {
                    *(it + idx) += 1;
                }
            });
            auto its_o = t.p_its_o();
            for (auto idx : indices) {
                REQUIRE(its_o[3][idx] == t2.p_its_o()[3][idx] + 1);
            }
            REQUIRE(!(t.nodes()[0] == t2.nodes()[0]));
            // Error throwing.
            using Catch::Matchers::Contains;
            t = t2;
            REQUIRE_THROWS_WITH(t.update_masses_u([](auto) { throw std::runtime_error("blab"); }), Contains("blab"));
            REQUIRE(t.nparts() == 0u);
            t = t2;
            REQUIRE_THROWS_WITH(t.update_masses_o([](auto) { throw std::runtime_error("blab"); }), Contains("blab"));
            REQUIRE(t.nparts() == 0u);
            // Check also with setting non-finite masses.
            t = t2;
            REQUIRE_THROWS(t.update_masses_u([](auto it) { *it = std::numeric_limits<fp_type>::infinity(); }));
            t = t2;
            REQUIRE_THROWS(t.update_masses_o([](auto it) { *it = std::numeric_limits<fp_type>::infinity(); }));
            t = t2;
            REQUIRE_THROWS(t.update_masses_u([](auto it) { *it = std::numeric_limits<fp_type>::quiet_NaN(); }));
            t = t2;
            REQUIRE_THROWS(t.update_masses_o([](auto it) { *it = std::numeric_limits<fp_type>::quiet_NaN(); }));
        });
    });
}
