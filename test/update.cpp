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
#include <iterator>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau::kwargs;
using namespace rakau_test;

using fp_types = std::tuple<float, double>;

static std::mt19937 rng;

TEST_CASE("update positions")
{
    tuple_for_each(fp_types{}, [](auto x) {
        using fp_type = decltype(x);
        constexpr auto bsize = static_cast<fp_type>(1);
        constexpr auto s = 10000u;
        auto parts = get_uniform_particles<3>(s, bsize, rng);
        octree<fp_type> t({parts.begin() + s, parts.begin() + 2u * s, parts.begin() + 3u * s, parts.begin()}, s,
                          box_size = bsize),
            t2(t);
        REQUIRE(t.perm() == t.last_perm());
        // Select randomly some particle indices to track.
        using size_type = typename decltype(t)::size_type;
        std::vector<size_type> track_idx(100);
        std::uniform_int_distribution<size_type> idist(0, s - 1u);
        std::generate(track_idx.begin(), track_idx.end(), [&idist]() { return idist(rng); });
        // First let's verify that the ordered iterators functions are working properly.
        auto pro = t.p_its_o();
        using oit_diff_t =
            typename std::iterator_traits<std::remove_reference_t<decltype(t.p_its_o()[0])>>::difference_type;
        for (auto idx : track_idx) {
            REQUIRE(pro[0][static_cast<oit_diff_t>(idx)] == parts[s + idx]);
            REQUIRE(pro[1][static_cast<oit_diff_t>(idx)] == parts[2 * s + idx]);
            REQUIRE(pro[2][static_cast<oit_diff_t>(idx)] == parts[3 * s + idx]);
            REQUIRE(pro[3][static_cast<oit_diff_t>(idx)] == parts[idx]);
        }
        // Next, let's run a position updater that does not actually update positions.
        // Update both in unordered and ordered fashion.
        t.update_particles_u([](const auto &) {});
        pro = t.p_its_o();
        for (auto idx : track_idx) {
            REQUIRE(pro[0][static_cast<oit_diff_t>(idx)] == parts[s + idx]);
            REQUIRE(pro[1][static_cast<oit_diff_t>(idx)] == parts[2 * s + idx]);
            REQUIRE(pro[2][static_cast<oit_diff_t>(idx)] == parts[3 * s + idx]);
            REQUIRE(pro[3][static_cast<oit_diff_t>(idx)] == parts[idx]);
        }
        t.update_particles_o([](const auto &) {});
        pro = t.p_its_o();
        for (auto idx : track_idx) {
            REQUIRE(pro[0][static_cast<oit_diff_t>(idx)] == parts[s + idx]);
            REQUIRE(pro[1][static_cast<oit_diff_t>(idx)] == parts[2 * s + idx]);
            REQUIRE(pro[2][static_cast<oit_diff_t>(idx)] == parts[3 * s + idx]);
            REQUIRE(pro[3][static_cast<oit_diff_t>(idx)] == parts[idx]);
        }
        // Let's also verify that the internal ordering did not change.
        auto pru = t.p_its_u();
        auto pru2 = t2.p_its_u();
        for (auto idx : track_idx) {
            REQUIRE(pru[0][idx] == pru2[0][idx]);
            REQUIRE(pru[1][idx] == pru2[1][idx]);
            REQUIRE(pru[2][idx] == pru2[2][idx]);
            REQUIRE(pru[3][idx] == pru2[3][idx]);
        }
        // Now let's use an updater that swaps around the coordinates
        // for the tracked particles.
        t.update_particles_o([&track_idx](const auto &p_its) {
            auto x_it = p_its[0], y_it = p_its[1], z_it = p_its[2];
            for (auto idx : track_idx) {
                // x, y, z -> y, x, z
                std::swap(*(x_it + static_cast<oit_diff_t>(idx)), *(y_it + static_cast<oit_diff_t>(idx)));
                // y, x, z -> y, z, x
                std::swap(*(y_it + static_cast<oit_diff_t>(idx)), *(z_it + static_cast<oit_diff_t>(idx)));
            }
        });
        // Now let's verify that the particles are yielded in the correct order.
        pro = t.p_its_o();
        for (auto idx : track_idx) {
            REQUIRE(pro[0][static_cast<oit_diff_t>(idx)] == parts[2 * s + idx]);
            REQUIRE(pro[1][static_cast<oit_diff_t>(idx)] == parts[3 * s + idx]);
            REQUIRE(pro[2][static_cast<oit_diff_t>(idx)] == parts[s + idx]);
            REQUIRE(pro[3][static_cast<oit_diff_t>(idx)] == parts[idx]);
        }
        // Let's swap back to the original order.
        t.update_particles_o([&track_idx](const auto &p_its) {
            auto x_it = p_its[0], y_it = p_its[1], z_it = p_its[2];
            for (auto idx : track_idx) {
                // y, z, x -> y, x, z
                std::swap(*(z_it + static_cast<oit_diff_t>(idx)), *(y_it + static_cast<oit_diff_t>(idx)));
                // y, x, z -> x, y, z
                std::swap(*(x_it + static_cast<oit_diff_t>(idx)), *(y_it + static_cast<oit_diff_t>(idx)));
            }
        });
        pro = t.p_its_o();
        for (auto idx : track_idx) {
            REQUIRE(pro[0][static_cast<oit_diff_t>(idx)] == parts[s + idx]);
            REQUIRE(pro[1][static_cast<oit_diff_t>(idx)] == parts[2 * s + idx]);
            REQUIRE(pro[2][static_cast<oit_diff_t>(idx)] == parts[3 * s + idx]);
            REQUIRE(pro[3][static_cast<oit_diff_t>(idx)] == parts[idx]);
        }
    });
}
