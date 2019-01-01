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
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau::kwargs;
using namespace rakau_test;

using fp_types = std::tuple<float, double>;
using macs = std::tuple<std::integral_constant<mac, mac::bh>, std::integral_constant<mac, mac::bh_geom>>;

static std::mt19937 rng;

TEST_CASE("update positions")
{
    tuple_for_each(macs{}, [](auto mac_type) {
        tuple_for_each(fp_types{}, [](auto x) {
            using fp_type = decltype(x);
            constexpr auto bsize = static_cast<fp_type>(1);
            constexpr auto s = 10000u;
            auto parts = get_uniform_particles<3>(s, bsize, rng);
            octree<fp_type, decltype(mac_type)::value> t(
                {parts.begin() + s, parts.begin() + 2u * s, parts.begin() + 3u * s, parts.begin()}, s,
                box_size = fp_type(10)),
                t2(t);
            REQUIRE(t.perm() == t.last_perm());
            // Select randomly some particle indices to track.
            using size_type = typename decltype(t)::size_type;
            std::vector<size_type> track_idx(1000);
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
            auto orig_perm = t.perm();
            auto orig_last_perm = t.last_perm();
            auto orig_inv_perm = t.inv_perm();
            t.update_particles_u([](const auto &) {});
            REQUIRE(orig_perm == t.perm());
            // NOTE: last_perm will now be the range [0, s).
            std::iota(orig_last_perm.begin(), orig_last_perm.end(), size_type(0));
            REQUIRE(orig_last_perm == t.last_perm());
            REQUIRE(orig_inv_perm == t.inv_perm());
            pro = t.p_its_o();
            for (auto idx : track_idx) {
                REQUIRE(pro[0][static_cast<oit_diff_t>(idx)] == parts[s + idx]);
                REQUIRE(pro[1][static_cast<oit_diff_t>(idx)] == parts[2 * s + idx]);
                REQUIRE(pro[2][static_cast<oit_diff_t>(idx)] == parts[3 * s + idx]);
                REQUIRE(pro[3][static_cast<oit_diff_t>(idx)] == parts[idx]);
            }
            t.update_particles_o([](const auto &) {});
            REQUIRE(orig_perm == t.perm());
            REQUIRE(orig_last_perm == t.last_perm());
            REQUIRE(orig_inv_perm == t.inv_perm());
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
            // Now let's use an updater that swaps around the coordinates.
            // Let's also save the x coordinate in Morton order for reference later.
            std::vector<fp_type> x_morton_old(t.p_its_u()[0], t.p_its_u()[0] + s), x_morton_orig(x_morton_old);
            t.update_particles_o([](const auto &p_its) {
                auto x_it = p_its[0], y_it = p_its[1], z_it = p_its[2];
                for (size_type idx = 0; idx < s; ++idx) {
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
            // Let's reorder x_morton according to last_perm.
            auto lp = t.last_perm();
            auto x_morton_new(x_morton_old);
            for (decltype(lp.size()) i = 0; i < lp.size(); ++i) {
                x_morton_new[i] = x_morton_old[lp[i]];
            }
            // We moved the x coordinate into the z coordinate. Verify that they match.
            REQUIRE(std::equal(x_morton_new.begin(), x_morton_new.end(), t.p_its_u()[2]));
            // Let's swap back to the original order.
            t.update_particles_o([](const auto &p_its) {
                auto x_it = p_its[0], y_it = p_its[1], z_it = p_its[2];
                for (size_type idx = 0; idx < s; ++idx) {
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
            // Same check as before.
            lp = t.last_perm();
            x_morton_old = x_morton_new;
            for (decltype(lp.size()) i = 0; i < lp.size(); ++i) {
                x_morton_new[i] = x_morton_old[lp[i]];
            }
            REQUIRE(std::equal(x_morton_new.begin(), x_morton_new.end(), t.p_its_u()[0]));
            REQUIRE(x_morton_new == x_morton_orig);
            // Add a constant to all coordinates.
            t.update_particles_u([](const auto &p_its) {
                for (size_type idx = 0; idx < s; ++idx) {
                    for (std::size_t j = 0; j < 3; ++j) {
                        *(p_its[j] + static_cast<oit_diff_t>(idx)) += fp_type(1);
                    }
                }
            });
            pro = t.p_its_o();
            for (auto idx : track_idx) {
                REQUIRE(pro[0][static_cast<oit_diff_t>(idx)] == parts[s + idx] + fp_type(1));
                REQUIRE(pro[1][static_cast<oit_diff_t>(idx)] == parts[2 * s + idx] + fp_type(1));
                REQUIRE(pro[2][static_cast<oit_diff_t>(idx)] == parts[3 * s + idx] + fp_type(1));
                REQUIRE(pro[3][static_cast<oit_diff_t>(idx)] == parts[idx]);
            }
            lp = t.last_perm();
            x_morton_old = x_morton_new;
            for (decltype(lp.size()) i = 0; i < lp.size(); ++i) {
                x_morton_new[i] = x_morton_old[lp[i]] + fp_type(1);
            }
            REQUIRE(std::equal(x_morton_new.begin(), x_morton_new.end(), t.p_its_u()[0]));
            // Divide by 2 all coordinates.
            t.update_particles_u([](const auto &p_its) {
                for (size_type idx = 0; idx < s; ++idx) {
                    for (std::size_t j = 0; j < 3; ++j) {
                        *(p_its[j] + static_cast<oit_diff_t>(idx)) /= fp_type(2);
                    }
                }
            });
            pro = t.p_its_o();
            for (auto idx : track_idx) {
                REQUIRE(pro[0][static_cast<oit_diff_t>(idx)] == (parts[s + idx] + fp_type(1)) / fp_type(2));
                REQUIRE(pro[1][static_cast<oit_diff_t>(idx)] == (parts[2 * s + idx] + fp_type(1)) / fp_type(2));
                REQUIRE(pro[2][static_cast<oit_diff_t>(idx)] == (parts[3 * s + idx] + fp_type(1)) / fp_type(2));
                REQUIRE(pro[3][static_cast<oit_diff_t>(idx)] == parts[idx]);
            }
            lp = t.last_perm();
            x_morton_old = x_morton_new;
            for (decltype(lp.size()) i = 0; i < lp.size(); ++i) {
                x_morton_new[i] = x_morton_old[lp[i]] / fp_type(2);
            }
            REQUIRE(std::equal(x_morton_new.begin(), x_morton_new.end(), t.p_its_u()[0]));
        });
    });
}
