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

#include <array>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau::kwargs;
using namespace rakau_test;

using fp_types = std::tuple<float, double>;

static std::mt19937 rng;

TEST_CASE("ctors")
{
    tuple_for_each(fp_types{}, [](auto x) {
        using fp_type = decltype(x);
        constexpr fp_type bsize = 10;
        constexpr unsigned N = 100;
        using tree_t = octree<fp_type>;
        // Default ctor.
        tree_t t0;
        REQUIRE(t0.get_box_size() == fp_type(0));
        REQUIRE(!t0.get_box_size_deduced());
        REQUIRE(t0.get_ncrit() == default_ncrit);
        REQUIRE(t0.get_max_leaf_n() == default_max_leaf_n);
        // Generate some particles in 3D.
        auto parts = get_uniform_particles<3>(N, bsize, rng);
        // Ctor from array of iterators, box size given, default ncrit/max_leaf_n.
        tree_t t1{std::array{parts.begin() + N, parts.begin() + 2u * N, parts.begin() + 3u * N, parts.begin()}, N,
                  box_size = bsize};
        REQUIRE(t1.get_box_size() == bsize);
        REQUIRE(!t1.get_box_size_deduced());
        REQUIRE(t1.get_max_leaf_n() == default_max_leaf_n);
        REQUIRE(t1.get_ncrit() == default_ncrit);
        // Non-default ncrit/max_leaf_n.
        tree_t t2{std::array{parts.begin() + N, parts.begin() + 2u * N, parts.begin() + 3u * N, parts.begin()}, N,
                  max_leaf_n = 4, ncrit = 5, box_size = bsize};
        REQUIRE(t2.get_box_size() == bsize);
        REQUIRE(!t2.get_box_size_deduced());
        REQUIRE(t2.get_max_leaf_n() == 4u);
        REQUIRE(t2.get_ncrit() == 5u);
        // Same, with ctors from init lists.
        tree_t t1a{
            {parts.begin() + N, parts.begin() + 2u * N, parts.begin() + 3u * N, parts.begin()}, N, box_size = bsize};
        REQUIRE(t1a.get_box_size() == bsize);
        REQUIRE(!t1a.get_box_size_deduced());
        REQUIRE(t1a.get_max_leaf_n() == default_max_leaf_n);
        REQUIRE(t1a.get_ncrit() == default_ncrit);
        tree_t t2a{{parts.begin() + N, parts.begin() + 2u * N, parts.begin() + 3u * N, parts.begin()},
                   N,
                   box_size = bsize,
                   max_leaf_n = 4,
                   ncrit = 5};
        REQUIRE(t2a.get_box_size() == bsize);
        REQUIRE(!t2a.get_box_size_deduced());
        REQUIRE(t2a.get_max_leaf_n() == 4u);
        REQUIRE(t2a.get_ncrit() == 5u);
        // Ctors with deduced box size.
        fp_type xcoords[] = {-10, 1, 2, 10}, ycoords[] = {-10, 1, 2, 10}, zcoords[] = {-10, 1, 2, 10},
                masses[] = {1, 1, 1, 1};
        tree_t t3{std::array{xcoords, ycoords, zcoords, masses}, 4};
        REQUIRE(t3.get_box_size() == fp_type(21));
        REQUIRE(t3.get_box_size_deduced());
        REQUIRE(t3.get_max_leaf_n() == default_max_leaf_n);
        REQUIRE(t3.get_ncrit() == default_ncrit);
        tree_t t4{std::array{xcoords, ycoords, zcoords, masses}, 4, max_leaf_n = 4, ncrit = 5};
        REQUIRE(t4.get_box_size() == fp_type(21));
        REQUIRE(t4.get_box_size_deduced());
        REQUIRE(t4.get_max_leaf_n() == 4u);
        REQUIRE(t4.get_ncrit() == 5u);
        tree_t t3a{{xcoords, ycoords, zcoords, masses}, 4};
        REQUIRE(t3a.get_box_size() == fp_type(21));
        REQUIRE(t3a.get_box_size_deduced());
        REQUIRE(t3a.get_max_leaf_n() == default_max_leaf_n);
        REQUIRE(t3a.get_ncrit() == default_ncrit);
        tree_t t4a{{xcoords, ycoords, zcoords, masses}, 4, max_leaf_n = 4, ncrit = 5};
        REQUIRE(t4a.get_box_size() == fp_type(21));
        REQUIRE(t4a.get_box_size_deduced());
        REQUIRE(t4a.get_max_leaf_n() == 4u);
        REQUIRE(t4a.get_ncrit() == 5u);
        // Provide explicit box size of zero, which generates an infinity
        // when trying to discretise.
        using Catch::Matchers::Contains;
        REQUIRE_THROWS_WITH((tree_t{{xcoords, ycoords, zcoords, masses}, 4, box_size = 0., max_leaf_n = 4, ncrit = 5}),
                            Contains("While trying to discretise the input coordinate"));
        // Box size too small.
        REQUIRE_THROWS_WITH((tree_t{{xcoords, ycoords, zcoords, masses}, 4, box_size = 3, max_leaf_n = 4, ncrit = 5}),
                            Contains("produced the floating-point value"));
        // Box size negative.
        REQUIRE_THROWS_WITH((tree_t{{xcoords, ycoords, zcoords, masses}, 4, box_size = -3, max_leaf_n = 4, ncrit = 5}),
                            Contains("The box size must be a finite non-negative value, but it is"));
        if (std::numeric_limits<fp_type>::has_infinity) {
            // Box size not finite.
            REQUIRE_THROWS_WITH((tree_t{{xcoords, ycoords, zcoords, masses},
                                        4,
                                        box_size = std::numeric_limits<fp_type>::infinity(),
                                        max_leaf_n = 4,
                                        ncrit = 5}),
                                Contains("The box size must be a finite non-negative value, but it is"));
        }
        // Wrong max_leaf_n/ncrit.
        REQUIRE_THROWS_WITH((tree_t{{xcoords, ycoords, zcoords, masses}, 4, max_leaf_n = 0, ncrit = 5}),
                            Contains("The maximum number of particles per leaf must be nonzero"));
        REQUIRE_THROWS_WITH((tree_t{{xcoords, ycoords, zcoords, masses}, 4, max_leaf_n = 4, ncrit = 0}),
                            Contains("The critical number of particles for the vectorised computation of the"));
        // Wrong number of elements in init list.
        REQUIRE_THROWS_WITH((tree_t{{xcoords, ycoords}, 4, max_leaf_n = 4, ncrit = 5}),
                            Contains("An initializer list containing 2 iterators was used in the construction of a "
                                     "3-dimensional tree, but a list with 4 iterators is required instead (3 iterators "
                                     "for the coordinates, 1 for the masses)"));
        // Copy ctor.
        tree_t t4a_copy(t4a);
        REQUIRE(t4a_copy.get_box_size() == fp_type(21));
        REQUIRE(t4a_copy.get_box_size_deduced());
        REQUIRE(t4a_copy.get_max_leaf_n() == 4u);
        REQUIRE(t4a_copy.get_ncrit() == 5u);
        // Move ctor.
        tree_t t4a_move(std::move(t4a_copy));
        REQUIRE(t4a_move.get_box_size() == fp_type(21));
        REQUIRE(t4a_move.get_box_size_deduced());
        REQUIRE(t4a_move.get_max_leaf_n() == 4u);
        REQUIRE(t4a_move.get_ncrit() == 5u);
        REQUIRE(t4a_copy.get_box_size() == fp_type(0));
        REQUIRE(!t4a_copy.get_box_size_deduced());
        REQUIRE(t4a_copy.get_max_leaf_n() == default_max_leaf_n);
        REQUIRE(t4a_copy.get_ncrit() == default_ncrit);
        // Copy assignment.
        t4a = t3a;
        REQUIRE(t4a.get_box_size() == fp_type(21));
        REQUIRE(t4a.get_box_size_deduced());
        REQUIRE(t4a.get_max_leaf_n() == default_max_leaf_n);
        REQUIRE(t4a.get_ncrit() == default_ncrit);
        // Move assignment.
        t4a = std::move(t3);
        REQUIRE(t4a.get_box_size() == fp_type(21));
        REQUIRE(t4a.get_box_size_deduced());
        REQUIRE(t4a.get_max_leaf_n() == default_max_leaf_n);
        REQUIRE(t4a.get_ncrit() == default_ncrit);
        REQUIRE(t3.get_box_size() == fp_type(0));
        REQUIRE(!t3.get_box_size_deduced());
        REQUIRE(t3.get_max_leaf_n() == default_max_leaf_n);
        REQUIRE(t3.get_ncrit() == default_ncrit);
        // Self assignments.
        t4a = *&t4a;
        REQUIRE(t4a.get_box_size() == fp_type(21));
        REQUIRE(t4a.get_box_size_deduced());
        REQUIRE(t4a.get_max_leaf_n() == default_max_leaf_n);
        REQUIRE(t4a.get_ncrit() == default_ncrit);
        t4a = std::move(t4a);
        REQUIRE(t4a.get_box_size() == fp_type(21));
        REQUIRE(t4a.get_box_size_deduced());
        REQUIRE(t4a.get_max_leaf_n() == default_max_leaf_n);
        REQUIRE(t4a.get_ncrit() == default_ncrit);
    });
}
