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
#include <random>
#include <vector>

#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau::kwargs;
using namespace rakau_test;

using fp_types = std::tuple<float, double>;

static std::mt19937 rng(2);

TEST_CASE("g constant accelerations")
{
    tuple_for_each(fp_types{}, [](auto x) {
        using fp_type = decltype(x);
        constexpr auto bsize = static_cast<fp_type>(1);
        constexpr auto s = 10000u;
        constexpr auto theta = fp_type(0.75);
        auto parts = get_uniform_particles<3>(s, bsize, rng);
        octree<fp_type> t({parts.begin() + s, parts.begin() + 2u * s, parts.begin() + 3u * s, parts.begin()}, s,
                          box_size = bsize);
        std::vector<fp_type> pots;
        t.pots_u(pots, theta);
        auto pots_u_orig(pots);
        t.pots_o(pots, theta);
        auto pots_o_orig(pots);
        t.pots_u(pots, theta, kwargs::G = fp_type(0));
        REQUIRE(std::all_of(pots.begin(), pots.end(), [](fp_type x) { return x == fp_type(0); }));
        t.pots_u(pots, theta, kwargs::G = fp_type(2));
        REQUIRE(std::all_of(boost::make_zip_iterator(boost::make_tuple(pots.begin(), pots_u_orig.begin())),
                            boost::make_zip_iterator(boost::make_tuple(pots.end(), pots_u_orig.end())),
                            [](auto t) { return boost::get<0>(t) == fp_type(2) * boost::get<1>(t); }));
        t.pots_o(pots, theta, kwargs::G = fp_type(1) / fp_type(2));
        REQUIRE(std::all_of(boost::make_zip_iterator(boost::make_tuple(pots.begin(), pots_o_orig.begin())),
                            boost::make_zip_iterator(boost::make_tuple(pots.end(), pots_o_orig.end())),
                            [](auto t) { return boost::get<0>(t) == boost::get<1>(t) / fp_type(2); }));
        // Check the exact pots as well.
        const auto epot_u_orig = t.exact_pot_u(42);
        const auto epot_o_orig = t.exact_pot_o(42);
        const auto epot_u = t.exact_pot_u(42, kwargs::G = fp_type(2));
        const auto epot_o = t.exact_pot_o(42, kwargs::G = fp_type(1) / fp_type(2));
        REQUIRE(epot_u_orig == epot_u / fp_type(2));
        REQUIRE(epot_o_orig == epot_o * fp_type(2));
    });
}
