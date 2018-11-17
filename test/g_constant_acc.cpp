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

static std::mt19937 rng(1);

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
        std::array<std::vector<fp_type>, 3> accs;
        t.accs_u(accs, theta);
        auto accs_u_orig(accs);
        t.accs_o(accs, theta);
        auto accs_o_orig(accs);
        t.accs_u(accs, theta, kwargs::G = fp_type(0));
        REQUIRE(std::all_of(accs[0].begin(), accs[0].end(), [](fp_type x) { return x == fp_type(0); }));
        REQUIRE(std::all_of(accs[1].begin(), accs[1].end(), [](fp_type x) { return x == fp_type(0); }));
        REQUIRE(std::all_of(accs[2].begin(), accs[2].end(), [](fp_type x) { return x == fp_type(0); }));
        t.accs_u(accs, theta, kwargs::G = fp_type(2));
        REQUIRE(std::all_of(boost::make_zip_iterator(boost::make_tuple(accs[0].begin(), accs_u_orig[0].begin())),
                            boost::make_zip_iterator(boost::make_tuple(accs[0].end(), accs_u_orig[0].end())),
                            [](auto t) { return boost::get<0>(t) == fp_type(2) * boost::get<1>(t); }));
        REQUIRE(std::all_of(boost::make_zip_iterator(boost::make_tuple(accs[1].begin(), accs_u_orig[1].begin())),
                            boost::make_zip_iterator(boost::make_tuple(accs[1].end(), accs_u_orig[1].end())),
                            [](auto t) { return boost::get<0>(t) == fp_type(2) * boost::get<1>(t); }));
        REQUIRE(std::all_of(boost::make_zip_iterator(boost::make_tuple(accs[2].begin(), accs_u_orig[2].begin())),
                            boost::make_zip_iterator(boost::make_tuple(accs[2].end(), accs_u_orig[2].end())),
                            [](auto t) { return boost::get<0>(t) == fp_type(2) * boost::get<1>(t); }));
        t.accs_o(accs, theta, kwargs::G = fp_type(1) / fp_type(2));
        REQUIRE(std::all_of(boost::make_zip_iterator(boost::make_tuple(accs[0].begin(), accs_o_orig[0].begin())),
                            boost::make_zip_iterator(boost::make_tuple(accs[0].end(), accs_o_orig[0].end())),
                            [](auto t) { return boost::get<0>(t) == boost::get<1>(t) / fp_type(2); }));
        REQUIRE(std::all_of(boost::make_zip_iterator(boost::make_tuple(accs[1].begin(), accs_o_orig[1].begin())),
                            boost::make_zip_iterator(boost::make_tuple(accs[1].end(), accs_o_orig[1].end())),
                            [](auto t) { return boost::get<0>(t) == boost::get<1>(t) / fp_type(2); }));
        REQUIRE(std::all_of(boost::make_zip_iterator(boost::make_tuple(accs[2].begin(), accs_o_orig[2].begin())),
                            boost::make_zip_iterator(boost::make_tuple(accs[2].end(), accs_o_orig[2].end())),
                            [](auto t) { return boost::get<0>(t) == boost::get<1>(t) / fp_type(2); }));
        // Check the exact accs as well.
        const auto eacc_u_orig = t.exact_acc_u(42);
        const auto eacc_o_orig = t.exact_acc_o(42);
        const auto eacc_u = t.exact_acc_u(42, kwargs::G = fp_type(2));
        const auto eacc_o = t.exact_acc_o(42, kwargs::G = fp_type(1) / fp_type(2));
        REQUIRE(eacc_u_orig[0] == eacc_u[0] / fp_type(2));
        REQUIRE(eacc_u_orig[1] == eacc_u[1] / fp_type(2));
        REQUIRE(eacc_u_orig[2] == eacc_u[2] / fp_type(2));
        REQUIRE(eacc_o_orig[0] == eacc_o[0] * fp_type(2));
        REQUIRE(eacc_o_orig[1] == eacc_o[1] * fp_type(2));
        REQUIRE(eacc_o_orig[2] == eacc_o[2] * fp_type(2));
    });
}
