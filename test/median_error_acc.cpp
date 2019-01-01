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
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

#include <rakau/config.hpp>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau_test;

using fp_types = std::tuple<float, double>;
using macs = std::tuple<std::integral_constant<mac, mac::bh>, std::integral_constant<mac, mac::bh_geom>>;

static std::mt19937 rng(1);

static const std::vector<double> sp =
#if defined(RAKAU_WITH_ROCM) || defined(RAKAU_WITH_CUDA)
    {0.5, 0.5}
#else
    {}
#endif
;

TEST_CASE("acceleration accuracy unordered")
{
    tuple_for_each(macs{}, [](auto mac_type) {
        tuple_for_each(fp_types{}, [mac_type](auto x) {
            using fp_type = decltype(x);
            std::array<std::vector<fp_type>, 3> accs;
            std::vector<fp_type> diffs;
            const auto thetas = {fp_type(0.2), fp_type(0.4), fp_type(0.6), fp_type(0.8)};
            constexpr auto nparts = 10000ul;
            constexpr fp_type bsize = 100;
            diffs.resize(nparts);
            const auto parts = get_uniform_particles<3>(nparts, bsize, rng);
            octree<fp_type, decltype(mac_type)::value> t(
                {parts.begin() + nparts, parts.begin() + 2u * nparts, parts.begin() + 3u * nparts, parts.begin()},
                nparts);
            for (auto theta : thetas) {
                t.accs_u(accs, theta, kwargs::split = sp);
                for (auto i = 0ul; i < nparts; ++i) {
                    const auto eacc = t.exact_acc_u(i);
                    const auto eacc_abs = std::sqrt(eacc[0] * eacc[0] + eacc[1] * eacc[1] + eacc[2] * eacc[2]);
                    const auto diff_x = eacc[0] - accs[0][i];
                    const auto diff_y = eacc[1] - accs[1][i];
                    const auto diff_z = eacc[2] - accs[2][i];
                    diffs[i] = std::sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z) / eacc_abs;
                }
                std::cout << "Median error for theta=" << theta << ", mac=" << static_cast<int>(mac_type()) << ": "
                          << median(diffs) << '\n';
            }
        });
    });
}
