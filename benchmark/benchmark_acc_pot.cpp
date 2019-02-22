// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include <tbb/task_scheduler_init.h>

#include <rakau/tree.hpp>

#include "common.hpp"

using namespace rakau;
using namespace rakau_benchmark;

int main(int argc, char **argv)
{
    std::cout.precision(20);

    const auto popts = parse_accpot_benchmark_options(argc, argv);

    std::optional<tbb::task_scheduler_init> t_init;
    if (std::get<4>(popts)) {
        t_init.emplace(std::get<4>(popts));
    }

    auto runner = [&popts](auto x) {
        using fp_type = decltype(x);

        auto inner = [&](auto m) {
            const auto [nparts, idx, max_leaf_n, ncrit, _1, bsize, a, mac_value, parinit, split, _2, mac_type] = popts;

            auto parts = get_plummer_sphere(nparts, static_cast<fp_type>(a), static_cast<fp_type>(bsize), parinit);

            octree<fp_type, decltype(m)::value> t{kwargs::x_coords = parts.data() + nparts,
                                                  kwargs::y_coords = parts.data() + 2 * nparts,
                                                  kwargs::z_coords = parts.data() + 3 * nparts,
                                                  kwargs::masses = parts.data(),
                                                  kwargs::nparts = nparts,
                                                  kwargs::max_leaf_n = max_leaf_n,
                                                  kwargs::ncrit = ncrit};
            std::cout << t << '\n';
            std::array<std::vector<fp_type>, 4> accs_pots;
            t.accs_pots_u(accs_pots, mac_value, kwargs::split = split);
            std::cout << accs_pots[0][t.inv_perm()[idx]] << ", " << accs_pots[1][t.inv_perm()[idx]] << ", "
                      << accs_pots[2][t.inv_perm()[idx]] << '\n';
            auto eacc = t.exact_acc_u(t.inv_perm()[idx]);
            std::cout << eacc[0] << ", " << eacc[1] << ", " << eacc[2] << '\n';
        };

        if (std::get<11>(popts) == "bh") {
            inner(std::integral_constant<mac, mac::bh>{});
        } else {
            inner(std::integral_constant<mac, mac::bh_geom>{});
        }
    };

    if (std::get<10>(popts) == "float") {
        runner(0.f);
    } else {
        runner(0.);
    }
}
