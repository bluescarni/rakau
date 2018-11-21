// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <iostream>
#include <optional>
#include <tuple>
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

        const auto [nparts, idx, max_leaf_n, ncrit, _1, bsize, a, theta, parinit, split, _2] = popts;

        auto parts = get_plummer_sphere(nparts, static_cast<fp_type>(a), static_cast<fp_type>(bsize), parinit);

        tree<3, fp_type> t({parts.data() + nparts, parts.data() + 2 * nparts, parts.data() + 3 * nparts, parts.data()},
                           nparts, kwargs::max_leaf_n = max_leaf_n, kwargs::ncrit = ncrit);
        std::cout << t << '\n';
        std::vector<fp_type> pots;
        t.pots_u(pots, theta, kwargs::split = split);
        std::cout << pots[t.inv_perm()[idx]] << '\n';
        auto epot = t.exact_pot_u(t.inv_perm()[idx]);
        std::cout << epot << '\n';
    };

    if (std::get<10>(popts) == "float") {
        runner(0.f);
    } else {
        runner(0.);
    }
}
