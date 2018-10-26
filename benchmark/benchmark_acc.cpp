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

#include <tbb/task_scheduler_init.h>

#include <rakau/tree.hpp>

#include "common.hpp"

using namespace rakau;
using namespace rakau_benchmark;

int main(int argc, char **argv)
{
    std::cout.precision(20);

    const auto [nparts, idx, max_leaf_n, ncrit, nthreads, bsize, a, theta, parinit]
        = parse_benchmark_options<float>(argc, argv);

    std::optional<tbb::task_scheduler_init> t_init;
    if (nthreads) {
        t_init.emplace(nthreads);
    }

    auto parts = get_plummer_sphere(nparts, a, bsize, parinit);

    tree<3, float> t({parts.data() + nparts, parts.data() + 2 * nparts, parts.data() + 3 * nparts, parts.data()},
                     nparts, max_leaf_n, ncrit);
    std::cout << t << '\n';
    std::array<std::vector<float>, 3> accs;
    t.accs_u(accs, theta);
    std::cout << accs[0][t.ord_ind()[idx]] << ", " << accs[1][t.ord_ind()[idx]] << ", " << accs[2][t.ord_ind()[idx]]
              << '\n';
    auto eacc = t.exact_acc_u(t.ord_ind()[idx]);
    std::cout << eacc[0] << ", " << eacc[1] << ", " << eacc[2] << '\n';
}
