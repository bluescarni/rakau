// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <optional>

#include <tbb/task_scheduler_init.h>

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

    using tree_t = tree<3, float>;

    tree<3, float> t(
        tree_t::frip{}, 0., true,
        std::array{parts.data() + nparts, parts.data() + 2 * nparts, parts.data() + 3 * nparts, parts.data()}, nparts,
        max_leaf_n);
    std::cout << t << '\n';
}
