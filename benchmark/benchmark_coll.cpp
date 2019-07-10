// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <iostream>
#include <numeric>
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

    const auto popts = parse_coll_benchmark_options(argc, argv);

    std::optional<tbb::task_scheduler_init> t_init;
    if (std::get<2>(popts)) {
        t_init.emplace(std::get<2>(popts));
    }

    auto runner = [&popts](auto x) {
        using fp_type = decltype(x);

        const auto [nparts, max_leaf_n, _1, bsize, a, parinit, _2, ordered, psize] = popts;

        auto parts = get_plummer_sphere(nparts, static_cast<fp_type>(a), static_cast<fp_type>(bsize), parinit);
        const std::vector<fp_type> aabb_sizes(nparts, psize);

        octree<fp_type> t{kwargs::x_coords = parts.data() + nparts,
                          kwargs::y_coords = parts.data() + 2 * nparts,
                          kwargs::z_coords = parts.data() + 3 * nparts,
                          kwargs::masses = parts.data(),
                          kwargs::nparts = nparts,
                          kwargs::max_leaf_n = max_leaf_n};

        std::cout << t << '\n';
        std::cout << detail::tree_level<3>(std::max_element(t.nodes().begin(), t.nodes().end(),
                                                            [](const auto &n1, const auto &n2) {
                                                                return detail::tree_level<3>(n1.code)
                                                                       < detail::tree_level<3>(n2.code);
                                                            })
                                               ->code)
                  << '\n';

        decltype(t.compute_cgraph_o(aabb_sizes.data())) cgraph;
        if (ordered) {
            cgraph = t.compute_cgraph_o(aabb_sizes.data());
        } else {
            cgraph = t.compute_cgraph_u(aabb_sizes.data());
        }

        const auto acc = std::accumulate(cgraph.begin(), cgraph.end(), 0ull,
                                         [](auto cur, const auto &c) { return cur + c.size(); });

        std::cout << "Total number of collisions detected: " << acc / 2u << '\n';
    };

    if (std::get<6>(popts) == "float") {
        runner(0.f);
    } else {
        runner(0.);
    }
}
