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
#include <atomic>
#include <initializer_list>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

#include <boost/iterator/permutation_iterator.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau::kwargs;
using namespace rakau_test;

std::mt19937 rng;

template <typename CGraph, typename Cmp>
inline void compare_results(const CGraph &cgraph, Cmp &cmp)
{
    REQUIRE(cgraph.size() == cmp.size());

    std::atomic<int> n_failures(0);
    std::atomic<unsigned long long> n_coll(0);

    tbb::parallel_for(tbb::blocked_range(decltype(cgraph.size())(0), cgraph.size()),
                      [&n_failures, &cgraph, &cmp, &n_coll](const auto &r) {
                          auto loc_n_coll = 0ull;

                          for (auto i = r.begin(); i != r.end(); ++i) {
                              if (cgraph[i].size() != cmp[i].size()) {
                                  ++n_failures;
                              }

                              std::sort(cmp[i].begin(), cmp[i].end());
                              if (!std::equal(cmp[i].begin(), cmp[i].end(), cgraph[i].begin())) {
                                  ++n_failures;
                              }

                              loc_n_coll += cmp[i].size();
                          }

                          n_coll += loc_n_coll;
                      });

    REQUIRE(n_failures.load() == 0);
    std::cout << "Total number of collisions detected: " << n_coll.load() / 2u << '\n';
}

TEST_CASE("compute_cgraph_2d")
{
    constexpr auto bsize = 1.;
    constexpr auto s = 200u;

    auto aabb_overlap = [](auto x1, auto y1, auto s1, auto x2, auto y2, auto s2) {
        if (s1 == 0 || s2 == 0) {
            return false;
        }

        auto xmin1 = x1 - s1 / 2;
        auto xmax1 = x1 + s1 / 2;
        auto ymin1 = y1 - s1 / 2;
        auto ymax1 = y1 + s1 / 2;

        auto xmin2 = x2 - s2 / 2;
        auto xmax2 = x2 + s2 / 2;
        auto ymin2 = y2 - s2 / 2;
        auto ymax2 = y2 + s2 / 2;

        return xmax1 >= xmin2 && xmin1 <= xmax2 && ymax1 >= ymin2 && ymin1 <= ymax2;
    };

    std::vector<double> aabb_sizes(s, 1.);

    // Start with an empty tree.
    quadtree<double> t;
    REQUIRE(t.compute_cgraph_o(aabb_sizes.data()).empty());
    REQUIRE(t.compute_cgraph_u(aabb_sizes.data()).empty());

    // Test with various leaf node sizes.
    for (auto mln : {1, 4, 8, 16, 400}) {
        // Test with a variety of aabb sizes, starting from
        // very small until encompassing the whole domain.
        for (long k = 15; k >= -1; --k) {
            std::cout << "Testing mln=" << mln << ", k=" << k << '\n';

            // Fill with random data.
            auto parts = get_uniform_particles<2>(s, bsize, rng);

            const auto xc_o = parts.begin() + s;
            const auto yc_o = parts.begin() + 2u * s;

            t = quadtree<double>{x_coords = xc_o, y_coords = yc_o,  masses = parts.begin(),
                                 nparts = s,      box_size = bsize, max_leaf_n = mln};

            const auto [xc_u, yc_u, m_u] = t.p_its_u();
            detail::ignore(m_u);

            // Collision graph that will be computed with the N**2 algorithm.
            std::vector<std::vector<decltype(t)::size_type>> cmp;
            cmp.resize(s);

            // All aabbs same (nonzero) size.
            // NOTE: for k == -1, pick an AABB size much larger than the domain.
            const auto aabb_size = k >= 0 ? 1. / static_cast<double>(1l << k) : 10.;
            std::fill(aabb_sizes.begin(), aabb_sizes.end(), aabb_size);

            // Unordered testing.
            auto cgraph_u = t.compute_cgraph_u(aabb_sizes.data());
            for (auto i = 0u; i < s; ++i) {
                for (auto j = i + 1u; j < s; ++j) {
                    if (aabb_overlap(xc_u[i], yc_u[i], aabb_size, xc_u[j], yc_u[j], aabb_size)) {
                        cmp[i].push_back(j);
                        cmp[j].push_back(i);
                    }
                }
            }
            compare_results(cgraph_u, cmp);
            // Clear cmp.
            for (auto &v : cmp) {
                v.clear();
            }

            // Ordered testing.
            auto cgraph_o = t.compute_cgraph_o(aabb_sizes.data());
            for (auto i = 0u; i < s; ++i) {
                for (auto j = i + 1u; j < s; ++j) {
                    if (aabb_overlap(xc_o[i], yc_o[i], aabb_size, xc_o[j], yc_o[j], aabb_size)) {
                        cmp[i].push_back(j);
                        cmp[j].push_back(i);
                    }
                }
            }
            compare_results(cgraph_o, cmp);
            // Clear cmp.
            for (auto &v : cmp) {
                v.clear();
            }

            // Set the aabb size to zero for half the points.
            for (auto i = 0u; i < s; i += 2u) {
                aabb_sizes[i] = 0;
            }
            // Build a version of aabb_sizes permuted according to the tree order.
            auto aabb_sizes_u(aabb_sizes);
            for (auto i = 0u; i < s; ++i) {
                aabb_sizes_u[i] = aabb_sizes[t.perm()[i]];
            }

            // Redo the testing, this time with retvals passed in.
            t.compute_cgraph_u(cgraph_u, aabb_sizes_u.data());
            for (auto i = 0u; i < s; ++i) {
                for (auto j = i + 1u; j < s; ++j) {
                    if (aabb_overlap(xc_u[i], yc_u[i], aabb_sizes_u[i], xc_u[j], yc_u[j], aabb_sizes_u[j])) {
                        cmp[i].push_back(j);
                        cmp[j].push_back(i);
                    }
                }
            }
            compare_results(cgraph_u, cmp);
            // Clear cmp.
            for (auto &v : cmp) {
                v.clear();
            }

            t.compute_cgraph_o(cgraph_o, aabb_sizes.data());
            for (auto i = 0u; i < s; ++i) {
                for (auto j = i + 1u; j < s; ++j) {
                    if (aabb_overlap(xc_o[i], yc_o[i], aabb_sizes[i], xc_o[j], yc_o[j], aabb_sizes[j])) {
                        cmp[i].push_back(j);
                        cmp[j].push_back(i);
                    }
                }
            }
            compare_results(cgraph_o, cmp);
            // Clear cmp.
            for (auto &v : cmp) {
                v.clear();
            }

            // Try different sizes.
            for (auto i = 0u; i < s; ++i) {
                aabb_sizes[i] += aabb_sizes[i] / (i + 1u);
            }
            aabb_sizes_u = aabb_sizes;
            for (auto i = 0u; i < s; ++i) {
                aabb_sizes_u[i] = aabb_sizes[t.perm()[i]];
            }

            cgraph_u = t.compute_cgraph_u(aabb_sizes_u.data());
            for (auto i = 0u; i < s; ++i) {
                for (auto j = i + 1u; j < s; ++j) {
                    if (aabb_overlap(xc_u[i], yc_u[i], aabb_sizes_u[i], xc_u[j], yc_u[j], aabb_sizes_u[j])) {
                        cmp[i].push_back(j);
                        cmp[j].push_back(i);
                    }
                }
            }
            compare_results(cgraph_u, cmp);
            // Clear cmp.
            for (auto &v : cmp) {
                v.clear();
            }

            cgraph_o = t.compute_cgraph_o(aabb_sizes.data());
            for (auto i = 0u; i < s; ++i) {
                for (auto j = i + 1u; j < s; ++j) {
                    if (aabb_overlap(xc_o[i], yc_o[i], aabb_sizes[i], xc_o[j], yc_o[j], aabb_sizes[j])) {
                        cmp[i].push_back(j);
                        cmp[j].push_back(i);
                    }
                }
            }
            compare_results(cgraph_o, cmp);
            // Clear cmp.
            for (auto &v : cmp) {
                v.clear();
            }

            // All zero aabb sizes.
            std::fill(aabb_sizes.begin(), aabb_sizes.end(), 0.);

            t.compute_cgraph_u(cgraph_u, aabb_sizes.data());
            for (const auto &c : cgraph_u) {
                REQUIRE(c.empty());
            }

            t.compute_cgraph_o(cgraph_o, aabb_sizes.data());
            for (const auto &c : cgraph_o) {
                REQUIRE(c.empty());
            }
        }
    }
}

TEST_CASE("compute_cgraph_3d")
{
    constexpr auto bsize = 1.;
    constexpr auto s = 200u;

    auto aabb_overlap = [](auto x1, auto y1, auto z1, auto s1, auto x2, auto y2, auto z2, auto s2) {
        if (s1 == 0 || s2 == 0) {
            return false;
        }

        auto xmin1 = x1 - s1 / 2;
        auto xmax1 = x1 + s1 / 2;
        auto ymin1 = y1 - s1 / 2;
        auto ymax1 = y1 + s1 / 2;
        auto zmin1 = z1 - s1 / 2;
        auto zmax1 = z1 + s1 / 2;

        auto xmin2 = x2 - s2 / 2;
        auto xmax2 = x2 + s2 / 2;
        auto ymin2 = y2 - s2 / 2;
        auto ymax2 = y2 + s2 / 2;
        auto zmin2 = z2 - s2 / 2;
        auto zmax2 = z2 + s2 / 2;

        return xmax1 >= xmin2 && xmin1 <= xmax2 && ymax1 >= ymin2 && ymin1 <= ymax2 && zmax1 >= zmin2 && zmin1 <= zmax2;
    };

    std::vector<double> aabb_sizes(s, 1.);

    // Start with an empty tree.
    octree<double> t;
    REQUIRE(t.compute_cgraph_o(aabb_sizes.data()).empty());
    REQUIRE(t.compute_cgraph_u(aabb_sizes.data()).empty());

    // Test with various leaf node sizes.
    for (auto mln : {1, 4, 8, 16, 400}) {
        // Test with a variety of aabb sizes, starting from
        // very small until encompassing the whole domain.
        for (long k = 15; k >= -1; --k) {
            std::cout << "Testing mln=" << mln << ", k=" << k << '\n';

            // Fill with random data.
            auto parts = get_uniform_particles<3>(s, bsize, rng);

            const auto xc_o = parts.begin() + s;
            const auto yc_o = parts.begin() + 2u * s;
            const auto zc_o = parts.begin() + 3u * s;

            t = octree<double>{x_coords = xc_o, y_coords = yc_o,  z_coords = zc_o, masses = parts.begin(),
                               nparts = s,      box_size = bsize, max_leaf_n = mln};

            const auto [xc_u, yc_u, zc_u, m_u] = t.p_its_u();
            detail::ignore(m_u);

            // Collision graph that will be computed with the N**2 algorithm.
            std::vector<std::vector<decltype(t)::size_type>> cmp;
            cmp.resize(s);

            // All aabbs same (nonzero) size.
            // NOTE: for k == -1, pick an AABB size much larger than the domain.
            const auto aabb_size = k >= 0 ? 1. / static_cast<double>(1l << k) : 10.;
            std::fill(aabb_sizes.begin(), aabb_sizes.end(), aabb_size);

            // Unordered testing.
            auto cgraph_u = t.compute_cgraph_u(aabb_sizes.data());
            for (auto i = 0u; i < s; ++i) {
                for (auto j = i + 1u; j < s; ++j) {
                    if (aabb_overlap(xc_u[i], yc_u[i], zc_u[i], aabb_size, xc_u[j], yc_u[j], zc_u[j], aabb_size)) {
                        cmp[i].push_back(j);
                        cmp[j].push_back(i);
                    }
                }
            }
            compare_results(cgraph_u, cmp);
            // Clear cmp.
            for (auto &v : cmp) {
                v.clear();
            }

            // Ordered testing.
            auto cgraph_o = t.compute_cgraph_o(aabb_sizes.data());
            for (auto i = 0u; i < s; ++i) {
                for (auto j = i + 1u; j < s; ++j) {
                    if (aabb_overlap(xc_o[i], yc_o[i], zc_o[i], aabb_size, xc_o[j], yc_o[j], zc_o[j], aabb_size)) {
                        cmp[i].push_back(j);
                        cmp[j].push_back(i);
                    }
                }
            }
            compare_results(cgraph_o, cmp);
            // Clear cmp.
            for (auto &v : cmp) {
                v.clear();
            }

            // Set the aabb size to zero for half the points.
            for (auto i = 0u; i < s; i += 2u) {
                aabb_sizes[i] = 0;
            }
            // Build a version of aabb_sizes permuted according to the tree order.
            auto aabb_sizes_u(aabb_sizes);
            for (auto i = 0u; i < s; ++i) {
                aabb_sizes_u[i] = aabb_sizes[t.perm()[i]];
            }

            // Redo the testing.
            t.compute_cgraph_u(cgraph_u, aabb_sizes_u.data());
            for (auto i = 0u; i < s; ++i) {
                for (auto j = i + 1u; j < s; ++j) {
                    if (aabb_overlap(xc_u[i], yc_u[i], zc_u[i], aabb_sizes_u[i], xc_u[j], yc_u[j], zc_u[j],
                                     aabb_sizes_u[j])) {
                        cmp[i].push_back(j);
                        cmp[j].push_back(i);
                    }
                }
            }
            compare_results(cgraph_u, cmp);
            // Clear cmp.
            for (auto &v : cmp) {
                v.clear();
            }

            t.compute_cgraph_o(cgraph_o, aabb_sizes.data());
            for (auto i = 0u; i < s; ++i) {
                for (auto j = i + 1u; j < s; ++j) {
                    if (aabb_overlap(xc_o[i], yc_o[i], zc_o[i], aabb_sizes[i], xc_o[j], yc_o[j], zc_o[j],
                                     aabb_sizes[j])) {
                        cmp[i].push_back(j);
                        cmp[j].push_back(i);
                    }
                }
            }
            compare_results(cgraph_o, cmp);
            // Clear cmp.
            for (auto &v : cmp) {
                v.clear();
            }

            // Try different sizes.
            for (auto i = 0u; i < s; ++i) {
                aabb_sizes[i] += aabb_sizes[i] / (i + 1u);
            }
            aabb_sizes_u = aabb_sizes;
            for (auto i = 0u; i < s; ++i) {
                aabb_sizes_u[i] = aabb_sizes[t.perm()[i]];
            }

            cgraph_u = t.compute_cgraph_u(aabb_sizes_u.data());
            for (auto i = 0u; i < s; ++i) {
                for (auto j = i + 1u; j < s; ++j) {
                    if (aabb_overlap(xc_u[i], yc_u[i], zc_u[i], aabb_sizes_u[i], xc_u[j], yc_u[j], zc_u[j],
                                     aabb_sizes_u[j])) {
                        cmp[i].push_back(j);
                        cmp[j].push_back(i);
                    }
                }
            }
            compare_results(cgraph_u, cmp);
            // Clear cmp.
            for (auto &v : cmp) {
                v.clear();
            }

            cgraph_o = t.compute_cgraph_o(aabb_sizes.data());
            for (auto i = 0u; i < s; ++i) {
                for (auto j = i + 1u; j < s; ++j) {
                    if (aabb_overlap(xc_o[i], yc_o[i], zc_o[i], aabb_sizes[i], xc_o[j], yc_o[j], zc_o[j],
                                     aabb_sizes[j])) {
                        cmp[i].push_back(j);
                        cmp[j].push_back(i);
                    }
                }
            }
            compare_results(cgraph_o, cmp);
            // Clear cmp.
            for (auto &v : cmp) {
                v.clear();
            }

            // All zero aabb sizes.
            std::fill(aabb_sizes.begin(), aabb_sizes.end(), 0.);

            t.compute_cgraph_u(cgraph_u, aabb_sizes.data());
            for (const auto &c : cgraph_u) {
                REQUIRE(c.empty());
            }

            t.compute_cgraph_o(cgraph_o, aabb_sizes.data());
            for (const auto &c : cgraph_o) {
                REQUIRE(c.empty());
            }
        }
    }
}
