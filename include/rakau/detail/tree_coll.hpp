// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef RAKAU_DETAIL_TREE_COLL_HPP
#define RAKAU_DETAIL_TREE_COLL_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/iterator/permutation_iterator.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/task_group.h>

#include <rakau/detail/simple_timer.hpp>
#include <rakau/detail/tree_fwd.hpp>

namespace rakau
{

// Return the largest node level such that the node
// size is larger than the value s. If s >= m_box_size,
// 0 will be returned.
template <std::size_t NDim, typename F, typename UInt, mac MAC>
inline UInt tree<NDim, F, UInt, MAC>::coll_size_to_level(F s) const
{
    assert(std::isfinite(s));
    // NOTE: we assume we never enter here
    // with a null node size.
    assert(s > F(0));

    UInt retval = 0;
    F cur_node_size = m_box_size;

    // Don't return a result larger than cbits
    // (the max level).
    for (; retval != cbits; ++retval) {
        const auto next_node_size = cur_node_size * F(1) / F(2);
        if (next_node_size <= s) {
            // The next node size is not larger
            // than the target. Break out and
            // return the current retval;
            break;
        }
        cur_node_size = next_node_size;
    }

    return retval;
}

// Return a vector of indices into tree structure
// representing the ordered set of leaf nodes
// NOTE: need to understand if/when/how this should be parallelised.
template <std::size_t NDim, typename F, typename UInt, mac MAC>
inline auto tree<NDim, F, UInt, MAC>::coll_leaves_permutation() const
{
    simple_timer st("coll_leaves_permutation");

    // Prepare the output.
    std::vector<size_type> retval;
    const auto tsize = m_tree.size();
    retval.reserve(static_cast<decltype(retval.size())>(tsize));

    for (size_type i = 0; i < tsize; ++i) {
        if (!m_tree[i].n_children) {
            // Leaf node, add it.
            retval.push_back(i);
        }
    }

#if !defined(NDEBUG)
    // Verify the output.
    size_type tot_parts = 0;
    for (auto idx : retval) {
        assert(m_tree[idx].n_children == 0u);
        tot_parts += m_tree[idx].end - m_tree[idx].begin;
    }
    assert(tot_parts == m_parts[0].size());
#endif

    return retval;
}

inline namespace detail
{

// Compute the coordinates and the codes of the vertices of an AABB of size aabb_size around
// the point p_pos. When computing the codes, the coordinates of the AABB vertices will be clamped
// to the domain boundaries (via inv_box_size).
template <typename UInt, typename F, std::size_t NDim>
inline auto tree_coll_get_aabb(const std::array<F, NDim> &p_pos, F aabb_size, F inv_box_size)
{
    // p_pos must contain finite values.
    assert(std::all_of(p_pos.begin(), p_pos.end(), [](F x) { return std::isfinite(x); }));
    // aabb_size finite and non-negative.
    // NOTE: empty AABBs are allowed.
    assert(std::isfinite(aabb_size) && aabb_size >= F(0));

    // The number of vertices of the AABB is 2**NDim.
    static_assert(NDim < unsigned(std::numeric_limits<std::size_t>::digits), "Overflow error.");
    constexpr auto n_points = std::size_t(1) << NDim;

    // Compute the min/max coordinates of the AABB (in 2D, the lower-left
    // and upper-right corners of the AABB).
    const auto aabb_minmax_coords = [half = aabb_size * (F(1) / F(2)), &p_pos]() {
        std::array<std::array<F, NDim>, 2> retval;

        for (std::size_t i = 0; i < NDim; ++i) {
            const auto min_c = p_pos[i] - half;
            const auto max_c = p_pos[i] + half;

            // Check them.
            if (rakau_unlikely(!std::isfinite(min_c) || !std::isfinite(max_c) || min_c > max_c)) {
                throw std::invalid_argument(
                    "The computation of the min/max coordinates of an AABB produced the invalid pair of values ("
                    + std::to_string(min_c) + ", " + std::to_string(max_c) + ")");
            }

            retval[0][i] = min_c;
            retval[1][i] = max_c;
        }

        return retval;
    }();

    // The return value.
    std::pair<std::array<std::array<F, NDim>, n_points>, std::array<UInt, n_points>> retval;
    auto &aabb_vertices = retval.first;
    auto &aabb_codes = retval.second;

    // Fill in aabb_vertices and aabb_codes. The idea here is that
    // we need to generate all the 2**NDim possible combinations of min/max
    // aabb coordinates. We do it via the bit-level representation
    // of the numbers from 0 to 2**NDim - 1u. For instance, in 3 dimensions,
    // we have the numbers from 0 to 7 included:
    //
    // 0 0 0 | i = 0
    // 0 0 1 | i = 1
    // 0 1 0 | i = 2
    // 0 1 1 | i = 3
    // ...
    // 1 1 1 | i = 7
    //
    // We interpret a zero bit as setting the min aabb coordinate,
    // a one bit as setting the max aabb coordinate. So, for instance,
    // i = 3 corresponds to the aabb point (min, max, max).
    std::array<UInt, NDim> tmp_disc;
    morton_encoder<NDim, UInt> me;
    for (std::size_t i = 0; i < n_points; ++i) {
        auto &cur_vertex = aabb_vertices[i];

        for (std::size_t j = 0; j < NDim; ++j) {
            const auto idx = (i >> j) & 1u;
            const auto cur_c = aabb_minmax_coords[idx][j];

            cur_vertex[j] = cur_c;
            // NOTE: discretize with clamping.
            tmp_disc[j] = disc_single_coord<NDim, UInt, true>(cur_c, inv_box_size);
        }
        aabb_codes[i] = me(tmp_disc.data());
    }

    return retval;
}

// Given a nodal code and its level, determine the
// closed range of [min, max] highest level nodal codes
// (that is, positional codes with an extra 1 bit on top)
// belonging to the node.
template <std::size_t NDim, typename UInt>
inline std::pair<UInt, UInt> tree_coll_node_range(UInt code, UInt level)
{
    constexpr auto cbits = cbits_v<UInt, NDim>;

    assert(tree_level<NDim>(code) == level);
    assert(cbits >= level);

    const auto shift_amount = (cbits - level) * NDim;

    // Move up, right filling zeroes.
    const UInt min = code << shift_amount;
    // Turn the filling zeroes into ones.
    const UInt max = min + (UInt(-1) >> (unsigned(std::numeric_limits<UInt>::digits) - shift_amount));

    return std::make_pair(min, max);
}

} // namespace detail

template <std::size_t NDim, typename F, typename UInt, mac MAC>
template <bool Ordered, typename It>
inline auto tree<NDim, F, UInt, MAC>::compute_cgraph_impl2(It it) const
{
    simple_timer st("overall cgraph computation");

    // Check that we can index into It at least
    // up to the number of particles.
    detail::it_diff_check<It>(m_parts[0].size());

    // The vector of permutations over the leaf nodes.
    decltype(coll_leaves_permutation()) clp;

    // The return value.
    std::vector<tbb::concurrent_vector<size_type>> cgraph;

    // Prepare storage in parallel.
    tbb::task_group tg;

    tg.run([this, &cgraph]() {
        simple_timer st("cgraph prepare");
        cgraph.resize(boost::numeric_cast<decltype(cgraph.size())>(m_parts[0].size()));
    });

    tg.run([this, &clp]() {
        simple_timer st("leaves permutation");
        clp = coll_leaves_permutation();
    });

    tg.wait();

    // Create the iterators for accessing the leaf nodes.
    // NOTE: make sure we don't overflow when indexing.
    detail::it_diff_check<decltype(m_tree.begin())>(m_tree.size());
    auto c_begin = boost::make_permutation_iterator(m_tree.begin(), clp.begin());
    auto c_end = boost::make_permutation_iterator(m_tree.end(), clp.end());

    // Pre-compute the inverse of the domain size.
    const auto inv_box_size = F(1) / m_box_size;

    // Build the collision graph
    tbb::parallel_for(tbb::blocked_range(c_begin, c_end), [this, it, inv_box_size, c_begin, c_end,
                                                           &cgraph](const auto &r) {
        // Temporary vector for the particle position, and its min/max aabb coordinates.
        std::array<F, NDim> p_pos, min_aabb, max_aabb;

        // Iteration over the leaf nodes.
        for (const auto &lnode : r) {
            // Fetch/cache some properties of the leaf node.
            const auto lcode = lnode.code;
            const auto llevel = lnode.level;

            // Iterate over the particles of the leaf node.
            const auto e = lnode.end;
            for (auto pidx = lnode.begin; pidx != e; ++pidx) {
                if (pidx == 102u || pidx == 107u) {
                    static std::mutex mutex;
                    std::lock_guard lock{mutex};
                    std::cout << "idx: " << pidx << '\n';
                    std::cout << "lnode dim: " << get_node_dim(llevel, m_box_size) << '\n';
                    F nc[NDim];
                    get_node_centre(nc, lcode, m_box_size);
                    std::cout << "lnode centre: ";
                    for (std::size_t j = 0; j < NDim; ++j) {
                        std::cout << nc[j] << ", ";
                    }
                    std::cout << '\n';
                }

                // Load the particle's AABB size.
                // NOTE: if Ordered, we must transform the original
                // pidx, as 'it' points to a vector of aabb sizes sorted
                // in the original order.
                const auto aabb_size = Ordered ? *(it + m_perm[pidx]) : *(it + pidx);

                // Check the aabb_size.
                if (rakau_unlikely(!std::isfinite(aabb_size))) {
                    throw std::invalid_argument(
                        "A non-finite AABB size was detected while computing the collision graph");
                }

                // Particles with a null AABB don't participate in collision
                // detection.
                if (aabb_size == F(0)) {
                    continue;
                }

                // Fill in the particle position and min/max aabb coordinates.
                for (std::size_t j = 0; j < NDim; ++j) {
                    p_pos[j] = m_parts[j][pidx];
                    min_aabb[j] = p_pos[j] - aabb_size * (F(1) / F(2));
                    max_aabb[j] = p_pos[j] + aabb_size * (F(1) / F(2));
                }

                // Helper to run AABB collision detection between the
                // current particle and all the particles in the input
                // node.
                auto aabb_node_coll = [this, pidx, it, &max_aabb, &min_aabb, &cgraph](const auto &node) {
                    for (auto idx_other = node.begin; idx_other != node.end; ++idx_other) {
                        if (idx_other == pidx) {
                            // Don't self collision check.
                            continue;
                        }

                        // Load the other AABB size.
                        const auto other_aabb_size = Ordered ? *(it + m_perm[idx_other]) : *(it + idx_other);

                        if (other_aabb_size == F(0)) {
                            // Null AABB, particle does not participate
                            // in collision detection.
                            continue;
                        }

                        // Check AABB overlap.
                        bool overlap = true;
                        for (std::size_t j = 0; j < NDim; ++j) {
                            // NOTE: if other_aabb_size is NaN, the comparisons
                            // below will both be false and no collision
                            // will be registered. Later, when we do the collision
                            // detection for idx_other, we will be checking for
                            // NaN the aabb_size (see above) and an exception will fire.
                            const auto min2 = m_parts[j][idx_other] - other_aabb_size * (F(1) / F(2));
                            const auto max2 = m_parts[j][idx_other] + other_aabb_size * (F(1) / F(2));

                            if (!(max_aabb[j] > min2 && min_aabb[j] < max2)) {
                                overlap = false;
                                break;
                            }
                        }

                        if (overlap) {
                            // if (pidx == 102u) {
                            //     std::cout << "102 first collides with: " << idx_other << '\n';
                            // }
                            // if (pidx == 107u) {
                            //     std::cout << "107 first collides with: " << idx_other << '\n';
                            // }
                            // if (idx_other == 102u) {
                            //     std::cout << "102 second collides with: " << pidx << '\n';
                            // }
                            // if (idx_other == 107u) {
                            //     std::cout << "107 second collides with: " << pidx << '\n';
                            // }

                            // Register the collision for the particle pidx.
                            if constexpr (Ordered) {
                                cgraph[m_perm[pidx]].push_back(m_perm[idx_other]);
                                cgraph[m_perm[idx_other]].push_back(m_perm[pidx]);
                            } else {
                                cgraph[pidx].push_back(idx_other);
                                cgraph[idx_other].push_back(pidx);
                            }
                        }
                    }
                };

                // Do the collision detection with the other
                // particles in the leaf node.
                aabb_node_coll(lnode);

                // Now we need to establish
                // if the particle pidx straddles outside the leaf node,
                // in which case we will have to run collision detection
                // with other leaf nodes.

                // Compute the AABB vertices and the corresponding clamped codes.
                const auto tcga = detail::tree_coll_get_aabb<UInt>(p_pos, aabb_size, inv_box_size);
                const auto &aabb_vertices = tcga.first;
                const auto &aabb_codes = tcga.second;

                // Fetch the number of vertices.
                constexpr auto n_vertices = std::tuple_size_v<uncvref_t<decltype(aabb_vertices)>>;

                // Check if the particle straddles.
                // NOTE: the idea here is that the codes for all positions in the current node share
                // the same initial llevel*NDim digits, which we can determine from the particle code.
                const auto straddles = [llevel, &aabb_codes, pcode = m_codes[pidx]]() {
                    const auto shift_amount = (cbits - llevel) * NDim;
                    const auto common_prefix = pcode >> shift_amount;

                    for (std::size_t i = 0; i < n_vertices; ++i) {
                        if (aabb_codes[i] >> shift_amount != common_prefix) {
                            return true;
                        }
                    }

                    return false;
                }();

                // if (pidx == 102u) {
                //     std::cout << "102 straddles: " << straddles << '\n';
                // }

                // if (pidx == 107u) {
                //     std::cout << "107 straddles: " << straddles << '\n';
                // }

                if (!straddles) {
                    // The particle does not straddle, move on.
                    continue;
                }

                // The particle straddles. To know into which leaf nodes it straddles,
                // the first step is to construct a set of adjacent nodes which are guaranteed
                // to contain the AABB of pidx. We can do that by establishing for each AABB
                // vertex the smallest node of size greater than aabb_size that contains
                // the vertex. The set of equal-sized nodes thus determined will completely
                // enclose the AABB of pidx. We will then figure out which leaf nodes
                // overlap the enclosing node set and test for collision with all the
                // particles in such leaf nodes.

                // Convert the AABB size to a node level.
                const auto aabb_level = coll_size_to_level(aabb_size);

                // Iterate over the AABB vertices.
                for (std::size_t i = 0; i < n_vertices; ++i) {
                    // Compute the code of the node at level aabb_level
                    // that encloses the vertex.
                    const auto s_code
                        = (aabb_codes[i] >> ((cbits - aabb_level) * NDim)) + (UInt(1) << (aabb_level * NDim));

                    if (aabb_level >= llevel && s_code >> ((aabb_level - llevel) * NDim) == lcode) {
                        // The s_code node fits within the current leaf node.
                        // We don't need to re-run collision detection on the same
                        // particles as already done above.
                        continue;
                    }

                    // Determine the code range encompassed by s_code.
                    const auto s_code_range = detail::tree_coll_node_range<NDim>(s_code, aabb_level);

                    // In the set of leaf nodes, determine either the last node whose
                    // code range precedes s_code_range, or the first one whose code
                    // range has some overlap with s_code range.
                    auto l_begin = std::lower_bound(c_begin, c_end, s_code_range, [](const auto &n, const auto &p) {
                        const auto [n_min, n_max] = detail::tree_coll_node_range<NDim>(n.code, n.level);
                        return n_max < p.first;
                    });

                    // Now determine the first node whose code range follows s_code_range.
                    // NOTE: in upper_bound(), the comparator parameter types are flipped around
                    // wrt lower_bound().
                    const auto l_end = std::upper_bound(c_begin, c_end, s_code_range, [](const auto &p, const auto &n) {
                        const auto [n_min, n_max] = detail::tree_coll_node_range<NDim>(n.code, n.level);
                        return p.second < n_min;
                    });

                    // if (pidx == 102u) {
                    //     std::cout << "102, vertex " << i << ", n nodes " << l_end - l_begin << '\n';
                    // }
                    // if (pidx == 107u) {
                    //     std::cout << "107, vertex " << i << ", n nodes " << l_end - l_begin << '\n';
                    // }

                    // Iterate over the leaf node range, do the collision detection.
                    for (; l_begin != l_end; ++l_begin) {
                        aabb_node_coll(*l_begin);
                    }
                }
            }
        }
    });

    tbb::parallel_for(tbb::blocked_range(cgraph.begin(), cgraph.end()), [](const auto &r) {
        for (auto &v : r) {
            detail::it_diff_check<decltype(v.begin())>(v.size());

            std::sort(v.begin(), v.end());
            const auto new_end = std::unique(v.begin(), v.end());
            v.resize(static_cast<decltype(v.size())>(new_end - v.begin()));
        }
    });

#if !defined(NDEBUG)
    // Verify the collision graph.
    for (decltype(cgraph.size()) i = 0; i < cgraph.size(); ++i) {
        for (auto idx : cgraph[i]) {
            assert(idx < cgraph.size());
            assert(std::find(cgraph[idx].begin(), cgraph[idx].end(), i) != cgraph[idx].end());
        }
    }

    const auto acc
        = std::accumulate(cgraph.begin(), cgraph.end(), 0ull, [](auto cur, const auto &c) { return cur + c.size(); });

    std::cout << "Total number of collisions detected: " << acc / 2u << '\n';
#endif

    return cgraph;
}

} // namespace rakau

#endif
