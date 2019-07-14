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
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
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

// Compute the codes of the vertices of an AABB of size aabb_size around
// the point p_pos. The coordinates of the AABB vertices will be clamped
// to the domain boundaries (via inv_box_size).
template <typename UInt, typename F, std::size_t NDim>
inline auto tree_coll_get_aabb_codes(const std::array<F, NDim> &p_pos, F aabb_size, F inv_box_size)
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
    std::array<UInt, n_points> aabb_codes;

    // Fill in aabb_codes. The idea here is that
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
        for (std::size_t j = 0; j < NDim; ++j) {
            const auto idx = (i >> j) & 1u;
            // NOTE: discretize with clamping.
            tmp_disc[j] = disc_single_coord<NDim, UInt, true>(aabb_minmax_coords[idx][j], inv_box_size);
        }
        aabb_codes[i] = me(tmp_disc.data());
    }

    return aabb_codes;
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
inline auto tree<NDim, F, UInt, MAC>::compute_cgraph_impl(It it) const
{
    simple_timer st("overall cgraph computation");

    // Check that we can index into It at least
    // up to the number of particles.
    detail::it_diff_check<It>(m_parts[0].size());

    // The vector for iterating over the leaf nodes.
    decltype(coll_leaves_permutation()) clp;

    // The vector of additional particles for each leaf node.
    std::vector<tbb::concurrent_vector<size_type>> v_add;

    // The return value.
    std::vector<tbb::concurrent_vector<size_type>> cgraph;

    // Prepare storage for cgraph in parallel
    // with the v_add computation.
    tbb::task_group tg;

    tg.run([this, &cgraph]() {
        simple_timer st("cgraph prepare");
        cgraph.resize(boost::numeric_cast<decltype(cgraph.size())>(m_parts[0].size()));
    });

    decltype(boost::make_permutation_iterator(m_tree.begin(), clp.begin())) c_begin, c_end;

    tg.run([this, &clp, &v_add, it, &c_begin, &c_end]() {
        {
            simple_timer st("leaves permutation");
            clp = coll_leaves_permutation();
        }

        {
            simple_timer st("v_add prepare");
            v_add.resize(boost::numeric_cast<decltype(v_add.size())>(clp.size()));
        }

        simple_timer st("v_add computation");

        // Create the iterators for accessing the leaf nodes.
        // NOTE: make sure we don't overflow when indexing.
        detail::it_diff_check<decltype(m_tree.begin())>(m_tree.size());
        c_begin = boost::make_permutation_iterator(m_tree.begin(), clp.begin());
        c_end = boost::make_permutation_iterator(m_tree.end(), clp.end());

        // Pre-compute the inverse of the domain size.
        const auto inv_box_size = F(1) / m_box_size;

        // Determine the additional particles for each node.
        tbb::parallel_for(tbb::blocked_range(c_begin, c_end), [this, &v_add, inv_box_size, it, c_begin,
                                                               c_end](const auto &r) {
            // Temporary vector for the particle position.
            std::array<F, NDim> p_pos;

            // Iteration over the leaf nodes.
            for (const auto &lnode : r) {
                // Fetch/cache some properties of the leaf node.
                const auto lcode = lnode.code;
                const auto llevel = lnode.level;

                // Iterate over the particles of the leaf node.
                const auto e = lnode.end;
                for (auto pidx = lnode.begin; pidx != e; ++pidx) {
                    // Load the particle's AABB size.
                    // NOTE: if Ordered, we must transform the original
                    // pidx, as 'it' points to a vector of aabb sizes sorted
                    // in the original order.
                    const auto aabb_size = Ordered ? *(it + m_perm[pidx]) : *(it + pidx);

                    // Check it.
                    if (rakau_unlikely(!std::isfinite(aabb_size))) {
                        throw std::invalid_argument(
                            "A non-finite AABB size was detected while computing the collision graph");
                    }

                    // Particles with a null AABB don't participate in collision
                    // detection and cannot straddle into other nodes.
                    if (aabb_size == F(0)) {
                        continue;
                    }

                    // Fill in the particle position.
                    for (std::size_t j = 0; j < NDim; ++j) {
                        p_pos[j] = m_parts[j][pidx];
                    }

                    // Compute the clamped codes of the AABB vertices.
                    const auto aabb_codes = detail::tree_coll_get_aabb_codes<UInt>(p_pos, aabb_size, inv_box_size);

                    // Fetch the number of vertices.
                    constexpr auto n_vertices = std::tuple_size_v<std::remove_const_t<decltype(aabb_codes)>>;

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
                    // overlap the enclosing node set.

                    // Convert the AABB size to a node level.
                    const auto aabb_level = coll_size_to_level(aabb_size);

                    // Iterate over the AABB vertices.
                    for (std::size_t i = 0; i < n_vertices; ++i) {
                        // Compute the code of the node at level aabb_level
                        // that encloses the vertex.
                        const auto s_code
                            = (aabb_codes[i] >> ((cbits - aabb_level) * NDim)) + (UInt(1) << (aabb_level * NDim));

                        if (aabb_level >= llevel && s_code >> ((aabb_level - llevel) * NDim) == lcode) {
                            // The s_code node fits within the current leaf node,
                            // no need for further checks (i.e., s_code does not overlap
                            // with any leaf node other than the original one).
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
                        // NOTE: start the search from l_begin, determined above, as we know
                        // that l_end must be l_begin or an iterator following it.
                        const auto l_end
                            = std::upper_bound(l_begin, c_end, s_code_range, [](const auto &p, const auto &n) {
                                  const auto [n_min, n_max] = detail::tree_coll_node_range<NDim>(n.code, n.level);
                                  return p.second < n_min;
                              });

                        // Iterate over the leaf node range, and add pidx as extra particle
                        // to each leaf node.
                        for (; l_begin != l_end; ++l_begin) {
                            // NOTE: don't add pidx as an extra particle
                            // to its original node.
                            if (l_begin->code != lcode) {
                                // NOTE: we checked earlier that c_begin's
                                // diff type can represent m_tree.size() and,
                                // by extension, v_add.size().
                                v_add[static_cast<decltype(v_add.size())>(l_begin - c_begin)].push_back(pidx);
                            }
                        }
                    }
                }
            }
        });
    });

    tg.wait();

#if !defined(NDEBUG)
    // Verify v_add.
    for (decltype(v_add.size()) i = 0; i < v_add.size(); ++i) {
        const auto &v = v_add[i];
        for (auto idx : v) {
            // The indices of the extra particles cannot
            // be referring to particles already in the node.
            assert(idx < m_tree[clp[i]].begin || idx >= m_tree[clp[i]].end);
        }
    }
#endif

    // Build the collision graph.
    tbb::parallel_for(tbb::blocked_range(c_begin, c_end), [this, &v_add, &cgraph, c_begin, it](const auto &r) {
        // A thread local vector to store all the particle indices
        // of a node, including the extra particles.
        std::vector<size_type> node_indices;

        // Vectors to hold the min/max aabb coordinates
        // for a particle.
        std::array<F, NDim> min_aabb, max_aabb;

        // Iteration over the leaf nodes.
        for (auto l_it = r.begin(); l_it != r.end(); ++l_it) {
            // Fetch the leaf node and its vector of extra particles.
            const auto &lnode = *l_it;
            auto &va = v_add[static_cast<decltype(v_add.size())>(l_it - c_begin)];

            // Fill in all the particle indices belonging to the node: its original
            // particles, plus the additional ones.
            node_indices.resize(boost::numeric_cast<decltype(node_indices.size())>(lnode.end - lnode.begin));
            std::iota(node_indices.begin(), node_indices.end(), lnode.begin);
            std::sort(va.begin(), va.end());
            const auto new_va_end = std::unique(va.begin(), va.end());
            node_indices.insert(node_indices.end(), va.begin(), new_va_end);

            // Run the N**2 collision detection.
            const auto tot_n = node_indices.size();
            for (decltype(node_indices.size()) i = 0; i < tot_n; ++i) {
                // Load the index of the first particle.
                const auto idx1 = node_indices[i];

                // Load the AABB size for particle idx1.
                const auto aabb_size1 = Ordered ? *(it + m_perm[idx1]) : *(it + idx1);

                if (aabb_size1 == F(0)) {
                    // Skip collision detection for this particle
                    // if the AABB is null.
                    continue;
                }

                // Determine the min/max aabb coordinates for the particle idx1.
                for (std::size_t k = 0; k < NDim; ++k) {
                    min_aabb[k] = m_parts[k][idx1] - aabb_size1 * (F(1) / F(2));
                    max_aabb[k] = m_parts[k][idx1] + aabb_size1 * (F(1) / F(2));
                }

                for (auto j = i + 1u; j < tot_n; ++j) {
                    // Load the index of the second particle.
                    const auto idx2 = node_indices[j];

                    assert(idx1 != idx2);

                    // Load the AABB size for particle idx2.
                    const auto aabb_size2 = Ordered ? *(it + m_perm[idx2]) : *(it + idx2);

                    if (aabb_size2 == F(0)) {
                        // Skip collision detection for this particle
                        // if the AABB is null.
                        continue;
                    }

                    // Check for AABB overlap.
                    bool overlap = true;
                    for (std::size_t k = 0; k < NDim; ++k) {
                        const auto min2 = m_parts[k][idx2] - aabb_size2 * (F(1) / F(2));
                        const auto max2 = m_parts[k][idx2] + aabb_size2 * (F(1) / F(2));

                        if (!(max_aabb[k] > min2 && min_aabb[k] < max2)) {
                            overlap = false;
                            break;
                        }
                    }

                    if (overlap) {
                        // Overlap detected, record it.
                        if constexpr (Ordered) {
                            cgraph[m_perm[idx1]].push_back(m_perm[idx2]);
                            cgraph[m_perm[idx2]].push_back(m_perm[idx1]);
                        } else {
                            cgraph[idx1].push_back(idx2);
                            cgraph[idx2].push_back(idx1);
                        }
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
#endif

    return cgraph;
}

} // namespace rakau

#endif
