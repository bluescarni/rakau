// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef RAKAU_TREE_HPP
#define RAKAU_TREE_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cassert>
#if defined(RAKAU_WITH_TIMER)
#include <chrono>
#endif
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/iterator/permutation_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/partitioner.h>
#include <tbb/task_group.h>

#include <xsimd/xsimd.hpp>

// Let's disable a few compiler warnings emitted by the libmorton code.
#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"

#if defined(__clang__)

#pragma GCC diagnostic ignored "-Wshorten-64-to-32"
#pragma GCC diagnostic ignored "-Wsign-conversion"

#endif

#endif

#include <rakau/config.hpp>
#include <rakau/detail/libmorton/morton.h>

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

#include <rakau/detail/di_aligned_allocator.hpp>
#include <rakau/detail/igor.hpp>
#include <rakau/detail/simd.hpp>
#include <rakau/detail/tree_fwd.hpp>

// likely/unlikely macros, for those compilers known to support them.
#if defined(__clang__) || defined(__GNUC__) || defined(__INTEL_COMPILER)

#define rakau_likely(condition) __builtin_expect(static_cast<bool>(condition), 1)
#define rakau_unlikely(condition) __builtin_expect(static_cast<bool>(condition), 0)

#else

#define rakau_likely(condition) (condition)
#define rakau_unlikely(condition) (condition)

#endif

namespace rakau
{

inline namespace detail
{

class simple_timer
{
public:
#if defined(RAKAU_WITH_TIMER)
    simple_timer(const char *desc) : m_desc(desc), m_start(std::chrono::high_resolution_clock::now()) {}
    double elapsed() const
    {
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - m_start)
                .count());
    }
    ~simple_timer()
    {
        std::cout << "Elapsed time for '" + m_desc + "': "
                  << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()
                                                                           - m_start)
                         .count()
                  << u8"μs\n";
    }

private:
    const std::string m_desc;
    const std::chrono::high_resolution_clock::time_point m_start;
#else
    simple_timer(const char *) {}
#endif
};

template <std::size_t NDim, typename Out>
struct morton_encoder {
};

template <>
struct morton_encoder<3, std::uint64_t> {
    template <typename It>
    std::uint64_t operator()(It it) const
    {
        const auto x = *it;
        const auto y = *(it + 1);
        const auto z = *(it + 2);
        assert(x < (1ul << 21));
        assert(y < (1ul << 21));
        assert(z < (1ul << 21));
        assert(
            !(libmorton::m3D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y), std::uint32_t(z))
              >> 63u));
        assert((libmorton::morton3D_64_encode(x, y, z)
                == libmorton::m3D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y),
                                                                       std::uint32_t(z))));
        return libmorton::m3D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y),
                                                                   std::uint32_t(z));
    }
};

template <>
struct morton_encoder<3, std::uint32_t> {
    template <typename It>
    std::uint32_t operator()(It it) const
    {
        const auto x = *it;
        const auto y = *(it + 1);
        const auto z = *(it + 2);
        assert(x < (1ul << 10));
        assert(y < (1ul << 10));
        assert(z < (1ul << 10));
        assert(
            !(libmorton::m3D_e_sLUT<std::uint32_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y), std::uint32_t(z))
              >> 31u));
        assert((libmorton::morton3D_32_encode(x, y, z)
                == libmorton::m3D_e_sLUT<std::uint32_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y),
                                                                       std::uint32_t(z))));
        return libmorton::m3D_e_sLUT<std::uint32_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y),
                                                                   std::uint32_t(z));
    }
};

// NOTE: the 2D versions still need to be tested.
template <>
struct morton_encoder<2, std::uint64_t> {
    template <typename It>
    std::uint64_t operator()(It it) const
    {
        const auto x = *it;
        const auto y = *(it + 1);
        assert(x < (1ul << 31));
        assert(y < (1ul << 31));
        assert(!(libmorton::m2D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y)) >> 63u));
        assert((libmorton::morton2D_64_encode(x, y)
                == libmorton::m2D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y))));
        return libmorton::m2D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y));
    }
};

template <>
struct morton_encoder<2, std::uint32_t> {
    template <typename It>
    std::uint32_t operator()(It it) const
    {
        const auto x = *it;
        const auto y = *(it + 1);
        assert(x < (1ul << 15));
        assert(y < (1ul << 15));
        assert(!(libmorton::m2D_e_sLUT<std::uint32_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y)) >> 31u));
        assert((libmorton::morton2D_32_encode(x, y)
                == libmorton::m2D_e_sLUT<std::uint32_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y))));
        return libmorton::m2D_e_sLUT<std::uint32_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y));
    }
};

// Small function to compare nodal codes.
template <std::size_t NDim, typename UInt>
inline bool node_compare(UInt n1, UInt n2)
{
    constexpr auto cbits = cbits_v<UInt, NDim>;
    const auto tl1 = tree_level<NDim>(n1);
    const auto tl2 = tree_level<NDim>(n2);
    const auto s_n1 = n1 << ((cbits - tl1) * NDim);
    const auto s_n2 = n2 << ((cbits - tl2) * NDim);
    return s_n1 < s_n2 || (s_n1 == s_n2 && tl1 < tl2);
}

// Apply the indirect sort defined by the vector of indices 'perm'
// to the 'values' vector. E.g., if in input
//
// values = [a, c, d, b]
// perm = [0, 3, 1, 2]
//
// then in output
//
// values = [a, b, c, d]
template <typename VVec, typename PVec>
inline void apply_isort(VVec &values, const PVec &perm)
{
    assert(values.size() == perm.size());
    VVec values_new;
    values_new.resize(values.size());
    tbb::parallel_for(tbb::blocked_range<decltype(perm.size())>(0u, perm.size()),
                      [&values_new, &values, &perm](const auto &range) {
                          for (auto i = range.begin(); i != range.end(); ++i) {
                              assert(perm[i] < values.size());
                              values_new[i] = values[perm[i]];
                          }
                      });
    values = std::move(values_new);
}

// Small helpers for the checked in-place addition of (atomic) unsigned integrals.
template <typename T>
inline void checked_uinc(T &out, T add)
{
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
    if (out > std::numeric_limits<T>::max() - add) {
        throw std::overflow_error("Overflow in the addition of two unsigned integral values");
    }
    out += add;
}

template <typename T>
inline void checked_uinc(std::atomic<T> &out, T add)
{
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
    const auto prev = out.fetch_add(add);
    if (prev > std::numeric_limits<T>::max() - add) {
        throw std::overflow_error("Overflow in the addition of two unsigned integral values");
    }
}

// Scalar FMA wrappers.
inline float fma_wrap(float x, float y, float z)
{
#if defined(FP_FAST_FMAF)
    return std::fma(x, y, z);
#else
    return x * y + z;
#endif
}

inline double fma_wrap(double x, double y, double z)
{
#if defined(FP_FAST_FMA)
    return std::fma(x, y, z);
#else
    return x * y + z;
#endif
}

inline long double fma_wrap(long double x, long double y, long double z)
{
#if defined(FP_FAST_FMAL)
    return std::fma(x, y, z);
#else
    return x * y + z;
#endif
}

// Helper to detect is simd is enabled. It is if the type F
// supports it, and if simd is not explicitly disabled via RAKAU_DISABLE_SIMD.
template <typename F>
inline constexpr bool simd_enabled_v =
#if defined(RAKAU_DISABLE_SIMD)
    false
#else
    has_simd<F>
#endif
    ;

// Helper to detect whether fast rsqrt intrinsics should be used or not
// for the batch type Batch. They are used if available and if not explicitly
// disabled by the RAKAU_DISABLE_RSQRT config option.
template <typename Batch>
inline constexpr bool use_fast_inv_sqrt =
#if defined(RAKAU_DISABLE_RSQRT)
    false
#else
    has_fast_inv_sqrt<Batch>
#endif
    ;

// Default values for the max_leaf_n and ncrit tree parameters.
// These are determined experimentally from benchmarks on various systems.

// NOTE: a value of 8 might get *slightly* better performance on the
// acc/pot computations, but it results in a tree twice as big.
inline constexpr unsigned default_max_leaf_n = 16;

// NOTE: so far we have tested only on x86 systems, where it seems for
// everything but AVX512 a good combination is 128, 16. For AVX512, 256
// seems to work better than 128.
inline constexpr unsigned default_ncrit =
#if defined(XSIMD_X86_INSTR_SET) && XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512_VERSION
    256
#else
    128
#endif
    ;

} // namespace detail

namespace kwargs
{

IGOR_MAKE_KWARG(box_size);
IGOR_MAKE_KWARG(max_leaf_n);
IGOR_MAKE_KWARG(ncrit);

IGOR_MAKE_KWARG(G);
IGOR_MAKE_KWARG(eps);

} // namespace kwargs

// NOTE: possible improvements:
// - it is still not yet clear to me what the NUMA picture is here. During tree traversal, the results
//   and the target node data are stored in thread local caches, so maybe we can try to ensure that
//   at the beginning of traversal we re-init these caches in order to make sure they are allocated
//   by the thread that's actually using them. Not entirely sure on how to do that however. The other
//   bit is the source node data that gets read during tree traversal. Perhaps we can assume that the data transfer in
//   the traversal routine for a node will mostly involve data adjacent to that node in the morton order. So perhaps we
//   can try to ensure that the TBB threads are scheduled with the same affinity as the affinity used to write initially
//   into the particle data vectors. TBB has an affinity partitioner, but it's not clear to me if we can rely on that
//   for efficient NUMA access. It's probably better to run some tests before embarking in this.
// - we should probably also think about replacing the morton encoder with some generic solution. It does not
//   need to be super high performance, as morton encoding is hardly a bottleneck here. It's more important for it
//   to be generic (i.e., work on a general number of dimensions), correct and compact.
// - add more MACs (e.g., the one from bonsai and the one from the gothic paper from Warren et al).
// - double precision benchmarking/tuning.
// - tuning for the potential computation (possibly not much improvement to be had there, but it should be investigated
//   a bit at least).
// - we currently define critical nodes those nodes with < ncrit particles. Some papers say that it's worth
//   to check also the node's size, as a crit node whose size is very large will likely result in traversal lists
//   which are not very similar to each other (which, in turn, means that during tree traversal the BH check
//   will fail often). It's probably best to start experimenting with such size as a free parameter, check the
//   performance with various values and then try to understand if there's any heuristic we can deduce from that.
// - quadrupole moments.
// - radix sort.
// - would be interesting to see if we can do the permutations in-place efficiently. If that worked, it would probably
//   help simplifying things on the GPU side. See for instance:
//   https://stackoverflow.com/questions/7365814/in-place-array-reordering
template <std::size_t NDim, typename F, typename UInt = std::size_t>
class tree
{
    // Need at least 1 dimension.
    static_assert(NDim, "The number of dimensions must be at least 1.");
    // We need to compute NDim + N safely. Currently, N is up to 2
    // (NDim + 2 vectors are needed in the temporary storage when both
    // accelerations and potentials are requested).
    static_assert(NDim <= std::numeric_limits<std::size_t>::max() - 2u, "The number of dimensions is too high.");
    // Only C++ FP types are supported at the moment.
    static_assert(std::is_floating_point_v<F>, "The type F must be a C++ floating-point type.");
    // UInt must be an unsigned integral.
    static_assert(std::is_integral_v<UInt> && std::is_unsigned_v<UInt>,
                  "The type UInt must be a C++ unsigned integral type.");
    // cbits shortcut.
    static constexpr auto cbits = cbits_v<UInt, NDim>;
    // simd_enabled shortcut.
    static constexpr bool simd_enabled = simd_enabled_v<F>;
    // Main vector type for storing floating-point values. It uses custom alignment to enable
    // aligned loads/stores whenever possible.
    using fp_vector = std::vector<F, di_aligned_allocator<F, XSIMD_DEFAULT_ALIGNMENT>>;

public:
    using size_type = tree_size_t<F>;

private:
    // Consistency check: the size type which was forward-defined
    // is the same as the actual size type.
    static_assert(std::is_same_v<size_type, typename fp_vector::size_type>);
    // The node type.
    using node_type = tree_node_t<NDim, F, UInt>;
    // The tree type.
    using tree_type = std::vector<node_type, di_aligned_allocator<node_type>>;
    // The critical node descriptor type (nodal code and particle range).
    using cnode_type = tree_cnode_t<F, UInt>;
    // List of critical nodes.
    using cnode_list_type = std::vector<cnode_type, di_aligned_allocator<cnode_type>>;
    // Small helper to get the square of the dimension of a node at the tree level node_level.
    F get_sqr_node_dim(UInt node_level) const
    {
        const auto tmp = m_box_size / static_cast<F>(UInt(1) << node_level);
        return tmp * tmp;
    }
    // A small functor to right shift an input UInt by a fixed amount.
    // Used in the tree construction functions.
    struct code_shifter {
        explicit code_shifter(UInt shift) : m_shift(shift)
        {
            assert(shift < static_cast<unsigned>(std::numeric_limits<UInt>::digits));
        }
        auto operator()(UInt code) const
        {
            return static_cast<UInt>(code >> m_shift);
        }
        UInt m_shift;
    };
    // Serial construction of a subtree. The parent of the subtree is the node with code parent_code,
    // at the level ParentLevel. The particles in the children nodes have codes in the [begin, end)
    // range. The children nodes will be appended in depth-first order to tree. crit_nodes is the local
    // list of critical nodes, crit_ancestor a flag signalling if the parent node or one of its
    // ancestors is a critical node.
    // NOTE: here and elsewhere the use of [[maybe_unused]] is a bit random, as we use to suppress
    // GCC warnings which also look rather random (e.g., it complains about some unused
    // arguments but not others).
    template <UInt ParentLevel, typename CIt>
    size_type build_tree_ser_impl(tree_type &tree, cnode_list_type &crit_nodes, [[maybe_unused]] UInt parent_code,
                                  [[maybe_unused]] CIt begin, [[maybe_unused]] CIt end, bool crit_ancestor)
    {
        if constexpr (ParentLevel < cbits) {
            assert(tree_level<NDim>(parent_code) == ParentLevel);
            assert(tree.size() && tree.back().code == parent_code);
            size_type retval = 0;
            // We should never be invoking this on an empty range.
            assert(begin != end);
            // On entry, the range [begin, end) contains the codes
            // of all the particles belonging to the parent node.
            // parent_code is the nodal code of the parent node.
            //
            // We want to iterate over the children nodes at the current level
            // (of which there might be up to 2**NDim). A child exists if
            // it contains at least 1 particle. If it contains > m_max_leaf_n particles,
            // it is an internal (i.e., non-leaf) node and we go deeper. If it contains <= m_max_leaf_n
            // particles, it is a leaf node, we stop going deeper and move to its sibling.
            //
            // This is the node prefix: it is the nodal code of the parent with the MSB switched off.
            // NOTE: overflow is prevented by the if constexpr above.
            const auto node_prefix = parent_code - (UInt(1) << (ParentLevel * NDim));
            // Init the code shifting functor, and the shifting iterators.
            // Example: consider an octree, and parent level 1 (i.e., current level 2). We will
            // want to shift down the codes so that only the most significant 2 * NDim = 2 * 3 = 6
            // bits survive.
            const code_shifter cs((cbits - ParentLevel - 1u) * NDim);
            const auto t_begin = boost::make_transform_iterator(begin, cs),
                       t_end = boost::make_transform_iterator(end, cs);
            // NOTE: overflow is prevented in the computation of cbits_v.
            for (UInt i = 0; i < (UInt(1) << NDim); ++i) {
                // We want to identify which particles belong to the current child of the parent node.
                // Example: consider a quadtree, parent level 1, parent nodal code 1 01. The node prefix,
                // computed above, is 1 01 - 1 00 = 01. Let's say we are in the child with index 10
                // (that is, i = 2, or the third child). Then, the codes of the particles belonging to the current
                // child node will begin (from the MSB) with the 4 bits 01 10 (node prefix + current child
                // index). In order to identify the particles with such initial bits in the code, we will
                // be doing a binary search on the codes using a transform iterator that will shift
                // down the codes on the fly.
                const auto [it_start, it_end]
                    = std::equal_range(t_begin, t_end, static_cast<UInt>((node_prefix << NDim) + i));
                // NOTE: the boost transform iterators inherit the difference type from the base iterator.
                // We checked outside that we can safely compute the distances in the base iterator.
                const auto npart = std::distance(it_start, it_end);
                assert(npart >= 0);
                if (npart) {
                    // npart > 0, we have a node. Compute its nodal code by moving up the
                    // parent nodal code by NDim and adding the current child node index i.
                    const auto cur_code = static_cast<UInt>((parent_code << NDim) + i);
                    // Create the new node.
                    node_type new_node{static_cast<size_type>(std::distance(m_codes.begin(), it_start.base())),
                                       static_cast<size_type>(std::distance(m_codes.begin(), it_end.base())),
                                       // NOTE: the children count gets inited to zero. It
                                       // will be filled in later.
                                       0,
                                       cur_code,
                                       ParentLevel + 1u,
                                       // NOTE: make sure the node props are initialised to zero.
                                       {},
                                       get_sqr_node_dim(ParentLevel + 1u)};
                    // Compute its properties.
                    compute_node_properties(new_node);
                    // Add the node to the tree.
                    tree.push_back(std::move(new_node));
                    // Store the tree size before possibly adding more nodes.
                    const auto tree_size = tree.size();
                    // Cast npart to the unsigned counterpart.
                    const auto u_npart
                        = static_cast<std::make_unsigned_t<decltype(std::distance(it_start, it_end))>>(npart);
                    // NOTE: we have a critical node only if there are no critical ancestors and either:
                    // - the number of particles is leq m_ncrit (i.e., the definition of a
                    //   critical node), or
                    // - the number of particles is leq m_max_leaf_n (in which case this node will have no
                    //   children, so it will be a critical node with a number of particles > m_ncrit), or
                    // - the next recursion level is the last possible one (in which case the node will also
                    //   have no children).
                    const bool critical_node
                        = !crit_ancestor
                          && (u_npart <= m_ncrit || u_npart <= m_max_leaf_n || ParentLevel + 1u == cbits);
                    if (critical_node) {
                        // The node is a critical one, add it to the list of critical nodes for this subtree.
                        crit_nodes.push_back({tree.back().code, tree.back().begin, tree.back().end});
                    }
                    if (u_npart > m_max_leaf_n) {
                        // The node is an internal one, go deeper, and update the children count
                        // for the newly-added node.
                        // NOTE: do this in 2 parts, rather than assigning directly the children count into
                        // the tree, as the computation of the children count might enlarge the tree and thus
                        // invalidate references to its elements.
                        const auto children_count = build_tree_ser_impl<ParentLevel + 1u>(
                            tree, crit_nodes, cur_code, it_start.base(), it_end.base(),
                            // NOTE: the children nodes have critical ancestors if either
                            // the newly-added node is critical or one of its ancestors is.
                            critical_node || crit_ancestor);
                        tree[tree_size - 1u].n_children = children_count;
                    }
                    // The total children count will be augmented by the children count of the
                    // newly-added node, +1 for the node itself.
                    checked_uinc(retval, tree[tree_size - 1u].n_children);
                    checked_uinc(retval, size_type(1));
                }
            }
            return retval;
        } else {
            // NOTE: if we end up here, it means we walked through all the recursion levels
            // and we cannot go any deeper.
            return 0;
        }
    }
    // Parallel tree construction. It will iterate in parallel over the children of a node with nodal code
    // parent_code at level ParentLevel, add single nodes to the 'trees' concurrent container, and recurse depth-first
    // until a node containing less than an implementation-defined number of particles is encountered.
    // From there, whole subtrees (rather than single nodes) will be
    // constructed and added to the 'trees' container via build_tree_ser_impl(). The particles in the children nodes
    // have codes in the [begin, end) range. crit_nodes is the global list of lists of critical nodes, crit_ancestor a
    // flag signalling if the parent node or one of its ancestors is a critical node.
    template <UInt ParentLevel, typename Out, typename CritNodes, typename CIt>
    size_type build_tree_par_impl(Out &trees, CritNodes &crit_nodes, [[maybe_unused]] UInt parent_code,
                                  [[maybe_unused]] CIt begin, [[maybe_unused]] CIt end,
                                  [[maybe_unused]] bool crit_ancestor)
    {
        if constexpr (ParentLevel < cbits) {
            assert(tree_level<NDim>(parent_code) == ParentLevel);
            // NOTE: the return value needs to be computed atomically as we are accumulating
            // results from multiple concurrent tasks.
            std::atomic<size_type> retval(0);
            // NOTE: similar to the previous function, see comments there.
            assert(begin != end);
            const auto node_prefix = parent_code - (UInt(1) << (ParentLevel * NDim));
            const code_shifter cs((cbits - ParentLevel - 1u) * NDim);
            const auto t_begin = boost::make_transform_iterator(begin, cs),
                       t_end = boost::make_transform_iterator(end, cs);
            tbb::parallel_for(tbb::blocked_range<UInt>(0u, UInt(1) << NDim), [node_prefix, t_begin, t_end, &trees,
                                                                              parent_code, this, &retval, crit_ancestor,
                                                                              &crit_nodes](const auto &range) {
                for (auto i = range.begin(); i != range.end(); ++i) {
                    const auto [it_start, it_end]
                        = std::equal_range(t_begin, t_end, static_cast<UInt>((node_prefix << NDim) + i));
                    const auto npart = std::distance(it_start, it_end);
                    assert(npart >= 0);
                    if (npart) {
                        const auto cur_code = static_cast<UInt>((parent_code << NDim) + i);
                        // Add a new tree, and fill its first node.
                        // NOTE: use push_back(tree_type{}) instead of emplace_back() because
                        // TBB hates clang apparently.
                        auto &new_tree = *trees.push_back(tree_type{});
                        node_type new_node{static_cast<size_type>(std::distance(m_codes.begin(), it_start.base())),
                                           static_cast<size_type>(std::distance(m_codes.begin(), it_end.base())),
                                           0,
                                           cur_code,
                                           ParentLevel + 1u,
                                           {},
                                           get_sqr_node_dim(ParentLevel + 1u)};
                        compute_node_properties(new_node);
                        new_tree.push_back(std::move(new_node));
                        const auto u_npart
                            = static_cast<std::make_unsigned_t<std::remove_const_t<decltype(npart)>>>(npart);
                        // NOTE: we have a critical node only if there are no critical ancestors and either:
                        // - the number of particles is leq m_ncrit (i.e., the definition of a
                        //   critical node), or
                        // - the number of particles is leq m_max_leaf_n (in which case this node will have no
                        //   children, so it will be a critical node with a number of particles > m_ncrit), or
                        // - the next recursion level is the last possible one (in which case the node will also
                        //   have no children).
                        const bool critical_node
                            = !crit_ancestor
                              && (u_npart <= m_ncrit || u_npart <= m_max_leaf_n || ParentLevel + 1u == cbits);
                        // Add a new entry to crit_nodes. If the only node of the new tree is critical, the new entry
                        // will contain 1 element, otherwise it will be an empty list. This empty list may remain empty,
                        // or be used to accumulate the list of critical nodes during the serial subtree construction.
                        auto &new_crit_nodes
                            = critical_node
                                  ? *crit_nodes.push_back({{new_tree[0].code, new_tree[0].begin, new_tree[0].end}})
                                  : *crit_nodes.push_back({});
                        if (u_npart > m_max_leaf_n) {
                            // NOTE: this is the smallest number of particles a node must
                            // contain in order to continue the tree construction in parallel.
                            // This number has been determined experimentally, and seems to work
                            // fine on different setups. Maybe in the future it could be made
                            // a tuning parameter, if needed.
                            constexpr auto split_nparts = 40000ul;
                            if (u_npart < split_nparts) {
                                // NOTE: like in the serial function, make sure we first compute the
                                // children count and only later we assign it into the tree, as the computation
                                // of the children count might end up modifying the tree.
                                const auto children_count = build_tree_ser_impl<ParentLevel + 1u>(
                                    new_tree, new_crit_nodes, cur_code, it_start.base(), it_end.base(),
                                    // NOTE: the children nodes have critical ancestors if either
                                    // the newly-added node is critical or one of its ancestors is.
                                    critical_node || crit_ancestor);
                                new_tree[0].n_children = children_count;
                            } else {
                                // We have enough particles in the node to continue the
                                // construction in parallel.
                                new_tree[0].n_children = build_tree_par_impl<ParentLevel + 1u>(
                                    trees, crit_nodes, cur_code, it_start.base(), it_end.base(),
                                    critical_node || crit_ancestor);
                            }
                        }
                        checked_uinc(retval, new_tree[0].n_children);
                        checked_uinc(retval, size_type(1));
                    }
                }
            });
            return retval.load();
        } else {
            return 0;
        }
    }
    void build_tree()
    {
        simple_timer st("node building");
        // Make sure we always have an empty tree when invoking this method.
        assert(m_tree.empty());
        assert(m_crit_nodes.empty());
        // Exit early if there are no particles.
        if (!m_codes.size()) {
            return;
        }
        // NOTE: in the tree builder code, we will be moving around in the codes
        // vector using random access iterators. Thus, we must ensure the difference
        // type of the iterator can represent the size of the codes vector.
        using it_diff_t = typename std::iterator_traits<decltype(m_codes.begin())>::difference_type;
        using it_udiff_t = std::make_unsigned_t<it_diff_t>;
        if (m_codes.size() > static_cast<it_udiff_t>(std::numeric_limits<it_diff_t>::max())) {
            throw std::overflow_error("The number of particles (" + std::to_string(m_codes.size())
                                      + ") is too large, and it results in an overflow condition");
        }
        // Vector of partial trees. This will eventually contain a sequence of single-node trees
        // and subtrees. A subtree starts with a node and contains all of its children, ordered
        // according to the nodal code.
        // NOTE: here we could use tbb::enumerable_thread_specific as well, but I see no reason
        // to at the moment.
        tbb::concurrent_vector<tree_type> trees;
        // List of lists of critical nodes. We will accumulate here partial ordered
        // lists of critical nodes, in a similar fashion to the vector of partial trees above.
        tbb::concurrent_vector<cnode_list_type> crit_nodes;
        // Add the root node.
        const auto root_sqr_node_dim = m_box_size * m_box_size;
        if (!std::isfinite(root_sqr_node_dim)) {
            throw std::invalid_argument(
                "The computation of the square of the dimension of the root node leads to the non-finite value "
                + std::to_string(root_sqr_node_dim));
        }
        m_tree.push_back(node_type{size_type(0),
                                   size_type(m_codes.size()),
                                   // NOTE: the children count gets inited to zero. It
                                   // will be filled in later.
                                   0,
                                   // Root code is 1.
                                   1,
                                   0,
                                   // NOTE: make sure mass and COM coords are initialised in a known state (i.e.,
                                   // zero for C++ floating-point).
                                   {},
                                   root_sqr_node_dim});

        // Compute the root node's properties. Do it concurrently with other computations.
        tbb::task_group tg;
        tg.run([&]() { compute_node_properties(m_tree.back()); });

        // Check if the root node is a critical node. It is a critical node if the number of particles is leq m_ncrit
        // (the definition of critical node) or m_max_leaf_n (in which case it will have no children).
        const bool root_is_crit = m_codes.size() <= m_ncrit || m_codes.size() <= m_max_leaf_n;
        if (root_is_crit) {
            // The root node is critical, add it to the global list.
            crit_nodes.push_back({{UInt(1), size_type(0), size_type(m_codes.size())}});
        }
        // Build the rest, if needed.
        if (m_codes.size() > m_max_leaf_n) {
            m_tree[0].n_children
                = build_tree_par_impl<0>(trees, crit_nodes, 1, m_codes.begin(), m_codes.end(), root_is_crit);
        }
        // Wait for the root node properties to be computed.
        tg.wait();

        // NOTE: the merge of the subtrees and of the critical nodes lists can be done independently.
        tg.run([&]() {
            // NOTE: this sorting and the computation of the cumulative sizes can be done also in parallel,
            // but it's probably not worth it since the size of trees should be rather small.
            //
            // Sort the subtrees according to the nodal code of the first node.
            std::sort(trees.begin(), trees.end(), [](const auto &t1, const auto &t2) {
                assert(t1.size() && t2.size());
                return node_compare<NDim>(t1[0].code, t2[0].code);
            });
            // Compute the cumulative sizes in trees.
            std::vector<size_type, di_aligned_allocator<size_type>> cum_sizes;
            // NOTE: start from 1 in order to account for the root node (which is not accounted
            // for in trees' sizes).
            cum_sizes.emplace_back(1u);
            for (const auto &t : trees) {
                cum_sizes.push_back(cum_sizes.back());
                checked_uinc(cum_sizes.back(), boost::numeric_cast<size_type>(t.size()));
            }
            // Resize the tree and copy over the data from trees.
            m_tree.resize(boost::numeric_cast<decltype(m_tree.size())>(cum_sizes.back()));
            tbb::parallel_for(
                tbb::blocked_range<decltype(cum_sizes.size())>(0u,
                                                               // NOTE: cum_sizes is 1 element larger than trees.
                                                               cum_sizes.size() - 1u),
                [this, &cum_sizes, &trees](const auto &out_range) {
                    for (auto i = out_range.begin(); i != out_range.end(); ++i) {
                        tbb::parallel_for(tbb::blocked_range<decltype(trees[i].size())>(0, trees[i].size()),
                                          [&](const auto &in_range) {
                                              std::copy(trees[i].data() + in_range.begin(),
                                                        trees[i].data() + in_range.end(),
                                                        m_tree.data() + cum_sizes[i] + in_range.begin());
                                          });
                    }
                });
        });

        tg.run([&]() {
            // NOTE: as above, we could do some of these things in parallel but it does not seem worth it at this time.
            // Sort the critical nodes lists according to the starting points of the ranges.
            std::sort(crit_nodes.begin(), crit_nodes.end(), [](const auto &v1, const auto &v2) {
                // NOTE: we may have empty lists, put them at the beginning.
                if (v1.empty()) {
                    // v1 empty, if v2 is not empty then v1 < v2, otherwise v1 >= v2.
                    return !v2.empty();
                }
                if (v2.empty()) {
                    // NOTE: v1 is not empty at this point, thus v1 >= v2.
                    return false;
                }
                // v1 and v2 are not empty, compare the starting points of their first critical nodes.
                return get<1>(v1[0]) < get<1>(v2[0]);
            });
            // Compute the cumulative sizes in crit_nodes.
            std::vector<size_type, di_aligned_allocator<size_type>> cum_sizes;
            cum_sizes.emplace_back(0u);
            for (const auto &c : crit_nodes) {
                cum_sizes.push_back(cum_sizes.back());
                checked_uinc(cum_sizes.back(), boost::numeric_cast<size_type>(c.size()));
            }
            // Resize the critical nodes list and copy over the data from crit_nodes.
            m_crit_nodes.resize(boost::numeric_cast<decltype(m_crit_nodes.size())>(cum_sizes.back()));
            tbb::parallel_for(
                tbb::blocked_range<decltype(cum_sizes.size())>(0u,
                                                               // NOTE: cum_sizes is 1 element larger than crit_nodes.
                                                               cum_sizes.size() - 1u),
                [this, &cum_sizes, &crit_nodes](const auto &out_range) {
                    for (auto i = out_range.begin(); i != out_range.end(); ++i) {
                        tbb::parallel_for(tbb::blocked_range<decltype(crit_nodes[i].size())>(0, crit_nodes[i].size()),
                                          [&](const auto &in_range) {
                                              std::copy(crit_nodes[i].data() + in_range.begin(),
                                                        crit_nodes[i].data() + in_range.end(),
                                                        m_crit_nodes.data() + cum_sizes[i] + in_range.begin());
                                          });
                    }
                });
        });
        tg.wait();

        // Various debug checks.
        // Check the tree is sorted according to the nodal code comparison.
        assert(std::is_sorted(m_tree.begin(), m_tree.end(),
                              [](const auto &n1, const auto &n2) { return node_compare<NDim>(n1.code, n2.code); }));
        // Check that all the nodes contain at least 1 element.
        assert(std::all_of(m_tree.begin(), m_tree.end(), [](const auto &t) { return t.end > t.begin; }));
        // In a non-empty domain, we must have at least 1 critical node.
        assert(!m_crit_nodes.empty());
        // The list of critical nodes must start with the first particle and end with the last particle.
        assert(get<1>(m_crit_nodes[0]) == 0u);
        assert(get<2>(m_crit_nodes.back()) == m_codes.size());
#if !defined(NDEBUG)
        // Verify that the critical nodes list contains all the particles in the domain,
        // that the critical nodes' limits are contiguous, and that the critical nodes are not empty.
        for (decltype(m_crit_nodes.size()) i = 0; i < m_crit_nodes.size() - 1u; ++i) {
            assert(get<1>(m_crit_nodes[i]) < get<2>(m_crit_nodes[i]));
            if (i == m_crit_nodes.size() - 1u) {
                break;
            }
            assert(get<2>(m_crit_nodes[i]) == get<1>(m_crit_nodes[i + 1u]));
        }
#endif
        // Check the critical nodes are ordered according to the nodal code.
        assert(std::is_sorted(m_crit_nodes.begin(), m_crit_nodes.end(), [](const auto &t1, const auto &t2) {
            return node_compare<NDim>(get<0>(t1), get<0>(t2));
        }));
        // Verify the node levels.
        assert(std::all_of(m_tree.begin(), m_tree.end(),
                           [](const auto &n) { return n.level == tree_level<NDim>(n.code); }));
        // Verify the node dim2.
        assert(std::all_of(m_tree.begin(), m_tree.end(),
                           [this](const auto &n) { return n.dim2 == get_sqr_node_dim(n.level); }));

        // NOTE: a couple of final checks to make sure we can use size_type to represent both the tree
        // size and the size of the list of critical nodes.
        if (m_tree.size() > std::numeric_limits<size_type>::max()) {
            throw std::overflow_error("The size of the tree (" + std::to_string(m_tree.size())
                                      + ") is too large, and it results in an overflow condition");
        }
        if (m_crit_nodes.size() > std::numeric_limits<size_type>::max()) {
            throw std::overflow_error("The size of the critical nodes list (" + std::to_string(m_crit_nodes.size())
                                      + ") is too large, and it results in an overflow condition");
        }
    }
    void compute_node_properties(node_type &node)
    {
        // Get the indices and the size for the current node.
        const auto begin = node.begin, end = node.end;
        assert(end > begin);
        const auto size = end - begin;
        // Compute the total mass.
        const auto tot_mass = std::accumulate(m_parts[NDim].data() + begin, m_parts[NDim].data() + end, F(0));
        // Compute the COM for the coordinates.
        const auto m_ptr = m_parts[NDim].data() + begin;
        for (std::size_t j = 0; j < NDim; ++j) {
            F acc(0);
            auto c_ptr = m_parts[j].data() + begin;
            for (std::remove_const_t<decltype(size)> k = 0; k < size; ++k) {
                acc = fma_wrap(m_ptr[k], c_ptr[k], acc);
            }
            node.props[j] = acc / tot_mass;
        }
        // Store the total mass.
        node.props[NDim] = tot_mass;
    }
    // Discretize the coordinates of the particle at index idx. The result will
    // be written into retval.
    void disc_coords(std::array<UInt, NDim> &retval, size_type idx) const
    {
        constexpr UInt factor = UInt(1) << cbits;
        for (std::size_t j = 0; j < NDim; ++j) {
            // Load the coordinate locally.
            const auto x = m_parts[j][idx];
            // Translate and rescale the coordinate so that -box_size/2 becomes zero
            // and box_size/2 becomes 1.
            auto tmp = x / m_box_size + F(.5);
            // Rescale by factor.
            tmp *= factor;
            // Check: don't end up with a nonfinite value.
            if (rakau_unlikely(!std::isfinite(tmp))) {
                throw std::invalid_argument("While trying to discretise the input coordinate " + std::to_string(x)
                                            + " in a box of size " + std::to_string(m_box_size)
                                            + ", the non-finite value " + std::to_string(tmp) + " was generated");
            }
            // Check: don't end up outside the [0, factor) range.
            if (rakau_unlikely(tmp < F(0) || tmp >= F(factor))) {
                throw std::invalid_argument("The discretisation of the input coordinate " + std::to_string(x)
                                            + " in a box of size " + std::to_string(m_box_size)
                                            + " produced the floating-point value " + std::to_string(tmp)
                                            + ", which is outside the allowed bounds");
            }
            // Cast to UInt and write to retval.
            retval[j] = static_cast<UInt>(tmp);
            // Last check, make sure we don't overflow.
            if (rakau_unlikely(retval[j] >= factor)) {
                throw std::invalid_argument("The discretisation of the input coordinate " + std::to_string(x)
                                            + " in a box of size " + std::to_string(m_box_size)
                                            + " produced the integral value " + std::to_string(retval[j])
                                            + ", which is outside the allowed bounds");
            }
        }
    }
    // Small helper to determine m_inv_perm based on the indirect sorting vector m_perm.
    // This is used when (re)building the tree.
    void perm_to_inv_perm()
    {
        // NOTE: it's *very* important here that we read/write only to/from m_perm and m_inv_perm.
        // This function is often run in parallel with other functions that touch other members
        // of the tree, and if we try to access those members here we'll end up with data races.
        assert(m_perm.size() == m_inv_perm.size());
        tbb::parallel_for(tbb::blocked_range<size_type>(0u, static_cast<size_type>(m_perm.size())),
                          [this](const auto &range) {
                              for (auto i = range.begin(); i != range.end(); ++i) {
                                  assert(i < m_perm.size());
                                  assert(m_perm[i] < m_inv_perm.size());
                                  m_inv_perm[m_perm[i]] = i;
                              }
                          });
    }
    // Indirect code sort. The input range, which must point to values of type size_type,
    // will be sorted so that, after sorting, [m_codes[*begin], m_codes[*(begin + 1)], ... ]
    // yields the values in m_codes in ascending order. This is used when (re)building the tree.
    template <typename It>
    void indirect_code_sort(It begin, It end) const
    {
        static_assert(std::is_same_v<size_type, typename std::iterator_traits<It>::value_type>);
        simple_timer st("indirect code sorting");
        tbb::parallel_sort(begin, end, [codes_ptr = m_codes.data()](const size_type &idx1, const size_type &idx2) {
            return codes_ptr[idx1] < codes_ptr[idx2];
        });
    }
    // Determine the box size from an input sequence of iterators representing the coordinates and
    // the masses (unused) of the particles in the simulation.
    // NOTE: this function assumes we can index into It up to N.
    // NOTE: this function is safe for N == 0 (it will return zero in that case).
    template <typename It>
    static F determine_box_size(const std::array<It, NDim + 1u> &cm_it, const size_type &N)
    {
        simple_timer st_m("box size deduction");
        using it_diff_t = typename std::iterator_traits<It>::difference_type;
        // NOTE: we will be indexing into It up to the value N below. Make sure we checked *outside*
        // this function that we can do that.
        assert(N <= static_cast<std::make_unsigned_t<it_diff_t>>(std::numeric_limits<it_diff_t>::max()));
        // Local maximum coordinates for each thread. For each thread, the initial value
        // will be std::array<F, NDim>{}, that is, all max coordinates will be zero.
        tbb::enumerable_thread_specific<std::array<F, NDim>> max_coords(std::array<F, NDim>{});
        tbb::parallel_for(tbb::blocked_range<size_type>(0u, N), [&cm_it, &max_coords](const auto &range) {
            // Copy locally the current max coords array.
            auto local_max = max_coords.local();
            for (auto i = range.begin(); i != range.end(); ++i) {
                for (std::size_t j = 0; j < NDim; ++j) {
                    const auto tmp = std::abs(*(cm_it[j] + static_cast<it_diff_t>(i)));
                    if (!std::isfinite(tmp)) {
                        throw std::invalid_argument("While trying to automatically determine the domain size, a "
                                                    "non-finite coordinate with absolute value "
                                                    + std::to_string(tmp) + " was encountered");
                    }
                    local_max[j] = std::max(local_max[j], tmp);
                }
            }
            // Store the updated max coords.
            max_coords.local() = local_max;
        });
        // Combine the maxima from all threads into a single maximum vector.
        // NOTE: if max_coords is empty (i.e., for N == 0), the combine function will return a
        // copy of the element used in the construction of the enumerable_thread_specific object,
        // that is, an array of zeroes.
        const auto mc = max_coords.combine([](const auto &c1, const auto &c2) {
            std::array<F, NDim> retval;
            for (std::size_t j = 0; j < NDim; ++j) {
                retval[j] = std::max(c1[j], c2[j]);
            }
            return retval;
        });
        // Pick the max of the NDim coordinates, multiply by 2 to get the box size.
        auto retval = *std::max_element(mc.begin(), mc.end()) * F(2);
        // Add a 5% slack.
        retval += retval / F(20);
        // Final check.
        if (!std::isfinite(retval)) {
            throw std::invalid_argument("The automatic deduction of the domain size produced the non-finite value "
                                        + std::to_string(retval));
        }
        return retval;
    }
    // Chunk size to be used in parallel operations when moving data. Set to a large value
    // because this is used in bulk transfer operations, where we don't want TBB to try to split
    // up the work in packages which are too small.
    static constexpr auto data_chunking = 1000000ul;
    // Implementation of the constructor.
    // NOTE: It needs to be a random access iterator, as we need to index into it for parallel iteration.
    template <typename It>
    void construct_impl(const F &box_size, bool box_size_deduced, const std::array<It, NDim + 1u> &cm_it,
                        const size_type &N, const size_type &max_leaf_n, const size_type &ncrit)
    {
        simple_timer st("overall tree construction");

        // Copy in data members.
        m_box_size = box_size;
        m_box_size_deduced = box_size_deduced;
        m_max_leaf_n = max_leaf_n;
        m_ncrit = ncrit;

        // Param consistency checks: if size is deduced, box_size must be zero.
        assert(!m_box_size_deduced || m_box_size == F(0));
        // Box size checks (if automatically deduced, m_box_size is set to zero, so it will
        // pass the checks).
        if (!std::isfinite(m_box_size) || m_box_size < F(0)) {
            throw std::invalid_argument("The box size must be a finite non-negative value, but it is "
                                        + std::to_string(box_size) + " instead");
        }
        // Check the max_leaf_n param.
        if (!max_leaf_n) {
            throw std::invalid_argument("The maximum number of particles per leaf must be nonzero");
        }
        // Check the ncrit param.
        if (!ncrit) {
            throw std::invalid_argument("The critical number of particles for the vectorised computation of the "
                                        "potentials/accelerations must be nonzero");
        }

        // NOTE: in the parallel for loops below, we need to index into the random-access iterator
        // type It up to the value N. Make sure we can do that.
        using it_diff_t = typename std::iterator_traits<It>::difference_type;
        // NOTE: for use in make_unsigned, it_diff_t must be a C++ integral. This should be ensured
        // by iterator_traits, at least for input iterators:
        // https://en.cppreference.com/w/cpp/iterator/iterator_traits
        if (N > static_cast<std::make_unsigned_t<it_diff_t>>(std::numeric_limits<it_diff_t>::max())) {
            throw std::overflow_error("The number of particles (" + std::to_string(m_parts[0].size())
                                      + ") is too large, and it results in an overflow condition");
        }

        // Prepare the vectors.
        for (auto &vc : m_parts) {
            vc.resize(N);
        }
        // NOTE: these ensure that, from now on, we can just cast
        // freely between the size types of the masses/coords and codes/indices vectors.
        m_codes.resize(boost::numeric_cast<decltype(m_codes.size())>(N));
        m_perm.resize(boost::numeric_cast<decltype(m_perm.size())>(N));
        m_last_perm.resize(boost::numeric_cast<decltype(m_last_perm.size())>(N));
        m_inv_perm.resize(boost::numeric_cast<decltype(m_inv_perm.size())>(N));

        {
            // Copy the input data.
            simple_timer st_m("data movement");
            // NOTE: we will be essentially doing a memcpy here. Let's try to fix a
            // large chunk size and let's use a simple partitioner, in order to
            // limit the parallel overhead while hopefully still getting some speedup.
            for (std::size_t j = 0; j < NDim + 1u; ++j) {
                tbb::parallel_for(tbb::blocked_range<size_type>(0u, N, boost::numeric_cast<size_type>(data_chunking)),
                                  [this, &cm_it, j](const auto &range) {
                                      std::copy(cm_it[j] + static_cast<it_diff_t>(range.begin()),
                                                cm_it[j] + static_cast<it_diff_t>(range.end()),
                                                m_parts[j].data() + range.begin());
                                  },
                                  tbb::simple_partitioner());
            }
            // Generate the initial m_perm data (this is just a iota).
            tbb::parallel_for(tbb::blocked_range<size_type>(0u, N, boost::numeric_cast<size_type>(data_chunking)),
                              [this](const auto &range) {
                                  std::iota(m_perm.data() + range.begin(), m_perm.data() + range.end(), range.begin());
                              },
                              tbb::simple_partitioner());
        }

        // Deduce the box size, if needed.
        if (m_box_size_deduced) {
            // NOTE: this function works ok if N == 0.
            m_box_size = determine_box_size(p_its_u(), N);
        }

        {
            // Do the Morton encoding.
            simple_timer st_m("morton encoding");
            tbb::parallel_for(tbb::blocked_range<size_type>(0u, N), [this](const auto &range) {
                // Temporary structure used in the encoding.
                std::array<UInt, NDim> tmp_dcoord;
                // The encoder object.
                morton_encoder<NDim, UInt> me;
                for (auto i = range.begin(); i != range.end(); ++i) {
                    disc_coords(tmp_dcoord, i);
                    m_codes[i] = me(tmp_dcoord.data());
                }
            });
        }
        // Do the sorting of m_perm.
        indirect_code_sort(m_perm.begin(), m_perm.end());
        {
            // Apply the permutation to the data members.
            // These steps can be done in parallel.
            // NOTE: it's not 100% clear if doing things in parallel here helps.
            // It seems to help a bit on the large skylake systems, hurt a bit
            // on a desktop ryzen. Perhaps revisit in the future.
            simple_timer st_p("permute");
            tbb::task_group tg;
            tg.run([this]() {
                apply_isort(m_codes, m_perm);
                // Make sure the sort worked as intended.
                assert(std::is_sorted(m_codes.begin(), m_codes.end()));
            });
            for (std::size_t j = 0; j < NDim + 1u; ++j) {
                tg.run([this, j]() { apply_isort(m_parts[j], m_perm); });
            }
            // Establish the inverse permutation vector.
            tg.run([this]() { perm_to_inv_perm(); });
            // Copy over m_perm to m_last_perm.
            tg.run([this, N]() {
                tbb::parallel_for(tbb::blocked_range<size_type>(0u, N, boost::numeric_cast<size_type>(data_chunking)),
                                  [this](const auto &range) {
                                      std::copy(m_perm.data() + range.begin(), m_perm.data() + range.end(),
                                                m_last_perm.data() + range.begin());
                                  },
                                  tbb::simple_partitioner());
            });
            tg.wait();
        }
        // Now let's proceed to the tree construction.
        // NOTE: this function works ok if N == 0.
        build_tree();
    }

public:
    // Default constructor.
    tree() : m_box_size(0), m_box_size_deduced(false), m_max_leaf_n(default_max_leaf_n), m_ncrit(default_ncrit) {}
    // Constructor from array of iterators.
    template <typename It, typename... KwArgs>
    explicit tree(const std::array<It, NDim + 1u> &cm_it, const size_type &N, KwArgs &&... args)
    {
        // Parse the kwargs.
        ::igor::parser p{args...};

        // Handle the box size.
        F box_size(0);
        bool box_size_deduced = true;
        if constexpr (p.is_provided(kwargs::box_size)) {
            box_size = boost::numeric_cast<F>(p(kwargs::box_size));
            box_size_deduced = false;
        }

        // Handle max_leaf_n and ncrit.
        size_type max_leaf_n = default_max_leaf_n, ncrit = default_ncrit;
        if constexpr (p.is_provided(kwargs::max_leaf_n)) {
            max_leaf_n = boost::numeric_cast<size_type>(p(kwargs::max_leaf_n));
        }
        if constexpr (p.is_provided(kwargs::ncrit)) {
            ncrit = boost::numeric_cast<size_type>(p(kwargs::ncrit));
        }

        // Do the actual construction.
        construct_impl(box_size, box_size_deduced, cm_it, N, max_leaf_n, ncrit);
    }

private:
    template <typename It>
    static auto ctor_ilist_to_array(std::initializer_list<It> ilist)
    {
        if (rakau_unlikely(ilist.size() != NDim + 1u)) {
            throw std::invalid_argument("An initializer list containing " + std::to_string(ilist.size())
                                        + " iterators was used in the construction of a " + std::to_string(NDim)
                                        + "-dimensional tree, but a list with " + std::to_string(NDim + 1u)
                                        + " iterators is required instead (" + std::to_string(NDim)
                                        + " iterators for the coordinates, 1 for the masses)");
        }
        std::array<It, NDim + 1u> retval;
        std::copy(ilist.begin(), ilist.end(), retval.begin());
        return retval;
    }

public:
    // Convenience overload with init list instead of array.
    template <typename It, typename... KwArgs>
    explicit tree(std::initializer_list<It> cm_it, const size_type &N, KwArgs &&... args)
        : tree(ctor_ilist_to_array(cm_it), N, std::forward<KwArgs>(args)...)
    {
    }

private:
    template <typename Allocator>
    static auto ctor_vecs_to_its(const std::array<std::vector<F, Allocator>, NDim + 1u> &coords)
    {
        std::array<decltype(coords[0].data()), NDim + 1u> retval;

        const auto s = coords[0].size();
        retval[0] = coords[0].data();
        for (std::size_t j = 1; j < NDim + 1u; ++j) {
            if (rakau_unlikely(coords[j].size() != s)) {
                throw std::invalid_argument("Inconsistent sizes detected in the construction of a tree from an array "
                                            "of vectors: the first vector has a size of "
                                            + std::to_string(s) + ", while the vector at index " + std::to_string(j)
                                            + " has a size of " + std::to_string(coords[j].size())
                                            + " (all the vectors in the input array must have the same size)");
            }
            retval[j] = coords[j].data();
        }

        return retval;
    }

public:
    // Convenience overload for the construction from an array of vectors of coords/masses.
    template <typename Allocator, typename... KwArgs>
    explicit tree(const std::array<std::vector<F, Allocator>, NDim + 1u> &coords, KwArgs &&... args)
        : tree(ctor_vecs_to_its(coords), boost::numeric_cast<size_type>(coords[0].size()),
               std::forward<KwArgs>(args)...)
    {
    }
    tree(const tree &) = default;
    tree(tree &&other) noexcept
        : m_box_size(other.m_box_size), m_box_size_deduced(other.m_box_size_deduced), m_max_leaf_n(other.m_max_leaf_n),
          m_ncrit(other.m_ncrit), m_parts(std::move(other.m_parts)), m_codes(std::move(other.m_codes)),
          m_perm(std::move(other.m_perm)), m_last_perm(std::move(other.m_last_perm)),
          m_inv_perm(std::move(other.m_inv_perm)), m_tree(std::move(other.m_tree)),
          m_crit_nodes(std::move(other.m_crit_nodes))
    {
        // Make sure other is left in a known state, otherwise we might
        // have in principle assertions failures in the destructor of other
        // in debug mode.
        // NOTE: it is not clear if we can rely on std::vector's move ctor
        // to clear() the moved-from object. The page on cppreference says
        // the moved-from vector is guaranteed to be empty(), but I was not able
        // to confirm this independently. So for now let's keep custom
        // implementations of move semantics until this is clarified.
        other.clear();
    }
    tree &operator=(const tree &other)
    {
        try {
            if (this != &other) {
                m_box_size = other.m_box_size;
                m_box_size_deduced = other.m_box_size_deduced;
                m_max_leaf_n = other.m_max_leaf_n;
                m_ncrit = other.m_ncrit;
                m_parts = other.m_parts;
                m_codes = other.m_codes;
                m_perm = other.m_perm;
                m_last_perm = other.m_last_perm;
                m_inv_perm = other.m_inv_perm;
                m_tree = other.m_tree;
                m_crit_nodes = other.m_crit_nodes;
            }
            return *this;
        } catch (...) {
            // NOTE: if we triggered an exception, this might now be
            // in an inconsistent state. Call clear()
            // to reset to a consistent state before re-throwing.
            clear();
            throw;
        }
    }
    tree &operator=(tree &&other) noexcept
    {
        if (this != &other) {
            m_box_size = other.m_box_size;
            m_box_size_deduced = other.m_box_size_deduced;
            m_max_leaf_n = other.m_max_leaf_n;
            m_ncrit = other.m_ncrit;
            m_parts = std::move(other.m_parts);
            m_codes = std::move(other.m_codes);
            m_perm = std::move(other.m_perm);
            m_last_perm = std::move(other.m_last_perm);
            m_inv_perm = std::move(other.m_inv_perm);
            m_tree = std::move(other.m_tree);
            m_crit_nodes = std::move(other.m_crit_nodes);
            // Make sure other is left in an empty state, otherwise we might
            // have in principle assertion failures in the destructor of other
            // in debug mode.
            other.clear();
        }
        return *this;
    }
    ~tree()
    {
#if !defined(NDEBUG)
        // Run various debug checks.
        for (std::size_t j = 0; j < NDim; ++j) {
            // All particle data vectors have the same size.
            assert(m_parts[NDim].size() == m_parts[j].size());
        }
        // Same number of particles and codes.
        assert(m_parts[0].size() == m_codes.size());
        // Codes are sorted.
        assert(std::is_sorted(m_codes.begin(), m_codes.end()));
        // The size of m_perm, m_last_perm and m_inv_perm is the number of particles.
        assert(m_parts[0].size() == m_perm.size());
        assert(m_parts[0].size() == m_last_perm.size());
        assert(m_parts[0].size() == m_inv_perm.size());
        // All coordinates must fit in the box, and they need to correspond
        // to the correct Morton code.
        std::array<UInt, NDim> tmp_dcoord;
        morton_encoder<NDim, UInt> me;
        for (size_type i = 0; i < m_parts[NDim].size(); ++i) {
            for (std::size_t j = 0; j < NDim; ++j) {
                assert(m_parts[j][i] < m_box_size / F(2));
                assert(m_parts[j][i] >= -m_box_size / F(2));
            }
            disc_coords(tmp_dcoord, i);
            assert(m_codes[i] == me(tmp_dcoord.data()));
        }
        // m_inv_perm and m_perm are consistent with each other.
        for (decltype(m_perm.size()) i = 0; i < m_perm.size(); ++i) {
            assert(m_perm[i] < m_inv_perm.size());
            assert(m_inv_perm[m_perm[i]] == i);
        }
        // m_perm does not contain duplicates.
        std::sort(m_perm.begin(), m_perm.end());
        assert(std::unique(m_perm.begin(), m_perm.end()) == m_perm.end());
        // m_last_perm does not contain duplicates.
        std::sort(m_last_perm.begin(), m_last_perm.end());
        assert(std::unique(m_last_perm.begin(), m_last_perm.end()) == m_last_perm.end());
        // Check min/max as well.
        if (m_parts[0].size()) {
            assert(m_last_perm[0] == 0u);
            assert(m_last_perm.back() == m_parts[0].size() - 1u);
        }
#endif
    }
    // Reset the state of the tree to a known one, i.e., a def-cted tree.
    void clear() noexcept
    {
        m_box_size = F(0);
        m_box_size_deduced = false;
        m_max_leaf_n = default_max_leaf_n;
        m_ncrit = default_ncrit;
        for (auto &p : m_parts) {
            p.clear();
        }
        m_codes.clear();
        m_perm.clear();
        m_last_perm.clear();
        m_inv_perm.clear();
        m_tree.clear();
        m_crit_nodes.clear();
    }
    // Tree pretty printing. Will print up to max_nodes, or all the nodes if max_nodes is zero.
    std::ostream &pprint(std::ostream &os, size_type max_nodes = 0) const
    {
        // NOTE: sanity check for the use of UInt in std::bitset.
        static_assert(unsigned(std::numeric_limits<UInt>::digits) <= std::numeric_limits<std::size_t>::max());
        const auto n_nodes = m_tree.size();
        os << "Box size                 : " << m_box_size << (m_box_size_deduced ? " (deduced)" : "") << '\n';
        os << "Total number of particles: " << m_codes.size() << '\n';
        os << "Total number of nodes    : " << n_nodes << "\n\n";
        if (!n_nodes) {
            // Exit early if there are no nodes.
            return os;
        }
        if (max_nodes && max_nodes < n_nodes) {
            // We have a limit and the limit is smaller than the number
            // of nodes. We will be printing only some of the nodes.
            os << "First " << max_nodes << " nodes:\n";
        } else {
            // Either we don't have a limit, or the limit is equal to or larger
            // than the number of nodes. We will be printing all nodes.
            os << "Nodes:\n";
        }
        // Init the number of nodes printed so far.
        size_type n_printed = 0;
        for (const auto &node : m_tree) {
            // Print the node.
            os << std::bitset<std::numeric_limits<UInt>::digits>(node.code) << '|' << node.begin << ',' << node.end
               << ',' << node.n_children << "|" << node.props[NDim] << "|[";
            for (std::size_t j = 0; j < NDim; ++j) {
                os << node.props[j];
                if (j < NDim - 1u) {
                    os << ", ";
                }
            }
            os << "]\n";
            // Increase the printed nodes counter and check against max_nodes.
            // NOTE: if max_nodes is zero, this condition will never be true, so we will end
            // up printing all the nodes.
            if (++n_printed == max_nodes) {
                // We have a limit for the max number of printed nodes, and we hit it. Break out.
                break;
            }
        }
        if (n_printed < n_nodes) {
            // We have not printed all the nodes in the tree. Add an ellipsis.
            os << "...\n";
        }
        return os;
    }
    friend std::ostream &operator<<(std::ostream &os, const tree &t)
    {
        // Hard code to 20 max nodes for the streaming operator.
        return t.pprint(os, 20);
    }

private:
    // Helpers to compute how many vectors we will need to store temporary
    // data during the BH check.
    // Q == 0 -> accelerations only, NDim + 1 vectors (temporary diffs + 1/dist3)
    // Q == 1 -> potentials only, 1 vector (1/dist)
    // Q == 2 -> accs + pots, NDim + 2u vectors (temporary diffs + 1/dist3 + 1/dist)
    template <unsigned Q>
    static constexpr std::size_t compute_nvecs_tmp()
    {
        static_assert(Q <= 2u);
        return static_cast<std::size_t>(Q == 0u ? NDim + 1u : (Q == 1u ? 1u : NDim + 2u));
    }
    template <unsigned Q>
    static constexpr std::size_t nvecs_tmp = compute_nvecs_tmp<Q>();
    // Storage for temporary data computed during the BH check (which will be re-used
    // by another function).
    template <unsigned Q>
    static auto &acc_pot_tmp_vecs()
    {
        static thread_local std::array<fp_vector, nvecs_tmp<Q>> tmp_vecs;
        return tmp_vecs;
    }
    // Helpers to compute how many vectors we will need to store the results
    // of the computation of the accelerations/potentials.
    // Q == 0 -> accelerations only, NDim vectors
    // Q == 1 -> potentials only, 1 vector
    // Q == 2 -> accs + pots, NDim + 1 vectors
    template <unsigned Q>
    static constexpr std::size_t nvecs_res = tree_nvecs_res<Q, NDim>;
    // Temporary storage to accumulate the accelerations/potentials induced on the
    // particles of a critical node. Data in here will be copied to
    // the output arrays after the accelerations/potentials from all the other
    // particles/nodes in the domain have been computed.
    template <unsigned Q>
    static auto &acc_pot_tmp_res()
    {
        static thread_local std::array<fp_vector, nvecs_res<Q>> tmp_res;
        return tmp_res;
    }
    // Temporary vectors to store the data of a target node during traversal.
    static auto &tgt_tmp_data()
    {
        static thread_local std::array<fp_vector, NDim + 1u> tmp_tgt;
        return tmp_tgt;
    }
    // Compute the element-wise accelerations on the batch of particles at xvec1, yvec1, zvec1 by the
    // particles at xvec2, yvec2, zvec2 with masses mvec2, and add the result into res_x_vec, res_y_vec,
    // res_z_vec. eps2_vec is the square of the softening length.
    template <typename B>
    static void batch_batch_3d_accs(B &res_x_vec, B &res_y_vec, B &res_z_vec, B xvec1, B yvec1, B zvec1, B xvec2,
                                    B yvec2, B zvec2, B mvec2, B eps2_vec)
    {
        const B diff_x = xvec2 - xvec1, diff_y = yvec2 - yvec1, diff_z = zvec2 - zvec1,
                dist2 = diff_x * diff_x + diff_y * diff_y + xsimd_fma(diff_z, diff_z, eps2_vec);
        B m2_dist3;
        if constexpr (use_fast_inv_sqrt<B>) {
            m2_dist3 = mvec2 * inv_sqrt_3(dist2);
        } else {
            const B dist = xsimd_sqrt(dist2);
            const B dist3 = dist * dist2;
            m2_dist3 = mvec2 / dist3;
        }
        res_x_vec = xsimd_fma(diff_x, m2_dist3, res_x_vec);
        res_y_vec = xsimd_fma(diff_y, m2_dist3, res_y_vec);
        res_z_vec = xsimd_fma(diff_z, m2_dist3, res_z_vec);
    }
    // Compute the element-wise mutual potential for 2 sets of particles at the specified coordinates and
    // with the specified masses. eps2_vec is the square of the softening length. The return value
    // is the negated mutual potential (that is, m1 * m2 / dist).
    template <typename B>
    static B batch_batch_3d_pots(B xvec1, B yvec1, B zvec1, B mvec1, B xvec2, B yvec2, B zvec2, B mvec2, B eps2_vec)
    {
        const B diff_x = xvec2 - xvec1, diff_y = yvec2 - yvec1, diff_z = zvec2 - zvec1,
                dist2 = diff_x * diff_x + diff_y * diff_y + xsimd_fma(diff_z, diff_z, eps2_vec);
        B m1_dist;
        if constexpr (use_fast_inv_sqrt<B>) {
            m1_dist = mvec1 * inv_sqrt(dist2);
        } else {
            m1_dist = mvec1 / xsimd_sqrt(dist2);
        }
        return m1_dist * mvec2;
    }
    // A combination of the 2 functions above, computing both accelerations and potentials between 2 batches.
    // For both the accs and pots, the results are accumulated into the return values.
    template <typename B>
    static void batch_batch_3d_accs_pots(B &res_x_vec, B &res_y_vec, B &res_z_vec, B &res_pot_vec, B xvec1, B yvec1,
                                         B zvec1, B mvec1, B xvec2, B yvec2, B zvec2, B mvec2, B eps2_vec)
    {
        const B diff_x = xvec2 - xvec1, diff_y = yvec2 - yvec1, diff_z = zvec2 - zvec1,
                dist2 = diff_x * diff_x + diff_y * diff_y + xsimd_fma(diff_z, diff_z, eps2_vec);
        B m2_dist, m2_dist3;
        if constexpr (use_fast_inv_sqrt<B>) {
            const auto tmp = inv_sqrt(dist2), tmp3 = tmp * tmp * tmp;
            m2_dist = mvec2 * tmp;
            m2_dist3 = mvec2 * tmp3;
        } else {
            // NOTE: as in the self interaction function and in the leaf-target interaction function,
            // we should check if it is worthwhile to reduce the number of divisions here.
            const auto dist = xsimd_sqrt(dist2), dist3 = dist2 * dist;
            m2_dist = mvec2 / dist;
            m2_dist3 = mvec2 / dist3;
        }
        // Write out the results.
        res_x_vec = xsimd_fma(diff_x, m2_dist3, res_x_vec);
        res_y_vec = xsimd_fma(diff_y, m2_dist3, res_y_vec);
        res_z_vec = xsimd_fma(diff_z, m2_dist3, res_z_vec);
        // NOTE: need a negated FMA for the potential.
        res_pot_vec = xsimd_fnma(mvec1, m2_dist, res_pot_vec);
    }
    // Function to compute the self-interactions within a target node. eps2 is the square of the softening length,
    // tgt_size is the number of particles in the target node, p_ptrs pointers to the target particles'
    // coordinates/masses, res_ptrs pointers to the output arrays. Q indicates which quantities will be computed
    // (accs, potentials, or both).
    template <unsigned Q>
    void tree_self_interactions(F eps2, size_type tgt_size, const std::array<const F *, NDim + 1u> &p_ptrs,
                                const std::array<F *, nvecs_res<Q>> &res_ptrs) const
    {
        if constexpr (simd_enabled && NDim == 3u) {
            // xsimd batch type.
            using batch_type = xsimd::simd_type<F>;
            // Size of batch_type.
            constexpr auto batch_size = batch_type::size;
            // Shortcuts to the node coordinates/masses.
            const auto [x_ptr, y_ptr, z_ptr, m_ptr] = p_ptrs;
            // Softening length, vector version.
            const batch_type eps2_vec(eps2);
            if constexpr (Q == 0u) {
                // Q == 0, accelerations only.
                //
                // Shortcuts to the result vectors.
                const auto [res_x, res_y, res_z] = res_ptrs;
                for (size_type i1 = 0; i1 < tgt_size; i1 += batch_size) {
                    // Load the first batch of particles.
                    const auto xvec1 = xsimd::load_aligned(x_ptr + i1), yvec1 = xsimd::load_aligned(y_ptr + i1),
                               zvec1 = xsimd::load_aligned(z_ptr + i1), mvec1 = xsimd::load_aligned(m_ptr + i1);
                    // Init the accumulators for the accelerations on the first batch of particles.
                    batch_type res_x_vec1(F(0)), res_y_vec1(F(0)), res_z_vec1(F(0));
                    // Now we iterate over the node particles starting 1 position past i1 (to avoid self interactions).
                    // This is the classical n body inner loop.
                    for (size_type i2 = i1 + 1u; i2 < tgt_size; ++i2) {
                        // Load the second batch of particles.
                        const auto xvec2 = xsimd::load_unaligned(x_ptr + i2), yvec2 = xsimd::load_unaligned(y_ptr + i2),
                                   zvec2 = xsimd::load_unaligned(z_ptr + i2), mvec2 = xsimd::load_unaligned(m_ptr + i2);
                        // NOTE: now we are going to do a slight repetition of batch_batch_3d_accs(), with the goal
                        // of avoiding doing extra needless computations.
                        // Compute the relative positions of 2 wrt 1, and the distance square.
                        const auto diff_x = xvec2 - xvec1, diff_y = yvec2 - yvec1, diff_z = zvec2 - zvec1,
                                   dist2 = diff_x * diff_x + diff_y * diff_y + xsimd_fma(diff_z, diff_z, eps2_vec);
                        // Compute m1/dist3 and m2/dist3.
                        batch_type m1_dist3, m2_dist3;
                        if constexpr (use_fast_inv_sqrt<batch_type>) {
                            const auto tmp = inv_sqrt_3(dist2);
                            m1_dist3 = mvec1 * tmp;
                            m2_dist3 = mvec2 * tmp;
                        } else {
                            // NOTE: here it might be beneficial to compute 1/sqrt()
                            // and get rid of a division, paying instead a couple of extra
                            // multiplications. To be benchmarked.
                            const auto dist = xsimd_sqrt(dist2);
                            const auto dist3 = dist * dist2;
                            m1_dist3 = mvec1 / dist3;
                            m2_dist3 = mvec2 / dist3;
                        }
                        // Add to the accumulators for 1 the accelerations due to the batch 2.
                        res_x_vec1 = xsimd_fma(diff_x, m2_dist3, res_x_vec1);
                        res_y_vec1 = xsimd_fma(diff_y, m2_dist3, res_y_vec1);
                        res_z_vec1 = xsimd_fma(diff_z, m2_dist3, res_z_vec1);
                        // Add *directly into the result buffer* the acceleration on 2 due to 1.
                        xsimd_fnma(diff_x, m1_dist3, xsimd::load_unaligned(res_x + i2)).store_unaligned(res_x + i2);
                        xsimd_fnma(diff_y, m1_dist3, xsimd::load_unaligned(res_y + i2)).store_unaligned(res_y + i2);
                        xsimd_fnma(diff_z, m1_dist3, xsimd::load_unaligned(res_z + i2)).store_unaligned(res_z + i2);
                    }
                    // Add the accumulated acceleration on 1 to the values already in the result buffer.
                    (xsimd::load_aligned(res_x + i1) + res_x_vec1).store_aligned(res_x + i1);
                    (xsimd::load_aligned(res_y + i1) + res_y_vec1).store_aligned(res_y + i1);
                    (xsimd::load_aligned(res_z + i1) + res_z_vec1).store_aligned(res_z + i1);
                }
            } else if constexpr (Q == 1u) {
                // Q == 1, potentials only.
                //
                // Shortcut to the result vector.
                const auto res = res_ptrs[0];
                for (size_type i1 = 0; i1 < tgt_size; i1 += batch_size) {
                    // Load the first batch of particles.
                    const auto xvec1 = xsimd::load_aligned(x_ptr + i1), yvec1 = xsimd::load_aligned(y_ptr + i1),
                               zvec1 = xsimd::load_aligned(z_ptr + i1), mvec1 = xsimd::load_aligned(m_ptr + i1);
                    // Init the accumulator for the potential on the first batch of particles.
                    batch_type res_vec(F(0));
                    // Now we iterate over the node particles starting 1 position past i1 (to avoid self interactions).
                    // This is the classical n body inner loop.
                    for (size_type i2 = i1 + 1u; i2 < tgt_size; ++i2) {
                        // Load the second batch of particles.
                        const auto xvec2 = xsimd::load_unaligned(x_ptr + i2), yvec2 = xsimd::load_unaligned(y_ptr + i2),
                                   zvec2 = xsimd::load_unaligned(z_ptr + i2), mvec2 = xsimd::load_unaligned(m_ptr + i2);
                        // NOTE: now we are going to do a slight repetition of batch_batch_3d_pots(), with the goal
                        // of avoiding doing extra needless computations.
                        // Compute the relative positions of 2 wrt 1, and the distance square.
                        const auto diff_x = xvec2 - xvec1, diff_y = yvec2 - yvec1, diff_z = zvec2 - zvec1,
                                   dist2 = diff_x * diff_x + diff_y * diff_y + xsimd_fma(diff_z, diff_z, eps2_vec);
                        // Compute m1/dist.
                        batch_type m1_dist;
                        if constexpr (use_fast_inv_sqrt<batch_type>) {
                            m1_dist = mvec1 * inv_sqrt(dist2);
                        } else {
                            m1_dist = mvec1 / xsimd_sqrt(dist2);
                        }
                        // Compute the mutual (negated) potential between 1 and 2.
                        const auto mut_pot = m1_dist * mvec2;
                        // Subtract it from the acccumulator for 1.
                        res_vec -= mut_pot;
                        // Subtract *directly from the result buffer* the mutual negated potential for 2.
                        (xsimd::load_unaligned(res + i2) - mut_pot).store_unaligned(res + i2);
                    }
                    // Add the accumulated potentials on 1 from the values already in the result buffer.
                    // NOTE: we are doing an add here because we already built res_vec with repeated
                    // subtractions, thus generating a negative value.
                    (xsimd::load_aligned(res + i1) + res_vec).store_aligned(res + i1);
                }
            } else {
                // Q == 2, accelerations and potentials.
                //
                // Shortcuts to the result vectors.
                const auto [res_x, res_y, res_z, res_pot] = res_ptrs;
                for (size_type i1 = 0; i1 < tgt_size; i1 += batch_size) {
                    // Load the first batch of particles.
                    const auto xvec1 = xsimd::load_aligned(x_ptr + i1), yvec1 = xsimd::load_aligned(y_ptr + i1),
                               zvec1 = xsimd::load_aligned(z_ptr + i1), mvec1 = xsimd::load_aligned(m_ptr + i1);
                    // Init the accumulators for the accelerations/potentials on the first batch of particles.
                    batch_type res_x_vec1(F(0)), res_y_vec1(F(0)), res_z_vec1(F(0)), res_pot_vec(F(0));
                    // Now we iterate over the node particles starting 1 position past i1 (to avoid self interactions).
                    // This is the classical n body inner loop.
                    for (size_type i2 = i1 + 1u; i2 < tgt_size; ++i2) {
                        // Load the second batch of particles.
                        const auto xvec2 = xsimd::load_unaligned(x_ptr + i2), yvec2 = xsimd::load_unaligned(y_ptr + i2),
                                   zvec2 = xsimd::load_unaligned(z_ptr + i2), mvec2 = xsimd::load_unaligned(m_ptr + i2);
                        // NOTE: now we are going to do a slight repetition of batch_batch_3d_accs_pots(), with the goal
                        // of avoiding doing extra needless computations.
                        // Compute the relative positions of 2 wrt 1, and the distance square.
                        const auto diff_x = xvec2 - xvec1, diff_y = yvec2 - yvec1, diff_z = zvec2 - zvec1,
                                   dist2 = diff_x * diff_x + diff_y * diff_y + xsimd_fma(diff_z, diff_z, eps2_vec);
                        // Compute m1/dist, m1/dist3 and m2/dist3.
                        batch_type m1_dist, m1_dist3, m2_dist3;
                        if constexpr (use_fast_inv_sqrt<batch_type>) {
                            const auto tmp = inv_sqrt(dist2);
                            m1_dist = mvec1 * tmp;
                            const auto tmp3 = tmp * tmp * tmp;
                            m1_dist3 = mvec1 * tmp3;
                            m2_dist3 = mvec2 * tmp3;
                        } else {
                            // NOTE: as hinted above in the Q == 0 case, perhaps we can avoid
                            // some of these divisions.
                            const auto dist = xsimd_sqrt(dist2);
                            m1_dist = mvec1 / dist;
                            const auto dist3 = dist * dist2;
                            m1_dist3 = mvec1 / dist3;
                            m2_dist3 = mvec2 / dist3;
                        }
                        // Compute the mutual (negated) potential between 1 and 2.
                        const auto mut_pot = m1_dist * mvec2;
                        // Add to the accumulators for 1 the accelerations due to the batch 2,
                        // and subtract the mutual negated potential from the acccumulator for 1.
                        res_x_vec1 = xsimd_fma(diff_x, m2_dist3, res_x_vec1);
                        res_y_vec1 = xsimd_fma(diff_y, m2_dist3, res_y_vec1);
                        res_z_vec1 = xsimd_fma(diff_z, m2_dist3, res_z_vec1);
                        res_pot_vec -= mut_pot;
                        // Add *directly into the result buffer* the acceleration on 2 due to 1,
                        // and subtract the negated potential.
                        xsimd_fnma(diff_x, m1_dist3, xsimd::load_unaligned(res_x + i2)).store_unaligned(res_x + i2);
                        xsimd_fnma(diff_y, m1_dist3, xsimd::load_unaligned(res_y + i2)).store_unaligned(res_y + i2);
                        xsimd_fnma(diff_z, m1_dist3, xsimd::load_unaligned(res_z + i2)).store_unaligned(res_z + i2);
                        (xsimd::load_unaligned(res_pot + i2) - mut_pot).store_unaligned(res_pot + i2);
                    }
                    // Add the accumulated accelerations/potentials on 1 to the values already in the result buffer.
                    (xsimd::load_aligned(res_x + i1) + res_x_vec1).store_aligned(res_x + i1);
                    (xsimd::load_aligned(res_y + i1) + res_y_vec1).store_aligned(res_y + i1);
                    (xsimd::load_aligned(res_z + i1) + res_z_vec1).store_aligned(res_z + i1);
                    // NOTE: the accumulated potential is added because it was constructed as a negative quantity.
                    (xsimd::load_aligned(res_pot + i1) + res_pot_vec).store_aligned(res_pot + i1);
                }
            }
        } else {
            // Pointer to the masses.
            const auto m_ptr = p_ptrs[NDim];
            // Temporary vectors to be used in the loops below.
            std::array<F, NDim> diffs, pos1;
            for (size_type i1 = 0; i1 < tgt_size; ++i1) {
                // Load the coords of the current particle.
                for (std::size_t j = 0; j < NDim; ++j) {
                    pos1[j] = p_ptrs[j][i1];
                }
                // Load the mass of the current particle.
                const auto m1 = m_ptr[i1];
                // The acceleration/potential vector for the current particle
                // (inited to zero).
                std::array<F, nvecs_res<Q>> a1{};
                for (size_type i2 = i1 + 1u; i2 < tgt_size; ++i2) {
                    // Determine dist2, dist and dist3.
                    F dist2(eps2);
                    for (std::size_t j = 0; j < NDim; ++j) {
                        diffs[j] = p_ptrs[j][i2] - pos1[j];
                        dist2 = fma_wrap(diffs[j], diffs[j], dist2);
                    }
                    const auto dist = std::sqrt(dist2), m2 = m_ptr[i2];
                    if constexpr (Q == 0u || Q == 2u) {
                        // Q == 0 or 2: accelerations are requested.
                        const auto dist3 = dist2 * dist, m2_dist3 = m2 / dist3, m1_dist3 = m1 / dist3;
                        // Accumulate the accelerations, both in the local
                        // accumulator for the current particle and in the global
                        // acc vector for the opposite acceleration.
                        for (std::size_t j = 0; j < NDim; ++j) {
                            a1[j] = fma_wrap(m2_dist3, diffs[j], a1[j]);
                            // NOTE: this is a fused negated multiply-add.
                            res_ptrs[j][i2] = fma_wrap(m1_dist3, -diffs[j], res_ptrs[j][i2]);
                        }
                    }
                    if constexpr (Q == 1u || Q == 2u) {
                        // Q == 1 or 2: potentials are requested.
                        // Establish the index of the potential in the result array:
                        // 0 if only the potentials are requested, NDim otherwise.
                        constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim);
                        // Compute the negated mutual potential.
                        const auto mut_pot = m1 / dist * m2;
                        // Subtract mut_pot from the accumulator for the current particle and from
                        // the total potential of particle i2.
                        a1[pot_idx] -= mut_pot;
                        res_ptrs[pot_idx][i2] -= mut_pot;
                    }
                }
                // Update the acceleration/potential on the first particle
                // in the temporary storage.
                if constexpr (Q == 0u || Q == 2u) {
                    for (std::size_t j = 0; j < NDim; ++j) {
                        res_ptrs[j][i1] += a1[j];
                    }
                }
                if constexpr (Q == 1u || Q == 2u) {
                    constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim);
                    // NOTE: addition, because the value in a1[pot_idx] was already built
                    // as a negative quantity.
                    res_ptrs[pot_idx][i1] += a1[pot_idx];
                }
            }
        }
    }
    // Function to compute the accelerations/potentials on a target node by all the particles of a leaf source node.
    // eps2 is the square of the softening length, src_idx is the index, in the tree structure, of the leaf node,
    // tgt_size the number of particles in the target node, p_ptrs pointers to the target particles' coordinates/masses,
    // res_ptrs pointers to the output arrays. Q indicates which quantities will be computed (accs, potentials, or
    // both).
    template <unsigned Q>
    void tree_acc_pot_leaf(F eps2, size_type src_idx, size_type tgt_size,
                           const std::array<const F *, NDim + 1u> &p_ptrs,
                           const std::array<F *, nvecs_res<Q>> &res_ptrs) const
    {
        // Get a reference to the source node.
        const auto &src_node = m_tree[src_idx];
        // Establish the range of the source node.
        const auto src_begin = src_node.begin, src_end = src_node.end;
        if constexpr (simd_enabled && NDim == 3u) {
            // The SIMD-accelerated version.
            using batch_type = xsimd::simd_type<F>;
            constexpr auto batch_size = batch_type::size;
            // The number of particles in the source node.
            const auto src_size = static_cast<size_type>(src_end - src_begin);
            // Vector version of eps2.
            const batch_type eps2_vec(eps2);
            // Pointers to the target node data.
            const auto [x_ptr1, y_ptr1, z_ptr1, m_ptr1] = p_ptrs;
            // Pointers to the source node data.
            const auto x_ptr2 = m_parts[0].data() + src_begin, y_ptr2 = m_parts[1].data() + src_begin,
                       z_ptr2 = m_parts[2].data() + src_begin, m_ptr2 = m_parts[3].data() + src_begin;
            if constexpr (Q == 0u) {
                // Q == 0, accelerations only.
                //
                // Pointers to the result data.
                const auto [res_x, res_y, res_z] = res_ptrs;
                for (size_type i = 0; i < tgt_size; i += batch_size) {
                    // Load the current batch of target data.
                    const auto xvec1 = batch_type(x_ptr1 + i, xsimd::aligned_mode{}),
                               yvec1 = batch_type(y_ptr1 + i, xsimd::aligned_mode{}),
                               zvec1 = batch_type(z_ptr1 + i, xsimd::aligned_mode{});
                    // Init the batches for computing the accelerations, loading the
                    // accumulated acceleration for the current batch.
                    auto res_x_vec = batch_type(res_x + i, xsimd::aligned_mode{}),
                         res_y_vec = batch_type(res_y + i, xsimd::aligned_mode{}),
                         res_z_vec = batch_type(res_z + i, xsimd::aligned_mode{});
                    for (size_type j = 0; j < src_size; ++j) {
                        // Compute the interaction with the source particle.
                        batch_batch_3d_accs(res_x_vec, res_y_vec, res_z_vec, xvec1, yvec1, zvec1, batch_type(x_ptr2[j]),
                                            batch_type(y_ptr2[j]), batch_type(z_ptr2[j]), batch_type(m_ptr2[j]),
                                            eps2_vec);
                    }
                    // Store the updated accelerations in the temporary vectors.
                    res_x_vec.store_aligned(res_x + i);
                    res_y_vec.store_aligned(res_y + i);
                    res_z_vec.store_aligned(res_z + i);
                }
            } else if constexpr (Q == 1u) {
                // Q == 1, potentials only.
                //
                // Pointer to the result data.
                const auto res = res_ptrs[0];
                for (size_type i = 0; i < tgt_size; i += batch_size) {
                    // Load the current batch of target data.
                    const auto xvec1 = batch_type(x_ptr1 + i, xsimd::aligned_mode{}),
                               yvec1 = batch_type(y_ptr1 + i, xsimd::aligned_mode{}),
                               zvec1 = batch_type(z_ptr1 + i, xsimd::aligned_mode{}),
                               mvec1 = batch_type(m_ptr1 + i, xsimd::aligned_mode{});
                    // Init the batch for computing the potentials, loading the
                    // accumulated potentials for the current batch.
                    auto res_vec = batch_type(res + i, xsimd::aligned_mode{});
                    for (size_type j = 0; j < src_size; ++j) {
                        // Compute the interaction with the source particle,
                        // and subtract the obtained potentials from the current
                        // accumulated value.
                        // NOTE: need a subtraction because batch_batch_3d_pots() returns
                        // the negated potential.
                        res_vec -= batch_batch_3d_pots(xvec1, yvec1, zvec1, mvec1, batch_type(x_ptr2[j]),
                                                       batch_type(y_ptr2[j]), batch_type(z_ptr2[j]),
                                                       batch_type(m_ptr2[j]), eps2_vec);
                    }
                    // Store the updated potentials in the temporary vector.
                    res_vec.store_aligned(res + i);
                }
            } else {
                // Q == 2, accelerations and potentials.
                //
                // Pointers to the result data.
                const auto [res_x, res_y, res_z, res_pot] = res_ptrs;
                for (size_type i = 0; i < tgt_size; i += batch_size) {
                    // Load the current batch of target data.
                    const auto xvec1 = batch_type(x_ptr1 + i, xsimd::aligned_mode{}),
                               yvec1 = batch_type(y_ptr1 + i, xsimd::aligned_mode{}),
                               zvec1 = batch_type(z_ptr1 + i, xsimd::aligned_mode{}),
                               mvec1 = batch_type(m_ptr1 + i, xsimd::aligned_mode{});
                    // Init the batches for computing the accelerations and the potentials, loading the
                    // accumulated values for the current batch.
                    auto res_x_vec = batch_type(res_x + i, xsimd::aligned_mode{}),
                         res_y_vec = batch_type(res_y + i, xsimd::aligned_mode{}),
                         res_z_vec = batch_type(res_z + i, xsimd::aligned_mode{}),
                         res_pot_vec = batch_type(res_pot + i, xsimd::aligned_mode{});
                    for (size_type j = 0; j < src_size; ++j) {
                        // Compute the interaction with the source particle.
                        batch_batch_3d_accs_pots(res_x_vec, res_y_vec, res_z_vec, res_pot_vec, xvec1, yvec1, zvec1,
                                                 mvec1, batch_type(x_ptr2[j]), batch_type(y_ptr2[j]),
                                                 batch_type(z_ptr2[j]), batch_type(m_ptr2[j]), eps2_vec);
                    }
                    // Store the updated accelerations/potentials in the temporary vectors.
                    res_x_vec.store_aligned(res_x + i);
                    res_y_vec.store_aligned(res_y + i);
                    res_z_vec.store_aligned(res_z + i);
                    res_pot_vec.store_aligned(res_pot + i);
                }
            }
        } else {
            // Local variables for the scalar computation.
            std::array<F, NDim> pos1, diffs;
            for (size_type i1 = 0; i1 < tgt_size; ++i1) {
                // Load the coordinates of the current particle
                // in the target node.
                for (std::size_t j = 0; j < NDim; ++j) {
                    pos1[j] = p_ptrs[j][i1];
                }
                // Load the target mass, but only if we are interested in the potentials.
                [[maybe_unused]] F m1;
                if constexpr (Q == 1u || Q == 2u) {
                    m1 = p_ptrs[NDim][i1];
                }
                // Iterate over the particles in the src node.
                for (size_type i2 = src_begin; i2 < src_end; ++i2) {
                    F dist2(eps2);
                    for (std::size_t j = 0; j < NDim; ++j) {
                        diffs[j] = m_parts[j][i2] - pos1[j];
                        dist2 = fma_wrap(diffs[j], diffs[j], dist2);
                    }
                    const auto dist = std::sqrt(dist2), m2 = m_parts[NDim][i2];
                    if constexpr (Q == 0u || Q == 2u) {
                        // Q == 0 or 2: accelerations are requested.
                        const auto dist3 = dist * dist2, m_dist3 = m2 / dist3;
                        for (std::size_t j = 0; j < NDim; ++j) {
                            res_ptrs[j][i1] = fma_wrap(diffs[j], m_dist3, res_ptrs[j][i1]);
                        }
                    }
                    if constexpr (Q == 1u || Q == 2u) {
                        // Q == 1 or 2: potentials are requested.
                        // Establish the index of the potential in the result array:
                        // 0 if only the potentials are requested, NDim otherwise.
                        constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim);
                        res_ptrs[pot_idx][i1] = fma_wrap(-m1, m2 / dist, res_ptrs[pot_idx][i1]);
                    }
                }
            }
        }
    }
    // Function to compute the accelerations/potentials due to the COM of a source node onto a target node. src_idx is
    // the index, in the tree structure, of the source node, tgt_size the number of particles in the target node,
    // p_ptrs pointers to the target particles' coordinates/masses, tmp_ptrs are pointers to the temporary data filled
    // in by the tree_acc_pot_bh_check() function (which will be re-used by this function), res_ptrs pointers to the
    // output arrays. Q indicates which quantities will be computed (accs, potentials, or both).
    template <unsigned Q>
    void tree_acc_pot_bh_com(size_type src_idx, size_type tgt_size, const std::array<const F *, NDim + 1u> &p_ptrs,
                             const std::array<F *, nvecs_tmp<Q>> &tmp_ptrs,
                             const std::array<F *, nvecs_res<Q>> &res_ptrs) const
    {
        // Load locally the mass of the source node.
        const auto m_src = m_tree[src_idx].props[NDim];
        if constexpr (simd_enabled && NDim == 3u) {
            using batch_type = xsimd::simd_type<F>;
            constexpr auto batch_size = batch_type::size;
            // Vector version of the source node mass.
            const batch_type m_src_vec(m_src);
            if constexpr (Q == 0u) {
                // Q == 0, accelerations only.
                //
                // Pointers to the temporary coordinate diffs and 1/dist3 values computed in the BH check.
                const auto [tmp_x, tmp_y, tmp_z, tmp_dist3] = tmp_ptrs;
                // Pointers to the result arrays.
                const auto [res_x, res_y, res_z] = res_ptrs;
                for (size_type i = 0; i < tgt_size; i += batch_size) {
                    // Compute m_src/dist**3 and load the differences.
                    const auto m_src_dist3_vec = use_fast_inv_sqrt<batch_type>
                                                     ? m_src_vec * batch_type(tmp_dist3 + i, xsimd::aligned_mode{})
                                                     : m_src_vec / batch_type(tmp_dist3 + i, xsimd::aligned_mode{}),
                               xdiff = batch_type(tmp_x + i, xsimd::aligned_mode{}),
                               ydiff = batch_type(tmp_y + i, xsimd::aligned_mode{}),
                               zdiff = batch_type(tmp_z + i, xsimd::aligned_mode{});
                    // Compute and accumulate the accelerations.
                    xsimd_fma(xdiff, m_src_dist3_vec, batch_type(res_x + i, xsimd::aligned_mode{}))
                        .store_aligned(res_x + i);
                    xsimd_fma(ydiff, m_src_dist3_vec, batch_type(res_y + i, xsimd::aligned_mode{}))
                        .store_aligned(res_y + i);
                    xsimd_fma(zdiff, m_src_dist3_vec, batch_type(res_z + i, xsimd::aligned_mode{}))
                        .store_aligned(res_z + i);
                }
            } else if constexpr (Q == 1u) {
                // Q == 1, potentials only.
                //
                // Pointer to the temporary 1/dist values computed in the BH check.
                const auto tmp_dist = tmp_ptrs[0];
                // Pointer to the target masses.
                const auto m_ptr = p_ptrs[3];
                // Pointer to the result array.
                const auto res = res_ptrs[0];
                for (size_type i = 0; i < tgt_size; i += batch_size) {
                    // Compute m_src/dist.
                    const auto m_src_dist_vec = use_fast_inv_sqrt<batch_type>
                                                    ? m_src_vec * batch_type(tmp_dist + i, xsimd::aligned_mode{})
                                                    : m_src_vec / batch_type(tmp_dist + i, xsimd::aligned_mode{});
                    // Compute and accumulate the potential.
                    xsimd_fnma(batch_type(m_ptr + i, xsimd::aligned_mode{}), m_src_dist_vec,
                               batch_type(res + i, xsimd::aligned_mode{}))
                        .store_aligned(res + i);
                }
            } else {
                // Q == 2, accelerations and potentials.
                //
                // Pointers to the temporary coordinate diffs, 1/dist3 and 1/dist values computed in the BH check.
                const auto [tmp_x, tmp_y, tmp_z, tmp_dist3, tmp_dist] = tmp_ptrs;
                // Pointer to the target masses.
                const auto m_ptr = p_ptrs[3];
                // Pointers to the result arrays.
                const auto [res_x, res_y, res_z, res_pot] = res_ptrs;
                for (size_type i = 0; i < tgt_size; i += batch_size) {
                    // Compute m_src/dist**3, m_src/dist and load the differences.
                    const auto m_src_dist3_vec = use_fast_inv_sqrt<batch_type>
                                                     ? m_src_vec * batch_type(tmp_dist3 + i, xsimd::aligned_mode{})
                                                     : m_src_vec / batch_type(tmp_dist3 + i, xsimd::aligned_mode{}),
                               m_src_dist_vec = use_fast_inv_sqrt<batch_type>
                                                    ? m_src_vec * batch_type(tmp_dist + i, xsimd::aligned_mode{})
                                                    : m_src_vec / batch_type(tmp_dist + i, xsimd::aligned_mode{}),
                               xdiff = batch_type(tmp_x + i, xsimd::aligned_mode{}),
                               ydiff = batch_type(tmp_y + i, xsimd::aligned_mode{}),
                               zdiff = batch_type(tmp_z + i, xsimd::aligned_mode{});
                    // Compute and accumulate the accelerations.
                    xsimd_fma(xdiff, m_src_dist3_vec, batch_type(res_x + i, xsimd::aligned_mode{}))
                        .store_aligned(res_x + i);
                    xsimd_fma(ydiff, m_src_dist3_vec, batch_type(res_y + i, xsimd::aligned_mode{}))
                        .store_aligned(res_y + i);
                    xsimd_fma(zdiff, m_src_dist3_vec, batch_type(res_z + i, xsimd::aligned_mode{}))
                        .store_aligned(res_z + i);
                    // Compute and accumulate the potential.
                    xsimd_fnma(batch_type(m_ptr + i, xsimd::aligned_mode{}), m_src_dist_vec,
                               batch_type(res_pot + i, xsimd::aligned_mode{}))
                        .store_aligned(res_pot + i);
                }
            }
        } else {
            // Init the pointer to the target masses, but only if potentials are requested.
            [[maybe_unused]] const F *m_ptr;
            if constexpr (Q == 1u || Q == 2u) {
                m_ptr = p_ptrs[NDim];
            }
            for (size_type i = 0; i < tgt_size; ++i) {
                if constexpr (Q == 0u || Q == 2u) {
                    // Q == 0 or 2: accelerations are requested.
                    const auto m_src_dist3 = m_src / tmp_ptrs[NDim][i];
                    for (std::size_t j = 0; j < NDim; ++j) {
                        res_ptrs[j][i] = fma_wrap(tmp_ptrs[j][i], m_src_dist3, res_ptrs[j][i]);
                    }
                }
                if constexpr (Q == 1u || Q == 2u) {
                    // Q == 1 or 2: potentials are requested.
                    // Establish the index of the potential in the result array:
                    // 0 if only the potentials are requested, NDim otherwise.
                    constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim);
                    // Establish the index of the dist values in the temp data:
                    // 0 if only the potentials are requested, NDim + 1 otherwise.
                    constexpr auto dist_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim + 1u);
                    res_ptrs[pot_idx][i] = fma_wrap(-m_ptr[i], m_src / tmp_ptrs[dist_idx][i], res_ptrs[pot_idx][i]);
                }
            }
        }
    }
    // Function to check if a source node satisfies the BH criterion and, possibly, to compute the
    // accelerations/potentials due to that source node. src_idx is the index, in the tree structure, of the source
    // node, theta2 the square of the opening angle, eps2 the square of the softening length, tgt_size the number of
    // particles in the target node, p_ptrs pointers to the coordinates/masses of the particles in the target node,
    // res_ptrs pointers to the output arrays. The return value is the index of the next source node in the tree
    // traversal. Q indicates which quantities will be computed (accs, potentials, or both).
    template <unsigned Q>
    size_type tree_acc_pot_bh_check(size_type src_idx, F theta2, F eps2, size_type tgt_size,
                                    const std::array<const F *, NDim + 1u> &p_ptrs,
                                    const std::array<F *, nvecs_res<Q>> &res_ptrs) const
    {
        // Temporary vectors to store the data computed during the BH criterion check.
        // We will re-use this data later in tree_acc_pot_bh_com().
        auto &tmp_vecs = acc_pot_tmp_vecs<Q>();
        static_assert(nvecs_tmp<Q> == std::tuple_size_v<std::remove_reference_t<decltype(tmp_vecs)>>);
        std::array<F *, nvecs_tmp<Q>> tmp_ptrs;
        // NOTE: the size of the vectors in tgt_tmp_data() might be
        // greater than tgt_size, due to SIMD padding. Fetch the
        // actual size.
        const auto pdata_size = tgt_tmp_data()[0].size();
        // Prepare the temporary data.
        if constexpr (Q == 0u || Q == 2u) {
            // Q == 0 or 2: accelerations are requested.
            for (std::size_t j = 0; j < NDim + 1u; ++j) {
                tmp_vecs[j].resize(pdata_size);
                tmp_ptrs[j] = tmp_vecs[j].data();
            }
        }
        if constexpr (Q == 1u || Q == 2u) {
            // Q == 1 or 2: potentials are requested.
            // Establish the index of the dist values in the temp data:
            // 0 if only the potentials are requested, NDim + 1 otherwise.
            constexpr auto dist_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim + 1u);
            tmp_vecs[dist_idx].resize(pdata_size);
            tmp_ptrs[dist_idx] = tmp_vecs[dist_idx].data();
        }
        // Local cache.
        const auto &src_node = m_tree[src_idx];
        // Copy locally the number of children of the source node.
        const auto n_children_src = src_node.n_children;
        // Copy locally the properties of the source node.
        const auto props = src_node.props;
        // Copy locally the dim2 of the source node.
        const auto src_dim2 = src_node.dim2;
        // The flag for the BH criterion check. Initially set to true,
        // it will be set to false if at least one particle in the
        // target node fails the check.
        bool bh_flag = true;
        if constexpr (simd_enabled && NDim == 3u) {
            // The SIMD-accelerated version.
            using batch_type = xsimd::simd_type<F>;
            constexpr auto batch_size = batch_type::size;
            // Splatted vector versions of the scalar variables.
            const batch_type src_dim2_vec(src_dim2), theta2_vec(theta2), eps2_vec(eps2), x_com_vec(props[0]),
                y_com_vec(props[1]), z_com_vec(props[2]);
            // Pointers to the coordinates.
            const auto [x_ptr, y_ptr, z_ptr, m_ptr] = p_ptrs;
            (void)m_ptr;
            if constexpr (Q == 0u) {
                // Q == 0, accelerations only.
                //
                // Pointers to the temporary data.
                const auto [tmp_x, tmp_y, tmp_z, tmp_dist3] = tmp_ptrs;
                for (size_type i = 0; i < tgt_size; i += batch_size) {
                    const auto diff_x = x_com_vec - batch_type(x_ptr + i, xsimd::aligned_mode{}),
                               diff_y = y_com_vec - batch_type(y_ptr + i, xsimd::aligned_mode{}),
                               diff_z = z_com_vec - batch_type(z_ptr + i, xsimd::aligned_mode{});
                    auto dist2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                    if (xsimd::any(src_dim2_vec >= theta2_vec * dist2)) {
                        // At least one particle in the current batch fails the BH criterion
                        // check. Mark the bh_flag as false, then break out.
                        bh_flag = false;
                        break;
                    }
                    // Add the softening length.
                    dist2 += eps2_vec;
                    diff_x.store_aligned(tmp_x + i);
                    diff_y.store_aligned(tmp_y + i);
                    diff_z.store_aligned(tmp_z + i);
                    if constexpr (use_fast_inv_sqrt<batch_type>) {
                        inv_sqrt_3(dist2).store_aligned(tmp_dist3 + i);
                    } else {
                        (xsimd_sqrt(dist2) * dist2).store_aligned(tmp_dist3 + i);
                    }
                }
            } else if constexpr (Q == 1u) {
                // Q == 1, potentials only.
                //
                // Pointer to the temporary data.
                const auto tmp = tmp_ptrs[0];
                for (size_type i = 0; i < tgt_size; i += batch_size) {
                    const auto diff_x = x_com_vec - batch_type(x_ptr + i, xsimd::aligned_mode{}),
                               diff_y = y_com_vec - batch_type(y_ptr + i, xsimd::aligned_mode{}),
                               diff_z = z_com_vec - batch_type(z_ptr + i, xsimd::aligned_mode{});
                    auto dist2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                    if (xsimd::any(src_dim2_vec >= theta2_vec * dist2)) {
                        // At least one particle in the current batch fails the BH criterion
                        // check. Mark the bh_flag as false, then break out.
                        bh_flag = false;
                        break;
                    }
                    // Add the softening length.
                    dist2 += eps2_vec;
                    if constexpr (use_fast_inv_sqrt<batch_type>) {
                        inv_sqrt(dist2).store_aligned(tmp + i);
                    } else {
                        xsimd_sqrt(dist2).store_aligned(tmp + i);
                    }
                }
            } else {
                // Q == 2, accelerations and potentials.
                //
                // Pointers to the temporary data.
                const auto [tmp_x, tmp_y, tmp_z, tmp_dist3, tmp_dist] = tmp_ptrs;
                for (size_type i = 0; i < tgt_size; i += batch_size) {
                    const auto diff_x = x_com_vec - batch_type(x_ptr + i, xsimd::aligned_mode{}),
                               diff_y = y_com_vec - batch_type(y_ptr + i, xsimd::aligned_mode{}),
                               diff_z = z_com_vec - batch_type(z_ptr + i, xsimd::aligned_mode{});
                    auto dist2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                    if (xsimd::any(src_dim2_vec >= theta2_vec * dist2)) {
                        // At least one particle in the current batch fails the BH criterion
                        // check. Mark the bh_flag as false, then break out.
                        bh_flag = false;
                        break;
                    }
                    // Add the softening length.
                    dist2 += eps2_vec;
                    diff_x.store_aligned(tmp_x + i);
                    diff_y.store_aligned(tmp_y + i);
                    diff_z.store_aligned(tmp_z + i);
                    if constexpr (use_fast_inv_sqrt<batch_type>) {
                        const auto inv_dist = inv_sqrt(dist2);
                        (inv_dist * inv_dist * inv_dist).store_aligned(tmp_dist3 + i);
                        inv_dist.store_aligned(tmp_dist + i);
                    } else {
                        const auto dist = xsimd_sqrt(dist2);
                        (dist2 * dist).store_aligned(tmp_dist3 + i);
                        dist.store_aligned(tmp_dist + i);
                    }
                }
            }
        } else {
            // The scalar version.
            for (size_type i = 0; i < tgt_size; ++i) {
                F dist2(0);
                for (std::size_t j = 0; j < NDim; ++j) {
                    const auto diff = props[j] - p_ptrs[j][i];
                    if constexpr (Q == 0u || Q == 2u) {
                        // Store the differences for later use, if we are computing
                        // accelerations.
                        tmp_ptrs[j][i] = diff;
                    }
                    dist2 = fma_wrap(diff, diff, dist2);
                }
                if (src_dim2 >= theta2 * dist2) {
                    // At least one of the particles in the target
                    // node is too close to the COM. Set the flag
                    // to false and exit.
                    bh_flag = false;
                    break;
                }
                // Add the softening length and compute the distance.
                dist2 += eps2;
                const auto dist = std::sqrt(dist2);
                if constexpr (Q == 0u || Q == 2u) {
                    // Q == 0 or 2: accelerations are requested, store dist3.
                    // NOTE: in the scalar part, we always store dist3.
                    tmp_ptrs[NDim][i] = dist * dist2;
                }
                if constexpr (Q == 1u || Q == 2u) {
                    // Q == 1 or 2: potentials are requested.
                    // Establish the index of the dist values in the temp data:
                    // 0 if only the potentials are requested, NDim + 1 otherwise.
                    constexpr auto dist_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim + 1u);
                    // NOTE: in the scalar part, we always store dist.
                    tmp_ptrs[dist_idx][i] = dist;
                }
            }
        }
        if (bh_flag) {
            // The source node satisfies the BH criterion for all the particles of the target node. Add the
            // acceleration due to the com of the source node.
            tree_acc_pot_bh_com<Q>(src_idx, tgt_size, p_ptrs, tmp_ptrs, res_ptrs);
            // We can now skip all the children of the source node.
            return static_cast<size_type>(src_idx + n_children_src + 1u);
        }
        // The source node does not satisfy the BH criterion. We check if it is a leaf
        // node, in which case we need to compute all the pairwise interactions.
        if (!n_children_src) {
            // Leaf node.
            tree_acc_pot_leaf<Q>(eps2, src_idx, tgt_size, p_ptrs, res_ptrs);
        }
        // In any case, we keep traversing the tree moving to the next node in depth-first order.
        return static_cast<size_type>(src_idx + 1u);
    }
    // Tree traversal for the computation of the accelerations/potentials. theta2 is the square of the opening angle,
    // eps2 the square of the softening length, tgt_size the number of particles in the target node, tgt_code its code,
    // p_ptrs are pointers to the coordinates/masses of the particles in the target node, res_ptrs pointers to the
    // output arrays. Q indicates which quantities will be computed (accs, potentials, or both).
    template <unsigned Q>
    void tree_acc_pot(F theta2, F eps2, size_type tgt_size, UInt tgt_code,
                      const std::array<const F *, NDim + 1u> &p_ptrs,
                      const std::array<F *, nvecs_res<Q>> &res_ptrs) const
    {
        assert(!m_tree.empty());
        // Tree level of the target node.
        // NOTE: the target level is available in the tree structure, so in principle we could get
        // it from there. However, the target node is taken from the list of critical nodes, which
        // currently does not store the target node index. Let's keep this in mind if we change
        // the tree building routine.
        const auto tgt_level = tree_level<NDim>(tgt_code);
        // Total size of the tree.
        const auto tree_size = static_cast<size_type>(m_tree.size());
        // Start the iteration over the source nodes.
        for (size_type src_idx = 0; src_idx < tree_size;) {
            // Get a reference to the current source node.
            const auto &src_node = m_tree[src_idx];
            // Extract the code of the source node.
            const auto src_code = src_node.code;
            // Number of children of the source node.
            const auto n_children_src = src_node.n_children;
            // Extract the level of the source node.
            const auto src_level = src_node.level;
            // Compute the shifted target code. This is tgt_code
            // shifted down by the difference between tgt_level
            // and src_level. For instance, in an octree,
            // if the target code is 1 000 000 001 000, then tgt_level
            // is 4, and, if src_level is 2, then the shifted code
            // will be 1 000 000.
            const auto s_tgt_code = static_cast<UInt>(tgt_code >> ((tgt_level - src_level) * NDim));
            if (s_tgt_code == src_code) {
                // The shifted target code coincides with the source code. This means
                // that either the source node is an ancestor of the target node, or it is
                // the target node itself. In the former cases, we just have to continue
                // the depth-first traversal by setting ++src_idx. In the latter case,
                // we want to bump up src_idx by n_children_src + 1 in order to skip
                // the target node and all its children. We will compute later the self
                // interactions in the target node.
                const auto tgt_eq_src_mask = static_cast<size_type>(-(src_code == tgt_code));
                src_idx += 1u + (n_children_src & tgt_eq_src_mask);
            } else {
                // The source node is not an ancestor of the target. We need to run the BH criterion
                // check. The tree_acc_pot_bh_check() function will return the index of the next node
                // in the traversal.
                src_idx = tree_acc_pot_bh_check<Q>(src_idx, theta2, eps2, tgt_size, p_ptrs, res_ptrs);
            }
        }

        // Compute the self interactions within the target node.
        tree_self_interactions<Q>(eps2, tgt_size, p_ptrs, res_ptrs);
    }
    // Top level function for the computation of the accelerations/potentials. out is the array of output iterators,
    // theta2 the square of the opening angle, G the grav constant, eps2 the square of the softening length. Q indicates
    // which quantities will be computed (accs, potentials, or both).
    template <unsigned Q, typename It>
    void acc_pot_impl(const std::array<It, nvecs_res<Q>> &out, F theta2, F G, F eps2) const
    {
        // NOTE: we will be adding padding to the target node data when SIMD is active,
        // in order to be able to use SIMD instructions on all the particles of the node.
        // The padding data must be crafted so that it does not interfere with the useful
        // calculations. This means that:
        // - we will set the masses to zero so that, when computing the self interactions
        //   in a target node, the extra accelerations/potentials due to the padding data will be zero,
        // - for the positions, we must ensure 2 things:
        //   - they must not overlap with any other real particle, in order to avoid singularities
        //     when computing self-interactions in the target node, hence they must be outside the box,
        //   - the distance of the padding particles from any point of the box must be large enough so
        //     that they never fail the BH criterion check, which fails when node_dim >= theta * dist.
        //     The maximum node_dim is the box size b_size, and thus we must ensure that
        //     dist > b_size / theta for the padding particles.
        //
        // The strategy is that we put the padding particles at coordinates (M, M, ...), with M to
        // be determined. The upper right corner of the box, with coordinates (b_size/2, b_size/2, ...)
        // will be the closest point of the box to the padding particles, and the corner-particles distance
        // will be sqrt(NDim * (M - b_size/2)**2), which simplifies to sqrt(NDim) / 2 * (2*M - b_size).
        // Now we require this distance to be large enough to always satisfy the BH criterion (as written
        // above), that is, sqrt(NDim) / 2 * (2*M - b_size) > b_size / theta, which yields the requirement
        // M > b_size / (theta * sqrt(NDim)) + b_size / 2.
        const auto M = m_box_size / (std::sqrt(theta2) * std::sqrt(F(NDim))) + m_box_size / F(2);
        // NOTE: M is mathematically always >= m_box_size / F(2), which puts it on the top right
        // corner of the box for theta2 == inf. To make it completely safe with respect to the requirement
        // of avoiding singularities in the self interaction routine, we double it.
        const auto pad_coord = M * F(2);
        if (!std::isfinite(pad_coord)) {
            throw std::invalid_argument("The calculation of the SIMD padding coordinate produced the non-finite value "
                                        + std::to_string(pad_coord));
        }
        tbb::parallel_for(
            tbb::blocked_range<decltype(m_crit_nodes.size())>(0u, m_crit_nodes.size()),
            [this, theta2, G, eps2, pad_coord, &out](const auto &range) {
                // Get references to the local temporary data.
                auto &tmp_res = acc_pot_tmp_res<Q>();
                auto &tmp_tgt = tgt_tmp_data();
                for (auto i = range.begin(); i != range.end(); ++i) {
                    const auto tgt_code = get<0>(m_crit_nodes[i]);
                    const auto tgt_begin = get<1>(m_crit_nodes[i]);
                    const auto tgt_size = static_cast<size_type>(get<2>(m_crit_nodes[i]) - tgt_begin);
                    // Size of the temporary vectors that will be used to store the target node
                    // data and the total accelerations/potentials on its particles. If simd is not enabled, this
                    // value will be tgt_size. Otherwise, we add extra padding at the end based on the
                    // simd vector size.
                    const auto pdata_size = [tgt_size]() {
                        if constexpr (simd_enabled) {
                            using batch_type = xsimd::simd_type<F>;
                            constexpr auto batch_size = batch_type::size;
                            static_assert(batch_size);
                            // NOTE: we will need batch_size - 1 extra elements at the end of the
                            // temp vectors, to ensure we can load/store a simd vector starting from
                            // the last element.
                            if ((batch_size - 1u) > (std::numeric_limits<size_type>::max() - tgt_size)) {
                                throw std::overflow_error("The number of particles in a critical node ("
                                                          + std::to_string(tgt_size)
                                                          + ") is too large, and it results in an overflow condition");
                            }
                            return static_cast<size_type>(tgt_size + (batch_size - 1u));
                        } else {
                            return tgt_size;
                        }
                    }();
                    // Prepare the temporary vectors containing the result.
                    for (std::size_t j = 0; j < nvecs_res<Q>; ++j) {
                        // Resize and fill with zeroes (everything, including the padding data).
                        tmp_res[j].resize(pdata_size);
                        std::fill(tmp_res[j].data(), tmp_res[j].data() + pdata_size, F(0));
                    }
                    // Prepare the temporary vectors containing the target node's data.
                    for (std::size_t j = 0; j < NDim; ++j) {
                        tmp_tgt[j].resize(pdata_size);
                        // From 0 to tgt_size we copy the actual node coordinates.
                        std::copy(m_parts[j].data() + tgt_begin, m_parts[j].data() + tgt_begin + tgt_size,
                                  tmp_tgt[j].data());
                        // From tgt_size to pdata_size (the padding values) we insert the padding coordinate.
                        std::fill(tmp_tgt[j].data() + tgt_size, tmp_tgt[j].data() + pdata_size, pad_coord);
                    }
                    // Copy the particle masses and set the padding masses to zero.
                    tmp_tgt[NDim].resize(pdata_size);
                    std::copy(m_parts[NDim].data() + tgt_begin, m_parts[NDim].data() + tgt_begin + tgt_size,
                              tmp_tgt[NDim].data());
                    std::fill(tmp_tgt[NDim].data() + tgt_size, tmp_tgt[NDim].data() + pdata_size, F(0));
                    // Prepare arrays of pointers to the temporary data.
                    std::array<F *, nvecs_res<Q>> res_ptrs;
                    for (std::size_t j = 0; j < nvecs_res<Q>; ++j) {
                        res_ptrs[j] = tmp_res[j].data();
                    }
                    std::array<const F *, NDim + 1u> p_ptrs;
                    for (std::size_t j = 0; j < NDim + 1u; ++j) {
                        p_ptrs[j] = tmp_tgt[j].data();
                    }
                    // Do the computation.
                    tree_acc_pot<Q>(theta2, eps2, tgt_size, tgt_code, p_ptrs, res_ptrs);
                    // Multiply by G, if needed.
                    if (G != F(1)) {
                        for (std::size_t j = 0; j < nvecs_res<Q>; ++j) {
                            auto r_ptr = res_ptrs[j];
                            if constexpr (simd_enabled) {
                                using batch_type = xsimd::simd_type<F>;
                                constexpr auto batch_size = batch_type::size;
                                const batch_type Gvec(G);
                                for (size_type k = 0; k < tgt_size; k += batch_size) {
                                    (xsimd::load_aligned(r_ptr + k) * Gvec).store_aligned(r_ptr + k);
                                }
                            } else {
                                for (size_type k = 0; k < tgt_size; ++k) {
                                    *(r_ptr + k) *= G;
                                }
                            }
                        }
                    }
                    // Write out the result.
                    for (std::size_t j = 0; j < nvecs_res<Q>; ++j) {
                        std::copy(
                            res_ptrs[j], res_ptrs[j] + tgt_size,
                            out[j]
                                + boost::numeric_cast<typename std::iterator_traits<It>::difference_type>(tgt_begin));
                    }
                }
#if defined(RAKAU_WITH_SIMD_COUNTERS)
                // For the current thread, add the thread local counters
                // to the global atomic ones, then reset them.
                simd_fma_counter += simd_fma_counter_tl;
                simd_fma_counter_tl = 0;

                simd_sqrt_counter += simd_sqrt_counter_tl;
                simd_sqrt_counter_tl = 0;

                simd_rsqrt_counter += simd_rsqrt_counter_tl;
                simd_rsqrt_counter_tl = 0;
#endif
            });
    }
    // Small helper to check the value of the softening length and its square.
    // Used more than once, hence factored out.
    static void check_eps_eps2(F eps, F eps2)
    {
        if (!std::isfinite(eps2) || eps2 < F(0)) {
            throw std::domain_error("The square of the softening length must be finite and non-negative, but it is "
                                    + std::to_string(eps2) + " instead");
        }
        if (eps < F(0)) {
            throw std::domain_error("The softening length must be non-negative, but it is " + std::to_string(eps)
                                    + " instead");
        }
    }
    // Small helper to check the value of the gravitational constant.
    static void check_G_const(F G)
    {
        if (!std::isfinite(G)) {
            throw std::domain_error("The value of the gravitational constant G must be finite, but it is "
                                    + std::to_string(G) + " instead");
        }
    }
    // Top level dispatcher for the accs/pots functions. It will run a few checks and then invoke acc_pot_impl().
    // out is the array of output iterators, theta the opening angle, G the grav const, eps the softening length.
    // Q indicates which quantities will be computed (accs, potentials, or both).
    template <bool Ordered, unsigned Q, typename It>
    void acc_pot_dispatch(const std::array<It, nvecs_res<Q>> &out, F theta, F G, F eps) const
    {
        simple_timer st("vector accs/pots computation");
        const auto theta2 = theta * theta, eps2 = eps * eps;
        // Input param check.
        if (!std::isfinite(theta2) || theta2 <= F(0)) {
            throw std::domain_error("The square of the theta parameter must be finite and positive, but it is "
                                    + std::to_string(theta2) + " instead");
        }
        if (theta < F(0)) {
            throw std::domain_error("The theta parameter must be non-negative, but it is " + std::to_string(theta)
                                    + " instead");
        }
        check_eps_eps2(eps, eps2);
        check_G_const(G);
        if constexpr (Ordered) {
            // Make sure we don't run into overflows when doing a permutated iteration
            // over the iterators in out.
            using diff_t = typename std::iterator_traits<It>::difference_type;
            if (m_parts[0].size() > static_cast<std::make_unsigned_t<diff_t>>(std::numeric_limits<diff_t>::max())) {
                throw std::overflow_error("The number of particles (" + std::to_string(m_parts[0].size())
                                          + ") is too large, and it results in an overflow condition when computing "
                                            "the accelerations/potentials");
            }
            using it_t = decltype(boost::make_permutation_iterator(out[0], m_perm.begin()));
            std::array<it_t, nvecs_res<Q>> out_pits;
            for (std::size_t j = 0; j < nvecs_res<Q>; ++j) {
                out_pits[j] = boost::make_permutation_iterator(out[j], m_perm.begin());
            }
            // NOTE: we are checking in the acc_pot_impl() function that we can index into
            // the permuted iterators without overflows (see the use of boost::numeric_cast()).
            acc_pot_impl<Q>(out_pits, theta2, G, eps2);
        } else {
            acc_pot_impl<Q>(out, theta2, G, eps2);
        }
    }
    // Helper overload for an array of vectors. It will prepare the vectors and then
    // call the other overload.
    template <bool Ordered, unsigned Q, typename Allocator>
    void acc_pot_dispatch(std::array<std::vector<F, Allocator>, nvecs_res<Q>> &out, F theta, F G, F eps) const
    {
        std::array<F *, nvecs_res<Q>> out_ptrs;
        for (std::size_t j = 0; j < nvecs_res<Q>; ++j) {
            out[j].resize(boost::numeric_cast<decltype(out[j].size())>(m_parts[0].size()));
            out_ptrs[j] = out[j].data();
        }
        acc_pot_dispatch<Ordered, Q>(out_ptrs, theta, G, eps);
    }
    // Helper overload for a single vector. It will prepare the vector and then
    // call the other overload. This is used for the potential-only computations.
    template <bool Ordered, unsigned Q, typename Allocator>
    void acc_pot_dispatch(std::vector<F, Allocator> &out, F theta, F G, F eps) const
    {
        static_assert(Q == 1u);
        out.resize(boost::numeric_cast<decltype(out.size())>(m_parts[0].size()));
        acc_pot_dispatch<Ordered, Q>(std::array{out.data()}, theta, G, eps);
    }
    // Small helper to turn an init list into an array, in the functions for the computation
    // of the accelerations/potentials. Q indicates which quantities will be computed (accs,
    // potentials, or both).
    template <unsigned Q, typename It>
    static auto acc_pot_ilist_to_array(std::initializer_list<It> ilist)
    {
        if (ilist.size() != nvecs_res<Q>) {
            throw std::invalid_argument(
                "An initializer list containing " + std::to_string(ilist.size())
                + " iterators was used as the output for the computation of the accelerations/potentials in a "
                + std::to_string(NDim) + "-dimensional tree, but a list with " + std::to_string(nvecs_res<Q>)
                + " iterators is required instead");
        }
        std::array<It, nvecs_res<Q>> retval;
        std::copy(ilist.begin(), ilist.end(), retval.begin());
        return retval;
    }
    // Helper to parse the keyword arguments for the acc/pot functions.
    template <typename... Args>
    static auto parse_accpot_kwargs(Args &&... args)
    {
        ::igor::parser p{args...};

        F G(1), eps(0);
        if constexpr (p.is_provided(kwargs::G)) {
            G = boost::numeric_cast<F>(p(kwargs::G));
        }
        if constexpr (p.is_provided(kwargs::eps)) {
            eps = boost::numeric_cast<F>(p(kwargs::eps));
        }

        return std::make_tuple(G, eps);
    }

public:
    template <typename Allocator, typename... KwArgs>
    void accs_u(std::array<std::vector<F, Allocator>, NDim> &out, F theta, KwArgs &&... args) const
    {
        const auto [G, eps] = parse_accpot_kwargs(std::forward<KwArgs>(args)...);
        acc_pot_dispatch<false, 0>(out, theta, G, eps);
    }
    template <typename It, typename... KwArgs>
    void accs_u(const std::array<It, NDim> &out, F theta, KwArgs &&... args) const
    {
        const auto [G, eps] = parse_accpot_kwargs(std::forward<KwArgs>(args)...);
        acc_pot_dispatch<false, 0>(out, theta, G, eps);
    }
    template <typename It, typename... KwArgs>
    void accs_u(std::initializer_list<It> out, F theta, KwArgs &&... args) const
    {
        accs_u(acc_pot_ilist_to_array<0>(out), theta, std::forward<KwArgs>(args)...);
    }
    template <typename Allocator, typename... KwArgs>
    void pots_u(std::vector<F, Allocator> &out, F theta, KwArgs &&... args) const
    {
        const auto [G, eps] = parse_accpot_kwargs(std::forward<KwArgs>(args)...);
        acc_pot_dispatch<false, 1>(out, theta, G, eps);
    }
    template <typename It, typename... KwArgs>
    void pots_u(It out, F theta, KwArgs &&... args) const
    {
        const auto [G, eps] = parse_accpot_kwargs(std::forward<KwArgs>(args)...);
        acc_pot_dispatch<false, 1>(std::array{out}, theta, G, eps);
    }
    template <typename Allocator, typename... KwArgs>
    void accs_pots_u(std::array<std::vector<F, Allocator>, NDim + 1u> &out, F theta, KwArgs &&... args) const
    {
        const auto [G, eps] = parse_accpot_kwargs(std::forward<KwArgs>(args)...);
        acc_pot_dispatch<false, 2>(out, theta, G, eps);
    }
    template <typename It, typename... KwArgs>
    void accs_pots_u(const std::array<It, NDim + 1u> &out, F theta, KwArgs &&... args) const
    {
        const auto [G, eps] = parse_accpot_kwargs(std::forward<KwArgs>(args)...);
        acc_pot_dispatch<false, 2>(out, theta, G, eps);
    }
    template <typename It, typename... KwArgs>
    void accs_pots_u(std::initializer_list<It> out, F theta, KwArgs &&... args) const
    {
        accs_pots_u(acc_pot_ilist_to_array<2>(out), theta, std::forward<KwArgs>(args)...);
    }
    template <typename Allocator, typename... KwArgs>
    void accs_o(std::array<std::vector<F, Allocator>, NDim> &out, F theta, KwArgs &&... args) const
    {
        const auto [G, eps] = parse_accpot_kwargs(std::forward<KwArgs>(args)...);
        acc_pot_dispatch<true, 0>(out, theta, G, eps);
    }
    template <typename It, typename... KwArgs>
    void accs_o(const std::array<It, NDim> &out, F theta, KwArgs &&... args) const
    {
        const auto [G, eps] = parse_accpot_kwargs(std::forward<KwArgs>(args)...);
        acc_pot_dispatch<true, 0>(out, theta, G, eps);
    }
    template <typename It, typename... KwArgs>
    void accs_o(std::initializer_list<It> out, F theta, KwArgs &&... args) const
    {
        accs_o(acc_pot_ilist_to_array<0>(out), theta, std::forward<KwArgs>(args)...);
    }
    template <typename Allocator, typename... KwArgs>
    void pots_o(std::vector<F, Allocator> &out, F theta, KwArgs &&... args) const
    {
        const auto [G, eps] = parse_accpot_kwargs(std::forward<KwArgs>(args)...);
        acc_pot_dispatch<true, 1>(out, theta, G, eps);
    }
    template <typename It, typename... KwArgs>
    void pots_o(It out, F theta, KwArgs &&... args) const
    {
        const auto [G, eps] = parse_accpot_kwargs(std::forward<KwArgs>(args)...);
        acc_pot_dispatch<true, 1>(std::array{out}, theta, G, eps);
    }
    template <typename Allocator, typename... KwArgs>
    void accs_pots_o(std::array<std::vector<F, Allocator>, NDim + 1u> &out, F theta, KwArgs &&... args) const
    {
        const auto [G, eps] = parse_accpot_kwargs(std::forward<KwArgs>(args)...);
        acc_pot_dispatch<true, 2>(out, theta, G, eps);
    }
    template <typename It, typename... KwArgs>
    void accs_pots_o(const std::array<It, NDim + 1u> &out, F theta, KwArgs &&... args) const
    {
        const auto [G, eps] = parse_accpot_kwargs(std::forward<KwArgs>(args)...);
        acc_pot_dispatch<true, 2>(out, theta, G, eps);
    }
    template <typename It, typename... KwArgs>
    void accs_pots_o(std::initializer_list<It> out, F theta, KwArgs &&... args) const
    {
        accs_pots_o(acc_pot_ilist_to_array<2>(out), theta, std::forward<KwArgs>(args)...);
    }

private:
    template <bool Ordered, unsigned Q>
    auto exact_acc_pot_impl(size_type orig_idx, F G, F eps) const
    {
        simple_timer st("exact acc/pot computation");
        const auto eps2 = eps * eps;
        // Check eps.
        check_eps_eps2(eps, eps2);
        // Check G.
        check_G_const(G);
        const auto size = m_parts[0].size();
        std::array<F, nvecs_res<Q>> retval{};
        std::array<F, NDim> diffs;
        const auto idx = Ordered ? m_inv_perm[orig_idx] : orig_idx;
        for (size_type i = 0; i < size; ++i) {
            if (i == idx) {
                continue;
            }
            F dist2(eps2);
            for (std::size_t j = 0; j < NDim; ++j) {
                diffs[j] = m_parts[j][i] - m_parts[j][idx];
                dist2 = fma_wrap(diffs[j], diffs[j], dist2);
            }
            const auto inv_dist = F(1) / std::sqrt(dist2), Gmi_dist = G * m_parts[NDim][i] * inv_dist;
            if constexpr (Q == 0u || Q == 2u) {
                // Q == 0 or 2: accelerations are requested.
                const auto Gmi_dist3 = inv_dist * inv_dist * Gmi_dist;
                for (std::size_t j = 0; j < NDim; ++j) {
                    retval[j] = fma_wrap(diffs[j], Gmi_dist3, retval[j]);
                }
            }
            if constexpr (Q == 1u || Q == 2u) {
                // Q == 1 or 2: potentials are requested.
                // Establish the index of the potential in the result array:
                // 0 if only the potentials are requested, NDim otherwise.
                constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim);
                retval[pot_idx] = fma_wrap(-Gmi_dist, m_parts[NDim][idx], retval[pot_idx]);
            }
        }
        return retval;
    }

public:
    std::array<F, NDim> exact_acc_u(size_type idx, F G = F(1), F eps = F(0)) const
    {
        return exact_acc_pot_impl<false, 0>(idx, G, eps);
    }
    F exact_pot_u(size_type idx, F G = F(1), F eps = F(0)) const
    {
        return exact_acc_pot_impl<false, 1>(idx, G, eps)[0];
    }
    std::array<F, NDim + 1u> exact_acc_pot_u(size_type idx, F G = F(1), F eps = F(0)) const
    {
        return exact_acc_pot_impl<false, 2>(idx, G, eps);
    }
    std::array<F, NDim> exact_acc_o(size_type idx, F G = F(1), F eps = F(0)) const
    {
        return exact_acc_pot_impl<true, 0>(idx, G, eps);
    }
    F exact_pot_o(size_type idx, F G = F(1), F eps = F(0)) const
    {
        return exact_acc_pot_impl<true, 1>(idx, G, eps)[0];
    }
    std::array<F, NDim + 1u> exact_acc_pot_o(size_type idx, F G = F(1), F eps = F(0)) const
    {
        return exact_acc_pot_impl<true, 2>(idx, G, eps);
    }

private:
    // Implementations of the functions to get (un)ordered iterators into the particles.
    // They are static functions because we need both const and non-const variants of this.
    template <typename Tr>
    static auto ord_p_its_impl(Tr &tr)
    {
        using it_t = decltype(boost::make_permutation_iterator(tr.m_parts[0].data(), tr.m_inv_perm.begin()));
        using diff_t = typename std::iterator_traits<it_t>::difference_type;
        using udiff_t = std::make_unsigned_t<diff_t>;
        // Ensure that the iterators we return can index up to the particle number.
        if (tr.m_parts[0].size() > static_cast<udiff_t>(std::numeric_limits<diff_t>::max())) {
            throw std::overflow_error("The number of particles (" + std::to_string(tr.m_parts[0].size())
                                      + ") is too large, and it results in an overflow condition when constructing "
                                        "ordered iterators to the particle data");
        }
        std::array<it_t, NDim + 1u> retval;
        for (std::size_t j = 0; j < NDim + 1u; ++j) {
            retval[j] = boost::make_permutation_iterator(tr.m_parts[j].data(), tr.m_inv_perm.begin());
        }
        return retval;
    }
    template <typename Tr>
    static auto unord_p_its_impl(Tr &tr)
    {
        using ptr_t = decltype(tr.m_parts[0].data());
        std::array<ptr_t, NDim + 1u> retval;
        for (std::size_t j = 0; j < NDim + 1u; ++j) {
            retval[j] = tr.m_parts[j].data();
        }
        return retval;
    }

public:
    auto p_its_u() const
    {
        return unord_p_its_impl(*this);
    }
    auto p_its_o() const
    {
        return ord_p_its_impl(*this);
    }
    const auto &perm() const
    {
        return m_perm;
    }
    const auto &last_perm() const
    {
        return m_last_perm;
    }
    const auto &inv_perm() const
    {
        return m_inv_perm;
    }

private:
    // After updating the particles' positions, this method must be called
    // to reconstruct the other data members according to the new positions.
    void sync()
    {
        // Get the number of particles.
        const auto nparts = m_parts[0].size();
        // Re-deduce the box size, if needed.
        if (m_box_size_deduced) {
            m_box_size = determine_box_size(p_its_u(), nparts);
        }

        // Establish the new codes.
        tbb::parallel_for(tbb::blocked_range<size_type>(0u, nparts), [this](const auto &range) {
            std::array<UInt, NDim> tmp_dcoord;
            morton_encoder<NDim, UInt> me;
            for (auto i = range.begin(); i != range.end(); ++i) {
                disc_coords(tmp_dcoord, i);
                m_codes[i] = me(tmp_dcoord.data());
            }
        });

        // Reset m_last_perm to a iota.
        tbb::parallel_for(tbb::blocked_range<size_type>(0u, nparts, boost::numeric_cast<size_type>(data_chunking)),
                          [this](const auto &range) {
                              std::iota(m_last_perm.data() + range.begin(), m_last_perm.data() + range.end(),
                                        range.begin());
                          },
                          tbb::simple_partitioner());
        // Do the indirect sorting onto m_last_perm.
        indirect_code_sort(m_last_perm.begin(), m_last_perm.end());
        {
            // Apply the indirect sorting.
            tbb::task_group tg;
            // NOTE: upon tree construction, we already checked that the number of particles does not
            // overflow the limit imposed by apply_isort().
            tg.run([this]() {
                apply_isort(m_codes, m_last_perm);
                // Make sure the sort worked as intended.
                assert(std::is_sorted(m_codes.begin(), m_codes.end()));
            });
            for (std::size_t j = 0; j < NDim + 1u; ++j) {
                tg.run([this, j]() { apply_isort(m_parts[j], m_last_perm); });
            }
            tg.run([this]() {
                // Apply the new indirect sorting to the original one.
                apply_isort(m_perm, m_last_perm);
                // Establish the indices for ordered iteration (in the original order).
                // NOTE: this goes in the same task as we need m_perm to be sorted
                // before calling this function.
                perm_to_inv_perm();
            });
            tg.wait();
        }
        // Re-construct the tree. Make sure we empty the tree structures
        // before doing it.
        m_tree.clear();
        m_crit_nodes.clear();
        build_tree();
    }
    // Invoke the particle update function with an
    // exception safe wrapper.
    template <bool Ordered, typename Func>
    void update_particles_dispatch(Func &&f)
    {
        simple_timer st("overall update_particles");
        try {
            if constexpr (Ordered) {
                // Apply the functor to the ordered iterators.
                std::forward<Func>(f)(ord_p_its_impl(*this));
            } else {
                // Apply the functor to the unordered iterators.
                std::forward<Func>(f)(unord_p_its_impl(*this));
            }
            // Sync the tree structures.
            sync();
        } catch (...) {
            // Erase everything before re-throwing.
            clear();
            throw;
        }
    }

public:
    template <typename Func>
    void update_particles_u(Func &&f)
    {
        update_particles_dispatch<false>(std::forward<Func>(f));
    }
    template <typename Func>
    void update_particles_o(Func &&f)
    {
        update_particles_dispatch<true>(std::forward<Func>(f));
    }
    F get_box_size() const
    {
        return m_box_size;
    }
    bool get_box_size_deduced() const
    {
        return m_box_size_deduced;
    }
    size_type get_ncrit() const
    {
        return m_ncrit;
    }
    size_type get_max_leaf_n() const
    {
        return m_max_leaf_n;
    }

private:
    // The size of the domain.
    F m_box_size;
    // Flag to signal if the domain size was deduced or explicitly specified.
    bool m_box_size_deduced;
    // The maximum number of particles in a leaf node.
    size_type m_max_leaf_n;
    // Number of particles in a critical node: if the number of particles in
    // a node is ncrit or less, then we will compute the accelerations/potentials on the
    // particles in that node in a vectorised fashion.
    size_type m_ncrit;
    // The particles: NDim coordinates plus masses.
    std::array<fp_vector, NDim + 1u> m_parts;
    // The particles' Morton codes.
    std::vector<UInt, di_aligned_allocator<UInt>> m_codes;
    // The indirect sorting vector. It establishes how to re-order the
    // original particle sequence so that the particles' Morton codes are
    // sorted in ascending order. E.g., if m_perm is [0, 3, 1, 2, ...],
    // then the first particle in Morton order is also the first particle in
    // the original order, the second particle in the Morton order is the
    // particle at index 3 in the original order, and so on.
    std::vector<size_type, di_aligned_allocator<size_type>> m_perm;
    // m_perm stores the permutation necessary to re-order the particles
    // from the *original* order (i.e., the order used during the construction
    // of the tree) to the current internal Morton order. m_last_perm is similar,
    // but it contains the permutation needed to bring the particle order *before*
    // the last call to update_particles() to the current internal Morton order.
    // In other words, m_last_perm tells us how update_particles() re-ordered the internal
    // order.
    std::vector<size_type, di_aligned_allocator<size_type>> m_last_perm;
    // Indices vector to iterate over the particles' data in the original order.
    // It establishes how to re-order the current internal Morton order to recover the original
    // particle order. This is the inverse of m_perm, and it's always possible to
    // compute one given the other. E.g., if m_perm is [0, 3, 1, 2, ...] then
    // m_inv_perm will be [0, 2, 3, 1, ...], meaning that the first particle in
    // the original order is also the first particle in the Morton order, the second
    // particle in the original order is the particle at index 2 in the Morton order,
    // and so on.
    std::vector<size_type, di_aligned_allocator<size_type>> m_inv_perm;
    // The tree structure.
    tree_type m_tree;
    // The list of critical nodes.
    cnode_list_type m_crit_nodes;
};

template <typename F>
using octree = tree<3, F>;

} // namespace rakau

#endif
