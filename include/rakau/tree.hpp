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
#include <deque>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// We use the BitScanReverse(64) intrinsic in the implementation of clz(), but
// only if we are *not* on clang-cl: there, we can use GCC-style intrinsics.
// https://msdn.microsoft.com/en-us/library/fbxyd7zd.aspx
#if !defined(__clang__) && defined(_MSC_VER)

#include <intrin.h>

#if _WIN64

#pragma intrinsic(_BitScanReverse64)

#endif

#pragma intrinsic(_BitScanReverse)

#endif

#include <boost/iterator/permutation_iterator.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
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

#include <rakau/detail/simd.hpp>

namespace rakau
{

inline namespace detail
{

// Helper to ignore unused args in functions.
template <typename... Args>
inline void ignore_args(const Args &...)
{
}

// Dependent false for static_assert in if constexpr.
// http://en.cppreference.com/w/cpp/language/if#Constexpr_If
template <typename T>
struct dependent_false : std::false_type {
};

template <typename T>
inline constexpr bool dependent_false_v = dependent_false<T>::value;

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

// Silence spurious GCC warning in tuple_for_each().
#if !defined(__clang__) && defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnoexcept"

#endif

// Tuple for_each(). It will apply the input functor f to each element of
// the input tuple tup, sequentially.
template <typename Tuple, typename F>
inline void tuple_for_each(Tuple &&tup, F &&f)
{
    std::apply(
        [&f](auto &&... items) {
            // NOTE: here we converting to void the results of the invocations
            // of f. This ensures that we are folding using the builtin comma
            // operator, which implies sequencing:
            // """
            //  Every value computation and side effect of the first (left) argument of the built-in comma operator is
            //  sequenced before every value computation and side effect of the second (right) argument.
            // """"
            // NOTE: we are writing this as a right fold, i.e., it will expand as:
            //
            // f(tup[0]), (f(tup[1]), (f(tup[2])...
            //
            // A left fold would also work guaranteeing the same sequencing.
            (void(std::forward<F>(f)(std::forward<decltype(items)>(items))), ...);
        },
        std::forward<Tuple>(tup));
}

#if !defined(__clang__) && defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

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
            !(::m3D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y), std::uint32_t(z)) >> 63u));
        assert((::morton3D_64_encode(x, y, z)
                == ::m3D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y), std::uint32_t(z))));
        return ::m3D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y), std::uint32_t(z));
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
            !(::m3D_e_sLUT<std::uint32_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y), std::uint32_t(z)) >> 31u));
        assert((::morton3D_32_encode(x, y, z)
                == ::m3D_e_sLUT<std::uint32_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y), std::uint32_t(z))));
        return ::m3D_e_sLUT<std::uint32_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y), std::uint32_t(z));
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
        assert(!(::m2D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y)) >> 63u));
        assert((::morton2D_64_encode(x, y)
                == ::m2D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y))));
        return ::m2D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y));
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
        assert(!(::m2D_e_sLUT<std::uint32_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y)) >> 31u));
        assert((::morton2D_32_encode(x, y)
                == ::m2D_e_sLUT<std::uint32_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y))));
        return ::m2D_e_sLUT<std::uint32_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y));
    }
};

template <typename UInt, std::size_t NDim>
inline constexpr unsigned cbits_v = []() {
    constexpr unsigned nbits = std::numeric_limits<UInt>::digits;
    static_assert(nbits > NDim, "The number of bits must be greater than the number of dimensions.");
    return static_cast<unsigned>(nbits / NDim - !(nbits % NDim));
}();

// clz wrapper. n must be a nonzero unsigned integral.
template <typename UInt>
inline unsigned clz(UInt n)
{
    static_assert(std::is_integral_v<UInt> && std::is_unsigned_v<UInt>);
    assert(n);
#if defined(__clang__) || defined(__GNUC__)
    // Implementation using GCC's and clang's builtins.
    if constexpr (std::is_same_v<UInt, unsigned>) {
        return static_cast<unsigned>(__builtin_clz(n));
    } else if constexpr (std::is_same_v<UInt, unsigned long>) {
        return static_cast<unsigned>(__builtin_clzl(n));
    } else if constexpr (std::is_same_v<UInt, unsigned long long>) {
        return static_cast<unsigned>(__builtin_clzll(n));
    } else {
        // In this case we are dealing with an unsigned integral type which
        // is not wider than unsigned int. Let's compute the result with n cast
        // to unsigned int first.
        const auto ret_u = static_cast<unsigned>(__builtin_clz(static_cast<unsigned>(n)));
        // We must now subtract the number of extra bits that unsigned
        // has over UInt.
        constexpr auto extra_nbits
            = static_cast<unsigned>(std::numeric_limits<unsigned>::digits - std::numeric_limits<UInt>::digits);
        return ret_u - extra_nbits;
    }
#elif defined(_MSC_VER)
    // Implementation using MSVC's intrinsics.
    unsigned long index;
    if constexpr (std::is_same_v<UInt, unsigned long long>) {
#if _WIN64
        // On 64-bit builds, we have a specific intrinsic for 64-bit ints.
        _BitScanReverse64(&index, n);
        return 63u - static_cast<unsigned>(index);
#else
        // On 32-bit builds, we split the computation in two parts.
        if (n >> 32) {
            // The high half of n contains something. The total bsr
            // will be the bsr of the high half augmented by 32 bits.
            _BitScanReverse(&index, static_cast<unsigned long>(n >> 32));
            return static_cast<unsigned>(31u - index);
        } else {
            // The high half of n does not contain anything. Only
            // the low half contributes to the bsr.
            _BitScanReverse(&index, static_cast<unsigned long>(n));
            return static_cast<unsigned>(63u - index);
        }
#endif
    } else {
        _BitScanReverse(&index, static_cast<unsigned long>(n));
        return static_cast<unsigned>(std::numeric_limits<UInt>::digits) - 1u - static_cast<unsigned>(index);
    }
#else
    static_assert(dependent_false_v<UInt>, "No clz() implementation is available on this platform.");
#endif
}

// Small helper to get the tree level of a nodal code.
template <std::size_t NDim, typename UInt>
inline unsigned tree_level(UInt n)
{
#if !defined(NDEBUG)
    constexpr unsigned cbits = cbits_v<UInt, NDim>;
#endif
    constexpr unsigned ndigits = std::numeric_limits<UInt>::digits;
    assert(n);
    assert(!((ndigits - 1u - clz(n)) % NDim));
    auto retval = static_cast<unsigned>((ndigits - 1u - clz(n)) / NDim);
    assert(cbits >= retval);
    assert((cbits - retval) * NDim < ndigits);
    return retval;
}

// Small function to compare nodal codes.
template <std::size_t NDim, typename UInt>
inline bool node_compare(UInt n1, UInt n2)
{
    constexpr unsigned cbits = cbits_v<UInt, NDim>;
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
                              values_new[i] = values[perm[i]];
                          }
                      });
    values = std::move(values_new);
}

// Little helper to verify that we can index into ElementIt
// up to at least the value max_index. This is used below to verify
// that a permuted iterator does not incur in overflows.
template <typename ElementIt, typename Index>
inline bool check_perm_it_range(Index max_index)
{
    using diff_t = typename std::iterator_traits<ElementIt>::difference_type;
    using udiff_t = std::make_unsigned_t<diff_t>;
    static_assert(std::is_integral_v<Index> && std::is_unsigned_v<Index>);
    return max_index <= static_cast<udiff_t>(std::numeric_limits<diff_t>::max());
}

// Small helpers for the checked in-place addition of (atomic) unsigned integrals.
template <typename T>
inline void checked_uinc(T &out, T add)
{
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
    if (out > std::numeric_limits<T>::max() - add) {
        throw std::overflow_error("overflow in the addition of two unsigned integral values");
    }
    out += add;
}

template <typename T>
inline void checked_uinc(std::atomic<T> &out, T add)
{
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
    const auto prev = out.fetch_add(add);
    if (prev > std::numeric_limits<T>::max() - add) {
        throw std::overflow_error("overflow in the addition of two unsigned integral values");
    }
}

// A trivial allocator that supports custom alignment and does
// default-initialisation instead of value-initialisation.
template <typename T, std::size_t Alignment = 0>
struct di_aligned_allocator {
    // Alignment must be either zero or:
    // - not less than the natural alignment of T,
    // - a power of 2.
    static_assert(!Alignment || (Alignment >= alignof(T) && (Alignment & (Alignment - 1u)) == 0u));
    // value_type must always be present.
    using value_type = T;
    // Make sure the size_type is consistent with the size type returned
    // by malloc() and friends.
    using size_type = std::size_t;
    // NOTE: rebind is needed because we have a non-type template parameter, which
    // prevents the automatic implementation of rebind from working.
    // http://en.cppreference.com/w/cpp/concept/Allocator#cite_note-1
    template <typename U>
    struct rebind {
        using other = di_aligned_allocator<U, Alignment>;
    };
    // Allocation.
    T *allocate(size_type n) const
    {
        // Total size in bytes. This is prevented from being too large
        // by the default implementation of max_size().
        const auto size = n * sizeof(T);
        void *retval;
        if constexpr (Alignment == 0u) {
            retval = std::malloc(size);
        } else {
            // For use in std::aligned_alloc, the size must be a multiple of the alignment.
            // http://en.cppreference.com/w/cpp/memory/c/aligned_alloc
            // A null pointer will be returned if invalid Alignment and/or size are supplied,
            // or if the allocation fails.
            // NOTE: some early versions of GCC put aligned_alloc in the root namespace rather
            // than std, so let's try to workaround.
            using namespace std;
            retval = aligned_alloc(Alignment, size);
        }
        if (!retval) {
            // Throw on failure.
            throw std::bad_alloc{};
        }
        return static_cast<T *>(retval);
    }
    // Deallocation.
    void deallocate(T *ptr, size_type) const
    {
        std::free(ptr);
    }
    // Trivial comparison operators.
    friend bool operator==(const di_aligned_allocator &, const di_aligned_allocator &)
    {
        return true;
    }
    friend bool operator!=(const di_aligned_allocator &, const di_aligned_allocator &)
    {
        return false;
    }
    // The construction function.
    template <typename U, typename... Args>
    void construct(U *p, Args &&... args) const
    {
        if constexpr (sizeof...(args) == 0u) {
            // When no construction arguments are supplied, do default
            // initialisation rather than value initialisation.
            ::new (static_cast<void *>(p)) U;
        } else {
            // This is the standard std::allocator implementation.
            // http://en.cppreference.com/w/cpp/memory/allocator/construct
            ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
        }
    }
};

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

} // namespace detail

template <typename UInt, typename F, std::size_t NDim>
class tree
{
    static_assert(NDim);
    static_assert(std::is_floating_point_v<F>);
    static_assert(std::is_integral_v<UInt> && std::is_unsigned_v<UInt>);
    // cbits shortcut.
    static constexpr unsigned cbits = cbits_v<UInt, NDim>;
    // simd_enabled shortcut.
    static constexpr bool simd_enabled = simd_enabled_v<F>;
    // Main vector type for storing floating-point values. It uses custom alignment to enable
    // aligned loads/stores whenever possible.
    using fp_vector = std::vector<F, di_aligned_allocator<F, XSIMD_DEFAULT_ALIGNMENT>>;

public:
    using size_type = typename fp_vector::size_type;

private:
    // The node type.
    using node_type = std::tuple<UInt, std::array<size_type, 3>, F, std::array<F, NDim>>;
    // The tree type.
    using tree_type = std::vector<node_type, di_aligned_allocator<node_type>>;
    // The critical node descriptor type (nodal code and particle range).
    using cnode_type = std::tuple<UInt, size_type, size_type>;
    // List of critical nodes.
    using cnode_list_type = std::vector<cnode_type, di_aligned_allocator<cnode_type>>;
    // Serial construction of a subtree. The parent of the subtree is the node with code parent_code,
    // at the level ParentLevel. The particles in the children nodes have codes in the [begin, end)
    // range. The children nodes will be appended in depth-first order to tree. crit_nodes is the local
    // list of critical nodes, crit_ancestor a flag signalling if the parent node or one of its
    // ancestors is a critical node.
    template <unsigned ParentLevel, typename CIt>
    size_type build_tree_ser_impl(tree_type &tree, cnode_list_type &crit_nodes, UInt parent_code, CIt begin, CIt end,
                                  bool crit_ancestor)
    {
        if constexpr (ParentLevel < cbits) {
            assert(tree_level<NDim>(parent_code) == ParentLevel);
            assert(tree.size() && get<0>(tree.back()) == parent_code);
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
            // This is the node prefix: it is the nodal code of the parent with the most significant bit
            // switched off.
            // NOTE: overflow is prevented by the if constexpr above.
            const auto node_prefix = parent_code - (UInt(1) << (ParentLevel * NDim));
            // NOTE: overflow is prevented in the computation of cbits_v.
            for (UInt i = 0; i < (UInt(1) << NDim); ++i) {
                // Compute the first and last possible codes for the current child node.
                // They both start with (from MSB to LSB):
                // - current node prefix,
                // - i.
                // The first possible code is then right-padded with all zeroes, the last possible
                // code is right-padded with ones.
                const auto p_first = static_cast<UInt>((node_prefix << ((cbits - ParentLevel) * NDim))
                                                       + (i << ((cbits - ParentLevel - 1u) * NDim)));
                const auto p_last
                    = static_cast<UInt>(p_first + ((UInt(1) << ((cbits - ParentLevel - 1u) * NDim)) - 1u));
                // Compute the starting point of the node: it_start will point to the first value equal to
                // or greater than p_first.
                const auto it_start = std::lower_bound(begin, end, p_first);
                // Determine the end of the child node: it_end will point to the first value greater
                // than the largest possible code for the current child node.
                const auto it_end = std::upper_bound(it_start, end, p_last);
                // Compute the number of particles.
                const auto npart = std::distance(it_start, it_end);
                assert(npart >= 0);
                if (npart) {
                    // npart > 0, we have a node. Compute its nodal code by moving up the
                    // parent nodal code by NDim and adding the current child node index i.
                    const auto cur_code = static_cast<UInt>((parent_code << NDim) + i);
                    // Add the node to the tree.
                    tree.emplace_back(
                        cur_code,
                        std::array<size_type, 3>{static_cast<size_type>(std::distance(m_codes.begin(), it_start)),
                                                 static_cast<size_type>(std::distance(m_codes.begin(), it_end)),
                                                 // NOTE: the children count gets inited to zero. It
                                                 // will be filled in later.
                                                 size_type(0)},
                        // NOTE: make sure mass and COM coords are initialised in a known state (i.e.,
                        // zero for C++ floating-point).
                        0, std::array<F, NDim>{});
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
                        crit_nodes.push_back({get<0>(tree.back()), get<1>(tree.back())[0], get<1>(tree.back())[1]});
                    }
                    if (u_npart > m_max_leaf_n) {
                        // The node is an internal one, go deeper, and update the children count
                        // for the newly-added node.
                        // NOTE: do this in 2 parts, rather than assigning directly the children count into
                        // the tree, as the computation of the children count might enlarge the tree and thus
                        // invalidate references to its elements.
                        const auto children_count = build_tree_ser_impl<ParentLevel + 1u>(
                            tree, crit_nodes, cur_code, it_start, it_end,
                            // NOTE: the children nodes have critical ancestors if either
                            // the newly-added node is critical or one of its ancestors is.
                            critical_node || crit_ancestor);
                        get<1>(tree[tree_size - 1u])[2] = children_count;
                    }
                    // The total children count will be augmented by the children count of the
                    // newly-added node, +1 for the node itself.
                    checked_uinc(retval, get<1>(tree[tree_size - 1u])[2]);
                    checked_uinc(retval, size_type(1));
                }
            }
            return retval;
        } else {
            // NOTE: if we end up here, it means we walked through all the recursion levels
            // and we cannot go any deeper.
            // GCC warnings about unused params.
            ignore_args(parent_code, begin, end);
            return 0;
        }
    }
    // Parallel tree construction. It will iterate in parallel over the children of a node with nodal code
    // parent_code at level ParentLevel, add single nodes to the 'trees' concurrent container, and recurse down
    // until the tree level split_level is reached. From there, whole subtrees (rather than single nodes) will be
    // constructed and added to the 'trees' container via build_tree_ser_impl(). The particles in the children nodes
    // have codes in the [begin, end) range. crit_nodes is the global list of lists of critical nodes, crit_ancestor a
    // flag signalling if the parent node or one of its ancestors is a critical node.
    template <unsigned ParentLevel, typename Out, typename CritNodes, typename CIt>
    size_type build_tree_par_impl(Out &trees, CritNodes &crit_nodes, UInt parent_code, CIt begin, CIt end,
                                  unsigned split_level, bool crit_ancestor)
    {
        if constexpr (ParentLevel < cbits) {
            assert(tree_level<NDim>(parent_code) == ParentLevel);
            // NOTE: the return value needs to be computed atomically as we are accumulating
            // results from multiple concurrent tasks.
            std::atomic<size_type> retval(0);
            // NOTE: similar to the previous function, see comments there.
            assert(begin != end);
            const auto node_prefix = parent_code - (UInt(1) << (ParentLevel * NDim));
            tbb::task_group tg;
            for (UInt i = 0; i < (UInt(1) << NDim); ++i) {
                tg.run([node_prefix, i, begin, end, &trees, parent_code, this, &retval, split_level, crit_ancestor,
                        &crit_nodes] {
                    const auto p_first = static_cast<UInt>((node_prefix << ((cbits - ParentLevel) * NDim))
                                                           + (i << ((cbits - ParentLevel - 1u) * NDim)));
                    const auto p_last
                        = static_cast<UInt>(p_first + ((UInt(1) << ((cbits - ParentLevel - 1u) * NDim)) - 1u));
                    const auto it_start = std::lower_bound(begin, end, p_first);
                    const auto it_end = std::upper_bound(it_start, end, p_last);
                    const auto npart = std::distance(it_start, it_end);
                    assert(npart >= 0);
                    if (npart) {
                        const auto cur_code = static_cast<UInt>((parent_code << NDim) + i);
                        // Add a new tree, and fill its first node.
                        // NOTE: use push_back(tree_type{}) instead of emplace_back() because
                        // TBB hates clang apparently.
                        auto &new_tree = *trees.push_back(tree_type{});
                        new_tree.emplace_back(
                            cur_code,
                            std::array<size_type, 3>{static_cast<size_type>(std::distance(m_codes.begin(), it_start)),
                                                     static_cast<size_type>(std::distance(m_codes.begin(), it_end)),
                                                     size_type(0)},
                            0, std::array<F, NDim>{});
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
                        // Add a new entry to crit_nodes. If the only node of the new tree is critical, the new entry
                        // will contain 1 element, otherwise it will be an empty list. This empty list may remain empty,
                        // or be used to accumulate the list of critical nodes during the serial subtree construction.
                        auto &new_crit_nodes
                            = critical_node ? *crit_nodes.push_back({{get<0>(new_tree[0]), get<1>(new_tree[0])[0],
                                                                      get<1>(new_tree[0])[1]}})
                                            : *crit_nodes.push_back({});
                        if (u_npart > m_max_leaf_n) {
                            if (ParentLevel + 1u == split_level) {
                                // NOTE: the current level is the split level: start building
                                // the complete subtree of the newly-added node in a serial fashion.
                                // NOTE: like in the serial function, make sure we first compute the
                                // children count and only later we assign it into the tree, as the computation
                                // of the children count might end up modifying the tree.
                                const auto children_count = build_tree_ser_impl<ParentLevel + 1u>(
                                    new_tree, new_crit_nodes, cur_code, it_start, it_end,
                                    // NOTE: the children nodes have critical ancestors if either
                                    // the newly-added node is critical or one of its ancestors is.
                                    critical_node || crit_ancestor);
                                get<1>(new_tree[0])[2] = children_count;
                            } else {
                                get<1>(new_tree[0])[2] = build_tree_par_impl<ParentLevel + 1u>(
                                    trees, crit_nodes, cur_code, it_start, it_end, split_level,
                                    critical_node || crit_ancestor);
                            }
                        }
                        checked_uinc(retval, get<1>(new_tree[0])[2]);
                        checked_uinc(retval, size_type(1));
                    }
                });
            }
            tg.wait();
            return retval.load();
        } else {
            ignore_args(parent_code, begin, end, split_level, crit_ancestor);
            return 0;
        }
    }
    void build_tree()
    {
        simple_timer st("node building");
        // Make sure we always have an empty tree when invoking this method.
        assert(m_tree.empty());
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
            throw std::overflow_error("the number of particles (" + std::to_string(m_codes.size())
                                      + ") is too large, and it results in an overflow condition");
        }
        // Computation of the level at which we start building the subtrees serially.
        // Based on the equation (2**NDim)**split_level >= number of cores. So, for instance,
        // on a 16-core machine and 3 dimensions, this results in split_level == 2: we are
        // switching to serial subtree building at the second level where we have up to
        // 64 nodes.
        const unsigned split_level = [] {
            if (const auto hc = std::thread::hardware_concurrency(); hc) {
                // NOTE: use UInt in the shift, as we now that UInt won't
                // overflow thanks to the checks in cbits.
                const auto tmp = std::log(hc) / std::log(UInt(1) << NDim);
                // Make sure we don't return zero on single-core machines.
                return std::max(1u, boost::numeric_cast<unsigned>(std::ceil(tmp)));
            }
            // Just return 1 if hardware_concurrency is not working.
            return 1u;
        }();
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
        m_tree.emplace_back(1,
                            std::array<size_type, 3>{size_type(0), size_type(m_codes.size()),
                                                     // NOTE: the children count gets inited to zero. It
                                                     // will be filled in later.
                                                     size_type(0)},
                            // NOTE: make sure mass and COM coords are initialised in a known state (i.e.,
                            // zero for C++ floating-point).
                            0, std::array<F, NDim>{});
        // Check if the root node is a critical node. It is a critical node if the number of particles is leq m_ncrit
        // (the definition of critical node) or m_max_leaf_n (in which case it will have no children).
        const bool root_is_crit = m_codes.size() <= m_ncrit || m_codes.size() <= m_max_leaf_n;
        if (root_is_crit) {
            // The root node is critical, add it to the global list.
            crit_nodes.push_back({{UInt(1), size_type(0), size_type(m_codes.size())}});
        }
        // Build the rest, if needed.
        if (m_codes.size() > m_max_leaf_n) {
            get<1>(m_tree[0])[2] = build_tree_par_impl<0>(trees, crit_nodes, 1, m_codes.begin(), m_codes.end(),
                                                          split_level, root_is_crit);
        }

        // NOTE: the merge of the subtrees and of the critical nodes lists can be done independently.
        tbb::task_group tg;
        tg.run([&]() {
            // NOTE: this sorting and the computation of the cumulative sizes can be done also in parallel,
            // but it's probably not worth it since the size of trees should be rather small.
            //
            // Sort the subtrees according to the nodal code of the first node.
            std::sort(trees.begin(), trees.end(), [](const auto &t1, const auto &t2) {
                assert(t1.size() && t2.size());
                return node_compare<NDim>(get<0>(t1[0]), get<0>(t2[0]));
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
                [this, &cum_sizes, &trees](const auto &range) {
                    for (auto i = range.begin(); i != range.end(); ++i) {
                        std::copy(trees[i].begin(), trees[i].end(),
                                  &m_tree[boost::numeric_cast<decltype(m_tree.size())>(cum_sizes[i])]);
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
                [this, &cum_sizes, &crit_nodes](const auto &range) {
                    for (auto i = range.begin(); i != range.end(); ++i) {
                        std::copy(crit_nodes[i].begin(), crit_nodes[i].end(),
                                  &m_crit_nodes[boost::numeric_cast<decltype(m_crit_nodes.size())>(cum_sizes[i])]);
                    }
                });
        });
        tg.wait();

        // Various debug checks.
        // Check the tree is sorted according to the nodal code comparison.
        assert(std::is_sorted(m_tree.begin(), m_tree.end(), [](const auto &t1, const auto &t2) {
            return node_compare<NDim>(get<0>(t1), get<0>(t2));
        }));
        // Check that all the nodes contain at least 1 element.
        assert(
            std::all_of(m_tree.begin(), m_tree.end(), [](const auto &tup) { return get<1>(tup)[1] > get<1>(tup)[0]; }));
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

        // NOTE: a couple of final checks to make sure we can use size_type to represent both the tree
        // size and the size of the list of critical nodes.
        if (m_tree.size() > std::numeric_limits<size_type>::max()) {
            throw std::overflow_error("the size of the tree (" + std::to_string(m_tree.size())
                                      + ") is too large, and it results in an overflow condition");
        }
        if (m_crit_nodes.size() > std::numeric_limits<size_type>::max()) {
            throw std::overflow_error("the size of the critical nodes list (" + std::to_string(m_crit_nodes.size())
                                      + ") is too large, and it results in an overflow condition");
        }
    }
    void build_tree_properties()
    {
        simple_timer st("tree properties");
        // NOTE: we check in build_tree() that m_tree.size() can be safely cast
        // to size_type.
        const auto tree_size = static_cast<size_type>(m_tree.size());
        tbb::parallel_for(tbb::blocked_range<size_type>(0u, tree_size), [this](const auto &range) {
            for (auto i = range.begin(); i != range.end(); ++i) {
                auto &tup = m_tree[i];
                // Get the indices and the size for the current node.
                const auto begin = get<1>(tup)[0];
                const auto end = get<1>(tup)[1];
                assert(end > begin);
                const auto size = end - begin;
                // Compute the total mass.
                const auto tot_mass = std::accumulate(m_masses.data() + begin, m_masses.data() + end, F(0));
                // Compute the COM for the coordinates.
                for (std::size_t j = 0; j < NDim; ++j) {
                    F acc(0);
                    auto m_ptr = m_masses.data() + begin;
                    auto c_ptr = m_coords[j].data() + begin;
                    for (std::remove_const_t<decltype(size)> k = 0; k < size; ++k) {
                        acc = fma_wrap(m_ptr[k], c_ptr[k], acc);
                    }
                    get<3>(tup)[j] = acc / tot_mass;
                }
                // Store the total mass.
                get<2>(tup) = tot_mass;
            }
        });
    }
    // Function to discretise the input NDim floating-point coordinates starting at 'it'
    // into a box of a given size box_size.
    template <typename It>
    static auto disc_coords(It it, const F &box_size)
    {
        constexpr UInt factor = UInt(1) << cbits;
        std::array<UInt, NDim> retval;
        for (std::size_t i = 0; i < NDim; ++i, ++it) {
            const auto &x = *it;
            // Translate and rescale the coordinate so that -box_size/2 becomes zero
            // and box_size/2 becomes 1.
            auto tmp = (x + box_size / F(2)) / box_size;
            // Rescale by factor.
            tmp *= factor;
            // Check: don't end up with a nonfinite value.
            if (!std::isfinite(tmp)) {
                throw std::invalid_argument("Not finite!");
            }
            // Check: don't end up outside the [0, factor) range.
            if (tmp < F(0) || tmp >= F(factor)) {
                throw std::invalid_argument("Out of bounds!");
            }
            // Cast to UInt and write to retval.
            retval[i] = static_cast<UInt>(tmp);
            // Last check, make sure we don't overflow.
            if (retval[i] >= factor) {
                throw std::invalid_argument("Out of bounds! (after cast)");
            }
        }
        return retval;
    }
    // Small helper to determine m_ord_ind based on the indirect sorting vector m_isort.
    // This is used when (re)building the tree.
    void isort_to_ord_ind()
    {
        tbb::parallel_for(tbb::blocked_range<size_type>(0u, static_cast<size_type>(m_codes.size())),
                          [this](const auto &range) {
                              for (auto i = range.begin(); i != range.end(); ++i) {
                                  assert(i < m_isort.size());
                                  assert(m_isort[i] < m_ord_ind.size());
                                  m_ord_ind[m_isort[i]] = i;
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

public:
    template <typename It>
    explicit tree(const F &box_size, It m_it, std::array<It, NDim> c_it, const size_type &N,
                  const size_type &max_leaf_n, const size_type &ncrit)
        : m_box_size(box_size), m_max_leaf_n(max_leaf_n), m_ncrit(ncrit)
    {
        simple_timer st("overall tree construction");
        // Check the box size.
        if (!std::isfinite(box_size) || box_size <= F(0)) {
            throw std::invalid_argument("the box size must be a finite positive value, but it is "
                                        + std::to_string(box_size) + " instead");
        }
        // Check the max_leaf_n param.
        if (!max_leaf_n) {
            throw std::invalid_argument("the maximum number of particles per leaf must be nonzero");
        }
        // Check the ncrit param.
        if (!ncrit) {
            throw std::invalid_argument("the critical number of particles for the vectorised computation of the "
                                        "accelerations must be nonzero");
        }
        // Get out soon if there's nothing to do.
        if (!N) {
            return;
        }
        // Prepare the vectors.
        m_masses.resize(N);
        for (auto &vc : m_coords) {
            vc.resize(N);
        }
        // NOTE: these ensure that, from now on, we can just cast
        // freely between the size types of the masses/coords and codes/indices vectors.
        m_codes.resize(boost::numeric_cast<decltype(m_codes.size())>(N));
        m_isort.resize(boost::numeric_cast<decltype(m_isort.size())>(N));
        m_ord_ind.resize(boost::numeric_cast<decltype(m_ord_ind.size())>(N));
        {
            // Do the Morton encoding.
            simple_timer st_m("morton encoding");
            // NOTE: in the parallel for, we need to index into the random-access iterator
            // type It up to the value N. Make sure we can do that.
            using it_diff_t = typename std::iterator_traits<It>::difference_type;
            // NOTE: for use in make_unsigned, it_diff_t must be a C++ integral. This should be ensured
            // by iterator_traits, at least for input iterators:
            // https://en.cppreference.com/w/cpp/iterator/iterator_traits
            using it_udiff_t = std::make_unsigned_t<it_diff_t>;
            if (m_masses.size() > static_cast<it_udiff_t>(std::numeric_limits<it_diff_t>::max())) {
                throw std::overflow_error("the number of particles (" + std::to_string(m_masses.size())
                                          + ") is too large, and it results in an overflow condition");
            }
            tbb::parallel_for(tbb::blocked_range<size_type>(0u, N), [this, &c_it, &m_it](const auto &range) {
                // Temporary structure used in the encoding.
                std::array<F, NDim> tmp_coord;
                // The encoder object.
                morton_encoder<NDim, UInt> me;
                // Determine the particles' codes, and fill in the particles' data.
                for (auto i = range.begin(); i != range.end(); ++i) {
                    // Write the coords in the temp structure and in the data members.
                    for (std::size_t j = 0; j < NDim; ++j) {
                        tmp_coord[j] = *(c_it[j] + static_cast<it_diff_t>(i));
                        m_coords[j][i] = *(c_it[j] + static_cast<it_diff_t>(i));
                    }
                    // Store the mass.
                    m_masses[i] = *(m_it + static_cast<it_diff_t>(i));
                    // Compute and store the code.
                    m_codes[i] = me(disc_coords(tmp_coord.begin(), m_box_size).begin());
                    // Store the index for indirect sorting.
                    m_isort[i] = i;
                }
            });
        }
        // Do the sorting of m_isort.
        indirect_code_sort(m_isort.begin(), m_isort.end());
        {
            // Apply the permutation to the data members.
            simple_timer st_p("permute");
            tbb::task_group tg;
            tg.run([this]() {
                apply_isort(m_codes, m_isort);
                // Make sure the sort worked as intended.
                assert(std::is_sorted(m_codes.begin(), m_codes.end()));
            });
            for (std::size_t j = 0; j < NDim; ++j) {
                tg.run([this, j]() { apply_isort(m_coords[j], m_isort); });
            }
            tg.run([this]() { apply_isort(m_masses, m_isort); });
            // Establish the indices for ordered iteration.
            tg.run([this]() { isort_to_ord_ind(); });
            tg.wait();
        }
        // Now let's proceed to the tree construction.
        build_tree();
        // Now move to the computation of the COM of the nodes.
        build_tree_properties();
        // NOTE: whenever we need ordered iteration on the particles' data,
        // we need to be able to index into the vectors' iterators with values
        // up to the total number of particles. Verify that we can actually do that.
        if (!check_perm_it_range<decltype(m_masses.begin())>(m_masses.size())) {
            throw std::overflow_error("the number of particles (" + std::to_string(m_masses.size())
                                      + ") is too large, and it results in an overflow condition");
        }
    }
    tree(const tree &) = default;

private:
    // Helper to clear all the internal containers.
    void clear_containers()
    {
        m_masses.clear();
        for (auto &coord : m_coords) {
            coord.clear();
        }
        m_codes.clear();
        m_isort.clear();
        m_ord_ind.clear();
        m_tree.clear();
        m_crit_nodes.clear();
    }

public:
    tree(tree &&other) noexcept
        : m_box_size(std::move(other.m_box_size)), m_max_leaf_n(other.m_max_leaf_n), m_ncrit(other.m_ncrit),
          m_masses(std::move(other.m_masses)), m_coords(std::move(other.m_coords)), m_codes(std::move(other.m_codes)),
          m_isort(std::move(other.m_isort)), m_ord_ind(std::move(other.m_ord_ind)), m_tree(std::move(other.m_tree)),
          m_crit_nodes(std::move(other.m_crit_nodes))
    {
        // Make sure other is left in an empty state, otherwise we might
        // have in principle assertions failures in the destructor of other
        // in debug mode.
        other.clear_containers();
    }
    tree &operator=(const tree &other)
    {
        try {
            if (this != &other) {
                m_box_size = other.m_box_size;
                m_max_leaf_n = other.m_max_leaf_n;
                m_ncrit = other.m_ncrit;
                m_masses = other.m_masses;
                m_coords = other.m_coords;
                m_codes = other.m_codes;
                m_isort = other.m_isort;
                m_ord_ind = other.m_ord_ind;
                m_tree = other.m_tree;
                m_crit_nodes = other.m_crit_nodes;
            }
            return *this;
        } catch (...) {
            // NOTE: if we triggered an exception, this might now be
            // in an inconsistent state. Clear out the internal containers
            // to reset to a consistent state before re-throwing.
            clear_containers();
            throw;
        }
    }
    tree &operator=(tree &&other) noexcept
    {
        if (this != &other) {
            m_box_size = std::move(other.m_box_size);
            m_max_leaf_n = other.m_max_leaf_n;
            m_ncrit = other.m_ncrit;
            m_masses = std::move(other.m_masses);
            m_coords = std::move(other.m_coords);
            m_codes = std::move(other.m_codes);
            m_isort = std::move(other.m_isort);
            m_ord_ind = std::move(other.m_ord_ind);
            m_tree = std::move(other.m_tree);
            m_crit_nodes = std::move(other.m_crit_nodes);
            // Make sure other is left in an empty state, otherwise we might
            // have in principle assertions failures in the destructor of other
            // in debug mode.
            other.clear_containers();
        }
        return *this;
    }
    ~tree()
    {
        // Run various debug checks.
#if !defined(NDEBUG)
        for (std::size_t j = 0; j < NDim; ++j) {
            assert(m_masses.size() == m_coords[j].size());
        }
#endif
        assert(m_masses.size() == m_codes.size());
        assert(std::is_sorted(m_codes.begin(), m_codes.end()));
        assert(m_masses.size() == m_isort.size());
        assert(m_masses.size() == m_ord_ind.size());
#if !defined(NDEBUG)
        for (decltype(m_isort.size()) i = 0; i < m_isort.size(); ++i) {
            assert(m_isort[i] < m_ord_ind.size());
            assert(m_ord_ind[m_isort[i]] == i);
        }
        std::sort(m_isort.begin(), m_isort.end());
        assert(std::unique(m_isort.begin(), m_isort.end()) == m_isort.end());
#endif
    }
    friend std::ostream &operator<<(std::ostream &os, const tree &t)
    {
        static_assert(unsigned(std::numeric_limits<UInt>::digits) <= std::numeric_limits<std::size_t>::max());
        os << "Total number of particles: " << t.m_codes.size() << '\n';
        os << "Total number of nodes    : " << t.m_tree.size() << "\n\n";
        os << "First few nodes:\n";
        constexpr unsigned max_nodes = 20;
        auto i = 0u;
        for (const auto &tup : t.m_tree) {
            if (i > max_nodes) {
                break;
            }
            os << std::bitset<std::numeric_limits<UInt>::digits>(get<0>(tup)) << '|' << get<1>(tup)[0] << ','
               << get<1>(tup)[1] << ',' << get<1>(tup)[2] << "|" << get<2>(tup) << "|[";
            for (std::size_t j = 0; j < NDim; ++j) {
                os << get<3>(tup)[j];
                if (j < NDim - 1u) {
                    os << ", ";
                }
            }
            os << "]\n";
            ++i;
        }
        if (i > max_nodes) {
            std::cout << "...\n";
        }
        return os;
    }

private:
    // Temporary storage used to store the distances between the particles
    // of a node and the COM of another node while traversing the tree.
    static auto &vec_acc_tmp_vecs()
    {
        static thread_local std::array<fp_vector, NDim + 1u> tmp_vecs;
        return tmp_vecs;
    }
    // Temporary storage to accumulate the accelerations induced on the
    // particles of a critical node. Data in here will be copied to
    // the output array after the accelerations from all the other
    // particles/nodes in the domain have been computed.
    static auto &vec_acc_tmp_res()
    {
        static thread_local std::array<fp_vector, NDim> tmp_res;
        return tmp_res;
    }
    // Temporary vectors to store the data of a target node during traversal.
    static auto &tgt_tmp_data()
    {
        static thread_local std::array<fp_vector, NDim + 1u> tmp_tgt;
        return tmp_tgt;
    }
    // Compute the element-wise attraction on the batch of particles at xvec1, yvec1, zvec1 by the
    // particle(s) at x2, y2, z2 with mass(es) mvec2, and add the result into res_x_vec, res_y_vec,
    // res_z_vec. B must be an xsimd batch. BS must be either the same as B, or the scalar type of B.
    template <typename B, typename BS>
    static void batch_bs_3d(B &res_x_vec, B &res_y_vec, B &res_z_vec, B xvec1, B yvec1, B zvec1, BS x2, BS y2, BS z2,
                            BS m2)
    {
        const B diff_x = x2 - xvec1;
        const B diff_y = y2 - yvec1;
        const B diff_z = z2 - zvec1;
        const B dist2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
        B m2_dist3;
        if constexpr (use_fast_inv_sqrt<B>) {
            m2_dist3 = m2 * inv_sqrt_3(dist2);
        } else {
            const B dist = xsimd::sqrt(dist2);
            const B dist3 = dist * dist2;
            m2_dist3 = m2 / dist3;
        }
        res_x_vec = xsimd::fma(diff_x, m2_dist3, res_x_vec);
        res_y_vec = xsimd::fma(diff_y, m2_dist3, res_y_vec);
        res_z_vec = xsimd::fma(diff_z, m2_dist3, res_z_vec);
    }
    // Compute the accelerations on the particles of a target node induced by the particles of another
    // node (the source). pidx and size are the starting index (in the particles arrays) and the size of the target
    // node. begin/end is the range, in the tree structure, encompassing the source node and its children.
    // node_size2 is the square of the size of the source node. The accelerations will be written into the
    // temporary storage provided by vec_acc_tmp_res(). SLevel is the tree level of the source node.
    template <unsigned SLevel>
    void vec_acc_from_node(const F &theta2, size_type pidx, size_type size, size_type begin, size_type end,
                           const F &node_size2) const
    {
        if constexpr (SLevel <= cbits) {
            // Check that SLevel is consistent with the tree data.
            assert(tree_level<NDim>(get<0>(m_tree[begin])) == SLevel);
            // Check that node_size2 is correct.
            assert(node_size2 == m_box_size / (UInt(1) << SLevel) * m_box_size / (UInt(1) << SLevel));
            // Prepare pointers to the input and output data.
            auto &tmp_res = vec_acc_tmp_res();
            auto &tmp_tgt = tgt_tmp_data();
            std::array<F *, NDim> res_ptrs;
            std::array<const F *, NDim> c_ptrs;
            for (std::size_t j = 0; j < NDim; ++j) {
                res_ptrs[j] = tmp_res[j].data();
                c_ptrs[j] = tmp_tgt[j].data();
            }
            // Temporary vectors to store the pos differences and dist3 below.
            // We will store the data generated in the BH criterion check because
            // we can re-use it later to compute the accelerations.
            auto &tmp_vecs = vec_acc_tmp_vecs();
            std::array<F *, NDim + 1u> tmp_ptrs;
            for (std::size_t j = 0; j < NDim + 1u; ++j) {
                tmp_vecs[j].resize(size);
                tmp_ptrs[j] = tmp_vecs[j].data();
            }
            // Copy locally the COM coords of the source.
            const auto com_pos = get<3>(m_tree[begin]);
            // Check the distances of all the particles of the target
            // node from the COM of the source.
            bool bh_flag = true;
            size_type i = 0;
            if constexpr (simd_enabled && NDim == 3u) {
                // The SIMD-accelerated part.
                auto x_ptr = c_ptrs[0], y_ptr = c_ptrs[1], z_ptr = c_ptrs[2];
                auto tmp_x = tmp_ptrs[0], tmp_y = tmp_ptrs[1], tmp_z = tmp_ptrs[2], tmp_dist3 = tmp_ptrs[3];
                tuple_for_each(simd_sizes<F>, [&](auto s) {
                    constexpr auto batch_size = s.value;
                    using batch_type = xsimd::batch<F, batch_size>;
                    const auto vec_size = static_cast<size_type>(size - size % batch_size);
                    const batch_type node_size2_vec(node_size2), theta2_vec(theta2), x_com_vec(com_pos[0]),
                        y_com_vec(com_pos[1]), z_com_vec(com_pos[2]);
                    for (; i < vec_size; i += batch_size, x_ptr += batch_size, y_ptr += batch_size, z_ptr += batch_size,
                                         tmp_x += batch_size, tmp_y += batch_size, tmp_z += batch_size,
                                         tmp_dist3 += batch_size) {
                        const auto diff_x = x_com_vec - batch_type(x_ptr, xsimd::aligned_mode{}),
                                   diff_y = y_com_vec - batch_type(y_ptr, xsimd::aligned_mode{}),
                                   diff_z = z_com_vec - batch_type(z_ptr, xsimd::aligned_mode{}),
                                   dist2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                        if (xsimd::any(node_size2_vec >= theta2_vec * dist2)) {
                            // At least one particle in the current batch fails the BH criterion
                            // check. Mark the bh_flag as false, and set i to size in order
                            // to skip the scalar calculation later. Then break out.
                            bh_flag = false;
                            i = size;
                            break;
                        }
                        diff_x.store_aligned(tmp_x);
                        diff_y.store_aligned(tmp_y);
                        diff_z.store_aligned(tmp_z);
                        if constexpr (use_fast_inv_sqrt<batch_type>) {
                            inv_sqrt_3(dist2).store_aligned(tmp_dist3);
                        } else {
                            (xsimd::sqrt(dist2) * dist2).store_aligned(tmp_dist3);
                        }
                    }
                });
            }
            for (; i < size; ++i) {
                F dist2(0);
                for (std::size_t j = 0; j < NDim; ++j) {
                    // Store the differences for later use.
                    const auto diff = com_pos[j] - c_ptrs[j][i];
                    tmp_ptrs[j][i] = diff;
                    dist2 = fma_wrap(diff, diff, dist2);
                }
                if (node_size2 >= theta2 * dist2) {
                    // At least one of the particles in the target
                    // node is too close to the COM. Set the flag
                    // to false and exit.
                    bh_flag = false;
                    break;
                }
                // Store dist3 for later use.
                // NOTE: in the scalar part, we always store dist3.
                tmp_ptrs[NDim][i] = std::sqrt(dist2) * dist2;
            }
            if (bh_flag) {
                // The source node satisfies the BH criterion for
                // all the particles of the target node. Add the accelerations.
                //
                // Load the mass of the COM of the sibling node.
                const auto m_com = get<2>(m_tree[begin]);
                i = 0;
                if constexpr (simd_enabled && NDim == 3u) {
                    // The SIMD-accelerated part.
                    auto tmp_x = tmp_ptrs[0], tmp_y = tmp_ptrs[1], tmp_z = tmp_ptrs[2], tmp_dist3 = tmp_ptrs[3];
                    auto res_x = res_ptrs[0], res_y = res_ptrs[1], res_z = res_ptrs[2];
                    tuple_for_each(simd_sizes<F>, [&](auto s) {
                        constexpr auto batch_size = s.value;
                        using batch_type = xsimd::batch<F, batch_size>;
                        const batch_type m_com_vec(m_com);
                        const auto vec_size = static_cast<size_type>(size - size % batch_size);
                        for (; i < vec_size; i += batch_size, tmp_x += batch_size, tmp_y += batch_size,
                                             tmp_z += batch_size, tmp_dist3 += batch_size, res_x += batch_size,
                                             res_y += batch_size, res_z += batch_size) {
                            const auto m_com_dist3_vec = use_fast_inv_sqrt<batch_type>
                                                             ? m_com_vec * batch_type(tmp_dist3, xsimd::aligned_mode{})
                                                             : m_com_vec / batch_type(tmp_dist3, xsimd::aligned_mode{}),
                                       xdiff = batch_type(tmp_x, xsimd::aligned_mode{}),
                                       ydiff = batch_type(tmp_y, xsimd::aligned_mode{}),
                                       zdiff = batch_type(tmp_z, xsimd::aligned_mode{});
                            xsimd::fma(xdiff, m_com_dist3_vec, batch_type(res_x, xsimd::aligned_mode{}))
                                .store_aligned(res_x);
                            xsimd::fma(ydiff, m_com_dist3_vec, batch_type(res_y, xsimd::aligned_mode{}))
                                .store_aligned(res_y);
                            xsimd::fma(zdiff, m_com_dist3_vec, batch_type(res_z, xsimd::aligned_mode{}))
                                .store_aligned(res_z);
                        }
                    });
                }
                for (; i < size; ++i) {
                    const auto m_com_dist3 = m_com / tmp_ptrs[NDim][i];
                    for (std::size_t j = 0; j < NDim; ++j) {
                        res_ptrs[j][i] = fma_wrap(tmp_ptrs[j][i], m_com_dist3, res_ptrs[j][i]);
                    }
                }
                return;
            }
            // At least one particle in the target node is too close to the
            // COM of the source node. If we can, we go deeper, otherwise we must compute
            // all the pairwise interactions between all the particles in the
            // target and source nodes.
            const auto n_children = get<1>(m_tree[begin])[2];
            if (!n_children) {
                // The source node is a leaf, compute all the accelerations induced by its
                // particles on the particles of the target node.
                //
                // NOTE: we do this here, rather than earlier, as it might be that the node
                // is far enough to satisfy the BH criterion. In such a case we save a lot
                // of operations, as we are avoiding all the pairwise interactions.
                //
                // Establish the range of the source node.
                const auto leaf_begin = get<1>(m_tree[begin])[0], leaf_end = get<1>(m_tree[begin])[1];
                if constexpr (simd_enabled && NDim == 3u) {
                    // Pointers to the target node data.
                    const auto x_ptr1 = c_ptrs[0], y_ptr1 = c_ptrs[1], z_ptr1 = c_ptrs[2];
                    // Pointers to the source node data.
                    const auto x_ptr2 = m_coords[0].data() + leaf_begin, y_ptr2 = m_coords[1].data() + leaf_begin,
                               z_ptr2 = m_coords[2].data() + leaf_begin, m_ptr2 = m_masses.data() + leaf_begin;
                    // Pointer to the result data.
                    const auto res_x = res_ptrs[0], res_y = res_ptrs[1], res_z = res_ptrs[2];
                    // The number of particles in the source node.
                    const auto size_leaf = static_cast<size_type>(leaf_end - leaf_begin);
                    // NOTE: we will now divide the source and target nodes into blocks whose sizes
                    // are multiples of the available simd vector sizes. For instance, if the
                    // available vector sizes are 16, 8, 4 (AVX512 float) and the target/source
                    // nodes have both size 45, then we will have a size-16 block in the [0, 32)
                    // range, a size-8 block in the [0, 40) range, a size-4 block in the [0, 44)
                    // range and a remainder block in the [44, 45) range. We then do a double iteration
                    // on the simd sizes, which yields the following pairs for the simd sizes (in order):
                    //
                    // 16-16, 16-8, 16-4,
                    // 8-16, 8-8, 8-4,
                    // 4-16, 4-8, 4-4.
                    //
                    // For each pair, we pick the minimum simd size (e.g., 16-8 will yield a simd size of 8,
                    // 4-16 a simd size of 4, etc.) and we compute block-block interactions via vector
                    // instructions. A worked out example:
                    //
                    // 16-16, i1 = 0, i2 = 0, simd_size = 16 -> [0, 32) x [0, 32) (simd-simd)
                    // 16-8, i1 = 0, i2 = 32, simd_size = 8 -> [0, 32) x [32, 40) (simd-simd)
                    // 16-4, i1 = 0, i2 = 40, simd_size = 4 -> [0, 32) x [40, 44) (simd-simd)
                    // remainder, i1 = 0, i2 = 44, simd_size = 16 -> [0, 32) x [40, 45) (simd-scalar)
                    //
                    // 8-16, i1 = 32, i2 = 0, simd_size = 8 -> [32, 40) x [0, 32) (simd-simd)
                    // 8-8, i1 = 32, i2 = 32, simd_size = 8 -> [32, 40) x [32, 40) (simd-simd)
                    // 8-4, i1 = 32, i2 = 40, simd_size = 4 -> [32, 40) x [40, 44) (simd-simd)
                    // remainder, i1 = 32, i2 = 44, simd_size = 8 -> [32, 40) x [40, 45) (simd-scalar)
                    //
                    // 4-16, i1 = 40, i2 = 0, simd_size = 4 -> [40, 44) x [0, 32) (simd-simd)
                    // 4-8, i1 = 40, i2 = 32, simd_size = 4 -> [40, 44) x [32, 40) (simd-simd)
                    // 4-4, i1 = 40, i2 = 40, simd_size = 4 -> [40, 44) x [40, 44) (simd-simd)
                    // remainder, i1 = 40, i2 = 44, simd_size = 4 -> [40, 44) x [40, 45) (simd-scalar)
                    size_type i1 = 0;
                    tuple_for_each(simd_sizes<F>, [&](auto s1) {
                        constexpr auto batch_size1 = s1.value;
                        using batch_type1 = xsimd::batch<F, batch_size1>;
                        const auto vec_size1 = static_cast<size_type>(size - size % batch_size1);
                        size_type i2 = 0;
                        tuple_for_each(simd_sizes<F>, [&](auto s2) {
                            constexpr auto batch_size2 = s2.value;
                            const auto vec_size2 = static_cast<size_type>(size_leaf - size_leaf % batch_size2);
                            // The simd vector size is the smallest between s1 and s2.
                            // NOTE: we use s1.value rather than batch_size1 to workaround a GCC ICE.
                            constexpr auto batch_size = std::min(s1.value, batch_size2);
                            using batch_type = xsimd::batch<F, batch_size>;
                            if (i2 == vec_size2) {
                                // Exit early if the inner loop has no iterations.
                                return;
                            }
                            for (auto idx1 = i1; idx1 < vec_size1; idx1 += batch_size) {
                                // Load the current batch of target data.
                                const auto xvec1 = batch_type(x_ptr1 + idx1, xsimd::aligned_mode{}),
                                           yvec1 = batch_type(y_ptr1 + idx1, xsimd::aligned_mode{}),
                                           zvec1 = batch_type(z_ptr1 + idx1, xsimd::aligned_mode{});
                                // Init the batches for computing the accelerations, loading the
                                // accumulated acceleration for the current batch.
                                auto res_x_vec = batch_type(res_x + idx1, xsimd::aligned_mode{}),
                                     res_y_vec = batch_type(res_y + idx1, xsimd::aligned_mode{}),
                                     res_z_vec = batch_type(res_z + idx1, xsimd::aligned_mode{});
                                for (auto idx2 = i2; idx2 < vec_size2; idx2 += batch_size) {
                                    auto xvec2 = batch_type(x_ptr2 + idx2), yvec2 = batch_type(y_ptr2 + idx2),
                                         zvec2 = batch_type(z_ptr2 + idx2), mvec2 = batch_type(m_ptr2 + idx2);
                                    batch_bs_3d(res_x_vec, res_y_vec, res_z_vec, xvec1, yvec1, zvec1, xvec2, yvec2,
                                                zvec2, mvec2);
                                    for (std::size_t j = 1; j < batch_size; ++j) {
                                        // Above we computed the element-wise accelerations of a source batch
                                        // onto a target batch. We need to rotate the source batch
                                        // batch_size - 1 times and perform again the computation in order
                                        // to compute all possible particle-particle interactions.
                                        xvec2 = rotate(xvec2);
                                        yvec2 = rotate(yvec2);
                                        zvec2 = rotate(zvec2);
                                        mvec2 = rotate(mvec2);
                                        batch_bs_3d(res_x_vec, res_y_vec, res_z_vec, xvec1, yvec1, zvec1, xvec2, yvec2,
                                                    zvec2, mvec2);
                                    }
                                }
                                // Store the updated accelerations in the temporary vectors.
                                res_x_vec.store_aligned(res_x + idx1);
                                res_y_vec.store_aligned(res_y + idx1);
                                res_z_vec.store_aligned(res_z + idx1);
                            }
                            // Update the index into the source node.
                            i2 = vec_size2;
                        });
                        if (i2 == size_leaf) {
                            // NOTE: exit early if possible. Note that we have to update
                            // i1 manually here (if we run the loop below, it will be updated
                            // in the loop header).
                            i1 = vec_size1;
                            return;
                        }
                        for (; i1 < vec_size1; i1 += batch_size1) {
                            const auto xvec1 = batch_type1(x_ptr1 + i1, xsimd::aligned_mode{}),
                                       yvec1 = batch_type1(y_ptr1 + i1, xsimd::aligned_mode{}),
                                       zvec1 = batch_type1(z_ptr1 + i1, xsimd::aligned_mode{});
                            auto res_x_vec = batch_type1(res_x + i1, xsimd::aligned_mode{}),
                                 res_y_vec = batch_type1(res_y + i1, xsimd::aligned_mode{}),
                                 res_z_vec = batch_type1(res_z + i1, xsimd::aligned_mode{});
                            for (auto idx2 = i2; idx2 < size_leaf; ++idx2) {
                                // NOTE: here we are doing batch/scalar interactions.
                                batch_bs_3d(res_x_vec, res_y_vec, res_z_vec, xvec1, yvec1, zvec1, x_ptr2[idx2],
                                            y_ptr2[idx2], z_ptr2[idx2], m_ptr2[idx2]);
                            }
                            res_x_vec.store_aligned(res_x + i1);
                            res_y_vec.store_aligned(res_y + i1);
                            res_z_vec.store_aligned(res_z + i1);
                        }
                    });
                    // These are the interactions between the remainder of the target node
                    // and the source node.
                    // NOTE: here we are not using any simd operations. In principle, we could either:
                    // - do scalar-batch interactions followed by an horizontal sum to accumulate
                    //   the contribution of a source batch onto a single target particle, or
                    // - do scalar-batch interactions followed by a store to a local buffer and a
                    //   scalar accumulation.
                    // The first option has bad performance, the second one does not seem to buy anything
                    // performance-wise. Let's keep in mind the implementation at
                    // 6a1618d9f5db8a61b76cd32e3c38fe2034bfe666
                    // if we ever want to revisit this.
                    for (; i1 < size; ++i1) {
                        const auto x1 = x_ptr1[i1], y1 = y_ptr1[i1], z1 = z_ptr1[i1];
                        // Load the current accelerations on the target particle into local variables.
                        auto rx = res_x[i1], ry = res_y[i1], rz = res_z[i1];
                        for (size_type i2 = 0; i2 < size_leaf; ++i2) {
                            // NOTE: scalar/scalar interactions.
                            const auto diff_x = x_ptr2[i2] - x1, diff_y = y_ptr2[i2] - y1, diff_z = z_ptr2[i2] - z1,
                                       dist2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z,
                                       dist = std::sqrt(dist2), dist3 = dist * dist2, m_dist3 = m_ptr2[i2] / dist3;
                            rx = fma_wrap(diff_x, m_dist3, rx);
                            ry = fma_wrap(diff_y, m_dist3, ry);
                            rz = fma_wrap(diff_z, m_dist3, rz);
                        }
                        // Store the updated accelerations.
                        res_x[i1] = rx;
                        res_y[i1] = ry;
                        res_z[i1] = rz;
                    }
                } else {
                    // Local variables for the scalar computation.
                    std::array<F, NDim> pos1, diffs;
                    for (size_type i1 = 0; i1 < size; ++i1) {
                        // Load the coordinates of the current particle
                        // in the target node.
                        for (std::size_t j = 0; j < NDim; ++j) {
                            pos1[j] = c_ptrs[j][i1];
                        }
                        // Iterate over the particles in the sibling node.
                        for (size_type i2 = leaf_begin; i2 < leaf_end; ++i2) {
                            F dist2(0);
                            for (std::size_t j = 0; j < NDim; ++j) {
                                diffs[j] = m_coords[j][i2] - pos1[j];
                                dist2 = fma_wrap(diffs[j], diffs[j], dist2);
                            }
                            const auto dist = std::sqrt(dist2), dist3 = dist * dist2, m_dist3 = m_masses[i2] / dist3;
                            for (std::size_t j = 0; j < NDim; ++j) {
                                tmp_res[j][i1] = fma_wrap(diffs[j], m_dist3, tmp_res[j][i1]);
                            }
                        }
                    }
                }
                return;
            }
            // We can go deeper in the tree.
            //
            // Determine the size of the node at the next level.
            const auto next_node_size = m_box_size / (UInt(1) << (SLevel + 1u));
            const auto next_node_size2 = next_node_size * next_node_size;
            // Bump up begin to move to the first child.
            for (++begin; begin != end; begin += get<1>(m_tree[begin])[2] + 1u) {
                vec_acc_from_node<SLevel + 1u>(theta2, pidx, size, begin, begin + get<1>(m_tree[begin])[2] + 1u,
                                               next_node_size2);
            }
        } else {
            ignore_args(pidx, size, begin, end);
            // NOTE: we cannot go deeper than the maximum level of the tree.
            // The n_children check above will prevent reaching this point at runtime.
            assert(false);
        }
    }
    // Compute the accelerations on the particles in a node due to node's particles themselves.
    // node_begin is the starting index of the node in the particles arrays. npart is the
    // number of particles in the node. The self accelerations will be added to the accelerations
    // in the temporary storage.
    void vec_node_self_interactions(size_type node_begin, size_type npart) const
    {
        // Prepare common pointers to the input and output data.
        auto &tmp_res = vec_acc_tmp_res();
        const auto m_ptr = m_masses.data() + node_begin;
        if constexpr (simd_enabled && NDim == 3u) {
            // xsimd batch type.
            using b_type = xsimd::simd_type<F>;
            // Size of b_type.
            constexpr auto b_size = b_type::size;
            // Shortcuts to the node coordinates/masses.
            const auto x_ptr = m_coords[0].data() + node_begin, y_ptr = m_coords[1].data() + node_begin,
                       z_ptr = m_coords[2].data() + node_begin;
            // Shortcuts to the result vectors.
            auto res_x = tmp_res[0].data(), res_y = tmp_res[1].data(), res_z = tmp_res[2].data();
            const auto vec_size = static_cast<size_type>(npart - npart % b_size);
            auto x_ptr1 = x_ptr, y_ptr1 = y_ptr, z_ptr1 = z_ptr;
            size_type i1 = 0;
            for (; i1 < vec_size; i1 += b_size, x_ptr1 += b_size, y_ptr1 += b_size, z_ptr1 += b_size, res_x += b_size,
                                  res_y += b_size, res_z += b_size) {
                // Load the current accelerations from the temporary result vectors.
                auto res_x_vec = xsimd::load_aligned(res_x), res_y_vec = xsimd::load_aligned(res_y),
                     res_z_vec = xsimd::load_aligned(res_z);
                // Load the data for the particles under consideration.
                const auto xvec1 = xsimd::load_unaligned(x_ptr1), yvec1 = xsimd::load_unaligned(y_ptr1),
                           zvec1 = xsimd::load_unaligned(z_ptr1);
                // Iterate over all the particles in the node and compute the accelerations
                // on the particles under consideration.
                auto x_ptr2 = x_ptr, y_ptr2 = y_ptr, z_ptr2 = z_ptr, m_ptr2 = m_ptr;
                size_type i2 = 0;
                for (; i2 < vec_size;
                     i2 += b_size, x_ptr2 += b_size, y_ptr2 += b_size, z_ptr2 += b_size, m_ptr2 += b_size) {
                    // NOTE: batch/batch interactions.
                    // Load the current batch of particles exerting gravity.
                    auto xvec2 = xsimd::load_unaligned(x_ptr2), yvec2 = xsimd::load_unaligned(y_ptr2),
                         zvec2 = xsimd::load_unaligned(z_ptr2), mvec2 = xsimd::load_unaligned(m_ptr2);
                    if (i2 != i1) {
                        // NOTE: if i2 == i1, we want to skip the first batch-batch
                        // permutation, as we don't want to compute self-accelerations.
                        batch_bs_3d(res_x_vec, res_y_vec, res_z_vec, xvec1, yvec1, zvec1, xvec2, yvec2, zvec2, mvec2);
                    }
                    // Iterate over all the other possible batch-batch permutations
                    // by rotating the data in xvec2, yvec2, zvec2 and mvec2.
                    for (std::size_t j = 1; j < b_size; ++j) {
                        xvec2 = rotate(xvec2);
                        yvec2 = rotate(yvec2);
                        zvec2 = rotate(zvec2);
                        mvec2 = rotate(mvec2);
                        batch_bs_3d(res_x_vec, res_y_vec, res_z_vec, xvec1, yvec1, zvec1, xvec2, yvec2, zvec2, mvec2);
                    }
                }
                // Do the remaining scalar part.
                for (; i2 < npart; ++i2, ++x_ptr2, ++y_ptr2, ++z_ptr2, ++m_ptr2) {
                    // NOTE: batch/scalar interactions.
                    // NOTE: i2 cannot be the same as i1, since i1 is the start of a simd-size
                    // block and i2 is now in a sub-simd-size block at the end of the particle list.
                    assert(i2 != i1);
                    batch_bs_3d(res_x_vec, res_y_vec, res_z_vec, xvec1, yvec1, zvec1, *x_ptr2, *y_ptr2, *z_ptr2,
                                *m_ptr2);
                }
                // Write out the updated accelerations.
                xsimd::store_aligned(res_x, res_x_vec);
                xsimd::store_aligned(res_y, res_y_vec);
                xsimd::store_aligned(res_z, res_z_vec);
            }
            // Do the remaining scalar part (that is, the remainder of the particle list modulo batch size).
            // NOTE: here we are still using horizontal add, which does not look like a performance win
            // in other situations. We should revisit its usage when we iterate over the self interactions
            // function.
            for (; i1 < npart; ++i1, ++x_ptr1, ++y_ptr1, ++z_ptr1, ++res_x, ++res_y, ++res_z) {
                auto x_ptr2 = x_ptr, y_ptr2 = y_ptr, z_ptr2 = z_ptr, m_ptr2 = m_ptr;
                const auto x1 = *x_ptr1, y1 = *y_ptr1, z1 = *z_ptr1;
                // Load locally the current accelerations.
                auto rx = *res_x, ry = *res_y, rz = *res_z;
                size_type i2 = 0;
                for (; i2 < vec_size;
                     i2 += b_size, x_ptr2 += b_size, y_ptr2 += b_size, z_ptr2 += b_size, m_ptr2 += b_size) {
                    // NOTE: scalar/batch interactions.
                    const auto diff_x = xsimd::load_unaligned(x_ptr2) - x1, diff_y = xsimd::load_unaligned(y_ptr2) - y1,
                               diff_z = xsimd::load_unaligned(z_ptr2) - z1, mvec2 = xsimd::load_unaligned(m_ptr2),
                               dist2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                    b_type m2_dist3;
                    if constexpr (use_fast_inv_sqrt<b_type>) {
                        m2_dist3 = mvec2 * inv_sqrt_3(dist2);
                    } else {
                        const auto dist = xsimd::sqrt(dist2);
                        const auto dist3 = dist * dist2;
                        m2_dist3 = mvec2 / dist3;
                    }
                    rx += xsimd::hadd(diff_x * m2_dist3);
                    ry += xsimd::hadd(diff_y * m2_dist3);
                    rz += xsimd::hadd(diff_z * m2_dist3);
                }
                // NOTE: we are about to do the remainder vs remainder interactions: i2 must be at the beginning
                // of the remainder, thus it cannot be greater than i1 which is an index in the remainder.
                assert(i2 <= i1);
                for (; i2 < npart; ++i2, ++x_ptr2, ++y_ptr2, ++z_ptr2, ++m_ptr2) {
                    // NOTE: scalar/scalar interactions.
                    if (i2 != i1) {
                        // Avoid self interactions.
                        const auto diff_x = *x_ptr2 - x1, diff_y = *y_ptr2 - y1, diff_z = *z_ptr2 - z1,
                                   dist2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z, dist = std::sqrt(dist2),
                                   dist3 = dist * dist2, m2_dist3 = *m_ptr2 / dist3;
                        rx = fma_wrap(diff_x, m2_dist3, rx);
                        ry = fma_wrap(diff_y, m2_dist3, ry);
                        rz = fma_wrap(diff_z, m2_dist3, rz);
                    }
                }
                // Store the updated accelerations.
                *res_x = rx;
                *res_y = ry;
                *res_z = rz;
            }
        } else {
            // Shortcuts to the input coordinates.
            std::array<const F *, NDim> c_ptrs;
            for (std::size_t j = 0; j < NDim; ++j) {
                c_ptrs[j] = m_coords[j].data() + node_begin;
            }
            // Temporary vectors to be used in the loops below.
            std::array<F, NDim> diffs, pos1;
            for (size_type i1 = 0; i1 < npart; ++i1) {
                // Load the coords of the current particle.
                for (std::size_t j = 0; j < NDim; ++j) {
                    pos1[j] = c_ptrs[j][i1];
                }
                // Load the mass of the current particle.
                const auto m1 = m_ptr[i1];
                // The acceleration vector on the current particle
                // (inited to zero).
                std::array<F, NDim> a1{};
                for (size_type i2 = i1 + 1u; i2 < npart; ++i2) {
                    // Determine dist2, dist and dist3.
                    F dist2(0);
                    for (std::size_t j = 0; j < NDim; ++j) {
                        diffs[j] = c_ptrs[j][i2] - pos1[j];
                        dist2 = fma_wrap(diffs[j], diffs[j], dist2);
                    }
                    const auto dist = std::sqrt(dist2), dist3 = dist2 * dist, m2_dist3 = m_ptr[i2] / dist3,
                               m1_dist3 = m1 / dist3;
                    // Accumulate the accelerations, both in the local
                    // accumulator for the current particle and in the global
                    // acc vector for the opposite acceleration.
                    for (std::size_t j = 0; j < NDim; ++j) {
                        a1[j] = fma_wrap(m2_dist3, diffs[j], a1[j]);
                        // NOTE: this is a fused multiply-sub.
                        tmp_res[j][i2] = fma_wrap(m1_dist3, -diffs[j], tmp_res[j][i2]);
                    }
                }
                // Update the acceleration on the first particle
                // in the temporary storage.
                for (std::size_t j = 0; j < NDim; ++j) {
                    tmp_res[j][i1] += a1[j];
                }
            }
        }
    }
    // Compute the total acceleration on the particles in a target node. node_begin is the starting index
    // of the target in the particles' arrays, npart the number of particles in the node. [sib_begin, sib_end)
    // is the index range, in the tree structure, encompassing the target, its parent and its parent's siblings at the
    // tree level Level. nodal_code is the nodal code of the target, node_level its level.
    template <unsigned Level>
    void vec_acc_on_node(const F &theta2, size_type node_begin, size_type npart, UInt nodal_code, size_type sib_begin,
                         size_type sib_end, unsigned node_level) const
    {
        if constexpr (Level <= cbits) {
            // Make sure the node level is consistent with the nodal code.
            assert(node_level == tree_level<NDim>(nodal_code));
            // We proceed breadth-first examining all the siblings of the target's parent
            // (or of the node itself, at the last iteration) at the current level.
            //
            // Compute the shifted code. This is the nodal code of the target's parent
            // at the current level (or the nodal code of the target itself at the last
            // iteration).
            const auto s_code = nodal_code >> ((node_level - Level) * NDim);
            auto new_sib_begin = sib_end;
            // Determine the size of the target at the current level.
            const auto node_size = m_box_size / (UInt(1) << Level);
            const auto node_size2 = node_size * node_size;
            for (auto idx = sib_begin; idx != sib_end; idx += get<1>(m_tree[idx])[2] + 1u) {
                if (get<0>(m_tree[idx]) == s_code) {
                    // We are in the target's parent, or the target itself.
                    if (Level == node_level) {
                        // Last iteration, we are in the target itself. Compute the
                        // self-interactions within the target.
                        vec_node_self_interactions(node_begin, npart);
                    } else {
                        // We identified the parent of the target at the current
                        // level. Store its starting index for later.
                        new_sib_begin = idx;
                    }
                } else {
                    // Compute the accelerations from the current sibling.
                    vec_acc_from_node<Level>(theta2, node_begin, npart, idx, idx + 1u + get<1>(m_tree[idx])[2],
                                             node_size2);
                }
            }
            if (Level != node_level) {
                // If we are not at the last iteration, we must have changed
                // new_sib_begin in the loop above.
                assert(new_sib_begin != sib_end);
            } else {
                ignore_args(new_sib_begin);
            }
            if (Level < node_level) {
                // We are not at the level of the target yet. Recurse down.
                vec_acc_on_node<Level + 1u>(theta2, node_begin, npart, nodal_code, new_sib_begin + 1u,
                                            new_sib_begin + 1u + get<1>(m_tree[new_sib_begin])[2], node_level);
            }
        } else {
            ignore_args(node_begin, npart, nodal_code, sib_begin, sib_end);
            // NOTE: we can never get to a level higher than cbits, which is the maximum node level.
            // This is prevented at runtime by the Level < node_level check.
            assert(false);
        }
    }
    // Top level function fo the vectorised computation of the accelerations.
    template <typename It>
    void vec_accs_impl(std::array<It, NDim> &out, const F &theta2) const
    {
        tbb::parallel_for(tbb::blocked_range<decltype(m_crit_nodes.size())>(0u, m_crit_nodes.size()),
                          [this, &theta2, &out](const auto &range) {
                              auto &tmp_res = vec_acc_tmp_res();
                              auto &tmp_tgt = tgt_tmp_data();
                              for (auto i = range.begin(); i != range.end(); ++i) {
                                  const auto nodal_code = get<0>(m_crit_nodes[i]);
                                  const auto node_begin = get<1>(m_crit_nodes[i]);
                                  const auto npart = static_cast<size_type>(get<2>(m_crit_nodes[i]) - node_begin);
                                  // NOTE: in principle this could be pre-computed during the construction of the tree
                                  // (perhaps for free?). Not sure if it's worth it.
                                  const auto node_level = tree_level<NDim>(nodal_code);
                                  // Prepare the temporary vectors containing the result.
                                  for (auto &v : tmp_res) {
                                      // Resize and fill with zeroes.
                                      v.resize(npart);
                                      std::fill(v.begin(), v.end(), F(0));
                                  }
                                  // Prepare the temporary vectors containing the target node's data.
                                  for (std::size_t j = 0; j < NDim; ++j) {
                                      tmp_tgt[j].resize(npart);
                                      std::copy(m_coords[j].data() + node_begin,
                                                m_coords[j].data() + node_begin + npart, tmp_tgt[j].data());
                                  }
                                  tmp_tgt[NDim].resize(npart);
                                  std::copy(m_masses.data() + node_begin, m_masses.data() + node_begin + npart,
                                            tmp_tgt[NDim].data());
                                  // Do the computation.
                                  vec_acc_on_node<0>(theta2, node_begin, npart, nodal_code, size_type(0),
                                                     size_type(m_tree.size()), node_level);
                                  // Write out the result.
                                  using it_diff_t = typename std::iterator_traits<It>::difference_type;
                                  for (std::size_t j = 0; j != NDim; ++j) {
                                      std::copy(tmp_res[j].data(), tmp_res[j].data() + tmp_res[j].size(),
                                                out[j] + boost::numeric_cast<it_diff_t>(node_begin));
                                  }
                              }
                          });
    }
    // Top level dispatcher for the accs functions. It will run a few checks and then invoke vec_accs_impl().
    template <bool Ordered, typename Output>
    void accs_dispatch(Output &out, const F &theta) const
    {
        simple_timer st("vector accs computation");
        const auto theta2 = theta * theta;
        // Input param check.
        if (!std::isfinite(theta2)) {
            throw std::domain_error("the value of the square of the theta parameter must be finite, but it is "
                                    + std::to_string(theta2) + " instead");
        }
        if (theta < F(0)) {
            throw std::domain_error("the value of the theta parameter must be non-negative, but it is "
                                    + std::to_string(theta) + " instead");
        }
        // In the implementation we need to be able to compute the square of the node size.
        // Check we can do it with the largest node size (i.e., the box size).
        if (!std::isfinite(m_box_size * m_box_size)) {
            throw std::overflow_error("the box size (" + std::to_string(m_box_size)
                                      + ") is too large, and it leads to non-finite values being generated during the "
                                        "computation of the accelerations");
        }
        if constexpr (Ordered) {
            using it_t = decltype(boost::make_permutation_iterator(out[0], m_isort.begin()));
            // Make sure we don't run into overflows when doing a permutated iteration
            // over the iterators in out.
            if (!check_perm_it_range<std::remove_reference_t<decltype(out[0])>>(m_masses.size())) {
                throw std::overflow_error(
                    "the number of particles (" + std::to_string(m_masses.size())
                    + ") is too large, and it results in an overflow condition when computing the accelerations");
            }
            std::array<it_t, NDim> out_pits;
            for (std::size_t j = 0; j != NDim; ++j) {
                out_pits[j] = boost::make_permutation_iterator(out[j], m_isort.begin());
            }
            vec_accs_impl(out_pits, theta2);
        } else {
            vec_accs_impl(out, theta2);
        }
    }
    // Helper overload for an array of vectors. It will prepare the vectors and then
    // call the other overload.
    template <bool Ordered, typename Allocator>
    void accs_dispatch(std::array<std::vector<F, Allocator>, NDim> &out, const F &theta) const
    {
        std::array<F *, NDim> out_ptrs;
        for (std::size_t j = 0; j != NDim; ++j) {
            out[j].resize(boost::numeric_cast<decltype(out[j].size())>(m_masses.size()));
            out_ptrs[j] = out[j].data();
        }
        accs_dispatch<Ordered>(out_ptrs, theta);
    }

public:
    template <typename Allocator>
    void accs_u(std::array<std::vector<F, Allocator>, NDim> &out, const F &theta) const
    {
        accs_dispatch<false>(out, theta);
    }
    template <typename It>
    void accs_u(std::array<It, NDim> &out, const F &theta) const
    {
        accs_dispatch<false>(out, theta);
    }
    template <typename Allocator>
    void accs_o(std::array<std::vector<F, Allocator>, NDim> &out, const F &theta) const
    {
        accs_dispatch<true>(out, theta);
    }
    template <typename It>
    void accs_o(std::array<It, NDim> &out, const F &theta) const
    {
        accs_dispatch<true>(out, theta);
    }

private:
    template <bool Ordered>
    std::array<F, NDim> exact_acc_impl(size_type orig_idx) const
    {
        simple_timer st("exact acc computation");
        const auto size = m_masses.size();
        std::array<F, NDim> retval{};
        const auto idx = Ordered ? m_ord_ind[orig_idx] : orig_idx;
        for (size_type i = 0; i < size; ++i) {
            if (i == idx) {
                continue;
            }
            F dist2(0);
            for (std::size_t j = 0; j < NDim; ++j) {
                dist2 += (m_coords[j][i] - m_coords[j][idx]) * (m_coords[j][i] - m_coords[j][idx]);
            }
            const auto dist = std::sqrt(dist2);
            const auto dist3 = dist * dist2;
            for (std::size_t j = 0; j < NDim; ++j) {
                retval[j] += (m_coords[j][i] - m_coords[j][idx]) * m_masses[i] / dist3;
            }
        }
        return retval;
    }

public:
    std::array<F, NDim> exact_acc_u(size_type idx) const
    {
        return exact_acc_impl<false>(idx);
    }
    std::array<F, NDim> exact_acc_o(size_type idx) const
    {
        return exact_acc_impl<true>(idx);
    }

private:
    template <typename Tr>
    static auto ord_c_ranges_impl(Tr &tr)
    {
        using it_t = decltype(boost::make_permutation_iterator(tr.m_coords[0].begin(), tr.m_ord_ind.begin()));
        std::array<std::pair<it_t, it_t>, NDim> retval;
        for (std::size_t j = 0; j != NDim; ++j) {
            retval[j] = std::make_pair(boost::make_permutation_iterator(tr.m_coords[j].begin(), tr.m_ord_ind.begin()),
                                       boost::make_permutation_iterator(tr.m_coords[j].end(), tr.m_ord_ind.end()));
        }
        return retval;
    }

public:
    auto c_ranges_u() const
    {
        std::array<std::pair<const F *, const F *>, NDim> retval;
        for (std::size_t j = 0; j != NDim; ++j) {
            retval[j] = std::make_pair(m_coords[j].data(), m_coords[j].data() + m_coords[j].size());
        }
        return retval;
    }
    auto c_ranges_o() const
    {
        return ord_c_ranges_impl(*this);
    }
    auto m_range_u() const
    {
        return std::make_pair(m_masses.data(), m_masses.data() + m_masses.size());
    }
    auto m_range_o() const
    {
        return std::make_pair(boost::make_permutation_iterator(m_masses.begin(), m_ord_ind.begin()),
                              boost::make_permutation_iterator(m_masses.end(), m_ord_ind.end()));
    }
    const auto &ord_ind() const
    {
        return m_ord_ind;
    }

private:
    // After updating the particles' positions, this method must be called
    // to reconstruct the other data members according to the new positions.
    void refresh()
    {
        // Let's start with generating the new codes.
        const auto nparts = m_masses.size();
        tbb::parallel_for(tbb::blocked_range<decltype(m_masses.size())>(0u, nparts), [this](const auto &range) {
            std::array<F, NDim> tmp_coord;
            morton_encoder<NDim, UInt> me;
            for (auto i = range.begin(); i != range.end(); ++i) {
                for (std::size_t j = 0; j != NDim; ++j) {
                    tmp_coord[j] = m_coords[j][i];
                }
                m_codes[i] = me(disc_coords(tmp_coord.begin(), m_box_size).begin());
            }
        });
        // Like on construction, do the indirect sorting of the new codes.
        // Use a new temp vector for the new indirect sorting.
        std::vector<size_type, di_aligned_allocator<size_type>> v_ind;
        v_ind.resize(boost::numeric_cast<decltype(v_ind.size())>(nparts));
        // NOTE: this is just a iota.
        tbb::parallel_for(tbb::blocked_range<decltype(m_masses.size())>(0u, nparts), [&v_ind](const auto &range) {
            for (auto i = range.begin(); i != range.end(); ++i) {
                v_ind[i] = i;
            }
        });
        // Do the sorting.
        indirect_code_sort(v_ind.begin(), v_ind.end());
        // Apply the indirect sorting.
        // NOTE: upon tree construction, we already checked that the number of particles does not
        // overflow the limit imposed by apply_isort().
        apply_isort(m_codes, v_ind);
        // Make sure the sort worked as intended.
        assert(std::is_sorted(m_codes.begin(), m_codes.end()));
        for (std::size_t j = 0; j < NDim; ++j) {
            apply_isort(m_coords[j], v_ind);
        }
        apply_isort(m_masses, v_ind);
        // Apply the new indirect sorting to the original one.
        apply_isort(m_isort, v_ind);
        // Establish the indices for ordered iteration (in the original order).
        isort_to_ord_ind();
        // Re-construct the tree.
        m_tree.clear();
        build_tree();
        build_tree_properties();
        // NOTE: we are not adding new particles, we don't need the permutation
        // iterator check that is present in the constructor.
    }
    template <bool Ordered, typename Func>
    void update_positions_impl(Func &&f)
    {
        if constexpr (Ordered) {
            // Create an array of ranges to the coordinates in the original order.
            using it_t = decltype(boost::make_permutation_iterator(m_coords[0].begin(), m_ord_ind.begin()));
            std::array<std::pair<it_t, it_t>, NDim> c_pranges;
            for (std::size_t j = 0; j != NDim; ++j) {
                c_pranges[j] = std::make_pair(boost::make_permutation_iterator(m_coords[j].begin(), m_ord_ind.begin()),
                                              boost::make_permutation_iterator(m_coords[j].end(), m_ord_ind.end()));
            }
            // Feed it to the functor.
            std::forward<Func>(f)(c_pranges);
        } else {
            // Create an array of ranges to the coordinates.
            std::array<std::pair<F *, F *>, NDim> c_ranges;
            for (std::size_t j = 0; j != NDim; ++j) {
                // NOTE: [data(), data() + size) is a valid range also for empty vectors.
                c_ranges[j] = std::make_pair(m_coords[j].data(), m_coords[j].data() + m_coords[j].size());
            }
            // Feed it to the functor.
            std::forward<Func>(f)(c_ranges);
        }
        // Refresh the tree.
        refresh();
    }
    template <bool Ordered, typename Func>
    void update_positions_dispatch(Func &&f)
    {
        simple_timer st("overall update_positions");
        try {
            update_positions_impl<Ordered>(std::forward<Func>(f));
        } catch (...) {
            // Erase everything before re-throwing.
            clear_containers();
            throw;
        }
    }

public:
    template <typename Func>
    void update_positions_u(Func &&f)
    {
        update_positions_dispatch<false>(std::forward<Func>(f));
    }
    template <typename Func>
    void update_positions_o(Func &&f)
    {
        update_positions_dispatch<true>(std::forward<Func>(f));
    }

private:
    // The size of the domain.
    F m_box_size;
    // The maximum number of particles in a leaf node.
    size_type m_max_leaf_n;
    // Number of particles in a critical node: if the number of particles in
    // a node is ncrit or less, then we will compute the accelerations on the
    // particles in that node in a vectorised fashion.
    size_type m_ncrit;
    // The particles' masses.
    fp_vector m_masses;
    // The particles' coordinates.
    std::array<fp_vector, NDim> m_coords;
    // The particles' Morton codes.
    std::vector<UInt, di_aligned_allocator<UInt>> m_codes;
    // The indirect sorting vector. It establishes how to re-order the
    // original particle sequence so that the particles' Morton codes are
    // sorted in ascending order. E.g., if m_isort is [0, 3, 1, 2, ...],
    // then the first particle in Morton order is also the first particle in
    // the original order, the second particle in the Morton order is the
    // particle at index 3 in the original order, and so on.
    std::vector<size_type, di_aligned_allocator<size_type>> m_isort;
    // Indices vector to iterate over the particles' data in the original order.
    // It establishes how to re-order the Morton order to recover the original
    // particle order. This is the dual of m_isort, and it's always possible to
    // compute one given the other. E.g., if m_isort is [0, 3, 1, 2, ...] then
    // m_ord_ind will be [0, 2, 3, 1, ...], meaning that the first particle in
    // the original order is also the first particle in the Morton order, the second
    // particle in the original order is the particle at index 2 in the Morton order,
    // and so on.
    std::vector<size_type, di_aligned_allocator<size_type>> m_ord_ind;
    // The tree structure.
    tree_type m_tree;
    // The list of critical nodes.
    cnode_list_type m_crit_nodes;
};

template <typename UInt, typename F>
using octree = tree<UInt, F, 3>;

} // namespace rakau

#endif
