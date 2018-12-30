// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef RAKAU_DETAIL_TREE_FWD_HPP
#define RAKAU_DETAIL_TREE_FWD_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <limits>
#include <tuple>
#include <type_traits>
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

#include <rakau/detail/di_aligned_allocator.hpp>

namespace rakau
{

// Multipole acceptance criteria.
enum class mac { bh, bh_geom };

inline namespace detail
{

// Dependent false for static_assert in if constexpr.
// http://en.cppreference.com/w/cpp/language/if#Constexpr_If
template <typename T>
struct dependent_false : std::false_type {
};

template <typename T>
inline constexpr bool dependent_false_v = dependent_false<T>::value;

// Size type for the tree class.
// NOTE: strictly speaking, the allocator we use in the tree may have a different
// alignment than zero. However, I don't think there's any way the size type
// of std::vector can depend on the actual alignment value.
template <typename F>
using tree_size_t = typename std::vector<F, di_aligned_allocator<F, 0>>::size_type;

// Tree node structure.
// NOTE: the default implementation is empty and it will error
// out if used.
template <std::size_t NDim, typename F, typename UInt, mac MAC>
struct tree_node_t {
    static_assert(dependent_false_v<F>,
                  "The specialisation of the tree node type for this set of parameters is not available.");
};

// Tree node data common to all node implementations.
template <std::size_t NDim, typename F, typename UInt>
struct base_tree_node_t {
    // Node begin/end (in the arrays of particle data) and number of children.
    tree_size_t<F> begin, end, n_children;
    // Node code and level.
    UInt code, level;
};

// Tree node for the BH MAC.
template <std::size_t NDim, typename F, typename UInt>
struct tree_node_t<NDim, F, UInt, mac::bh> : base_tree_node_t<NDim, F, UInt> {
    // Node properties (COM coordinates + mass) and square of the node dimension.
    F props[NDim + 1u], dim2;
};

// Critical node.
template <typename F, typename UInt>
using tree_cnode_t = std::tuple<UInt, tree_size_t<F>, tree_size_t<F>>;

// Computation of the number of vectors needed to store the result
// of an acceleration/potential computation.
template <unsigned Q, std::size_t NDim>
constexpr std::size_t compute_tree_nvecs_res()
{
    static_assert(Q <= 2u);
    return static_cast<std::size_t>(Q == 0u ? NDim : (Q == 1u ? 1u : NDim + 1u));
}

template <unsigned Q, std::size_t NDim>
inline constexpr std::size_t tree_nvecs_res = compute_tree_nvecs_res<Q, NDim>();

// Computation of the cbits value (number of bits to use
// in the morton encoding for each coordinate).
template <typename UInt, std::size_t NDim>
constexpr auto compute_cbits_v()
{
    constexpr unsigned nbits = std::numeric_limits<UInt>::digits;
    static_assert(nbits > NDim, "The number of bits must be greater than the number of dimensions.");
    return static_cast<UInt>(nbits / NDim - !(nbits % NDim));
}

template <typename UInt, std::size_t NDim>
inline constexpr auto cbits_v = compute_cbits_v<UInt, NDim>();

// clz wrapper. n must be a nonzero unsigned integral.
template <typename UInt>
inline unsigned clz(UInt n)
#if defined(__HCC_ACCELERATOR__)
    [[hc]]
#endif
{
    static_assert(std::is_integral<UInt>::value && std::is_unsigned<UInt>::value);
    assert(n);
#if defined(__clang__) || defined(__GNUC__)
    // Implementation using GCC's and clang's builtins.
    if (std::is_same<UInt, unsigned>::value) {
        return static_cast<unsigned>(__builtin_clz(n));
    } else if (std::is_same<UInt, unsigned long>::value) {
        return static_cast<unsigned>(__builtin_clzl(n));
    } else if (std::is_same<UInt, unsigned long long>::value) {
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
inline UInt tree_level(UInt n)
#if defined(__HCC_ACCELERATOR__)
    [[hc]]
#endif
{
#if !defined(NDEBUG)
    constexpr auto cbits = cbits_v<UInt, NDim>;
#endif
    constexpr auto ndigits = static_cast<unsigned>(std::numeric_limits<UInt>::digits);
    assert(n);
    assert(!((ndigits - 1u - clz(n)) % NDim));
    auto retval = static_cast<UInt>((ndigits - 1u - clz(n)) / NDim);
    assert(cbits >= retval);
    assert((cbits - retval) * NDim < ndigits);
    return retval;
}

} // namespace detail
} // namespace rakau

#endif
