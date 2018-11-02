// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef RAKAU_DETAIL_PH_HPP
#define RAKAU_DETAIL_PH_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__BMI2__)

#include <x86intrin.h>

#endif

namespace rakau
{

inline namespace detail
{

template <typename UInt>
inline UInt ph_pdep(UInt src, UInt mask)
{
    UInt result = 0, mask_bit = 1, src_bit = 1;

    for (int i = 0; i < std::numeric_limits<UInt>::digits; ++i, mask_bit <<= 1) {
        if (mask & mask_bit) {
            if (src & src_bit) {
                result |= mask_bit;
            }
            src_bit <<= 1;
        }
    }

    return result;
}

#if defined(__BMI2__)

template <>
inline std::uint32_t ph_pdep<std::uint32_t>(std::uint32_t src, std::uint32_t mask)
{
    return _pdep_u32(src, mask);
}

template <>
inline std::uint64_t ph_pdep<std::uint64_t>(std::uint64_t src, std::uint64_t mask)
{
    return _pdep_u64(src, mask);
}

#endif

template <std::size_t NBits, typename UInt, std::size_t NDim>
inline UInt ph_interleave_mask()
{
    UInt retval = 1;
    for (std::size_t i = 1; i < NBits; ++i) {
        retval += static_cast<UInt>(UInt(1) << (NDim * i));
    }
    return retval;
}

template <std::size_t NBits, typename UInt, std::size_t NDim, typename T>
inline UInt ph_interleave(const T &in)
{
    // NOTE: the interleaving must be done in inverse order,
    // from the end of the input array to the beginning.
    auto mask = ph_interleave_mask<NBits, UInt, NDim>();
    auto retval = ph_pdep(in[NDim - 1u], mask);
    for (std::size_t i = 1; i < NDim; ++i) {
        retval |= ph_pdep(in[NDim - 1u - i], mask <<= 1);
    }
    return retval;
}

template <std::size_t NBits, typename UInt, std::size_t NDim, typename T>
inline UInt ph_encode_impl(const T &in)
{
    static_assert(NBits > 0u, "The number of bits must be positive.");
    static_assert(NDim > 0u, "The number of dimensions must be positive.");
    static_assert(NBits <= std::numeric_limits<std::size_t>::max() / NDim, "Overflow error.");
    static_assert(NBits * NDim <= std::numeric_limits<UInt>::digits, "Too many bits and/or dimensions.");
    static_assert(std::is_integral<UInt>::value && std::is_unsigned<UInt>::value,
                  "Only unsigned integrals are supported.");

#if !defined(NDEBUG)
    for (std::size_t i = 0; i < NDim; ++i) {
        if constexpr (NBits < std::numeric_limits<UInt>::digits) {
            assert(in[i] <= ((UInt(1) << NBits) - 1u));
        }
    }
#endif

    // Make a copy of the input array.
    // NOTE: init and then copy to accommodate the case in which T is an array.
    T arr;
    for (std::size_t i = 0; i < NDim; ++i) {
        arr[i] = in[i];
    }

    constexpr auto M = static_cast<UInt>(UInt(1) << (NBits - 1u));

    for (auto Q = M; Q > 1u; Q >>= 1) {
        const auto P = static_cast<UInt>(Q - 1u);
        for (std::size_t i = 0; i < NDim; ++i) {
            if (arr[i] & Q) {
                arr[0] ^= P;
            } else {
                const auto t = static_cast<UInt>((arr[0] ^ arr[i]) & P);
                arr[0] ^= t;
                arr[i] ^= t;
            }
        }
    }

    for (std::size_t i = 1; i < NDim; ++i) {
        arr[i] ^= arr[i - 1u];
    }

    UInt t = 0;
    for (auto Q = M; Q > 1u; Q >>= 1) {
        if (arr[NDim - 1u] & Q) {
            t ^= Q - 1u;
        }
    }

    for (std::size_t i = 0; i < NDim; ++i) {
        arr[i] ^= t;
    }

    return ph_interleave<NBits, UInt, NDim>(arr);
}

template <std::size_t NBits, typename UInt, std::size_t NDim>
inline UInt ph_encode(const UInt (&in)[NDim])
{
    return ph_encode_impl<NBits, UInt, NDim>(in);
}

template <std::size_t NBits, typename UInt, std::size_t NDim>
inline UInt ph_encode(const std::array<UInt, NDim> &in)
{
    return ph_encode_impl<NBits, UInt, NDim>(in);
}

} // namespace detail

} // namespace rakau

#endif
