// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef RAKAU_DETAIL_SIMD_HPP
#define RAKAU_DETAIL_SIMD_HPP

#include <array>
#include <atomic>
#include <cstddef>
#include <type_traits>

#include <xsimd/xsimd.hpp>

#if defined(XSIMD_X86_INSTR_SET)

#include <x86intrin.h>

#endif

namespace rakau
{

inline namespace detail
{

// Small helper to get the scalar type from a batch type B.
template <typename B>
using xsimd_scalar_t = typename xsimd::revert_simd_traits<B>::type;

// Global atomic counters for certain simd operations.
// NOTE: ATOMIC_VAR_INIT can be used with variables with static storage duration,
// and inline variables do have static storage duration.
inline std::atomic<unsigned long long> simd_fma_counter = ATOMIC_VAR_INIT(0);
inline std::atomic<unsigned long long> simd_sqrt_counter = ATOMIC_VAR_INIT(0);
inline std::atomic<unsigned long long> simd_rsqrt_counter = ATOMIC_VAR_INIT(0);

// The corresponding thread-local counters.
inline thread_local unsigned long long simd_fma_counter_tl = 0;
inline thread_local unsigned long long simd_sqrt_counter_tl = 0;
inline thread_local unsigned long long simd_rsqrt_counter_tl = 0;

// Wrappers around some xsimd function. They will increase the corresponding
// thread-local counters if compiled with RAKAU_WITH_SIMD_COUNTERS.
template <typename B>
inline auto xsimd_fma(B x, B y, B z)
{
#if defined(RAKAU_WITH_SIMD_COUNTERS)
    ++simd_fma_counter_tl;
#endif
    return xsimd::fma(x, y, z);
}

template <typename B>
inline auto xsimd_fnma(B x, B y, B z)
{
#if defined(RAKAU_WITH_SIMD_COUNTERS)
    ++simd_fma_counter_tl;
#endif
    return xsimd::fnma(x, y, z);
}

template <typename B>
inline auto xsimd_sqrt(B x)
{
#if defined(RAKAU_WITH_SIMD_COUNTERS)
    ++simd_sqrt_counter_tl;
#endif
    return xsimd::sqrt(x);
}

// Newton iteration step for the computation of 1/sqrt(x) with starting point y0:
// y1 = y0/2 * (3 - x*y0**2).
template <typename F, std::size_t N>
inline xsimd::batch<F, N> inv_sqrt_newton_iter(xsimd::batch<F, N> y0, xsimd::batch<F, N> x)
{
    const xsimd::batch<F, N> three(F(3));
    const xsimd::batch<F, N> xy0 = x * y0;
    const xsimd::batch<F, N> half_y0 = y0 * (F(1) / F(2));
    const xsimd::batch<F, N> three_minus_muls = xsimd_fnma(xy0, y0, three);
    return half_y0 * three_minus_muls;
}

// Computation of 1/sqrt(x)**3 via fast rsqrt.
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512_VERSION

inline xsimd::batch<float, 16> inv_sqrt(xsimd::batch<float, 16> x)
{
    // NOTE: AVX512-ER has an intrinsic for 28-bit precision (rather than 14-bit)
    // rsqrt, but it does not seem to be widely available yet.
#if defined(RAKAU_WITH_SIMD_COUNTERS)
    // Increase the corresponding TL counter, if requested.
    // NOTE: these inv_sqrt functions are the only places where
    // we use the rsqrt intrinsics.
    ++simd_rsqrt_counter_tl;
#endif
    return inv_sqrt_newton_iter(xsimd::batch<float, 16>(_mm512_rsqrt14_ps(x)), x);
}

inline xsimd::batch<float, 16> inv_sqrt_3(xsimd::batch<float, 16> x)
{
    const auto tmp = inv_sqrt(x);
    return tmp * tmp * tmp;
}

#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION

inline xsimd::batch<float, 8> inv_sqrt(xsimd::batch<float, 8> x)
{
#if defined(RAKAU_WITH_SIMD_COUNTERS)
    ++simd_rsqrt_counter_tl;
#endif
    return inv_sqrt_newton_iter(xsimd::batch<float, 8>(_mm256_rsqrt_ps(x)), x);
}

inline xsimd::batch<float, 8> inv_sqrt_3(xsimd::batch<float, 8> x)
{
    const auto tmp = inv_sqrt(x);
    return tmp * tmp * tmp;
}

#endif

// Minimal traits class for xsimd batches.
template <typename T>
struct simd_traits {
};

template <typename T, std::size_t N>
struct simd_traits<xsimd::batch<T, N>> {
    using scalar_type = T;
    static constexpr auto size = N;
};

// Machinery for the implementation of the functions below. This
// is a way of using index_sequence with variadic lambdas. See:
// http://aherrmann.github.io/programming/2016/02/28/unpacking-tuples-in-cpp14/
template <typename F, std::size_t... I>
inline auto index_apply_impl(const F &f, const std::index_sequence<I...> &)
{
    return f(std::integral_constant<std::size_t, I>{}...);
}

template <std::size_t N, typename F>
inline auto index_apply(const F &f)
{
    return index_apply_impl(f, std::make_index_sequence<N>{});
}

// Create an array of batches of type B containing all zeroes.
template <typename B, std::size_t N>
inline auto batches_zero()
{
    return index_apply<N>([](auto... I) {
        using scalar_t = typename simd_traits<B>::scalar_type;
        return std::array{(void(I), B(scalar_t(0)))...};
    });
}

// Create an array of batches loading data from the input pointers. The batch type
// will be B, its scalar type T. Aligned specifies whether the pointers point to
// aligned memory or not. An optional offset will be added to the pointers.
template <bool Aligned, typename B, typename T, std::size_t N>
inline auto batches_load(const std::array<T *, N> &ptrs, std::size_t offset = 0)
{
    // Double check that the scalar type of B is T or const T.
    static_assert(std::is_same_v<std::remove_const_t<T>, typename simd_traits<B>::scalar_type>);
    return index_apply<N>([&ptrs, offset](auto... I) {
        if constexpr (Aligned) {
            return std::array{B(std::get<I>(ptrs) + offset, xsimd::aligned_mode{})...};
        } else {
            return std::array{B(std::get<I>(ptrs) + offset, xsimd::unaligned_mode{})...};
        }
    });
}

// Store the data in the input batches at the addresses specified by ptrs.
// Aligned specifies whether the pointers point to aligned memory or not. An optional offset will
// be added to the pointers.
template <bool Aligned, typename B, typename T, std::size_t N>
inline void batches_store(const std::array<B, N> &batches, const std::array<T *, N> &ptrs, std::size_t offset = 0)
{
    index_apply<N>([&batches, &ptrs, offset](auto... I) {
        if constexpr (Aligned) {
            ((std::get<I>(batches).store_aligned(std::get<I>(ptrs) + offset)), ...);
        } else {
            ((std::get<I>(batches).store_unaligned(std::get<I>(ptrs) + offset)), ...);
        }
    });
}

// Compute the square of the softened l2 norm of a.
template <typename B, std::size_t N>
inline auto batches_softened_norm2(const std::array<B, N> &a, const B &eps2)
{
    // NOTE: we will skip iterating over the last element of the array.
    // The last element will instead be used for init in conjunction with
    // an FMA operation.
    static_assert(N > 0u);
    return index_apply<N - 1u>([&a, &eps2](auto... I) {
        return ((std::get<I>(a) * std::get<I>(a)) + ... + (xsimd_fma(std::get<N - 1u>(a), std::get<N - 1u>(a), eps2)));
    });
}

// Small variable template helper to establish if a fast implementation
// of the inverse sqrt for an xsimd batch of type B is available.
// Currently, this is true for:
// - AVX 8-floats batches,
// - AVX512 16-floats batches.
// NOTE: there are intrinsics in SSE for rsqrt as well, but they don't seem to
// improve performance for our use case. I could not understand why exactly that's
// the case.
template <typename B>
inline constexpr bool has_fast_inv_sqrt =
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
    (std::is_same_v<xsimd_scalar_t<B>, float> && B::size == 8u)
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512_VERSION
    || (std::is_same_v<xsimd_scalar_t<B>, float> && B::size == 16u)
#endif
#else
    false
#endif
    ;

// Small helper to establish if simd is available for the type F.
template <typename F, typename = void>
struct has_simd_impl : std::false_type {
};

template <typename F>
struct has_simd_impl<F, std::enable_if_t<(xsimd::simd_type<F>::size > 1u)>> : std::true_type {
};

template <typename F>
inline constexpr bool has_simd = has_simd_impl<F>::value;

} // namespace detail

} // namespace rakau

#endif
