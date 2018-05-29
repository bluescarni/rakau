// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef RAKAU_DETAIL_SIMD_HPP
#define RAKAU_DETAIL_SIMD_HPP

#include <algorithm>
#include <type_traits>

#include <xsimd/xsimd.hpp>

#if defined(XSIMD_X86_INSTR_SET)

#include <x86intrin.h>

#endif

namespace rakau
{

inline namespace detail
{

// Rotate left by 1 the input batch x. That is, if the input
// contains the values [0, 1, 2, 3], the output will contain
// the values [1, 2, 3, 0].
template <typename F, std::size_t N>
inline xsimd::batch<F, N> rotate(xsimd::batch<F, N> x)
{
    // The generic (slow) implementation.
    constexpr auto align = xsimd::simd_batch_traits<xsimd::batch<F, N>>::align;
    alignas(align) F tmp[N];
    x.store_aligned(tmp);
    std::rotate(tmp, tmp + 1, tmp + N);
    return xsimd::batch<F, N>(tmp, xsimd::aligned_mode{});
}

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512F_VERSION

// NOTE: on AVX/SSE this looks like a *right* rotation because
// of the way vector element are indexed in AVX/SSE intrinsics.

// AVX512, float.
template <>
inline xsimd::batch<float, 16> rotate(xsimd::batch<float, 16> x)
{
    return _mm512_permutexvar_ps(_mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1), x);
}

#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX2_VERSION

// AVX2 for floats.
template <>
inline xsimd::batch<float, 8> rotate(xsimd::batch<float, 8> x)
{
    // This instruction is available only on AVX2, and only for floats.
    return _mm256_permutevar8x32_ps(x, _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1));
}

#elif XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION

// AVX1 for floats.
template <>
inline xsimd::batch<float, 8> rotate(xsimd::batch<float, 8> x)
{
    // If we have only AVX1, we use a mix of permute and blend for floats.
    //
    // See the second answer here:
    // https://stackoverflow.com/questions/19516585/shifting-sse-avx-registers-32-bits-left-and-right-while-shifting-in-zeros
    //
    // This is slightly modified in the second line as we don't want the zeroing
    // feature provided by _mm256_permute2f128_ps.
    //
    // NOTE: 0x39 == 0011 1001, 0x88 == 1000 1000.
    // NOTE: x = [x7 x6 ... x0].
    const auto t0 = _mm256_permute_ps(x, 0x39);        // [x4  x7  x6  x5  x0  x3  x2  x1]
    const auto t1 = _mm256_permute2f128_ps(t0, t0, 1); // [x0  x3  x2  x1  x4  x7  x6  x5]
    return _mm256_blend_ps(t0, t1, 0x88);              // [x0  x7  x6  x5  x4  x3  x2  x1]
}

#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION

// AVX1/2, double.
template <>
inline xsimd::batch<double, 4> rotate(xsimd::batch<double, 4> x)
{
    // NOTE: same idea as above, just different constants.
    // NOTE: x = [x3 x2 x1 x0].
    const auto t0 = _mm256_permute_pd(x, 5);           // [x2 x3 x0 x1]
    const auto t1 = _mm256_permute2f128_pd(t0, t0, 1); // [x0 x1 x2 x3]
    return _mm256_blend_pd(t0, t1, 10);                // [x0 x3 x2 x1]
}

#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_SSE2_VERSION

// SSE2, float.
template <>
inline xsimd::batch<float, 4> rotate(xsimd::batch<float, 4> x)
{
    return _mm_shuffle_ps(x, x, 0b00111001);
}

// SSE2, double.
template <>
inline xsimd::batch<double, 2> rotate(xsimd::batch<double, 2> x)
{
    return _mm_shuffle_pd(x, x, 1);
}

#endif

// Small variable template helper to establish if a fast implementation
// of the inverse sqrt for an xsimd batch of type B is available.
// Currently, this is true for:
// - AVX 8-floats batches,
// - AVX512 16-floats batches.
// NOTE: there are intrinsics in SSE for rsqrt as well, but they don't seem to
// improve performance for our use case.
template <typename B>
inline constexpr bool has_fast_inv_sqrt =
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION
    (std::is_same_v<typename xsimd::simd_batch_traits<B>::value_type, float> && xsimd::simd_batch_traits<B>::size == 8u)
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512F_VERSION
    || (std::is_same_v<typename xsimd::simd_batch_traits<B>::value_type,
                       float> && xsimd::simd_batch_traits<B>::size == 16u)
#endif
#else
    false
#endif
    ;

// Newton iteration step for the computation of 1/sqrt(x) with starting point y0:
// y1 = y0/2 * (3 - x*y0**2).
template <typename F, std::size_t N>
inline xsimd::batch<F, N> inv_sqrt_newton_iter(xsimd::batch<F, N> y0, xsimd::batch<F, N> x)
{
    const xsimd::batch<F, N> three(F(3));
    const xsimd::batch<F, N> xy0 = x * y0;
    const xsimd::batch<F, N> half_y0 = y0 * (F(1) / F(2));
    const xsimd::batch<F, N> three_minus_muls = xsimd::fnma(xy0, y0, three);
    return half_y0 * three_minus_muls;
}

// Compute 1/sqrt(x)**3. This will use fast rsqrt if available.
template <typename F, std::size_t N>
inline xsimd::batch<F, N> inv_sqrt_3(xsimd::batch<F, N> x)
{
    // The default implementation.
    const auto tmp = F(1) / xsimd::sqrt(x);
    return tmp * tmp * tmp;
}

// On various versions of AVX, we have intrinsics for fast rsqrt.
#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512F_VERSION

template <>
inline xsimd::batch<float, 16> inv_sqrt_3(xsimd::batch<float, 16> x)
{
    // NOTE: AVX512-ER has an intrinsic for 28-bit precision (rather than 14-bit)
    // rsqrt, but it does not seem to be widely available yet.
    const auto tmp = inv_sqrt_newton_iter(xsimd::batch<float, 16>(_mm512_rsqrt14_ps(x)), x);
    return tmp * tmp * tmp;
}

#endif

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX_VERSION

template <>
inline xsimd::batch<float, 8> inv_sqrt_3(xsimd::batch<float, 8> x)
{
    const auto tmp = inv_sqrt_newton_iter(xsimd::batch<float, 8>(_mm256_rsqrt_ps(x)), x);
    return tmp * tmp * tmp;
}

#endif

// Return true if at least 1 element of a is greater-than or equal to
// the corresponding element of b, false otherwise.
template <typename F, std::size_t N>
inline bool any_geq(xsimd::batch<F, N> a, xsimd::batch<F, N> b)
{
    return xsimd::any(a >= b);
}

#if XSIMD_X86_INSTR_SET >= XSIMD_X86_AVX512F_VERSION

// We can optimise this in AVX512.
template <>
inline bool any_geq(xsimd::batch<float, 16> a, xsimd::batch<float, 16> b)
{
    return _mm512_cmp_ps_mask(a, b, _CMP_GE_OQ);
}

#endif

} // namespace detail

} // namespace rakau

#endif
