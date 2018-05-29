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

// These should detect x86 and x86_64 on GCC, clang, MSVC and MinGW, at least. See:
// https://sourceforge.net/p/predef/wiki/Architectures/
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86) || defined(_X86_)

#include <x86intrin.h>

#endif

#include <xsimd/xsimd.hpp>

namespace rakau
{

inline namespace detail
{

// Highest AVX version available.
inline constexpr unsigned avx_version =
#if defined(__AVX512F__)
    3
#elif defined(__AVX2__)
    2
#elif defined(__AVX__)
    1
#else
    0
#endif
    ;

// Highest SSE version available. See:
// https://xsimd.readthedocs.io/en/latest/api/instr_macros.html
// https://stackoverflow.com/questions/18563978/detect-the-availability-of-sse-sse2-instruction-set-in-visual-studio
inline constexpr unsigned sse_version =
#if defined(__SSE4_2__)
    6
#elif defined(__SSE4_1__)
    5
#elif defined(__SSSE3__)
    4
#elif defined(__SSE3__)
    3
#elif (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
    2
#elif (defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1))
    1
#else
    0
#endif
    ;

// Rotate left by 1 the input batch x. That is, if the input
// contains the values [0, 1, 2, 3], the output will contain
// the values [1, 2, 3, 0].
// NOTE: on AVX/SSE this looks like a *right* rotation because
// of the way vector element are indexed in AVX/SSE intrinsics.
template <typename F, std::size_t N>
inline xsimd::batch<F, N> rotate(xsimd::batch<F, N> x)
{
    // NOTE: these need the preprocessor #if in addition to if constexpr
    // because inside we are calling functions which do not depend
    // on any template parameter, and thus name lookup happens before
    // instantiation.
#if defined(__AVX512F__)
    if constexpr (avx_version == 3u && std::is_same_v<F, float> && N == 16u) {
        return _mm512_permutexvar_ps(x, _mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));
    } else if constexpr (avx_version == 3u && std::is_same_v<F, double> && N == 8u) {
        return _mm512_permutexvar_pd(x, _mm512_set_epi64(0, 7, 6, 5, 4, 3, 2, 1));
    } else
#endif
#if defined(__AVX2__)
        if constexpr (avx_version == 2u && std::is_same_v<F, float> && N == 8u) {
        // This instruction is available on AVX2 for floats.
        return _mm256_permutevar8x32_ps(x, _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1));
    } else
#endif
        if constexpr (avx_version == 1u && std::is_same_v<F, float> && N == 8u) {
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
    } else if constexpr (avx_version && avx_version <= 2u && std::is_same_v<F, double> && N == 4u) {
        // Double precision on AVX<=2.
        // NOTE: same idea as above, just different constants.
        // NOTE: x = [x3 x2 x1 x0].
        const auto t0 = _mm256_permute_pd(x, 5);           // [x2 x3 x0 x1]
        const auto t1 = _mm256_permute2f128_pd(t0, t0, 1); // [x0 x1 x2 x3]
        return _mm256_blend_pd(t0, t1, 10);                // [x0 x3 x2 x1]
    } else if constexpr (sse_version >= 2u && std::is_same_v<F, float> && N == 4u) {
        // SSE2, float.
        return _mm_shuffle_ps(x, x, 0b00111001);
    } else if constexpr (sse_version >= 2u && std::is_same_v<F, double> && N == 2u) {
        // SSE2, double.
        return _mm_shuffle_pd(x, x, 1);
    } else {
        // The generic (slow) implementation.
        constexpr auto align = xsimd::simd_batch_traits<xsimd::batch<F, N>>::align;
        alignas(align) F tmp[N];
        x.store_aligned(tmp);
        std::rotate(tmp, tmp + 1, tmp + N);
        return xsimd::batch<F, N>(tmp, xsimd::aligned_mode{});
    }
}

// Small variable template helper to establish if a fast implementation
// of the inverse sqrt for an xsimd batch of type B is available.
template <typename B>
inline constexpr bool has_fast_inv_sqrt = (avx_version == 3u
                                           && std::is_same_v<typename xsimd::simd_batch_traits<B>::value_type,
                                                             float> && xsimd::simd_batch_traits<B>::size == 16u)
                                          || (avx_version
                                              && std::is_same_v<typename xsimd::simd_batch_traits<B>::value_type,
                                                                float> && xsimd::simd_batch_traits<B>::size == 8u);

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
    // On AVX, use the rsqrt intrinsics refined with a Newton iteration.
    if constexpr (avx_version == 3u && std::is_same_v<F, float> && N == 16u) {
        const auto tmp = inv_sqrt_newton_iter(xsimd::batch<F, N>(_mm512_rsqrt14_ps(x)), x);
        return tmp * tmp * tmp;
    } else if constexpr (avx_version && std::is_same_v<F, float> && N == 8u) {
        const auto tmp = inv_sqrt_newton_iter(xsimd::batch<F, N>(_mm256_rsqrt_ps(x)), x);
        return tmp * tmp * tmp;
    } else {
        const auto tmp = F(1) / xsimd::sqrt(x);
        return tmp * tmp * tmp;
    }
}

} // namespace detail

} // namespace rakau

#endif
