// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the mp++ library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <rakau/detail/simd.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <numeric>
#include <type_traits>

#include <xsimd/xsimd.hpp>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau_test;

using fp_types = std::tuple<float, double>;

TEST_CASE("rotate")
{
    tuple_for_each(fp_types{}, [](auto x) {
        using fp_t = decltype(x);
        using b_type = xsimd::simd_type<fp_t>;
        constexpr auto align = xsimd::simd_batch_traits<b_type>::align;
        constexpr auto b_size = b_type::size;
        alignas(align) fp_t tmp[b_size];
        std::iota(tmp, tmp + b_size, fp_t(0));
        b_type b(tmp, xsimd::aligned_mode{});
        rotate(b).store_aligned(tmp);
        REQUIRE(tmp[b_size - 1u] == fp_t(0));
        for (std::remove_const_t<decltype(b_type::size)> i = 0; i < b_size - 1u; ++i) {
            REQUIRE(tmp[i] == fp_t(i) + fp_t(1));
        }
    });
}
