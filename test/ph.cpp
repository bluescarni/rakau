// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <rakau/detail/ph.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <tuple>
#include <type_traits>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau_test;

using uint_types = std::tuple<unsigned, unsigned long, unsigned long long>;

template <std::size_t N>
using sc = std::integral_constant<std::size_t, N>;

using ndims = std::tuple<sc<1>, sc<2>, sc<3>>;

static std::mt19937 rng;

static const int ntrials = 10000;

TEST_CASE("ph test")
{
    tuple_for_each(uint_types{}, [](auto x) {
        using uint_t = decltype(x);
        tuple_for_each(ndims{}, [](auto d) {
            constexpr auto ndim = decltype(d)::value;

            std::array<uint_t, ndim> arr{};

            if constexpr (ndim == 1u) {
                std::uniform_int_distribution<uint_t> udist;

                for (int i = 0; i < ntrials; ++i) {
                    arr[0] = udist(rng);
                    REQUIRE(ph_encode<static_cast<std::size_t>(std::numeric_limits<uint_t>::digits)>(arr) == arr[0]);
                }
            } else {
                constexpr auto nbits = static_cast<std::size_t>(std::numeric_limits<uint_t>::digits / ndim);
                std::uniform_int_distribution<uint_t> udist(0, (uint_t(1) << nbits) - 1u);

                for (int i = 0; i < ntrials; ++i) {
                    for (std::size_t j = 0; j < ndim; ++j) {
                        arr[j] = udist(rng);
                    }
                    REQUIRE(
                        (ph_encode<nbits>(arr) || std::all_of(arr.begin(), arr.end(), [](auto n) { return n == 0u; })));
                }
            }
        });
    });
}
