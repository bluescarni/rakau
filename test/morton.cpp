// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <rakau/tree.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <type_traits>

#include "test_utils.hpp"

using namespace rakau;
using namespace rakau_test;

using uint_types = std::tuple<std::uint32_t, std::uint64_t>;
using dims = std::tuple<std::integral_constant<std::size_t, 2>, std::integral_constant<std::size_t, 3>>;

static std::mt19937 rng;

static constexpr int ntrials = 10000;

// Encode a bunch of randomly-generated arrays, and verify
// that decoding the result gives back the original array.
TEST_CASE("morton")
{
    tuple_for_each(uint_types{}, [](auto u) {
        tuple_for_each(dims{}, [u](auto dim) {
            using uint_t = decltype(u);

            constexpr auto d = dim();
            constexpr auto cbits = cbits_v<uint_t, d>;

            morton_encoder<d, uint_t> me;
            morton_decoder<d, uint_t> md;

            std::uniform_int_distribution<uint_t> udist(0, (uint_t(1) << cbits) - 1u);

            uint_t buffer1[d], buffer2[d];
            for (auto i = 0; i < ntrials; ++i) {
                std::generate_n(buffer1, d, [&udist]() { return udist(rng); });
                const auto code = me(&buffer1[0]);
                md(&buffer2[0], code);
                REQUIRE(std::equal(&buffer1[0], &buffer1[0] + d, &buffer2[0]));
            }
        });
    });
}
