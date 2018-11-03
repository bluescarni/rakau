// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cstdint>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <rakau/detail/ph.hpp>
#include <rakau/tree.hpp>

using namespace rakau;

int main(int argc, char **argv)
{
    constexpr auto ntot = 100l;

    std::array<std::uint64_t, 3> tmp{};

    std::vector<std::uint64_t> retval(boost::numeric_cast<std::vector<std::uint64_t>::size_type>(ntot * ntot * ntot));

    simple_timer st("encoding");

    auto counter = 0ul;
    for (auto i = 0l; i < ntot; ++i) {
        tmp[0] = static_cast<std::uint64_t>((1l << 20) / ntot * i);
        for (auto j = 0l; j < ntot; ++j) {
            tmp[1] = static_cast<std::uint64_t>((1l << 20) / ntot * j);
            for (auto k = 0l; k < ntot; ++k) {
                tmp[2] = static_cast<std::uint64_t>((1l << 20) / ntot * k);
                retval[counter] = ph_encode<21>(tmp);
                ++counter;
            }
        }
    }
}
