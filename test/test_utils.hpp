// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef RAKAU_TEST_UTILS_HPP
#define RAKAU_TEST_UTILS_HPP

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <random>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

namespace rakau_test
{

// Generate n uniformly-distributed particles in a D-dimensional box of given size, using
// the random number engine rng.
template <std::size_t D, typename F, typename Rng>
inline std::vector<F> get_uniform_particles(std::size_t n, F size, Rng &rng)
{
    std::vector<F> retval(n * (D + 1u));
    // Mass.
    std::uniform_real_distribution<F> mdist(F(0), F(1));
    std::generate(
        retval.begin(),
        retval.begin()
            + boost::numeric_cast<typename std::iterator_traits<decltype(retval.begin())>::difference_type>(n),
        [&mdist, &rng]() { return mdist(rng); });
    // Positions.
    std::uniform_real_distribution<F> rdist(-size / F(2), size / F(2));
    std::generate(
        retval.begin()
            + boost::numeric_cast<typename std::iterator_traits<decltype(retval.begin())>::difference_type>(n),
        retval.end(), [&rdist, &rng]() { return rdist(rng); });
    return retval;
}

} // namespace rakau_test

#endif
