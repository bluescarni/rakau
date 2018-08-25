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
#include <cassert>
#include <cstddef>
#include <iterator>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

namespace rakau_test
{

// Median.
template <typename T>
inline T median(std::vector<T> &v)
{
    assert(v.size());
    std::sort(v.begin(), v.end());
    const auto half_size = v.size() / 2u;
    if (v.size() % 2u) {
        return v[half_size];
    }
    return (v[half_size - 1u] + v[half_size]) / T(2);
}

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

// Silence spurious GCC warning in tuple_for_each().
#if !defined(__clang__) && defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnoexcept"

#endif

// Tuple for_each(). It will apply the input functor f to each element of
// the input tuple tup, sequentially.
template <typename Tuple, typename F>
inline void tuple_for_each(Tuple &&tup, F &&f)
{
    std::apply(
        [&f](auto &&... items) {
            // NOTE: here we are converting to void the results of the invocations
            // of f. This ensures that we are folding using the builtin comma
            // operator, which implies sequencing:
            // """
            //  Every value computation and side effect of the first (left) argument of the built-in comma operator is
            //  sequenced before every value computation and side effect of the second (right) argument.
            // """"
            // NOTE: we are writing this as a right fold, i.e., it will expand as:
            //
            // f(tup[0]), (f(tup[1]), (f(tup[2])...
            //
            // A left fold would also work guaranteeing the same sequencing.
            (void(std::forward<F>(f)(std::forward<decltype(items)>(items))), ...);
        },
        std::forward<Tuple>(tup));
}

#if !defined(__clang__) && defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

} // namespace rakau_test

#endif
