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
#include <initializer_list>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

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
    std::generate(retval.begin(), retval.begin() + n, [&mdist, &rng]() { return mdist(rng); });
    // Positions.
    std::uniform_real_distribution<F> rdist(-size / F(2), size / F(2));
    std::generate(retval.begin() + n, retval.end(), [&rdist, &rng]() { return rdist(rng); });
    return retval;
}

inline namespace impl
{

template <typename T, typename F, std::size_t... Is>
inline void apply_to_each_item(T &&t, const F &f, std::index_sequence<Is...>)
{
    (void)std::initializer_list<int>{0, (void(f(std::get<Is>(std::forward<T>(t)))), 0)...};
}
} // namespace impl

// Tuple for_each(). Execute the functor f on each element of the input Tuple.
// https://isocpp.org/blog/2015/01/for-each-arg-eric-niebler
// https://www.reddit.com/r/cpp/comments/2tffv3/for_each_argumentsean_parent/
// https://www.reddit.com/r/cpp/comments/33b06v/for_each_in_tuple/
template <class Tuple, class F>
inline void tuple_for_each(Tuple &&t, const F &f)
{
    apply_to_each_item(std::forward<Tuple>(t), f, std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>{});
}

} // namespace rakau_test

#endif
