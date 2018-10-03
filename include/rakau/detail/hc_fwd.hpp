// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef RAKAU_DETAIL_HC_FWD_HPP
#define RAKAU_DETAIL_HC_FWD_HPP

#include <array>
#include <cstddef>

namespace rakau
{
inline namespace detail
{

template <typename F, std::size_t NDim, typename UInt>
void first_hc_function(std::size_t, std::array<F *, 3> &) __attribute__((visibility("default")));

} // namespace detail
} // namespace rakau

#endif
