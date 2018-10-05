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

#include <rakau/detail/tree_fwd.hpp>

namespace rakau
{
inline namespace detail
{

template <unsigned Q, std::size_t NDim, typename F, typename UInt>
void acc_pot_impl_hcc(const std::array<F *, tree_nvecs_res<Q, NDim>> &, const tree_cnode_t<F, UInt> *, tree_size_t<F>,
                      const tree_node_t<NDim, F, UInt> *, tree_size_t<F>, const std::array<const F *, NDim + 1u> &,
                      tree_size_t<F>, F, F, F) __attribute__((visibility("default")));

} // namespace detail
} // namespace rakau

#endif
