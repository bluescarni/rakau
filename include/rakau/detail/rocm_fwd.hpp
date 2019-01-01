// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef RAKAU_DETAIL_ROCM_FWD_HPP
#define RAKAU_DETAIL_ROCM_FWD_HPP

#include <array>
#include <cstddef>

#include <rakau/detail/tree_fwd.hpp>

namespace rakau
{
inline namespace detail
{

unsigned rocm_min_size() __attribute__((visibility("default")));

bool rocm_has_accelerator() __attribute__((visibility("default")));

template <std::size_t NDim, typename F, typename UInt, mac MAC>
class __attribute__((visibility("default"))) rocm_state
{
public:
    explicit rocm_state(const std::array<const F *, NDim + 1u> &, const UInt *, int,
                        const tree_node_t<NDim, F, UInt, MAC> *, int);

    // NOTE: make sure we don't end up accidentally copying/moving
    // objects of this class.
    rocm_state(const rocm_state &) = delete;
    rocm_state(rocm_state &&) = delete;
    rocm_state &operator=(const rocm_state &) = delete;
    rocm_state &operator=(rocm_state &&) = delete;
    ~rocm_state();

    template <unsigned Q>
    void acc_pot(int, int, const std::array<F *, tree_nvecs_res<Q, NDim>> &, F, F, F) const;

private:
    void *m_state;
};

} // namespace detail
} // namespace rakau

#endif
