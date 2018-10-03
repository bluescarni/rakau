#include <array>
#include <cstddef>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wdeprecated-dynamic-exception-spec"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"

#include <hc.hpp>

#pragma GCC diagnostic pop

#include <rakau/detail/hc_fwd.hpp>

#include <vector>

#include <iostream>

namespace rakau
{
inline namespace detail
{

template <typename F, std::size_t NDim, typename UInt>
void first_hc_function(std::size_t tgt_size, std::array<F *, 3> &res_ptrs)
{
    hc::array_view<F, 1> x_acc(tgt_size, res_ptrs[0]);
    hc::array_view<F, 1> y_acc(tgt_size, res_ptrs[1]);
    hc::array_view<F, 1> z_acc(tgt_size, res_ptrs[2]);
    hc::parallel_for_each(hc::extent<1>(1), [x_acc, y_acc, z_acc](hc::index<1> i) [[hc]] {
        x_acc[i] += 1;
        y_acc[i] += 1;
        z_acc[i] += 1;
    });
    // x_acc.synchronize();
    // y_acc.synchronize();
    // z_acc.synchronize();
}

// Explicit instantiations.
template void first_hc_function<float, 3, std::size_t>(std::size_t, std::array<float *, 3> &);
template void first_hc_function<double, 3, std::size_t>(std::size_t, std::array<double *, 3> &);

} // namespace detail
} // namespace rakau
