#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>

#include <boost/numeric/conversion/cast.hpp>

#include <rakau/detail/hc_fwd.hpp>
#include <rakau/detail/tree_fwd.hpp>

namespace rakau
{

inline namespace detail
{

template <typename T>
auto make_scoped_cu_array(std::size_t n)
{
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::bad_alloc();
    }
    T *ret;
    auto res = ::cudaMalloc(&ret, n * sizeof(T));
    if (res != ::cudaSuccess) {
        throw std::bad_alloc();
    }
    return std::unique_ptr<T, decltype(::cudaFree) *>(ret, ::cudaFree);
}

template <typename T>
class scoped_cu_array
{
    using ptr_t = decltype(make_scoped_cu_array<T>(0));

public:
    scoped_cu_array() : m_ptr(nullptr, ::cudaFree) {}
    explicit scoped_cu_array(std::size_t n) : m_ptr(make_scoped_cu_array<T>(n)) {}
    auto get() const
    {
        return m_ptr.get();
    }

private:
    ptr_t m_ptr;
};

template <unsigned Q, std::size_t NDim, typename F, typename UInt>
__global__ void acc_pot_kernel(F *__restrict__ res_x, F *__restrict__ res_y, F *__restrict__ res_z,
                               const F *__restrict__ ptr_x, const F *__restrict__ ptr_y, const F *__restrict__ ptr_z,
                               const F *__restrict__ ptr_mass, const UInt *__restrict__ codes, const int nparts,
                               const tree_node_t<NDim, F, UInt> *__restrict__ tree_ptr, const int tree_size,
                               const F theta2, const F G, const F eps2)
{
    int pidx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pidx >= nparts) {
        // Don't do anything if we are in the remainder
        // of the last tile.
        return;
    }

    // Results, inited to zero.
    F r_x = 0, r_y = 0, r_z = 0;

    // Load the particle code, position and mass.
    const auto code = codes[pidx];
    const auto x = ptr_x[pidx], y = ptr_y[pidx], z = ptr_z[pidx], mass = ptr_mass[pidx];

    // Add a 1 bit just above the highest possible bit position for the particle code.
    // This value is used in the loop, we precompute it here.
    const auto s_p_code_init = static_cast<UInt>(code | (UInt(1) << (cbits_v<UInt, NDim> * NDim)));

    // Loop over the tree.
    for (auto src_idx = 0; src_idx < tree_size;) {
        // Get a reference to the current source node, and cache locally a few quantities.
        const auto &src_node = tree_ptr[src_idx];
        // Code of the source node.
        const auto src_code = src_node.code;
        // Range of the source node.
        const auto src_begin = static_cast<int>(src_node.begin), src_end = static_cast<int>(src_node.end);
        // Number of children of the source node.
        const auto n_children_src = static_cast<int>(src_node.n_children);
        // Node properties.
        auto node_x = src_node.props[0], node_y = src_node.props[1], node_z = src_node.props[2],
             node_mass = src_node.props[3];
        // Level of the source node.
        const auto src_level = src_node.level;
        // Square of the dimension of the source node.
        const auto src_dim2 = src_node.dim2;

        // Compute the shifted particle code. This is the particle code with one extra
        // top bit and then shifted down according to the level of the source node, so that
        // the top 1 bits of s_p_code and src_code are at the same position.
        const auto s_p_code = static_cast<UInt>(s_p_code_init >> ((cbits_v<UInt, NDim> - src_level) * NDim));
        // We need now to determine if the source node contains the target particle.
        // If it does, in all but one specific case we will have to correct
        // the source node COM coordinates & mass with the removal of the target particle.
        // The only exception is if the source node contains *only* the target particle,
        // in which case we want to avoid the correction because it would result in
        // infinities.
        //
        // The check s_p_code == src_code tells us if the source node contains the target
        // particle. The check (src_end - src_begin) != 1 tells us that the source node
        // contains other particles in addition to the target particle.
        if (s_p_code == src_code && (src_end - src_begin) != 1) {
            // Update the COM position.
            const auto new_node_mass = node_mass - mass;
            node_x = (node_x * node_mass - mass * x) / new_node_mass;
            node_y = (node_y * node_mass - mass * y) / new_node_mass;
            node_z = (node_z * node_mass - mass * z) / new_node_mass;
            // Don't forget to update the node mass as well.
            node_mass = new_node_mass;
        }

        // Compute the distance between target particle and source COM.
        // NOTE: if we are in a source node which contains only the target particle,
        // then dist2 and dist_vec will be zero.
        F diff_x = node_x - x, diff_y = node_y - y, diff_z = node_z - z,
          dist2 = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
        // Now let's run the BH check on *all* the target particles in the same wavefront.
        // NOTE: if we are in a source node which contains only the target particle,
        // then dist2 will have been set to zero above and the check always fails.
        if (__all_sync(unsigned(-1), src_dim2 < theta2 * dist2)) {
            // We are not in a leaf node containing only the target particle,
            // and the source node satisfies the BH criterion for the target
            // particle. We will then add the (approximated) contribution of the source node
            // to the final result.
            //
            // Start by adding the softening.
            dist2 += eps2;
            // Compute the (softened) distance.
            const auto dist = sqrt(dist2);
            const auto node_mass_dist3 = node_mass / (dist * dist2);
            r_x += diff_x * node_mass_dist3;
            r_y += diff_y * node_mass_dist3;
            r_z += diff_z * node_mass_dist3;
            // We can now skip all the children of the source node.
            src_idx += n_children_src + 1;
        } else {
            // Either the source node fails the BH criterion, or we are in a source node which
            // contains only the target particle.
            if (!n_children_src) {
                // We are in a leaf node. Compute all the interactions with the target particle.
                // NOTE: if we are in a source node which contains only the target particle,
                // then the loop will have just 1 iteration and the use of the is_tgt_particle
                // variable will ensure that all interactions of the particle with itself
                // amount to zero.
                for (auto i = src_begin; i < src_end; ++i) {
                    // Test if the current particle of the source leaf node coincides
                    // with the target particle.
                    const bool is_tgt_particle = pidx == i;
                    // Init the distance with the softening, plus add some extra
                    // softening if i is the target particle. This will avoid
                    // infinites when dividing by the distance below.
                    dist2 = eps2 + is_tgt_particle;
                    diff_x = ptr_x[i] - x;
                    diff_y = ptr_y[i] - y;
                    diff_z = ptr_z[i] - z;
                    dist2 += diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                    // Compute the distance, load the current source mass.
                    const auto dist = sqrt(dist2), m_i = ptr_mass[i];
                    // Q == 0 or 2: accelerations are requested.
                    const auto m_i_dist3 = m_i / (dist * dist2);
                    r_x += diff_x * m_i_dist3;
                    r_y += diff_y * m_i_dist3;
                    r_z += diff_z * m_i_dist3;
                }
            }
            // In any case, we keep traversing the tree moving to the next node
            // in depth-first order.
            ++src_idx;
        }
    }

    // Handle the G constant and write out the result.
    res_x[pidx] = G * r_x;
    res_y[pidx] = G * r_y;
    res_z[pidx] = G * r_z;
} // namespace detail

template <unsigned Q, std::size_t NDim, typename F, typename UInt>
void acc_pot_impl_cuda(const std::array<F *, tree_nvecs_res<Q, NDim>> &out, const tree_node_t<NDim, F, UInt> *tree,
                       tree_size_t<F> tree_size, const std::array<const F *, NDim + 1u> &p_parts, const UInt *codes,
                       tree_size_t<F> nparts, F theta2, F G, F eps2, tree_size_t<F> ncrit)
{
    static_assert(Q == 0u);

    // TODO error handling for memcopy.
    std::array<scoped_cu_array<F>, tree_nvecs_res<Q, NDim>> res_arrs;
    F *res_ptrs[tree_nvecs_res<Q, NDim>];
    for (std::size_t j = 0; j < tree_nvecs_res<Q, NDim>; ++j) {
        res_arrs[j] = scoped_cu_array<F>(boost::numeric_cast<std::size_t>(nparts));
        res_ptrs[j] = res_arrs[j].get();
    }

    scoped_cu_array<tree_node_t<NDim, F, UInt>> tree_arr(boost::numeric_cast<std::size_t>(tree_size));
    ::cudaMemcpy(tree_arr.get(), tree, sizeof(tree_node_t<NDim, F, UInt>) * tree_size, ::cudaMemcpyHostToDevice);
    const tree_node_t<NDim, F, UInt> *tree_ptr = tree_arr.get();

    std::array<scoped_cu_array<F>, NDim + 1u> parts_arrs;
    const F *parts_ptrs[NDim + 1u];
    for (std::size_t j = 0; j < NDim + 1u; ++j) {
        parts_arrs[j] = scoped_cu_array<F>(boost::numeric_cast<std::size_t>(nparts));
        ::cudaMemcpy(parts_arrs[j].get(), p_parts[j], sizeof(F) * nparts, ::cudaMemcpyHostToDevice);
        parts_ptrs[j] = parts_arrs[j].get();
    }

    scoped_cu_array<UInt> codes_arr(boost::numeric_cast<std::size_t>(nparts));
    ::cudaMemcpy(codes_arr.get(), codes, sizeof(UInt) * nparts, ::cudaMemcpyHostToDevice);
    const UInt *codes_ptr = codes_arr.get();

    // TODO overflow checks on ints?
    acc_pot_kernel<Q, NDim, F, UInt><<<(nparts + 31u) / 32u, 32u>>>(
        res_ptrs[0], res_ptrs[1], res_ptrs[2], parts_ptrs[0], parts_ptrs[1], parts_ptrs[2], parts_ptrs[3], codes_ptr,
        boost::numeric_cast<int>(nparts), tree_ptr, boost::numeric_cast<int>(tree_size), theta2, G, eps2);

    for (std::size_t j = 0; j < tree_nvecs_res<Q, NDim>; ++j) {
        ::cudaMemcpy(out[j], res_ptrs[j], sizeof(F) * nparts, ::cudaMemcpyDeviceToHost);
    }
}

// Explicit instantiations.
#define RAKAU_CUDA_EXPLICIT_INST(Q, NDim, F, UInt)                                                                     \
    template void acc_pot_impl_cuda<Q, NDim, F, UInt>(                                                                 \
        const std::array<F *, tree_nvecs_res<Q, NDim>> &, const tree_node_t<NDim, F, UInt> *, tree_size_t<F>,          \
        const std::array<const F *, NDim + 1u> &, const UInt *, tree_size_t<F>, F, F, F, tree_size_t<F>)

RAKAU_CUDA_EXPLICIT_INST(0, 3, float, std::uint64_t);

RAKAU_CUDA_EXPLICIT_INST(0, 3, double, std::uint64_t);

RAKAU_CUDA_EXPLICIT_INST(0, 3, float, std::uint32_t);

RAKAU_CUDA_EXPLICIT_INST(0, 3, double, std::uint32_t);

#undef RAKAU_CUDA_EXPLICIT_INST

} // namespace detail
} // namespace rakau
