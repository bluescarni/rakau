#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/preprocessor/seq/elem.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>

#include <rakau/detail/tree_fwd.hpp>

namespace rakau
{

inline namespace detail
{

// Minimum number of particles needed for running the cuda implementation.
unsigned cuda_min_size()
{
    return 1000u;
}

// Get the number of cuda devices.
unsigned cuda_device_count()
{
    int ret;
    if (::cudaGetDeviceCount(&ret) != ::cudaSuccess) {
        throw std::runtime_error("Cannot determine the number of CUDA devices");
    }
    return static_cast<unsigned>(ret);
}

// Small helper to create a unique_ptr to managed memory
// with enough storage for n objects of type T.
template <typename T>
auto make_scoped_cu_array(std::size_t n)
{
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::bad_alloc{};
    }
    void *ret;
    if (::cudaMallocManaged(&ret, n * sizeof(T)) != ::cudaSuccess) {
        throw std::bad_alloc{};
    }
    return std::unique_ptr<T, decltype(::cudaFree) *>(static_cast<T *>(ret), ::cudaFree);
}

// Small wrapper to handle arrays in managed memory.
template <typename T>
class scoped_cu_array
{
    using ptr_t = decltype(make_scoped_cu_array<T>(0));

public:
    // Def ctor, inits to nullptr.
    scoped_cu_array() : m_ptr(nullptr, ::cudaFree) {}
    // Constructor from size.
    explicit scoped_cu_array(std::size_t n) : m_ptr(make_scoped_cu_array<T>(n)) {}
    // Get a pointer to the start of the array.
    T *get() const
    {
        return m_ptr.get();
    }

private:
    ptr_t m_ptr;
};

// A few CUDA API wrappers with some minimal error checking.

static inline void cuda_memcpy(void *dst, const void *src, std::size_t count, ::cudaMemcpyKind kind)
{
    if (::cudaMemcpy(dst, src, count, kind) != ::cudaSuccess) {
        throw std::runtime_error("cudaMemcpy() returned an error code");
    }
}

static inline void cuda_device_synchronize()
{
    if (::cudaDeviceSynchronize() != ::cudaSuccess) {
        throw std::runtime_error("cudaDeviceSynchronize() returned an error code");
    }
}

static inline void cuda_set_device(int device)
{
    if (::cudaSetDevice(device) != ::cudaSuccess) {
        throw std::runtime_error("cudaSetDevice() returned an error code");
    }
}

template <typename T, std::size_t N>
struct arr_wrap {
    T value[N];
};

template <unsigned Q, std::size_t NDim, typename F, typename UInt>
__global__ void acc_pot_kernel(arr_wrap<F *, tree_nvecs_res<Q, NDim>> res_ptrs, int p_begin, int p_end,
                               const tree_node_t<NDim, F, UInt> *tree_ptr, int tree_size,
                               arr_wrap<const F *, NDim + 1u> parts_ptrs, const UInt *codes_ptr, F theta2, F G, F eps2)
{
    // Get the local and global particle indices.
    const auto loc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto pidx = loc_idx + p_begin;
    if (pidx >= p_end) {
        // Don't do anything if we are in the remainder
        // of the last block.
        return;
    }

    // Array of results, inited to zeroes.
    constexpr auto res_array_size = tree_nvecs_res<Q, NDim>;
    F res_array[res_array_size]{};

    // Load the particle code, position and mass.
    const auto p_code = codes_ptr[pidx];
    F p_pos[NDim];
    for (std::size_t j = 0; j < NDim; ++j) {
        p_pos[j] = parts_ptrs.value[j][pidx];
    }
    const auto p_mass = parts_ptrs.value[NDim][pidx];

    // Temporary arrays that will be used in the loop.
    F dist_vec[NDim], props[NDim + 1u];

    // Add a 1 bit just above the highest possible bit position for the particle code.
    // This value is used in the loop, we precompute it here.
    const auto s_p_code_init = static_cast<UInt>(p_code | (UInt(1) << (cbits_v<UInt, NDim> * NDim)));

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
        for (std::size_t j = 0; j < NDim + 1u; ++j) {
            props[j] = src_node.props[j];
        }
        // Level of the source node.
        const auto src_level = src_node.level;
        // Square of the dimension of the source node.
        const auto src_dim2 = src_node.dim2;

        // Compute the shifted particle code. This is the particle code with one extra
        // top bit and then shifted down according to the level of the source node, so that
        // the top 1 bits of s_p_code and src_code are at the same position.
        // If s_p_code == src_code, then it means that the source node contains the target particle
        // (or, in other words, the source node is an ancestor of the leaf node containing
        // the target particle).
        const auto s_p_code = s_p_code_init >> ((cbits_v<UInt, NDim> - src_level) * NDim);
        // If the source node contains the target particle, we will need to account for self interactions
        // in the tree traversal. There are two different approaches that can be taken.
        //
        // The first is to modify on-the-fly the properties of the source node with the removal of the target
        // particle. In the classic BH scheme, this will alter the COM position of the source node
        // and its mass. The alteration needs to take place if the source node is an ancestor of
        // NT, the leaf node of the target particle. However, if the source node coincides with NT
        // and NT contains *only* the target particle, then the alteration must not take place because
        // otherwise we generate infinities (the COM of a system of only 1 particle is the particle
        // itself). The alteration can be done by defining a mass factor mf as
        //
        // mf = orig_node_mass / (orig_node_mass - p_mass * needs_alteration),
        //
        // where needs_alteration is a boolean that expresses whether the source node needs to be
        // adjusted or not (so that mf == 1 if no adjustment needs to happen). It can then be shown
        // that the target particle's distance from the adjusted COM is
        //
        // new_dist = mf * orig_dist,
        //
        // where orig_dist is the (vector) distance from the original COM. The new node mass will be:
        //
        // new_node_mass = orig_node_mass - p_mass * needs_alteration.
        //
        // The other approach is not to modify the properties of the COM, and instead just continue
        // in the tree traversal as if the current source node didn't satisfy the BH check. By doing this
        // we will eventually land into the leaf node of the target particle, where we will compute
        // local particle-particle interactions in the usual N**2 way (avoiding self interactions for the
        // target particle).
        //
        // The first method is more arithmetically-intensive and requires less flow control. The other
        // method will result in longer tree traversals and higher flow control, but requires less arithmetics.
        // At the moment it seems like the first method might be a bit faster on the GPU, but it's also not
        // entirely clear how more complicated/intensive the source node alteration would become once we
        // implement quadrupole moments and other MACs. Thus, for now, let's go with the second approach.

        // Compute the distance between target particle and source COM.
        // NOTE: if we are in a source node which contains only the target particle,
        // then dist2 and dist_vec will be zero.
        F dist2(0);
        for (std::size_t j = 0; j < NDim; ++j) {
            const auto diff = props[j] - p_pos[j];
            dist2 += diff * diff;
            dist_vec[j] = diff;
        }

        // Now let's run the BH/ancestor check on all the target particles in the same warp.
        if (__all_sync(unsigned(-1), s_p_code != src_code && src_dim2 < theta2 * dist2)) {
            // The source node does not contain the target particle and it satisfies the BH check.
            // We will then add the (approximated) contribution of the source node
            // to the final result.
            //
            // Start by adding the softening.
            dist2 += eps2;
            // Compute the (softened) distance.
            const auto dist = sqrt(dist2);
            if (Q == 0u || Q == 2u) {
                // Q == 0 or 2: accelerations are requested.
                const auto node_mass_dist3 = props[NDim] / (dist * dist2);
                for (std::size_t j = 0; j < NDim; ++j) {
                    res_array[j] += dist_vec[j] * node_mass_dist3;
                }
            }
            if (Q == 1u || Q == 2u) {
                // Q == 1 or 2: potentials are requested.
                // Establish the index of the potential in the result array:
                // 0 if only the potentials are requested, NDim otherwise.
                constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim);
                // Add the potential due to the node.
                res_array[pot_idx] -= p_mass * props[NDim] / dist;
            }
            // We can now skip all the children of the source node.
            src_idx += n_children_src + 1;
        } else {
            // Either the source node contains the target particle, or it fails the BH check.
            if (!n_children_src) {
                // We are in a leaf node (possibly containing the target particle).
                // Compute all the interactions with the target particle.
                for (auto i = src_begin; i < src_end; ++i) {
                    // Test if the current particle of the source leaf node coincides
                    // with the target particle.
                    const bool is_tgt_particle = pidx == i;
                    // Init the distance with the softening, plus add some extra
                    // softening if i is the target particle. This will avoid
                    // infinites when dividing by the distance below.
                    dist2 = eps2 + is_tgt_particle;
                    for (std::size_t j = 0; j < NDim; ++j) {
                        const auto diff = parts_ptrs.value[j][i] - p_pos[j];
                        dist2 += diff * diff;
                        dist_vec[j] = diff;
                    }
                    // Compute the distance, load the current source mass.
                    const auto dist = sqrt(dist2), m_i = parts_ptrs.value[NDim][i];
                    if (Q == 0u || Q == 2u) {
                        // Q == 0 or 2: accelerations are requested.
                        const auto m_i_dist3 = m_i / (dist * dist2);
                        for (std::size_t j = 0; j < NDim; ++j) {
                            // NOTE: if i == pidx, then dist_vec will be a vector
                            // of zeroes and res_array will not be modified.
                            res_array[j] += dist_vec[j] * m_i_dist3;
                        }
                    }
                    if (Q == 1u || Q == 2u) {
                        // Q == 1 or 2: potentials are requested.
                        // Establish the index of the potential in the result array:
                        // 0 if only the potentials are requested, NDim otherwise.
                        constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim);
                        // NOTE: for the potential, we need an extra multiplication by
                        // !is_tgt_particle to set the rhs to zero in case i == pidx (for
                        // the accelerations, the same effect was achieved via dist_vec).
                        res_array[pot_idx] -= !is_tgt_particle * p_mass * m_i / dist;
                    }
                }
            }
            // Keep traversing the tree moving to the next node in depth-first order.
            ++src_idx;
        }
    }

    // Handle the G constant and write out the result.
    for (std::size_t j = 0; j < tree_nvecs_res<Q, NDim>; ++j) {
        // NOTE: for writing the results, we use the local index.
        res_ptrs.value[j][loc_idx] = G * res_array[j];
    }
}

template <unsigned Q, std::size_t NDim, typename F, typename UInt>
void cuda_acc_pot_impl(const std::array<F *, tree_nvecs_res<Q, NDim>> &out,
                       const std::vector<tree_size_t<F>> &split_indices, const tree_node_t<NDim, F, UInt> *tree,
                       tree_size_t<F> tree_size, const std::array<const F *, NDim + 1u> &p_parts, const UInt *codes,
                       tree_size_t<F> nparts, F theta2, F G, F eps2)
{
    assert(split_indices.size() && split_indices.size() - 1u <= cuda_device_count());

    // Fetch how many gpus we will actually be using.
    // NOTE: this is ensured to be not greater than the value returned
    // by the CUDA api, due to the checks we do outside this function.
    // So, we can freely cast it around to unsigned and signed int as well.
    const auto ngpus = static_cast<unsigned>(split_indices.size() - 1u);

    // Create the arrays that will hold the results.
    std::vector<arr_wrap<scoped_cu_array<F>, tree_nvecs_res<Q, NDim>>> res_arrs;
    std::vector<arr_wrap<F *, tree_nvecs_res<Q, NDim>>> res_ptrs;
    for (auto i = 0u; i < ngpus; ++i) {
        typename decltype(res_arrs)::value_type tmp_arrs;
        typename decltype(res_ptrs)::value_type tmp_ptrs;
        for (std::size_t j = 0; j < tree_nvecs_res<Q, NDim>; ++j) {
            tmp_arrs.value[j]
                = scoped_cu_array<F>(boost::numeric_cast<std::size_t>(split_indices[i + 1u] - split_indices[i]));
            tmp_ptrs.value[j] = tmp_arrs.value[j].get();
        }
        res_arrs.emplace_back(std::move(tmp_arrs));
        res_ptrs.emplace_back(std::move(tmp_ptrs));
    }

    // Copy over the tree data.
    scoped_cu_array<tree_node_t<NDim, F, UInt>> tree_arr(boost::numeric_cast<std::size_t>(tree_size));
    cuda_memcpy(tree_arr.get(), tree, sizeof(tree_node_t<NDim, F, UInt>) * tree_size, ::cudaMemcpyDefault);
    const auto *tree_ptr = tree_arr.get();

    // Copy over the particles' data.
    arr_wrap<scoped_cu_array<F>, NDim + 1u> parts_arrs;
    arr_wrap<const F *, NDim + 1u> parts_ptrs;
    for (std::size_t j = 0; j < NDim + 1u; ++j) {
        parts_arrs.value[j] = scoped_cu_array<F>(boost::numeric_cast<std::size_t>(nparts));
        cuda_memcpy(parts_arrs.value[j].get(), p_parts[j], sizeof(F) * nparts, ::cudaMemcpyDefault);
        parts_ptrs.value[j] = parts_arrs.value[j].get();
    }

    // Copy over the codes.
    scoped_cu_array<UInt> codes_arr(boost::numeric_cast<std::size_t>(nparts));
    cuda_memcpy(codes_arr.get(), codes, sizeof(UInt) * nparts, ::cudaMemcpyDefault);
    const auto *codes_ptr = codes_arr.get();

    // NOTE: not 100% sure this is necessary here, as the docs say that memory copy
    // functions have "mostly" synchronizing behaviour. Better safe than sorry, I guess?
    cuda_device_synchronize();

    // Run the computations on the devices.
    for (auto i = 0u; i < ngpus; ++i) {
        // Set the device.
        cuda_set_device(static_cast<int>(i));

        // Number of particles for which we will be
        // computing the accelerations/potentials for
        // this device.
        const auto loc_nparts = split_indices[i + 1u] - split_indices[i];

        // Run the kernel.
        // TODO overflow checks on kernel launch param?
        acc_pot_kernel<Q, NDim, F, UInt><<<(loc_nparts + 31u) / 32u, 32u>>>(
            res_ptrs[i], boost::numeric_cast<int>(split_indices[i]), boost::numeric_cast<int>(split_indices[i + 1u]),
            tree_ptr, boost::numeric_cast<int>(tree_size), parts_ptrs, codes_ptr, theta2, G, eps2);
    }

    // Wait for all the kernels to finish.
    cuda_device_synchronize();

    // Write out the results.
    for (auto i = 0u; i < ngpus; ++i) {
        for (std::size_t j = 0; j < tree_nvecs_res<Q, NDim>; ++j) {
            cuda_memcpy(out[j] + split_indices[i], res_ptrs[i].value[j],
                        sizeof(F) * (split_indices[i + 1u] - split_indices[i]), ::cudaMemcpyDefault);
        }
    }

    // Last sync.
    cuda_device_synchronize();
}

// Explicit instantiations of the templates implemented above. We are going to use Boost.Preprocessor.
// It's gonna look ugly, but it will allow us to avoid a lot of typing.

// Define the values/types that we will use for the concrete instantiations.

// Only quadtrees and octrees for the moment.
#define RAKAU_CUDA_INST_DIM_SEQUENCE (2)(3)

// float and double only on the gpu.
#define RAKAU_CUDA_INST_FP_SEQUENCE (float)(double)

// 32/64bit types for the particle codes.
#define RAKAU_CUDA_INST_UINT_SEQUENCE (std::uint32_t)(std::uint64_t)

// Computation of accelerations, potentials or both.
#define RAKAU_CUDA_INST_Q_SEQUENCE (0)(1)(2)

// Macro for the instantiation of the main function. NDim, F, UInt and Q will be passed in
// as a sequence named Args (in that order).
#define RAKAU_CUDA_EXPLICIT_INST_FUN(r, Args)                                                                          \
    template void cuda_acc_pot_impl<BOOST_PP_SEQ_ELEM(3, Args), BOOST_PP_SEQ_ELEM(0, Args),                            \
                                    BOOST_PP_SEQ_ELEM(1, Args), BOOST_PP_SEQ_ELEM(2, Args)>(                           \
        const std::array<BOOST_PP_SEQ_ELEM(1, Args) *,                                                                 \
                         tree_nvecs_res<BOOST_PP_SEQ_ELEM(3, Args), BOOST_PP_SEQ_ELEM(0, Args)>> &,                    \
        const std::vector<tree_size_t<BOOST_PP_SEQ_ELEM(1, Args)>> &,                                                  \
        const tree_node_t<BOOST_PP_SEQ_ELEM(0, Args), BOOST_PP_SEQ_ELEM(1, Args), BOOST_PP_SEQ_ELEM(2, Args)> *,       \
        tree_size_t<BOOST_PP_SEQ_ELEM(1, Args)>,                                                                       \
        const std::array<const BOOST_PP_SEQ_ELEM(1, Args) *, BOOST_PP_SEQ_ELEM(0, Args) + 1u> &,                       \
        const BOOST_PP_SEQ_ELEM(2, Args) *, tree_size_t<BOOST_PP_SEQ_ELEM(1, Args)>, BOOST_PP_SEQ_ELEM(1, Args),       \
        BOOST_PP_SEQ_ELEM(1, Args), BOOST_PP_SEQ_ELEM(1, Args));

// Do the actual instantiation via a cartesian product over the sequences.
// clang-format off
BOOST_PP_SEQ_FOR_EACH_PRODUCT(RAKAU_CUDA_EXPLICIT_INST_FUN, (RAKAU_CUDA_INST_DIM_SEQUENCE)(RAKAU_CUDA_INST_FP_SEQUENCE)(RAKAU_CUDA_INST_UINT_SEQUENCE)(RAKAU_CUDA_INST_Q_SEQUENCE));
// clang-format on

} // namespace detail
} // namespace rakau
