#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wdeprecated-dynamic-exception-spec"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"

#include <hc.hpp>
#include <hc_math.hpp>

#pragma GCC diagnostic pop

#include <boost/numeric/conversion/cast.hpp>

#include <rakau/detail/hc_fwd.hpp>
#include <rakau/detail/tree_fwd.hpp>

namespace rakau
{

inline namespace detail
{

// Machinery to transform an array of pointers into a tuple of array views.
template <typename T, std::size_t N, std::size_t... I>
inline auto ap2tv_impl(const std::array<T *, N> &a, int size, std::index_sequence<I...>)
{
    return std::make_tuple(hc::array_view<T, 1>(size, a[I])...);
}

template <typename T, std::size_t N>
inline auto ap2tv(const std::array<T *, N> &a, int size)
{
    return ap2tv_impl(a, size, std::make_index_sequence<N>{});
}

// Helper to turn a tuple of values of the same type into an array.
template <typename Tuple, std::size_t... I>
inline auto t2a_impl(const Tuple &tup, std::index_sequence<I...>)
#if defined(__HCC_ACCELERATOR__)
    [[hc]]
#endif
{
    return std::array<std::tuple_element_t<0, Tuple>, std::tuple_size_v<Tuple>>{std::get<I>(tup)...};
}

template <typename Tuple>
inline auto t2a(const Tuple &tup)
#if defined(__HCC_ACCELERATOR__)
    [[hc]]
#endif
{
    return t2a_impl(tup, std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

template <unsigned Q, std::size_t NDim, typename F, typename UInt>
void acc_pot_impl_hcc(const std::array<F *, tree_nvecs_res<Q, NDim>> &out, const tree_node_t<NDim, F, UInt> *tree,
                      tree_size_t<F> tree_size, const std::array<const F *, NDim + 1u> &p_parts, const UInt *codes,
                      tree_size_t<F> nparts, F theta2, F G, F eps2, tree_size_t<F> ncrit)
{
    hc::array_view<const tree_node_t<NDim, F, UInt>, 1> tree_view(boost::numeric_cast<int>(tree_size), tree);
    hc::array_view<const UInt, 1> codes_view(boost::numeric_cast<int>(nparts), codes);
    // Turn the input particles data and the result buffers into tuples of views, for use
    // in the parallel_for_each().
    auto pt = ap2tv(p_parts, boost::numeric_cast<int>(nparts));
    auto rt = ap2tv(out, boost::numeric_cast<int>(nparts));
    hc::parallel_for_each(
        hc::extent<1>(boost::numeric_cast<int>(nparts)).tile(boost::numeric_cast<int>(ncrit)),
        [
            tree_view, tree_size = static_cast<int>(tree_size), nparts = static_cast<int>(nparts), pt, codes_view, rt,
            theta2, G,
            eps2
        ](hc::tiled_index<1> thread_id) [[hc]] {
            // Get the global particle index.
            const auto pidx = thread_id.global[0];
            if (pidx >= nparts) {
                // Don't do anything if we are in the remainder
                // of the last tile.
                return;
            }

            // Turn the tuples back into arrays.
            auto p_view = t2a(pt);
            auto res_view = t2a(rt);

            // Array of results, inited to zeroes.
            std::array<F, tree_nvecs_res<Q, NDim>> res_array{};

            // Load the particle code, position and mass.
            const auto p_code = codes_view[pidx];
            std::array<F, NDim> p_pos;
            for (std::size_t j = 0; j < NDim; ++j) {
                p_pos[j] = p_view[j][pidx];
            }
            const F p_mass = p_view[NDim][pidx];

            // Temporary arrays that will be used in the loop.
            std::array<F, NDim> dist_vec, com_pos;

            // Add a 1 bit just above the highest possible bit position for the particle code.
            // This value is used in the loop, we precompute it here.
            const auto s_p_code_init = static_cast<UInt>(p_code | (UInt(1) << (cbits_v<UInt, NDim> * NDim)));

            // Loop over the tree.
            for (auto src_idx = 0; src_idx < tree_size;) {
                // Get a reference to the current source node, and cache locally a few quantities.
                const auto &src_node = tree_view[src_idx];
                // Code of the source node.
                const auto src_code = std::get<0>(src_node);
                // Range of the source node.
                const auto src_begin = static_cast<int>(std::get<1>(src_node)[0]),
                           src_end = static_cast<int>(std::get<1>(src_node)[1]);
                // Number of children of the source node.
                const auto n_children_src = static_cast<int>(std::get<1>(src_node)[2]);
                // Total mass in the source node.
                auto node_mass = std::get<2>(src_node);
                // Position of the centre of mass of the source node.
                com_pos = std::get<3>(src_node);
                // Level of the source node.
                const auto src_level = std::get<4>(src_node);
                // Square of the dimension of the source node.
                const auto src_dim2 = std::get<5>(src_node);

                // Compute the shifted particle code. This is the particle code with one extra
                // top bit and then shifted down according to the level of the source node, so that
                // the top 1 bits of s_p_code and src_code are at the same position.
                const auto s_p_code = static_cast<UInt>(s_p_code_init >> ((cbits_v<UInt, NDim> - src_level) * NDim));
                // Determine if the source node is an ancestor of the leaf node of the
                // target particle. It is if the code of the source node and the shifted
                // particle code coincide.
                // NOTE: according to this test, the leaf node of the target particle
                // is also considered an ancestor.
                const bool ancestor_node = s_p_code == src_code;
                // Determine if we are in the leaf node of the target particle, and if
                // the leaf node has only the particle itself.
                const bool unitary_leaf_node = ancestor_node && (src_end - src_begin) == 1;
                // If we are dealing with an ancestor node, normally we will have to
                // subtract from it the target particle, otherwise we will have
                // self interactions involving the target particle. The only exception
                // is if we are in the leaf node of the target particle and the leaf
                // node contains only that particle. In such case, we will not subtract
                // the particle, otherwise we will run into infinities below.
                const bool subtract_p = ancestor_node && !unitary_leaf_node;

                // Do the particle subtraction, which will update the COM position and the total mass of the node.
                const F sub_mass = subtract_p * p_mass, new_node_mass = node_mass - sub_mass,
                        rec_new_node_mass = F(1) / new_node_mass;
                for (std::size_t j = 0; j < NDim; ++j) {
                    com_pos[j] = (com_pos[j] * node_mass - sub_mass * p_pos[j]) * rec_new_node_mass;
                }
                // Don't forget to update the node mass as well.
                node_mass = new_node_mass;

                // Compute the distance between target particle and source COM.
                // NOTE: in case we are in the leaf node of the target particle and
                // the leaf node contains only that particle, then dist2 will be zero
                // (or very close to it).
                F dist2(0);
                for (std::size_t j = 0; j < NDim; ++j) {
                    const auto diff = com_pos[j] - p_pos[j];
                    dist2 += diff * diff;
                    dist_vec[j] = diff;
                }
                if (!unitary_leaf_node && src_dim2 < theta2 * dist2) {
                    // We are not in a leaf node containing only the target particle,
                    // and the source node satisfies the BH criterion for the target
                    // particle. We will then add the (approximated) contribution of the source node
                    // to the final result.
                    //
                    // Start by adding the softening.
                    dist2 += eps2;
                    // Compute the (softened) distance.
                    const F dist = sqrt(dist2);
                    if constexpr (Q == 0u || Q == 2u) {
                        // Q == 0 or 2: accelerations are requested.
                        const auto node_mass_dist3 = node_mass / (dist * dist2);
                        for (std::size_t j = 0; j < NDim; ++j) {
                            res_array[j] += dist_vec[j] * node_mass_dist3;
                        }
                    }
                    if constexpr (Q == 1u || Q == 2u) {
                        // Q == 1 or 2: potentials are requested.
                        // Establish the index of the potential in the result array:
                        // 0 if only the potentials are requested, NDim otherwise.
                        constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim);
                        // Add the potential due to the node.
                        res_array[pot_idx] -= p_mass * node_mass / dist;
                    }
                    // We can now skip all the children of the source node.
                    src_idx += n_children_src + 1;
                } else {
                    // The source node is either a unitary leaf node or it does not satisfy the BH criterion.
                    if (!n_children_src) {
                        // We are in a leaf node, possibly containing the target particle. Compute all
                        // the interactions with the target particle, taking care of avoiding self interactions.
                        for (auto i = src_begin + (src_begin == pidx); i < src_end; i = (i + 1) + ((i + 1) == pidx)) {
                            // Init the distance with the softening.
                            dist2 = eps2;
                            for (std::size_t j = 0; j < NDim; ++j) {
                                const auto diff = p_view[j][i] - p_pos[j];
                                dist2 += diff * diff;
                                dist_vec[j] = diff;
                            }
                            // Compute the distance, load the current source mass.
                            const auto dist = sqrt(dist2), m_i = p_view[NDim][i];
                            if constexpr (Q == 0u || Q == 2u) {
                                // Q == 0 or 2: accelerations are requested.
                                const auto m_i_dist3 = m_i / (dist * dist2);
                                for (std::size_t j = 0; j < NDim; ++j) {
                                    res_array[j] += dist_vec[j] * m_i_dist3;
                                }
                            }
                            if constexpr (Q == 1u || Q == 2u) {
                                // Q == 1 or 2: potentials are requested.
                                // Establish the index of the potential in the result array:
                                // 0 if only the potentials are requested, NDim otherwise.
                                constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim);
                                res_array[pot_idx] -= p_mass * m_i / dist;
                            }
                        }
                    }
                    // In any case, we keep traversing the tree moving to the next node in depth-first
                    // order.
                    ++src_idx;
                }
            }

            // Handle the G constant and write out the result.
            for (std::size_t j = 0; j < tree_nvecs_res<Q, NDim>; ++j) {
                res_view[j][pidx] = G * res_array[j];
            }
        })
        .get();
    for (auto &v : t2a(rt)) {
        v.synchronize();
    }
}

// Explicit instantiations.
#define RAKAU_HC_EXPLICIT_INST(Q, NDim, F, UInt)                                                                       \
    template void acc_pot_impl_hcc<Q, NDim, F, UInt>(                                                                  \
        const std::array<F *, tree_nvecs_res<Q, NDim>> &, const tree_node_t<NDim, F, UInt> *, tree_size_t<F>,          \
        const std::array<const F *, NDim + 1u> &, const UInt *, tree_size_t<F>, F, F, F, tree_size_t<F>)

RAKAU_HC_EXPLICIT_INST(0, 3, float, std::size_t);
RAKAU_HC_EXPLICIT_INST(1, 3, float, std::size_t);
RAKAU_HC_EXPLICIT_INST(2, 3, float, std::size_t);

RAKAU_HC_EXPLICIT_INST(0, 3, double, std::size_t);
RAKAU_HC_EXPLICIT_INST(1, 3, double, std::size_t);
RAKAU_HC_EXPLICIT_INST(2, 3, double, std::size_t);

#undef RAKAU_HC_EXPLICIT_INST

} // namespace detail
} // namespace rakau
