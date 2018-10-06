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

template <unsigned Q, std::size_t NDim, typename F, typename TreeView, typename PView, typename ResView>
inline void tree_acc_pot_leaf_hcc(TreeView tree_view, F eps2, int src_idx, int pidx, PView p_view, ResView res_view)
    [[hc]]
{
    // Local cache.
    const auto &src_node = tree_view[src_idx];
    // Establish the range of the source node.
    const auto src_begin = static_cast<int>(std::get<1>(src_node)[0]),
               src_end = static_cast<int>(std::get<1>(src_node)[1]);
    // Local variables for the scalar computation.
    std::array<F, NDim> pos1, diffs;
    // Load the coordinates of the current particle
    // in the target node.
    for (std::size_t j = 0; j < NDim; ++j) {
        pos1[j] = p_view[j][pidx];
    }
    // Load the target mass, but only if we are interested in the potentials.
    [[maybe_unused]] F m1;
    if constexpr (Q == 1u || Q == 2u) {
        m1 = p_view[NDim][pidx];
    }
    // Iterate over the particles in the src node.
    for (auto i = src_begin; i < src_end; ++i) {
        F dist2(eps2);
        for (std::size_t j = 0; j < NDim; ++j) {
            diffs[j] = p_view[j][i] - pos1[j];
            dist2 = fma(diffs[j], diffs[j], dist2);
        }
        const auto dist = sqrt(dist2), m2 = p_view[NDim][i];
        if constexpr (Q == 0u || Q == 2u) {
            // Q == 0 or 2: accelerations are requested.
            const auto dist3 = dist * dist2, m_dist3 = m2 / dist3;
            for (std::size_t j = 0; j < NDim; ++j) {
                res_view[j][pidx] = fma(diffs[j], m_dist3, res_view[j][pidx]);
            }
        }
        if constexpr (Q == 1u || Q == 2u) {
            // Q == 1 or 2: potentials are requested.
            // Establish the index of the potential in the result array:
            // 0 if only the potentials are requested, NDim otherwise.
            constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0 : NDim);
            res_view[pot_idx][pidx] = fma(-m1, m2 / dist, res_view[pot_idx][pidx]);
        }
    }
}

template <unsigned Q, std::size_t NDim, typename F, typename TreeView, typename PView, typename ResView>
inline void tree_self_interactions_hcc(TreeView tree_view, F eps2, int pidx, int tgt_begin, int tgt_end, PView p_view,
                                       ResView res_view) [[hc]]
{
    // Temporary vectors to be used in the loops below.
    std::array<F, NDim> diffs, pos;
    // Load the coords of the current particle.
    for (std::size_t j = 0; j < NDim; ++j) {
        pos[j] = p_view[j][pidx];
    }
    // Load the mass of the current particle.
    const auto m1 = p_view[NDim][pidx];
    // The acceleration/potential vector for the current particle
    // (inited to zero).
    std::array<F, tree_nvecs_res<Q, NDim>> a1{};
    for (int i = tgt_begin; i < tgt_end; ++i) {
        if (i == pidx) {
            continue;
        }
        // Determine dist2, dist and dist3.
        F dist2(eps2);
        for (std::size_t j = 0; j < NDim; ++j) {
            const auto tmp = p_view[j][i] - pos[j];
            diffs[j] = tmp;
            dist2 = fma(tmp, tmp, dist2);
        }
        const auto dist = sqrt(dist2), m2 = p_view[NDim][i];
        if constexpr (Q == 0u || Q == 2u) {
            // Q == 0 or 2: accelerations are requested.
            const auto dist3 = dist2 * dist, m2_dist3 = m2 / dist3, m1_dist3 = m1 / dist3;
            // Accumulate the accelerations.
            for (std::size_t j = 0; j < NDim; ++j) {
                a1[j] = fma(m2_dist3, diffs[j], a1[j]);
            }
        }
        if constexpr (Q == 1u || Q == 2u) {
            // Q == 1 or 2: potentials are requested.
            // Establish the index of the potential in the result array:
            // 0 if only the potentials are requested, NDim otherwise.
            constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim);
            // Compute the negated mutual potential.
            const auto mut_pot = m1 / dist * m2;
            // Subtract mut_pot from the accumulator for the current particle.
            a1[pot_idx] -= mut_pot;
        }
    }
    // Update the acceleration/potential on the first particle
    // in the temporary storage.
    if constexpr (Q == 0u || Q == 2u) {
        for (std::size_t j = 0; j < NDim; ++j) {
            res_view[j][pidx] += a1[j];
        }
    }
    if constexpr (Q == 1u || Q == 2u) {
        constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim);
        // NOTE: addition, because the value in a1[pot_idx] was already built
        // as a negative quantity.
        res_view[pot_idx][pidx] += a1[pot_idx];
    }
}

template <unsigned Q, std::size_t NDim, typename F, typename TreeView, typename PView, typename ResView>
inline void tree_acc_pot_bh_com_hcc(TreeView tree_view, F eps2, int src_idx, int pidx, PView p_view, ResView res_view,
                                    F dist2, const std::array<F, NDim> &dist_vec) [[hc]]
{
    // Add the softening length to dist2.
    dist2 += eps2;
    // Load locally the mass of the source node.
    const auto m_src = std::get<2>(tree_view[src_idx]);
    // Compute the distance.
    const F dist = sqrt(dist2);
    if constexpr (Q == 0u || Q == 2u) {
        // Q == 0 or 2: accelerations are requested.
        const auto m_src_dist3 = m_src / (dist * dist2);
        for (std::size_t j = 0; j < NDim; ++j) {
            res_view[j][pidx] = fma(dist_vec[j], m_src_dist3, res_view[j][pidx]);
        }
    }
    if constexpr (Q == 1u || Q == 2u) {
        // Q == 1 or 2: potentials are requested.
        // Establish the index of the potential in the result array:
        // 0 if only the potentials are requested, NDim otherwise.
        constexpr auto pot_idx = static_cast<std::size_t>(Q == 1u ? 0u : NDim);
        // Load the target mass and compute the potential.
        res_view[pot_idx][pidx] = fma(-p_view[NDim][pidx], m_src / dist, res_view[pot_idx][pidx]);
    }
}

template <unsigned Q, std::size_t NDim, typename F, typename TreeView, typename PView, typename ResView>
inline int tree_acc_pot_bh_check_hcc(TreeView tree_view, int src_idx, F theta2, F eps2, int pidx, PView p_view,
                                     ResView res_view) [[hc]]
{
    // Local cache.
    const auto &src_node = tree_view[src_idx];
    // Copy locally the number of children of the source node.
    const auto n_children_src = static_cast<int>(std::get<1>(src_node)[2]);
    // Copy locally the COM coords of the source.
    const auto com_pos = std::get<3>(src_node);
    // Copy locally the dim2 of the source node.
    const auto src_dim2 = std::get<5>(src_node);
    // Compute the distance between target particle and source COM.
    F dist2(0);
    std::array<F, 3> dist_vec;
    for (std::size_t j = 0; j < NDim; ++j) {
        const auto diff = com_pos[j] - p_view[j][pidx];
        dist2 = fma(diff, diff, dist2);
        dist_vec[j] = diff;
    }
    // const F dx = com_pos[0] - p_view[0][pidx], dy = com_pos[1] - p_view[1][pidx], dz = com_pos[2] - p_view[2][pidx];
    // const F dist2 = dx * dx + dy * dy + dz * dz;
    if (src_dim2 < theta2 * dist2) {
        // The source node satisfies the BH criterion for all the particles of the target node. Add the
        // acceleration due to the com of the source node.
        tree_acc_pot_bh_com_hcc<Q, NDim>(tree_view, eps2, src_idx, pidx, p_view, res_view, dist2, dist_vec);
        // We can now skip all the children of the source node.
        return src_idx + n_children_src + 1;
    }
    // The source node does not satisfy the BH criterion. We check if it is a leaf
    // node, in which case we need to compute all the pairwise interactions.
    if (!n_children_src) {
        // Leaf node.
        tree_acc_pot_leaf_hcc<Q, NDim>(tree_view, eps2, src_idx, pidx, p_view, res_view);
    }
    // In any case, we keep traversing the tree moving to the next node in depth-first order.
    return src_idx + 1;
}

template <unsigned Q, std::size_t NDim, typename F, typename UInt>
void acc_pot_impl_hcc(const std::array<F *, tree_nvecs_res<Q, NDim>> &out, const tree_cnode_t<F, UInt> *cnodes,
                      tree_size_t<F> cnodes_size, const tree_node_t<NDim, F, UInt> *tree, tree_size_t<F> tree_size,
                      const std::array<const F *, NDim + 1u> &p_parts, tree_size_t<F> nparts, F theta2, F G, F eps2,
                      tree_size_t<F> ncrit)
{
    using size_type = tree_size_t<F>;
    hc::array_view<const tree_node_t<NDim, F, UInt>, 1> tree_view(boost::numeric_cast<int>(tree_size), tree);
    hc::array_view<const tree_cnode_t<F, UInt>, 1> cnodes_view(boost::numeric_cast<int>(cnodes_size), cnodes);
    // Turn the input particles data and the result buffers into tuples of views, for use
    // in the parallel_for_each().
    auto pt = ap2tv(p_parts, boost::numeric_cast<int>(nparts));
    auto rt = ap2tv(out, boost::numeric_cast<int>(nparts));
    // TODO overflow checks on ncrit*ncodes.
    hc::parallel_for_each(
        hc::extent<1>(boost::numeric_cast<int>(ncrit * cnodes_size)).tile(boost::numeric_cast<int>(ncrit)),
        [cnodes_view, tree_view, tsize = static_cast<int>(tree_size), pt, rt, theta2, G,
         eps2](hc::tiled_index<1> thread_id) [[hc]] {
            // Turn the tuples back into arrays.
            auto p_view = t2a(pt);
            auto res_view = t2a(rt);

            // Fetch the info about the critical node to which the current particle belongs.
            const auto tgt_code = std::get<0>(cnodes_view[thread_id.tile]);
            const auto tgt_begin = static_cast<int>(std::get<1>(cnodes_view[thread_id.tile]));
            const auto tgt_end = static_cast<int>(std::get<2>(cnodes_view[thread_id.tile]));
            const auto tgt_level = tree_level<NDim>(tgt_code);

            // The tile size is the maximum number of particles in a critical node.
            // Many critical nodes will have fewer particles than that, so we must ensure
            // that we are not going out of the boundaries of the critical node.
            if (thread_id.local[0] >= (tgt_end - tgt_begin)) {
                return;
            }
            // Compute the global index in the particles arrays of the current particle.
            const int pidx = tgt_begin + thread_id.local[0];

            // Start looping over the source data.
            for (int src_idx = 0; src_idx < tsize;) {
                // Get a reference to the current source node.
                const auto &src_node = tree_view[src_idx];
                // Extract the code of the source node.
                const auto src_code = std::get<0>(src_node);
                // Number of children of the source node.
                const auto n_children_src = static_cast<int>(std::get<1>(src_node)[2]);
                if (src_code == tgt_code) {
                    // If src_code == tgt_code, we are currently visiting the target node.
                    // Compute the self interactions and skip all the children of the target node.
                    tree_self_interactions_hcc<Q, NDim>(tree_view, eps2, pidx, tgt_begin, tgt_end, p_view, res_view);
                    src_idx += n_children_src + 1;
                    continue;
                }
                // Extract the level of the source node.
                const auto src_level = std::get<4>(src_node);
                // Compute the shifted target code. This is tgt_code
                // shifted down by the difference between tgt_level
                // and src_level. For instance, in an octree,
                // if the target code is 1 000 000 001 000, then tgt_level
                // is 4, and, src_level is 2, then the shifted code
                // will be 1 000 000.
                const auto s_tgt_code = static_cast<UInt>(tgt_code >> ((tgt_level - src_level) * NDim));
                // Is the source node an ancestor of the target node? It is if the
                // shifted target code coincides with the source code.
                if (s_tgt_code == src_code) {
                    // If the source node is an ancestor of the target, then we continue the
                    // depth-first traversal.
                    ++src_idx;
                    continue;
                }
                // The source node is not an ancestor of the target. We need to run the BH criterion
                // check. The tree_acc_pot_bh_check() function will return the index of the next node
                // in the traversal.
                src_idx = tree_acc_pot_bh_check_hcc<Q, NDim>(tree_view, src_idx, theta2, eps2, pidx, p_view, res_view);
            }
        })
        .get();
    for (auto &v : t2a(rt)) {
        v.synchronize();
    }
} // namespace detail

// Explicit instantiations.
#define RAKAU_HC_EXPLICIT_INST(Q, NDim, F, UInt)                                                                       \
    template void acc_pot_impl_hcc<Q, NDim, F, UInt>(                                                                  \
        const std::array<F *, tree_nvecs_res<Q, NDim>> &, const tree_cnode_t<F, UInt> *, tree_size_t<F>,               \
        const tree_node_t<NDim, F, UInt> *, tree_size_t<F>, const std::array<const F *, NDim + 1u> &, tree_size_t<F>,  \
        F, F, F, tree_size_t<F>)

RAKAU_HC_EXPLICIT_INST(0, 3, float, std::size_t);
RAKAU_HC_EXPLICIT_INST(1, 3, float, std::size_t);
RAKAU_HC_EXPLICIT_INST(2, 3, float, std::size_t);

RAKAU_HC_EXPLICIT_INST(0, 3, double, std::size_t);
RAKAU_HC_EXPLICIT_INST(1, 3, double, std::size_t);
RAKAU_HC_EXPLICIT_INST(2, 3, double, std::size_t);

#undef RAKAU_HC_EXPLICIT_INST

} // namespace detail
} // namespace rakau
