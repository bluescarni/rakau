#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/sort/spreadsort/spreadsort.hpp>

#include "libmorton/morton.h"

template <std::size_t NDim, typename Out>
struct morton_encoder {
};

template <>
struct morton_encoder<3, std::uint64_t> {
    template <typename It>
    std::uint64_t operator()(It it) const
    {
        const auto x = *it;
        const auto y = *(it + 1);
        const auto z = *(it + 2);
        assert(x < (1ul << 21));
        assert(y < (1ul << 21));
        assert(z < (1ul << 21));
        assert(!(::m3D_e_sLUT<std::uint64_t, std::uint32_t>(x, y, z) >> 63u));
        assert((::morton3D_64_encode(x, y, z) == ::m3D_e_sLUT<std::uint64_t, std::uint32_t>(x, y, z)));
        return ::m3D_e_sLUT<std::uint64_t, std::uint32_t>(std::uint32_t(x), std::uint32_t(y), std::uint32_t(z));
    }
};

template <>
struct morton_encoder<2, std::uint64_t> {
    std::uint64_t operator()(std::uint64_t x, std::uint64_t y) const
    {
        // TODO fix.
        assert(x < (1ul << 31));
        assert(y < (1ul << 31));
        assert(!(::morton2D_64_encode(x, y) >> 62u));
        return ::morton2D_64_encode(x, y);
    }
};

template <typename UInt, std::size_t NDim>
constexpr unsigned get_cbits()
{
    constexpr unsigned nbits = std::numeric_limits<UInt>::digits;
    static_assert(nbits > NDim, "The number of bits must be greater than the number of dimensions.");
    return static_cast<unsigned>(nbits / NDim - !(nbits % NDim));
}

// Check that the input UInt N is representable
// by the difference typeof the iterator type It.
template <typename It, typename UInt>
constexpr bool size_iter_check(UInt N)
{
    using diff_t = typename std::iterator_traits<It>::difference_type;
    using udiff_t = std::make_unsigned_t<diff_t>;
    return N <= static_cast<udiff_t>(std::numeric_limits<diff_t>::max());
}

template <std::size_t NDim, typename CIt, typename PIt>
inline void get_particle_codes(const typename std::iterator_traits<PIt>::value_type &size, CIt begin, CIt end, PIt p_it)
{
    using code_t = typename std::iterator_traits<CIt>::value_type;
    using float_t = typename std::iterator_traits<PIt>::value_type;
    static_assert(get_cbits<code_t, NDim>() < std::numeric_limits<code_t>::digits);
    constexpr code_t factor = code_t(1) << get_cbits<code_t, NDim>();
    // Function to discretise the input floating-point coordinates starting at 'it'
    // into a box of a given size.
    auto disc_coords = [&size](PIt it) {
        std::array<code_t, NDim> retval;
        for (std::size_t i = 0; i < NDim; ++i, ++it) {
            const auto &x = *it;
            // Translate and rescale the coordinate so that -size/2 becomes zero
            // and size/2 becomes 1.
            auto tmp = (x + size / float_t(2)) / size;
            // Rescale by factor.
            tmp *= factor;
            // Check: don't end up with a nonfinite value.
            if (!std::isfinite(tmp)) {
                throw std::invalid_argument("Not finite!");
            }
            // Check: don't end up outside the [0, factor) range.
            if (tmp < float_t(0) || tmp >= float_t(factor)) {
                throw std::invalid_argument("Out of bounds!");
            }
            // Cast to code_t and write to retval.
            retval[i] = static_cast<code_t>(tmp);
            // Last check, make sure we don't overflow.
            if (retval[i] >= factor) {
                throw std::invalid_argument("Out of bounds! (after cast)");
            }
        }
        return retval;
    };
    // Do the encoding. We will store code, mass and coordinates
    // in a temporary structure which we will use for sorting.
    static_assert(NDim < std::numeric_limits<std::size_t>::max());
    static thread_local std::vector<std::pair<code_t, std::array<float_t, NDim + 1u>>> tmp;
    tmp.resize(0);
    // The encoder object.
    morton_encoder<NDim, code_t> me;
    // Check that it's safe to add (NDim + 1u) to the mass/coords iterator.
    static_assert(size_iter_check<PIt>(NDim + 1u));
    auto p_it_copy = p_it;
    for (auto it = begin; it != end; ++it, p_it_copy += NDim + 1u) {
        tmp.emplace_back();
        tmp.back().first = me(disc_coords(p_it_copy + 1).begin());
        std::copy(p_it_copy, p_it_copy + (NDim + 1u), tmp.back().second.begin());
    }
    // Now let's sort.
    boost::sort::spreadsort::integer_sort(
        tmp.begin(), tmp.end(), [](const auto &p, unsigned offset) { return static_cast<code_t>(p.first >> offset); },
        [](const auto &p1, const auto &p2) { return p1.first < p2.first; });
    // Copy over the data.
    for (const auto &p : tmp) {
        // Write the code to the output iter.
        *begin = p.first;
        // Copy mass and coordinates.
        std::copy(p.second.begin(), p.second.end(), p_it);
        // Increase the iterators.
        ++begin;
        p_it += NDim + 1u;
    }
}

// Helper to increase all elements in a tuple.
template <typename T, std::size_t... N>
inline void increase_tuple_elements(T &tup, const std::index_sequence<N...> &)
{
    // TODO: overflow checking?
    (..., ++std::get<N>(tup));
}

// Small helper to get the tree level of a nodal code.
template <std::size_t NDim, typename UInt>
inline unsigned tree_level(UInt n)
{
    // TODO clzl wrappers.
#if !defined(NDEBUG)
    constexpr unsigned cbits = get_cbits<UInt, NDim>();
#endif
    constexpr unsigned ndigits = std::numeric_limits<UInt>::digits;
    assert(n);
    assert(!((ndigits - 1u - unsigned(__builtin_clzl(n))) % NDim));
    auto retval = static_cast<unsigned>((ndigits - 1u - unsigned(__builtin_clzl(n))) / NDim);
    assert(cbits >= retval);
    assert((cbits - retval) * NDim < ndigits);
    return retval;
}

// Small functor to compare nodal codes.
template <std::size_t NDim>
struct node_comparer {
    template <typename UInt>
    bool operator()(UInt n1, UInt n2) const
    {
        constexpr unsigned cbits = get_cbits<UInt, NDim>();
        const auto tl1 = tree_level<NDim>(n1);
        const auto tl2 = tree_level<NDim>(n2);
        const auto s_n1 = n1 << ((cbits - tl1) * NDim);
        const auto s_n2 = n2 << ((cbits - tl2) * NDim);
        return s_n1 < s_n2 || (s_n1 == s_n2 && tl1 < tl2);
    }
};

template <unsigned ParentLevel, std::size_t NDim, typename UInt,
          typename = std::integral_constant<bool, (get_cbits<UInt, NDim>() == ParentLevel)>>
struct tree_builder {
    template <typename Tree, typename CTuple, typename CIt, typename PIt>
    void operator()(Tree &tree, std::deque<UInt> &children_count, CTuple &ct, UInt parent_code, CIt begin, CIt end,
                    PIt p_it) const
    {
        // Shortcuts.
        using code_t = typename std::iterator_traits<CIt>::value_type;
        using float_t = typename std::iterator_traits<PIt>::value_type;
        constexpr auto cbits = get_cbits<UInt, NDim>();
        // Make sure UInt is the same as the particle code type.
        static_assert(std::is_same_v<code_t, UInt>);
        // We can have an empty parent node only at the root level. Otherwise,
        // if we are here we must have some particles in the node.
        assert(ParentLevel == 0u || begin != end);
        // On entry, the ranges [begin, end) and [p_it, ...) contain
        // the codes, masses and positions of all the particles
        // belonging to the parent node. parent_code is the nodal code
        // of the parent node.
        //
        // We want to iterate over the current children nodes
        // (of which there might be up to 2**NDim). A child exists if
        // it contains at least 1 particle. If it contains > 1 particles,
        // it is an internal (i.e., non-leaf) node and we go deeper. If it contains 1 particle,
        // it is a leaf node, we stop going deeper and move to its sibling.
        //
        // This is the node prefix: it is the nodal code of the parent without the most significant bit.
        const auto node_prefix = parent_code - (code_t(1) << (ParentLevel * NDim));
        // Temporary structure that we will use to compute the COM. It contains
        // the total mass and the position of the COM.
        std::array<float_t, NDim + 1u> com;
        for (code_t i = 0; i < (code_t(1) << NDim); ++i) {
            // Compute the first and last possible codes for the current child node.
            // They both start with (from MSB to LSB):
            // - current node prefix,
            // - i.
            // The first possible code is then right-padded with all zeroes, the last possible
            // code is right-padded with ones.
            const auto p_first = static_cast<code_t>((node_prefix << ((cbits - ParentLevel) * NDim))
                                                     + (i << ((cbits - ParentLevel - 1u) * NDim)));
            const auto p_last
                = static_cast<code_t>(p_first + ((code_t(1) << ((cbits - ParentLevel - 1u) * NDim)) - 1u));
            // Verify that begin contains the first value equal to or greater than p_first.
            assert(begin == std::lower_bound(begin, end, p_first));
            // Determine the end of the child node: it_end will point to the first value greater
            // than the largest possible code for the current child node.
            const auto it_end = std::upper_bound(begin, end, p_last);
            // Compute the number of particles.
            using part_count_t = decltype(std::distance(begin, it_end));
            const auto npart = std::distance(begin, it_end);
            // TODO runtime check instead of assert.
            assert(npart >= 0);
            if (npart) {
                // npart > 0, we have a node. Compute its nodal code by moving up the
                // parent nodal code by NDim and adding the current child node index i.
                const auto cur_code = static_cast<code_t>((parent_code << NDim) + i);
                // Init the COM with the properties of the first particle.
                auto it = p_it;
                com[0] = *it;
                for (std::size_t j = 1; j < NDim + 1u; ++j) {
                    com[j] = *it * *(it + j);
                }
                // Move to the second particle.
                it += NDim + 1u;
                // Do the rest.
                for (part_count_t i = 1; i < npart; ++i, it += NDim + 1u) {
                    // Update total mass.
                    com[0] += *it;
                    // Update the COM position.
                    for (std::size_t j = 1; j < NDim + 1u; ++j) {
                        com[j] += *it * *(it + j);
                    }
                }
                // Do the final division for the COM.
                for (std::size_t j = 1; j < NDim + 1u; ++j) {
                    com[j] /= com[0];
                }
                // Add the node to the tree.
                // NOTE: the children count in the node gets inited to zero, it will
                // be copied over from the deque later.
                tree.first.insert(tree.first.end(), {cur_code, code_t(0)});
                tree.second.insert(tree.second.end(), com.begin(), com.end());
                // Add a counter to the deque. The newly-added node has zero children initially.
                children_count.emplace_back(0);
                // Increase the children count of the parents.
                increase_tuple_elements(ct, std::make_index_sequence<std::tuple_size_v<CTuple>>{});
                if (npart > 1) {
                    // Add a new element to the children counter tuple, pointing to
                    // the value we just added to the deque.
                    auto new_ct = std::tuple_cat(ct, std::tie(children_count.back()));
                    // The node is an internal one, go deeper.
                    tree_builder<ParentLevel + 1u, NDim, UInt>{}(tree, children_count, new_ct, cur_code, begin, it_end,
                                                                 p_it);
                }
            }
            // Move to the next child node.
            // TODO: overflow checks?
            std::advance(begin, npart);
            std::advance(p_it, npart * (NDim + 1u));
        }
    }
};

template <unsigned ParentLevel, std::size_t NDim, typename UInt>
struct tree_builder<ParentLevel, NDim, UInt, std::true_type> {
    template <typename Tree, typename CTuple, typename CIt, typename PIt>
    void operator()(Tree &, std::deque<UInt> &, CTuple &, UInt, CIt, CIt, PIt) const
    {
    }
};

template <std::size_t NDim, typename CIt, typename PIt>
inline auto build_tree(CIt begin, CIt end, PIt p_it)
{
    using code_t = typename std::iterator_traits<CIt>::value_type;
    using float_t = typename std::iterator_traits<PIt>::value_type;
    std::pair<std::vector<code_t>, std::vector<float_t>> retval;
    // The structure to keep track of the children count. This
    // has to be a deque (rather than a vector) because we need
    // to keep references to elements in it as we grow it.
    static thread_local std::deque<code_t> children_count;
    children_count.resize(0);
    children_count.emplace_back(0);
    // The initial tuple containing the children counters for the parents
    // of a node.
    auto c_tuple = std::tie(children_count.back());
    // Depth-first tree construction starting from the root (level 0, nodal code 1).
    tree_builder<0, NDim, code_t>{}(retval, children_count, c_tuple, 1, begin, end, p_it);
    // Double check the return value.
    assert(!(retval.first.size() % 2u));
    assert(!(retval.second.size() % (NDim + 1u)));
    assert(retval.first.size() / 2u == retval.second.size() / (NDim + 1u));
    assert(retval.first.size() / 2u == children_count.size() - 1u);
    // Copy over the children count.
    auto c_it = retval.first.begin();
    for (auto it = children_count.begin() + 1; it != children_count.end(); ++it, c_it += 2) {
        *(c_it + 1) = *it;
    }
#if !defined(NDEBUG)
    // Check that the tree is sorted wrt the nodal code.
    if (retval.first.size()) {
        node_comparer<NDim> nc;
        for (decltype(retval.first.size()) i = 0; i < retval.first.size() - 2u; i += 2u) {
            assert(nc(retval.first[i], retval.first[i + 2u]));
        }
    }
#endif
    return retval;
}

template <unsigned ParentLevel, std::size_t NDim, typename UInt,
          typename = std::integral_constant<bool, (get_cbits<UInt, NDim>() == ParentLevel)>>
struct particle_acc {
    template <typename Tree, typename PIt>
    auto operator()(const Tree &tree, const typename std::iterator_traits<PIt>::value_type &dsize,
                    const typename std::iterator_traits<PIt>::value_type &theta, UInt code, UInt begin, UInt end,
                    PIt p_it) const
    {
        // Shortcuts.
        using float_t = typename std::iterator_traits<PIt>::value_type;
        constexpr unsigned cbits = get_cbits<UInt, NDim>();
        // This is the nodal code of the node in which the particle is at the current level.
        // We compute it via the following:
        // - add an extra 1 bit in the MSB direction,
        // - shift down the result depending on the current level.
        const auto part_node_code
            = static_cast<UInt>(((UInt(1) << (cbits * NDim)) + code) >> ((cbits - ParentLevel - 1u) * NDim));
        // This will eventually become the index pointing to the node
        // hosting the particle at the current level. It will be established
        // in the loop below.
        auto part_node_idx = end;
        // Init the return value.
        float_t retval(0);
        for (auto idx = begin; idx != end;
             // NOTE: when incrementing idx, we need to add 1 to the
             // total number of children in order to point to the next sibling.
             idx += tree.first[idx * 2u + 1u] + 1u) {
            // Get the nodal code of the current sibling.
            const auto cur_node_code = tree.first[idx * 2u];
            if (part_node_code == cur_node_code) {
                // We are in the sibling that contains the particle. Store
                // its index and don't do anything, just move to the next
                // sibling.
                part_node_idx = idx;
            } else {
                // Compute the acceleration from the current sibling node.
                retval += acc_from_node<ParentLevel>{}(tree, dsize, theta, idx, idx + tree.first[idx * 2u + 1u] + 1u,
                                                       p_it);
            }
        }
        // We must have set part_node_idx to something other than end
        // in the loop above.
        assert(part_node_idx != end);
        // Check if the node containing the particle at the current level is a leaf.
        if (!tree.first[part_node_idx * 2u + 1u]) {
            // The particle's node has no children.
            // TODO fixme for multiple particles in a leaf.
            return retval;
        }
        // Go one level deeper in the particle's current node. The new indices range must start from the position
        // immediately past part_node_idx (i.e., the first children node) and have a size equal to the number
        // of children.
        retval += particle_acc<ParentLevel + 1u, NDim, UInt>{}(tree, dsize, theta, code, part_node_idx + 1u,
                                                               part_node_idx + 1u + tree.first[part_node_idx * 2u + 1u],
                                                               p_it);
        return retval;
    }
    template <unsigned PLevel, typename = void>
    struct acc_from_node {
        template <typename Tree, typename PIt>
        auto operator()(const Tree &tree, const typename std::iterator_traits<PIt>::value_type &dsize,
                        const typename std::iterator_traits<PIt>::value_type &theta, UInt begin, UInt end,
                        PIt p_it) const
        {
            using float_t = typename std::iterator_traits<PIt>::value_type;
            // Determine the distance**2 between the particle and the COM of the node.
            float_t dist2(0);
            for (std::size_t j = 1; j < NDim + 1u; ++j) {
                // Curent node COM coordinate.
                const auto &node_x = tree.second[begin * (NDim + 1u) + j];
                dist2 += (*(p_it + j) - node_x) * (*(p_it + j) - node_x);
            }
            // Check if the current node is a leaf.
            if (end == begin + 1u) {
                // The current node has no children, as the next node is a sibling.
                // The acceleration is M / dist2.
                // TODO what happens if this leaf node has more than 1 particle?
                // Direct sum or just do the COM?
                return tree.second[begin * (NDim + 1u)] / dist2;
            }
            // Determine the size of the node.
            const auto node_size = dsize / (UInt(1) << (PLevel + 1u));
            // Check the BH acceptance criterion.
            if ((node_size * node_size) / dist2 < theta * theta) {
                // We can approximate the acceleration with the COM of the
                // current node.
                return tree.second[begin * (NDim + 1u)] / dist2;
            }
            // We need to go deeper in the tree.
            float_t retval(0);
            // We can now bump up the index because we know this node has at least 1 child.
            for (++begin; begin != end; begin += tree.first[begin * 2u + 1u] + 1u) {
                retval += acc_from_node<PLevel + 1u>{}(tree, dsize, theta, begin,
                                                       begin + tree.first[begin * 2u + 1u] + 1u, p_it);
            }
            return retval;
        }
    };
    template <typename T>
    struct acc_from_node<get_cbits<UInt, NDim>(), T> {
        template <typename Tree, typename PIt>
        auto operator()(const Tree &, const typename std::iterator_traits<PIt>::value_type &,
                        const typename std::iterator_traits<PIt>::value_type &, UInt, UInt, PIt) const
        {
            using float_t = typename std::iterator_traits<PIt>::value_type;
            return float_t(0);
        }
    };
};

template <unsigned ParentLevel, std::size_t NDim, typename UInt>
struct particle_acc<ParentLevel, NDim, UInt, std::true_type> {
    template <typename Tree, typename PIt>
    auto operator()(const Tree &, const typename std::iterator_traits<PIt>::value_type &,
                    const typename std::iterator_traits<PIt>::value_type &, UInt, UInt, UInt, PIt) const
    {
        using float_t = typename std::iterator_traits<PIt>::value_type;
        return float_t(0);
    }
};

#include <bitset>
#include <chrono>
#include <iostream>
#include <random>

class simple_timer
{
public:
    simple_timer() : m_start(std::chrono::high_resolution_clock::now()) {}
    double elapsed() const
    {
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - m_start)
                .count());
    }
    ~simple_timer()
    {
        std::cout << "Elapsed time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()
                                                                           - m_start)
                         .count()
                  << u8"μs\n";
    }

private:
    const std::chrono::high_resolution_clock::time_point m_start;
};

static constexpr std::size_t nparts = 600'000;
static constexpr auto bsize = 10.;

static std::mt19937 rng;

template <std::size_t NDim, typename F>
inline std::vector<F> get_uniform_particles(std::size_t n, F size)
{
    std::vector<F> retval(n * (NDim + 1u));
    // Mass distribution.
    std::uniform_real_distribution<F> mdist(F(0), F(1));
    // Positions distribution.
    std::uniform_real_distribution<F> rdist(-size / F(2), size / F(2));
    for (auto it = retval.begin(); it != retval.end(); it += NDim + 1u) {
        *it = mdist(rng);
        std::generate_n(it + 1, NDim, [&rdist]() { return rdist(rng); });
    }
    return retval;
}

int main()
{
    std::cout.precision(40);
    auto parts = get_uniform_particles<3>(nparts, bsize);
    std::vector<std::uint64_t> codes(nparts);
    {
        simple_timer st;
        get_particle_codes<3>(bsize, codes.begin(), codes.end(), parts.begin());
    }
    decltype(build_tree<3>(codes.begin(), codes.end(), parts.begin())) tree;
    {
        simple_timer st;
        tree = build_tree<3>(codes.begin(), codes.end(), parts.begin());
        std::cout << "Tree size: " << tree.first.size() << '\n';
        std::cout << tree.first[0] << '\n';
        std::cout << tree.first[1] << '\n';
        std::cout << tree.second[0] << '\n';
        std::cout << tree.second[1] << '\n';
        std::cout << tree.second[2] << '\n';
        std::cout << tree.second[3] << '\n';
    }
    {
        simple_timer st;
        double ret = 0;
        for (std::size_t i = 0; i < 1; ++i) {
            const auto code = codes[i];
            ret += particle_acc<0, 3, std::uint64_t>{}(tree, bsize, 0.1, code, 0, tree.first.size() / 2u,
                                                       parts.begin() + i * 4u);
        }
        std::cout << "Total: " << ret << '\n';
    }
    {
        simple_timer st;
        double ret = 0.;
        const auto x0 = parts[1];
        const auto y0 = parts[2];
        const auto z0 = parts[3];
        for (std::size_t i = 1; i < nparts; ++i) {
            const auto x = parts[i * 4 + 1];
            const auto y = parts[i * 4 + 2];
            const auto z = parts[i * 4 + 3];
            ret += parts[i * 4] / ((x0 - x) * (x0 - x) + (y0 - y) * (y0 - y) + (z0 - z) * (z0 - z));
        }
        std::cout << "Total: " << ret << '\n';
    }
}
