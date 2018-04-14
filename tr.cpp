#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/sort/spreadsort/spreadsort.hpp>

#include "libmorton/morton.h"

#include <array>
#include <bitset>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>

template <std::size_t NDim, typename Out>
struct morton_encoder {
};

template <>
struct morton_encoder<3u, std::uint64_t> {
    std::uint64_t operator()(std::uint64_t x, std::uint64_t y, std::uint64_t z) const
    {
        assert(x < (1ul << 21));
        assert(y < (1ul << 21));
        assert(z < (1ul << 21));
        assert(!(::m3D_e_sLUT<std::uint64_t, std::uint32_t>(x, y, z) >> 63u));
        assert((::morton3D_64_encode(x, y, z) == ::m3D_e_sLUT<std::uint64_t, std::uint32_t>(x, y, z)));
        return ::m3D_e_sLUT<std::uint64_t, std::uint32_t>(x, y, z);
    }
};

template <>
struct morton_encoder<2u, std::uint64_t> {
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

template <typename F, typename PIt, typename MIt, typename... CIts>
inline void get_particle_codes(F size, std::size_t nparts, PIt p_it, MIt m_it, CIts... c_its)
{
    using code_t = typename std::iterator_traits<PIt>::value_type;
    constexpr std::size_t ndim = sizeof...(CIts);
    static_assert(get_cbits<code_t, ndim>() < std::numeric_limits<code_t>::digits);
    constexpr code_t factor = code_t(1) << get_cbits<code_t, ndim>();
    // Make a copy of the original mass/coord iterators.
    auto its_orig = std::make_tuple(m_it, c_its...);
    // Function to discretise the input floating-point coordinate x
    // into a box of a given size.
    auto disc_coord = [size](const auto &x) {
        // Translate and rescale the coordinate so that -size/2 becomes zero
        // and size/2 becomes 1.
        auto tmp = (x + size / F(2)) / size;
        // Rescale by factor.
        tmp *= factor;
        // Check: don't end up with a nonfinite value.
        if (!std::isfinite(tmp)) {
            throw std::invalid_argument("Not finite!");
        }
        // Check: don't end up outside the [0, factor) range.
        if (tmp < F(0) || tmp >= F(factor)) {
            throw std::invalid_argument("Out of bounds!");
        }
        // Cast to code_t.
        auto retval = static_cast<code_t>(tmp);
        // Last check, make sure we don't overflow.
        if (retval >= factor) {
            throw std::invalid_argument("Out of bounds! (after cast)");
        }
        return retval;
    };
    // Do the encoding. We will store code, mass and coordinates
    // in a temporary structure which we will use for sorting.
    static thread_local std::vector<std::pair<code_t, decltype(std::make_tuple(*m_it, *c_its...))>> tmp;
    tmp.resize(nparts);
    // The encoder object.
    morton_encoder<ndim, code_t> me;
    for (auto &p : tmp) {
        // Compute the code.
        p.first = me(disc_coord(*c_its)...);
        // Store the mass and the real coordinates.
        p.second = std::tie(*m_it, *c_its...);
        // Increase the other iterators.
        ++m_it, (..., ++c_its);
    }
    // Now let's sort.
    boost::sort::spreadsort::integer_sort(
        tmp.begin(), tmp.end(), [](const auto &p, unsigned offset) { return static_cast<code_t>(p.first >> offset); },
        [](const auto &p1, const auto &p2) { return p1.first < p2.first; });
    // Reset the original iterators.
    std::tie(m_it, c_its...) = its_orig;
    // Copy over the data.
    for (const auto &p : tmp) {
        *p_it = p.first;
        std::tie(*m_it, *c_its...) = p.second;
        // Increase the iterators.
        ++p_it, ++m_it, (..., ++c_its);
    }
}

template <unsigned ParentLevel, std::size_t NDim, typename UInt,
          typename = std::integral_constant<bool, (get_cbits<UInt, NDim>() == ParentLevel)>>
struct tree_builder {
    template <typename Tree, typename PIt, typename MIt, typename... CIts>
    void operator()(Tree &tree, UInt parent_code, PIt cbegin, PIt cend, MIt m_it, CIts... c_its) const
    {
        // Make sure NDim is consistent.
        static_assert(NDim == sizeof...(CIts));
        // Shortcuts.
        using code_t = typename std::iterator_traits<PIt>::value_type;
        using mass_t = typename std::iterator_traits<MIt>::value_type;
        constexpr auto cbits = get_cbits<UInt, NDim>();
        // Make sure UInt is the same as the particle code type.
        static_assert(std::is_same_v<code_t, UInt>);
        // We can have an empty parent node only at the root level. Otherwise,
        // if we are here we must have some particles in the node.
        assert(ParentLevel == 0u || cbegin != cend);
        // On entry, the range [cbegin, cend) contains the codes of all the particles
        // belonging to the parent node (the particles' properties are
        // encoded in the other iterators). parent_code is the nodal code
        // of the parent node.
        //
        // We want to iterate over the current children nodes
        // (of which there might be up to 2**NDim). A child exists if
        // it contains at least 1 particle. If it contains > 1 particles,
        // it is an internal (i.e., non-leaf) node and we go deeper. If it contains 1 particle,
        // it is a leaf node, we stop going deeper and move to its sibling.
        //
        // This is the node prefix: it is the nodal code without the most significant bit.
        const auto node_prefix = parent_code - (code_t(1) << (ParentLevel * NDim));
        for (code_t i = 0; i < (code_t(1) << NDim); ++i) {
            // Compute the first and last possible codes for the current child node.
            // They both start with (from MSB to LSB):
            // - current node prefix,
            // - i.
            // The first possible code then contains all zeros, the last possible
            // codes contains all ones.
            const auto p_first = static_cast<code_t>((node_prefix << ((cbits - ParentLevel) * NDim))
                                                     + (i << ((cbits - ParentLevel - 1u) * NDim)));
            const auto p_last
                = static_cast<code_t>(p_first + ((code_t(1) << ((cbits - ParentLevel - 1u) * NDim)) - 1u));
            // Verify that cbegin contains the first value equal to or greater than p_first.
            assert(cbegin == std::lower_bound(cbegin, cend, p_first));
            // Determine the end of the child node: it_end will point to the first value greater
            // than the largest possible code for the current child node.
            const auto it_end = std::upper_bound(cbegin, cend, p_last);
            // Compute the number of particles.
            const auto npart = std::distance(cbegin, it_end);
            if (npart) {
                // npart > 0, we have a node. Compute its nodal code by moving up the
                // parent nodal code by NDim and adding the current child node index i.
                const auto cur_code = static_cast<code_t>((parent_code << NDim) + i);
                // Compute the total mass in the node.
                const auto M = std::accumulate(m_it, m_it + npart, mass_t(0));
                // Compute the COM of the node, and add the node to the tree.
                tree.emplace_back(
                    cur_code, M,
                    std::inner_product(m_it, m_it + npart, c_its, static_cast<decltype(*c_its * mass_t(0))>(0)) / M...);
                if (npart > 1) {
                    // The node is an internal one, go deeper.
                    tree_builder<ParentLevel + 1u, NDim, UInt>{}(tree, cur_code, cbegin, it_end, m_it, c_its...);
                }
            }
            // Move to the next child node.
            std::advance(cbegin, npart);
            std::advance(m_it, npart);
            (..., std::advance(c_its, npart));
        }
    }
};

template <unsigned ParentLevel, std::size_t NDim, typename UInt>
struct tree_builder<ParentLevel, NDim, UInt, std::true_type> {
    template <typename Tree, typename PIt, typename MIt, typename... CIts>
    void operator()(Tree &, UInt, PIt, PIt, MIt, CIts...) const
    {
    }
};

template <typename Tree, typename PIt, typename MIt, typename... CIts>
inline void build_tree(Tree &tree, std::size_t nparts, PIt p_it, MIt m_it, CIts... c_its)
{
    using code_t = typename std::iterator_traits<PIt>::value_type;
    // Depth-first tree construction starting from the root (level 0, nodal code 1).
    tree_builder<0, sizeof...(CIts), code_t>{}(tree, code_t(1), p_it, p_it + nparts, m_it, c_its...);
}

static std::mt19937 rng;

template <std::size_t D, typename F>
inline std::vector<F> get_uniform_particles(std::size_t n, F size)
{
    std::vector<F> retval(n * (D + 1u));
    // Mass.
    std::uniform_real_distribution<F> mdist(F(0), F(1));
    std::generate(retval.begin(), retval.begin() + n, [&mdist]() { return mdist(rng); });
    // Positions.
    std::uniform_real_distribution<F> rdist(-size / F(2), size / F(2));
    std::generate(retval.begin() + n, retval.end(), [&rdist]() { return rdist(rng); });
    return retval;
}

static constexpr unsigned nparts = 1'000'000;
static constexpr double bsize = 10.;

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
struct particle_acc {
    // Some metaprogramming to establish the mass, radius**2 and acceleration types.
    template <typename TreeIt>
    using mass_t = decltype(std::get<1>(*std::declval<const TreeIt &>()));
    template <typename... Coords>
    using radius2_t = decltype((... + (std::declval<const Coords &>() * std::declval<const Coords &>())));
    template <typename TreeIt, typename... Coords>
    using acc_t = decltype(std::declval<const mass_t<TreeIt> &>() / std::declval<const radius2_t<Coords...> &>());
    template <typename F, typename TreeIt, typename... Coords>
    acc_t<TreeIt, Coords...> operator()(const F &dsize, double theta, UInt code, TreeIt begin, TreeIt end,
                                        const Coords &... coords) const
    {
        // Verify consistency of dimensions.
        static_assert(sizeof...(Coords) == NDim);
        static_assert(std::tuple_size_v<std::remove_reference_t<decltype(*begin)>> == NDim + 2u);
        constexpr unsigned cbits = get_cbits<UInt, NDim>();
        // This is the nodal code of the node in which the particle is at the current level.
        // We compute it via the following:
        // - add an extra 1 bit in the MSB direction,
        // - shift down the result depending on the current level.
        const auto part_node_code
            = static_cast<UInt>(((UInt(1) << (cbits * NDim)) + code) >> ((cbits - ParentLevel - 1u) * NDim));
        // This is part_node_code with the least significant NDim bits zeroed out.
        // We will use it to locate sibling nodes.
        auto sib_code = static_cast<UInt>(part_node_code & ~((UInt(1) << NDim) - 1u));
        // This will eventually become the iterator pointing to the node
        // hosting the particle at the current level. It will be established
        // in the loop below.
        auto part_node_it = end;
        // Init the return value.
        acc_t<TreeIt, Coords...> retval(0);
        // Now let's iterate over the sibling nodes.
        node_comparer<NDim> nc;
        for (UInt i = 0; i < (UInt(1) << NDim); ++i, ++sib_code) {
            // Try to locate the current sibling node.
            begin = std::lower_bound(begin, end, sib_code,
                                     [nc](const auto &t, const auto &n) { return nc(std::get<0>(t), n); });
            if (sib_code == part_node_code) {
                // Don't do anything if the current sibling is the particle
                // node itself.
                // std::cout << "Skipping particle node :" << part_node_code << '\n';
                // NOTE: we *must* have located the node here, as this node
                // is the one in which the particle is residing.
                assert(begin != end);
                assert(std::get<0>(*begin) == sib_code);
                // Record the iterator.
                part_node_it = begin;
                // Update begin to the next node (same as below).
                ++begin;
                continue;
            }
            if (begin != end && std::get<0>(*begin) == sib_code) {
                // We found a lower bound, and it corresponds
                // to the sibling node we are looking for.
                // std::cout << "Sibling nodal code: " << std::bitset<64>(std::get<0>(*begin)) << '\n';
                retval += acc_from_node<ParentLevel>{}(dsize, theta, begin, end, coords...);
                // Since we found the node we were looking for,
                // we can start the search in the next iteration from the next
                // element of the tree. Note that we cannot be at end here,
                // so ++begin is well defined.
                ++begin;
            }
        }
        assert(part_node_it != end);
        // Check if the node containing the particle at the current level is a leaf.
        if (part_node_it == end
            || tree_level<NDim>(std::get<0>(*part_node_it)) >= tree_level<NDim>(std::get<0>(*(part_node_it + 1)))) {
            // TODO fixme for multiple particles in a leaf.
            return retval;
        }
        // Go one level deeper in the particle's current node.
        retval += particle_acc<ParentLevel + 1u, NDim, UInt>{}(dsize, theta, code, part_node_it, end, coords...);
        return retval;
    }
    // Euclidean distance**2 between a particle (with coordinates in the tuple t1) and the COM of a node
    // (with coordinates starting from index 2 of the tuple t2).
    template <typename T1, typename T2, typename std::size_t... I>
    static auto tuple_dist2(const T1 &t1, const T2 &t2, const std::index_sequence<I...> &)
    {
        return (... + ((std::get<I>(t1) - std::get<I + 2u>(t2)) * (std::get<I>(t1) - std::get<I + 2u>(t2))));
    }
    template <unsigned PLevel, typename = void>
    struct acc_from_node {
        template <typename F, typename TreeIt, typename... Coords>
        acc_t<TreeIt, Coords...> operator()(const F &dsize, double theta, TreeIt node_it, TreeIt end,
                                            const Coords &... coords) const
        {
            // Determine the distance**2 between the particle and the COM of the node.
            const auto dist2
                = tuple_dist2(std::tie(coords...), *node_it, std::make_index_sequence<sizeof...(Coords)>{});
            if (node_it + 1 == end
                || tree_level<NDim>(std::get<0>(*node_it)) >= tree_level<NDim>(std::get<0>(*(node_it + 1)))) {
                // This is either the last node of the tree, or its tree level is not less than the tree level
                // of the next node: a leaf node.
                // std::cout << "Leaf node.\n";
                // The acceleration is M / r**2.
                return std::get<1>(*node_it) / dist2;
            }
            // std::cout << "Internal node.\n";
            // Determine the size of the node.
            const auto node_size = dsize / (UInt(1) << (PLevel + 1u));
            // std::cout << "Node size: " << node_size << '\n';
            // Check the BH acceptance criterion.
            if ((node_size * node_size) / dist2 < theta * theta) {
                // We can approximate the acceleration with the COM of the
                // current node.
                return std::get<1>(*node_it) / dist2;
            }
            // We need to go deeper in the tree.
            acc_t<TreeIt, Coords...> retval(0);
            // Compute the first possible code of a child node.
            auto child_code = static_cast<UInt>(std::get<0>(*node_it) << NDim);
            // We can now bump up the node iterator because we know this node has at least 1 child.
            ++node_it;
            node_comparer<NDim> nc;
            for (UInt i = 0; i < (UInt(1) << NDim); ++i, ++child_code) {
                // Try to locate the current child.
                node_it = std::lower_bound(node_it, end, child_code,
                                           [nc](const auto &t, const auto &n) { return nc(std::get<0>(t), n); });
                if (node_it != end && std::get<0>(*node_it) == child_code) {
                    // We found a child node, let's proceed to the computation of the force.
                    retval += acc_from_node<PLevel + 1u>{}(dsize, theta, node_it, end, coords...);
                }
            }
            return retval;
        }
    };
    template <typename T>
    struct acc_from_node<get_cbits<UInt, NDim>(), T> {
        template <typename F, typename TreeIt, typename... Coords>
        acc_t<TreeIt, Coords...> operator()(const F &, double, TreeIt, TreeIt, const Coords &...) const
        {
            return acc_t<TreeIt, Coords...>(0);
        }
    };
};

template <unsigned ParentLevel, std::size_t NDim, typename UInt>
struct particle_acc<ParentLevel, NDim, UInt, std::true_type> {
    // Some metaprogramming to establish the mass, radius**2 and acceleration types.
    template <typename TreeIt>
    using mass_t = decltype(std::get<1>(*std::declval<const TreeIt &>()));
    template <typename... Coords>
    using radius2_t = decltype((... + (std::declval<const Coords &>() * std::declval<const Coords &>())));
    template <typename TreeIt, typename... Coords>
    using acc_t = decltype(std::declval<const mass_t<TreeIt> &>() / std::declval<const radius2_t<Coords...> &>());
    template <typename F, typename TreeIt, typename... Coords>
    acc_t<TreeIt, Coords...> operator()(const F &, double, UInt, TreeIt, TreeIt, const Coords &...) const
    {
        return acc_t<TreeIt, Coords...>(0);
    }
};

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

int main()
{
    std::cout.precision(40);
    auto parts = get_uniform_particles<3>(nparts, bsize);
    std::vector<std::uint64_t> codes(nparts);
    {
        simple_timer st;
        get_particle_codes(bsize, nparts, codes.begin(), parts.begin(), parts.begin() + nparts,
                           parts.begin() + 2u * nparts, parts.begin() + 3u * nparts);
    }
    std::cout << "Done sorting.\n\n";
    std::cout.flush();
    std::cout << "Code: " << std::bitset<64>(codes.back()) << '\n';
    std::cout << "mass: " << *(parts.begin() + nparts - 1) << '\n';
    std::cout << "x coord: " << *(parts.begin() + 2 * nparts - 1) << '\n';
    std::cout << "y coord: " << *(parts.begin() + 3 * nparts - 1) << '\n';
    std::cout << "z coord: " << *(parts.begin() + 4 * nparts - 1) << '\n';
    std::cout.flush();
    std::vector<std::tuple<std::uint64_t, double, double, double, double>> tree;
    {
        simple_timer st;
        build_tree(tree, nparts, codes.cbegin(), parts.cbegin(), parts.cbegin() + nparts, parts.cbegin() + 2u * nparts,
                   parts.cbegin() + 3u * nparts);
    }
    std::cout << "Tree size: " << tree.size() << '\n';
    std::cout << "First few: \n";
    for (auto it = tree.cbegin(); it != tree.cbegin() + 10; ++it) {
        std::cout << std::bitset<64>(std::get<0>(*it)) << '\n';
    }
    std::cout << "....\n\n";
    std::cout << "Tree sorted: " << std::is_sorted(tree.begin(), tree.end(), [](const auto &n1, const auto &n2) {
        return node_comparer<3>{}(std::get<0>(n1), std::get<0>(n2));
    }) << '\n';

    // Let's pick a "random" particle.
    const std::size_t pidx = nparts / 3;
    const auto code = codes[pidx];
    std::cout << "Selected particle code      : " << std::bitset<64>(code) << '\n';
    std::cout << "Next Selected particle code : " << std::bitset<64>(codes[pidx + 1u]) << '\n';
    std::cout << "Next2 Selected particle code: " << std::bitset<64>(codes[pidx + 2u]) << '\n';
    {
        simple_timer st;
        std::cout << particle_acc<0, 3, std::uint64_t>{}(
                         bsize, .75, code, tree.begin(), tree.end(), *(parts.begin() + nparts + pidx),
                         *(parts.begin() + nparts * 2 + pidx), *(parts.begin() + nparts * 3 + pidx))
                  << '\n';
    }
    {
        simple_timer st;
        double tot_acc = 0.;
        const auto x0 = *(parts.begin() + nparts + pidx);
        const auto y0 = *(parts.begin() + 2 * nparts + pidx);
        const auto z0 = *(parts.begin() + 3 * nparts + pidx);
        for (std::size_t i = 0; i < nparts; ++i) {
            if (i == pidx) {
                continue;
            }
            const auto m = *(parts.begin() + i);
            const auto x = *(parts.begin() + nparts + i);
            const auto y = *(parts.begin() + 2 * nparts + i);
            const auto z = *(parts.begin() + 3 * nparts + i);
            tot_acc += m / ((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));
        }
        std::cout << tot_acc << '\n';
    }

#if 0
    // Manual.
    // Let's pick a "random" particle.
    const std::size_t pidx = nparts / 2;
    const auto code = std::get<0>(tree[pidx]);
    std::cout << "Code:" << std::bitset<64>(code) << '\n';
    auto get_tl = [](auto c) {
        return (unsigned(std::numeric_limits<std::uint64_t>::digits) - 1u - unsigned(__builtin_clzl(c))) / 3u;
    };
    const auto tl = get_tl(code);
    std::cout << "Tree level: " << tl << '\n';
    const auto f_oct = ((code >> ((tl - 1u) * 3u))) & 7u;
    std::cout << "First octant: " << std::bitset<3>(f_oct) << '\n';
    auto begin = tree.begin();
    auto end = tree.end();
    for (auto i = 0u; i < 8u; ++i) {
        if (i == f_oct) {
            continue;
        }
        auto it = std::lower_bound(begin, end, i, [get_tl](const auto &t, const auto b) {
            const auto c = std::get<0>(t);
            const auto tl = get_tl(c);
            return ((c >> ((tl - 1u) * 3u)) & 7u) < b;
        });
        std::cout << std::bitset<64>(std::get<0>(*it)) << '\n';
        std::cout << std::get<1>(*it) << '\n';
    }
#endif
}
