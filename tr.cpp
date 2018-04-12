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
            // Verify that cbegin is actually starting at the beginning of the
            // current child node, which could also be the end in case the node is empty
            // (see the next comment for an explanation).
            assert(cbegin
                   == std::lower_bound(cbegin, cend,
                                       static_cast<code_t>((node_prefix << ((cbits - ParentLevel) * NDim))
                                                           + (i << ((cbits - ParentLevel - 1u) * NDim)))));
            // This call will determine the end of range containing the particles
            // belonging to the current child node. Here we are basically constructing the
            // code of the first possible particle in the *next* child node, which is made by
            // (looking from MSB to LSB):
            // - the current node prefix,
            // - i + 1,
            // - right-padding zeroes.
            // Note that if we are in the last iteration of the for loop, concatenating
            // the current node prefix with i + 1 will essentially bump up by one the node
            // prefix (meaning that the first possible particle in the next child node
            // actually belongs to the next parent node).
            const auto it_end
                = std::lower_bound(cbegin, cend,
                                   static_cast<code_t>((node_prefix << ((cbits - ParentLevel) * NDim))
                                                       + ((i + 1u) << ((cbits - ParentLevel - 1u) * NDim))));
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

static constexpr unsigned nparts = 60'000;
static constexpr double bsize = 10.;

template <std::size_t NDim>
struct node_comparer {
    template <typename UInt>
    bool operator()(UInt n1, UInt n2) const
    {
        constexpr unsigned cbits = get_cbits<UInt, NDim>();
        constexpr unsigned ndigits = std::numeric_limits<UInt>::digits;
        // TODO clzl, assert nonzero.
        assert(!((ndigits - 1u - unsigned(__builtin_clzl(n1))) % NDim));
        assert(!((ndigits - 1u - unsigned(__builtin_clzl(n2))) % NDim));
        const auto tl1 = static_cast<unsigned>((ndigits - 1u - unsigned(__builtin_clzl(n1))) / NDim);
        const auto tl2 = static_cast<unsigned>((ndigits - 1u - unsigned(__builtin_clzl(n2))) / NDim);
        assert(cbits >= tl1);
        assert(cbits >= tl2);
        assert((cbits - tl1) * NDim < ndigits);
        assert((cbits - tl2) * NDim < ndigits);
        const auto s_n1 = n1 << ((cbits - tl1) * NDim);
        const auto s_n2 = n2 << ((cbits - tl2) * NDim);
        return s_n1 < s_n2 || (s_n1 == s_n2 && tl1 < tl2);
    }
};

template <typename F, unsigned ParentLevel, std::size_t NDim, typename UInt>
struct particle_force {
    template <typename TreeIt>
    F operator()(double theta, UInt code, TreeIt begin, TreeIt end) const
    {
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
        // Now let's iterate over the sibling nodes.
        node_comparer<NDim> nc;
        for (UInt i = 0; i < (UInt(1) << NDim); ++i, ++sib_code) {
            // Try to locate the current sibling node.
            begin = std::lower_bound(begin, end, sib_code,
                                     [nc](const auto &t, const auto &n) { return nc(std::get<0>(t), n); });
            if (sib_code == part_node_code) {
                // Don't do anything if the current sibling is the particle
                // node itself.
                std::cout << "Skipping particle node :" << part_node_code << '\n';
                // NOTE: we *must* have located the node here, as this node
                // is the one in which the particle is residing.
                assert(begin != end);
                assert(std::get<0>(*begin) == sib_code);
                // Update begin to the next node (same as below).
                ++begin;
                continue;
            }
            if (begin != end && std::get<0>(*begin) == sib_code) {
                // We found a lower bound, and it corresponds
                // to the sibling node we were looking for.
                std::cout << "Sibling nodal code: " << std::bitset<64>(std::get<0>(*begin)) << '\n';
                // Since we found the node we were looking for,
                // we can start the search in the next iteration from the next
                // element of the tree. Note that we cannot be at end here,
                // so ++begin is well defined.
                ++begin;
            }
        }
        return F(theta);
    }
};

int main()
{
    std::cout.precision(40);
    auto parts = get_uniform_particles<3>(nparts, bsize);
    std::vector<std::uint64_t> codes(nparts);
    get_particle_codes(bsize, nparts, codes.begin(), parts.begin(), parts.begin() + nparts, parts.begin() + 2u * nparts,
                       parts.begin() + 3u * nparts);
    std::cout << "Done sorting.\n\n";
    std::cout.flush();
    std::cout << "Code: " << std::bitset<64>(codes.back()) << '\n';
    std::cout << "mass: " << *(parts.begin() + nparts - 1) << '\n';
    std::cout << "x coord: " << *(parts.begin() + 2 * nparts - 1) << '\n';
    std::cout << "y coord: " << *(parts.begin() + 3 * nparts - 1) << '\n';
    std::cout << "z coord: " << *(parts.begin() + 4 * nparts - 1) << '\n';
    std::cout.flush();
    std::vector<std::tuple<std::uint64_t, double, double, double, double>> tree;
    build_tree(tree, nparts, codes.cbegin(), parts.cbegin(), parts.cbegin() + nparts, parts.cbegin() + 2u * nparts,
               parts.cbegin() + 3u * nparts);
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
    const std::size_t pidx = nparts / 2;
    const auto code = codes[pidx];
    std::cout << "Selected particle code: " << std::bitset<64>(code) << '\n';
    particle_force<double, 0, 3, std::uint64_t>{}(.5, code, tree.begin(), tree.end());

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
