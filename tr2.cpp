#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
    // TODO reserve.
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

template <unsigned ParentLevel, std::size_t NDim, typename UInt,
          typename = std::integral_constant<bool, (get_cbits<UInt, NDim>() == ParentLevel)>>
struct tree_builder {
    template <typename Tree, typename CIt, typename PIt>
    void operator()(Tree &tree, UInt parent_code, CIt begin, CIt end, PIt p_it) const
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
                tree.first.emplace_back(cur_code);
                tree.second.insert(tree.second.end(), com.begin(), com.end());
                if (npart > 1) {
                    // The node is an internal one, go deeper.
                    tree_builder<ParentLevel + 1u, NDim, UInt>{}(tree, cur_code, begin, it_end, p_it);
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
    template <typename Tree, typename CIt, typename PIt>
    void operator()(Tree &, UInt, CIt, CIt, PIt) const
    {
    }
};

template <std::size_t NDim, typename CIt, typename PIt>
inline auto build_tree(CIt begin, CIt end, PIt p_it)
{
    using code_t = typename std::iterator_traits<CIt>::value_type;
    using float_t = typename std::iterator_traits<PIt>::value_type;
    std::pair<std::vector<code_t>, std::vector<float_t>> retval;
    // Depth-first tree construction starting from the root (level 0, nodal code 1).
    tree_builder<0, NDim, code_t>{}(retval, 1, begin, end, p_it);
    return retval;
}

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

static constexpr std::size_t nparts = 1'000'000;
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
    {
        simple_timer st;
        auto tree = build_tree<3>(codes.begin(), codes.end(), parts.begin());
        std::cout << "Tree size: " << tree.first.size() << '\n';
        std::cout << tree.first[0] << '\n';
        std::cout << tree.second[0] << '\n';
        std::cout << tree.second[1] << '\n';
        std::cout << tree.second[2] << '\n';
        std::cout << tree.second[3] << '\n';
    }
}
