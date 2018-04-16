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
        return ::m3D_e_sLUT<std::uint64_t, std::uint32_t>(x, y, z);
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

template <std::size_t NDim, typename PIt, typename CIt>
inline void get_particle_codes(const typename std::iterator_traits<PIt>::value_type &size, PIt begin, PIt end, CIt c_it)
{
    using code_t = typename std::iterator_traits<CIt>::value_type;
    using float_t = typename std::iterator_traits<PIt>::value_type;
    static_assert(get_cbits<code_t, NDim>() < std::numeric_limits<code_t>::digits);
    constexpr code_t factor = code_t(1) << get_cbits<code_t, NDim>();
    // Function to discretise the input floating-point coordinates starting at it
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
    for (auto it = begin; it != end; it += (NDim + 1u)) {
        tmp.emplace_back();
        tmp.back().first = me(disc_coords(it + 1).begin());
        std::copy(it, it + (NDim + 1u), tmp.back().second.begin());
    }
    // Now let's sort.
    boost::sort::spreadsort::integer_sort(
        tmp.begin(), tmp.end(), [](const auto &p, unsigned offset) { return static_cast<code_t>(p.first >> offset); },
        [](const auto &p1, const auto &p2) { return p1.first < p2.first; });
    // Copy over the data.
    for (const auto &p : tmp) {
        // Write the code to the output iter.
        *c_it = p.first;
        // Copy mass and coordinates.
        std::copy(p.second.begin(), p.second.end(), begin);
        // Increase the iterators.
        ++c_it;
        begin += (NDim + 1u);
    }
}

template <unsigned Level, std::size_t NDim, typename UInt,
          typename = std::integral_constant<bool, (get_cbits<UInt, NDim>() == Level)>>
struct tree_builder {
    template <typename Tree, typename PIt, typename CIt>
    void operator()(Tree &tree, PIt begin, PIt end, CIt c_it) const
    {
        // Shortcuts.
        using code_t = typename std::iterator_traits<CIt>::value_type;
        using float_t = typename std::iterator_traits<PIt>::value_type;
        constexpr auto cbits = get_cbits<UInt, NDim>();
        // Make sure UInt is the same as the particle code type.
        static_assert(std::is_same_v<code_t, UInt>);
        for (; begin != end; begin += (NDim + 1u), ++c_it) {
            *c_it >> (cbits * NDim - Level);
        }
    }
};

template <std::size_t NDim, typename PIt, typename CIt>
inline auto build_tree(PIt begin, PIt end, CIt c_it)
{
    using code_t = typename std::iterator_traits<CIt>::value_type;
    using float_t = typename std::iterator_traits<PIt>::value_type;
    std::pair<std::vector<code_t>, std::vector<float_t>> retval;
    if (begin == end) {
        // Special case the empty tree.
        return retval;
    }
    // Fill in the first node with the first particle.
    retval.first.emplace_back(1);
    static_assert(NDim < std::numeric_limits<std::size_t>::max());
    static_assert(size_iter_check<PIt>(NDim + 1u));
    retval.second.insert(retval.second.end(), begin, begin + (NDim + 1u));
    // Update the iterators.
    begin += (NDim + 1u), ++c_it;
    // Build the rest of the tree, starting from the root.
    tree_builder<0, NDim, code_t>{}(retval, begin, end, c_it);
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
static constexpr double bsize = 10.;

static std::mt19937 rng;

template <std::size_t NDim, typename F>
inline std::vector<F> get_uniform_particles(std::size_t n, F size)
{
    std::vector<F> retval(n * (NDim + 1u));
    // Mass distribution.
    std::uniform_real_distribution<F> mdist(F(0), F(1));
    // Positions distribution.
    std::uniform_real_distribution<F> rdist(-size / F(2), size / F(2));
    for (auto it = retval.begin(); it != retval.end(); it += (NDim + 1u)) {
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
        get_particle_codes<3>(bsize, parts.begin(), parts.end(), codes.begin());
    }
    {
        simple_timer st;
        auto tree = build_tree<3>(parts.begin(), parts.end(), codes.begin());
        std::cout << tree.first[0] << '\n';
        std::cout << tree.second[0] << '\n';
        std::cout << tree.second[3] << '\n';
    }
}
