#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/sort/spreadsort/spreadsort.hpp>

#include "libmorton/morton.h"

inline namespace detail
{

class simple_timer
{
public:
    simple_timer(const std::string &desc) : m_desc(desc), m_start(std::chrono::high_resolution_clock::now()) {}
    double elapsed() const
    {
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - m_start)
                .count());
    }
    ~simple_timer()
    {
        std::cout << "Elapsed time for '" + m_desc + "': "
                  << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()
                                                                           - m_start)
                         .count()
                  << u8"μs\n";
    }

private:
    const std::string m_desc;
    const std::chrono::high_resolution_clock::time_point m_start;
};

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

template <typename UInt, std::size_t NDim>
inline constexpr unsigned cbits_v = get_cbits<UInt, NDim>();

template <typename T, std::size_t... N>
inline void increase_children_count(T &tup, const std::index_sequence<N...> &)
{
    auto inc = [](auto &n) {
        if (n == std::numeric_limits<std::remove_reference_t<decltype(n)>>::max()) {
            throw std::overflow_error(
                "overflow error when incrementing the children count during the construction of a tree");
        }
        ++n;
    };
    (void)inc;
    (..., inc(std::get<N>(tup)));
}

// Small helper to get the tree level of a nodal code.
template <std::size_t NDim, typename UInt>
inline unsigned tree_level(UInt n)
{
    // TODO clzl wrappers.
#if !defined(NDEBUG)
    constexpr unsigned cbits = cbits_v<UInt, NDim>;
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
        constexpr unsigned cbits = cbits_v<UInt, NDim>;
        const auto tl1 = tree_level<NDim>(n1);
        const auto tl2 = tree_level<NDim>(n2);
        const auto s_n1 = n1 << ((cbits - tl1) * NDim);
        const auto s_n2 = n2 << ((cbits - tl2) * NDim);
        return s_n1 < s_n2 || (s_n1 == s_n2 && tl1 < tl2);
    }
};

template <unsigned ParentLevel, std::size_t NDim, typename UInt, typename F,
          typename = std::integral_constant<bool, (cbits_v<UInt, NDim> == ParentLevel)>>
struct tree_builder {
    template <typename Tree, typename Deque, typename CTuple, typename CIt, typename SizeType>
    void operator()(Tree &tree, Deque &children_count, CTuple &ct, UInt parent_code, CIt begin, CIt end, CIt top,
                    SizeType max_leaf_n) const
    {
        // Shortcut.
        constexpr auto cbits = cbits_v<UInt, NDim>;
        // We should never be invoking this on an empty range.
        assert(begin != end);
        // Double check max_leaf_n.
        assert(max_leaf_n);
        // On entry, the range [begin, end) contains the codes
        // of all the particles belonging to the parent node.
        // parent_code is the nodal code of the parent node.
        // top is the beginning of the full list of codes (used
        // to compute the offsets of the code iterators).
        // max_leaf_n is the maximum number of particles per leaf.
        //
        // We want to iterate over the current children nodes
        // (of which there might be up to 2**NDim). A child exists if
        // it contains at least 1 particle. If it contains > max_leaf_n particles,
        // it is an internal (i.e., non-leaf) node and we go deeper. If it contains <= max_leaf_n
        // particles, it is a leaf node, we stop going deeper and move to its sibling.
        //
        // This is the node prefix: it is the nodal code of the parent without the most significant bit.
        const auto node_prefix = parent_code - (UInt(1) << (ParentLevel * NDim));
        for (UInt i = 0; i < (UInt(1) << NDim); ++i) {
            // Compute the first and last possible codes for the current child node.
            // They both start with (from MSB to LSB):
            // - current node prefix,
            // - i.
            // The first possible code is then right-padded with all zeroes, the last possible
            // code is right-padded with ones.
            const auto p_first = static_cast<UInt>((node_prefix << ((cbits - ParentLevel) * NDim))
                                                   + (i << ((cbits - ParentLevel - 1u) * NDim)));
            const auto p_last = static_cast<UInt>(p_first + ((UInt(1) << ((cbits - ParentLevel - 1u) * NDim)) - 1u));
            // Verify that begin contains the first value equal to or greater than p_first.
            assert(begin == std::lower_bound(begin, end, p_first));
            // Determine the end of the child node: it_end will point to the first value greater
            // than the largest possible code for the current child node.
            const auto it_end = std::upper_bound(begin, end, p_last);
            // Compute the number of particles.
            const auto npart = std::distance(begin, it_end);
            assert(npart >= 0);
            if (npart) {
                // npart > 0, we have a node. Compute its nodal code by moving up the
                // parent nodal code by NDim and adding the current child node index i.
                const auto cur_code = static_cast<UInt>((parent_code << NDim) + i);
                // Add the node to the tree.
                tree.emplace_back(cur_code,
                                  std::array<SizeType, 3>{static_cast<SizeType>(std::distance(top, begin)),
                                                          static_cast<SizeType>(std::distance(top, it_end)),
                                                          // NOTE: the children count gets inited to zero. It will
                                                          // be filled in later.
                                                          SizeType(0)},
                                  // NOTE: make sure mass and coords are initialised in a known state (i.e., zero for
                                  // C++ floating-point).
                                  0, std::array<F, NDim>{});
                // Add a counter to the deque. The newly-added node has zero children initially.
                children_count.emplace_back(0);
                // Increase the children count of the parents.
                increase_children_count(ct, std::make_index_sequence<std::tuple_size_v<CTuple>>{});
                if (static_cast<std::make_unsigned_t<decltype(std::distance(begin, it_end))>>(npart) > max_leaf_n) {
                    // Add a new element to the children counter tuple, pointing to
                    // the value we just added to the deque.
                    auto new_ct = std::tuple_cat(ct, std::tie(children_count.back()));
                    // The node is an internal one, go deeper.
                    tree_builder<ParentLevel + 1u, NDim, UInt, F>{}(tree, children_count, new_ct, cur_code, begin,
                                                                    it_end, top, max_leaf_n);
                }
            }
            // Move to the next child node.
            begin += npart;
        }
    }
};

template <unsigned ParentLevel, std::size_t NDim, typename UInt, typename F>
struct tree_builder<ParentLevel, NDim, UInt, F, std::true_type> {
    template <typename Tree, typename Deque, typename CTuple, typename CIt, typename SizeType>
    void operator()(Tree &, Deque &, CTuple &, UInt, CIt, CIt, CIt, SizeType) const
    {
    }
};

template <unsigned ParentLevel, std::size_t NDim, typename UInt,
          typename = std::integral_constant<bool, (cbits_v<UInt, NDim> == ParentLevel)>>
struct particle_acc {
    template <typename Tree, typename PIt>
    auto operator()(const Tree &tree, const typename std::iterator_traits<PIt>::value_type &dsize,
                    const typename std::iterator_traits<PIt>::value_type &theta2, UInt code, UInt begin, UInt end,
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
                retval += acc_from_node<ParentLevel>{}(tree, dsize, theta2, idx, idx + tree.first[idx * 2u + 1u] + 1u,
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
        retval += particle_acc<ParentLevel + 1u, NDim, UInt>{}(tree, dsize, theta2, code, part_node_idx + 1u,
                                                               part_node_idx + 1u + tree.first[part_node_idx * 2u + 1u],
                                                               p_it);
        return retval;
    }
    template <unsigned PLevel, typename = void>
    struct acc_from_node {
        template <typename Tree, typename PIt>
        auto operator()(const Tree &tree, const typename std::iterator_traits<PIt>::value_type &dsize,
                        const typename std::iterator_traits<PIt>::value_type &theta2, UInt begin, UInt end,
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
            if ((node_size * node_size) / dist2 < theta2) {
                // We can approximate the acceleration with the COM of the
                // current node.
                return tree.second[begin * (NDim + 1u)] / dist2;
            }
            // We need to go deeper in the tree.
            float_t retval(0);
            // We can now bump up the index because we know this node has at least 1 child.
            for (++begin; begin != end; begin += tree.first[begin * 2u + 1u] + 1u) {
                retval += acc_from_node<PLevel + 1u>{}(tree, dsize, theta2, begin,
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

} // namespace detail

template <typename UInt, typename F, std::size_t NDim>
class tree
{
    // cbits shortcut.
    static constexpr unsigned cbits = cbits_v<UInt, NDim>;
    // Main vector type.
    template <typename T>
    using v_type = std::vector<T>;

public:
    using size_type = typename v_type<F>::size_type;

private:
    static auto &get_v_ind()
    {
        static thread_local v_type<std::pair<UInt, size_type>> v_ind;
        return v_ind;
    }
    void build_tree()
    {
        simple_timer st("node building");
        // NOTE: in the tree builder code, we will be moving around in the codes
        // vector using random access iterators. Thus, we must ensure the difference
        // type of the iterator can represent the size of the codes vector.
        using it_diff_t = typename std::iterator_traits<decltype(m_codes.begin())>::difference_type;
        using it_udiff_t = std::make_unsigned_t<it_diff_t>;
        if (m_codes.size() > static_cast<it_udiff_t>(std::numeric_limits<it_diff_t>::max())) {
            throw std::overflow_error("the number of particles (" + std::to_string(m_codes.size())
                                      + ") is too large, and it results in an overflow condition");
        }
        // The structure to keep track of the children count. This
        // has to be a deque (rather than a vector) because we need
        // to keep references to elements in it as we grow it.
        static thread_local std::deque<UInt> children_count;
        children_count.resize(0);
        // The initial tuple containing the children counters for the parents
        // of a node.
        auto c_tuple = std::tie();
        tree_builder<0, NDim, UInt, F>{}(m_tree, children_count, c_tuple, 1, m_codes.begin(), m_codes.end(),
                                         m_codes.begin(), m_max_leaf_n);
        // Check the result.
        assert(children_count.size() == m_tree.size());
        // Check the tree is sorted according to the nodal code comparison.
        assert(std::is_sorted(m_tree.begin(), m_tree.end(), [](const auto &t1, const auto &t2) {
            node_comparer<NDim> nc;
            return nc(std::get<0>(t1), std::get<0>(t2));
        }));
        // Check that all the nodes contain at least 1 element.
        assert(std::all_of(m_tree.begin(), m_tree.end(),
                           [](const auto &tup) { return std::get<1>(tup)[1] > std::get<1>(tup)[0]; }));
        // Copy over the children count.
        auto tree_it = m_tree.begin();
        for (auto it = children_count.begin(); it != children_count.end(); ++it, ++tree_it) {
            std::get<1>(*tree_it)[2] = *it;
        }
    }
    void build_tree_properties()
    {
        simple_timer st("tree properties");
        for (auto &tup : m_tree) {
            // Get the indices and the size for the current node.
            const auto begin = std::get<1>(tup)[0];
            const auto end = std::get<1>(tup)[1];
            assert(end > begin);
            const auto size = end - begin;
            // Compute the total mass.
            const auto tot_mass = std::accumulate(m_masses.data() + begin, m_masses.data() + end, F(0));
            // Compute the COM for the coordinates.
            for (std::size_t j = 0; j < NDim; ++j) {
                F acc(0);
                auto m_ptr = m_masses.data() + begin;
                auto c_ptr = m_coords[j].data() + begin;
                for (std::remove_const_t<decltype(size)> i = 0; i < size; ++i) {
                    acc += m_ptr[i] * c_ptr[i];
                }
                std::get<3>(tup)[j] = acc / tot_mass;
            }
            // Store the total mass.
            std::get<2>(tup) = tot_mass;
        }
    }

public:
    template <typename It>
    explicit tree(const F &box_size, It m_it, std::array<It, NDim> c_it, size_type N, size_type max_leaf_n)
        : m_box_size(box_size), m_max_leaf_n(max_leaf_n)
    {
        simple_timer st("overall tree construction");
        // Check the box size.
        if (!std::isfinite(box_size) || box_size <= F(0)) {
            throw std::invalid_argument("the box size must be a finite positive value, but it is "
                                        + std::to_string(box_size) + " instead");
        }
        // Check the max_leaf_n param.
        if (!max_leaf_n) {
            throw std::invalid_argument("the maximum number of particles per leaf must be nonzero");
        }
        // Prepare the vectors.
        m_masses.resize(N);
        for (auto &vc : m_coords) {
            vc.resize(N);
        }
        // NOTE: this ensures that, from now on, we can just cast
        // freely between the size types of the masses/coords and codes vectors.
        m_codes.resize(boost::numeric_cast<decltype(m_codes.size())>(N));
        // Get out soon if there's nothing to do.
        if (!m_codes.size()) {
            return;
        }
        // Function to discretise the input NDim floating-point coordinates starting at 'it'
        // into a box of a given size box_size.
        auto disc_coords = [&box_size](auto it) {
            constexpr UInt factor = UInt(1) << cbits;
            std::array<UInt, NDim> retval;
            for (std::size_t i = 0; i < NDim; ++i, ++it) {
                const auto &x = *it;
                // Translate and rescale the coordinate so that -box_size/2 becomes zero
                // and box_size/2 becomes 1.
                auto tmp = (x + box_size / F(2)) / box_size;
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
                // Cast to UInt and write to retval.
                retval[i] = static_cast<UInt>(tmp);
                // Last check, make sure we don't overflow.
                if (retval[i] >= factor) {
                    throw std::invalid_argument("Out of bounds! (after cast)");
                }
            }
            return retval;
        };
        // Temporary structure used in the encoding.
        std::array<F, NDim> tmp_coord;
        // The encoder object.
        morton_encoder<NDim, UInt> me;
        // Determine the particles' codes, and store them in a temporary vector for indirect
        // sorting. Remember also the original coord iters for later use.
        auto old_c_it = c_it;
        auto &v_ind = get_v_ind();
        v_ind.resize(boost::numeric_cast<decltype(v_ind.size())>(N));
        for (size_type i = 0; i < N; ++i) {
            // Write the coords in the temp structure.
            for (std::size_t j = 0; j < NDim; ++j) {
                tmp_coord[j] = *c_it[j];
            }
            // Compute and store the code.
            v_ind[i].first = me(disc_coords(tmp_coord.begin()).begin());
            // Store the index.
            v_ind[i].second = i;
            // Increase the coordinates iterators.
            for (auto &_ : c_it) {
                ++_;
            }
        }
        // Do the actual sorting of v_ind.
        boost::sort::spreadsort::integer_sort(
            v_ind.begin(), v_ind.end(),
            [](const auto &p, unsigned offset) { return static_cast<UInt>(p.first >> offset); },
            [](const auto &p1, const auto &p2) { return p1.first < p2.first; });
        // Now let's write the masses, coords and the codes in the correct order.
        // NOTE: we will need to index into It, so make sure its difference type
        // can represent the total number of particles.
        using it_diff_t = typename std::iterator_traits<It>::difference_type;
        using it_udiff_t = std::make_unsigned_t<it_diff_t>;
        if (N > static_cast<it_udiff_t>(std::numeric_limits<it_diff_t>::max())) {
            throw std::overflow_error("the number of particles (" + std::to_string(N)
                                      + ") is too large, and it results in an overflow condition");
        }
        for (size_type i = 0; i < N; ++i) {
            const auto idx = static_cast<it_diff_t>(v_ind[i].second);
            m_masses[i] = m_it[idx];
            for (std::size_t j = 0; j < NDim; ++j) {
                m_coords[j][i] = old_c_it[j][idx];
            }
            m_codes[i] = v_ind[i].first;
        }
        // Now let's proceed to the tree construction.
        build_tree();
        // Now move to the computation of the COM of the nodes.
        build_tree_properties();
    }
    friend std::ostream &operator<<(std::ostream &os, const tree &t)
    {
        static_assert(unsigned(std::numeric_limits<UInt>::digits) <= std::numeric_limits<std::size_t>::max());
        os << "Total number of particles: " << t.m_codes.size() << '\n';
        os << "Total number of nodes    : " << t.m_tree.size() << "\n\n";
        os << "First few nodes:\n";
        constexpr unsigned max_nodes = 20;
        auto i = 0u;
        for (const auto &tup : t.m_tree) {
            if (i > max_nodes) {
                break;
            }
            os << std::bitset<std::numeric_limits<UInt>::digits>(std::get<0>(tup)) << '|' << std::get<1>(tup)[0] << ','
               << std::get<1>(tup)[1] << ',' << std::get<1>(tup)[2] << "|" << std::get<2>(tup) << "|[";
            for (std::size_t j = 0; j < NDim; ++j) {
                os << std::get<3>(tup)[j];
                if (j + 1u < NDim) {
                    os << ", ";
                }
            }
            os << "]\n";
            ++i;
        }
        if (i > max_nodes) {
            std::cout << "...\n";
        }
        return os;
    }
    template <typename OutIt>
    void compute_accelerations(OutIt out_it)
    {
    }

private:
    F m_box_size;
    size_type m_max_leaf_n;
    v_type<F> m_masses;
    std::array<v_type<F>, NDim> m_coords;
    v_type<UInt> m_codes;
    v_type<std::tuple<UInt, std::array<size_type, 3>, F, std::array<F, NDim>>> m_tree;
};

#include <random>

#include <boost/math/constants/constants.hpp>

static constexpr unsigned nparts = 1'000'000;
static constexpr float bsize = 10.f;

static std::mt19937 rng;

template <std::size_t D, typename F>
inline std::vector<F> get_uniform_particles(std::size_t n, F size)
{
    std::vector<F> retval(n * (D + 1u));
    // Mass.
    std::uniform_real_distribution<F> mdist(F(0), F(1));
    std::generate(retval.begin(), retval.begin() + n, [&mdist]() { return mdist(rng); });
    // Positions.
    std::uniform_real_distribution<F> rdist(-size / F(2) /*+ size * F(0.01)*/, size / F(2) /*- size * F(0.01)*/);
    std::generate(retval.begin() + n, retval.end(), [&rdist]() { return rdist(rng); });
    return retval;
}

// See: http://www.artcompsci.org/kali/vol/plummer/ch03.html
template <typename F>
inline std::vector<F> get_plummer_sphere(std::size_t n, F size)
{
    std::vector<F> retval(n * 4u);
    // Uniform [0, 1) dist.
    std::uniform_real_distribution<F> udist(F(0), F(1));
    // Particle mass is always 1/n.
    std::fill(retval.begin(), retval.begin() + n, F(1) / F(n));
    for (std::size_t i = 0; i < n; ++i) {
        // Generate a random radius.
        const F r = F(1) / std::sqrt(std::pow(udist(rng), F(-2) / F(3)) - F(1));
        // Generate random u, v for sphere picking.
        const F u = udist(rng), v = udist(rng);
        // Longitude.
        const F lon
            = std::clamp(F(2) * boost::math::constants::pi<F>() * u, F(0), F(2) * boost::math::constants::pi<F>());
        // Colatitude.
        const F colat = std::acos(std::clamp(F(2) * v - F(1), F(-1), F(1)));
        // Compute x, y, z.
        const F x = r * std::cos(lon) * std::sin(colat);
        const F y = r * std::sin(lon) * std::sin(colat);
        const F z = r * std::cos(colat);
        if (x >= -size / F(2) && x < size / F(2) && y >= -size / F(2) && y < size / F(2) && z >= -size / F(2)
            && z < size / F(2)) {
            // Assign coordinates only if we fall into the domain. Otherwise, try again.
            *(retval.begin() + n + i) = x;
            *(retval.begin() + 2 * n + i) = y;
            *(retval.begin() + 3 * n + i) = z;
        }
    }
    return retval;
}

int main()
{
    auto parts = get_uniform_particles<3>(nparts, bsize);
    // auto parts = get_plummer_sphere(nparts, bsize);
    std::cout << tree<std::uint64_t, float, 3>(
                     bsize, parts.begin(),
                     {parts.begin() + nparts, parts.begin() + 2 * nparts, parts.begin() + 3 * nparts}, nparts, 16)
              << '\n';
}
