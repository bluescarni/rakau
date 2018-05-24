#include <random>

#include <boost/math/constants/constants.hpp>

#include <rakau/tree.hpp>

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
    // Uniform [0.1, 1.9) dist.
    std::uniform_real_distribution<F> mdist(F(0.1), F(1.9));
    // Average mass of 1.
    std::generate(retval.begin(), retval.begin() + n, [&mdist]() { return mdist(rng); });
    for (std::size_t i = 0; i < n;) {
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
            ++i;
        }
    }
    return retval;
}

#include <boost/lexical_cast.hpp>

using namespace rakau;

int main(int argc, char **argv)
{
    if (argc < 2) {
        throw std::runtime_error("Need at least 4 arguments, but only " + std::to_string(argc) + " was/were provided");
    }
    std::cout.precision(20);
    const auto idx = boost::lexical_cast<std::size_t>(argv[1]);
    const auto max_leaf_n = boost::lexical_cast<std::size_t>(argv[2]);
    const auto ncrit = boost::lexical_cast<std::size_t>(argv[3]);
    // auto parts = get_uniform_particles<3>(nparts, bsize);
    auto parts = get_plummer_sphere(nparts, bsize);
    tree<std::uint64_t, float, 3> t(bsize, parts.begin(),
                                    {parts.begin() + nparts, parts.begin() + 2 * nparts, parts.begin() + 3 * nparts},
                                    nparts, max_leaf_n, ncrit);
    std::cout << t << '\n';
    std::array<std::vector<float>, 3> accs;
    // std::cout << accs[idx * 3] << ", " << accs[idx * 3 + 1] << ", " << accs[idx * 3 + 2] << '\n';
    t.accs_u(accs, 0.75f);
    std::cout << accs[0][t.ord_ind()[idx]] << ", " << accs[1][t.ord_ind()[idx]] << ", " << accs[2][t.ord_ind()[idx]]
              << '\n';
    auto eacc = t.exact_acc_u(t.ord_ind()[idx]);
    std::cout << eacc[0] << ", " << eacc[1] << ", " << eacc[2] << '\n';
    t.accs_o(accs, 0.75f);
    std::cout << accs[0][idx] << ", " << accs[1][idx] << ", " << accs[2][idx] << '\n';
    eacc = t.exact_acc_o(idx);
    std::cout << eacc[0] << ", " << eacc[1] << ", " << eacc[2] << '\n';
    std::cout << "ZERO\n";
    std::cout << accs[0][0] << ", " << accs[1][0] << ", " << accs[2][0] << '\n';
    eacc = t.exact_acc_o(0);
    std::cout << eacc[0] << ", " << eacc[1] << ", " << eacc[2] << '\n';
    // Slightly change the position of a single particle.
    t.update_positions_o([idx](auto s) {
        for (std::size_t j = 0; j < 3u; ++j) {
            s[j].first[idx] += 1E-6;
        }
    });
    t.accs_o(accs, 0.75f);
    std::cout << accs[0][idx] << ", " << accs[1][idx] << ", " << accs[2][idx] << '\n';
    eacc = t.exact_acc_o(idx);
    std::cout << eacc[0] << ", " << eacc[1] << ", " << eacc[2] << '\n';
    std::cout << accs[0][0] << ", " << accs[1][0] << ", " << accs[2][0] << '\n';
    eacc = t.exact_acc_o(0);
    std::cout << eacc[0] << ", " << eacc[1] << ", " << eacc[2] << '\n';
#if 0
    // Test ordered iters.
    auto test_ordered_iters = [&]() {
        auto [x_r, y_r, z_r] = t.ord_c_ranges();
        std::cout << *x_r.first << ", " << *y_r.first << ", " << *z_r.first << '\n';
        std::cout << *(parts.begin() + nparts) << ", " << *(parts.begin() + 2 * nparts) << ", "
                  << *(parts.begin() + 3 * nparts) << '\n';
        x_r.first += 5;
        y_r.first += 5;
        z_r.first += 5;
        std::cout << *x_r.first << ", " << *y_r.first << ", " << *z_r.first << '\n';
        std::cout << *(parts.begin() + nparts + 5) << ", " << *(parts.begin() + 2 * nparts + 5) << ", "
                  << *(parts.begin() + 3 * nparts + 5) << '\n';
    };
    test_ordered_iters();
    // Update positions, identity.
    t.update_positions([](const auto &) {});
    test_ordered_iters();
    // Again.
    t.update_positions([](const auto &) {});
    test_ordered_iters();
    // Update positions, divide by two.
    t.update_positions([](auto s) {
        for (std::size_t j = 0; j < 3u; ++j) {
            for (; s[j].first != s[j].second; ++s[j].first) {
                *s[j].first /= 2;
            }
        }
    });
    test_ordered_iters();
    // Update positions, take square root.
    t.update_positions([](auto s) {
        for (std::size_t j = 0; j < 3u; ++j) {
            for (; s[j].first != s[j].second; ++s[j].first) {
                const auto sign = *s[j].first >= 0;
                if (sign) {
                    *s[j].first = std::sqrt(*s[j].first);
                } else {
                    *s[j].first = -std::sqrt(-*s[j].first);
                }
            }
        }
    });
    test_ordered_iters();
    // Rotate around the z axis by 180 degrees.
    t.vec_accs_u(accs, 0.75f);
    std::cout << "vec acc before rotation at idx 42: " << accs[0][t.ord_ind()[42]] << ", " << accs[1][t.ord_ind()[42]]
              << ", " << accs[2][t.ord_ind()[42]] << '\n';
    t.update_positions([](const auto &s) {
        for (auto i = 0u; i < nparts; ++i) {
            const auto x0 = s[0].first[i];
            const auto y0 = s[1].first[i];
            const auto z0 = s[2].first[i];
            const auto r0 = std::hypot(x0, y0, z0);
            const auto th0 = std::acos(z0 / r0);
            const auto phi0 = std::atan2(y0, x0);
            const auto phi1 = phi0 + boost::math::constants::pi<std::remove_reference_t<decltype(s[0].first[0])>>();
            s[0].first[i] = r0 * std::sin(th0) * std::cos(phi1);
            s[1].first[i] = r0 * std::sin(th0) * std::sin(phi1);
            s[2].first[i] = r0 * std::cos(th0);
        }
    });
    t.vec_accs_u(accs, 0.75f);
    std::cout << "vec acc after rotation at idx 42: " << accs[0][t.ord_ind()[42]] << ", " << accs[1][t.ord_ind()[42]]
              << ", " << accs[2][t.ord_ind()[42]] << '\n';
    // Rotate again.
    t.update_positions([](const auto &s) {
        for (auto i = 0u; i < nparts; ++i) {
            const auto x0 = s[0].first[i];
            const auto y0 = s[1].first[i];
            const auto z0 = s[2].first[i];
            const auto r0 = std::hypot(x0, y0, z0);
            const auto th0 = std::acos(z0 / r0);
            const auto phi0 = std::atan2(y0, x0);
            const auto phi1 = phi0 + boost::math::constants::pi<std::remove_reference_t<decltype(s[0].first[0])>>();
            s[0].first[i] = r0 * std::sin(th0) * std::cos(phi1);
            s[1].first[i] = r0 * std::sin(th0) * std::sin(phi1);
            s[2].first[i] = r0 * std::cos(th0);
        }
    });
    t.vec_accs_u(accs, 0.75f);
    std::cout << "vec acc after rotation at idx 42: " << accs[0][t.ord_ind()[42]] << ", " << accs[1][t.ord_ind()[42]]
              << ", " << accs[2][t.ord_ind()[42]] << '\n';
#endif
}
