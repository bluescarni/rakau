// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <memory>
#include <random>

#include <boost/math/constants/constants.hpp>

#include <tbb/task_scheduler_init.h>

#include <rakau/tree.hpp>

static constexpr unsigned nparts = 1'000'000;
static constexpr float bsize = 10.f;

static std::mt19937 rng;

template <std::size_t D, typename F>
inline std::vector<F> get_uniform_particles(std::size_t n, F size)
{
    rakau::simple_timer st("uniform init");
    std::vector<F> retval(n * (D + 1u));
    // Mass.
    std::uniform_real_distribution<F> mdist(F(0), F(1));
    std::generate(retval.begin(), retval.begin() + n, [&mdist]() { return mdist(rng); });
    // Positions.
    const auto size_limit = size / F(2) - F(0.01);
    std::uniform_real_distribution<F> rdist(-size_limit, size_limit);
    std::generate(retval.begin() + n, retval.end(), [&rdist]() { return rdist(rng); });
    return retval;
}

// See: http://www.artcompsci.org/kali/vol/plummer/ch03.html
template <typename F>
inline std::vector<F> get_plummer_sphere(std::size_t n, F size)
{
    rakau::simple_timer st("plummer init");
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
        const auto size_limit = size / F(2) - F(0.01);
        if (x >= -size_limit && x < size_limit && y >= -size_limit && y < size_limit && z >= -size_limit
            && z < size_limit) {
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
    if (argc < 4) {
        throw std::runtime_error("Need at least 3 arguments, but only " + std::to_string(argc) + " was/were provided");
    }
    std::cout.precision(20);
    const auto idx = boost::lexical_cast<std::size_t>(argv[1]);
    const auto max_leaf_n = boost::lexical_cast<std::size_t>(argv[2]);
    const auto ncrit = boost::lexical_cast<std::size_t>(argv[3]);
    auto s_init = argc >= 5 ? std::make_unique<tbb::task_scheduler_init>(boost::lexical_cast<unsigned>(argv[4]))
                            : std::unique_ptr<tbb::task_scheduler_init>(nullptr);

    // auto parts = get_uniform_particles<3>(nparts, bsize);
    auto parts = get_plummer_sphere(nparts, bsize);
    tree<3, float> t(
        bsize,
        std::array{parts.begin() + nparts, parts.begin() + 2 * nparts, parts.begin() + 3 * nparts, parts.begin()},
        nparts, max_leaf_n, ncrit);
    std::cout << t << '\n';
    std::array<std::vector<float>, 3> accs;
    t.accs_u(accs, 0.75f);
    std::cout << accs[0][t.ord_ind()[idx]] << ", " << accs[1][t.ord_ind()[idx]] << ", " << accs[2][t.ord_ind()[idx]]
              << '\n';
    auto eacc = t.exact_acc_u(t.ord_ind()[idx]);
    std::cout << eacc[0] << ", " << eacc[1] << ", " << eacc[2] << '\n';
}
