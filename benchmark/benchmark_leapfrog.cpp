// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/math/constants/constants.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <boost/program_options.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>

#include <rakau/tree.hpp>

template <typename F>
inline const auto pi_const = boost::math::constants::pi<F>();

// Generate a plummer distribution. All particles will have the same mass.
//
// - G: grav constant
// - M: total mass
// - a: plummer core radius
// - n: number of particles
// - rng: random engine
// - with_velocities: generate velocities (if false, only the positions
//   will be generated)
//
// Returns tuple with:
// - x_pos, y_pos, z_pos
// - x_vel, y_vel, z_vel
template <typename F, typename Rng>
inline auto plummer(F G, F M, F a, unsigned long n, Rng &rng, bool with_velocities)
{
    // Mass of a single particle.
    const auto m = M / n;
    // Various distributions we will be using in the generation.
    std::uniform_real_distribution<F> dist1, dist2(F(-1), F(1)), dist3(F(0), F(2) * pi_const<F>),
        dist4(F(0), F(1) / F(10));
    // The output vectors.
    const auto v_size = boost::numeric_cast<typename std::vector<F>::size_type>(n);
    std::vector<F> x_pos(v_size), y_pos(v_size), z_pos(v_size), x_vel(v_size), y_vel(v_size), z_vel(v_size);
    // Small helper for rejection sampling.
    auto rej_sample = [&dist1, &dist4, &rng]() {
        auto x = F(0), y = F(1) / F(10);
        while (y > x * x * std::pow(F(1) - x * x, F(7) / F(2))) {
            x = dist1(rng);
            y = dist4(rng);
        }
        return x;
    };
    for (typename std::vector<F>::size_type i = 0; i < v_size; ++i) {
        // Positions.
        const auto r = a / std::sqrt(std::pow(dist1(rng), -F(2) / F(3)) - F(1)), theta = std::acos(dist2(rng)),
                   phi = dist3(rng);
        x_pos[i] = r * std::sin(theta) * std::cos(phi);
        y_pos[i] = r * std::sin(theta) * std::sin(phi);
        z_pos[i] = r * std::cos(theta);
        if (with_velocities) {
            // Velocities.
            const auto q = rej_sample();
            const auto v = q * std::sqrt(F(2) * G * M / a) * std::pow(F(1) + r * r / (a * a), -F(1) / F(4));
            const auto theta_v = std::acos(dist2(rng)), phi_v = dist3(rng);
            x_vel[i] = v * std::sin(theta_v) * std::cos(phi_v);
            y_vel[i] = v * std::sin(theta_v) * std::sin(phi_v);
            z_vel[i] = v * std::cos(theta_v);
        }
        // NOTE: if velocities are not requested, they are already zeroed
        // out by the std::vector constructor.
    }
    // Energy checks, meaningful only if velocities
    // have been requested.
    if (with_velocities) {
        F tot_kin = 0;
        for (typename std::vector<F>::size_type i = 0; i < v_size; ++i) {
            tot_kin += (F(1) / F(2)) * m * (x_vel[i] * x_vel[i] + y_vel[i] * y_vel[i] + z_vel[i] * z_vel[i]);
        }
        const auto ex_tot_en = (-F(3) * pi_const<F> / F(64)) * G * M * M / a,
                   ex_tot_pot = (-F(3) * pi_const<F> / F(32)) * G * M * M / a;
        std::cout << "Computed kin energy  : " << tot_kin << '\n';
        std::cout << "Expected kin energy  : " << (ex_tot_en - ex_tot_pot) << '\n';
    }
    return std::tuple{std::move(x_pos), std::move(y_pos), std::move(z_pos),
                      std::move(x_vel), std::move(y_vel), std::move(z_vel)};
}

static std::mt19937 rng;

using namespace rakau;
namespace po = boost::program_options;

int main(int argc, char **argv)
{
    unsigned long nparts;
    unsigned max_leaf_n, ncrit, nthreads;
    double a, mac_value, timestep;
    std::vector<double> split;
    std::string fp_type, mac_type;

    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")(
        "nparts", po::value<unsigned long>(&nparts)->default_value(1'000'000ul), "number of particles")(
        "max_leaf_n", po::value<unsigned>(&max_leaf_n)->default_value(rakau::default_max_leaf_n),
        "max number of particles in a leaf node")("ncrit",
                                                  po::value<unsigned>(&ncrit)->default_value(rakau::default_ncrit),
                                                  "maximum number of particles in a critical node")(
        "a", po::value<double>(&a)->default_value(1.),
        "Plummer core radius")("nthreads", po::value<unsigned>(&nthreads)->default_value(0u),
                               "number of threads to use (0 for auto-detection)")(
        "mac_value", po::value<double>(&mac_value)->default_value(0.75), "MAC value")(
        "split", po::value<std::vector<double>>()->multitoken(), "split vector for heterogeneous computations")(
        "fp_type", po::value<std::string>(&fp_type)->default_value("float"),
        "floating-point type to use in the computations")(
        "mac_type", po::value<std::string>(&mac_type)->default_value("bh"),
        "MAC type")("timestep", po::value<double>(&timestep)->default_value(1E-4), "integration timestep");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        std::exit(0);
    }

    if (nparts == 0u) {
        throw std::invalid_argument("The number of particles cannot be zero");
    }

    if (vm.count("split")) {
        split = vm["split"].as<std::vector<double>>();
    }

    if (fp_type != "float" && fp_type != "double") {
        throw std::invalid_argument("Only the 'float' and 'double' floating-point types are supported, but the type '"
                                    + fp_type + "' was specified instead");
    }

    if (mac_type != "bh" && mac_type != "bh_geom") {
        throw std::invalid_argument("'" + mac_type + "' is not a valid MAC type");
    }

    if (!std::isfinite(a) || a <= 0.) {
        throw std::invalid_argument("The Plummer core radius must be finite and positive, but it is "
                                    + std::to_string(a) + " instead");
    }

    if (!std::isfinite(timestep) || timestep <= 0.) {
        throw std::invalid_argument("The integration timestep must be finite and positive, but it is "
                                    + std::to_string(timestep) + " instead");
    }

    // Setup the number of threads.
    std::optional<tbb::task_scheduler_init> t_init;
    if (nthreads) {
        t_init.emplace(nthreads);
    }

    auto runner = [&](auto x) {
        using F = decltype(x);

        auto inner = [&](auto m) {
            // Build the Plummer distribution.
            std::cout << "Building the Plummer distribution...\n";
            std::vector<F> x_pos, y_pos, z_pos, x_vel, y_vel, z_vel;
            std::tie(x_pos, y_pos, z_pos, x_vel, y_vel, z_vel)
                = plummer(F(1), F(1), static_cast<F>(a), nparts, rng, true);
            // Clip the plummer distribution at 10 core radiuses.
            {
                typename std::vector<F>::size_type idx = 0;
                for (typename std::vector<F>::size_type i = 0; i < nparts; ++i) {
                    const auto dist2 = x_pos[i] * x_pos[i] + y_pos[i] * y_pos[i] + z_pos[i] * z_pos[i];
                    if (dist2 < F(100) * a * a) {
                        x_pos[idx] = x_pos[i];
                        y_pos[idx] = y_pos[i];
                        z_pos[idx] = z_pos[i];
                        x_vel[idx] = x_vel[i];
                        y_vel[idx] = y_vel[i];
                        z_vel[idx] = z_vel[i];
                        ++idx;
                    }
                }
                // Resize to the clipped size, and update the value of nparts.
                x_pos.resize(idx);
                y_pos.resize(idx);
                z_pos.resize(idx);
                x_vel.resize(idx);
                y_vel.resize(idx);
                z_vel.resize(idx);
                nparts = static_cast<unsigned long long>(idx);
                std::cout << "After clipping, nparts is " << nparts << '\n';
            }
            std::cout << "Done\n";

            // Softening length.
            // NOTE: this is the Athanassoula heuristic for concentrated distributions.
            // https://arxiv.org/pdf/astro-ph/0011568v1.pdf
            // This should be used for 1E3 <= nparts <= 1E5.
            const auto eps = static_cast<F>(0.45) * std::pow(static_cast<F>(nparts), static_cast<F>(-0.73));
            std::cout << "Softening length: " << eps << '\n';

            // Create the vector of masses.
            std::vector<F> masses(boost::numeric_cast<typename std::vector<F>::size_type>(nparts), F(1) / nparts);

            // Create the octree.
            octree<F, decltype(m)::value> t{kwargs::x_coords = x_pos,        kwargs::y_coords = y_pos,
                                            kwargs::z_coords = z_pos,        kwargs::masses = masses,
                                            kwargs::max_leaf_n = max_leaf_n, kwargs::ncrit = ncrit};

            std::cout << "Box size: " << t.box_size() << '\n';

            // Prepare the acceleration buffers.
            std::vector<F> acc_x(boost::numeric_cast<typename std::vector<F>::size_type>(nparts)),
                acc_y(boost::numeric_cast<typename std::vector<F>::size_type>(nparts)),
                acc_z(boost::numeric_cast<typename std::vector<F>::size_type>(nparts));
            auto acc_its = std::array{acc_x.data(), acc_y.data(), acc_z.data()};

            // Prepare the buffers to contain the kicked velocities.
            std::vector<F> kick_x_vel(boost::numeric_cast<typename std::vector<F>::size_type>(nparts)),
                kick_y_vel(boost::numeric_cast<typename std::vector<F>::size_type>(nparts)),
                kick_z_vel(boost::numeric_cast<typename std::vector<F>::size_type>(nparts));

            // Precompute half timestep.
            const F half_timestep = timestep / F(2);

            // Small helper to re-order the input vector vec according to the last reordering permutation in t.
            // We will re-use x_pos as a temporary buffer, and clear up y_pos and z_pos (which are unused).
            auto &tmp_buffer = x_pos;
            y_pos.clear();
            z_pos.clear();
            auto reorder = [&t, &tmp_buffer](auto &vec) {
                tbb::parallel_for(tbb::blocked_range<decltype(t.last_perm().size())>(0, t.last_perm().size()),
                                  [&tmp_buffer, &vec, &last_perm = t.last_perm()](const auto &range) {
                                      for (auto i = range.begin(); i != range.end(); ++i) {
                                          tmp_buffer[i] = vec[last_perm[i]];
                                      }
                                  });
                vec.swap(tmp_buffer);
            };

            // Reorder the velocities according to the internal tree order.
            reorder(x_vel);
            reorder(y_vel);
            reorder(z_vel);

            // Run the initial accelerations computation.
            t.accs_u(acc_its, mac_value, kwargs::split = split, kwargs::eps = eps);

            while (true) {
                // Compute the kicked velocities.
                tbb::parallel_for(tbb::blocked_range(0ul, nparts), [&](const auto &range) {
                    for (auto i = range.begin(); i != range.end(); ++i) {
                        kick_x_vel[i] = std::fma(acc_its[0][i], half_timestep, x_vel[i]);
                        kick_y_vel[i] = std::fma(acc_its[1][i], half_timestep, y_vel[i]);
                        kick_z_vel[i] = std::fma(acc_its[2][i], half_timestep, z_vel[i]);
                    }
                });

                // Update the particle positions in the tree according to the kicked velocities.
                t.update_particles_u([&](const auto &p_its) {
                    tbb::parallel_for(tbb::blocked_range(0ul, nparts), [&](const auto &range) {
                        const auto [x_it, y_it, z_it, m_it] = p_its;
                        (void)m_it;
                        for (auto i = range.begin(); i != range.end(); ++i) {
                            *(x_it + i) = std::fma(kick_x_vel[i], timestep, *(x_it + i));
                            *(y_it + i) = std::fma(kick_y_vel[i], timestep, *(y_it + i));
                            *(z_it + i) = std::fma(kick_z_vel[i], timestep, *(z_it + i));
                        }
                    });
                });

                // Compute the accelerations in the new positions.
                t.accs_u(acc_its, mac_value, kwargs::split = split, kwargs::eps = eps);

                // Update the velocities.
                tbb::parallel_for(tbb::blocked_range(0ul, nparts), [&](const auto &range) {
                    const auto &lp = t.last_perm();
                    for (auto i = range.begin(); i != range.end(); ++i) {
                        x_vel[i] = std::fma(acc_its[0][i], half_timestep, kick_x_vel[lp[i]]);
                        y_vel[i] = std::fma(acc_its[1][i], half_timestep, kick_y_vel[lp[i]]);
                        z_vel[i] = std::fma(acc_its[2][i], half_timestep, kick_z_vel[lp[i]]);
                    }
                });
            }
        };

        if (mac_type == "bh") {
            inner(std::integral_constant<mac, mac::bh>{});
        } else {
            inner(std::integral_constant<mac, mac::bh_geom>{});
        }
    };

    if (fp_type == "float") {
        runner(0.f);
    } else {
        runner(0.);
    }
}
