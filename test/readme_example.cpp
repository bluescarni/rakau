// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <initializer_list>
#include <vector>

#include <rakau/tree.hpp>

using namespace rakau;
using namespace rakau::kwargs;

int main()
{
    // Create an octree from a set of particle coordinates and masses.
    octree<float> t{x_coords = {1, 2, 3}, y_coords = {4, 5, 6}, z_coords = {7, 8, 9}, masses = {1, 1, 1}};

    // Prepare output vectors for the accelerations.
    std::array<std::vector<float>, 3> accs;

    // Compute the accelerations with a theta parameter of 0.4.
    t.accs_u(accs, 0.4f);
}
