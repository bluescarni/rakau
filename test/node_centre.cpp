// Copyright 2018 Francesco Biscani (bluescarni@gmail.com)
//
// This file is part of the rakau library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <rakau/tree.hpp>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <vector>

using namespace rakau;

// Verify the computation of the geometrical node centre in a few simple cases.
TEST_CASE("node centre quadtree")
{
    using tree_t = quadtree<double>;
    tree_t t;
    REQUIRE(t.nodes().size() == 0u);
    double buffer[2];
    t = tree_t{std::array{std::vector{-1.}, std::vector{1.}, std::vector{1.}}, kwargs::box_size = 10};
    REQUIRE(t.nodes().size() == 1u);
    node_centre(buffer, t.nodes()[0].code, t.box_size());
    REQUIRE(std::abs(buffer[0]) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1]) < 10. * std::numeric_limits<double>::epsilon());
    t = tree_t{std::array{std::vector{-1., -1., 1., 1.}, std::vector{-1., 1., -1., 1.}, std::vector{1., 1., 1., 1.}},
               kwargs::box_size = 10, kwargs::max_leaf_n = 1};
    REQUIRE(t.nodes().size() == 5u);
    node_centre(buffer, t.nodes()[0].code, t.box_size());
    REQUIRE(std::abs(buffer[0]) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1]) < 10. * std::numeric_limits<double>::epsilon());
    node_centre(buffer, t.nodes()[1].code, t.box_size());
    REQUIRE(std::abs(buffer[0] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    node_centre(buffer, t.nodes()[2].code, t.box_size());
    REQUIRE(std::abs(buffer[0] - (t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    node_centre(buffer, t.nodes()[3].code, t.box_size());
    REQUIRE(std::abs(buffer[0] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1] - (t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    node_centre(buffer, t.nodes()[4].code, t.box_size());
    REQUIRE(std::abs(buffer[0] - (t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1] - (t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
}

TEST_CASE("node centre octree")
{
    using tree_t = octree<double>;
    tree_t t;
    REQUIRE(t.nodes().size() == 0u);
    double buffer[3];
    t = tree_t{std::array{std::vector{-1.}, std::vector{1.}, std::vector{1.}, std::vector{1.}}, kwargs::box_size = 10};
    REQUIRE(t.nodes().size() == 1u);
    node_centre(buffer, t.nodes()[0].code, t.box_size());
    REQUIRE(std::abs(buffer[0]) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1]) < 10. * std::numeric_limits<double>::epsilon());
    t = tree_t{std::array{std::vector{-1., +1., -1., +1., -1., +1., -1., +1.},
                          std::vector{-1., -1., +1., +1., -1., -1., +1., +1.},
                          std::vector{-1., -1., -1., -1., +1., +1., +1., +1.},
                          std::vector{1., 1., 1., 1., 1., 1., 1., 1.}},
               kwargs::box_size = 10, kwargs::max_leaf_n = 1};
    REQUIRE(t.nodes().size() == 9u);
    node_centre(buffer, t.nodes()[0].code, t.box_size());
    REQUIRE(std::abs(buffer[0]) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1]) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[2]) < 10. * std::numeric_limits<double>::epsilon());
    node_centre(buffer, t.nodes()[1].code, t.box_size());
    REQUIRE(std::abs(buffer[0] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[2] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    node_centre(buffer, t.nodes()[2].code, t.box_size());
    REQUIRE(std::abs(buffer[0] - (+t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[2] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    node_centre(buffer, t.nodes()[3].code, t.box_size());
    REQUIRE(std::abs(buffer[0] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1] - (+t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[2] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    node_centre(buffer, t.nodes()[4].code, t.box_size());
    REQUIRE(std::abs(buffer[0] - (+t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1] - (+t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[2] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    node_centre(buffer, t.nodes()[5].code, t.box_size());
    REQUIRE(std::abs(buffer[0] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[2] - (+t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    node_centre(buffer, t.nodes()[6].code, t.box_size());
    REQUIRE(std::abs(buffer[0] - (+t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[2] - (+t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    node_centre(buffer, t.nodes()[7].code, t.box_size());
    REQUIRE(std::abs(buffer[0] - (-t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1] - (+t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[2] - (+t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    node_centre(buffer, t.nodes()[8].code, t.box_size());
    REQUIRE(std::abs(buffer[0] - (+t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[1] - (+t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
    REQUIRE(std::abs(buffer[2] - (+t.box_size() / 4.)) < 10. * std::numeric_limits<double>::epsilon());
}
