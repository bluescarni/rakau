[![Build Status](https://img.shields.io/travis/bluescarni/rakau/master.svg?logo=travis&style=for-the-badge)](https://travis-ci.org/bluescarni/rakau)
![language](https://img.shields.io/badge/language-C%2B%2B17-red.svg?style=for-the-badge)
![license](https://img.shields.io/badge/license-MPL2-blue.svg?style=for-the-badge)
[![Join the chat at https://gitter.im/rakau_nbody/community](https://img.shields.io/badge/gitter-join--chat-green.svg?logo=gitter-white&style=for-the-badge)](https://gitter.im/rakau_nbody/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

rakau
=====

rakau is a C++17 library for the computation of accelerations and potentials in gravitational
[N-body simulations](https://en.wikipedia.org/wiki/N-body_simulation).

The core of the library is a high-performance implementation of the
[Barnes-Hut tree algorithm](https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation), capable of
taking advantage of modern heterogeneous hardware architectures. Specifically, rakau can run on:

* multicore CPUs, where it takes advantage of both multithreading and [SIMD](https://en.wikipedia.org/wiki/SIMD) instructions,
* AMD GPUs, via [ROCm](https://rocm.github.io/),
* Nvidia GPUs, via [CUDA](https://en.wikipedia.org/wiki/CUDA).

On CPUs, multithreaded parallelism is implemented on top of the [TBB library](https://www.threadingbuildingblocks.org/).
Vectorisation is achieved via the [xsimd](https://github.com/QuantStack/xsimd) intrinsics wrapper library
(which means that, e.g., on x86-64 CPUs rakau can use all the available vector extensions, from SSE2 to AVX-512).

At the present time, rakau can accelerate on GPUs the tree traversal part of the Barnes-Hut algorithm.
During tree traversal, the CPU and the GPU can be used concurrently, and the user is free to choose
(depending on the available hardware) how to split the computation between CPU and GPU.

Work is ongoing to accelerate on GPUs other parts of the Barnes-Hut algorithm (most notably, tree
construction).

Performance
-----------

The following table lists the runtime for the computation of the gravitational accelerations
(i.e., the tree traversal part of the Barnes-Hut algorithm) in a system of 4 million particles
distributed according to the [Plummer model](https://en.wikipedia.org/wiki/Plummer_model).
Various hardware configurations are tested. The theta parameter is set to 0.75,
the computation is done in single precision.

| Hardware                           | Type                 | Compiler     | Runtime |
| :--------------------------------- | :------------------- | :----------- | ------: |
| 2 x Intel Xeon Gold 6148 | CPU (AVX-512, 40 cores + SMT) | GCC 8        |   82 ms |
| 2 x Intel Xeon E5-2698      | CPU (AVX2, 40 cores)       | GCC 7        |  133 ms |
| AMD Ryzen 1700              | CPU (AVX2, 8 cores + SMT)  | GCC 8        |  580 ms |
| AMD Ryzen 1700              | CPU (AVX2, 8 cores + SMT)  | HCC 1.9      |  688 ms |
| AMD Radeon RX 570                  | GPU (Polaris)        | HCC 1.9      |  256 ms |
| Ryzen 1700 + RX 570                | CPU+GPU              | HCC 1.9      |  195 ms |
| Nvidia GeForce GTX 1080 Ti         | GPU (Pascal)         | NVCC         |  140 ms |
| Nvidia V100                        | GPU (Volta)          | NVCC         |   95 ms |
| Intel Core i7-3610QM         | CPU (AVX, 4 cores + SMT)  | GCC 8        | 1510 ms |
| Nvidia GeForce GT 650M             | GPU (Kepler)         | NVCC         | 2818 ms |
| i7-3610QM + GT 650M                | CPU+GPU              | GCC 8 + NVCC | 1080 ms |

Features
--------

Current:

* single and double precision<sup>1</sup>,
* 2D and 3D<sup>2</sup>,
* computation of accelerations and/or potentials,
* support for multiple MACs (multipole acceptance criteria),
* highly configurable tree structure,
* ergonomic API based on modern C++ idioms.

Planned:

* higher multipole moments,
* support for integration schemes based on hierarchical timesteps,
* better support for multi-GPU setups<sup>3</sup>,
* Python interface.

<sup>1</sup>``long double`` is supported as well,
but it is available only on the CPU and there's no SIMD support for extended precision
on any architecture at this time.

<sup>2</sup>The 2D CPU codepaths have not beem SIMDified yet.

<sup>3</sup>Multi-GPU support is available on CUDA (and potentially ROCm,
if I can get my hands on a multi-GPU ROCm machine), but it currently exhibits poor
scaling properties.

Dependencies
------------

rakau has the following mandatory dependencies:

* the [TBB library](https://www.threadingbuildingblocks.org/),
* the [xsimd library](https://github.com/QuantStack/xsimd),
* the [Boost libraries](https://www.boost.org) (the header-only parts are sufficient,
  apart from the benchmark suite which needs the compiled Boost.Program_options library).

In order to run on AMD GPUs, rakau must be compiled with the HCC compiler from the
[ROCm toolchain](https://rocm.github.io/). Support for Nvidia GPUs requires the
[CUDA](https://en.wikipedia.org/wiki/CUDA) software stack.

rakau is written in C++17, thus a reasonably recent (and conforming) C++ compiler is required.
GCC 7/8 and clang 6/7 are the main compilers used during development.

Installation
------------

rakau uses the [CMake](https://cmake.org/) build system. The main configuration variables
are:

* ``RAKAU_BUILD_BENCHMARKS``: build the benchmark suite,
* ``RAKAU_BUILD_TESTS``: build the test suite,
* ``RAKAU_WITH_ROCM``: enable support for AMD GPUs via ROCm,
* ``RAKAU_WITH_CUDA``: enable support for Nvidia GPUs via CUDA.

If no GPU support is enabled, rakau is a header-only library. If support
for AMD or Nvidia GPUs is enabled, a dynamic library will be built and installed
in addition to the header files.

rakau's build system installs a CMake config-file package which allows to easily
find and use rakau from other CMake-based projects. A minimal example:

```cmake
# Locate rakau on the system.
find_package(rakau)

# Link rakau (and its dependencies) to an executable.
target_link_libraries(my_executable rakau::rakau)
```

Usage
-----

rakau's usual workflow involves two basic operations:

* the construction of the tree structure from a distribution of
  particles in space,
* the traversal of the tree structure for the computation of the
  gravitational accelerations/potentials on the particles.

A minimal example:

```c++
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
```

More examples and details are available in the user guide (TODO).

Acknowledgements
----------------

The development of rakau was in part supported by the German
Deutsche Forschungsgemeinschaft (DFG) priority program 1833, "Building a Habitable Earth".

<img src="https://github.com/bluescarni/rakau/raw/master/spp1833.png" width="150">
