#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
export PATH="$deps_dir/bin:$PATH"

if [[ "${RAKAU_BUILD}" == "gcc7_debug" ]]; then
    CXX=g++-7 CC=gcc-7 cmake -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DRAKAU_BUILD_TESTS=yes -DCMAKE_CXX_FLAGS="-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC" ../;
    make -j2 VERBOSE=1;
    ctest -V;
elif [[ "${RAKAU_BUILD}" == "gcc7_debug_nosimd" ]]; then
    CXX=g++-7 CC=gcc-7 cmake -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DRAKAU_BUILD_TESTS=yes -DCMAKE_CXX_FLAGS="-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -DRAKAU_DISABLE_SIMD" ../;
    make -j2 VERBOSE=1;
    ctest -V;
fi
