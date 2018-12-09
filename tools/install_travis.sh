#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
export PATH="$deps_dir/bin:$PATH"

if [[ "${TRAVIS_OS_NAME}" != "osx" ]]; then
    echo "Processor name: `cat /proc/cpuinfo|grep -i "model name"|uniq|awk -F ':' '{print $2}'`"
fi

if [[ "${RAKAU_BUILD}" == "osx_debug" ]]; then
    CXX=clang++ cmake -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DRAKAU_BUILD_TESTS=yes -DRAKAU_TEST_NSPLIT=${TEST_NSPLIT} -DRAKAU_TEST_SPLIT_NUM=${SPLIT_TEST_NUM} ../;
    make -j2 VERBOSE=1;
    ctest -V;
elif [[ "${RAKAU_BUILD}" == "gcc7_debug" ]]; then
    CXX=g++-7 cmake -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DRAKAU_BUILD_TESTS=yes -DCMAKE_CXX_FLAGS="-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC" -DRAKAU_TEST_NSPLIT=${TEST_NSPLIT} -DRAKAU_TEST_SPLIT_NUM=${SPLIT_TEST_NUM} ../;
    make -j2 VERBOSE=1;
    ctest -V;
elif [[ "${RAKAU_BUILD}" == "gcc7_debug_nosimd" ]]; then
    CXX=g++-7 cmake -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DRAKAU_BUILD_TESTS=yes -DCMAKE_CXX_FLAGS="-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -DRAKAU_DISABLE_SIMD" -DRAKAU_TEST_NSPLIT=${TEST_NSPLIT} -DRAKAU_TEST_SPLIT_NUM=${SPLIT_TEST_NUM} ../;
    make -j2 VERBOSE=1;
    ctest -V;
elif [[ "${RAKAU_BUILD}" == "gcc7_debug_native" ]]; then
    CXX=g++-7 cmake -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DRAKAU_BUILD_TESTS=yes -DCMAKE_CXX_FLAGS="-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -march=native" -DRAKAU_TEST_NSPLIT=${TEST_NSPLIT} -DRAKAU_TEST_SPLIT_NUM=${SPLIT_TEST_NUM} ../;
    make -j2 VERBOSE=1;
    ctest -V;
elif [[ "${RAKAU_BUILD}" == "gcc7_debug_native_norsqrt" ]]; then
    CXX=g++-7 cmake -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DRAKAU_BUILD_TESTS=yes -DRAKAU_ENABLE_RSQRT=no -DCMAKE_CXX_FLAGS="-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -march=native" -DRAKAU_TEST_NSPLIT=${TEST_NSPLIT} -DRAKAU_TEST_SPLIT_NUM=${SPLIT_TEST_NUM} ../;
    make -j2 VERBOSE=1;
    ctest -V;
fi
