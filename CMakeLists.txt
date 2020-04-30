cmake_minimum_required(VERSION 3.3.0)

# Set default build type to "Release".
# NOTE: this should be done before the project command since the latter can set
# CMAKE_BUILD_TYPE itself (it does so for nmake).
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
	FORCE)
endif()

project(rakau VERSION 0.1 LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/yacma")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "rakau version: ${rakau_VERSION}")

# The build options.
option(RAKAU_BUILD_TESTS "Build unit tests." OFF)
option(RAKAU_BUILD_BENCHMARKS "Build benchmarks." OFF)
option(RAKAU_ENABLE_RSQRT "Enable the use of rsqrt intrinsics." ON)
option(RAKAU_WITH_ROCM "Enable support for ROCm." OFF)
option(RAKAU_WITH_CUDA "Enable support for CUDA." OFF)

# YACMA compiler setup.
include(YACMACompilerLinkerSettings)

if(NOT RAKAU_ENABLE_RSQRT)
  set(RAKAU_DISABLE_RSQRT "#define RAKAU_DISABLE_RSQRT")
endif()

if(RAKAU_WITH_CUDA AND RAKAU_WITH_ROCM)
  message(FATAL_ERROR "ROCm and CUDA support cannot be activated together.")
endif()

if(RAKAU_WITH_ROCM)
  if(NOT YACMA_COMPILER_IS_HCC)
    message(FATAL_ERROR "ROCm support requires the hcc compiler.")
  endif()
  set(RAKAU_ENABLE_ROCM "#define RAKAU_WITH_ROCM")
endif()

if(RAKAU_WITH_CUDA)
  set(RAKAU_ENABLE_CUDA "#define RAKAU_WITH_CUDA")
  enable_language(CUDA)
endif()

# Assemble the flags.
set(RAKAU_CXX_FLAGS_DEBUG ${YACMA_CXX_FLAGS} ${YACMA_CXX_FLAGS_DEBUG})
set(RAKAU_CXX_FLAGS_RELEASE ${YACMA_CXX_FLAGS})
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND YACMA_COMPILER_IS_CLANGXX)
  message(STATUS "Clang compiler on OSX detected, setting the standard library to 'libc++'.")
  list(APPEND RAKAU_CXX_FLAGS_DEBUG "-stdlib=libc++")
  list(APPEND RAKAU_CXX_FLAGS_RELEASE "-stdlib=libc++")
endif()
if(YACMA_COMPILER_IS_MSVC)
  # Disable the idiotic minmax macros on MSVC, some annoying warnings,
  # and enable the bigobj option.
  list(APPEND RAKAU_CXX_FLAGS_DEBUG "-DNOMINMAX" "/wd4459" "/wd4127" "/bigobj")
  list(APPEND RAKAU_CXX_FLAGS_RELEASE "-DNOMINMAX" "/wd4459" "/wd4127" "/bigobj")
  if(YACMA_COMPILER_IS_CLANGXX)
    # clang-cl emits various warnings from Boost, let's just silence them.
    # NOTE: at one point in the recent past, MSVC added an options similar to GCC's isystem:
    # https://blogs.msdn.microsoft.com/vcblog/2017/12/13/broken-warnings-theory/
    # We probably just need to wait for this to be picked up by CMake/clang-cl. Let's
    # revisit the issue in the future.
    list(APPEND _RAKAU_CLANG_CL_DISABLED_WARNINGS
        "-Wno-deprecated-dynamic-exception-spec"
        "-Wno-old-style-cast"
        "-Wno-sign-conversion")
    list(APPEND RAKAU_CXX_FLAGS_DEBUG ${_RAKAU_CLANG_CL_DISABLED_WARNINGS})
    list(APPEND RAKAU_CXX_FLAGS_RELEASE ${_RAKAU_CLANG_CL_DISABLED_WARNINGS})
    unset(_RAKAU_CLANG_CL_DISABLED_WARNINGS)
  endif()
endif()
if(YACMA_COMPILER_IS_INTELXX)
  # NOTE: on MSVC we use the push/pop pragmas, but they do not seem to work on Intel (the pragmas
  # in icc influence the behaviour at instantiation point, not at definition point).
  list(APPEND RAKAU_CXX_FLAGS_DEBUG "-diag-disable" "3373,1682")
  list(APPEND RAKAU_CXX_FLAGS_RELEASE "-diag-disable" "3373,1682")
endif()
if(MINGW)
  # In MinGW some tests generate big object files.
  list(APPEND RAKAU_CXX_FLAGS_DEBUG "-Wa,-mbig-obj")
  list(APPEND RAKAU_CXX_FLAGS_RELEASE "-Wa,-mbig-obj")
endif()

# Find the dependencies.

# Boost.
if (RAKAU_BUILD_BENCHMARKS)
  set(_RAKAU_BOOST_COMPONENTS program_options)
else()
  set(_RAKAU_BOOST_COMPONENTS)
endif()
include(RakauFindBoost)
unset(_RAKAU_BOOST_COMPONENTS)

# xsimd.
find_package(xsimd REQUIRED)

# tbb.
find_package(TBB REQUIRED)

# Threading setup.
include(RakauFindThreads)

# NOTE: on Unix systems, the correct library installation path
# could be something other than just "lib", such as "lib64",
# "lib32", etc., depending on platform/configuration. Apparently,
# CMake provides this information via the GNUInstallDirs module.
# Let's enable this for now on all Unixes except OSX.
# NOTE: potentially, this could be applicable to Cygwin as well.
#
# https://cmake.org/cmake/help/v3.15/module/GNUInstallDirs.html
# https://cmake.org/pipermail/cmake/2013-July/055375.html
if(UNIX AND NOT APPLE)
    include(GNUInstallDirs)
    set(_RAKAU_INSTALL_LIBDIR_DEFAULT "${CMAKE_INSTALL_LIBDIR}")
else()
    set(_RAKAU_INSTALL_LIBDIR_DEFAULT "lib")
endif()
if(NOT RAKAU_INSTALL_LIBDIR)
    set(RAKAU_INSTALL_LIBDIR "${_RAKAU_INSTALL_LIBDIR_DEFAULT}" CACHE STRING
        "Library installation directory." FORCE)
endif()
mark_as_advanced(RAKAU_INSTALL_LIBDIR)
message(STATUS "Library installation directory: ${RAKAU_INSTALL_LIBDIR}")

# Initial setup of the rakau library.
if(RAKAU_WITH_ROCM)
  add_library(rakau SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/rakau_rocm.cpp")
elseif(RAKAU_WITH_CUDA)
  add_library(rakau SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/rakau_cuda.cu")
else()
  add_library(rakau INTERFACE)
endif()

# Add the include dirs to the rakau target's interface.
target_include_directories(rakau INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# Link the deps.
target_link_libraries(rakau INTERFACE Boost::boost xsimd TBB::tbb Threads::Threads)

# Additional ROCm-specific setup.
if(RAKAU_WITH_ROCM)
  # Introduce a separate variable containing extra compiler flags
  # to be used *exclusively* when building ROCm device code (and not,
  # e.g., when building the tests or the benchmarks). We can use this
  # variable for setting, e.g., -ffast-math only for device code, or
  # for emitting device code only for a specific gfx arch, etc.
  set(RAKAU_ROCM_EXTRA_FLAGS "" CACHE STRING "Extra flags for ROCm device code.")
  mark_as_advanced(RAKAU_ROCM_EXTRA_FLAGS)
  string(REPLACE " " ";" RAKAU_ROCM_EXTRA_FLAGS_ "${RAKAU_ROCM_EXTRA_FLAGS}")

  # NOTE: we need to manually add -hc and the extra flags to the link flags as well.
  # NOTE: very recent CMake versions have new target properties for the link flags,
  # let's keep it in mind.
  set_target_properties(rakau PROPERTIES LINK_FLAGS "-hc ${RAKAU_ROCM_EXTRA_FLAGS}")

  # Visibility settings.
  # NOTE: disable these for the time being, as they do not seem to work
  # properly. To be revisited in later hcc versions.
  # set_target_properties(rakau PROPERTIES CXX_VISIBILITY_PRESET hidden)
  # set_target_properties(rakau PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)

  # Assemble the flags.
  target_compile_options(rakau PRIVATE "$<$<CONFIG:DEBUG>:${RAKAU_CXX_FLAGS_DEBUG}>" "$<$<CONFIG:RELEASE>:${RAKAU_CXX_FLAGS_RELEASE}>"
    "-hc" "-Wno-unused-command-line-argument" "${RAKAU_ROCM_EXTRA_FLAGS_}")
  unset(RAKAU_ROCM_EXTRA_FLAGS_)

  # Set the C++17 requirement.
  set_property(TARGET rakau PROPERTY CXX_STANDARD 17)
  set_property(TARGET rakau PROPERTY CXX_STANDARD_REQUIRED YES)
  set_property(TARGET rakau PROPERTY CXX_EXTENSIONS NO)

  # Add the rakau includes to the private include dirs of the library.
  target_include_directories(rakau PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)
endif()

# Additional CUDA setup.
if(RAKAU_WITH_CUDA)
  # Visibility settings.
  set_target_properties(rakau PROPERTIES CXX_VISIBILITY_PRESET hidden)
  set_target_properties(rakau PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)

  # Highest supported C++ version is C++14 currently.
  target_compile_features(rakau PRIVATE cxx_std_14)

  # Add the rakau includes to the private include dirs of the library.
  target_include_directories(rakau PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)
endif()

# Configure config.hpp.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/include/rakau/config.hpp" @ONLY)

# Installation.
# Setup of the export.
set(_RAKAU_CONFIG_OPTIONAL_DEPS)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/rakau-config.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/rakau-config.cmake" @ONLY)
if(RAKAU_WITH_ROCM OR RAKAU_WITH_CUDA)
  install(TARGETS rakau EXPORT rakau_export LIBRARY DESTINATION "${RAKAU_INSTALL_LIBDIR}")
else()
  install(TARGETS rakau EXPORT rakau_export)
endif()
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rakau-config.cmake" DESTINATION "${RAKAU_INSTALL_LIBDIR}/cmake/rakau")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RakauFindBoost.cmake" DESTINATION "${RAKAU_INSTALL_LIBDIR}/cmake/rakau")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTBB.cmake" DESTINATION "${RAKAU_INSTALL_LIBDIR}/cmake/rakau")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RakauFindThreads.cmake" DESTINATION "${RAKAU_INSTALL_LIBDIR}/cmake/rakau")
install(EXPORT rakau_export NAMESPACE rakau:: DESTINATION ${RAKAU_INSTALL_LIBDIR}/cmake/rakau)
# Take care of versioning.
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/rakau-config-version.cmake" VERSION ${rakau_VERSION}
	COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rakau-config-version.cmake" DESTINATION "${RAKAU_INSTALL_LIBDIR}/cmake/rakau")
unset(_RAKAU_CONFIG_OPTIONAL_DEPS)

# Installation of the header files.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/rakau" DESTINATION include)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/rakau/config.hpp" DESTINATION include/rakau)

# This is just a simple counter variable, internal use only.
set(_RAKAU_TEST_NUM "0")
# Check splitting options. These need to be set from the command line.
# - RAKAU_TEST_NSPLIT: number of chunks into which the unit tests will be divided (must be > 1).
# - RAKAU_TEST_SPLIT_NUM: 0-based index of the chunk to run.
if(RAKAU_TEST_NSPLIT AND "${RAKAU_TEST_SPLIT_NUM}" STREQUAL "")
	message(FATAL_ERROR "Test splitting was requested, but the RAKAU_TEST_SPLIT_NUM variable was not set.")
elseif(NOT RAKAU_TEST_NSPLIT AND NOT "${RAKAU_TEST_SPLIT_NUM}" STREQUAL "")
	message(FATAL_ERROR "The RAKAU_TEST_SPLIT_NUM variable was set, but test splitting was not requested.")
endif()
if(RAKAU_TEST_NSPLIT)
	message(STATUS "Tests will be split into ${RAKAU_TEST_NSPLIT} chunks. The chunk with index ${RAKAU_TEST_SPLIT_NUM} will be processed.")
endif()

if(RAKAU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(RAKAU_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
