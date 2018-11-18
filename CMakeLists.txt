cmake_minimum_required(VERSION 3.3.0)

project(rakau VERSION 0.1 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/yacma")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "rakau version: ${rakau_VERSION}")

# Set default build type to "Release".
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
	FORCE)
endif()

# The build options.
option(RAKAU_BUILD_TESTS "Build unit tests." OFF)
option(RAKAU_BUILD_BENCHMARKS "Build benchmarks." OFF)
option(RAKAU_ENABLE_RSQRT "Enable the use of rsqrt intrinsics." ON)
option(RAKAU_WITH_ROCM "Enable support for ROCm." OFF)

if(NOT RAKAU_ENABLE_RSQRT)
  set(RAKAU_DISABLE_RSQRT "#define RAKAU_DISABLE_RSQRT")
endif()

include(YACMACompilerLinkerSettings)
include(CheckCXXCompilerFlag)

if(RAKAU_WITH_ROCM)
  if(NOT YACMA_COMPILER_IS_HCC)
    message(FATAL_ERROR "ROCm support requires hcc compiler.")
  endif()
  set(RAKAU_ENABLE_ROCM "#define RAKAU_WITH_ROCM")
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

# Setup of the rakau interface library.
add_library(rakau INTERFACE)

# Find Boost.
if (RAKAU_BUILD_BENCHMARKS)
  set(_RAKAU_BOOST_COMPONENTS program_options)
else()
  set(_RAKAU_BOOST_COMPONENTS)
endif()
include(RakauFindBoost)
unset(_RAKAU_BOOST_COMPONENTS)

# Find xsimd.
find_package(xsimd REQUIRED)

# Find TBB.
find_package(TBB REQUIRED)

# Find the threading library.
include(RakauFindThreads)

# Link the deps.
target_link_libraries(rakau INTERFACE Boost::boost xsimd TBB::tbb Threads::Threads)

# Finish setting up the rakau interface target.
target_include_directories(rakau INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

if(RAKAU_WITH_ROCM)
  set(RAKAU_ROCM_DEVICE_FLAGS "" CACHE STRING "Flags for ROCm device code compilation.")
  mark_as_advanced(RAKAU_ROCM_DEVICE_FLAGS)
  string(REPLACE " " ";" RAKAU_ROCM_DEVICE_FLAGS_ "${RAKAU_ROCM_DEVICE_FLAGS}")
  add_library(rakau_rocm SHARED "${CMAKE_CURRENT_SOURCE_DIR}/src/rakau_rocm.cpp")
  set_target_properties(rakau_rocm PROPERTIES LINK_FLAGS "-hc ${RAKAU_ROCM_DEVICE_FLAGS}")
  target_compile_options(rakau_rocm PRIVATE "$<$<CONFIG:DEBUG>:${RAKAU_CXX_FLAGS_DEBUG}>" "$<$<CONFIG:RELEASE>:${RAKAU_CXX_FLAGS_RELEASE}>"
    "-hc" "-Wno-unused-command-line-argument" "${RAKAU_ROCM_DEVICE_FLAGS_}")
  unset(RAKAU_ROCM_DEVICE_FLAGS_)
  set_property(TARGET rakau_rocm PROPERTY CXX_STANDARD 17)
  set_property(TARGET rakau_rocm PROPERTY CXX_STANDARD_REQUIRED YES)
  set_property(TARGET rakau_rocm PROPERTY CXX_EXTENSIONS NO)

  target_link_libraries(rakau INTERFACE rakau_rocm)
  target_link_libraries(rakau_rocm PRIVATE rakau)
endif()

# Configure config.hpp.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in" "${CMAKE_CURRENT_BINARY_DIR}/include/rakau/config.hpp" @ONLY)

# Installation.
# Setup of the export.
install(TARGETS rakau EXPORT rakau_export)
if(RAKAU_WITH_ROCM)
  install(TARGETS rakau_rocm EXPORT rakau_export LIBRARY DESTINATION "lib")
endif()
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/rakau-config.cmake" DESTINATION "lib/cmake/rakau")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RakauFindBoost.cmake" DESTINATION "lib/cmake/rakau")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTBB.cmake" DESTINATION "lib/cmake/rakau")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/RakauFindThreads.cmake" DESTINATION "lib/cmake/rakau")
install(EXPORT rakau_export NAMESPACE rakau:: DESTINATION lib/cmake/rakau)
# Take care of versioning.
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/rakau-config-version.cmake" VERSION ${rakau_VERSION}
	COMPATIBILITY SameMajorVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rakau-config-version.cmake" DESTINATION "lib/cmake/rakau")

# Installation of the header files.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/rakau" DESTINATION include)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/rakau/config.hpp" DESTINATION include/rakau)

if(RAKAU_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(RAKAU_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()
