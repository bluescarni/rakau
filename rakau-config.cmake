# Get current dir.
get_filename_component(_RAKAU_CONFIG_SELF_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)

# Find the deps. Alter the cmake module path.
set(_RAKAU_CONFIG_OLD_MODULE_PATH "${CMAKE_MODULE_PATH}")
list(APPEND CMAKE_MODULE_PATH "${_RAKAU_CONFIG_SELF_DIR}")
include(RakauFindBoost)
find_package(xsimd REQUIRED)
find_package(TBB REQUIRED)
# Restore the original module path.
set(CMAKE_MODULE_PATH "${_RAKAU_CONFIG_OLD_MODULE_PATH}")
unset(_RAKAU_CONFIG_OLD_MODULE_PATH)

include(${_RAKAU_CONFIG_SELF_DIR}/rakau_export.cmake)

# Clean up.
unset(_RAKAU_CONFIG_SELF_DIR)
