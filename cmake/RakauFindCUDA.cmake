find_package(CUDA REQUIRED)

message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")

add_library(rakau_cuda UNKNOWN IMPORTED)
set_target_properties(rakau_cuda PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
set_target_properties(rakau_cuda PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${CUDA_LIBRARIES}")
