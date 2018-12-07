find_package(CUDA REQUIRED)

message(STATUS "CUDA include dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA cudart library: ${CUDA_cudart_static_LIBRARY}")

add_library(rakau_cuda UNKNOWN IMPORTED)
set_target_properties(rakau_cuda PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
# NOTE: the cuda rt static library seems to be enough.
set_target_properties(rakau_cuda PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    IMPORTED_LOCATION "${CUDA_cudart_static_LIBRARY}")
# NOTE: apparently we need to link -ldl as well when using the cuda runtime.
target_link_libraries(rakau_cuda INTERFACE ${CMAKE_DL_LIBS})
