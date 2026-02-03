cmake_minimum_required(VERSION 3.24)
project(hello_clang_cuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Configure these two as you like ----
set(CUDA_HOME "/home/bittervan/Tools/miniforge3/envs/cuda12" CACHE PATH "CUDA Toolkit root (conda env path)")
set(CUDA_GPU_ARCH "sm_70" CACHE STRING "CUDA GPU arch for clang (e.g. sm_70)")

# CUDA include/lib paths (conda-style layout)
set(CUDA_INC_DIR1 "${CUDA_HOME}/include")
set(CUDA_INC_DIR2 "${CUDA_HOME}/targets/x86_64-linux/include")
set(CUDA_LIB_DIR  "${CUDA_HOME}/targets/x86_64-linux/lib")

# Find libcudart
find_library(CUDART_LIBRARY
  NAMES cudart
  PATHS "${CUDA_LIB_DIR}"
  NO_DEFAULT_PATH
)
if (NOT CUDART_LIBRARY)
  message(FATAL_ERROR "Could not find libcudart in: ${CUDA_LIB_DIR}")
endif()

add_executable(hello hello.cu)

# Treat .cu as C++ source, compile with clang as CUDA via -x cuda
set_source_files_properties(hello.cu PROPERTIES LANGUAGE CXX)

target_compile_options(hello PRIVATE
  "SHELL:-x cuda"
  "SHELL:--cuda-path=${CUDA_HOME}"
  "SHELL:--cuda-gpu-arch=${CUDA_GPU_ARCH}"
)

target_include_directories(hello PRIVATE
  "${CUDA_INC_DIR1}"
  "${CUDA_INC_DIR2}"
)

target_link_libraries(hello PRIVATE
  "${CUDART_LIBRARY}"
)

# So you don't need to export LD_LIBRARY_PATH to run
set_target_properties(hello PROPERTIES
  BUILD_RPATH "${CUDA_LIB_DIR}"
  INSTALL_RPATH "${CUDA_LIB_DIR}"
)
