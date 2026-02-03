cmake_minimum_required(VERSION 3.24)

# Use clang for CUDA support (must be before project())
set(CMAKE_CXX_COMPILER "clang++")

project(hello_clang_cuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for LSP support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use clang for CUDA support
set(CMAKE_CXX_COMPILER "clang++")

# ---- Read from environment variables ----
set(CUDA_HOME "$ENV{CUDA_HOME}" CACHE PATH "CUDA Toolkit root (conda env path)")
set(CUDA_GPU_ARCH "$ENV{CUDA_GPU_ARCH}" CACHE STRING "CUDA GPU arch for clang (e.g. sm_70)")

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

add_executable(hello
  src/main.cu
  src/saxpy.cu
)

# Treat .cu as C++ source, compile with clang as CUDA via -x cuda
set_source_files_properties(src/main.cu src/saxpy.cu PROPERTIES LANGUAGE CXX)

target_include_directories(hello PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CUDA_INC_DIR1}"
  "${CUDA_INC_DIR2}"
)

target_compile_options(hello PRIVATE
  "SHELL:-x cuda"
  "SHELL:--cuda-path=${CUDA_HOME}"
  "SHELL:--cuda-gpu-arch=${CUDA_GPU_ARCH}"
)

target_link_libraries(hello PRIVATE
  "${CUDART_LIBRARY}"
)

# So you don't need to export LD_LIBRARY_PATH to run
set_target_properties(hello PROPERTIES
  BUILD_RPATH "${CUDA_LIB_DIR}"
  INSTALL_RPATH "${CUDA_LIB_DIR}"
)
