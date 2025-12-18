cmake_minimum_required(VERSION 3.15)
project(mlip_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Create the Python module
pybind11_add_module(_core MODULE
    src/cpp/src/density.cpp
    src/cpp/src/rdf.cpp
    src/cpp/src/diffusion.cpp
    src/cpp/bindings.cpp
)

target_include_directories(_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/include
)

# Enable OpenMP if available (optional, for future parallelization)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_core PRIVATE USE_OPENMP)
endif()

# Compiler optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(_core PRIVATE -O3 -march=native -ffast-math)
endif()

# Install the module to the mlip_plot package
install(TARGETS _core DESTINATION mlip_plot)
