# SPDX-License-Identifier: PolyForm-Small-Business-1.0.0
cmake_minimum_required(VERSION 3.28)

set(CMAKE_CUDA_ARCHITECTURES 89)
project(quantim_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(QUANTIM_GPU_TESTS "Build and register tests" ON)
# The vendored project uses BUILD_TESTING through CTest, so both test switches must agree.
set(BUILD_TESTING "${QUANTIM_GPU_TESTS}" CACHE BOOL "Build tests" FORCE)
include(CTest)

add_subdirectory(extern/quantim/cpp/quantum)

add_executable(gpu_probe tools/probe.cu)
target_compile_options(gpu_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo;-Xcompiler=-Wall,-Wextra>)

find_path(NVML_INCLUDE_DIR nvml.h)
find_library(NVML_LIBRARY NAMES nvidia-ml)
if(NVML_INCLUDE_DIR AND NVML_LIBRARY)
    target_include_directories(gpu_probe PRIVATE "${NVML_INCLUDE_DIR}")
    target_link_libraries(gpu_probe PRIVATE "${NVML_LIBRARY}")
    target_compile_definitions(gpu_probe PRIVATE QUANTIM_HAVE_NVML=1)
endif()

find_package(CUDAToolkit REQUIRED)

set(QUANTIM_GPU_CUDA_FLAGS --fmad=false -ftz=false -prec-div=true -prec-sqrt=true -lineinfo)

add_library(quantim_gpu_state STATIC src/device_state.cu)
target_include_directories(quantim_gpu_state PUBLIC src)
target_compile_options(quantim_gpu_state PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${QUANTIM_GPU_CUDA_FLAGS};-Xcompiler=-Wall,-Wextra>)
target_link_libraries(quantim_gpu_state PUBLIC CUDA::cudart)

add_library(quantim_gpu_runner STATIC src/runner.cpp)
target_link_libraries(quantim_gpu_runner PUBLIC quantim_gpu_state quantim_quantum)
target_compile_options(quantim_gpu_runner PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror
    -ffp-contract=off)

add_executable(quantim-gpu src/main.cpp)
target_link_libraries(quantim-gpu PRIVATE quantim_gpu_runner)
target_compile_options(quantim-gpu PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror)

if(QUANTIM_GPU_TESTS)
    add_executable(gpu_state_tests tests/state_test.cpp)
    target_include_directories(gpu_state_tests PRIVATE extern/quantim/cpp/quantum/src)
    target_link_libraries(gpu_state_tests PRIVATE quantim_gpu_state quantim_quantum)
    target_compile_options(gpu_state_tests PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror
        -ffp-contract=off)
    add_test(NAME gpu_state_tests COMMAND gpu_state_tests)
    set_tests_properties(gpu_state_tests PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)

    add_executable(gpu_measure_tests tests/measure_test.cpp)
    target_include_directories(gpu_measure_tests PRIVATE extern/quantim/cpp/quantum/src)
    target_link_libraries(gpu_measure_tests PRIVATE quantim_gpu_state quantim_quantum)
    target_compile_options(gpu_measure_tests PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror
        -ffp-contract=off)
    add_test(NAME gpu_measure_tests COMMAND gpu_measure_tests)
    set_tests_properties(gpu_measure_tests PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)

    add_executable(gpu_runner_tests tests/runner_test.cpp)
    target_include_directories(gpu_runner_tests PRIVATE extern/quantim/cpp/quantum/src)
    target_link_libraries(gpu_runner_tests PRIVATE quantim_gpu_runner)
    target_compile_options(gpu_runner_tests PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror
        -ffp-contract=off)
    add_test(NAME gpu_runner_tests COMMAND gpu_runner_tests)
    set_tests_properties(gpu_runner_tests PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu)

    # Run by hand, one size at a time, under tools/gpu_guard.sh.
    add_executable(gpu_large_check tests/large_check.cpp)
    target_include_directories(gpu_large_check PRIVATE extern/quantim/cpp/quantum/src)
    target_link_libraries(gpu_large_check PRIVATE quantim_gpu_state quantim_quantum)
    target_compile_options(gpu_large_check PRIVATE -Wall -Wextra -Wpedantic -Wshadow -Werror
        -ffp-contract=off)
endif()
