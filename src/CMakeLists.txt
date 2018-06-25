cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(CuBin_main LANGUAGES CXX CUDA)


target_compile_features(CuBin_main PUBLIC cxx_std_11)

set(PROJECT_SOURCES CuBin_main.cu CuBin_gpu.cuh bit_vector_kernels.cuh float_kernels.cuh CuBin_cpu.h config.h io_and_allocation.hpp updates_and_measures.cuh helper/args_parser.h helper/rngpu.hpp helper/cuda_helpers.cuh)

add_executable(CuBin ${PROJECT_SOURCES})
target_include_directories(CuBin PRIVATE helper)
