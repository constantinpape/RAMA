project(parallel_gaec LANGUAGES CXX CUDA)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -rdynamic -lineinfo")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")

cmake_minimum_required(VERSION 2.8.12)

set(CMAKE_CUDA_ARCHITECTURES 70) #Use JIT compilation? 

include_directories(external/thrust/)
include_directories(external/thrust/dependencies/cub)

add_library(parallel_gaec INTERFACE)
target_include_directories(parallel_gaec INTERFACE include/)
target_compile_features(parallel_gaec INTERFACE cxx_std_17)
target_compile_options(parallel_gaec INTERFACE -march=native)
target_compile_options(parallel_gaec INTERFACE -fPIC)
target_include_directories(parallel_gaec INTERFACE external/Eigen)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(parallel_gaec INTERFACE OpenMP::OpenMP_CXX)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_CPP_STANDARD -std=c++17)
set(PYBIND11_INSTALL ON CACHE BOOL "enable pybind11 bindings.")
add_subdirectory(external/pybind11)

add_library(multicut_text_parser multicut_text_parser.cpp)

add_library(parallel_gaec_eigen parallel_gaec_eigen.cpp)
target_link_libraries(parallel_gaec_eigen parallel_gaec)

add_executable(parallel_gaec_eigen_test parallel_gaec_eigen_test.cpp)
target_link_libraries(parallel_gaec_eigen_test parallel_gaec_eigen)

# SpECK:
add_subdirectory(external/spECK)

add_library(dCSR dCSR.cu)
target_include_directories(dCSR PUBLIC external/spECK/include)

target_link_libraries(dCSR cusparse ECL-CC_11 spECKLib)

add_library(ECL-CC_11 external/ECL-CC/ECL-CC_11.cu)
target_include_directories(ECL-CC_11 INTERFACE external/ECL-CC)

add_executable(cc_gpu_test cc_gpu_test.cu)
target_link_libraries(cc_gpu_test ECL-CC_11 cusparse)

add_library(icp_small_cycles icp_small_cycles.cu)
target_link_libraries(icp_small_cycles dCSR cusparse)

add_library(maximum_matching maximum_matching/maximum_matching.cu)
add_library(maximum_matching_vertex_based maximum_matching/maximum_matching_vertex_based.cu)
target_link_libraries(maximum_matching_vertex_based cusparse)

add_library(parallel_gaec_cuda parallel_gaec_cuda.cu)
target_link_libraries(parallel_gaec_cuda dCSR cusparse icp_small_cycles maximum_matching maximum_matching_vertex_based)

add_executable(icp_test icp_test.cu)
target_link_libraries(icp_test cusparse icp_small_cycles)

add_executable(parallel_gaec_cuda_test parallel_gaec_cuda_test.cpp)
target_link_libraries(parallel_gaec_cuda_test parallel_gaec_cuda)

pybind11_add_module(parallel_gaec_py parallel_gaec_py.cpp)
target_link_libraries(parallel_gaec_py PRIVATE parallel_gaec_eigen multicut_text_parser parallel_gaec_cuda)

add_executable(multicut_parallel_gaec_eigen_text_input multicut_parallel_gaec_eigen_text_input.cpp)
target_link_libraries(multicut_parallel_gaec_eigen_text_input multicut_text_parser parallel_gaec_eigen)

add_executable(multicut_parallel_gaec_cuda_text_input multicut_parallel_gaec_cuda_text_input.cpp)
target_link_libraries(multicut_parallel_gaec_cuda_text_input multicut_text_parser parallel_gaec_cuda dCSR)

add_executable(maximum_matching_vertex_based_test maximum_matching/maximum_matching_vertex_based_test.cu)
target_link_libraries(maximum_matching_vertex_based_test cusparse dCSR maximum_matching_vertex_based)

add_executable(icp_cuda_text_input icp_cuda_text_input.cpp)
target_link_libraries(icp_cuda_text_input icp_small_cycles multicut_text_parser)
