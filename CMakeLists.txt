### CmakeLists.txt for CUDA 1D Swept Rule


cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# NVCC compiler flags
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_35,code=sm_35 -lm -std=c++11
)

# Target and source files respectively
cuda_add_executable(

    SweptOut
    1DSweptRule_maintester.cu

)
