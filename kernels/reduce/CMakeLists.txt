cmake_minimum_required(VERSION 3.18)
project(mat_transpose_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(reduce reduce.cu)

target_include_directories(reduce PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
)

target_link_libraries(reduce
    CUDA::cudart
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -O3 \
    -U__CUDA_NO_HALF_OPERATORS__ \
    -U__CUDA_NO_HALF_CONVERSIONS__ \
    -U__CUDA_NO_HALF2_OPERATORS__ \
    -U__CUDA_NO_BFLOAT16_CONVERSIONS__ \
    --expt-relaxed-constexpr \
    --expt-extended-lambda \
    --use_fast_math \
    -lineinfo \
")

set_target_properties(reduce PROPERTIES
    CUDA_ARCHITECTURES "80"
)