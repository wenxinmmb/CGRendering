cmake_minimum_required(VERSION 2.8)
project( CudaAssignment )

find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -L /usr/local/cuda-7.5/lib64 -I /usr/local/cuda-7.5/include/
    )


# For compilation ...
# Specify target & source files to compile it from
file(GLOB  hdr *.hpp *.h )
file(GLOB  cu  *.cu)
SET (cpp_files main.cpp)

CUDA_ADD_EXECUTABLE(cudaAssignment ${cpp_files} ${hdr} ${cu})

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(
    cudaAssignment
    ${OpenCV_LIBS}
)
