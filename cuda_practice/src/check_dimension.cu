#include <cuda_runtime.h>
#include <stdio.h>


__global__
void checkIndex()
{
    auto [x, y, z] = threadIdx;
    auto [bx, by, bz] = blockIdx;
    if (x + y + z + bx + by + bz == 0) {
        printf("blockDim.x: %d, blockDim.y: %d, blockDim.z: %d\n", blockDim.x, blockDim.y, blockDim.z);
        printf("gridDim.x: %d, gridDim.y: %d, gridDim.z: %d\n", gridDim.x, gridDim.y, gridDim.z);
    }
}

int main()
{
    // checkIndex<<<10, 10>>>();
    checkIndex<<<dim3(1, 2, 3), 10>>>();
    // checkIndex<<<10, dim3(10, 10, 10)>>>();
    cudaDeviceSynchronize();
    return 0;
}