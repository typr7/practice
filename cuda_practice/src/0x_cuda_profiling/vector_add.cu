#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>


__global__
void vector_add_fp32_kernel_v1(
    int vec_len,
    const float* __restrict__ vec_a,
    const float* __restrict__ vec_b,
    float* __restrict__ vec_res
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_len) {
        vec_res[idx] = vec_a[idx] + vec_b[idx];
    }
}

void init_vec(std::vector<float>& vec)
{
    srand(time(nullptr));

    for (std::size_t i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<float>(rand() % 100) / 100.f;
    }
}

int main()
{
    constexpr int n = 1 << 24;
    constexpr std::size_t byte_size = sizeof(float) * n;
    constexpr int block_size = 256;

    std::vector<float> vec_a(n);
    std::vector<float> vec_b(n);
    std::vector<float> vec_res(n);

    init_vec(vec_a);
    init_vec(vec_b);

    float* vec_a_buffer = nullptr;
    float* vec_b_buffer = nullptr;
    float* vec_res_buffer = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&vec_a_buffer), byte_size);
    cudaMalloc(reinterpret_cast<void**>(&vec_b_buffer), byte_size);
    cudaMalloc(reinterpret_cast<void**>(&vec_res_buffer), byte_size);
    cudaMemcpy(vec_a_buffer, vec_a.data(), byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(vec_b_buffer, vec_b.data(), byte_size, cudaMemcpyHostToDevice);

    constexpr int grid_size = (n + (block_size - 1)) / block_size;
    vector_add_fp32_kernel_v1<<<grid_size, block_size>>>(
        n,
        vec_a_buffer,
        vec_b_buffer,
        vec_res_buffer
    );

    cudaMemcpy(vec_res.data(), vec_res_buffer, byte_size, cudaMemcpyDeviceToHost);

    cudaFree(vec_a_buffer);
    cudaFree(vec_b_buffer);
    cudaFree(vec_res_buffer);

    return 0;
}