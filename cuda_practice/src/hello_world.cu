#include <cstdio>
#include <cstdlib>
#include <ctime>

__global__
void actualVecAdd(const float* d_a, const float* d_b, float* d_c, const int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_c[idx] = d_a[idx] + d_b[idx];
    }
}

void vecAdd(const float* A, const float* B, float* C, const int n)
{
    int size = n * sizeof(float);
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    // part 1: allocate device memory for A, B, C
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy A and B to device memory
    cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);

    // part 2: call kernel - to launch a grid of threads
    // to perform the actual vector addition
    constexpr int block_size = 128;
    actualVecAdd<<<(n + block_size - 1) / block_size, block_size>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    // part3: copy C from the device memory
    cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);

    // Free device vectors
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void plainVecAdd(const float* A, const float* B, float* C, int n)
{
    for (int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    srand(time(nullptr));

    const int vec_length = 10000000;
    const int vec_size = vec_length * sizeof(float);

    float* vec_a = static_cast<float*>(malloc(vec_size));
    float* vec_b = static_cast<float*>(malloc(vec_size));
    float* vec_c_cuda = static_cast<float*>(malloc(vec_size));
    float* vec_c_host = static_cast<float*>(malloc(vec_size));

    // random init
    for (int i = 0; i < vec_length; i++) {
        vec_a[i] = (rand() % 1000) / 1000.f;
        vec_b[i] = (rand() % 1000) / 1000.f;
        vec_c_cuda[i] = vec_c_host[i] = 0.f;
    }

    vecAdd(vec_a, vec_b, vec_c_cuda, vec_length);
    plainVecAdd(vec_a, vec_b, vec_c_host, vec_length);

    // check
    float sum_diff = 0.f;
    for (int i = 0; i < vec_length; i++) {
        if (vec_c_host[i] != vec_c_cuda[i]) {
            sum_diff += std::abs(vec_c_host[i] - vec_c_cuda[i]);
        }
    }

    printf("sum_diff: %f\n", sum_diff);

    return 0;
}
