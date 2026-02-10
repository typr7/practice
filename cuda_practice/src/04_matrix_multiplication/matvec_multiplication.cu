#include <cstdio>
#include <cstdlib>
#include <ctime>

static constexpr int B_ROWS = 5;
static constexpr int B_COLS = 5;

static constexpr int C_DIM = 5;

__global__ void
matvecMul(const float* B_d, const float* C_d, float* a_d)
{
    const int ai = blockDim.x * blockIdx.x + threadIdx.x;
    if (ai < B_ROWS) {
        double sum = 0.0;
        for (int i = 0; i < B_COLS; i++) {
            const int bi = ai * B_COLS + i;
            const int ci = i;
            sum += B_d[bi] * C_d[ci];
        }
        a_d[ai] = static_cast<float>(sum);
    }
}

void plainMatvecMul(const float* B, const float* C, float* a)
{
    for (int ai = 0; ai < B_ROWS; ai++) {
        double sum = 0.0;
        for (int i = 0; i < B_COLS; i++) {
            const int bi = ai * B_COLS + i;
            const int ci = i;
            sum += B[bi] * C[ci];
        }
        a[ai] = static_cast<float>(sum);
    }
}

void init_matvec(float* mat,
                 const int mat_size,
                 float* vec,
                 const int vec_size)
{
    srand(time(nullptr));
    for (int i = 0; i < mat_size; i++) {
        mat[i] = (rand() % 1000000) / 1000.f;
    }
    for (int i = 0; i < vec_size; i++) {
        vec[i] = (rand() % 1000000) / 1000.f;
    }
}

float check(const float* a, const float* a_plain)
{
    float sum = 0.f;
    for (int i = 0; i < B_ROWS; i++) {
        sum += std::abs(a[i] - a_plain[i]);
    }
    return sum;
}

int main()
{
    const int B_size = sizeof(float) * B_ROWS * B_COLS;
    const int C_size = sizeof(float) * C_DIM;
    const int a_size = sizeof(float) * B_ROWS;
    float* B = static_cast<float*>(malloc(B_size));
    float* C = static_cast<float*>(malloc(C_size));
    float* a = static_cast<float*>(malloc(a_size));
    float* a_plain = static_cast<float*>(malloc(a_size));
    init_matvec(B, B_ROWS * B_COLS, C, C_DIM);

    float* B_d = nullptr;
    float* C_d = nullptr;
    float* a_d = nullptr;
    cudaMalloc(&B_d, B_size);
    cudaMalloc(&C_d, C_size);
    cudaMalloc(&a_d, a_size);

    cudaMemcpy(B_d, B, B_size, cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, C_size, cudaMemcpyHostToDevice);

    dim3 grid_dim(2, 1, 1);
    dim3 block_dim(std::ceil(B_ROWS / 2.f), 1, 1);
    matvecMul<<<grid_dim, block_dim>>>(B_d, C_d, a_d);
    cudaDeviceSynchronize();

    cudaMemcpy(a, a_d, a_size, cudaMemcpyDeviceToHost);

    plainMatvecMul(B, C, a_plain);

    printf("err_sum: %f\n", check(a, a_plain));

    free(B);
    free(C);
    free(a);
    free(a_plain);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaFree(a_d);

    return 0;
}