#include <cstdio>
#include <cstdlib>
#include <ctime>


static constexpr int A_ROWS = 5;
static constexpr int A_COLS = 5;

static constexpr int B_ROWS = 5;
static constexpr int B_COLS = 5;

static_assert(A_COLS == B_ROWS, "");

void init_matrices(float* A,
                   const int A_size,
                   float* B,
                   const int B_size)
{
    srand(time(nullptr));
    for (int i = 0; i < A_size; i++) {
        A[i] = (rand() % 1000000) / 1000.f;
    }
    for (int i = 0; i < B_size; i++) {
        B[i] = (rand() % 1000000) / 1000.f;
    }
}

__global__ void
matrixMul(const float* A_d, const float* B_d, float* C_d)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < B_COLS && y < A_ROWS) {
        const int ci = y * B_COLS + x;
        double sum = 0.0;
        for (int i = 0; i < A_COLS; i++) {
            const int ai = y * A_COLS + i;
            const int bi = i * B_COLS + x;
            sum += A_d[ai] * B_d[bi];
        }
        C_d[ci] = static_cast<float>(sum);
    }
}

__global__ void
matrixMulCol(const float* A_d, const float* B_d, float* C_d)
{
    const int cx = blockDim.x * blockIdx.x + threadIdx.x;
    if (cx < B_COLS) {
        for (int cy = 0; cy < A_ROWS; cy++) {
            const int ci = cy * B_COLS + cx;
            double sum = 0.0;
            for (int i = 0; i < A_COLS; i++) {
                const int ai = cy * A_COLS + i;
                const int bi = i * B_COLS + cx;
                sum += A_d[ai] * B_d[bi];
            }
            C_d[ci] = static_cast<float>(sum);
        }
    }
}

__global__ void
matrixMulRow(const float* A_d, const float* B_d, float* C_d)
{
    const int cy = blockDim.x * blockIdx.x + threadIdx.x;
    if (cy < A_ROWS) {
        for (int cx = 0; cx < B_COLS; cx++) {
            const int ci = cy * B_COLS + cx;
            double sum = 0.0;
            for (int i = 0; i < A_COLS; i++) {
                const int ai = cy * A_COLS + i;
                const int bi = i * B_COLS + cx;
                sum += A_d[ai] * B_d[bi];
            }
            C_d[ci] = static_cast<float>(sum);
        }
    }
}

void plainMatrixMul(const float* A, const float* B, float* C)
{
    for (int cy = 0; cy < A_ROWS; cy++) {
        for (int cx = 0; cx < B_COLS; cx++) {
            const int ci = cy * B_COLS + cx;
            double sum = 0.0;
            for (int i = 0; i < A_COLS; i++) {
                const int ai = cy * A_COLS + i;
                const int bi = i * B_COLS + cx;
                sum += A[ai] * B[bi];
            }
            C[ci] = sum;
        }
    }
}

float check(const float* C, const float* C_plain)
{
    float err_sum = 0.f;
    for (int y = 0; y < A_ROWS; y++) {
        for (int x = 0; x < B_COLS; x++) {
            const int i = y * B_COLS + x;
            err_sum += std::abs(C[i] - C_plain[i]);
        }
    }
    return err_sum;
}

int main()
{
    const int A_size = sizeof(float) * A_ROWS * A_COLS;
    const int B_size = sizeof(float) * B_ROWS * B_COLS;
    const int C_size = sizeof(float) * A_ROWS * B_COLS;
    float* A = static_cast<float*>(malloc(A_size));
    float* B = static_cast<float*>(malloc(B_size));
    float* C = static_cast<float*>(malloc(C_size));
    float* C_plain = static_cast<float*>(malloc(C_size));
    init_matrices(A, A_ROWS * A_COLS, B, B_ROWS * B_COLS);

    float* A_d = nullptr;
    float* B_d = nullptr;
    float* C_d = nullptr;
    cudaMalloc(&A_d, A_size);
    cudaMalloc(&B_d, B_size);
    cudaMalloc(&C_d, C_size);


    cudaMemcpy(A_d, A, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, B_size, cudaMemcpyHostToDevice);

    // per element
    // dim3 block_dim(16, 16, 1);
    // dim3 grid_dim(std::ceil(B_COLS / 16.f), std::ceil(A_ROWS / 16.f), 1);
    // matrixMul<<<grid_dim, block_dim>>>(A_d, B_d, C_d);

    dim3 block_dim(2, 1, 1);
    dim3 grid_dim(std::ceil(B_COLS / 2.f), 1, 1);
    // per col
    // matrixMulCol<<<grid_dim, block_dim>>>(A_d, B_d, C_d);
    // per row
    matrixMulRow<<<grid_dim, block_dim>>>(A_d, B_d, C_d);

    cudaDeviceSynchronize();
    plainMatrixMul(A, B, C_plain);

    cudaMemcpy(C, C_d, C_size, cudaMemcpyDeviceToHost);

    printf("err_sum: %f\n", check(C, C_plain));

    free(A);
    free(B);
    free(C);
    free(C_plain);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
    return 0;
}