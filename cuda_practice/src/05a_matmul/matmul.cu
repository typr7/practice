namespace
{

constexpr int WARP_SIZE = 32;

// (a + b - 1) // b
__device__ __host__
constexpr
int cdiv(int a, int b)
{
    return (a + b - 1) / b;
}

__global__
void matmul_v1_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row > M || col > N) {
        return;
    }

    float acc = 0.f;
    for (int i = 0; i < K; i++) {
        acc += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = acc;
}

template <int BLOCK_SIZE>
__global__
void matmul_v2_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;

    const int offset_n = blockIdx.x * BLOCK_SIZE;
    const int offset_m = blockIdx.y * BLOCK_SIZE;

    __shared__ float A_smem[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_smem[BLOCK_SIZE][BLOCK_SIZE];

    A += offset_m * K;
    B += offset_n;
    C += offset_m * N + offset_n;

    float acc = 0.f;
    for (int offset_k = 0; offset_k < K; offset_k += BLOCK_SIZE) {
        A_smem[tidy][tidx] = (offset_k + tidx < K) && (offset_m + tidy < M) ? A[tidy * K + tidx]: 0.f;
        B_smem[tidy][tidx] = (offset_n + tidx < N) && (offset_k + tidy < K) ? B[tidy * N + tidx]: 0.f;
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            acc += A_smem[tidy][k] * B_smem[k][tidx];
        }
        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;
    }

    if ((offset_n + tidx < N) && (offset_m + tidy < M)) {
        C[tidy * N + tidx] = acc;
    }
}

template <int TB_SIZE, int BLOCK_M, int BLOCK_N>
__device__ __forceinline__
void load_block_to_smem(
    const float* src,
    float smem[BLOCK_M][BLOCK_N],
    int src_stride_n,
    int src_max_m, // offset_m + y < M
    int src_max_n
) {
    const int tid = threadIdx.x;
    for (int i = tid; i < BLOCK_M * BLOCK_N; i += TB_SIZE) {
        const int y = i / BLOCK_N;
        const int x = i % BLOCK_N;
        smem[y][x] = (y < src_max_m) && (x < src_max_n) ? src[y * src_stride_n + x]: 0.f;
    }
}

template <int TB_SIZE, int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__
void matmul_v3_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int tid = threadIdx.x;
    const int offset_m = blockIdx.y * BLOCK_M;
    const int offset_n = blockIdx.x * BLOCK_N;
    constexpr int block_size = BLOCK_M * BLOCK_N;

    A += offset_m * K;
    B += offset_n;
    C += offset_m * N + offset_n;

    __shared__ float A_smem[BLOCK_M][BLOCK_K];
    __shared__ float B_smem[BLOCK_K][BLOCK_N];

    static_assert(block_size % TB_SIZE == 0);
    float acc_smem[block_size / TB_SIZE] = {0.f};
    for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
        load_block_to_smem<TB_SIZE, BLOCK_M, BLOCK_K>(A, A_smem, K, M - offset_m, K - offset_k);
        load_block_to_smem<TB_SIZE, BLOCK_K, BLOCK_N>(B, B_smem, N, K - offset_k, N - offset_n);
        __syncthreads();

        for (int i = tid, acc_idx = 0; i < block_size; i += TB_SIZE, acc_idx++) {
            const int y = i / BLOCK_N;
            const int x = i % BLOCK_N;
            for (int k = 0; k < BLOCK_K; k++) {
                acc_smem[acc_idx] += A_smem[y][k] * B_smem[k][x];
            }
        }
        __syncthreads();

        A += BLOCK_K;
        B += BLOCK_K * N;
    }

    for (int i = tid, acc_idx = 0; i < block_size; i += TB_SIZE, acc_idx++) {
        const int y = i / BLOCK_N;
        const int x = i % BLOCK_N;
        if ((y + offset_m < M) && (x + offset_n < N)) {
            C[y * N + x] = acc_smem[acc_idx];
        }
    }
}

template <int TB_SIZE, int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_M, int THREAD_N>
__global__
void matmul_v4_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    const int tid = threadIdx.x;

    const int offset_m = blockIdx.y * BLOCK_M;
    const int offset_n = blockIdx.x * BLOCK_N;

    const int tile_offset_m = (tid / (BLOCK_N / THREAD_N)) * THREAD_M;
    const int tile_offset_n = (tid % (BLOCK_N / THREAD_N)) * THREAD_N;

    A += offset_m * K;
    B += offset_n;
    C += (offset_m + tile_offset_m) * N + offset_n + tile_offset_n;

    __shared__ float A_smem[BLOCK_M][BLOCK_K];
    __shared__ float B_smem[BLOCK_K][BLOCK_N];

    float acc[THREAD_M][THREAD_N] = {0.f};
    for (int offset_k = 0; offset_k < K; offset_k += BLOCK_K) {
        load_block_to_smem<TB_SIZE, BLOCK_M, BLOCK_K>(A, A_smem, K, M - offset_m, K - offset_k);
        load_block_to_smem<TB_SIZE, BLOCK_K, BLOCK_N>(B, B_smem, N, K - offset_k, N - offset_n);
        __syncthreads();

        for (int k = 0; k < BLOCK_K; k++) {
            float A_reg[THREAD_M];
            float B_reg[THREAD_N];

            for (int i = 0; i < THREAD_M; i++) {
                A_reg[i] = A_smem[tile_offset_m + i][k];
            }

            for (int i = 0; i < THREAD_N; i++) {
                B_reg[i] = B_smem[k][tile_offset_n + i];
            }

            for (int y = 0; y < THREAD_M; y++) {
                for (int x = 0; x < THREAD_N; x++) {
                    acc[y][x] += A_reg[y] * B_reg[x];
                }
            }
        } 
        __syncthreads();

        A += BLOCK_K;
        B += BLOCK_K * N;
    }

    for (int y = 0; y < THREAD_M; y++) {
        for (int x = 0; x < THREAD_N; x++) {
            if ((offset_m + tile_offset_m + y < M) && (offset_n + tile_offset_n + x < N)) {
                C[y * N + x] = acc[y][x];
            }
        }
    }
}

}

void matmul_v1(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    int block_size_total;
    int min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size_total, matmul_v1_kernel, 0, 0);

    dim3 block_size(WARP_SIZE, block_size_total / WARP_SIZE);
    dim3 grid_size(cdiv(N, WARP_SIZE), cdiv(M, block_size.y));
    matmul_v1_kernel<<<grid_size, block_size>>>(A, B, C, M, N, K);
}

void matmul_v2(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    constexpr int BLOCK_SIZE = 16;
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(cdiv(N, BLOCK_SIZE), cdiv(M, BLOCK_SIZE));
    matmul_v2_kernel<BLOCK_SIZE><<<grid_size, block_size>>>(A, B, C, M, N, K);
}

void matmul_v3(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 32;
    constexpr int TB_SIZE = 256;
    dim3 grid_size(cdiv(N, BLOCK_N), cdiv(M, BLOCK_M));
    matmul_v3_kernel<TB_SIZE, BLOCK_M, BLOCK_N, BLOCK_K>
        <<<grid_size, TB_SIZE>>>(A, B, C, M, N, K);
}

void matmul_v4(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    constexpr int BLOCK_M  = 128;
    constexpr int BLOCK_N  = 128;
    constexpr int BLOCK_K  = 32;
    constexpr int THREAD_M = 8;
    constexpr int THREAD_N = 8;
    constexpr int TB_SIZE  = (BLOCK_M * BLOCK_N) / (THREAD_M * THREAD_N);

    static_assert((BLOCK_M * BLOCK_N) % (THREAD_M * THREAD_N) == 0);
    static_assert((BLOCK_M % THREAD_M == 0) && (BLOCK_N % THREAD_N == 0));

    dim3 grid_size(cdiv(N, BLOCK_N), cdiv(M, BLOCK_M));
    matmul_v4_kernel<TB_SIZE, BLOCK_M, BLOCK_N, BLOCK_K, THREAD_M, THREAD_N>
        <<<grid_size, TB_SIZE>>>(A, B, C, M, N, K);
}