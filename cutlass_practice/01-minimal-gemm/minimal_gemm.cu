#include <optional>

#include <torch/extension.h>
#include <cute/tensor.hpp>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAGuard.h>


namespace {

// return a // b
constexpr int cdiv(int a, int b) {
  return (a + b - 1) / b;
};

namespace spec {

template <typename T_, int kTileM_ = 16, int kTileN_ = 8, int kTileK_ = 8>
struct KernelSpec {
  using T = T_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;

  using MmaOp = cute::SM80_16x8x8_F16F16F16F16_TN;
  using TiledMMA = decltype(cute::make_tiled_mma(MmaOp{}));

  static constexpr int kNumThreads = cute::size(TiledMMA{});
  static constexpr int kSmemSize = 0;
};

}

template <typename Spec, bool kIsGemm>
__global__
void minimal_gemm(void* C, const void* A, const void* B, int m, int n, int k)
{
  using namespace cute;

  using T = typename Spec::T;
  using TiledMMA = typename Spec::TiledMMA;
  
  constexpr int kTileM = Spec::kTileM;
  constexpr int kTileN = Spec::kTileN;
  constexpr int kTileK = Spec::kTileK;

  Tensor mA = make_tensor(make_gmem_ptr(static_cast<const T*>(A)),
                          make_shape(m, k),
                          make_stride(k, Int<1>{}));
  Tensor mB = make_tensor(make_gmem_ptr(static_cast<const T*>(B)),
                          make_shape(n, k),
                          make_stride(k, Int<1>{}));
  Tensor mC = make_tensor(make_gmem_ptr(static_cast<T*>(C)),
                          make_shape(m, n),
                          make_stride(n, Int<1>{}));
  
  auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
  auto coord = make_coord(0, 0, 0);

  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{});

  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);

  Tensor tCgA = thr_mma.partition_A(gA);
  Tensor tCgB = thr_mma.partition_B(gB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrA = thr_mma.partition_fragment_A(gA);
  Tensor tCrB = thr_mma.partition_fragment_B(gB);
  Tensor tCrC = thr_mma.partition_fragment_C(gC);

  auto copy_atom = AutoVectorizingCopy{};
  copy(copy_atom, tCgA, tCrA);
  copy(copy_atom, tCgB, tCrB);

  if constexpr (kIsGemm) {
    clear(tCrC);
  } else {
    copy(copy_atom, tCgC, tCrC);
  }

  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

  copy(copy_atom, tCrC, tCgC);
}

}

torch::Tensor minimal_gemm(const torch::Tensor& A, // [M, K]
                           const torch::Tensor& B, // [N, K]
                           std::optional<torch::Tensor>& C /* [M, N] */) {
  at::cuda::CUDAGuard device_guard{A.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  constexpr int M = 16;
  constexpr int N = 8;
  constexpr int K = 8;

  torch::Tensor c;
  bool is_gemm;

  if (C.has_value()) {
    c = C.value();
    is_gemm = false;
  } else {
    c = torch::empty({M, N}, A.options());
    is_gemm = true;
  }

  using Spec = spec::KernelSpec<cute::half_t, M, N, K>;
  dim3 block = Spec::kNumThreads;
  dim3 grid(cdiv(N, Spec::kTileN), cdiv(M, Spec::kTileM));
  int smem_size = Spec::kSmemSize;

  if (is_gemm) {
    minimal_gemm<Spec, true><<<grid, block, smem_size, stream>>>(c.data_ptr(), A.data_ptr(), B.data_ptr(), M, N, K);
  } else {
    minimal_gemm<Spec, false><<<grid, block, smem_size, stream>>>(c.data_ptr(), A.data_ptr(), B.data_ptr(), M, N, K);
  }

  return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("minimal_gemm", &::minimal_gemm, "m16n8k8 gemm");
}