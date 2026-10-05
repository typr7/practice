import os
import torch
import torch.utils.cpp_extension
import triton.testing


current_dir = os.path.dirname(os.path.abspath(__file__))
cutlass_include_path = os.path.join(current_dir, '../third_party/cutlass/include')

module = torch.utils.cpp_extension.load(
  'module',
  sources='01-minimal-gemm/minimal_gemm.cu',
  extra_cuda_cflags=[
    '-O3',
    '-lineinfo',
    '-Xptxas=-v',
    '-std=c++20',
    f'-I{cutlass_include_path}'
  ],
  verbose=True
)

M = 16
N = 8
K = 8

A = torch.randn(M, K, device='cuda', dtype=torch.half)
B = torch.randn(N, K, device='cuda', dtype=torch.half)
C = torch.randn(M, N, device='cuda', dtype=torch.half)

mm_output_ref = torch.matmul(A, B.T)
mm_output = module.minimal_gemm(A, B, None)
mma_output_ref = torch.addmm(C, A, B.T)
mma_output = module.minimal_gemm(A, B, C)

triton.testing.assert_close(mm_output_ref, mm_output)
triton.testing.assert_close(mma_output_ref, mma_output)

print('ok')