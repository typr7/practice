import torch
import torch.utils.cpp_extension
from triton.testing import do_bench

def benchmark(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median")

module = torch.utils.cpp_extension.load(
    'module',
    sources=['matmul_v1.cpp', 'matmul_v1.cu'],
    extra_cuda_cflags=[
        '-O3',
        '-lineinfo',
        '-Xptxas=-v'
    ],
    verbose=True
)

input1 = torch.randn((4096, 4096), dtype=torch.bfloat16).cuda()
input2 = torch.randn((4096, 4096), dtype=torch.bfloat16).cuda()

output_ref = module.matmul_cublas(input1, input2)
output_v1  = module.matmul_v1(input1, input2)

torch.testing.assert_close(output_v1, output_ref)

print(f'cublas: {benchmark(module.matmul_cublas, input1, input2)}')
print(f'v1: {benchmark(module.matmul_v1, input1, input2)}')