#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>


__device__ float
clamp(const float val, const float lo, const float hi)
{
    return fmaxf(fminf(val, hi), lo);
}

__global__ void
imageRGBToGray(const uint8_t*   input,
               const int        width,
               const int        height,
               uint8_t*         output)
{
    const int px = blockDim.x * blockIdx.x + threadIdx.x;
    const int py = blockDim.y * blockIdx.y + threadIdx.y;
    if (px < width && py < height) {
        const int pi = (width * py + px) * 3;
        const uint8_t gray = static_cast<uint8_t>(
            clamp(
                0.21f * input[pi] + 0.72f * input[pi + 1] + 0.07f * input[pi + 2],
                0.f,
                255.f
            )
        );
        output[width * py + px] = gray;
    }
}

int main()
{
    int width, height, channels;
    const uint8_t* data = stbi_load(
        "images.jpg",
        &width,
        &height,
        &channels,
        0
    );

    if (data) {
        printf(
            "width: %d, height: %d, channels: %d\n",
            width,
            height,
            channels
        );
    } else {
        printf("read image failed\n");
    }

    uint8_t* data_d = nullptr;
    cudaMalloc(&data_d, sizeof(uint8_t) * width * height * 3);
    uint8_t* res_d = nullptr;
    cudaMalloc(&res_d, sizeof(uint8_t) * width * height);

    cudaMemcpy(data_d, data, sizeof(uint8_t) * width * height * 3, cudaMemcpyHostToDevice);

    const dim3 grid_dim(std::ceil(width / 16.0), std::ceil(height / 16.0), 1);
    const dim3 block_dim(16, 16, 1);

    imageRGBToGray<<<grid_dim, block_dim>>>(data_d, width, height, res_d);
    cudaDeviceSynchronize();

    uint8_t* res = static_cast<uint8_t*>(malloc(sizeof(uint8_t) * width * height));
    cudaMemcpy(res, res_d, sizeof(uint8_t) * width * height, cudaMemcpyDeviceToHost);

    const int ok = stbi_write_jpg("output.jpg", width, height, 1, res, width);

    if (!ok) {
        printf("save processed image failed\n");
    }

    cudaFree(data_d);
    cudaFree(res_d);

    stbi_image_free(const_cast<uint8_t*>(data));
    free(res);

    return 0;
}