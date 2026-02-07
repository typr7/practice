#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>


static constexpr int BLUR_SIZE = 7;

__device__ float
clamp(const float val, const float lo, const float hi)
{
    return fmaxf(fminf(val, hi), lo);
}

__global__ void
imageBlur(const uint8_t*    input,
          const int         width,
          const int         height,
          uint8_t*          output)
{
    const int px = blockDim.x * blockIdx.x + threadIdx.x;
    const int py = blockDim.y * blockIdx.y + threadIdx.y;
    if (px < width && py < height) {
        int pixel_count = 0;
        float sum_r = 0;
        float sum_g = 0;
        float sum_b = 0;

        for (int dy = -BLUR_SIZE; dy < BLUR_SIZE + 1; dy++) {
            const int cy = py + dy;
            if (cy >= 0 && cy < height) {
                for (int dx = -BLUR_SIZE; dx < BLUR_SIZE + 1; dx++) {
                    const int cx = px + dx;
                    if (cx >= 0 && cx < width) {
                        const int i = (cy * width + cx) * 3;

                        pixel_count++;
                        sum_r += input[i];
                        sum_g += input[i + 1];
                        sum_b += input[i + 2];
                    }
                }
            }
        }

        const int i = (py * width + px) * 3;
        output[i] = static_cast<uint8_t>(sum_r / pixel_count);
        output[i + 1] = static_cast<uint8_t>(sum_g / pixel_count);
        output[i + 2] = static_cast<uint8_t>(sum_b / pixel_count);
    }
}

int main()
{
    int width, height, channels;
    const uint8_t* data = stbi_load(
        "image.jpg",
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
    cudaMalloc(&res_d, sizeof(uint8_t) * width * height * 3);

    cudaMemcpy(data_d, data, sizeof(uint8_t) * width * height * 3, cudaMemcpyHostToDevice);

    const dim3 grid_dim(std::ceil(width / 16.0), std::ceil(height / 16.0), 1);
    const dim3 block_dim(16, 16, 1);

    imageBlur<<<grid_dim, block_dim>>>(data_d, width, height, res_d);
    cudaDeviceSynchronize();

    uint8_t* res = static_cast<uint8_t*>(malloc(sizeof(uint8_t) * width * height * 3));
    cudaMemcpy(res, res_d, sizeof(uint8_t) * width * height * 3, cudaMemcpyDeviceToHost);

    const int ok = stbi_write_jpg("output.jpg", width, height, 3, res, width * 3);

    if (!ok) {
        printf("save processed image failed\n");
    }

    cudaFree(data_d);
    cudaFree(res_d);

    stbi_image_free(const_cast<uint8_t*>(data));
    free(res);

    return 0;
}