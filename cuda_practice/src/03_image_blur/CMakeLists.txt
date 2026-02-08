add_executable(image_blur image_blur.cu)
target_link_libraries(image_blur PRIVATE CUDA::cudart stb_image)