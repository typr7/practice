add_executable(rgb_to_gray rgb_to_gray.cu)
target_link_libraries(rgb_to_gray PRIVATE
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
)
set_target_properties(rgb_to_gray PROPERTIES CUDA_ARCHITECTURES 86)