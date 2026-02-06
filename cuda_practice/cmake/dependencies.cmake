include(FetchContent)

FetchContent_Declare(
    opencv
    GIT_REPOSITORY  https://github.com/opencv/opencv.git
    GIT_TAG         4.13.0
    GIT_SHALLOW     TRUE
)

set(BUILD_LIST "core,imgcodecs,imgproc" CACHE STRING "" FORCE)

set(BUILD_SHARED_LIBS        OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS              OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS         OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES           OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps        OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3     OFF CACHE BOOL "" FORCE)
set(BUILD_JAVA               OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_js          OFF CACHE BOOL "" FORCE)

set(WITH_GTK                 OFF CACHE BOOL "" FORCE)
set(WITH_QT                  OFF CACHE BOOL "" FORCE)
set(WITH_FFMPEG              OFF CACHE BOOL "" FORCE)
set(WITH_GSTREAMER           OFF CACHE BOOL "" FORCE)
set(WITH_V4L                 OFF CACHE BOOL "" FORCE)
set(WITH_1394                OFF CACHE BOOL "" FORCE)
set(WITH_CUDA                OFF CACHE BOOL "" FORCE)
set(WITH_OPENCL              OFF CACHE BOOL "" FORCE)
set(WITH_IPP                 OFF CACHE BOOL "" FORCE)
set(WITH_PROTOBUF            OFF CACHE BOOL "" FORCE)

set(WITH_PNG                 ON  CACHE BOOL "" FORCE)
set(WITH_JPEG                ON  CACHE BOOL "" FORCE)
set(WITH_TIFF                OFF CACHE BOOL "" FORCE)
set(WITH_WEBP                OFF CACHE BOOL "" FORCE)
set(WITH_OPENJPEG            OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(opencv)