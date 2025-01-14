cmake_minimum_required(VERSION 3.0)
project(SegmentAnything-OnnxRunner)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    # Set the MSVC input code encoding format to utf-8
    set(CMAKE_C_FLAGS /source-charset:utf-8)
endif()

message(STATUS "Project: SegmentAnything-OnnxRunner")
message(STATUS "Root Path: ${CMAKE_SOURCE_DIR}")

include(cmake/platform.cmake)  # checking platform

# link third-party library onnxruntime-win-x64-1.14.1 and opencv4
set(ONNXRUNTIME_DIR  ${CMAKE_SOURCE_DIR}/third_party/onnxruntime114)
message(STATUS "ONNXRUNTIME_DIR Path: ${ONNXRUNTIME_DIR}")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories("${ONNXRUNTIME_DIR}/lib")
set(ONNXRUNTIME_LIB onnxruntime)

# OpenCV configuration
set(OPENCV_DIR /opt/homebrew/opt/opencv)
message(STATUS "OPENCV_DIR Path: ${OPENCV_DIR}")
include_directories(${OPENCV_DIR}/include/opencv4)
link_directories(${OPENCV_DIR}/lib)

# 添加所需的 OpenCV 库
set(OpenCV_LIBS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    opencv_videoio
)

# compile own file
file(GLOB SRC_LIST 
    ${CMAKE_SOURCE_DIR}/src/*.cpp
)
add_executable(main ${SRC_LIST})

# 链接 ONNX Runtime 和 OpenCV 库
target_link_libraries(main 
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
)