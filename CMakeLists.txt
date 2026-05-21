cmake_minimum_required(VERSION 3.15)

project(trttemplate)

set(CMAKE_CXX_STANDARD 17)

# GNU 最高优化级别
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native -mtune=native -ffast-math -funroll-loops -flto")

# 设置 Release 模式为默认
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(LIB_TYPE SHARED)
set(CUDA_ROOT_DIR "/usr/local/cuda")
set(TensorRT_ROOT_DIR "/home/ros/app/TensorRT-10.7.0.23")



# set lib type
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message("-- Window operating system")

    # for opencv cmake config
    set(OpenCV_ROOT_DIR "E:/lib/opencv/opencv-4.8.0/build/x64/vc16/lib")
    set(CMAKE_PREFIX_PATH "${OpenCV_ROOT_DIR};${CMAKE_PREFIX_PATH}")
    
    # set TensorRT and cuda library and include directory for windows system
    set(TensorRT_INCLUDE_DIR "${TensorRT_ROOT_DIR}/include")
    set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")
    file(GLOB CUDA_LIBS "${CUDA_ROOT_DIR}/lib/x64/*.lib")
    file(GLOB TensorRT_LIBS "${TensorRT_ROOT_DIR}/lib/*.lib")
    message("-- TensorRT libs : ${CUDA_LIBS}")
    message("-- Cuda libs : ${TensorRT_LIBS}")
else()
# set TensorRT and cuda library and include directory for linux system
    set(TensorRT_INCLUDE_DIR "${TensorRT_ROOT_DIR}/include")
    # apt-installed CUDA: headers in /usr/include, runtime lib in /usr/lib/x86_64-linux-gnu
    set(CUDA_INCLUDE_DIR "/usr/include")
    file(GLOB CUDA_LIBS "/usr/lib/x86_64-linux-gnu/libcudart.so")
    set(TensorRT_LIBS
        "${TensorRT_ROOT_DIR}/lib/libnvinfer.so"
        "${TensorRT_ROOT_DIR}/lib/libnvinfer_plugin.so")
    message("-- Cuda libs : ${CUDA_LIBS}")
    message("-- TensorRT libs : ${TensorRT_LIBS}")
endif()

# find dependencies library
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)


# include
include_directories(
    ${PROJECT_SOURCE_DIR}/TRTInfer
    ${TensorRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
# generate library
add_library(trtemplate ${LIB_TYPE}
    ${PROJECT_SOURCE_DIR}/TRTInfer/StreamContextDecl.h
    ${PROJECT_SOURCE_DIR}/TRTInfer/StreamContextDecl.cc
    ${PROJECT_SOURCE_DIR}/TRTInfer/config.h
    ${PROJECT_SOURCE_DIR}/TRTInfer/TRTinfer.h
    ${PROJECT_SOURCE_DIR}/TRTInfer/benchmark.h
    ${PROJECT_SOURCE_DIR}/TRTInfer/utility.h
    ${PROJECT_SOURCE_DIR}/TRTInfer/TRTinfer.cc
    ${PROJECT_SOURCE_DIR}/TRTInfer/utility.cc
)

# 为共享库添加BUILD_SHARED定义（用于Windows DLL导出）
if(${LIB_TYPE} STREQUAL "SHARED")
    message("-- BUILD SHARED LIBS")
    target_compile_definitions(trtemplate PRIVATE BUILD_SHARED)
endif()

target_link_libraries(trtemplate 
PRIVATE
    ${CUDA_LIBS}
    ${TensorRT_LIBS}
PUBLIC
    ${OpenCV_LIBS}
)


add_executable(igev example/IGEV.cc)
target_link_libraries(igev
    trtemplate
)

add_executable(yolo example/YOLO.cc)
target_link_libraries(yolo
    trtemplate
)


add_executable(fcn example/FCN.cc)
target_link_libraries(fcn
    trtemplate
)



add_executable(segformer example/Segformer.cc)
target_link_libraries(segformer
    trtemplate
)



add_executable(s2m2 example/s2m2.cc)
target_link_libraries(s2m2
    trtemplate
)

add_executable(liteanystereo example/LiteAnyStereo.cc)
target_link_libraries(liteanystereo
    trtemplate
)

add_executable(yolo_video example/YOLOVideo.cc)
target_link_libraries(yolo_video
    trtemplate
)

add_executable(liteanystereo_video example/LiteAnyStereoVideo.cc)
target_link_libraries(liteanystereo_video
    trtemplate
)
