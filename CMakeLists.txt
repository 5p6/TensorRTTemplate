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
# ---- 可通过 -D 覆盖的路径与选项 ----
# 例:
#   cmake -S . -B build \
#     -DTensorRT_ROOT_DIR=/opt/TensorRT-10.7.0.23 \
#     -DCUDA_ROOT_DIR=/usr/local/cuda-12.4 \
#     -DLIB_TYPE=STATIC
# 也支持别名 -DTensorRT_ROOT / -DCUDA_ROOT。
set(TensorRT_ROOT_DIR "/home/ros/app/TensorRT-10.7.0.23" CACHE PATH "TensorRT 10.x 安装根目录")
set(CUDA_ROOT_DIR     "/usr"                  CACHE PATH "CUDA Toolkit 安装根目录")
set(LIB_TYPE          "SHARED"                           CACHE STRING "库类型: SHARED 或 STATIC")

# 兼容常见别名 -DTensorRT_ROOT / -DCUDA_ROOT
if(DEFINED TensorRT_ROOT)
    set(TensorRT_ROOT_DIR "${TensorRT_ROOT}" CACHE PATH "TensorRT 10.x 安装根目录" FORCE)
endif()
if(DEFINED CUDA_ROOT)
    set(CUDA_ROOT_DIR "${CUDA_ROOT}" CACHE PATH "CUDA Toolkit 安装根目录" FORCE)
endif()

message(STATUS "TensorRT_ROOT_DIR = ${TensorRT_ROOT_DIR}")
message(STATUS "CUDA_ROOT_DIR     = ${CUDA_ROOT_DIR}")
message(STATUS "LIB_TYPE          = ${LIB_TYPE}")

if(NOT EXISTS "${TensorRT_ROOT_DIR}/include/NvInfer.h")
    message(FATAL_ERROR "在 ${TensorRT_ROOT_DIR} 找不到 TensorRT (缺少 include/NvInfer.h)。\n"
                        "请用 -DTensorRT_ROOT_DIR=<路径> 指定 TensorRT 10.x 根目录。")
endif()

# set lib type
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message("-- Window operating system --")

    # for opencv cmake config
    set(OpenCV_ROOT_DIR "E:/lib/opencv/opencv-4.8.0/build/x64/vc16/lib")
    set(CMAKE_PREFIX_PATH "${OpenCV_ROOT_DIR};${CMAKE_PREFIX_PATH}")

    # set TensorRT and cuda library and include directory for windows system
    set(TensorRT_INCLUDE_DIR "${TensorRT_ROOT_DIR}/include")
    set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")
    file(GLOB CUDA_LIBS "${CUDA_ROOT_DIR}/lib/x64/*.lib")
    file(GLOB TensorRT_LIBS "${TensorRT_ROOT_DIR}/lib/*.lib")
    message("-- TensorRT libs : ${TensorRT_LIBS}")
    message("-- Cuda libs : ${CUDA_LIBS}")
else()
    # set TensorRT and cuda library and include directory for linux system
    set(TensorRT_INCLUDE_DIR "${TensorRT_ROOT_DIR}/include")
    set(TensorRT_LIBS
        "${TensorRT_ROOT_DIR}/lib/libnvinfer.so"
        "${TensorRT_ROOT_DIR}/lib/libnvinfer_plugin.so")

    # CUDA: 优先用 CUDA_ROOT_DIR (标准 toolkit 布局)，找不到则回退到发行版(apt)布局
    if(EXISTS "${CUDA_ROOT_DIR}/include/cuda_runtime.h")
        set(CUDA_INCLUDE_DIR "${CUDA_ROOT_DIR}/include")
        find_library(CUDART_LIB cudart
            HINTS "${CUDA_ROOT_DIR}/lib64" "${CUDA_ROOT_DIR}/lib"
                  "${CUDA_ROOT_DIR}/targets/x86_64-linux/lib"
            NO_DEFAULT_PATH)
    else()
        message(STATUS "在 ${CUDA_ROOT_DIR} 未找到 CUDA toolkit，回退到系统(apt)路径")
        set(CUDA_INCLUDE_DIR "/usr/include")
        find_library(CUDART_LIB cudart HINTS /usr/lib/x86_64-linux-gnu)
    endif()
    if(NOT CUDART_LIB)
        message(FATAL_ERROR "找不到 libcudart。请用 -DCUDA_ROOT_DIR=<路径> 指定 CUDA Toolkit 根目录。")
    endif()
    set(CUDA_LIBS "${CUDART_LIB}")
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
<<<<<<< HEAD
=======
    ${PROJECT_SOURCE_DIR}/TRTInfer/StreamContextDecl.h
    ${PROJECT_SOURCE_DIR}/TRTInfer/StreamContextDecl.cc
>>>>>>> main_multistream
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

add_executable(yolo_batch example/YOLOBatch.cc)
target_link_libraries(yolo_batch
    trtemplate
)

add_executable(liteanystereo_video example/LiteAnyStereoVideo.cc)
target_link_libraries(liteanystereo_video
    trtemplate
)
