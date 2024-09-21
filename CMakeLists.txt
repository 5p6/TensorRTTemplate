cmake_minimum_required(VERSION 3.20)

project(trttemplate)

set(TensorRT_INCLUDE_DIR "E:/lib/Tensorrt/TensorRT-10.0.1.6/include")
set(CUDA_INCLUDE_DIR "E:/lib/cudalib-11.8/development/include")
file(GLOB CUDA_LIBS "E:/lib/cudalib-11.8/development/lib/x64/*.lib")
file(GLOB TensorRT_LIBS "E:/lib/Tensorrt/TensorRT-10.0.1.6/lib/*.lib")

find_package(OpenCV REQUIRED)


include_directories(
    ${PROJECT_SOURCE_DIR}/TRTInfer
    ${TensorRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
add_library(trtemplate STATIC
    TRTInfer/TRTinfer.cc
    TRTInfer/utility.cc
)
target_link_libraries(trtemplate PRIVATE
    ${CUDA_LIBS}
    ${TensorRT_LIBS}
    ${OpenCV_LIBS}
)


add_executable(main main.cc)

target_link_libraries(main
    trtemplate
)


add_executable(example example.cc)

target_link_libraries(example
    trtemplate
)