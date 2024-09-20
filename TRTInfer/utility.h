#ifndef UTILITY_H
#define UTILITY_H
#include <cuda_runtime_api.h>
#include <iostream>
#include <NvInfer.h>
#include <opencv2/core.hpp>

namespace utilty
{
    /**
     * @brief 安全分配gpu内存
     * @param memSize 需要的gpu内存字节数
     * @return 分配的GPU内存地址块的地址头
     */
    void *safeCudaMalloc(size_t memSize);

    /**
     * @brief 安全释放gpu内存
     * @param ptr gpu地址块的地址头
     * @return 是否释放成功的标志
     */
    bool safeCudaFree(void *&ptr);

    /**
     * @brief 获取TensorRT基本类型的字节数
     * @brief type tensorrt的基本类型
     * @return 字节数
     */
    int getTypebytes(const nvinfer1::DataType &type);

    /**
     * @brief 输入一个矩阵的维度和基本类型，输出这个矩阵的全部字节数
     * @param dim 矩阵的维度
     * @param type TensorRT的基本数据类型
     */
    int getTensorbytes(const nvinfer1::Dims &dim, const nvinfer1::DataType &type);

    /**
     * @brief convert the cv type to nvidia tensorrt type
     * @param cv_type the type of opencv
     * @return nvidia type ,default float32
     */
    nvinfer1::DataType typeCv2Rt(const int &cv_type);

    /**
     * @brief convert the nvidia tensorrt type to cv type
     * @param rt_type the type of nvidia tensorrt
     * @return cv type ,default CV_32F
     */
    int typeRt2Cv(const nvinfer1::DataType &rt_type);
}

#endif