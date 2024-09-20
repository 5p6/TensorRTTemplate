#ifndef TRTINFER_H
#define TRTINFER_H
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
// #include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <opencv2/opencv.hpp>
#include "utility.h"

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override;
};

class TRTInfer
{
    // init -> load engine -> allocate cuda memory  -> inference
public:
    TRTInfer() = delete;
    /**
     * @param engine_path 引擎权重路径
     */
    TRTInfer(const std::string &engine_path);
    /**
     * @brief 模型推理，内部调用infer函数
     * @param input_blob 输出的blob数据，第一个数据是输入张量的名称，第二个就是张量数据的地址头、
     * @return 输出张量
     */
    std::unordered_map<std::string, void *> operator()(const std::unordered_map<std::string, void *> &input_blob);
        /**
     * @brief 模型推理，基于cv::Mat的数据类型，内部调用 infer
     * @param input_blob 输出的blob数据，第一个数据是输入张量的名称，第二个就是张量数据的地址头、
     * @return 输出张量
     */
    std::unordered_map<std::string, cv::Mat> operator()(const std::unordered_map<std::string, cv::Mat> &input_blob);

    ~TRTInfer();

private:
    void load_engine(const std::string &engine_path);

    void get_InputNames();

    void get_OutputNames();

    void get_bindings();

    std::unordered_map<std::string, void *> infer(const std::unordered_map<std::string, void *> &input_blob);

    // for opencv Mat data
    std::unordered_map<std::string, cv::Mat> infer(const std::unordered_map<std::string, cv::Mat> &input_blob);

private:
    // plugin
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    Logger logger;

    // data
    std::vector<std::string> input_names, output_names;
    std::unordered_map<std::string, size_t> input_size, output_size;
    std::unordered_map<std::string,std::vector<int>> output_shape;
    cv::Size size;

    // bindings
    std::unordered_map<std::string, cv::Mat> input_Bindings, output_Bindings;
    std::unordered_map<std::string, void *> inputBindings, outputBindings;
};

#endif // !RTDETRinferENCE_HPP
