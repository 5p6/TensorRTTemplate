#ifndef STREAMCONTEXTDECL_H
#define STREAMCONTEXTDECL_H

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <NvInfer.h>

/**
 * @brief Stream + Context + 显存/锁页缓冲 的资源捆绑 (RAII)
 *
 * 每个工作线程独占一个实例，整个生命周期内复用同一条 CUDA 流、执行上下文、
 * I/O 显存以及锁页暂存缓冲。析构时自动释放全部资源。
 *
 * 仅可移动、禁止拷贝，避免重复释放。资源由持有方 (TRTInfer::Impl) 分配，
 * 但销毁统一在本结构析构中完成。
 */
struct StreamContextDecl
{
    cudaStream_t stream = nullptr;                                          /**< @brief CUDA 流 */
    nvinfer1::IExecutionContext *context = nullptr;                         /**< @brief TensorRT 执行上下文 */
    std::unordered_map<std::string, void *> inputBindings, outputBindings; /**< @brief 输入/输出显存 (device) */
    std::unordered_map<std::string, void *> inputBlobsPin;                  /**< @brief 输入锁页暂存 (pinned host) */
    std::unordered_map<std::string, void *> outputBlobsPin;                 /**< @brief 输出锁页暂存 (pinned host) */

    StreamContextDecl() = default;

    StreamContextDecl(const StreamContextDecl &) = delete;
    StreamContextDecl &operator=(const StreamContextDecl &) = delete;

    StreamContextDecl(StreamContextDecl &&other) noexcept { moveFrom(other); }
    StreamContextDecl &operator=(StreamContextDecl &&other) noexcept
    {
        if (this != &other)
        {
            destroy();
            moveFrom(other);
        }
        return *this;
    }

    ~StreamContextDecl() { destroy(); }

private:
    /// 释放全部 CUDA 资源 (定义在 .cc，依赖 utility/cuda)
    void destroy() noexcept;

    void moveFrom(StreamContextDecl &other) noexcept
    {
        stream = other.stream;
        context = other.context;
        inputBindings = std::move(other.inputBindings);
        outputBindings = std::move(other.outputBindings);
        inputBlobsPin = std::move(other.inputBlobsPin);
        outputBlobsPin = std::move(other.outputBlobsPin);
        other.stream = nullptr;
        other.context = nullptr;
    }
};

#endif // STREAMCONTEXTDECL_H
