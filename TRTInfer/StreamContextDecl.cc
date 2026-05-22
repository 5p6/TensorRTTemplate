#include "StreamContextDecl.h"
#include "utility.h"

/**
 * @brief 释放本捆绑持有的全部 CUDA 资源
 *
 * 顺序：销毁流 -> 删除执行上下文 -> 释放 I/O 显存 -> 释放锁页缓冲。
 * 调用 engine 仍存活时进行（持有方需保证 engine 在所有 ctx 之后析构）。
 */
void StreamContextDecl::destroy() noexcept
{
    if (stream != nullptr)
    {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
    if (context != nullptr)
    {
        delete context;
        context = nullptr;
    }

    for (auto &bind : inputBindings)
        utility::safeCudaFree(bind.second);
    for (auto &bind : outputBindings)
        utility::safeCudaFree(bind.second);
    for (auto &bind : inputBlobsPin)
        utility::safeCudaFreeHost(bind.second);
    for (auto &bind : outputBlobsPin)
        utility::safeCudaFreeHost(bind.second);

    inputBindings.clear();
    outputBindings.clear();
    inputBlobsPin.clear();
    outputBlobsPin.clear();
}
