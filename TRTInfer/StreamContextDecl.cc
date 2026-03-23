#include "StreamContextDecl.h"
#include "utility.h"
#include <iostream>
/**
 * @brief 构造函数
 */
StreamPool::StreamPool()
    : engine_(nullptr), shutting_down_(false), initialized_(false), num_streams_(4)
{
}

/**
 * @brief 获取单例实例
 */
StreamPool::StreamPool(nvinfer1::ICudaEngine *engine, int num_streams)
{
    init(engine, num_streams);
}

/**
 * @brief 初始化 StreamPool
 */
void StreamPool::init(nvinfer1::ICudaEngine *engine, int num_streams)
{
    std::lock_guard<std::mutex> lock(const_cast<std::mutex &>(mutex_));

    if (initialized_)
    {
        std::cout << "[StreamPool] Already initialized, ignoring init call" << std::endl;
        return;
    }

    engine_ = engine;
    num_streams_ = num_streams;

    // 创建 Stream + Context 配对
    for (int i = 0; i < num_streams; i++)
    {
        StreamContextDecl Decl;

        // 创建 CUDA Stream
        cudaStreamCreate(&Decl.stream);

        // 创建 ExecutionContext（使用裸指针，StreamPool 管理生命周期）
        Decl.context = createContext(engine_);

        // 移动语义
        pool_.push(std::move(Decl));
    }

    initialized_ = true;
    std::cout << "[StreamPool] Initialized with " << num_streams << " Stream+Context pairs" << std::endl;
}

/**
 * @brief 创建 ExecutionContext
 */
nvinfer1::IExecutionContext *StreamPool::createContext(nvinfer1::ICudaEngine *engine)
{
    if (engine == nullptr)
    {
        return nullptr;
    }
    return engine->createExecutionContext();
}

/**
 * @brief 析构函数
 */
StreamPool::~StreamPool()
{
    std::cout << "[StreamPool] Shutdown" << std::endl;
    shutdown();

    // 销毁所有配对
    while (!pool_.empty())
    {
        auto &Decl = pool_.front();
        if (Decl.stream != nullptr)
        {
            cudaStreamDestroy(Decl.stream);
        }
        if (Decl.context != nullptr)
        {
            delete Decl.context;
        }

        for (auto &bind : Decl.outputBindings)
            utility::safeCudaFree(bind.second);

        for (auto &bind : Decl.inputBindings)
            utility::safeCudaFree(bind.second);

        for (auto &bind : Decl.outputBlobsPin)
            utility::safeCudaFreeHost(bind.second);
        pool_.pop(); // 弹出
    }

    std::cout << "[StreamPool] Destroyed" << std::endl;
}

/**
 * @brief 获取可用的配对（阻塞）
 */
StreamContextDecl StreamPool::acquire()
{
    std::unique_lock<std::mutex> lock(mutex_);

    // 等待直到有可用或关闭
    cond_.wait(lock, [this]()
               { return hasAvailable() || shutting_down_; });

    if (shutting_down_ || !initialized_)
    {
        return StreamContextDecl();
    }

    // 找到可用配对并标记为使用中
    for (size_t i = 0; i < pool_.size(); i++)
    {
        if (pool_.empty())
            break;
        StreamContextDecl result = std::move(pool_.front());
        pool_.pop();
        return result;
    }

    return StreamContextDecl();
}

/**
 * @brief 归还配对
 */
void StreamPool::release(StreamContextDecl &&Decl)
{
    std::lock_guard<std::mutex> lock(mutex_);
    pool_.push(std::move(Decl));
    cond_.notify_one();
}

/**
 * @brief 非阻塞尝试获取配对
 */
StreamContextDecl StreamPool::tryAcquire()
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (shutting_down_ || !initialized_ || !hasAvailable())
    {
        return StreamContextDecl();
    }

    // 找到可用配对并标记为使用中
    for (size_t i = 0; i < pool_.size(); i++)
    {
        if (pool_.empty())
            break;
        StreamContextDecl result = std::move(pool_.front());
        pool_.pop();
        return result;
    }

    return StreamContextDecl();
}

/**
 * @brief 等待所有配对可用
 */
void StreamPool::syncAll()
{
    std::unique_lock<std::mutex> lock(mutex_);

    cond_.wait(lock, [this]()
               { return !hasAvailableInUse() || shutting_down_; });
}

/**
 * @brief 关闭池
 */
void StreamPool::shutdown()
{
    std::lock_guard<std::mutex> lock(mutex_);
    shutting_down_ = true;
    cond_.notify_all();
}

/**
 * @brief 获取可用数量
 */
int StreamPool::availableCount() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return pool_.size();
}

/**
 * @brief 获取总数量
 */
int StreamPool::size() const
{
    return pool_.size();
}

/**
 * @brief 检查是否关闭
 */
bool StreamPool::isShuttingDown() const
{
    return shutting_down_;
}

/**
 * @brief 检查是否初始化
 */
bool StreamPool::isInitialized() const
{
    return initialized_;
}

/**
 * @brief 检查是否有可用配对
 */
bool StreamPool::hasAvailable() const
{
    return !pool_.empty();
}

/**
 * @brief 检查是否有使用中的配对
 */
bool StreamPool::hasAvailableInUse() const
{
    return pool_.size() < num_streams_ ? true : false;
}
