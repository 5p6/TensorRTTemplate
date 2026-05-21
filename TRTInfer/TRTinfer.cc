#include "TRTinfer.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <cstring>
#include "utility.h"
#include "StreamContextDecl.h"

/**
 * @brief 输出 nvinfer1::Dims 维度的流操作符
 */
std::ostream &operator<<(std::ostream &cout, const nvinfer1::Dims &dim)
{
    for (int i = 0; i < dim.nbDims; i++)
    {
        if (i < dim.nbDims - 1)
        {
            cout << dim.d[i] << " X ";
        }
        else
            cout << dim.d[i];
    }
    return cout;
}

/**
 * @brief 输出 nvinfer1::DataType 数据类型的流操作符
 */
std::ostream &operator<<(std::ostream &cout, const nvinfer1::DataType &type)
{
    switch (type)
    {
    case nvinfer1::DataType::kBF16:
        cout << "kBF16";
        break;
    case nvinfer1::DataType::kBOOL:
        cout << "kBOOL";
        break;
    case nvinfer1::DataType::kFLOAT:
        cout << "kFLOAT";
        break;
    case nvinfer1::DataType::kFP8:
        cout << "kFP8";
        break;
    case nvinfer1::DataType::kHALF:
        cout << "kHALF";
        break;
    case nvinfer1::DataType::kINT32:
        cout << "kINT32";
        break;
    case nvinfer1::DataType::kINT4:
        cout << "kINT4";
        break;
    case nvinfer1::DataType::kINT64:
        cout << "kINT64";
        break;
    case nvinfer1::DataType::kINT8:
        cout << "kINT8";
        break;
    case nvinfer1::DataType::kUINT8:
        cout << "kUINT8";
        break;
    default:
        break;
    }
    return cout;
}

/**
 * @brief TensorRT 日志记录器
 *
 * 过滤 INFO 级别日志，只输出 WARNING 及以上级别
 */
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity != Severity::kINFO)
        {
            std::cout << msg << std::endl;
        }
    }
};

namespace TRT
{
    /**
     * @brief 推理任务封装类
     *
     * 只持有输入与 promise；具体推理由工作线程用自己绑定的 ctx 执行。
     */
    class InferTask
    {
    public:
        explicit InferTask(const BlobType &args) : args_(args) {}

        const BlobType &args() const { return args_; }
        std::promise<BlobType> &promise() { return promise_; }
        std::future<BlobType> get_future() { return promise_.get_future(); }

    private:
        const BlobType args_; // 按值持有：异步任务执行前调用方可能已销毁原始 blob
        std::promise<BlobType> promise_;
    };
    /**
     * @brief TRTInfer 实现类 (Pimpl 模式)
     */
    class TRTInfer::Impl
    {
    public:
        /** @brief 构造函数 */
        Impl(const std::string &engine_path, int num_thread, TRTInfer *parent);

        /** @brief 析构函数 */
        ~Impl();

        /** @brief 推送任务到队列 */
        std::future<BlobType> PostQueue(const BlobType &input_blob);

        /** @brief 同步推理 */
        BlobType infer(const BlobType &input_blob);

        /** @brief 实际推理任务执行 (使用调用线程绑定的 ctx) */
        BlobType infer_task(StreamContextDecl &ctx, const BlobType &input_blob);

        /** @brief 初始化引擎和资源 */
        void Initialized();

    public:
        /** @brief 获取输入张量名称 */
        std::vector<std::string> getInputNames() const { return input_names_; }

        /** @brief 获取输出张量名称 */
        std::vector<std::string> getOutputNames() const { return output_names_; }

        /** @brief 获取输入张量形状 */
        std::vector<int> getInputShapeVec(const std::string &name) const
        {
            auto it = current_input_shapes_.find(name);
            return (it != current_input_shapes_.end()) ? it->second : std::vector<int>();
        }

        /** @brief 获取输出张量形状 */
        std::vector<int> getOutputShapeVec(const std::string &name) const
        {
            auto it = output_shape_.find(name);
            return (it != output_shape_.end()) ? it->second : std::vector<int>();
        }

    private:
        /** @brief 工作线程函数 (绑定固定的 stream/context) */
        void workThread(int idx);

        /** @brief 从文件加载引擎 */
        void LoadEngine(const std::string &engine_path);

        /** @brief 获取输入张量属性 */
        void getInputProperty();

        /** @brief 获取输出张量属性 */
        void getOutputProperty();

        /** @brief 分配 Stream/Context/内存 */
        void allocatePair();

        /** @brief 创建工作线程 */
        void createWorkthreads();

        /** @brief 分配输入输出显存 */
        void allocBindings(std::unordered_map<std::string, void *> &inputBindings,
                           std::unordered_map<std::string, void *> &outputBindings,
                           nvinfer1::IExecutionContext *context);

        /** @brief 分配输入主机锁页内存 (H2D 暂存) */
        void allocInBlobPinned(std::unordered_map<std::string, void *> &inputBlobPin);

        /** @brief 分配输出主机锁页内存 */
        void allocOutBlobPinned(std::unordered_map<std::string, void *> &outputBlobPin);

        /** @brief 上传输入数据到 GPU (host -> pinned -> device) */
        void uploadInput(const std::string &name, const cv::Mat &mat, StreamContextDecl &ctx);

        /** @brief 下载输出数据到 CPU (device -> pinned) */
        void downloadOutputPin(const std::string &name, StreamContextDecl &ctx);

    private:
        std::string engine_path_;  /**< @brief 引擎文件路径 */
        bool initialized_ = false; /**< @brief 初始化标志 */

        TRTInfer *parent_; /**< @brief 父类指针 */

        std::unique_ptr<nvinfer1::IRuntime> runtime_;   /**< @brief TensorRT 运行时 */
        std::unique_ptr<nvinfer1::ICudaEngine> engine_; /**< @brief TensorRT 引擎 */

        std::unordered_map<std::string, std::vector<int>> current_input_shapes_; /**< @brief 当前输入形状 */

        std::vector<std::string> input_names_, output_names_;              /**< @brief 输入输出名称 */
        std::unordered_map<std::string, size_t> input_size_, output_size_; /**< @brief 输入/输出已分配的最大字节容量 */
        std::unordered_map<std::string, std::vector<int>> output_shape_;   /**< @brief 输出形状(引擎声明，动态维为 -1) */
        std::unordered_map<std::string, bool> input_is_dynamic_;           /**< @brief 输入是否含动态维 */
        std::unordered_map<std::string, std::vector<int>> input_max_dims_; /**< @brief 输入最大维度(动态取 profile kMAX) */
        Logger logger;                                                     /**< @brief 日志记录器 */

    private:
        std::queue<std::unique_ptr<InferTask>> task_queues_; /**< @brief 任务队列 */

        std::vector<StreamContextDecl> stream_ctxs_; /**< @brief 每线程独占的 stream/context/缓冲 (须在 engine_ 之后销毁) */

        int num_threads_;                     /**< @brief 线程数量 */
        bool b_stop_ = false;                 /**< @brief 停止标志 */
        std::vector<std::thread> thread_pool; /**< @brief 线程池 */

        std::condition_variable cond_; /**< @brief 条件变量 */
        std::mutex task_mutext_;       /**< @brief 任务互斥锁 */
    };

    // TRTInfer 公共接口实现

    TRTInfer::TRTInfer(const std::string &engine_path, int num_thread)
        : pImpl(std::make_unique<Impl>(engine_path, num_thread, this))
    {
    }

    TRTInfer::~TRTInfer() = default;

    BlobType TRTInfer::operator()(const BlobType &input_blob)
    {
        return pImpl->infer(input_blob);
    }

    std::future<BlobType> TRTInfer::PostQueue(const BlobType &input_blob)
    {
        return pImpl->PostQueue(input_blob);
    }

    std::vector<std::string> TRTInfer::getInputNames() const
    {
        return pImpl->getInputNames();
    }

    std::vector<std::string> TRTInfer::getOutputNames() const
    {
        return pImpl->getOutputNames();
    }

    TensorShape TRTInfer::getInputShape(const std::string &name) const
    {
        return utility::vectorToShape(pImpl->getInputShapeVec(name));
    }

    TensorShape TRTInfer::getOutputShape(const std::string &name) const
    {
        return utility::vectorToShape(pImpl->getOutputShapeVec(name));
    }

    // TRTInfer::Impl 实现

    TRTInfer::Impl::Impl(const std::string &engine_path, int num_thread, TRTInfer *parent)
        : engine_path_(engine_path), num_threads_(num_thread), parent_(parent), logger()
    {
    }

    /**
     * @brief 初始化引擎：加载引擎、获取张量信息、分配资源、创建线程
     */
    void TRTInfer::Impl::Initialized()
    {
        LoadEngine(engine_path_);
        getInputProperty();
        getOutputProperty();
        allocatePair();
        createWorkthreads();
    }

    /**
     * @brief 析构函数：停止线程并释放资源
     */
    TRTInfer::Impl::~Impl()
    {
        std::cout << "[TRTInfer::Impl] 开始关闭" << std::endl;
        {
            std::lock_guard<std::mutex> lock(task_mutext_);
            b_stop_ = true;
        }
        cond_.notify_all();
        for (auto &thread : thread_pool)
            thread.join();
        std::cout << "[TRTInfer::Impl] 释放" << std::endl;
    }

    /**
     * @brief 从文件加载 TensorRT 引擎
     */
    void TRTInfer::Impl::LoadEngine(const std::string &engine_path)
    {
        // 读取引擎文件
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good())
        {
            file.close();
            std::cerr << "[TRTInfer::Impl] Error reading engine file" << std::endl;
            throw std::runtime_error("Error reading engine file: " + engine_path);
        }

        file.seekg(0, file.end);
        const size_t fsize = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> engineData(fsize);
        file.read(engineData.data(), fsize);
        file.close();

        // 创建运行时
        runtime_.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime_)
        {
            std::cerr << "[TRTInfer::Impl] Failed to create runtime" << std::endl;
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        // 初始化插件并反序列化引擎
        initLibNvInferPlugins(&logger, "TRT");
        engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), fsize));
        if (!engine_)
        {
            std::cerr << "[TRTInfer::Impl] Failed to create engine" << std::endl;
            throw std::runtime_error("Failed to deserialize TensorRT engine");
        }
    }

    /// std::vector<int> -> nvinfer1::Dims
    static nvinfer1::Dims makeDims(const std::vector<int> &v)
    {
        nvinfer1::Dims d;
        d.nbDims = static_cast<int>(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            d.d[i] = v[i];
        return d;
    }

    static std::vector<int> dimsToVec(const nvinfer1::Dims &d)
    {
        return std::vector<int>(d.d, d.d + d.nbDims);
    }

    /**
     * @brief 获取所有输入张量信息
     *
     * 记录引擎声明形状(动态维为 -1)；对动态输入用 optimization profile 的 kMAX
     * 求最大维度，据此确定按最大容量分配的字节数。
     */
    void TRTInfer::Impl::getInputProperty()
    {
        for (int i = 0; i < engine_->getNbIOTensors(); i++)
        {
            const char *name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT)
                continue;

            std::cout << "[TRTInfer::Impl] input tensor name : " << name
                      << ", tensor shape : " << engine_->getTensorShape(name)
                      << ", tensor type : " << engine_->getTensorDataType(name)
                      << ", tensor format : " << engine_->getTensorFormatDesc(name)
                      << std::endl;

            const std::string sname(name);
            input_names_.emplace_back(sname);

            // 引擎声明形状(动态维为 -1)，用于对外查询
            const nvinfer1::Dims eng_dims = engine_->getTensorShape(name);
            current_input_shapes_[sname] = dimsToVec(eng_dims);

            // 是否含动态维
            bool dynamic = false;
            for (int k = 0; k < eng_dims.nbDims; ++k)
                if (eng_dims.d[k] < 0)
                    dynamic = true;
            input_is_dynamic_[sname] = dynamic;

            // 最大维度：动态取 profile(0) 的 kMAX，静态即声明形状
            nvinfer1::Dims max_dims = dynamic
                ? engine_->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX)
                : eng_dims;
            input_max_dims_[sname] = dimsToVec(max_dims);

            // 按最大容量分配
            input_size_[sname] = utility::getTensorbytes(max_dims, engine_->getTensorDataType(name));

            if (dynamic)
                std::cout << "[TRTInfer::Impl]   dynamic input, max shape : " << max_dims << std::endl;
        }
    }

    /**
     * @brief 获取所有输出张量信息
     */
    void TRTInfer::Impl::getOutputProperty()
    {
        for (int i = 0; i < engine_->getNbIOTensors(); i++)
        {
            const char *name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
            {
                std::cout << "[TRTInfer::Impl] output tensor name : " << name
                          << ", tensor shape : " << engine_->getTensorShape(name)
                          << ", tensor type : " << engine_->getTensorDataType(name)
                          << ", tensor format : " << engine_->getTensorFormatDesc(name)
                          << std::endl;

                output_names_.emplace_back(std::string(name));
                output_size_[std::string(name)] = utility::getTensorbytes(
                    engine_->getTensorShape(name), engine_->getTensorDataType(name));

                nvinfer1::Dims dims = engine_->getTensorShape(name);
                std::vector<int> dim;
                dim.reserve(dims.nbDims);
                for (int i = 0; i < dims.nbDims; i++)
                    dim.emplace_back(dims.d[i]);
                output_shape_[std::string(name)] = dim;
            }
        }
    }

    /**
     * @brief 为输入输出分配 GPU 显存
     */
    void TRTInfer::Impl::allocBindings(std::unordered_map<std::string, void *> &inputBindings,
                                       std::unordered_map<std::string, void *> &outputBindings,
                                       nvinfer1::IExecutionContext *context)
    {
        // 分配输入显存
        for (int i = 0; i < input_names_.size(); i++)
        {
            void *ptr = utility::safeCudaMalloc(input_size_[input_names_[i]]);
            if (!ptr)
                throw std::runtime_error("Failed to allocate GPU memory");
            inputBindings[input_names_[i]] = ptr;
            context->setInputTensorAddress(input_names_[i].c_str(), inputBindings[input_names_[i]]);
        }

        // 分配输出显存
        for (int i = 0; i < output_names_.size(); i++)
        {
            void *ptr = utility::safeCudaMalloc(output_size_[output_names_[i]]);
            if (!ptr)
                throw std::runtime_error("Failed to allocate GPU memory");
            outputBindings[output_names_[i]] = ptr;
            context->setOutputTensorAddress(output_names_[i].c_str(), outputBindings[output_names_[i]]);
        }
    }


    void TRTInfer::Impl::allocInBlobPinned(std::unordered_map<std::string, void *> &inputBlobPin)
    {
        for (const auto &name : input_names_)
        {
            void *ptr = utility::safeCudaMallocHost(input_size_[name]);
            if (!ptr)
                throw std::runtime_error("Failed to allocate pinned host memory");
            inputBlobPin[name] = ptr;
        }
    }

    void TRTInfer::Impl::allocOutBlobPinned(std::unordered_map<std::string, void *> &outputBlobPin)
    {
        for (const auto &name : output_names_)
        {
            size_t datasize = output_size_[name];
            void *ptr = utility::safeCudaMallocHost(datasize);
            if (!ptr)
                throw std::runtime_error("Failed to allocate pinned host memory");
            outputBlobPin[name] = ptr;
        }
    }

    /**
     * @brief 同步推理
     */
    BlobType TRTInfer::Impl::infer(const BlobType &input_blob)
    {
        auto future = this->PostQueue(input_blob);
        BlobType results = std::move(future.get());
        return results;
    }

    /**
     * @brief 执行推理任务 (使用调用线程独占的 ctx，无需加锁获取资源)
     */
    BlobType TRTInfer::Impl::infer_task(StreamContextDecl &ctx, const BlobType &input_blob)
    {
        // 上传输入
        for (const auto &[name, mat] : input_blob)
        {
            uploadInput(name, mat, ctx);
        }

        // 执行推理
        if (!ctx.context->enqueueV3(ctx.stream))
        {
            std::cerr << "[TRTInfer::Impl] enqueueV3 failed" << std::endl;
            throw std::runtime_error("[TRTInfer::Impl] enqueueV3 failed");
        }

        // 下载输出
        for (const auto &name : output_names_)
        {
            downloadOutputPin(name, ctx);
        }

        // 等待完成
        cudaStreamSynchronize(ctx.stream);

        // 封装结果 (按本次推理的实际输出形状)
        BlobType tmp_results;
        for (auto &name : output_names_)
        {
            std::vector<int> dims = dimsToVec(ctx.context->getTensorShape(name.c_str()));
            cv::Mat temp(
                static_cast<int>(dims.size()),
                dims.data(),
                utility::typeRt2Cv(engine_->getTensorDataType(name.c_str())),
                ctx.outputBlobsPin[name]);
            tmp_results[name] = temp.clone();
        }
        return tmp_results;
    }

    /**
     * @brief 上传输入数据到 GPU
     */
    void TRTInfer::Impl::uploadInput(const std::string &name, const cv::Mat &mat, StreamContextDecl &ctx)
    {
        auto bind_it = ctx.inputBindings.find(name);
        if (bind_it == ctx.inputBindings.end())
            return;

        cv::Mat cpu_mat = mat;

        // 类型转换 (按需)
        const nvinfer1::DataType dtype = engine_->getTensorDataType(name.c_str());
        if (utility::typeCv2Rt(cpu_mat.type()) != dtype)
            cpu_mat.convertTo(cpu_mat, utility::typeRt2Cv(dtype));

        // 非连续数据 memcpy 会拷错，clone 成连续内存
        if (!cpu_mat.isContinuous())
        {
            std::cerr << "[TRTInfer::Impl - WARNING] Input cv::Mat for '" << name << "' is not continuous, cloning" << std::endl;
            cpu_mat = cpu_mat.clone();
        }

        const size_t capacity = input_size_.at(name); // 已分配的最大字节容量
        const size_t mat_size = cpu_mat.total() * cpu_mat.elemSize();

        if (input_is_dynamic_.at(name))
        {
            // 动态输入：按 cv::Mat 实际维度设置本次推理的输入形状
            std::vector<int> dims;
            dims.reserve(cpu_mat.dims);
            for (int k = 0; k < cpu_mat.dims; ++k)
                dims.push_back(cpu_mat.size[k]);
            if (!ctx.context->setInputShape(name.c_str(), makeDims(dims)))
            {
                std::cerr << "[TRTInfer::Impl - ERROR] setInputShape failed for '" << name
                          << "' (shape out of profile range?)" << std::endl;
                throw std::runtime_error("[TRTInfer::Impl] setInputShape failed for " + name);
            }
            if (mat_size > capacity)
                throw std::runtime_error("[TRTInfer::Impl] Input '" + name + "' exceeds max profile capacity");
        }
        else if (mat_size != capacity)
        {
            std::cerr << "[TRTInfer::Impl - ERROR] Input tensor size mismatch for '" << name << "': "
                      << "required " << capacity << " bytes, but cv::Mat has " << mat_size << " bytes" << std::endl;
            throw std::runtime_error("[TRTInfer::Impl] Input tensor size mismatch");
        }

        // host -> pinned -> device：先拷进锁页暂存，再异步 H2D，使传输可与其他流计算重叠
        void *pinned = ctx.inputBlobsPin.at(name);
        std::memcpy(pinned, cpu_mat.data, mat_size); // 拷贝数据

        cudaError_t err = cudaMemcpyAsync(bind_it->second, pinned, mat_size, cudaMemcpyHostToDevice, ctx.stream);
        if (err != cudaSuccess)
        {
            std::cerr << "[TRTInfer::Impl] CUDA memcpyAsync (H2D) failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error(cudaGetErrorString(err));
        }
        ctx.context->setInputTensorAddress(name.c_str(), bind_it->second);
    }

    /**
     * @brief 下载输出数据到 CPU 锁页内存
     */
    void TRTInfer::Impl::downloadOutputPin(const std::string &name, StreamContextDecl &ctx)
    {
        // 获取实际输出形状
        nvinfer1::Dims out_shape = ctx.context->getTensorShape(name.c_str());
        size_t actual_size = utility::getTensorbytes(out_shape, engine_->getTensorDataType(name.c_str()));

        // 验证不超过已分配的最大容量 (动态时实际可小于容量)
        if (actual_size > output_size_.at(name))
        {
            std::cerr << "[ERROR] Output exceeds allocated capacity for '" << name << "': "
                      << "actual " << actual_size << " bytes, capacity " << output_size_.at(name)
                      << " bytes" << std::endl;
            throw std::runtime_error("Output exceeds allocated capacity");
        }

        // 拷贝到主机
        cudaError_t err = cudaMemcpyAsync(ctx.outputBlobsPin.at(name), ctx.outputBindings.at(name),
                                          actual_size, cudaMemcpyDeviceToHost, ctx.stream);
        if (err != cudaSuccess)
        {
            std::cerr << "[TRTInfer::Impl] CUDA memcpyAsync (D2H) failed: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error(cudaGetErrorString(err));
        }
    }

    /**
     * @brief 推送任务到队列
     */
    std::future<BlobType> TRTInfer::Impl::PostQueue(const BlobType &input_blob)
    {
        auto task = std::make_unique<InferTask>(input_blob);
        auto future = task->get_future();
        {
            std::lock_guard<std::mutex> lock(task_mutext_);
            task_queues_.push(std::move(task));
        }
        cond_.notify_one();
        return future;
    }

    /**
     * @brief 工作线程函数：独占 stream_ctxs_[idx]，循环取任务执行
     */
    void TRTInfer::Impl::workThread(int idx)
    {
        StreamContextDecl &ctx = stream_ctxs_[idx];
        while (true)
        {
            std::unique_ptr<InferTask> task;
            {
                std::unique_lock<std::mutex> lock(task_mutext_);
                cond_.wait(lock, [this]()
                           { return b_stop_ || !task_queues_.empty(); });
                // 停止时先排空剩余任务，避免 future 收到 broken_promise
                if (task_queues_.empty())
                    return;
                task = std::move(task_queues_.front());
                task_queues_.pop();
            }

            if (task)
            {
                try
                {
                    task->promise().set_value(infer_task(ctx, task->args()));
                }
                catch (...)
                {
                    task->promise().set_exception(std::current_exception());
                }
            }
        }
    }

    /**
     * @brief 为每个工作线程分配独占的 stream/context/显存/锁页缓冲
     */
    void TRTInfer::Impl::allocatePair()
    {
        stream_ctxs_.reserve(num_threads_);
        for (int i = 0; i < num_threads_; i++)
        {
            StreamContextDecl ctx;
            cudaStreamCreate(&ctx.stream);
            ctx.context = engine_->createExecutionContext();
            if (!ctx.context)
                throw std::runtime_error("Failed to create execution context");

            // 动态输入先设到最大形状，以便推导输出最大容量并按最大分配
            for (const auto &name : input_names_)
                if (input_is_dynamic_[name])
                    if (!ctx.context->setInputShape(name.c_str(), makeDims(input_max_dims_[name])))
                        throw std::runtime_error("setInputShape(max) failed for " + name);

            // 在最大输入下推导每个输出的最大容量字节
            for (const auto &name : output_names_)
                output_size_[name] = utility::getTensorbytes(
                    ctx.context->getTensorShape(name.c_str()), engine_->getTensorDataType(name.c_str()));

            allocBindings(ctx.inputBindings, ctx.outputBindings, ctx.context);
            allocInBlobPinned(ctx.inputBlobsPin);
            allocOutBlobPinned(ctx.outputBlobsPin);

            stream_ctxs_.push_back(std::move(ctx));
        }
    }

    /**
     * @brief 创建工作线程，每个线程绑定一个 ctx 下标
     */
    void TRTInfer::Impl::createWorkthreads()
    {
        for (int i = 0; i < num_threads_; i++)
            thread_pool.emplace_back([this, i]()
                                     { workThread(i); });
    }

    /**
     * @brief 初始化引擎
     */
    void TRTInfer::Init()
    {
        pImpl->Initialized();
    }

    /**
     * @brief 工厂方法：创建 TRTInfer 实例
     */
    std::shared_ptr<TRTInfer> TRTInfer::create(const std::string &engine_path, int num_thread)
    {
        auto instance_ = std::shared_ptr<TRTInfer>(new TRTInfer(engine_path, num_thread));
        instance_->Init();
        return instance_;
    }
}
