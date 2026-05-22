## <div align="center">📄 TensorRT Template</div>

<<<<<<< HEAD
### 🛠️ 简介
这是一个支持 OpenCV cv::Mat 类型数据的 TensorRT 推理模板库，支持多输入多输出数据。

### ✒️ 环境要求
* Windows 11 / Ubuntu 20.04
* Visual Studio 2022 ~ 2026 / GNU
* CMake 3.20+
* TensorRT 10.x
* OpenCV > 4.5
* CUDA 11.x / 12.x


### 📊 速度对比
测试设备
* GPU : Nvidia 4060Ti 16G
* CPU : i5-12600kf 12核16线程


测试模型包括 目标检测、语义分割、立体匹配算法，在三个图像尺寸上进行测试，每张表格的数字以ms为单位。

* 目标检测 - YOLO26

| 模型 | 480×640 | 480×1280 | 736×1280 |
|:---|:---:|:---:|:---:|
| YOLO26l (Pytorch - FP32) | 19.29 | 22.06 | 41.06 |
| YOLO26l (TensorRT - FP32) | 7.01 | 11.99| 18.77 |
| YOLO26l (TensorRT - FP16) | 3.36 | 4.92 | 6.08 |
| YOLO26l (TensorRT - INT8) | 3.29 | 3.38 | 4.03 |


* 语义分割

todo

* 立体视觉 - IGEV-Stereo

| 模型 | 480×736 | 480×1280 | 736×1280 |
|:---|:---:|:---:|:---:|
| IGEV-Stereo (Pytorch - FP32) | 120.9 | 206.1 | 326.8 |
| IGEV-Stereo (TensorRT - FP32) | 65.2 | 118.9 | 207.2|
| IGEV-Stereo (TensorRT - FP16) | 30.1 | 45.89 | 88.34  |
| IGEV-Stereo (TensorRT - INT8) | 18.01 | 33.98 | 67.35 |

todo



### ⚙️ 常用配置

#### CMake 配置
构建前需要在 `CMakeLists.txt` 中配置库路径：

**Windows:**
```cmake
set(CUDA_ROOT_DIR "E:/lib/cuda/12.1")
set(TensorRT_root_dir "E:/lib/TensorRT/TensorRT-10.10.0.31")
set(OpenCV_root_dir "E:/lib/opencv/opencv-4.8.0/build/x64/vc16/lib")
set(LIB_TYPE SHARED)  # SHARED (.dll) 或 STATIC (.lib)
```

**Linux:**
```cmake
set(CUDA_ROOT_DIR "/usr/local/cuda")
set(TensorRT_root_dir "/usr/local/TensorRT-10.10.0.31")
set(LIB_TYPE SHARED)  # SHARED (.so) 或 STATIC (.a)
```

#### 库类型选项
- `LIB_TYPE = SHARED`: 动态库 (Windows: `.dll`, Linux: `.so`)
- `LIB_TYPE = STATIC`: 静态库 (Windows: `.lib`, Linux: `.a`)
=======
### 🛠️ Introduction

`trtemplate` is a C++17 template library that wraps **TensorRT 10.x** inference behind an
OpenCV-friendly API. The whole public surface speaks
`std::unordered_map<std::string, cv::Mat>` (aliased `TRT::BlobType`), so a model with any
number of named inputs/outputs is just a map of `cv::Mat`. It is built for high throughput:
multiple CUDA streams run concurrently, work is dispatched through an asynchronous queue, and
host↔device transfers go through pinned memory.

- **Multi-Stream Concurrency** — each worker thread owns its own CUDA stream + execution
  context, so `num_thread` inferences run in parallel with no per-call locking.
- **Async Queue** — submit with `PostQueue()` and collect `std::future`s later to saturate the
  streams; or call `operator()` for a blocking single inference.
- **Multi-Input / Multi-Output** — wrap any set of named tensors in one `BlobType`.
- **Dynamic Batch / Dynamic Shape** — engines built with an optimization profile are detected
  automatically; the input shape is set per inference from the `cv::Mat` you pass in. Static
  engines keep working unchanged.
- **Pinned-Memory Staging** — inputs (H2D) and outputs (D2H) are staged through page-locked host
  buffers so transfers overlap with compute on other streams.
- **Cross-Platform** — Windows (Visual Studio) and Linux (GCC / CMake).

### 🧩 Architecture

The library is three thin layers; everything is dispatched through a worker-thread queue.

| Layer | File | Responsibility |
|-------|------|----------------|
| `TRT::TRTInfer` | `TRTInfer/TRTinfer.{h,cc}` | Public API (Pimpl). Header depends only on OpenCV + std — no TensorRT/CUDA leakage. |
| Queue + thread pool | inside `TRTInfer::Impl` | `PostQueue` enqueues an `InferTask`; `num_thread` workers pop and run. |
| `StreamContextDecl` | `TRTInfer/StreamContextDecl.{h,cc}` | RAII bundle of one stream + execution context + device I/O buffers + pinned staging. One per worker. |
| `utility` / `benchmark` | `TRTInfer/utility.*`, `benchmark.h` | dtype mapping, byte-size math, guarded CUDA alloc/free, timing harness. |

`num_thread` is the real concurrency knob — it equals the number of CUDA streams / execution
contexts. Each worker thread is permanently bound to one `StreamContextDecl`, so the hot path
has no resource-pool contention.
>>>>>>> main_multistream

### ✒️ Environment

| Component | Version Requirement |
|-----------|---------------------|
| OS | Windows 11 / Ubuntu 20.04+ |
| Compiler | Visual Studio 2022+ / GCC (C++17) |
| CUDA | 11.x / 12.x |
| TensorRT | **10.x** (8.x is *not* compatible) |
| OpenCV | 4.5+ |
| CMake | 3.15+ |
| C++ | C++17 |

### 📦 构建

<<<<<<< HEAD
#### Windows

1. **安装依赖:**
   - 下载并安装 [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - 下载并安装 [TensorRT](https://developer.nvidia.com/tensorrt) 10.x
   - 下载并编译 [OpenCV](https://opencv.org/releases/)

2. **配置 CMakeLists.txt:**
   ```cmake
   set(CUDA_ROOT_DIR "Your/CUDA/Path")
   set(TensorRT_root_dir "Your/TensorRT/Path")
   set(OpenCV_root_dir "Your/OpenCV/Path")
   ```

3. **构建:**
   ```bash
   cmake -S . -B build
   cmake --build build --config release
   ```

4. **输出:**
   - 库: `build/Release/trtemplate.dll` 和 `build/Release/trtemplate.lib`
   - 可执行文件: `build/Release/yolo.exe`, `build/Release/fcn.exe`

#### Linux

1. **安装依赖:**
   ```bash
   sudo apt update
   sudo apt install cuda-toolkit-12-x
   sudo apt install libopencv-dev
   ```

2. **安装 TensorRT:**
   从 [NVIDIA 官网](https://developer.nvidia.com/tensorrt) 下载并安装

3. **配置 CMakeLists.txt:**
   ```cmake
   set(CUDA_ROOT_DIR "/usr/local/cuda")
   set(TensorRT_root_dir "/path/to/TensorRT")
   ```

4. **构建:**
   ```bash
   cmake -S . -B build
   cmake --build build --config release -j 12
   ```

5. **输出:**
   - 库: `build/libtrtemplate.so`
   - 可执行文件: `build/yolo`, `build/fcn`

#### 常见构建问题

1. **TensorRT 版本兼容性:**
   - TensorRT 8.x 不兼容
   - 必须使用 TensorRT 10.x

2. **CUDA 版本匹配:**
   - 确保 CUDA 版本与 TensorRT 匹配
   - TensorRT 10.x 需要 CUDA 11.8 或 12.x

3. **OpenCV 路径问题:**
   - Windows: 指向包含 `*.lib` 文件的 `lib` 目录
   - Linux: 确保 `pkg-config opencv4 --cflags --libs` 可用

4. **构建类型:**
   - 生产环境使用 Release (`--config release`)
   - 开发调试使用 Debug


### ✨ 使用示例

首先，从链接 [GoogleDrive](https://drive.google.com/drive/folders/19UBgYWeEADKTA1w44HIDkzn2oPxKATOH?usp=drive_link) 下载 YOLOv8 和 FCN 的 onnx 文件。

#### YOLOv8 - 目标检测示例
将 onnx 文件转换为 engine 文件：
```bash
trtexec \
--onnx=./pretrain/yolov8n.onnx \
--saveEngine=./yolov8n.engine
```

构建示例：
```bash
cmake -S . -B build
cmake --build ./build --config release -j 12
```

运行：
```bash
# Windows
./build/Release/yolo.exe
# Linux
./build/yolo
```

详细代码见 [YOLO.cc](example/YOLO.cc)

![alt text](demo/image.png)

#### FCN / Segformer - 语义分割示例
将 onnx 文件转换为 engine 文件：
```bash
trtexec \
--onnx=./pretrain/fcn.onnx \
--saveEngine=./fcn.engine
```

构建和运行方式同上，详细代码见 [FCN.cc](example/FCN.cc)，Segformer 使用方式类似。

![alt text](demo/image-1.png)


### 🔧 API 使用

#### 创建模型实例

```cpp
// 使用工厂方法创建实例（推荐）
auto model = TRTInfer::create("model.engine");

// 手动初始化（可选，用于预热或提前验证）
model->Init();

// 推理调用
std::unordered_map<std::string, cv::Mat> input;
input["images"] = cv::imread("test.jpg");

auto output = (*model)(input);
cv::Mat result = output["output"];
```

#### 查询张量信息

```cpp
// 获取输入/输出张量名称
auto input_names = model->getInputNames();
auto output_names = model->getOutputNames();

// 获取指定张量形状
TensorShape shape = model->getInputShape("images");
std::cout << "Batch: " << shape.n << ", Channel: " << shape.c
          << ", Height: " << shape.h << ", Width: " << shape.w << std::endl;
```

#### 动态批处理

```cpp
// 设置不同的批大小
model->setInputShape("input", {1, 3, 640, 640});   // 单张
auto output1 = (*model)(input1);

model->setInputShape("input", {4, 3, 640, 640});   // 批大小为4
auto output2 = (*model)(input_batch);
```


### 📋 模型模板

如果需要编写自己的模型推理加速，请按以下步骤操作：
1. 导出模型为 ONNX
2. 将 ONNX 转换为 engine
3. 编写预处理和后处理代码

以下是一个预处理和后处理代码模板：

```cpp
#include "TRTinfer.h"
#include <opencv2/opencv.hpp>

namespace model {

// 输入预处理
std::unordered_map<std::string, cv::Mat> preprocess(cv::Mat &left, cv::Mat &right)
{
    // ...
}

// 输出后处理
std::unordered_map<std::string, cv::Mat> postprocess(
    std::unordered_map<std::string, cv::Mat> output)
{
    // ...
}

} // namespace model

int main(int argc, char *argv[])
{
    cv::Mat tensor1 = cv::imread("...");
    cv::Mat tensor2 = cv::imread("...");

    // 预处理
    auto input_blob = model::preprocess(tensor1, tensor2);

    // 模型推理
    auto model = TRTInfer::create("*.engine");
    auto output_blob = (*model)(input_blob);

    // 后处理
    model::postprocess(output_blob);

    // 可视化
    // ...

    return 1;
}
```

### ⚠️ 注意事项

可能导致错误的几个关键点：

* **张量名称**
  * 某些权重文件的输入张量名称不一致
  * 实现前使用 `polygraphy` 验证张量名称

* **数据类型**
  * 内部可能涉及数据类型转换
  * 常见转换: float32 ↔ float16 (FP16), int8, uint8

* **输入数据形状**
  * 预处理包含 resize，但模型内部可能出错
  * 确保 NCHW vs NHWC 格式匹配模型预期


### ✅ 使用 Polygraphy 验证模型

建议执行前使用 Python 库 `polygraphy` 进行检查。例如，检查 YOLOv8：

```bash
$ polygraphy run yolov8n.onnx --onnxrt
[I] RUNNING | Command: /root/miniconda3/envs/dlpy310/bin/polygraphy run yolov8n.onnx --onnxrt
[I] onnxrt-runner-N0-11/21/25-13:22:08 | Activating and starting inference
[I] Creating ONNX-Runtime Inference Session with providers: ['CPUExecutionProvider']
[I] onnxrt-runner-N0-11/21/25-13:22:08
    ---- Inference Input(s) ----
    {images [dtype=float32, shape=(1, 3, 640, 480)]}
[I] onnxrt-runner-N0-11/21/25-13:22:08
    ---- Inference Output(s) ----
    {output0 [dtype=float32, shape=(1, 84, 6300)]}
[I] onnxrt-runner-N0-11/21/25-13:22:08 | Completed 1 iteration(s) in 59.11 ms | Average inference time: 59.11 ms.
[I] PASSED | Runtime: 1.037s | Command: /root/miniconda3/envs/dlpy310/bin/polygraphy run yolov8n.onnx --onnxrt
```

可以清楚看到张量的名称、类型和维度。


### 🔄 Engine 转换选项

将 ONNX 转换为 TensorRT engine 时，可以使用各种优化选项：

**基本转换:**
=======
#### Configure paths with `-D` flags (no need to edit CMakeLists.txt)

The TensorRT / CUDA roots and the library type are cache variables, overridable on the command
line:

```bash
cmake -S . -B build \
  -DTensorRT_ROOT_DIR=/path/to/TensorRT-10.x \
  -DCUDA_ROOT_DIR=/usr/local/cuda-12.x \
  -DLIB_TYPE=SHARED            # SHARED (.so/.dll) or STATIC (.a/.lib)
cmake --build build --config release -j
```

- Aliases `-DTensorRT_ROOT` / `-DCUDA_ROOT` are also accepted.
- On Linux, if `${CUDA_ROOT_DIR}/include/cuda_runtime.h` is not found, the build falls back to
  the distribution (apt) layout (`/usr/include` + `/usr/lib/x86_64-linux-gnu`), so an
  apt-installed CUDA works out of the box (`-DCUDA_ROOT_DIR=/usr`).
- A missing TensorRT or `libcudart` produces a clear `FATAL_ERROR` telling you which `-D` to set.

#### Linux

>>>>>>> main_multistream
```bash
sudo apt update
sudo apt install libopencv-dev          # OpenCV
# install CUDA Toolkit and TensorRT 10.x per NVIDIA's guides

cmake -S . -B build -DTensorRT_ROOT_DIR=/path/to/TensorRT-10.x
cmake --build build --config release -j
```

Outputs: `build/libtrtemplate.so` plus the example executables
(`yolo`, `yolo_video`, `yolo_batch`, `fcn`, `segformer`, `igev`, `liteanystereo`,
`liteanystereo_video`, `s2m2`). Run them from the repo root so relative paths like
`./demo/bus.jpg` and `./yolov8n.engine` resolve:

```bash
./build/yolo
```

#### Windows

Set `OpenCV_ROOT_DIR` in `CMakeLists.txt` (or via `-DCMAKE_PREFIX_PATH`), then:

```bash
cmake -S . -B build -DTensorRT_ROOT_DIR=E:/lib/TensorRT-10.x -DCUDA_ROOT_DIR="E:/lib/cuda/12.1"
cmake --build build --config release
```

Outputs land in `build/Release/` (`trtemplate.dll` + `trtemplate.lib`, `yolo.exe`, …).

> Release builds use aggressive flags (`-O3 -march=native -ffast-math -flto`); they are
> non-portable across CPUs by design.

### ⚙️ Generating engines

Examples consume `.engine` files (gitignored — generate them from ONNX with `trtexec`).

```bash
# Basic
trtexec --onnx=model.onnx --saveEngine=model.engine
<<<<<<< HEAD
```

**FP16 精度 (更快，精度略降):**
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

**INT8 精度 (最快，需要校准):**
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --int8 --calib=calibration.cache
```

**批大小配置:**
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=input:1x3x640x640 --optShapes=input:1x3x640x640 --maxShapes=input:1x3x640x640
```

**工作空间大小:**
```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --workspace=4096  # MB
```

**详细输出:**
=======
# FP16 (faster, slightly lower accuracy)
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
# INT8 (fastest, needs calibration)
trtexec --onnx=model.onnx --saveEngine=model.engine --int8 --calib=calibration.cache
```

Verify tensor names / shapes / dtypes with `polygraphy run model.onnx --onnxrt` before wiring up
a new model — mismatched tensor names are the most common failure.

### ✨ Base API

The constructor is private; create instances through the factory (returns a `shared_ptr`, eagerly
initialized):

```cpp
#include "TRTinfer.h"
#include <opencv2/opencv.hpp>
using namespace TRT;

int main() {
    // Load engine with 4 worker threads / streams
    auto model = TRTInfer::create("model.engine", 4);

    cv::Mat img = cv::imread("test.jpg");
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.0, cv::Size(640, 640));

    BlobType input;                 // = std::unordered_map<std::string, cv::Mat>
    input["images"] = blob;         // key must match the engine's input tensor name

    // (1) synchronous
    BlobType output = (*model)(input);

    // (2) asynchronous — submit many, then collect, to saturate the streams
    std::vector<std::future<BlobType>> futures;
    for (int i = 0; i < 100; ++i) futures.push_back(model->PostQueue(input));
    for (auto& f : futures) { BlobType out = f.get(); /* ... */ }

    return 0;
}
```

Introspection helpers: `getInputNames()`, `getOutputNames()`, `getInputShape(name)`,
`getOutputShape(name)`.

### 🔁 Dynamic batch

The library auto-detects dynamic dimensions (a `-1` in the engine's declared shape), sizes all
buffers to the profile's `kMAX`, and calls `setInputShape` per inference based on the actual
`cv::Mat` you pass. To use it:

**1. Build an engine with an optimization profile.** The ONNX must have a dynamic batch axis. If
yours is fixed (e.g. exported with batch=1), either re-export with `dynamic=True`, or make the
batch axis dynamic via ONNX graph surgery (set the input/output batch dim to a symbol and change
each `Reshape` constant's leading `1` to `0` = "copy input dim").

```bash
trtexec --onnx=yolov8n_dyn.onnx --saveEngine=yolov8n_dyn.engine --fp16 \
  --minShapes=images:1x3x640x480 \
  --optShapes=images:4x3x640x480 \
  --maxShapes=images:8x3x640x480
```

**2. Submit a batched blob** (`cv::dnn::blobFromImages` → `N×C×H×W`). The output comes back as a
batched `cv::Mat`; slice per image in postprocess. See `example/YOLOBatch.cc`.

>>>>>>> main_multistream
```bash
./build/yolo_batch ./yolov8n_dyn.engine 4 500
```

<<<<<<< HEAD

### 📁 项目结构

```
TensorRTTemplate/
├── TRTInfer/                 # 核心推理库
│   ├── TRTinfer.h           # 头文件
│   ├── TRTinfer.cc           # 实现文件
│   ├── utility.h             # 工具函数
│   ├── utility.cc            # 工具实现
│   ├── config.h              # 配置
│   └── benchmark.h            # 性能测试
├── example/                  # 示例代码
│   ├── YOLO.cc              # YOLOv8 目标检测
│   ├── FCN.cc               # FCN 语义分割
│   ├── Segformer.cc         # Segformer 语义分割
│   ├── IGEV.cc              # IGEV 双目匹配
│   ├── LiteAnyStereo.cc     # LiteAnyStereo 双目匹配
│   └── LiteAnyStereoVideo.cc # 视频流双目匹配
├── demo/                     # 示例图片
├── pretrain/                 # 预训练模型
├── CMakeLists.txt           # CMake 配置
└── README.md                # 说明文档
```
=======
### 📊 Benchmark

Test device — GPU: NVIDIA RTX 4060 Ti 16G · CPU: i5-12600KF.

**Dynamic-batch throughput** — yolov8n, FP16 dynamic engine, `num_thread=4` (`yolo_batch`):

| batch | ms / infer | images / sec |
|:---:|:---:|:---:|
| 1 | 1.88 | 533 |
| 2 | 2.42 | 827 |
| 4 | 3.78 | 1057 |
| 8 | 7.20 | **1112** |

Batching 8 ≈ **2.1×** the single-image throughput; with 4 streams that is up to 32 images in
flight per instance.

**Latency** (numbers in **milliseconds**, three image sizes):

* Object Detection — YOLO26

| Model | 480×640 | 480×1280 | 736×1280 |
|:---|:---:|:---:|:---:|
| YOLO26l (PyTorch FP32) | 19.29 | 22.06 | 41.06 |
| YOLO26l (TensorRT FP32) | 7.01 | 11.99 | 18.77 |
| YOLO26l (TensorRT FP16) | 3.36 | 4.92 | 6.08 |
| YOLO26l (TensorRT INT8) | 3.29 | 3.38 | 4.03 |

* Stereo Match — IGEV-Stereo

| Model | 480×736 | 480×1280 | 736×1280 |
|:---|:---:|:---:|:---:|
| IGEV-Stereo (PyTorch FP32) | 120.9 | 206.1 | 326.8 |
| IGEV-Stereo (TensorRT FP32) | 65.2 | 118.9 | 207.2 |
| IGEV-Stereo (TensorRT FP16) | 30.1 | 45.89 | 88.34 |
| IGEV-Stereo (TensorRT INT8) | 18.01 | 33.98 | 67.35 |

### 🧪 Examples

Per-model preprocess/inference/postprocess programs live in [`example/`](example/README.md).
Pretrained ONNX files: [GoogleDrive](https://drive.google.com/drive/folders/19UBgYWeEADKTA1w44HIDkzn2oPxKATOH?usp=drive_link).

| Executable | Task | Notes |
|------------|------|-------|
| `yolo` | Object detection | QPS benchmark (inference-only + end-to-end) |
| `yolo_video` | Detection on video | Pipelined across streams |
| `yolo_batch` | Detection, batched | Dynamic-batch throughput sweep |
| `fcn`, `segformer` | Semantic segmentation | |
| `igev`, `liteanystereo`, `s2m2` | Stereo matching | |
| `liteanystereo_video` | Stereo on a video pair | |

See [example/README.md](example/README.md) for inputs, engine names, and run commands.

<p align="center"><img src="demo/image.png" width="70%"></p>

### 🧱 Template — adding your own model

1. Export your model to ONNX, then convert to a `.engine` with `trtexec`.
2. Write `preprocess` (return a `BlobType` whose keys are the engine's **input** tensor names)
   and `postprocess` (consume the output `BlobType`).
3. Register the executable in `CMakeLists.txt` (`add_executable` + `target_link_libraries(<name> trtemplate)`).

```cpp
#include "TRTinfer.h"
#include <opencv2/opencv.hpp>
using namespace TRT;

namespace model {
    BlobType preprocess(const cv::Mat& img) { /* return {{"input", blob}} */ }
    void     postprocess(const BlobType& out) { /* ... */ }
}

int main() {
    auto model = TRTInfer::create("model.engine", 4);
    auto out   = (*model)(model::preprocess(cv::imread("test.jpg")));
    model::postprocess(out);
}
```

Common pitfalls:

- **Tensor names** — `BlobType` keys must exactly match the engine's I/O names. Verify with
  `polygraphy run model.onnx --onnxrt`.
- **Contiguity & size** — the input `cv::Mat` must be contiguous and byte-exact;
  `cv::dnn::blobFromImage[s]` gives a correct NCHW blob. Non-contiguous mats are cloned with a
  warning; size mismatches throw.
- **Dtype** — inputs are auto-converted to the engine's expected type, but confirm FP32/FP16/INT8
  expectations match how the engine was built.

### 🩺 Verifying a model with polygraphy

```bash
$ polygraphy run yolov8n.onnx --onnxrt
    ---- Inference Input(s) ----
    {images [dtype=float32, shape=(1, 3, 640, 480)]}
    ---- Inference Output(s) ----
    {output0 [dtype=float32, shape=(1, 84, 6300)]}
```

The tensor name, dtype, and dimensions are shown clearly — match these in your `preprocess`.

### ❓ Common build issues

1. **TensorRT version** — must be 10.x; 8.x is incompatible.
2. **CUDA mismatch** — TensorRT 10.x requires CUDA 11.8 or 12.x.
3. **OpenCV not found** — Windows: point `OpenCV_ROOT_DIR` at the `lib` dir; Linux: ensure
   `pkg-config opencv4 --cflags --libs` works.
4. **Wrong paths** — pass `-DTensorRT_ROOT_DIR` / `-DCUDA_ROOT_DIR`; the configure step reports
   exactly which one is missing.
>>>>>>> main_multistream
