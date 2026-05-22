## <div align="center">📄 TensorRT Template</div>

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

```bash
./build/yolo_batch ./yolov8n_dyn.engine 4 500
```

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
