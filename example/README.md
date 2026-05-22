# Examples

Each program here is a self-contained `preprocess → inference → postprocess` pipeline built on
`TRT::TRTInfer`. They show how to wrap a specific model's I/O into `TRT::BlobType`
(`std::unordered_map<std::string, cv::Mat>`).

> **Run from the repo root**, not from `example/` or `build/`, so relative paths like
> `./demo/bus.jpg` and `./yolov8n.engine` resolve. Engines are not committed — generate them
> first (see [Generating engines](../README.md#️-generating-engines)). Pretrained ONNX:
> [GoogleDrive](https://drive.google.com/drive/folders/19UBgYWeEADKTA1w44HIDkzn2oPxKATOH?usp=drive_link).

| Source | Executable | Task | Engine (default) | Input tensor(s) |
|--------|------------|------|------------------|-----------------|
| `YOLO.cc` | `yolo` | Object detection + QPS benchmark | `./yolov8n.engine` | `images` |
| `YOLOVideo.cc` | `yolo_video` | Detection on video (pipelined) | `./yolov8n.engine` | `images` |
| `YOLOBatch.cc` | `yolo_batch` | Detection, dynamic batch sweep | `./yolov8n_dyn.engine` | `images` |
| `FCN.cc` | `fcn` | Semantic segmentation | `fcn.engine` | `input` |
| `Segformer.cc` | `segformer` | Semantic segmentation | `segformer.engine` | `input` |
| `IGEV.cc` | `igev` | Stereo matching | *(edit path in source)* | `left`, `right` |
| `LiteAnyStereo.cc` | `liteanystereo` | Stereo matching | `liteanystereo.engine` | `left_image`, `right_image` |
| `LiteAnyStereoVideo.cc` | `liteanystereo_video` | Stereo on a video pair | `liteanystereo.engine` | `left_image`, `right_image` |
| `s2m2.cc` | `s2m2` | Stereo matching | `S2M2.engine` | `input_left`, `input_right` |

---

## Detection — YOLO

### `yolo` — single-image detection + benchmark

Loads `demo/bus.jpg`, resizes to 480×640, runs YOLOv8 (`images` → `output0`, `1×84×6300`), and
reports two QPS numbers: **inference-only** (blob reused) and **end-to-end** (preprocess each
iteration). Result saved to `demo/yolo_output.png`.

```bash
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.engine --fp16
# yolo [engine] [num_thread] [iters]
./build/yolo ./yolov8n.engine 4 2000
```

`YOLO_SHOW=1 ./build/yolo` opens a window with the drawn boxes.

### `yolo_video` — pipelined video detection

Decodes a video, keeps `num_thread` frames **in flight** (submit-ahead, in-order collect) so all
streams stay busy, and reports per-stage timing + wall-clock throughput.

```bash
# yolo_video [video] [engine] [num_thread] [save]
./build/yolo_video ./demo/bus_40s_60fps.mp4 ./yolov8n.engine 4
./build/yolo_video ./demo/bus_40s_60fps.mp4 ./yolov8n.engine 4 save   # writes *_output.mp4
```

A 40s/60fps test clip can be generated with [`demo/make_bus_video.sh`](../demo/make_bus_video.sh).

### `yolo_batch` — dynamic batch throughput

Packs N copies of `demo/bus.jpg` into one `N×3×640×480` blob with `cv::dnn::blobFromImages`,
runs a single batched inference, and slices the `N×84×6300` output per image for NMS. Sweeps
batch sizes `{1,2,4,8}` and prints images/sec. **Requires a dynamic engine** (see
[Dynamic batch](../README.md#-dynamic-batch)).

```bash
trtexec --onnx=yolov8n_dyn.onnx --saveEngine=yolov8n_dyn.engine --fp16 \
  --minShapes=images:1x3x640x480 --optShapes=images:4x3x640x480 --maxShapes=images:8x3x640x480
# yolo_batch [engine] [num_thread] [iters]
./build/yolo_batch ./yolov8n_dyn.engine 4 500
```

Output of image 0 saved to `demo/yolo_batch_output.png`.

---

## Segmentation — FCN / Segformer

Both normalize the image, resize to 512×512, feed the `input` tensor, then colorize the predicted
mask. `fcn` reads `demo/bus.jpg`; `segformer` reads `demo/image.png`.

```bash
trtexec --onnx=fcn.onnx       --saveEngine=fcn.engine
trtexec --onnx=segformer.onnx --saveEngine=segformer.engine
./build/fcn          # -> demo/fcn_output.png, demo/fcn_mask.png
./build/segformer    # -> demo/segformer_output.png, demo/segformer_mask.png
```

---

## Stereo matching — IGEV / LiteAnyStereo / S2M2

All take a rectified left/right pair and produce a colorized disparity map.

### `igev`

Inputs `left` / `right`, output `disparity`. **The engine and image paths are hardcoded at the
top of `IGEV.cc`** to the author's machine — edit `engine_path`, `left_path`, `right_path`
(e.g. `./demo/left.png`, `./demo/right.png`) before running. Result saved to
`demo/disp_output.png`.

```bash
trtexec --onnx=igev_480_752.onnx --saveEngine=igev_480_752.engine --fp16
./build/igev
```

### `liteanystereo` — CLI driven

Inputs `left_image` / `right_image` (output `disparity` or `output`). Configurable via flags:

```bash
./build/liteanystereo \
  --left_img demo/left.png --right_img demo/right.png \
  --engine liteanystereo.engine --output_dir ./output \
  --target_size 480,752 --warmup 5 --runs 50 [--normalize]
```

`liteanystereo_video` is the same model over a synchronized left/right video pair (run with
`--help` for its flags).

### `s2m2`

Inputs `input_left` / `input_right`. Reads `rect_left.png` / `rect_right.png`, writes
`demo/s2m2_disp.png`.

```bash
trtexec --onnx=S2M2.onnx --saveEngine=S2M2.engine --fp16
./build/s2m2
```

---

## Adding your own example

1. Create `example/MyModel.cc` with `preprocess` (returns a `BlobType` keyed by the engine's
   **input** tensor names) and `postprocess`.
2. Register it in `../CMakeLists.txt`:
   ```cmake
   add_executable(mymodel example/MyModel.cc)
   target_link_libraries(mymodel trtemplate)
   ```
3. Reconfigure (`cmake -S . -B build`) and build. See the
   [template section](../README.md#-template--adding-your-own-model) for the skeleton and pitfalls.
