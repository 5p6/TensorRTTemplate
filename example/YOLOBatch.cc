#include "TRTinfer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>

using namespace TRT;

namespace YOLO
{
    // 把 N 张图打包成 N×3×640×480 的 NCHW blob
    cv::Mat preprocessBatch(const std::vector<cv::Mat> &imgs)
    {
        std::vector<cv::Mat> resized;
        resized.reserve(imgs.size());
        for (const auto &im : imgs)
        {
            cv::Mat r;
            if (im.size() != cv::Size(480, 640))
                cv::resize(im, r, cv::Size(480, 640));
            else
                r = im;
            resized.push_back(r);
        }
        return cv::dnn::blobFromImages(resized, 1 / 255.f, cv::Size(), cv::Scalar(), true, false);
    }

    // 对批量输出 (N×84×6300) 中第 idx 张图做 NMS 并画框
    cv::Mat postprocessOne(const cv::Mat &out_batched, int idx, const cv::Mat &img, const cv::Size2f &scale)
    {
        cv::Mat imgc = img.clone();

        // 取第 idx 张的 84×6300 切片 (输出连续，每张图 84*6300 个 float)
        const int rows = out_batched.size[1]; // 84
        const int cols = out_batched.size[2]; // 6300
        cv::Mat one(rows, cols, CV_32F, const_cast<float *>(out_batched.ptr<float>(0) + (size_t)idx * rows * cols));

        cv::Mat pred;
        cv::transpose(one, pred); // 6300×84

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;

        const float confTh = 0.5f, nmsTh = 0.5f;
        for (int i = 0; i < pred.rows; ++i)
        {
            const float *row = pred.ptr<float>(i);
            cv::Mat cls(1, 80, CV_32F, const_cast<float *>(row + 4));
            cv::Point cls_id;
            double maxScore;
            cv::minMaxLoc(cls, nullptr, &maxScore, nullptr, &cls_id);
            if (maxScore > confTh)
            {
                float x = row[0], y = row[1], w = row[2], h = row[3];
                int left = int((x - 0.5f * w) * scale.width);
                int top = int((y - 0.5f * h) * scale.height);
                boxes.emplace_back(left, top, int(w * scale.width), int(h * scale.height));
                scores.push_back((float)maxScore);
                class_ids.push_back(cls_id.x);
            }
        }

        std::vector<int> keep;
        cv::dnn::NMSBoxes(boxes, scores, confTh, nmsTh, keep);
        for (int idx2 : keep)
        {
            cv::rectangle(imgc, boxes[idx2], cv::Scalar(0, 255, 0), 2);
            cv::putText(imgc, std::to_string(class_ids[idx2]), cv::Point(boxes[idx2].x, boxes[idx2].y),
                        cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0));
        }
        return imgc;
    }
}

int main(int argc, char *argv[])
{
    // yolo_batch [engine] [num_thread] [iters]
    std::string image_path = "./demo/bus.jpg";
    std::string engine_path = argc > 1 ? argv[1] : "./yolov8n_dyn.engine";
    int num_thread = argc > 2 ? std::atoi(argv[2]) : 4;
    int iters = argc > 3 ? std::atoi(argv[3]) : 500;

    using Blob = std::unordered_map<std::string, cv::Mat>;
    using FutureBlob = std::future<Blob>;

    auto model = TRT::TRTInfer::create(engine_path, num_thread);

    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return -1;
    }

    const float scalew = static_cast<float>(image.cols) / 480.f;
    const float scaleh = static_cast<float>(image.rows) / 640.f;
    const cv::Size2f scale_factor(scalew, scaleh);

    std::cout << "\n=== Config: engine=" << engine_path
              << ", num_thread=" << num_thread << ", iters=" << iters << " ===" << std::endl;
    std::cout << "=== Dynamic batch throughput sweep (images/sec) ===" << std::endl;

    const std::vector<int> batch_sizes = {1, 2, 4, 8};
    cv::Mat last_output;

    for (int B : batch_sizes)
    {
        // 预生成批量 blob (复用，预处理不计入)
        std::vector<cv::Mat> imgs(B, image);
        Blob blob;
        blob["images"] = YOLO::preprocessBatch(imgs);

        // 预热
        {
            std::vector<FutureBlob> warm;
            for (int i = 0; i < 50; ++i)
                warm.emplace_back(model->PostQueue(blob));
            for (auto &f : warm)
                f.get();
        }

        // 计时：iters 次批量推理，全部提交后再回收，喂满多流
        std::vector<FutureBlob> results;
        results.reserve(iters);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
            results.emplace_back(model->PostQueue(blob));
        Blob last;
        for (auto &f : results)
            last = f.get();
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double images = static_cast<double>(iters) * B;
        std::cout << "  batch=" << B
                  << "   " << ms / iters << " ms/infer"
                  << "   " << images * 1000.0 / ms << " images/sec"
                  << "   (" << iters * 1000.0 / ms << " infer/sec)" << std::endl;

        // 用最后一次结果的第 0 张做正确性校验
        last_output = YOLO::postprocessOne(last["output0"], 0, image, scale_factor);
    }

    cv::imwrite("./demo/yolo_batch_output.png", last_output);
    std::cout << "\nSaved batch output (image 0) to ./demo/yolo_batch_output.png" << std::endl;
    return 0;
}
