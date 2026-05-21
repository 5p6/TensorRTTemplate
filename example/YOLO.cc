#include "TRTinfer.h"
#include "benchmark.h"
#include <opencv2/opencv.hpp>
#include <random>
#include <cstdlib>
using namespace TRT;
namespace YOLO
{
    struct Detection
    {
        int class_id{0};
        std::string className{};
        float confidence{0.0};
        cv::Scalar color{};
        cv::Rect box{};
    };
    std::unordered_map<std::string, cv::Mat> preprocess(const cv::Mat &img)
    {
        cv::Mat imgc = img.clone();
        if (imgc.size() != cv::Size(480, 640))
            cv::resize(imgc, imgc, cv::Size(480, 640));
        cv::Mat blob = cv::dnn::blobFromImage(imgc, 1 / 255.f, cv::Size(), cv::Scalar(), true, false);
        std::unordered_map<std::string, cv::Mat> input_blob;
        input_blob["images"] = blob;
        return input_blob;
    }
    cv::Mat postprocess(const cv::Mat &output_blob, const cv::Mat &img, const cv::Size2f &scale)
    {
        cv::Mat imgc = img.clone();
        // reshape
        cv::Mat output_blobc = output_blob.clone().reshape(1, 84);
        output_blobc.convertTo(output_blobc, CV_32F);
        cv::transpose(output_blobc, output_blobc);

        // data
        std::vector<cv::Rect> boxes;
        std::vector<float> scores_classs;
        std::vector<int> indices;

        // NMS
        float confidenceThreshold = 0.5;
        float nmsThreshold = 0.5;

        // convert data
        for (int i = 0; i < output_blobc.rows; i++)
        {
            float *classes_scores = (float *)output_blobc.row(i).data + 4;
            cv::Mat scores(cv::Size(80, 1), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;
            // maximum and the location
            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
            if (maxClassScore > confidenceThreshold)
            {
                scores_classs.push_back(maxClassScore);
                indices.push_back(class_id.x);
                float x = output_blobc.at<float>(i, 0);
                float y = output_blobc.at<float>(i, 1);
                float w = output_blobc.at<float>(i, 2);
                float h = output_blobc.at<float>(i, 3);
                int left = int((x - 0.5 * w) * scale.width);
                int top = int((y - 0.5 * h) * scale.height);

                int width = int(w * scale.width);
                int height = int(h * scale.height);

                boxes.push_back(cv::Rect(left, top, width, height));
            }
            // break;
        }
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, scores_classs, confidenceThreshold, nmsThreshold, nms_result);
        for (unsigned long i = 0; i < nms_result.size(); ++i)
        {
            int idx = nms_result[i];

            Detection result;
            result.class_id = indices[idx];
            result.confidence = scores_classs[idx];

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(100, 255);
            result.color = cv::Scalar(dis(gen),
                                      dis(gen),
                                      dis(gen));

            result.className = std::to_string(indices[idx]);
            result.box = boxes[idx];
            cv::rectangle(imgc, boxes[idx], result.color, 4);
            cv::putText(imgc, result.className, cv::Point(boxes[idx].x, boxes[idx].y), cv::FONT_HERSHEY_COMPLEX, 1.0, result.color);
        }
        return imgc;
    }

}

int main(int argc, char *argv[])
{
    // 参数 (可命令行覆盖): yolo [engine] [num_thread] [iters]
    std::string image_path = "./demo/bus.jpg";
    std::string engine_path = argc > 1 ? argv[1] : "./yolov8n.engine";
    int num_thread = argc > 2 ? std::atoi(argv[2]) : 4;
    int iters = argc > 3 ? std::atoi(argv[3]) : 2000;

    using Blob = std::unordered_map<std::string, cv::Mat>;
    using FutureBlob = std::future<Blob>;

    // 加载模型 (使用工厂方法，延迟初始化)
    auto model = TRT::TRTInfer::create(engine_path, num_thread);

    // 加载图像
    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image from " << image_path << std::endl;
        return -1;
    }

    // for rescale factor
    float scalew = static_cast<float>(image.size().width) / 480.f;
    float scaleh = static_cast<float>(image.size().height) / 640.f;
    cv::Size2f scale_factor(scalew, scaleh);

    // 预生成一组 blob 复用 (避免一次性占用过多主机内存)
    const int pool_size = 64;
    std::vector<Blob> blob_pool;
    for (int i = 0; i < pool_size; i++)
        blob_pool.emplace_back(YOLO::preprocess(image));

    // 预热
    std::cout << "\n=== Warmup ===" << std::endl;
    {
        std::vector<FutureBlob> warm;
        for (int i = 0; i < 300; i++)
            warm.emplace_back(model->PostQueue(blob_pool[i % pool_size]));
        for (auto &f : warm)
            f.get();
    }

    std::cout << "\n=== Config: engine=" << engine_path
              << ", num_thread=" << num_thread << ", iters=" << iters << " ===" << std::endl;

    cv::Mat output;

    // (1) 纯推理 QPS —— blob 复用, 预处理不计入
    {
        std::vector<FutureBlob> results;
        results.reserve(iters);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++)
            results.emplace_back(model->PostQueue(blob_pool[i % pool_size]));
        for (auto &result : results)
            output = result.get()["output0"];
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[inference-only]            "
                  << ms / iters << " ms/infer, " << iters * 1000.0 / ms << " QPS" << std::endl;
    }

    // (2) 端到端 QPS —— 每次都做 CPU 预处理后提交 (更贴近真实部署)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<FutureBlob> results;
        results.reserve(iters);
        for (int i = 0; i < iters; i++)
        {
            Blob blob = YOLO::preprocess(image);
            results.emplace_back(model->PostQueue(blob));
        }
        for (auto &result : results)
            result.get();
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[end-to-end +preprocess]    "
                  << ms / iters << " ms/infer, " << iters * 1000.0 / ms << " QPS" << std::endl;
    }

    // post process (使用最后一次输出)
    cv::Mat result = YOLO::postprocess(output, image, scale_factor);

    // 保存结果
    cv::imwrite("./demo/yolo_output.png", result);
    std::cout << "Saved output to ./demo/yolo_output.png" << std::endl;

    // 仅在显式要求时弹窗 (YOLO_SHOW=1), 避免基准测试时 waitKey 阻塞
    if (std::getenv("YOLO_SHOW"))
    {
        cv::imshow("output", result);
        cv::waitKey();
    }

    return 0;
}
