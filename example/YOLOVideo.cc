#include "TRTinfer.h"
#include "benchmark.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <deque>
#include <algorithm>

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
        cv::Mat output_blobc = output_blob.clone().reshape(1, 84);
        output_blobc.convertTo(output_blobc, CV_32F);
        cv::transpose(output_blobc, output_blobc);

        std::vector<cv::Rect> boxes;
        std::vector<float> scores_classs;
        std::vector<int> indices;

        float confidenceThreshold = 0.5;
        float nmsThreshold = 0.5;

        for (int i = 0; i < output_blobc.rows; i++)
        {
            float *classes_scores = (float *)output_blobc.row(i).data + 4;
            cv::Mat scores(cv::Size(80, 1), CV_32FC1, classes_scores);
            cv::Point class_id;
            double maxClassScore;
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
        }

        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, scores_classs, confidenceThreshold, nmsThreshold, nms_result);
        for (unsigned long i = 0; i < nms_result.size(); ++i)
        {
            int idx = nms_result[i];
            Detection result;
            result.class_id = indices[idx];
            result.confidence = scores_classs[idx];
            result.className = std::to_string(indices[idx]);
            result.box = boxes[idx];
            cv::rectangle(imgc, boxes[idx], cv::Scalar(0, 255, 0), 2);
            cv::putText(imgc, result.className, cv::Point(boxes[idx].x, boxes[idx].y),
                        cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0));
        }
        return imgc;
    }
}

int main(int argc, char *argv[])
{
    std::string video_path = argc > 1 ? argv[1] : "./demo/bus_40s_60fps.mp4";
    std::string engine_path = argc > 2 ? argv[2] : "./yolov8n.engine";
    int num_thread = argc > 3 ? std::atoi(argv[3]) : 4;
    bool save_output = argc > 4 && std::string(argv[4]) == "save";

    using Blob = std::unordered_map<std::string, cv::Mat>;
    using FutureBlob = std::future<Blob>;

    auto model = TRT::TRTInfer::create(engine_path, num_thread);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video: " << video_path << std::endl;
        return -1;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "\n=== Video: " << video_path
              << ", " << width << "x" << height << "@" << fps << "fps, "
              << total_frames << " frames ===" << std::endl;
    std::cout << "=== Engine: " << engine_path
              << ", num_thread=" << num_thread << " ===" << std::endl;

    cv::VideoWriter writer;
    if (save_output)
    {
        std::string out_path = video_path.substr(0, video_path.rfind('.')) + "_output.mp4";
        writer.open(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                    fps, cv::Size(width, height));
        std::cout << "=== Saving output to: " << out_path << " ===" << std::endl;
    }

    // Warmup
    std::cout << "\n=== Warmup (300 frames) ===" << std::endl;
    cv::Mat warmup_frame;
    std::vector<FutureBlob> warmup_futures;
    for (int i = 0; i < 300; ++i)
    {
        cap >> warmup_frame;
        if (warmup_frame.empty())
        {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            cap >> warmup_frame;
        }
        auto blob = YOLO::preprocess(warmup_frame);
        warmup_futures.emplace_back(model->PostQueue(blob));
    }
    for (auto &f : warmup_futures)
        f.get();

    // Seek back to start
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);

    // 流水线深度：保持 num_thread 帧在途，喂满所有流，真正发挥多流并发
    const int pipeline_depth = std::max(1, num_thread);

    struct InFlight
    {
        cv::Mat frame;      // 原图，后处理时画框用
        FutureBlob fut;     // 推理结果 future
        cv::Size2f scale;   // 还原坐标比例
    };
    std::deque<InFlight> inflight;

    double sum_capture = 0, sum_pre = 0, sum_post = 0;
    double min_capture = 1e9, min_pre = 1e9, min_post = 1e9;
    double max_capture = 0, max_pre = 0, max_post = 0;
    int done = 0;
    cv::Mat output;

    auto record = [](double v, double &sum, double &mn, double &mx)
    {
        sum += v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    };

    // 回收一个在途结果：等待 -> 后处理 -> 写出
    auto collect = [&](InFlight &item)
    {
        Blob result = item.fut.get();
        auto tp0 = std::chrono::high_resolution_clock::now();
        output = YOLO::postprocess(result["output0"], item.frame, item.scale);
        auto tp1 = std::chrono::high_resolution_clock::now();
        record(std::chrono::duration<double, std::milli>(tp1 - tp0).count(), sum_post, min_post, max_post);
        if (save_output)
            writer << output;
        if (std::getenv("YOLO_SHOW"))
        {
            cv::imshow("output", output);
            cv::waitKey(1);
        }
        ++done;
    };

    auto wall_start = std::chrono::high_resolution_clock::now();
    std::cout << "\n=== Processing (pipeline depth=" << pipeline_depth << ") ===" << std::endl;

    cv::Mat frame;
    int submitted = 0;
    for (int fi = 0; fi < total_frames; ++fi)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        cap >> frame;
        auto t1 = std::chrono::high_resolution_clock::now();
        if (frame.empty())
            break;
        record(std::chrono::duration<double, std::milli>(t1 - t0).count(), sum_capture, min_capture, max_capture);

        auto t2 = std::chrono::high_resolution_clock::now();
        auto blob = YOLO::preprocess(frame);
        auto t3 = std::chrono::high_resolution_clock::now();
        record(std::chrono::duration<double, std::milli>(t3 - t2).count(), sum_pre, min_pre, max_pre);

        float scalew = static_cast<float>(frame.size().width) / 480.f;
        float scaleh = static_cast<float>(frame.size().height) / 640.f;
        inflight.push_back(InFlight{frame.clone(), model->PostQueue(blob), cv::Size2f(scalew, scaleh)});
        ++submitted;

        // 在途窗口满了，回收最旧的一帧，保持 pipeline_depth 帧并行
        if (static_cast<int>(inflight.size()) >= pipeline_depth)
        {
            collect(inflight.front());
            inflight.pop_front();
        }

        if (submitted % 100 == 0)
        {
            double wall = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wall_start).count();
            std::cout << "  frame " << submitted << "/" << total_frames
                      << "  throughput=" << (wall > 0 ? done / wall : 0.0) << " fps"
                      << "  capture=" << sum_capture / submitted << "ms"
                      << "  preprocess=" << sum_pre / submitted << "ms"
                      << "\n";
        }
    }

    // 排空剩余在途帧
    while (!inflight.empty())
    {
        collect(inflight.front());
        inflight.pop_front();
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_sec = std::chrono::duration<double>(wall_end - wall_start).count();

    int n = done;
    auto avg = [&](double sum) { return n ? sum / n : 0.0; };

    std::cout << "\n=== Summary (" << n << " frames, " << wall_sec << "s wall) ===" << std::endl;
    std::cout << std::fixed;
    std::cout << "  capture     avg=" << avg(sum_capture) << "ms  min=" << min_capture << "ms  max=" << max_capture << "ms" << std::endl;
    std::cout << "  preprocess  avg=" << avg(sum_pre) << "ms  min=" << min_pre << "ms  max=" << max_pre << "ms" << std::endl;
    std::cout << "  postprocess avg=" << avg(sum_post) << "ms  min=" << min_post << "ms  max=" << max_post << "ms" << std::endl;
    std::cout << "  throughput  " << (wall_sec > 0 ? n / wall_sec : 0.0) << " fps  (" << pipeline_depth << " streams in flight)" << std::endl;

    cap.release();
    if (save_output)
        writer.release();

    return 0;
}