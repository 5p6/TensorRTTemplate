#include "TRTinfer.h"
#include <opencv2/opencv.hpp>
std::unordered_map<std::string, cv::Mat> preprocess(cv::Mat &left, cv::Mat &right)
{
    if (left.size() != cv::Size(512, 320))
    {
        cv::resize(left, left, cv::Size(512, 320));
    }
    if (right.size() != cv::Size(512, 320))
    {
        cv::resize(right, right, cv::Size(512, 320));
    }
    std::unordered_map<std::string, cv::Mat> input_blob;
    input_blob["left"] = cv::dnn::blobFromImage(left);
    input_blob["right"] = cv::dnn::blobFromImage(right);

    float *ptr = input_blob["left"].ptr<float>(0);
    return input_blob;
}
void postprocess(const cv::Mat &disp, cv::Mat &disp_vis)
{
    cv::Mat disp_c = disp.clone();
    // disp_vis.release();
    double min, max;
    cv::minMaxLoc(disp_c, &min, &max);
    cv::Mat disp_norm = ((disp_c - min) / (max - min)) * 255;
    disp_norm.convertTo(disp_vis, CV_8U);
    cv::applyColorMap(disp_vis, disp_vis, cv::COLORMAP_INFERNO);
}

int main(int argc, char *argv[])
{
    cv::Mat left = cv::imread("E:/code/python/CVRecon/IGEV-plusplus/demo-imgs/PipesH/im0.png");
    cv::Mat right = cv::imread("E:/code/python/CVRecon/IGEV-plusplus/demo-imgs/PipesH/im1.png");

    // // 预处理
    auto input_blob = preprocess(left, right);
    // // 模型
    TRTInfer model("E:/code/python/CVRecon/IGEV-plusplus/igev_320.engine");
    // // // 输出
    auto output_blob = model(input_blob);
    cv::Mat dst;
    postprocess(output_blob["disparity"].reshape(1, 320),dst);
    cv::imshow("disp",dst);
    cv::waitKey();
    // cv::Mat x = cv::Mat::ones(cv::Size(28,28),CV_32FC1);
    // TRTInfer model("E:/code/python/CVRecon/IGEV-plusplus/igev_320_fp16.engine");
    // auto output = model(x);
    return 1;
}