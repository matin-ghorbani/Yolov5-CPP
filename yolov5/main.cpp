#include <fstream>
#include <opencv2/opencv.hpp>
#include "utils.h"

using namespace std;


int main(int argc, const char *argv[]) {
    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    Yolov5Detector detector("../assets/models/yolov5s.onnx", "../assets/classes.txt", is_cuda);

    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Error loading image\n";
        return -1;
    }

    Yolov5Detector::DetectionData output;
    output = detector.detect(image, true);

    cv::Mat resultImg = output.image;
    vector<Yolov5Detector::Detection> detections = output.detections;
    


    // As giving fixed value for width, and protecting original image ratio, output resolution declares in here.
    int targetWidth = 800;
    int targetHeight = static_cast<int>(image.rows * static_cast<float>(targetWidth) / image.cols);

    cv::imshow("result image", resultImg);
    cv::waitKey(0);

    return 0;
}