#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>

using namespace std;

struct Detection {
  int class_id;
  float confidence;
  cv::Rect box;
};

// Function declarations
vector<string> load_class_list(const string filePath);
cv::dnn::Net load_net(const string modelPath, bool is_cuda);
cv::Mat format_yolov5(const cv::Mat& source);
void detect(cv::Mat& image, cv::dnn::Net& net, vector<Detection>& output, const vector<string>& className);

// Constants (if applicable)
extern const vector<cv::Scalar> colors;
extern const float INPUT_WIDTH;
extern const float INPUT_HEIGHT;
extern const float SCORE_THRESHOLD;
extern const float NMS_THRESHOLD;
extern const float CONFIDENCE_THRESHOLD;

#endif
