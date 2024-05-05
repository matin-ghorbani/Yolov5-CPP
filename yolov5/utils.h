#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace std;

class Yolov5Detector
{
private:
    // Function to load the YOLOv5 model
    cv::dnn::Net loadNet(const string modelPath, bool is_cuda);

    // Function to load class names from a text file
    vector<string> loadClassesList(const string filePath);

    // Function to format an image for YOLOv5 input
    cv::Mat format2Yolov5(const cv::Mat source);

    // Internal variables
    cv::dnn::Net net;

    // Internal constants
    static constexpr float INPUT_WIDTH = 640.0f;
    static constexpr float INPUT_HEIGHT = 640.0f;
    static constexpr float SCORE_THRESHOLD = 0.2f;
    static constexpr float NMS_THRESHOLD = 0.4f;
    static constexpr float CONFIDENCE_THRESHOLD = 0.4f;

public:
    // Constructor
    Yolov5Detector(const string modelPath, const string classesFilePath, const bool useGPU);

    // Struct to store detection result
    struct Detection
    {
        int class_id;
        float confidence;
        cv::Rect box;
    };

    // Struct to bundle image, bounding boxes, and detections
    struct DetectionData
    {
        cv::Mat image;
        cv::Rect rect;
        vector<Detection> detections;
    };

    vector<string> classNames;

    // Function to perform object detection on an image
    DetectionData detect(cv::Mat image, const bool draw = true);

    cv::Mat draw(cv::Mat image, const vector<Detection> detections);
    const vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255),
                                       cv::Scalar(255, 0, 0)};
};

#endif
