#include <fstream>
#include <opencv2/opencv.hpp>
#include "utils.h"

using namespace std;

void predictOnVideo(const auto webcamId, const bool isCuda = false)
{
    Yolov5Detector detector("../assets/models/yolov5s.onnx", "../assets/classes.txt", isCuda);
    Yolov5Detector::DetectionData output;

    cv::VideoCapture cap(webcamId);
    cv::Mat frame, resultImg;

    cap.set(3, 640);
    cap.set(4, 480);

    while (true)
    {
        cap.read(frame);
        output = detector.detect(frame, true);
        resultImg = output.image;
        cv::imshow("Webcam " + webcamId, resultImg);

        if ((cv::waitKey(1) & 0xFF) == 27)
        {
            break;
        }
    }
}

void predictOnImage(string imagePath, bool isCuda = false)
{
    Yolov5Detector detector("../assets/models/yolov5s.onnx", "../assets/classes.txt", isCuda);

    cv::Mat image = cv::imread(imagePath);
    if (image.empty())
    {
        cerr << "Error loading image\n";
    }

    Yolov5Detector::DetectionData output;
    output = detector.detect(image, true);

    cv::Mat resultImg = output.image;
    vector<Yolov5Detector::Detection> detections = output.detections;

    cv::imshow("result image", resultImg);
    cv::waitKey(0);
}

int findType(const string &path)
{
    vector<string> image_extensions = {".jpg", ".jpeg", ".png"};
    for (const string &ext : image_extensions)
    {
        string lower_path = path;
        transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);
        if (lower_path.size() >= ext.size() && lower_path.substr(lower_path.size() - ext.size()) == ext)
        {
            return 0;
        }
    }

    vector<string> video_extensions = {".mp4", ".avi"};
    for (const string &ext : video_extensions)
    {
        if (path.size() >= ext.size() && path.substr(path.size() - ext.size()) == ext)
        {
            return 1;
        }
    }

    try
    {
        stod(path);
        return 2;
    }
    catch (const invalid_argument &)
    {
        // Not a number, continue
    }
    catch (const out_of_range &)
    {
        // Not a number, continue
    }

    return -1;
}

int main(int argc, const char *argv[])
{
    bool isCuda = argc > 1 && strcmp(argv[1], "cuda") == 0;

    string input = argv[1];
    int type = findType(input);
    if (type == -1)
    {
        cout << "Please specify a valid type" << endl;
        return -1;
    }
    else if (type == 0)
    {
        predictOnImage(input, isCuda);
    }
    else if (type == 1)
    {
        predictOnVideo(input, isCuda);
    }
    else if (type == 2)
    {
        predictOnVideo(stoi(input), isCuda);
    }
}