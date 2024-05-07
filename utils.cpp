#include <utils.h>

cv::dnn::Net Yolov5Detector::loadNet(const string modelPath, bool is_cuda = false)
{
    cv::dnn::dnn4_v20211004::Net model = cv::dnn::readNet(modelPath);
    if (is_cuda)
    {
        cout << "Attempty to use CUDA\n";
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        cout << "Running on CPU\n";
        model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    return model;
}

vector<string> Yolov5Detector::loadClassesList(const string filePath)
{
    vector<string> class_list;
    ifstream ifs(filePath);
    string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}

cv::Mat Yolov5Detector::format2Yolov5(const cv::Mat source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

Yolov5Detector::Yolov5Detector(const string modelPath, const string classesFilePath, const bool useGPU)
{
    this->net = this->loadNet(modelPath, useGPU);
    this->classNames = this->loadClassesList(classesFilePath);
}

Yolov5Detector::DetectionData Yolov5Detector::detect(cv::Mat image, const bool draw)
{
    cv::Mat blob;

    cv::Mat input_image = this->format2Yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true,
                           false);

    this->net.setInput(blob);
    vector<cv::Mat> outputs;
    this->net.forward(outputs, this->net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 25200;

    vector<int> class_ids;
    vector<float> confidences;
    vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float *classes_scores = data + 5;
            cv::Mat scores(1, this->classNames.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD)
            {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                // Width, height and x,y coordinates of bounding box

                int x1 = int((x - 0.5 * w) * x_factor);
                int y1 = int((y - 0.5 * h) * y_factor);
                int x2 = int(w * x_factor);
                int y2 = int(h * y_factor);
                boxes.push_back(cv::Rect(x1, y1, x2, y2));
            }
        }
        data += 85;
    }

    vector<Detection> output;
    vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }

    DetectionData ret;
    ret.detections = output;
    ret.image = this->draw(image, output);

    return ret;
}

cv::Mat Yolov5Detector::draw(cv::Mat image, const vector<Detection> detections)
{
    for (const Detection &detection : detections)
    {
        cv::Rect box = detection.box;
        int classId = detection.class_id;
        const auto &color = this->colors[classId % this->colors.size()];
        cv::rectangle(image, box, color, 3);
        cv::rectangle(image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
        cv::putText(image, this->classNames[classId], cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));
    }

    return image;
}
