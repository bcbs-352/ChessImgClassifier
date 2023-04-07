#include "opencv2/dnn/dnn.hpp"
using namespace cv;
using namespace cv::dnn;
using namespace std;
void Classification_good()
{ // 装载模型，设置参数
    clock_t st = clock();
    string model = "C:\\Users\\Ring\\Desktop\\A_jupyter\\pytorch\\test\\onnx_model_name.onnx";
    ClassificationModel dnn_model(model);
    dnn_model.setPreferableBackend(DNN_BACKEND_CUDA);
    dnn_model.setPreferableTarget(DNN_TARGET_CUDA);
    float scale = 1.0 / 255;
    int inpWidth = 224, inpHeight = 224;
    Scalar mean(0, 0, 0);
    dnn_model.setInputParams(scale, Size(inpWidth, inpHeight), mean, true, false);
    clock_t end = clock();
    cout << end - st << endl; // 图像文件夹遍历检测
    String folder = "C:\\Users\\Ring\\Desktop\\A_jupyter\\pytorch\\test\\Neu\\val\\Rs/";
    vector<String> imagePathList;
    glob(folder, imagePathList);
    cout << "test In C++!" << endl;
    for (int i = 0; i < imagePathList.size(); i++)
    {
        Mat img = imread(imagePathList[i]);
        resize(img, img, Size(224, 224), 0, 0, INTER_LANCZOS4);
        Mat img_t = Mat::zeros(img.size(), CV_32FC1);
        for (int ii = 0; ii < img.cols; ii++)
        {
            for (int jj = 0; jj < img.rows; jj++)
            {
                img_t.at<float>(ii, jj) = img.at<uchar>(ii, jj);
            }
        }
        int classIds;
        float confs;
        double time1 = static_cast<double>(getTickCount());
        dnn_model.classify(img, classIds, confs); // 前向推理，classIds是类别索引,classIds=0是划痕，classIds=1是颗粒
        double time2 = (static_cast<double>(getTickCount()) - time1) / getTickFrequency();
        cout << classIds << endl;
        cout << "time: " << time2 << endl;
    }
}