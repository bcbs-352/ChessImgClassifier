#include <opencv2/opencv.hpp>
#include <algorithm>

using namespace std;
using namespace cv;
const int H = 32;
const int W = 32;
vector<string> chess_list = {"黑士", "红仕", "黑象", "红相", "黑炮", "红炮", "黑将", "红帅", "黑马", "红马", "黑卒", "红兵", "黑車", "红車"};

void ModelTest()
{
    dnn::Net net = dnn::readNetFromONNX("../myModel_new.onnx");
    Mat testImg = imread("../Sample_Images/2.jpg");
    Mat dst;
    resize(testImg, dst, Size(H, W), INTER_LANCZOS4);
    imshow("norm", dst);
    // dst = (dst / 255. - 0.5) / 0.5; // 归一化,对应transform.Normalize()

    auto blob = dnn::blobFromImage(dst, 2. / 255, Size(H, W), Scalar(128, 128, 128), true, false);
    net.setInput(blob);
    auto output = net.forward();

    float maxVal = -1e37, index = 0;
    for (int i = 0; i < output.cols; ++i)
    {
        cout << output.at<float>(0, i) << endl;
        if (output.at<float>(0, i) > maxVal)
        {
            maxVal = output.at<float>(0, i);
            index = i;
        }
    }
    cout << "当前棋子：" << chess_list.at(index) << endl;
    set<int> s;

    return;
}

int main(int argc, char *argv[])
{
    ModelTest();

    waitKey();
    return 0;
}