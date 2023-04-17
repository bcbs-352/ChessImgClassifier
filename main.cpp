#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdlib.h>
#include <string>

using namespace cv;
using namespace std;

static string folderPath = "C:\\Users\\asus\\Desktop\\ChessDataset\\train\\R_r\\*.*";

Mat PepperNoise(const Mat &inImg, float ratio)
{
    Mat img = inImg.clone();
    clamp<float>(ratio, 0.001, 0.1);
    for (int i = 0; i < img.rows * img.cols * ratio; ++i)
    {
        int x = std::rand() % img.cols;
        int y = std::rand() % img.rows;
        if (img.type() == CV_8UC1)
        {
            img.at<uchar>(y, x) = rand() % 2 == 0 ? 0 : 255;
        }
        else if (img.type() == CV_8UC3)
        {
            int val = rand() % 2 == 0 ? 0 : 255;
            img.at<Vec3b>(y, x)[0] = val;
            img.at<Vec3b>(y, x)[1] = val;
            img.at<Vec3b>(y, x)[2] = val;
        }
    }
    // imshow("noise", img);
    return img;
}

Mat Rotate(const Mat &inImg, double angle)
{
    Mat img = inImg.clone();
    int minLen = min(img.cols, img.rows);
    Point2f center(img.cols / 2., img.rows / 2.);
    Mat rotateMat = getRotationMatrix2D(center, angle, 1.);
    warpAffine(img, img, rotateMat, inImg.size());
    resize(img, img, Size(minLen, minLen));

    // imshow("rotate", img);
    return img;
}

// 直方图均衡化，效果不好
Mat Normalize(const Mat &inImg)
{
    Mat img;
    normalize(inImg, img, 255, 0, NORM_MINMAX, CV_8UC3);
    vector<Mat> channels;
    split(inImg, channels);

    equalizeHist(channels[0], channels[0]);
    equalizeHist(channels[1], channels[1]);
    // equalizeHist(channels[2], channels[2]);
    merge(channels, img);

    // imshow("normalize", img);
    return img;
}

Mat ModifyBright(const Mat &inImg, float ratio)
{
    Mat img = inImg.clone();
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            Vec3b color = img.at<Vec3b>(i, j);
            for (int k = 0; k < 3; ++k)
            {
                img.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(ratio * color[k]);
            }
        }
    }
    // imshow("bright", img);
    return img;
}

Mat MedianBlur(const Mat &inImg, int ksize)
{
    Mat img;
    vector<Mat> channels;
    split(inImg, channels);
    medianBlur(channels[0], channels[0], ksize);
    medianBlur(channels[0], channels[0], ksize);
    medianBlur(channels[0], channels[0], ksize);

    merge(channels, img);
    // imshow("medBlur", img);
    return img;
}

Mat AmplifyImage(const Mat &inImg)
{
    Mat img = inImg.clone();
    Size originSize = inImg.size();
    resize(img, img, Size(100, 100));
    Point2f upleft(rand() % 20, rand() % 20);
    img = img(Rect(upleft, upleft + Point2f(80, 80)));
    resize(img, img, originSize, 0, 0, INTER_CUBIC);
    return img;
}

Mat LessenImage(const Mat &inImg, float ratio)
{
    Mat img = inImg.clone();
    Size originSize = inImg.size();
    ratio -= (int)ratio;
    int horizon = originSize.width * ratio / 2, vertical = originSize.height * ratio / 2;
    Scalar scalar = inImg.type() == CV_8UC3 ? Scalar(0, 0, 0) : Scalar(0);
    copyMakeBorder(img, img, vertical, vertical, horizon, horizon, BORDER_REPLICATE, scalar);

    resize(img, img, originSize);
    // imshow("lessen", img);
    return img;
}

void SaveImage(Mat img, string fileName, int height = 32, int width = 32)
{
    if (img.rows != height || img.cols != width)
        resize(img, img, Size(height, width), 0, 0, INTER_CUBIC);
    imwrite(fileName, img);
    return;
}

int main(int argc, char *argv[])
{
    vector<string> fileName;
    glob(folderPath, fileName, false);
    for (size_t i = 0; i < fileName.size(); ++i)
    {
        Mat img = imread(fileName[i]);
        fileName[i] = fileName[i].substr(0, fileName[i].rfind("."));
        // imshow("origin img", img);

        SaveImage(img, fileName[i] + "_copy1.jpg");
        SaveImage(img, fileName[i] + "_copy2.jpg");

        SaveImage(Rotate(img, rand() % 360), fileName[i] + "_rotate1.jpg");
        SaveImage(Rotate(img, rand() % 360), fileName[i] + "_rotate2.jpg");
        SaveImage(Rotate(img, rand() % 360), fileName[i] + "_rotate3.jpg");

        Mat tmp = Rotate(img, rand() % 360);
        Mat tmp1 = Rotate(img, rand() % 360);
        Mat tmp2 = Rotate(img, rand() % 360);
        // GaussianBlur(img, gausBlur_img, Size(5, 5), 0.8, 0.8);

        SaveImage(PepperNoise(img, 0.03), fileName[i] + "_noise1.jpg");
        SaveImage(PepperNoise(img, 0.01), fileName[i] + "_noise2.jpg");
        SaveImage(PepperNoise(img, 0.005), fileName[i] + "_noise3.jpg");
        SaveImage(PepperNoise(tmp, 0.03), fileName[i] + "_noise4.jpg");
        SaveImage(MedianBlur(img, 5), fileName[i] + "_medianBlur1.jpg");
        SaveImage(MedianBlur(tmp1, 5), fileName[i] + "_medianBlur2.jpg");
        SaveImage(MedianBlur(tmp2, 5), fileName[i] + "_medianBlur3.jpg");

        SaveImage(ModifyBright(img, 0.7), fileName[i] + "_dark0.jpg");
        SaveImage(ModifyBright(img, 0.85), fileName[i] + "_dark0.jpg");
        SaveImage(ModifyBright(img, 1.2), fileName[i] + "_bright1.jpg");
        SaveImage(ModifyBright(img, 1.3), fileName[i] + "_bright1.jpg");
        SaveImage(ModifyBright(tmp1, 0.6), fileName[i] + "_dark2.jpg");
        SaveImage(ModifyBright(tmp2, 1.3), fileName[i] + "_bright2.jpg");

        SaveImage(AmplifyImage(img), fileName[i] + "_amp1.jpg");
        SaveImage(AmplifyImage(img), fileName[i] + "_amp2.jpg");
        SaveImage(AmplifyImage(tmp1), fileName[i] + "_amp3.jpg");
        SaveImage(LessenImage(img, 0.2), fileName[i] + "_less1.jpg");
        SaveImage(LessenImage(img, 0.1), fileName[i] + "_less2.jpg");
        SaveImage(LessenImage(tmp, 0.2), fileName[i] + "_less2.jpg");
        SaveImage(LessenImage(tmp2, 0.2), fileName[i] + "_less3.jpg");
    }
    waitKey();
    return 0;
}