#include "opencv2/opencv.hpp"

//参数：源图像，输出图像，低阈值，高阈值
void ImageStretch(cv::Mat &originalMat,cv::Mat &processedMat,int lowThre, int highThre)
{
    // y=kx+b,重新映射灰度
    double k = 255.0 / (highThre - lowThre);
    double b = -k * lowThre;
    using namespace cv;

    Mat hsvMat;    //设置中间变量hsvMat是为了解决多线程时的安全问题

    cvtColor(originalMat,hsvMat,COLOR_BGR2HSV);   //从RGB转换到HSV，只对亮度V做变换

//    LUT(input, lookupTable, output);          // TODO利用LUT提高速度
    
    //亮度V重映射
    double y;
    for (int i = 0; i < hsvMat.rows; i++)
    {
        for (int j = 0; j < hsvMat.cols; j++)
        {

            y = hsvMat.at<Vec3b>(i, j)[2] * k + b;  //重映射公式
            if (y > 255.0)   //重映射后高于255的设为255
            {
                hsvMat.at<Vec3b>(i, j)[1] = 0;
                hsvMat.at<Vec3b>(i, j)[2] = 255;
            }
            else if (y < 0.0)  //重映射后低于0的设为0
            {
                 hsvMat.at<Vec3b>(i, j)[2] = 0;
            }
            else
            {
                hsvMat.at<Vec3b>(i, j)[2] = uchar(y);
            }

        }
    }
    cvtColor(hsvMat, processedMat, COLOR_HSV2BGR);   //再从HSV转换回RGB
}

int main()
{
	using namespace cv;
	Mat originalMat = imread("C:/Users/wulala/Desktop/087.jpg");
	Mat processedMat;
	ImageStretch(originalMat,processedMat,30,200);
	imshow("image",processedMat);
	waitKey(0);
	return 0;
}