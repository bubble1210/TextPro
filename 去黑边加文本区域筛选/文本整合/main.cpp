#pragma comment(linker, "/STACK:10240000000,10240000000")  
#include "core/core.hpp"  
#include "highgui/highgui.hpp"  
#include "imgproc/imgproc.hpp"  
#include "iostream"
#include <string>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>
#include <vector>
#include <dlib/svm_threaded.h>
#include "stdafx.h"
#include<io.h>
#include<stack>

using namespace std;
using namespace cv;

typedef dlib::matrix<float, 13 * 2 * 2 + 2, 1> sample_type;
typedef dlib::one_vs_one_trainer<dlib::any_trainer<sample_type> > ovo_trainer;
typedef dlib::polynomial_kernel<sample_type> poly_kernel;
typedef dlib::radial_basis_kernel<sample_type> rbf_kernel;

int vec[4];
Mat imageSource, imagegray, image, imagebinary, imagefull, ans;
int ss = 0, redsum, redfac,maxm;
stack<int> srow, scol, sk;

bool roi_svm(Mat &ROI, dlib::one_vs_one_decision_function<ovo_trainer, dlib::decision_function<rbf_kernel>>  df3);
void Connected_push(Mat &image, Mat &labels, int row, int col, int fac, int max)
{
	if (row >= image.rows) return;
	if (col >= image.cols) return;
	if (row < 0 || col < 0) return;
	if (fac > max) return;
	if (labels.at<uchar>(row, col) != 0  && labels.at<uchar>(row, col) <= fac) return;
	//cout << row << " " << col << " " << fac<< endl;
	labels.at<uchar>(row, col) = fac;
	srow.push(row);
	scol.push(col);
	sk.push(fac);
}
//bool Connected_serch(Mat &image, Mat &labels, int row, int col, int fac, int max)
//{
//	
//	bool ans = false;
//	//imshow("image",image);
//	//waitKey(0);
//	//cout << row << " " << col << " " << fac << endl;
//	if (row >= image.rows) return false;
//	if (row >= image.rows) return false;
//	if (row < 0 || col < 0) return false;
//	if (fac > max) return false;
//	if (row < image.rows * 14.0 / 15 && row > image.rows * 1.0 / 15 && col < image.cols * 14.0 / 15 && col > image.cols * 1.0 / 15) ans = true;
//	if (labels.at<uchar>(row, col) == 1 && labels.at<uchar>(row, col) <= fac) return false;
//	labels.at<uchar>(row, col) = fac;
//	if (image.at<uchar>(row, col) == 0)
//	{
//		if (row < vec[0]) vec[0] = row;
//		if (row > vec[1]) vec[1] = row;
//		if (col < vec[2]) vec[2] = col;
//		if (col > vec[3]) vec[3] = col;
//		redfac += imageSource.at<Vec3b>(row, col)[2] + imageSource.at<Vec3b>(row, col)[1] + imageSource.at<Vec3b>(row, col)[0];
//		redsum += imageSource.at<Vec3b>(row, col)[2];
//		ans = Connected_serch(image, labels, row + 1, col, 1, max) || ans;
//		ans = Connected_serch(image, labels, row, col + 1, 1, max) || ans;
//		ans = Connected_serch(image, labels, row, col - 1, 1, max) || ans;
//		ans = Connected_serch(image, labels, row - 1, col, 1, max) || ans;
//	}
//	else
//	{
//		ans = Connected_serch(image, labels, row + 1, col, fac + 1, max) || ans;
//		ans = Connected_serch(image, labels, row, col + 1, fac + 1, max) || ans;
//		ans = Connected_serch(image, labels, row - 1, col, fac + 1, max) || ans;
//		ans = Connected_serch(image, labels, row, col - 1, fac + 1, max) || ans;
//	}
//	return ans;
//}
bool Connected_serch(Mat &image, Mat &labels)
{
	
	bool ans = false;
	while (!sk.empty())
	{
		int row, Col,max,fac;
		row = srow.top();
		Col = scol.top();
		fac = sk.top();
		sk.pop();
		srow.pop();
		scol.pop();
		max = maxm;
		if (row < image.rows * 14.0 / 15 && row > image.rows * 1.0 / 15 && Col < image.cols * 14.0 / 15 && Col > image.cols * 1.0 / 15) ans = true;
		if (image.at<uchar>(row, Col) == 0)
		{
			if (row < vec[0]) vec[0] = row;
			if (row > vec[1]) vec[1] = row;
			if (Col < vec[2]) vec[2] = Col;
			if (Col > vec[3]) vec[3] = Col;
			redfac += imageSource.at<Vec3b>(row, Col)[2] + imageSource.at<Vec3b>(row, Col)[1] + imageSource.at<Vec3b>(row, Col)[0];
			redsum += imageSource.at<Vec3b>(row, Col)[2];
			Connected_push(image, labels, row + 1, Col, 1, maxm);
			Connected_push(image, labels, row, Col + 1, 1, maxm);
			Connected_push(image, labels, row, Col - 1, 1, maxm);
			Connected_push(image, labels, row - 1, Col, 1, maxm);
		}
		else
		{
			Connected_push(image, labels, row + 1, Col, fac + 1, maxm);
			Connected_push(image, labels, row, Col + 1, fac + 1, maxm);
			Connected_push(image, labels, row - 1, Col, fac + 1, maxm);
			Connected_push(image, labels, row, Col - 1, fac + 1, maxm);
		}
		
	}
	return ans;
}
Rect Connected_Component(Mat &image, int max, int k, Rect rct)
{
	maxm = max;
	dlib::one_vs_one_decision_function<ovo_trainer,
		dlib::decision_function<rbf_kernel>    // This is the output of the rbf_trainer
	> df3;

	// load the function back in from disk and store it in df3.  
	dlib::deserialize("df.dat") >> df3;
	Rect ans = rct;
	cvtColor(image, imagegray, COLOR_BGR2GRAY);
	threshold(imagegray, imagebinary, 210, 255, CV_THRESH_BINARY);
	imshow("gray", imagebinary);
	Mat labels = Mat::zeros(imagebinary.size(), CV_8U);
	for (int row = 0; row < imagebinary.rows; row++)
		for (int col = 0; col < imagebinary.cols; col++)
		{
			if (!rct.contains(Point(col, row)) && imagebinary.at<uchar>(row, col) == 0 && labels.at<uchar>(row, col) == 0)
			{
				redsum = redfac = 0;
				vec[0] = vec[1] = row;
				vec[2] = vec[3] = col;
				while (!srow.empty()) srow.pop();
				while (!scol.empty()) scol.pop();
				while (!sk.empty()) sk.pop();
				srow.push(row);
				scol.push(col);
				sk.push(0);
				bool flag = Connected_serch(imagebinary,labels);

				Rect rect(vec[2], vec[0], vec[3] - vec[2] + 1, vec[1] - vec[0] + 1);
				if ((vec[3] - vec[2] + 1) * (vec[1] - vec[0] + 1) >= 0.95*imagebinary.rows*imagebinary.cols)
				{
					for (int row2 = 0; row2 < imagebinary.rows; row2++)
						for (int col2 = 0; col2 < imagebinary.cols; col2++)
							if (labels.at<uchar>(row2, col2) != 0)
							{
								image.at<Vec3b>(row2, col2)[0] = 0;
								image.at<Vec3b>(row2, col2)[1] = 0;
								image.at<Vec3b>(row2, col2)[2] = 255;
							}
					continue;
				}
				if ((1.0*(vec[3] - vec[2] + 1) / (vec[1] - vec[0] + 1)>10 || 1.0*(vec[3] - vec[2] + 1) / (vec[1] - vec[0] + 1) < 0.1) && (vec[0]<1.0 / 60 * imagebinary.rows || vec[2]<1.0 / 60 * imagebinary.cols || vec[1]>59.0 / 60 * imagebinary.rows || vec[3]>59.0 / 60 * imagebinary.cols))
				{
					Point root_points[1][4];
					root_points[0][0] = Point(vec[2], vec[0]);
					root_points[0][1] = Point(vec[3], vec[0]);
					root_points[0][2] = Point(vec[3], vec[1]);
					root_points[0][3] = Point(vec[2], vec[1]);
					const Point* ppt[1] = { root_points[0] };
					int npt[] = { 4 };
					fillPoly(image, ppt, npt, 1, Scalar(0, 0, 255));
					continue;
				}
				if (1.0*redsum / redfac>0.35)
				{
					ans = ans | rect;
					continue;
				}
				if ((vec[3] - vec[2] + 1) * (vec[1] - vec[0] + 1) > 150000)
				{
					ans = ans | rect;
					continue;
				}
				if ((vec[3] - vec[2] + 1) * (vec[1] - vec[0] + 1) < 200 || ((1.0*(vec[3] - vec[2] + 1) / (vec[1] - vec[0] + 1)>10 || 1.0*(vec[3] - vec[2] + 1) / (vec[1] - vec[0] + 1) < 0.1))) continue;
				if (flag) rectangle(image, rect, Scalar(0, 255, 0), 2, 8, 0);
				if (!flag) rectangle(image, rect, Scalar(0, 0, 255), 2, 8, 0);
				if (flag)
				{
					ans = ans | rect;
					continue;
				}
				ss++;
			
				//imwrite("D:\\分类\\"; + to_string(k) + "_" + to_string(ss) + ".jpg", imagegray(rect));
				if (roi_svm(imagegray(rect), df3))
					ans = ans | rect;
			}
		}
	
	return ans;
}
bool roi_svm(Mat &ROI, dlib::one_vs_one_decision_function<ovo_trainer,dlib::decision_function<rbf_kernel>>  df3)
{
	dlib::array2d<unsigned char> _src(ROI.rows, ROI.cols);
	dlib::array2d<unsigned char> _img(30, 30);
	sample_type sample;
	for (int row = 0; row < ROI.rows; row++)
	{
		for (int col = 0; col < ROI.cols; col++)
		{
			_src[row][col] = ROI.at<unsigned char>(row, col);
		}
	}
	resize_image(_src, _img);
	dlib::array2d<dlib::matrix<float, 31, 1> > hog;
	dlib::extract_fhog_features(_img, hog);
	for (int i = 0; i < hog.nr(); i++)
	{
		for (int j = 0; j < hog.nc(); j++)
		{
			for (int k = 0; k < 13; k++)
			{
				sample((i*hog.nc() + j) * 13 + k) = hog[i][j](k);
			}
		}
	}
	sample(13 * 2 * 2) = ROI.rows / 100.0;
	sample(13 * 2 * 2 + 1) = ROI.cols / 100.0;
	int label = df3(sample);
	//std::cout << "start predict ...." << std::endl;
	//std::cout << "predicted label: " << df3(sample) << std::endl;
	//dlib::image_window win(_src);
	//getchar();
	if (label == 0)
		return true;
	else
		return false;
}
int main()
{



	//namedWindow("roi", WINDOW_NORMAL);
	/*namedWindow("Source Image", WINDOW_NORMAL);
	namedWindow("Test", WINDOW_NORMAL);*/
	namedWindow("文本有效区域", WINDOW_NORMAL);
	namedWindow("初始文本", WINDOW_NORMAL);

	for (int i = 1; i < 2056; i++)
	{
		imageSource = imread("C:\\Users\\Administrator\\Desktop\\所有文本图片分类\\表格\\1-"+to_string(i)+"_1.jpg");
		cv::Size dsize = cv::Size(round(0.25*imageSource.cols), round(0.25*imageSource.rows));
		//cv::resize(imageSource, imageSource, dsize);
		//connected_component_stats_demo(imageSource);
		//detect(imageSource);
		Rect rect = Connected_Component(imageSource, 6, i, Rect(1.0 / 15 * imageSource.cols, 1.0 / 15 * imageSource.rows, 13.0 / 15 * imageSource.cols, 13.0 / 15 * imageSource.rows));
		rectangle(imageSource, rect, Scalar(0, 0, 255), 2, 8, 0);
		imshow("文本有效区域", imageSource);
		//imageSource = imread("E:\\ok\\" + to_string(i) + ".jpg");
		//imshow("初始文本", imageSource);
		waitKey(0);
		printf("%d is ok\n", i);
	}
}
