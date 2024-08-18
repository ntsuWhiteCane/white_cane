#include <algorithm>
#include <iostream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "main.hpp"
#include "initial_map.hpp"
#include "path.hpp"

using namespace std;
using namespace cv;

InitialMap::InitialMap(Mat &src, int addition_point_interval, int max_corners){
	this->img = src;
	this->addition_point_interval = addition_point_interval > 0? addition_point_interval: 1;
	this->maxCorners = MAX(max_corners, 1);
}

InitialMap::~InitialMap(){

}

void InitialMap::cornerFun(){
	goodFeaturesToTrack(this->img, corners, maxCorners, qualityLevel, minDistance, Mat(), blockSize, gradientSize, useHarrisDetector, k);
}

// get the corner point by this class
// you need to run corner function first.
vector<Point2f> InitialMap::getCorner(){
	return corners;
}

void InitialMap::mapPreprocessing(){
	Mat _element;
	_element = getStructuringElement(MORPH_RECT, Size(3, 3));
	// binarization
	Mat mask;
	inRange(img, Scalar(205), Scalar(205), mask);
	img.setTo(Scalar(0), mask);
	threshold(img, img, 200, 255, THRESH_OTSU);
	// filter out(close) noise and dilate the obstacle
	// because obstale is black so we need to do the opposite operation(open and erode)
	morphologyEx(img, img, MORPH_OPEN, _element);
	morphologyEx(img, img, MORPH_ERODE, _element, Point2f(-1, -1), 1);
	
}

// it will copy the image it's cost many time to copy image.
// Please reduce the use of this function.
Mat InitialMap::getMap(){
	return this->img;
}

// it will copy the image it's cost many time to copy image.
// Please reduce the use of this function.
Mat InitialMap::getRGBMap(){
	return this->img_rgb;
}

void InitialMap::setGrayMap(Mat &inputImg){
	inputImg.copyTo(this->img);
}

void InitialMap::setRGBMap(Mat &inputImg){
	inputImg.copyTo(this->img_rgb);
}

// get obstacle points
void InitialMap::addtionPoint(Mat &outputImage){
	vector<vector<Point>> contours;
	int count;

	// find the contours
	findContours(img, contours, noArray(), RETR_LIST, CHAIN_APPROX_NONE);
	// drawContours(initial_map_img_rgb, contours, 117, Scalar(255, 255, 0));
	// cout << "contours.size: " << contours.size() << endl;

	img_rgb.copyTo(outputImage);

	// get obstacle points
	for (int i = 0; i< contours.size(); ++i){
		count = (contours[i].size() / addition_point_interval);
		points.push_back(contours[i][0]);
		circle(outputImage, contours[i][0], 2, Scalar(0, 0, 255), -1);	
		for(int j = 0; j< count; ++j){
			points.push_back(contours[i][(j+1)*(addition_point_interval) - 1]);
			circle(outputImage, contours[i][(j+1)*(addition_point_interval)-1], 3, Scalar(0, 0, 255), -1);
		}
	}
}

vector<Point2f> InitialMap::getAdditionPoint(){
	return points;
}

void InitialMap::grayMap2RGBMap(){
	cvtColor(img, img_rgb, COLOR_GRAY2BGR);
}